package ane

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math/bits"
	"syscall"
	"time"
	"unsafe"

	"github.com/aurora-silicon/vane/internal/ref"
)

// Linux accel/ane ABI v1 (H13) constants matching uapi/drm/ane_accel.h and .anec format.
const (
	TileCount = 0x20
	FifoNID   = 0x40
	CmdGran   = 0x10

	CmdBoInit = 0x01
	CmdBoFree = 0x02
	CmdSubmit = 0x03

	MaxTDCount     = 0xffff
	MinTDSize      = 4
	MaxTDSize      = 0x40000
	AnecHeaderSize = 0x800
)

// BoInitArgs matches struct drm_ane_bo_init (24 bytes).
type BoInitArgs struct {
	Handle uint32
	Pad    uint32
	Size   uint64
	Offset uint64
}

// BoFreeArgs matches struct drm_ane_bo_free (8 bytes).
type BoFreeArgs struct {
	Handle uint32
	Pad    uint32
}

// SubmitArgs matches struct drm_ane_submit (152 bytes).
type SubmitArgs struct {
	TskSize    uint64
	TDCount    uint32
	TDSize     uint32
	Handles    [TileCount]uint32
	BtspHandle uint32
	Pad        uint32
}

// ABI v1 ioctl request numbers.
var (
	IoctlBoInit = drmIOC(iocRW, CmdBoInit, unsafe.Sizeof(BoInitArgs{}))
	IoctlBoFree = drmIOC(iocRW, CmdBoFree, unsafe.Sizeof(BoFreeArgs{}))
	IoctlSubmit = drmIOC(iocRW, CmdSubmit, unsafe.Sizeof(SubmitArgs{}))
)

func (LinuxTransport) BoInit(fd int, size uint64) (BoInitArgs, error) {
	args := BoInitArgs{Size: size}
	if err := linuxIoctl(fd, IoctlBoInit, unsafe.Pointer(&args)); err != nil {
		return BoInitArgs{}, err
	}
	return args, nil
}

func (LinuxTransport) BoFree(fd int, handle uint32) error {
	args := BoFreeArgs{Handle: handle}
	return linuxIoctl(fd, IoctlBoFree, unsafe.Pointer(&args))
}

func (LinuxTransport) Submit(fd int, args SubmitArgs) error {
	return linuxIoctl(fd, IoctlSubmit, unsafe.Pointer(&args))
}

// AnecHeader holds the decoded fields of the 0x800-byte .anec container header.
type AnecHeader struct {
	Size     uint64
	TDSize   uint32
	TDCount  uint32
	TskSize  uint64
	KrnSize  uint64
	SrcCount uint32
	DstCount uint32
	Tiles    [TileCount]uint32
	NCHW     [TileCount][6]uint64
}

// TileBytes returns the byte allocation for BAR slot bdx (tiles[bdx] * 16 KiB).
func (h AnecHeader) TileBytes(bdx int) uint64 {
	if bdx < 0 || bdx >= TileCount {
		return 0
	}
	return uint64(h.Tiles[bdx]) << 14
}

// DstSlot returns the BAR slot index for output surface idx (4 + idx).
func (h AnecHeader) DstSlot(idx int) int {
	return 4 + idx
}

// SrcSlot returns the BAR slot index for input surface idx (4 + dst_count + idx).
func (h AnecHeader) SrcSlot(idx int) int {
	return 4 + int(h.DstCount) + idx
}

// AnecProgram holds a validated .anec header and its command+weights payload.
type AnecProgram struct {
	Header  AnecHeader
	Payload []byte
}

func alignUpU64(n, align uint64) (uint64, bool) {
	if align == 0 {
		return n, true
	}
	rem := n % align
	if rem == 0 {
		return n, true
	}
	add := align - rem
	if n > ^uint64(0)-add {
		return 0, false
	}
	return n + add, true
}

// ParseAnec parses and validates an ABI v1 .anec binary container.
func ParseAnec(data []byte) (*AnecProgram, error) {
	if len(data) < AnecHeaderSize {
		return nil, fmt.Errorf("anec truncated: %d < %d header bytes", len(data), AnecHeaderSize)
	}

	var h AnecHeader
	h.Size = binary.LittleEndian.Uint64(data[0x000:0x008])
	h.TDSize = binary.LittleEndian.Uint32(data[0x008:0x00c])
	h.TDCount = binary.LittleEndian.Uint32(data[0x00c:0x010])
	h.TskSize = binary.LittleEndian.Uint64(data[0x010:0x018])
	h.KrnSize = binary.LittleEndian.Uint64(data[0x018:0x020])
	h.SrcCount = binary.LittleEndian.Uint32(data[0x020:0x024])
	h.DstCount = binary.LittleEndian.Uint32(data[0x024:0x028])

	for i := 0; i < TileCount; i++ {
		off := 0x028 + 4*i
		h.Tiles[i] = binary.LittleEndian.Uint32(data[off : off+4])
	}
	for i := 0; i < TileCount; i++ {
		for j := 0; j < 6; j++ {
			off := 0x0a8 + (i*6+j)*8
			h.NCHW[i][j] = binary.LittleEndian.Uint64(data[off : off+8])
		}
	}

	if h.Size == 0 {
		return nil, errors.New("anec header size must be > 0")
	}
	if uint64(len(data)-AnecHeaderSize) != h.Size {
		return nil, fmt.Errorf("anec payload size mismatch: file has %d payload bytes, header declares %d",
			len(data)-AnecHeaderSize, h.Size)
	}
	if h.TDCount == 0 || h.TDCount > MaxTDCount {
		return nil, fmt.Errorf("anec td_count %d out of range [1, %d]", h.TDCount, MaxTDCount)
	}
	if h.TDSize < MinTDSize || h.TDSize > MaxTDSize || h.TDSize%4 != 0 {
		return nil, fmt.Errorf("anec td_size %d must be in [%d, %d] and 4-byte aligned",
			h.TDSize, MinTDSize, MaxTDSize)
	}
	if uint64(h.TDSize) > h.Size {
		return nil, fmt.Errorf("anec td_size %d exceeds payload size %d", h.TDSize, h.Size)
	}
	if h.TskSize == 0 {
		return nil, errors.New("anec tsk_size must be > 0")
	}
	hi, tdBytes := bits.Mul64(uint64(h.TDCount), uint64(h.TDSize))
	if hi != 0 || tdBytes > h.TskSize {
		return nil, fmt.Errorf("anec td_count(%d)*td_size(%d) exceeds tsk_size(%d)",
			h.TDCount, h.TDSize, h.TskSize)
	}
	if h.Tiles[0] == 0 {
		return nil, errors.New("anec tiles[0] must be > 0")
	}
	if h.Tiles[1] != 0 {
		return nil, fmt.Errorf("anec tiles[1] must be 0 (derived by kernel), got %d", h.Tiles[1])
	}
	tile0Bytes := h.TileBytes(0)
	if h.Size > tile0Bytes {
		return nil, fmt.Errorf("anec payload size %d exceeds tiles[0] allocation %d", h.Size, tile0Bytes)
	}
	if h.TskSize >= tile0Bytes {
		return nil, fmt.Errorf("anec tsk_size %d must be strictly smaller than tiles[0] allocation %d",
			h.TskSize, tile0Bytes)
	}
	if h.KrnSize > 0 {
		tskAligned, ok := alignUpU64(h.TskSize, CmdGran)
		if !ok || tskAligned > h.Size || h.KrnSize > h.Size-tskAligned {
			return nil, fmt.Errorf("anec round_up(tsk_size=%d, 0x10) + krn_size(%d) exceeds payload size %d",
				h.TskSize, h.KrnSize, h.Size)
		}
	}
	if h.DstCount < 1 {
		return nil, errors.New("anec dst_count must be >= 1")
	}
	if uint64(4)+uint64(h.DstCount)+uint64(h.SrcCount) > TileCount {
		return nil, fmt.Errorf("anec surface slots 4 + dst_count(%d) + src_count(%d) exceed %d",
			h.DstCount, h.SrcCount, TileCount)
	}
	for i := 0; i < int(h.DstCount); i++ {
		bdx := h.DstSlot(i)
		if h.Tiles[bdx] == 0 {
			return nil, fmt.Errorf("anec output slot %d has zero tile allocation", bdx)
		}
	}
	for i := 0; i < int(h.SrcCount); i++ {
		bdx := h.SrcSlot(i)
		if h.Tiles[bdx] == 0 {
			return nil, fmt.Errorf("anec input slot %d has zero tile allocation", bdx)
		}
	}

	payload := make([]byte, len(data)-AnecHeaderSize)
	copy(payload, data[AnecHeaderSize:])
	return &AnecProgram{
		Header:  h,
		Payload: payload,
	}, nil
}

// PatchBootstrapNID patches TD0 word 0 bits [23:16] with the FIFO network ID.
func PatchBootstrapNID(td0 []byte, nid uint8) error {
	if len(td0) < 4 {
		return fmt.Errorf("bootstrap TD0 truncated: %d < 4 bytes", len(td0))
	}
	hdr0 := binary.LittleEndian.Uint32(td0[0:4])
	hdr0 = (hdr0 & 0x0f00ffff) | (uint32(nid) << 16)
	binary.LittleEndian.PutUint32(td0[0:4], hdr0)
	return nil
}

// ExecuteAnec allocates ABI v1 tile & bootstrap BOs, maps them, executes warmup + timed SUBMIT
// calls with 0xff output poisoning and repeatability checks, and frees every BO before returning.
func ExecuteAnec(
	tr Transport,
	fd int,
	caps DeviceCaps,
	prog *AnecProgram,
	inputs [][]byte,
	opts ExecOptions,
) ([]byte, RunReport, error) {
	if prog == nil {
		return nil, RunReport{}, errors.New("nil AnecProgram")
	}
	if opts.InPlace {
		return nil, RunReport{}, errors.New("--in-place is not supported for ABI v1 execution (BOs are always mmap-backed)")
	}
	runs := opts.Runs
	if runs <= 0 {
		runs = 1
	}
	warmup := opts.Warmup
	if warmup < 0 {
		return nil, RunReport{}, fmt.Errorf("--warmup must be >= 0, got %d", warmup)
	}
	totalRuns := warmup + runs

	dstCount := int(prog.Header.DstCount)
	srcCount := int(prog.Header.SrcCount)

	resolvedInputs := make([][]byte, srcCount)
	if len(inputs) == 0 {
		for k := 0; k < srcCount; k++ {
			resolvedInputs[k] = make([]byte, prog.Header.TileBytes(prog.Header.SrcSlot(k)))
		}
	} else {
		if len(inputs) != srcCount {
			return nil, RunReport{}, fmt.Errorf("provided %d input buffers, .anec expects %d", len(inputs), srcCount)
		}
		for k := 0; k < srcCount; k++ {
			slot := prog.Header.SrcSlot(k)
			tileBytes := prog.Header.TileBytes(slot)
			if uint64(len(inputs[k])) > tileBytes {
				return nil, RunReport{}, fmt.Errorf("input %d (slot %d) size %d exceeds tile allocation %d",
					k, slot, len(inputs[k]), tileBytes)
			}
			resolvedInputs[k] = inputs[k]
		}
	}

	var resolvedAltInputs [][]byte
	if len(opts.AltInputs) > 0 {
		if runs < 2 {
			return nil, RunReport{}, fmt.Errorf("--alt-input requires --runs >= 2, got %d", runs)
		}
		if len(opts.AltInputs) != srcCount {
			return nil, RunReport{}, fmt.Errorf("provided %d alt-input buffers, .anec expects %d", len(opts.AltInputs), srcCount)
		}
		resolvedAltInputs = make([][]byte, srcCount)
		for k := 0; k < srcCount; k++ {
			slot := prog.Header.SrcSlot(k)
			tileBytes := prog.Header.TileBytes(slot)
			if uint64(len(opts.AltInputs[k])) > tileBytes {
				return nil, RunReport{}, fmt.Errorf("alt-input %d (slot %d) size %d exceeds tile allocation %d",
					k, slot, len(opts.AltInputs[k]), tileBytes)
			}
			resolvedAltInputs[k] = opts.AltInputs[k]
		}
	}

	type allocBO struct {
		handle uint32
		mapped []byte
	}
	var allocated []allocBO
	defer func() {
		for i := len(allocated) - 1; i >= 0; i-- {
			if allocated[i].mapped != nil {
				_ = tr.Munmap(allocated[i].mapped)
			}
			_ = tr.BoFree(fd, allocated[i].handle)
		}
	}()

	var handles [TileCount]uint32
	var tileMaps [TileCount][]byte

	for bdx := 0; bdx < TileCount; bdx++ {
		if prog.Header.Tiles[bdx] == 0 {
			continue
		}
		boSize := prog.Header.TileBytes(bdx)
		bo, err := tr.BoInit(fd, boSize)
		if err != nil {
			return nil, RunReport{}, fmt.Errorf("BO_INIT tile[%d] (%d bytes): %w", bdx, boSize, err)
		}
		idx := len(allocated)
		allocated = append(allocated, allocBO{handle: bo.Handle})
		mm, err := tr.Mmap(fd, int64(bo.Offset), int(boSize), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
		if err != nil {
			return nil, RunReport{}, fmt.Errorf("mmap tile[%d] (%d bytes): %w", bdx, boSize, err)
		}
		allocated[idx].mapped = mm
		handles[bdx] = bo.Handle
		tileMaps[bdx] = mm
	}

	btspSize, ok := alignUpU64(uint64(prog.Header.TDSize), PageSize16K)
	if !ok {
		return nil, RunReport{}, errors.New("bootstrap BO size overflow")
	}
	btspBO, err := tr.BoInit(fd, btspSize)
	if err != nil {
		return nil, RunReport{}, fmt.Errorf("BO_INIT btsp (%d bytes): %w", btspSize, err)
	}
	btspIdx := len(allocated)
	allocated = append(allocated, allocBO{handle: btspBO.Handle})
	btspMap, err := tr.Mmap(fd, int64(btspBO.Offset), int(btspSize), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return nil, RunReport{}, fmt.Errorf("mmap btsp (%d bytes): %w", btspSize, err)
	}
	allocated[btspIdx].mapped = btspMap

	// Populate command+weights tile[0] and bootstrap TD0 with FifoNID.
	clear(tileMaps[0])
	copy(tileMaps[0], prog.Payload)
	clear(btspMap)
	copy(btspMap[:prog.Header.TDSize], prog.Payload[:prog.Header.TDSize])
	if err := PatchBootstrapNID(btspMap[:prog.Header.TDSize], FifoNID); err != nil {
		return nil, RunReport{}, err
	}

	submitArgs := SubmitArgs{
		TskSize:    prog.Header.TskSize,
		TDCount:    prog.Header.TDCount,
		TDSize:     prog.Header.TDSize,
		Handles:    handles,
		BtspHandle: btspBO.Handle,
		Pad:        0,
	}

	outSlot := prog.Header.DstSlot(0)
	outSize := int(prog.Header.TileBytes(outSlot))

	latencies := make([]float64, 0, runs)
	var firstOutput []byte
	var firstAltOutput []byte

	recordOutput := func(r int, isAlt bool, outBytes []byte) error {
		if isAlt {
			if firstAltOutput == nil {
				firstAltOutput = outBytes
			} else if !bytes.Equal(outBytes, firstAltOutput) {
				s1 := sha256.Sum256(firstAltOutput)
				s2 := sha256.Sum256(outBytes)
				return fmt.Errorf("non-deterministic alt-input output on run %d: sha256 %s != run 2 sha256 %s",
					r+1, hex.EncodeToString(s2[:]), hex.EncodeToString(s1[:]))
			}
			return nil
		}
		if firstOutput == nil {
			firstOutput = outBytes
		} else if !bytes.Equal(outBytes, firstOutput) {
			s1 := sha256.Sum256(firstOutput)
			s2 := sha256.Sum256(outBytes)
			return fmt.Errorf("non-deterministic output on run %d: sha256 %s != run 1 sha256 %s",
				r+1, hex.EncodeToString(s2[:]), hex.EncodeToString(s1[:]))
		}
		return nil
	}

	for r := 0; r < totalRuns; r++ {
		isAlt := resolvedAltInputs != nil && (r%2 == 1)
		curInputs := resolvedInputs
		if isAlt {
			curInputs = resolvedAltInputs
		}
		for k := 0; k < srcCount; k++ {
			slot := prog.Header.SrcSlot(k)
			clear(tileMaps[slot])
			copy(tileMaps[slot], curInputs[k])
		}
		for i := 0; i < dstCount; i++ {
			dstMap := tileMaps[prog.Header.DstSlot(i)]
			for j := range dstMap {
				dstMap[j] = 0xff
			}
		}

		t0 := time.Now()
		err := tr.Submit(fd, submitArgs)
		dt := float64(time.Since(t0).Nanoseconds()) / 1e6
		if err != nil {
			return nil, RunReport{}, fmt.Errorf("DRM_IOCTL_ANE_SUBMIT (run %d): %w", r+1, err)
		}
		if r >= warmup {
			latencies = append(latencies, dt)
		}

		outBytes := make([]byte, outSize)
		copy(outBytes, tileMaps[outSlot][:outSize])
		if err := recordOutput(r, isAlt, outBytes); err != nil {
			return nil, RunReport{}, err
		}
	}

	minMS, medianMS, p95MS, meanMS := ComputeLatencyStats(latencies)
	sum := sha256.Sum256(firstOutput)
	var altSHA string
	if firstAltOutput != nil {
		as := sha256.Sum256(firstAltOutput)
		altSHA = hex.EncodeToString(as[:])
	}

	var compPtr *ref.ComparisonResult
	if opts.Expected != nil {
		comp, err := ref.CompareFP16Bytes(firstOutput, opts.Expected, opts.Atol, opts.Rtol, opts.MaxULP)
		if err != nil {
			return nil, RunReport{}, fmt.Errorf("reference comparison failed: %w", err)
		}
		compPtr = &comp
	}

	surfSizes := make([]uint64, 0, dstCount+srcCount)
	surfDirs := make([]uint32, 0, dstCount+srcCount)
	for i := 0; i < dstCount; i++ {
		surfSizes = append(surfSizes, prog.Header.TileBytes(prog.Header.DstSlot(i)))
		surfDirs = append(surfDirs, 2)
	}
	for k := 0; k < srcCount; k++ {
		surfSizes = append(surfSizes, prog.Header.TileBytes(prog.Header.SrcSlot(k)))
		surfDirs = append(surfDirs, 1)
	}

	report := RunReport{
		Device:            caps.Path,
		ABIVersion:        caps.ABIVersion,
		ChipFamily:        caps.ChipFamily,
		Mode:              "abi-v1",
		Runs:              runs,
		Warmup:            warmup,
		SurfaceCount:      len(surfSizes),
		SurfaceSizes:      surfSizes,
		SurfaceDirections: surfDirs,
		OutputSize:        uint64(outSize),
		OutputSHA256:      hex.EncodeToString(sum[:]),
		AltOutputSHA256:   altSHA,
		LatenciesMS:       latencies,
		MinMS:             minMS,
		MedianMS:          medianMS,
		P95MS:             p95MS,
		MeanMS:            meanMS,
		Repeatable:        true,
		Comparison:        compPtr,
	}
	return firstOutput, report, nil
}
