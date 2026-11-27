package ane

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"time"
	"unsafe"

	"github.com/aurora-silicon/vane/internal/ref"
)

// Linux accel/ane ABI v2 (H14) constants matching uapi/drm/ane_accel.h.
const (
	MaxSectionsV2 = 9
	MaxBindsV2    = 64

	CmdProgLoad   = 0x04
	CmdProcCreate = 0x05
	CmdExecV2     = 0x06

	MinPriorityV2     = 2
	MaxPriorityV2     = 7
	DefaultPriorityV2 = 3
	MaxTimeoutMSV2    = 5000
)

// SectionV2 matches struct drm_ane_section (24 bytes).
type SectionV2 struct {
	ID       uint32
	BoHandle uint32
	Size     uint64
	Offset   uint64
}

// GenericBindV2 matches struct drm_ane_generic_bind (24 bytes).
type GenericBindV2 struct {
	BufferID uint32
	BoHandle uint32
	Type     uint32
	Pad      uint32
	Size     uint64
}

// ProgLoadV2 matches struct drm_ane_prog_load (32 bytes).
type ProgLoadV2 struct {
	SectionsPtr  uint64
	GenericPtr   uint64
	SectionCount uint32
	GenericCount uint32
	ProgIDOut    uint32
	Pad          uint32
}

// ProcCreateV2 matches struct drm_ane_proc_create (8 bytes).
type ProcCreateV2 struct {
	ProgID    uint32
	ProcIDOut uint32
}

// ExecIOV2 matches struct drm_ane_exec_io (32 bytes).
type ExecIOV2 struct {
	BufferID uint32
	BoHandle uint32
	Type     uint32
	Flags    uint32
	Reserved uint64
	Size     uint64
}

// ExecV2 matches struct drm_ane_exec (32 bytes).
type ExecV2 struct {
	ProgID    uint32
	ProcID    uint32
	Priority  uint32
	TimeoutMS uint32
	Count     uint32
	Pad       uint32
	IOPtr     uint64
}

// Compile-time exact wire struct size assertions matching uapi/drm/ane_accel.h.
var (
	_ [24 - unsafe.Sizeof(SectionV2{})]byte
	_ [unsafe.Sizeof(SectionV2{}) - 24]byte
	_ [24 - unsafe.Sizeof(GenericBindV2{})]byte
	_ [unsafe.Sizeof(GenericBindV2{}) - 24]byte
	_ [32 - unsafe.Sizeof(ProgLoadV2{})]byte
	_ [unsafe.Sizeof(ProgLoadV2{}) - 32]byte
	_ [8 - unsafe.Sizeof(ProcCreateV2{})]byte
	_ [unsafe.Sizeof(ProcCreateV2{}) - 8]byte
	_ [32 - unsafe.Sizeof(ExecIOV2{})]byte
	_ [unsafe.Sizeof(ExecIOV2{}) - 32]byte
	_ [32 - unsafe.Sizeof(ExecV2{})]byte
	_ [unsafe.Sizeof(ExecV2{}) - 32]byte
)

// ABI v2 ioctl request numbers.
var (
	IoctlProgLoad   = drmIOC(iocRW, CmdProgLoad, unsafe.Sizeof(ProgLoadV2{}))
	IoctlProcCreate = drmIOC(iocRW, CmdProcCreate, unsafe.Sizeof(ProcCreateV2{}))
	IoctlExec       = drmIOC(iocRW, CmdExecV2, unsafe.Sizeof(ExecV2{}))
)

func (LinuxTransport) ProgLoad(fd int, args ProgLoadV2, sections []SectionV2, genericBinds []GenericBindV2) (ProgLoadV2, error) {
	if args.Pad != 0 {
		return ProgLoadV2{}, syscall.EINVAL
	}
	if len(sections) < 1 || len(sections) > MaxSectionsV2 {
		return ProgLoadV2{}, syscall.EINVAL
	}
	if args.SectionCount != 0 && int(args.SectionCount) != len(sections) {
		return ProgLoadV2{}, syscall.EINVAL
	}
	if len(genericBinds) > MaxBindsV2 {
		return ProgLoadV2{}, syscall.EINVAL
	}
	if args.GenericCount != 0 && int(args.GenericCount) != len(genericBinds) {
		return ProgLoadV2{}, syscall.EINVAL
	}
	for _, g := range genericBinds {
		if g.Pad != 0 {
			return ProgLoadV2{}, syscall.EINVAL
		}
	}

	args.SectionsPtr = uint64(uintptr(unsafe.Pointer(&sections[0])))
	args.SectionCount = uint32(len(sections))
	if len(genericBinds) > 0 {
		args.GenericPtr = uint64(uintptr(unsafe.Pointer(&genericBinds[0])))
		args.GenericCount = uint32(len(genericBinds))
	} else {
		args.GenericPtr = 0
		args.GenericCount = 0
	}

	err := linuxIoctl(fd, IoctlProgLoad, unsafe.Pointer(&args))
	runtime.KeepAlive(sections)
	runtime.KeepAlive(genericBinds)
	return args, err
}

func (LinuxTransport) ProcCreate(fd int, args ProcCreateV2) (ProcCreateV2, error) {
	err := linuxIoctl(fd, IoctlProcCreate, unsafe.Pointer(&args))
	return args, err
}

func (LinuxTransport) ExecV2(fd int, args ExecV2, ios []ExecIOV2) error {
	if args.Pad != 0 {
		return syscall.EINVAL
	}
	if len(ios) < 1 || len(ios) > MaxBindsV2 {
		return syscall.EINVAL
	}
	if args.Count != 0 && int(args.Count) != len(ios) {
		return syscall.EINVAL
	}
	for _, ioBuf := range ios {
		if ioBuf.Flags != 0 || ioBuf.Reserved != 0 {
			return syscall.EINVAL
		}
	}

	args.IOPtr = uint64(uintptr(unsafe.Pointer(&ios[0])))
	args.Count = uint32(len(ios))

	err := linuxIoctl(fd, IoctlExec, unsafe.Pointer(&args))
	runtime.KeepAlive(ios)
	return err
}

// V2SectionSpec describes one firmware program section in an ABI v2 bundle.
type V2SectionSpec struct {
	ID     uint32 `json:"id"`
	File   string `json:"file,omitempty"`
	Offset uint64 `json:"offset,omitempty"`
	Data   []byte `json:"-"`
}

// V2GenericBindSpec describes an optional generic buffer binding in an ABI v2 bundle.
type V2GenericBindSpec struct {
	BufferID uint32 `json:"buffer_id"`
	BoHandle uint32 `json:"bo_handle,omitempty"`
	Type     uint32 `json:"type"`
	Size     uint64 `json:"size"`
}

// V2IOSpec describes one input, output, or scratch buffer for ABI v2 EXEC.
type V2IOSpec struct {
	BufferID  uint32 `json:"buffer_id"`
	Type      uint32 `json:"type"`
	Direction string `json:"direction"`
	Size      uint64 `json:"size"`
}

// V2Bundle holds a validated ABI v2 program bundle with loaded section bytes.
type V2Bundle struct {
	Sections     []V2SectionSpec     `json:"sections"`
	GenericBinds []V2GenericBindSpec `json:"generic_binds,omitempty"`
	IOs          []V2IOSpec          `json:"ios"`
	Priority     uint32              `json:"priority"`
	TimeoutMS    uint32              `json:"timeout_ms"`
}

type rawV2Bundle struct {
	Sections     []V2SectionSpec     `json:"sections"`
	GenericBinds []V2GenericBindSpec `json:"generic_binds"`
	IOs          []V2IOSpec          `json:"ios"`
	Priority     *uint32             `json:"priority"`
	TimeoutMS    *uint32             `json:"timeout_ms"`
}

// ValidateV2Bundle enforces all ABI v2 manifest and kernel constraints on bundle.
func ValidateV2Bundle(b *V2Bundle) error {
	if b == nil {
		return errors.New("nil V2Bundle")
	}
	if len(b.Sections) < 1 || len(b.Sections) > MaxSectionsV2 {
		return fmt.Errorf("v2 bundle section count %d out of range [1, %d]", len(b.Sections), MaxSectionsV2)
	}
	seenSecIDs := make(map[uint32]bool, len(b.Sections))
	for i, s := range b.Sections {
		if s.ID < 1 || s.ID > MaxSectionsV2 {
			return fmt.Errorf("v2 bundle sections[%d] id %d out of range [1, %d]", i, s.ID, MaxSectionsV2)
		}
		if seenSecIDs[s.ID] {
			return fmt.Errorf("v2 bundle duplicate section id %d", s.ID)
		}
		seenSecIDs[s.ID] = true
		if len(s.Data) == 0 {
			return fmt.Errorf("v2 bundle section id %d has empty payload", s.ID)
		}
		sz := uint64(len(s.Data))
		if sz > ^uint64(0)-s.Offset {
			return fmt.Errorf("v2 bundle section id %d offset(%d) + size(%d) overflows uint64", s.ID, s.Offset, sz)
		}
		end := s.Offset + sz
		aligned, ok := alignUpU64(end, PageSize16K)
		if !ok || aligned > uint64(math.MaxInt) {
			return fmt.Errorf("v2 bundle section id %d allocation size overflows", s.ID)
		}
	}
	if len(b.GenericBinds) > MaxBindsV2 {
		return fmt.Errorf("v2 bundle generic_binds count %d exceeds maximum %d", len(b.GenericBinds), MaxBindsV2)
	}
	if len(b.IOs) < 1 || len(b.IOs) > MaxBindsV2 {
		return fmt.Errorf("v2 bundle ios count %d out of range [1, %d]", len(b.IOs), MaxBindsV2)
	}
	seenBufIDs := make(map[uint32]bool, len(b.IOs))
	outputCount := 0
	for i, ioSpec := range b.IOs {
		switch ioSpec.Direction {
		case "input", "scratch":
		case "output":
			outputCount++
		default:
			return fmt.Errorf("v2 bundle ios[%d] invalid direction %q (must be input, output, or scratch)", i, ioSpec.Direction)
		}
		if seenBufIDs[ioSpec.BufferID] {
			return fmt.Errorf("v2 bundle duplicate io buffer_id %d", ioSpec.BufferID)
		}
		seenBufIDs[ioSpec.BufferID] = true
		if ioSpec.Size == 0 {
			return fmt.Errorf("v2 bundle ios[%d] (buffer_id=%d) size must be > 0", i, ioSpec.BufferID)
		}
		aligned, ok := alignUpU64(ioSpec.Size, PageSize16K)
		if !ok || aligned > uint64(math.MaxInt) {
			return fmt.Errorf("v2 bundle ios[%d] (buffer_id=%d) size %d overflows", i, ioSpec.BufferID, ioSpec.Size)
		}
	}
	if outputCount < 1 {
		return errors.New("v2 bundle must declare at least 1 output io")
	}
	if b.Priority < MinPriorityV2 || b.Priority > MaxPriorityV2 {
		return fmt.Errorf("v2 bundle priority %d out of range [%d, %d]", b.Priority, MinPriorityV2, MaxPriorityV2)
	}
	if b.TimeoutMS > MaxTimeoutMSV2 {
		return fmt.Errorf("v2 bundle timeout_ms %d exceeds maximum %d", b.TimeoutMS, MaxTimeoutMSV2)
	}
	return nil
}

// ParseV2Bundle parses an ABI v2 JSON bundle manifest, loads section payloads relative
// to baseDir, and validates the resulting bundle.
func ParseV2Bundle(data []byte, baseDir string) (*V2Bundle, error) {
	var raw rawV2Bundle
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&raw); err != nil {
		return nil, fmt.Errorf("parse v2 bundle manifest: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return nil, errors.New("parse v2 bundle manifest: trailing data after JSON object")
	}

	prio := uint32(DefaultPriorityV2)
	if raw.Priority != nil {
		prio = *raw.Priority
	}
	var timeoutMS uint32
	if raw.TimeoutMS != nil {
		timeoutMS = *raw.TimeoutMS
	}

	if len(raw.Sections) < 1 || len(raw.Sections) > MaxSectionsV2 {
		return nil, fmt.Errorf("v2 bundle section count %d out of range [1, %d]", len(raw.Sections), MaxSectionsV2)
	}
	seenSecIDs := make(map[uint32]bool, len(raw.Sections))
	for i, s := range raw.Sections {
		if s.ID < 1 || s.ID > MaxSectionsV2 {
			return nil, fmt.Errorf("v2 bundle sections[%d] id %d out of range [1, %d]", i, s.ID, MaxSectionsV2)
		}
		if seenSecIDs[s.ID] {
			return nil, fmt.Errorf("v2 bundle duplicate section id %d", s.ID)
		}
		seenSecIDs[s.ID] = true
		if s.File == "" {
			return nil, fmt.Errorf("v2 bundle sections[%d] (id=%d) missing file path", i, s.ID)
		}
	}

	sections := make([]V2SectionSpec, len(raw.Sections))
	for i, s := range raw.Sections {
		secPath := s.File
		if !filepath.IsAbs(secPath) {
			secPath = filepath.Join(baseDir, secPath)
		}
		payload, err := os.ReadFile(secPath)
		if err != nil {
			return nil, fmt.Errorf("read v2 section id %d (%q): %w", s.ID, secPath, err)
		}
		s.Data = payload
		sections[i] = s
	}

	b := &V2Bundle{
		Sections:     sections,
		GenericBinds: raw.GenericBinds,
		IOs:          raw.IOs,
		Priority:     prio,
		TimeoutMS:    timeoutMS,
	}
	if err := ValidateV2Bundle(b); err != nil {
		return nil, err
	}
	return b, nil
}

// LoadV2Bundle reads an ABI v2 JSON manifest from manifestPath, resolving relative
// section files against the manifest's directory.
func LoadV2Bundle(manifestPath string) (*V2Bundle, error) {
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read v2 bundle %q: %w", manifestPath, err)
	}
	return ParseV2Bundle(raw, filepath.Dir(manifestPath))
}

func validateV2Caps(caps DeviceCaps) error {
	if caps.ABIVersion != ABIV2 {
		return fmt.Errorf("device %q has ABI v%d, expected ABI v%d", caps.Path, caps.ABIVersion, ABIV2)
	}
	wantSec := uint32(unsafe.Sizeof(SectionV2{}))
	wantBind := uint32(unsafe.Sizeof(GenericBindV2{}))
	wantExecIO := uint32(unsafe.Sizeof(ExecIOV2{}))
	if caps.SectionSize != wantSec {
		return fmt.Errorf("device %q section_size=%d, want %d", caps.Path, caps.SectionSize, wantSec)
	}
	if caps.BindSize != wantBind {
		return fmt.Errorf("device %q bind_size=%d, want %d", caps.Path, caps.BindSize, wantBind)
	}
	if caps.ExecIOSize != wantExecIO {
		return fmt.Errorf("device %q exec_io_size=%d, want %d", caps.Path, caps.ExecIOSize, wantExecIO)
	}
	return nil
}

// ExecuteV2 uploads section BOs, calls PROG_LOAD and PROC_CREATE, allocates and maps
// I/O BOs, executes warmup + timed EXEC calls with 0xff output pre-poisoning and
// repeatability verification, and strictly munmaps before BO_FREE on all exit paths.
func ExecuteV2(
	tr Transport,
	fd int,
	caps DeviceCaps,
	bundle *V2Bundle,
	inputs [][]byte,
	opts ExecOptions,
) ([]byte, RunReport, error) {
	if err := validateV2Caps(caps); err != nil {
		return nil, RunReport{}, err
	}
	if err := ValidateV2Bundle(bundle); err != nil {
		return nil, RunReport{}, fmt.Errorf("invalid V2Bundle: %w", err)
	}
	if opts.InPlace {
		return nil, RunReport{}, errors.New("--in-place is not supported for ABI v2 execution (BOs are always mmap-backed)")
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

	var inputIndices []int
	var outputIndices []int
	for i, ioSpec := range bundle.IOs {
		switch ioSpec.Direction {
		case "input":
			inputIndices = append(inputIndices, i)
		case "output":
			outputIndices = append(outputIndices, i)
		}
	}

	resolvedInputs := make([][]byte, len(inputIndices))
	if len(inputs) == 0 {
		for k, ioIdx := range inputIndices {
			resolvedInputs[k] = make([]byte, bundle.IOs[ioIdx].Size)
		}
	} else {
		if len(inputs) != len(inputIndices) {
			return nil, RunReport{}, fmt.Errorf("provided %d input buffers, bundle expects %d", len(inputs), len(inputIndices))
		}
		for k, ioIdx := range inputIndices {
			maxSz := bundle.IOs[ioIdx].Size
			if uint64(len(inputs[k])) > maxSz {
				return nil, RunReport{}, fmt.Errorf("input %d (buffer_id=%d) size %d exceeds declared size %d",
					k, bundle.IOs[ioIdx].BufferID, len(inputs[k]), maxSz)
			}
			resolvedInputs[k] = inputs[k]
		}
	}

	var resolvedAltInputs [][]byte
	if len(opts.AltInputs) > 0 {
		if runs < 2 {
			return nil, RunReport{}, fmt.Errorf("--alt-input requires --runs >= 2, got %d", runs)
		}
		if len(opts.AltInputs) != len(inputIndices) {
			return nil, RunReport{}, fmt.Errorf("provided %d alt-input buffers, bundle expects %d", len(opts.AltInputs), len(inputIndices))
		}
		resolvedAltInputs = make([][]byte, len(inputIndices))
		for k, ioIdx := range inputIndices {
			maxSz := bundle.IOs[ioIdx].Size
			if uint64(len(opts.AltInputs[k])) > maxSz {
				return nil, RunReport{}, fmt.Errorf("alt-input %d (buffer_id=%d) size %d exceeds declared size %d",
					k, bundle.IOs[ioIdx].BufferID, len(opts.AltInputs[k]), maxSz)
			}
			resolvedAltInputs[k] = opts.AltInputs[k]
		}
	}

	allocAlign := uint64(PageSize16K)
	if ps := tr.PageSize(); ps > PageSize16K {
		allocAlign = uint64(ps)
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
				allocated[i].mapped = nil
			}
			_ = tr.BoFree(fd, allocated[i].handle)
		}
	}()

	// 1. Allocate, populate, and unmap section BOs.
	secEntries := make([]SectionV2, len(bundle.Sections))
	for i, sec := range bundle.Sections {
		secEnd := sec.Offset + uint64(len(sec.Data))
		boSize, ok := alignUpU64(secEnd, allocAlign)
		if !ok || boSize > uint64(math.MaxInt) {
			return nil, RunReport{}, fmt.Errorf("section id %d BO size overflow", sec.ID)
		}
		bo, err := tr.BoInit(fd, boSize)
		if err != nil {
			return nil, RunReport{}, fmt.Errorf("BO_INIT section id %d (%d bytes): %w", sec.ID, boSize, err)
		}
		idx := len(allocated)
		allocated = append(allocated, allocBO{handle: bo.Handle})
		mm, err := tr.Mmap(fd, int64(bo.Offset), int(boSize), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
		if err != nil {
			return nil, RunReport{}, fmt.Errorf("mmap section id %d (%d bytes): %w", sec.ID, boSize, err)
		}
		allocated[idx].mapped = mm
		clear(mm)
		copy(mm[sec.Offset:secEnd], sec.Data)
		if err := tr.Munmap(mm); err != nil {
			return nil, RunReport{}, fmt.Errorf("munmap section id %d: %w", sec.ID, err)
		}
		allocated[idx].mapped = nil
		secEntries[i] = SectionV2{
			ID:       sec.ID,
			BoHandle: bo.Handle,
			Size:     uint64(len(sec.Data)),
			Offset:   sec.Offset,
		}
	}

	// 2. PROG_LOAD and PROC_CREATE.
	var genericBinds []GenericBindV2
	if len(bundle.GenericBinds) > 0 {
		genericBinds = make([]GenericBindV2, len(bundle.GenericBinds))
		for i, gb := range bundle.GenericBinds {
			genericBinds[i] = GenericBindV2{
				BufferID: gb.BufferID,
				BoHandle: gb.BoHandle,
				Type:     gb.Type,
				Pad:      0,
				Size:     gb.Size,
			}
		}
	}
	progRes, err := tr.ProgLoad(fd, ProgLoadV2{
		SectionCount: uint32(len(secEntries)),
		GenericCount: uint32(len(genericBinds)),
		Pad:          0,
	}, secEntries, genericBinds)
	if err != nil {
		return nil, RunReport{}, fmt.Errorf("DRM_IOCTL_ANE_PROG_LOAD: %w", err)
	}
	procRes, err := tr.ProcCreate(fd, ProcCreateV2{ProgID: progRes.ProgIDOut})
	if err != nil {
		return nil, RunReport{}, fmt.Errorf("DRM_IOCTL_ANE_PROC_CREATE: %w", err)
	}

	// 3. Allocate and mmap I/O BOs.
	execIOs := make([]ExecIOV2, len(bundle.IOs))
	ioMaps := make([][]byte, len(bundle.IOs))
	for i, spec := range bundle.IOs {
		boSize, ok := alignUpU64(spec.Size, allocAlign)
		if !ok || boSize > uint64(math.MaxInt) {
			return nil, RunReport{}, fmt.Errorf("io[%d] (buffer_id=%d) BO size overflow", i, spec.BufferID)
		}
		bo, err := tr.BoInit(fd, boSize)
		if err != nil {
			return nil, RunReport{}, fmt.Errorf("BO_INIT io[%d] (buffer_id=%d, %d bytes): %w", i, spec.BufferID, boSize, err)
		}
		idx := len(allocated)
		allocated = append(allocated, allocBO{handle: bo.Handle})
		mm, err := tr.Mmap(fd, int64(bo.Offset), int(boSize), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
		if err != nil {
			return nil, RunReport{}, fmt.Errorf("mmap io[%d] (buffer_id=%d, %d bytes): %w", i, spec.BufferID, boSize, err)
		}
		allocated[idx].mapped = mm
		clear(mm)
		ioMaps[i] = mm
		execIOs[i] = ExecIOV2{
			BufferID: spec.BufferID,
			BoHandle: bo.Handle,
			Type:     spec.Type,
			Flags:    0,
			Reserved: 0,
			Size:     spec.Size,
		}
	}

	// 4. Execute warmup + timed runs with 0xff output pre-poisoning and repeatability checks.
	execArgs := ExecV2{
		ProgID:    progRes.ProgIDOut,
		ProcID:    procRes.ProcIDOut,
		Priority:  bundle.Priority,
		TimeoutMS: bundle.TimeoutMS,
		Count:     uint32(len(execIOs)),
		Pad:       0,
	}

	primaryOutIdx := outputIndices[0]
	outSize := int(bundle.IOs[primaryOutIdx].Size)

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
		for k, ioIdx := range inputIndices {
			clear(ioMaps[ioIdx])
			copy(ioMaps[ioIdx], curInputs[k])
		}
		for _, ioIdx := range outputIndices {
			outMap := ioMaps[ioIdx]
			for j := range outMap {
				outMap[j] = 0xff
			}
		}

		t0 := time.Now()
		err := tr.ExecV2(fd, execArgs, execIOs)
		dt := float64(time.Since(t0).Nanoseconds()) / 1e6
		if err != nil {
			return nil, RunReport{}, fmt.Errorf("DRM_IOCTL_ANE_EXEC (run %d): %w", r+1, err)
		}
		if r >= warmup {
			latencies = append(latencies, dt)
		}

		outBytes := make([]byte, outSize)
		copy(outBytes, ioMaps[primaryOutIdx][:outSize])
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

	surfSizes := make([]uint64, len(bundle.IOs))
	surfDirs := make([]uint32, len(bundle.IOs))
	for i, ioSpec := range bundle.IOs {
		surfSizes[i] = ioSpec.Size
		switch ioSpec.Direction {
		case "input":
			surfDirs[i] = 1
		case "output":
			surfDirs[i] = 2
		default:
			surfDirs[i] = 0
		}
	}

	report := RunReport{
		Device:            caps.Path,
		ABIVersion:        caps.ABIVersion,
		ChipFamily:        caps.ChipFamily,
		Mode:              "abi-v2",
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
