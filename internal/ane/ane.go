// Package ane implements the Linux accel/ane ABI v3 device interface,
// host-built LOAD section derivation from HWX binaries, and program execution.
package ane

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"syscall"
	"time"
	"unsafe"

	"github.com/aurora-silicon/vane/internal/ref"
)

// Linux accel/ane ABI v3 constants matching uapi/drm/ane_accel.h.
const (
	ABIV1 = 1
	ABIV2 = 2
	ABIV3 = 3

	ChipH13 = 13
	ChipH14 = 14
	ChipH16 = 16
	ChipH17 = 17

	CmdGetCaps = 0x00
	CmdH16Load = 0x07
	CmdH16Exec = 0x08

	SectionGeneric        uint32 = 1
	SectionOperation      uint32 = 2
	SectionProcedure      uint32 = 3
	SectionTextProperties uint32 = 4
	SectionsCount         uint32 = 4

	SectionMin = 0x28
	SectionMax = 0x10000

	ResourceGeneric        uint32 = 1002
	ResourceOperation      uint32 = 1003
	ResourceProcedure      uint32 = 1004
	ResourceTextProperties uint32 = 1005
	ResourceText           uint32 = 1006
	ResourceKernel         uint32 = 1007

	MaxSurfaces = 4

	ExecInPlace uint32 = 1 << 0

	PageSize16K        = 0x4000
	PrologueFrameBytes = 16
	TaskFrameAlign     = 16
	ExitSkip           = 4

	iocWrite       = 1
	iocRead        = 2
	iocRW          = iocRead | iocWrite
	drmIoctlBase   = 'd'
	drmCommandBase = 0x40
)

func drmIOC(dir, nr, size uintptr) uintptr {
	return (dir << 30) | (size << 16) | (drmIoctlBase << 8) | (drmCommandBase + nr)
}

// MmapOffset returns the accel fd mmap offset for surface index i (0-based).
func MmapOffset(surfaceIndex int) int64 {
	return int64(surfaceIndex+1) << 32
}

// GetCapsArgs matches struct drm_ane_get_caps (32 bytes).
type GetCapsArgs struct {
	Flags       uint32
	Size        uint32
	ABIVersion  uint32
	ChipFamily  uint32
	SectionSize uint32
	BindSize    uint32
	ExecIOSize  uint32
	Pad         uint32
}

// H16Section matches struct drm_ane_h16_section (16 bytes).
type H16Section struct {
	Kind    uint32
	Size    uint32
	DataPtr uint64
}

// H16LoadArgs matches struct drm_ane_h16_load (88 bytes).
type H16LoadArgs struct {
	HwxPtr           uint64
	HwxSize          uint64
	SectionsPtr      uint64
	SectionCount     uint32
	Flags            uint32
	SurfaceCount     uint32
	Pad              uint32
	SurfaceSize      [MaxSurfaces]uint64
	SurfaceDirection [MaxSurfaces]uint32
}

// H16Buffer matches struct drm_ane_h16_buffer (16 bytes).
type H16Buffer struct {
	Ptr  uint64
	Size uint64
}

// H16ExecArgs matches struct drm_ane_h16_exec (16 bytes).
type H16ExecArgs struct {
	BuffersPtr uint64
	Count      uint32
	Flags      uint32
}

// Active ABI v3 ioctl request numbers.
var (
	IoctlGetCaps = drmIOC(iocRW, CmdGetCaps, unsafe.Sizeof(GetCapsArgs{}))
	IoctlH16Load = drmIOC(iocRW, CmdH16Load, unsafe.Sizeof(H16LoadArgs{}))
	IoctlH16Exec = drmIOC(iocWrite, CmdH16Exec, unsafe.Sizeof(H16ExecArgs{}))
)

// SectionFilenames maps section kind 1..4 to canonical on-disk filenames.
var SectionFilenames = map[uint32]string{
	SectionGeneric:        "generic.bin",
	SectionOperation:      "operation.bin",
	SectionProcedure:      "procedure.bin",
	SectionTextProperties: "text-properties.bin",
}

// ErrNoMatchingDevice indicates no matching accel/ane device is present (exit code 4).
var ErrNoMatchingDevice = errors.New("no matching accel/ane device found")

// ---------------------------------------------------------------------------
// Pure-Go host-built LOAD section builder from HWX bytes
// ---------------------------------------------------------------------------

func readFixedASCII(buf []byte, off, width int) (string, error) {
	if off+width > len(buf) {
		return "", fmt.Errorf("fixed ASCII string at 0x%x exceeds file size", off)
	}
	raw := buf[off : off+width]
	nul := bytes.IndexByte(raw, 0)
	if nul < 0 {
		return "", fmt.Errorf("missing NUL terminator in fixed string at 0x%x", off)
	}
	return string(raw[:nul]), nil
}

// ExpectedChipFromHWX reads the HWX CPU subtype and returns ChipH16 or ChipH17.
func ExpectedChipFromHWX(hwx []byte) (uint32, error) {
	if len(hwx) < 32 {
		return 0, fmt.Errorf("HWX file truncated (%d < 32 bytes)", len(hwx))
	}
	subtype := binary.LittleEndian.Uint32(hwx[8:12])
	switch subtype {
	case 7:
		return ChipH16, nil
	case 9:
		return ChipH17, nil
	default:
		return 0, fmt.Errorf("unsupported HWX CPU subtype %d", subtype)
	}
}

// BuildSectionsFromHWX derives the 4 host-built LOAD sections deterministically from HWX bytes,
// sorting OPERATION BAR slot bindings canonically in ascending slot order.
func BuildSectionsFromHWX(hwx []byte) (map[uint32][]byte, error) {
	if len(hwx) < 32 {
		return nil, errors.New("HWX too small (< 32 bytes)")
	}
	magic := binary.LittleEndian.Uint32(hwx[0x00:])
	cpuType := binary.LittleEndian.Uint32(hwx[0x04:])
	cpuSubtype := binary.LittleEndian.Uint32(hwx[0x08:])
	fileType := binary.LittleEndian.Uint32(hwx[0x0c:])
	ncmds := binary.LittleEndian.Uint32(hwx[0x10:])
	sizeofcmds := binary.LittleEndian.Uint32(hwx[0x14:])
	flags := binary.LittleEndian.Uint32(hwx[0x18:])
	reserved := binary.LittleEndian.Uint32(hwx[0x1c:])

	if magic != 0xbeefface || cpuType != 0x80 || fileType != 2 || flags != 0x200000 || reserved != 0 {
		return nil, errors.New("invalid HWX file header")
	}
	if cpuSubtype != 7 && cpuSubtype != 9 {
		return nil, fmt.Errorf("unsupported HWX CPU subtype %d", cpuSubtype)
	}
	cmdEnd := 32 + int(sizeofcmds)
	if cmdEnd > len(hwx) {
		return nil, errors.New("HWX load commands exceed file length")
	}

	type textSecInfo struct {
		addr   uint64
		size   uint64
		offset uint32
	}
	type scratchInfo struct {
		addr uint64
		size uint64
	}
	type surfaceInfo struct {
		direction uint32
		addr      uint64
		vmsize    uint64
	}

	var textSec *textSecInfo
	var kernelAddr *uint64
	var scratch *scratchInfo
	var surfaces []surfaceInfo
	var progDesc []byte

	cursor := 32
	for ci := uint32(0); ci < ncmds; ci++ {
		if cursor+8 > cmdEnd {
			return nil, fmt.Errorf("truncated load command %d", ci)
		}
		cmd := binary.LittleEndian.Uint32(hwx[cursor:])
		cmdsize := int(binary.LittleEndian.Uint32(hwx[cursor+4:]))
		if cmdsize < 8 || cursor+cmdsize > cmdEnd {
			return nil, fmt.Errorf("invalid cmdsize %d at 0x%x", cmdsize, cursor)
		}
		switch cmd {
		case 0x19: // LC_SEGMENT_64
			if cmdsize < 72 {
				return nil, fmt.Errorf("LC_SEGMENT_64 cmdsize %d < 72", cmdsize)
			}
			segname, err := readFixedASCII(hwx, cursor+8, 16)
			if err != nil {
				return nil, err
			}
			vmsize := binary.LittleEndian.Uint64(hwx[cursor+0x20:])
			initprot := binary.LittleEndian.Uint32(hwx[cursor+0x3c:])
			nsects := int(binary.LittleEndian.Uint32(hwx[cursor+0x40:]))
			if cmdsize != 72+nsects*80 {
				return nil, fmt.Errorf("segment %q cmdsize mismatch", segname)
			}
			for si := 0; si < nsects; si++ {
				soff := cursor + 72 + si*80
				sectname, err := readFixedASCII(hwx, soff, 16)
				if err != nil {
					return nil, err
				}
				saddr := binary.LittleEndian.Uint64(hwx[soff+0x20:])
				ssize := binary.LittleEndian.Uint64(hwx[soff+0x28:])
				sfileoff := binary.LittleEndian.Uint32(hwx[soff+0x30:])
				switch {
				case segname == "__TEXT" && sectname == "__text":
					textSec = &textSecInfo{addr: saddr, size: ssize, offset: sfileoff}
				case segname == "__KERN_0" && sectname == "__kern_0":
					a := saddr
					kernelAddr = &a
				case segname == "__DATA" && sectname == "__bss":
					scratch = &scratchInfo{addr: saddr, size: ssize}
				case segname == "__FVMLIB":
					surfaces = append(surfaces, surfaceInfo{direction: initprot, addr: saddr, vmsize: vmsize})
				}
			}
		case 4: // Descriptor
			if cmdsize < 12 {
				return nil, errors.New("descriptor command too small")
			}
			subtype := binary.LittleEndian.Uint32(hwx[cursor+8:])
			if subtype == 4 {
				progDesc = hwx[cursor : cursor+cmdsize]
			}
		}
		cursor += cmdsize
	}

	if textSec == nil || progDesc == nil {
		return nil, errors.New("HWX missing __TEXT,__text or program descriptor")
	}
	textEndOff := int(textSec.offset) + int(textSec.size)
	if textEndOff > len(hwx) {
		return nil, errors.New("__TEXT,__text extends past end of file")
	}
	textBytes := hwx[textSec.offset:textEndOff]

	// Walk task stream to count tasks and collect distinct BAR slots (bit 29 set)
	streamOff := PrologueFrameBytes
	taskCount := uint32(0)
	barSlotSeen := map[uint32]bool{}
	var barSlots []uint32

	for streamOff < len(textBytes) {
		if streamOff+TaskFrameAlign > len(textBytes) {
			return nil, fmt.Errorf("truncated task stream at 0x%x", streamOff)
		}
		w0 := binary.LittleEndian.Uint32(textBytes[streamOff:])
		words := int((w0 >> 16) & 0x7ff)
		if words == 0 {
			streamOff += TaskFrameAlign
			continue
		}
		taskEnd := streamOff + 4*words
		if taskEnd > len(textBytes) {
			return nil, fmt.Errorf("task at 0x%x extends past __text", streamOff)
		}
		w8 := binary.LittleEndian.Uint32(textBytes[streamOff+32:])
		hdrWords := 9
		if (w8 & 3) == 3 {
			hdrWords = 10
		}
		rec := streamOff + 4*hdrWords
		for rec < taskEnd {
			h := binary.LittleEndian.Uint32(textBytes[rec:])
			rec += 4
			var count int
			if ((h >> 31) & 1) == 1 {
				mask := uint16((h >> 15) & 0xffff)
				count = 1 + bits.OnesCount16(mask)
			} else {
				count = int((h>>15)&0x3f) + 1
				if ((h >> 29) & 1) == 1 {
					slot := (h >> 23) & 0x3f
					if !barSlotSeen[slot] {
						barSlotSeen[slot] = true
						barSlots = append(barSlots, slot)
					}
				}
			}
			rec += 4 * count
		}
		taskCount++
		if taskEnd < len(textBytes) {
			streamOff = ref.AlignUp(taskEnd, TaskFrameAlign)
		} else {
			streamOff = taskEnd
		}
	}

	// Assign buffer IDs: output = 4, inputs = 5, 6, ..., scratch = next
	nextIn := uint32(5)
	surfIDs := make([]uint32, len(surfaces))
	for i, s := range surfaces {
		if s.direction == 2 {
			surfIDs[i] = 4
		} else {
			surfIDs[i] = nextIn
			nextIn++
		}
	}
	var scratchID uint32
	if scratch != nil {
		scratchID = nextIn
	}

	// 1. GENERIC
	entries := len(surfaces)
	if scratch != nil {
		entries++
	}
	gen := make([]byte, 0x208+entries*0x30)
	binary.LittleEndian.PutUint32(gen[0x000:], 1)
	binary.LittleEndian.PutUint32(gen[0x004:], 0x10)
	binary.LittleEndian.PutUint32(gen[0x204:], uint32(entries))
	for i, s := range surfaces {
		base := 0x208 + i*0x30
		binary.LittleEndian.PutUint32(gen[base+0x00:], 1)
		binary.LittleEndian.PutUint32(gen[base+0x04:], surfIDs[i])
		binary.LittleEndian.PutUint32(gen[base+0x08:], s.direction-1)
		binary.LittleEndian.PutUint32(gen[base+0x10:], s.direction)
		binary.LittleEndian.PutUint64(gen[base+0x20:], s.vmsize)
		binary.LittleEndian.PutUint32(gen[base+0x28:], 0xffff)
	}
	if scratch != nil {
		base := 0x208 + len(surfaces)*0x30
		binary.LittleEndian.PutUint32(gen[base+0x00:], 1)
		binary.LittleEndian.PutUint32(gen[base+0x04:], scratchID)
		binary.LittleEndian.PutUint32(gen[base+0x08:], 0)
		binary.LittleEndian.PutUint32(gen[base+0x10:], 1)
		binary.LittleEndian.PutUint64(gen[base+0x20:], scratch.size)
		binary.LittleEndian.PutUint32(gen[base+0x28:], 0xffff)
	}

	// 2. OPERATION
	type barBinding struct {
		slot uint32
		tag  uint32
	}
	var bindings []barBinding
	hasKernelBinding := false
	for _, slot := range barSlots {
		off := int(0x10 * (slot + 1))
		if off+8 > len(progDesc) {
			return nil, fmt.Errorf("BAR slot %d offset 0x%x out of program descriptor range", slot, off)
		}
		vmBase := binary.LittleEndian.Uint64(progDesc[off:])
		if vmBase == 0 {
			continue
		}
		found := false
		var tag uint32
		for i, s := range surfaces {
			if s.addr == vmBase {
				tag = surfIDs[i]
				found = true
				break
			}
		}
		if !found && scratch != nil && scratch.addr == vmBase {
			tag = scratchID
			found = true
		}
		if !found && kernelAddr != nil && *kernelAddr == vmBase {
			tag = ResourceKernel
			hasKernelBinding = true
			found = true
		}
		if !found {
			return nil, fmt.Errorf("BAR slot %d VM base 0x%x does not match any resource", slot, vmBase)
		}
		bindings = append(bindings, barBinding{slot: slot, tag: tag})
	}
	if len(bindings) == 0 {
		return nil, errors.New("task stream references no non-zero BAR slots")
	}
	if kernelAddr != nil && !hasKernelBinding {
		return nil, errors.New("HWX has __KERN_0 but no kernel BAR binding")
	}
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].slot < bindings[j].slot })

	op := make([]byte, 0x410)
	binary.LittleEndian.PutUint32(op[0x00:], 1)
	binary.LittleEndian.PutUint32(op[0x0c:], uint32(len(bindings)))
	for j, b := range bindings {
		binary.LittleEndian.PutUint32(op[0x10+8*j:], b.slot)
		binary.LittleEndian.PutUint32(op[0x14+8*j:], b.tag)
	}

	// 3. PROCEDURE
	proc := make([]byte, 0x38)
	binary.LittleEndian.PutUint32(proc[0x00:], 1)
	binary.LittleEndian.PutUint32(proc[0x04:], 0)
	binary.LittleEndian.PutUint64(proc[0x08:], 0x18)
	binary.LittleEndian.PutUint64(proc[0x10:], 0x20)
	procWords := [8]uint32{1, 3, 0, 0, 1, 0, 0xffffffff, 4}
	for i, w := range procWords {
		binary.LittleEndian.PutUint32(proc[0x18+4*i:], w)
	}

	// 4. TEXT_PROPERTIES
	tprop := make([]byte, 0x28)
	binary.LittleEndian.PutUint32(tprop[0x00:], 1)
	binary.LittleEndian.PutUint32(tprop[0x0c:], taskCount)
	binary.LittleEndian.PutUint64(tprop[0x20:], textSec.size)

	return map[uint32][]byte{
		SectionGeneric:        gen,
		SectionOperation:      op,
		SectionProcedure:      proc,
		SectionTextProperties: tprop,
	}, nil
}

// LoadSectionsFromDir reads the 4 host-built LOAD sections from a directory.
func LoadSectionsFromDir(dir string) (map[uint32][]byte, error) {
	out := make(map[uint32][]byte, 4)
	for kind, fname := range SectionFilenames {
		b, err := os.ReadFile(filepath.Join(dir, fname))
		if err != nil {
			return nil, fmt.Errorf("read section %s: %w", fname, err)
		}
		out[kind] = b
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Transport interface & Linux accel/ane implementation
// ---------------------------------------------------------------------------

// DeviceCaps describes capabilities returned by DRM_IOCTL_ANE_GET_CAPS.
type DeviceCaps struct {
	Path        string `json:"path"`
	Size        uint32 `json:"size"`
	ABIVersion  uint32 `json:"abi_version"`
	ChipFamily  uint32 `json:"chip_family"`
	SectionSize uint32 `json:"section_size"`
	BindSize    uint32 `json:"bind_size"`
	ExecIOSize  uint32 `json:"exec_io_size"`
}

// IsKnownANE reports whether caps identifies a recognized Linux accel/ane ABI and chip family.
func IsKnownANE(caps DeviceCaps) bool {
	switch caps.ABIVersion {
	case ABIV1, ABIV2, ABIV3:
	default:
		return false
	}
	switch caps.ChipFamily {
	case ChipH13, ChipH14, ChipH16, ChipH17:
		return true
	default:
		return false
	}
}

// LoadedProgram describes surface metadata returned by DRM_IOCTL_ANE_H16_LOAD.
type LoadedProgram struct {
	SurfaceCount      int
	SurfaceSizes      []uint64
	SurfaceDirections []uint32
}

// Transport abstracts OS-level accel/ane operations for offline mock testing.
type Transport interface {
	ListDevices() ([]string, error)
	Open(path string) (int, error)
	Close(fd int) error
	GetCaps(fd int) (GetCapsArgs, error)
	BoInit(fd int, size uint64) (BoInitArgs, error)
	BoFree(fd int, handle uint32) error
	Submit(fd int, args SubmitArgs) error
	H16Load(fd int, hwx []byte, sections [4]struct {
		Kind uint32
		Data []byte
	}) (H16LoadArgs, error)
	H16Exec(fd int, bufs [][]byte, surfaceCount int, inPlace bool) error
	Mmap(fd int, offset int64, length int, prot int, flags int) ([]byte, error)
	Munmap(b []byte) error
	PageSize() int
}

// LinuxTransport implements Transport against real Linux /dev/accel/accel* nodes.
type LinuxTransport struct{}

func (LinuxTransport) ListDevices() ([]string, error) {
	matches, err := filepath.Glob("/dev/accel/accel*")
	if err != nil {
		return nil, err
	}
	sort.Strings(matches)
	return matches, nil
}

func (LinuxTransport) Open(path string) (int, error) {
	return syscall.Open(path, syscall.O_RDWR|syscall.O_CLOEXEC, 0)
}

func (LinuxTransport) Close(fd int) error {
	return syscall.Close(fd)
}

func linuxIoctl(fd int, req uintptr, arg unsafe.Pointer) error {
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), req, uintptr(arg))
	if errno != 0 {
		return errno
	}
	return nil
}

func (LinuxTransport) GetCaps(fd int) (GetCapsArgs, error) {
	var args GetCapsArgs
	if err := linuxIoctl(fd, IoctlGetCaps, unsafe.Pointer(&args)); err != nil {
		return GetCapsArgs{}, err
	}
	return args, nil
}

func (LinuxTransport) H16Load(fd int, hwx []byte, sections [4]struct {
	Kind uint32
	Data []byte
}) (H16LoadArgs, error) {
	var secEntries [4]H16Section
	for i := range sections {
		secEntries[i] = H16Section{
			Kind:    sections[i].Kind,
			Size:    uint32(len(sections[i].Data)),
			DataPtr: uint64(uintptr(unsafe.Pointer(&sections[i].Data[0]))),
		}
	}
	args := H16LoadArgs{
		HwxPtr:       uint64(uintptr(unsafe.Pointer(&hwx[0]))),
		HwxSize:      uint64(len(hwx)),
		SectionsPtr:  uint64(uintptr(unsafe.Pointer(&secEntries[0]))),
		SectionCount: SectionsCount,
	}
	err := linuxIoctl(fd, IoctlH16Load, unsafe.Pointer(&args))
	runtime.KeepAlive(hwx)
	runtime.KeepAlive(&sections)
	runtime.KeepAlive(&secEntries)
	return args, err
}

func (LinuxTransport) H16Exec(fd int, bufs [][]byte, surfaceCount int, inPlace bool) error {
	if inPlace {
		execArgs := H16ExecArgs{
			BuffersPtr: 0,
			Count:      uint32(surfaceCount),
			Flags:      ExecInPlace,
		}
		return linuxIoctl(fd, IoctlH16Exec, unsafe.Pointer(&execArgs))
	}
	if len(bufs) == 0 {
		return errors.New("H16Exec requires at least 1 buffer in copy mode")
	}
	entries := make([]H16Buffer, len(bufs))
	for i := range bufs {
		if len(bufs[i]) == 0 {
			return fmt.Errorf("H16Exec buffer %d is empty", i)
		}
		entries[i] = H16Buffer{
			Ptr:  uint64(uintptr(unsafe.Pointer(&bufs[i][0]))),
			Size: uint64(len(bufs[i])),
		}
	}
	execArgs := H16ExecArgs{
		BuffersPtr: uint64(uintptr(unsafe.Pointer(&entries[0]))),
		Count:      uint32(surfaceCount),
		Flags:      0,
	}
	err := linuxIoctl(fd, IoctlH16Exec, unsafe.Pointer(&execArgs))
	runtime.KeepAlive(bufs)
	runtime.KeepAlive(entries)
	return err
}

func (LinuxTransport) Mmap(fd int, offset int64, length int, prot int, flags int) ([]byte, error) {
	return syscall.Mmap(fd, offset, length, prot, flags)
}

func (LinuxTransport) Munmap(b []byte) error {
	return syscall.Munmap(b)
}

func (LinuxTransport) PageSize() int {
	return os.Getpagesize()
}

// QueryCaps issues DRM_IOCTL_ANE_GET_CAPS on an open accel fd.
func QueryCaps(tr Transport, fd int, path string) (DeviceCaps, error) {
	args, err := tr.GetCaps(fd)
	if err != nil {
		return DeviceCaps{}, err
	}
	return DeviceCaps{
		Path:        path,
		Size:        args.Size,
		ABIVersion:  args.ABIVersion,
		ChipFamily:  args.ChipFamily,
		SectionSize: args.SectionSize,
		BindSize:    args.BindSize,
		ExecIOSize:  args.ExecIOSize,
	}, nil
}

// ProbeDevicesFilter enumerates accel nodes and returns all known ANE devices across
// ABI v1, v2, and v3 (or only those matching abiFilter when abiFilter != 0).
func ProbeDevicesFilter(tr Transport, devicePath string, abiFilter uint32) ([]DeviceCaps, error) {
	var candidates []string
	if devicePath != "" {
		candidates = []string{devicePath}
	} else {
		var err error
		candidates, err = tr.ListDevices()
		if err != nil {
			return nil, err
		}
	}

	var found []DeviceCaps
	for _, path := range candidates {
		fd, err := tr.Open(path)
		if err != nil {
			if errors.Is(err, syscall.ENOENT) || errors.Is(err, os.ErrNotExist) {
				if devicePath == "" {
					continue
				}
				return nil, fmt.Errorf("%w: %s", ErrNoMatchingDevice, path)
			}
			return nil, fmt.Errorf("open %s: %w", path, err)
		}
		caps, err := QueryCaps(tr, fd, path)
		_ = tr.Close(fd)
		if err != nil {
			continue
		}
		if !IsKnownANE(caps) {
			continue
		}
		if abiFilter != 0 && caps.ABIVersion != abiFilter {
			continue
		}
		found = append(found, caps)
	}
	return found, nil
}

// ProbeDevices enumerates accel nodes and returns all known ANE devices across ABI v1, v2, and v3.
func ProbeDevices(tr Transport, devicePath string) ([]DeviceCaps, error) {
	return ProbeDevicesFilter(tr, devicePath, 0)
}

// DiscoverDeviceABI opens and returns (fd, DeviceCaps) for the first matching accel/ane node.
// expectedABI == 0 accepts any known ABI; expectedChip == 0 accepts any known chip family.
func DiscoverDeviceABI(tr Transport, devicePath string, expectedABI uint32, expectedChip uint32) (int, DeviceCaps, error) {
	var candidates []string
	if devicePath != "" {
		candidates = []string{devicePath}
	} else {
		var err error
		candidates, err = tr.ListDevices()
		if err != nil {
			return -1, DeviceCaps{}, err
		}
	}
	if len(candidates) == 0 {
		return -1, DeviceCaps{}, fmt.Errorf("%w: no /dev/accel/accel* devices found", ErrNoMatchingDevice)
	}

	for _, path := range candidates {
		fd, err := tr.Open(path)
		if err != nil {
			if errors.Is(err, syscall.ENOENT) || errors.Is(err, os.ErrNotExist) {
				if devicePath == "" {
					continue
				}
				return -1, DeviceCaps{}, fmt.Errorf("%w: device %q does not exist", ErrNoMatchingDevice, path)
			}
			return -1, DeviceCaps{}, fmt.Errorf("open %s: %w", path, err)
		}

		caps, err := QueryCaps(tr, fd, path)
		if err != nil {
			_ = tr.Close(fd)
			continue
		}
		if !IsKnownANE(caps) {
			_ = tr.Close(fd)
			continue
		}
		if expectedABI != 0 && caps.ABIVersion != expectedABI {
			_ = tr.Close(fd)
			continue
		}
		if expectedChip != 0 && caps.ChipFamily != expectedChip {
			_ = tr.Close(fd)
			continue
		}
		return fd, caps, nil
	}
	return -1, DeviceCaps{}, ErrNoMatchingDevice
}

// DiscoverDevice opens and returns (fd, DeviceCaps) for the first matching ABI v3 accel/ane node.
// expectedChip == 0 accepts either H16 or H17.
func DiscoverDevice(tr Transport, devicePath string, expectedChip uint32) (int, DeviceCaps, error) {
	return DiscoverDeviceABI(tr, devicePath, ABIV3, expectedChip)
}

// LoadProgram issues DRM_IOCTL_ANE_H16_LOAD with the HWX and 4 host-built LOAD sections.
func LoadProgram(tr Transport, fd int, hwx []byte, sections map[uint32][]byte) (LoadedProgram, error) {
	if len(hwx) == 0 {
		return LoadedProgram{}, errors.New("empty HWX buffer")
	}
	kinds := [4]uint32{SectionGeneric, SectionOperation, SectionProcedure, SectionTextProperties}
	var ordered [4]struct {
		Kind uint32
		Data []byte
	}
	for i, kind := range kinds {
		data, ok := sections[kind]
		if !ok {
			return LoadedProgram{}, fmt.Errorf("missing required LOAD section kind %d", kind)
		}
		if len(data) < SectionMin || len(data) > SectionMax || len(data)%4 != 0 {
			return LoadedProgram{}, fmt.Errorf("section kind %d has invalid size %d", kind, len(data))
		}
		ordered[i].Kind = kind
		ordered[i].Data = data
	}

	args, err := tr.H16Load(fd, hwx, ordered)
	if err != nil {
		return LoadedProgram{}, fmt.Errorf("DRM_IOCTL_ANE_H16_LOAD failed: %w", err)
	}

	sc := int(args.SurfaceCount)
	if sc < 2 || sc > MaxSurfaces {
		return LoadedProgram{}, fmt.Errorf("kernel returned invalid surface_count %d", sc)
	}
	sizes := make([]uint64, sc)
	dirs := make([]uint32, sc)
	for i := 0; i < sc; i++ {
		if args.SurfaceSize[i] == 0 {
			return LoadedProgram{}, fmt.Errorf("kernel returned zero size for surface %d", i)
		}
		sizes[i] = args.SurfaceSize[i]
		dirs[i] = args.SurfaceDirection[i]
	}
	return LoadedProgram{
		SurfaceCount:      sc,
		SurfaceSizes:      sizes,
		SurfaceDirections: dirs,
	}, nil
}

// ExecOptions configures program execution and optional reference checking.
type ExecOptions struct {
	Runs      int
	Warmup    int
	InPlace   bool
	AltInputs [][]byte // Optional alternate input surfaces used on odd runs (run 2, 4, ...) when Runs > 1.
	Expected  []byte
	Atol      float64
	Rtol      float64
	MaxULP    *int
}

// RunReport summarizes device execution, repeatability, latency, and optional comparison.
type RunReport struct {
	Device            string                `json:"device"`
	ABIVersion        uint32                `json:"abi_version"`
	ChipFamily        uint32                `json:"chip_family"`
	Mode              string                `json:"mode"`
	Runs              int                   `json:"runs"`
	Warmup            int                   `json:"warmup"`
	SurfaceCount      int                   `json:"surface_count"`
	SurfaceSizes      []uint64              `json:"surface_sizes"`
	SurfaceDirections []uint32              `json:"surface_directions"`
	OutputSize        uint64                `json:"output_size"`
	OutputSHA256      string                `json:"output_sha256"`
	AltOutputSHA256   string                `json:"alt_output_sha256,omitempty"`
	LatenciesMS       []float64             `json:"latencies_ms"`
	MinMS             float64               `json:"min_ms"`
	MedianMS          float64               `json:"median_ms"`
	P95MS             float64               `json:"p95_ms"`
	MeanMS            float64               `json:"mean_ms"`
	Repeatable        bool                  `json:"repeatable"`
	Comparison        *ref.ComparisonResult `json:"comparison"`
}

// ComputeLatencyStats computes (min, median, p95, mean) in milliseconds over a non-empty slice of latencies.
// p95 uses standard linear interpolation across sorted samples (matching NumPy's default percentile method).
func ComputeLatencyStats(latencies []float64) (minMS, medianMS, p95MS, meanMS float64) {
	n := len(latencies)
	if n == 0 {
		return 0, 0, 0, 0
	}
	sorted := make([]float64, n)
	sum := 0.0
	for i, v := range latencies {
		sorted[i] = v
		sum += v
	}
	sort.Float64s(sorted)
	minMS = sorted[0]
	meanMS = sum / float64(n)
	if n%2 == 1 {
		medianMS = sorted[n/2]
	} else {
		medianMS = 0.5 * (sorted[n/2-1] + sorted[n/2])
	}
	if n == 1 {
		p95MS = sorted[0]
	} else {
		pos := 0.95 * float64(n-1)
		lo := int(math.Floor(pos))
		hi := int(math.Ceil(pos))
		frac := pos - float64(lo)
		p95MS = sorted[lo] + frac*(sorted[hi]-sorted[lo])
	}
	return minMS, medianMS, p95MS, meanMS
}

// ExecuteProgram executes a loaded program opts.Warmup warmup runs followed by opts.Runs timed runs
// in copy or in-place mmap mode. Copy mode poisons the output buffer with 0xff before every run.
// Repeatability is checked across ALL (Warmup + Runs) executions, while LatenciesMS and summary stats
// (MinMS, MedianMS, P95MS, MeanMS) are computed strictly over the opts.Runs timed runs.
// When opts.AltInputs is non-empty and opts.Runs > 1, even runs (r=0,2,4,...) execute
// inputs and odd runs (r=1,3,5,...) execute opts.AltInputs, checking independent
// determinism for both input sets.
func ExecuteProgram(
	tr Transport,
	fd int,
	caps DeviceCaps,
	loaded LoadedProgram,
	inputs [][]byte,
	opts ExecOptions,
) ([]byte, RunReport, error) {
	runs := opts.Runs
	if runs <= 0 {
		runs = 1
	}
	warmup := opts.Warmup
	if warmup < 0 {
		return nil, RunReport{}, fmt.Errorf("--warmup must be >= 0, got %d", warmup)
	}
	totalRuns := warmup + runs

	var inIndices []int
	outIdx := -1
	for i, d := range loaded.SurfaceDirections {
		switch d {
		case 1:
			inIndices = append(inIndices, i)
		case 2:
			if outIdx != -1 {
				return nil, RunReport{}, errors.New("expected exactly 1 output surface, found multiple")
			}
			outIdx = i
		}
	}
	if outIdx < 0 {
		return nil, RunReport{}, errors.New("program has no output surface")
	}
	outSize := int(loaded.SurfaceSizes[outIdx])

	resolvedInputs := make([][]byte, len(inIndices))
	if len(inputs) == 0 {
		for k, sIdx := range inIndices {
			resolvedInputs[k] = make([]byte, loaded.SurfaceSizes[sIdx])
		}
	} else {
		if len(inputs) != len(inIndices) {
			return nil, RunReport{}, fmt.Errorf("provided %d input buffers, program expects %d", len(inputs), len(inIndices))
		}
		for k, sIdx := range inIndices {
			expSz := int(loaded.SurfaceSizes[sIdx])
			if len(inputs[k]) != expSz {
				return nil, RunReport{}, fmt.Errorf("input surface %d size mismatch: got %d, expected %d", sIdx, len(inputs[k]), expSz)
			}
			resolvedInputs[k] = inputs[k]
		}
	}

	var resolvedAltInputs [][]byte
	if len(opts.AltInputs) > 0 {
		if runs < 2 {
			return nil, RunReport{}, fmt.Errorf("--alt-input requires --runs >= 2, got %d", runs)
		}
		if len(opts.AltInputs) != len(inIndices) {
			return nil, RunReport{}, fmt.Errorf("provided %d alt-input buffers, program expects %d", len(opts.AltInputs), len(inIndices))
		}
		resolvedAltInputs = make([][]byte, len(inIndices))
		for k, sIdx := range inIndices {
			expSz := int(loaded.SurfaceSizes[sIdx])
			if len(opts.AltInputs[k]) != expSz {
				return nil, RunReport{}, fmt.Errorf("alt-input surface %d size mismatch: got %d, expected %d", sIdx, len(opts.AltInputs[k]), expSz)
			}
			resolvedAltInputs[k] = opts.AltInputs[k]
		}
	}

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

	if opts.InPlace {
		psize := tr.PageSize()
		if psize != PageSize16K {
			return nil, RunReport{}, fmt.Errorf("--in-place requires 16 KiB host page size (%d), got %d", PageSize16K, psize)
		}
		mappings := make([][]byte, loaded.SurfaceCount)
		defer func() {
			for _, mm := range mappings {
				if mm != nil {
					_ = tr.Munmap(mm)
				}
			}
		}()
		for i := 0; i < loaded.SurfaceCount; i++ {
			mapLen := ref.AlignUp(int(loaded.SurfaceSizes[i]), PageSize16K)
			prot := syscall.PROT_READ
			if loaded.SurfaceDirections[i] == 1 {
				prot = syscall.PROT_READ | syscall.PROT_WRITE
			}
			mm, err := tr.Mmap(fd, MmapOffset(i), mapLen, prot, syscall.MAP_SHARED)
			if err != nil {
				return nil, RunReport{}, fmt.Errorf("mmap surface %d: %w", i, err)
			}
			mappings[i] = mm
		}

		for r := 0; r < totalRuns; r++ {
			isAlt := resolvedAltInputs != nil && (r%2 == 1)
			curInputs := resolvedInputs
			if isAlt {
				curInputs = resolvedAltInputs
			}
			for k, sIdx := range inIndices {
				copy(mappings[sIdx][:len(curInputs[k])], curInputs[k])
			}
			t0 := time.Now()
			err := tr.H16Exec(fd, nil, loaded.SurfaceCount, true)
			dt := float64(time.Since(t0).Nanoseconds()) / 1e6
			if err != nil {
				return nil, RunReport{}, fmt.Errorf("DRM_IOCTL_ANE_H16_EXEC (in-place run %d): %w", r+1, err)
			}
			if r >= warmup {
				latencies = append(latencies, dt)
			}

			outBytes := make([]byte, outSize)
			copy(outBytes, mappings[outIdx][:outSize])
			if err := recordOutput(r, isAlt, outBytes); err != nil {
				return nil, RunReport{}, err
			}
		}
	} else {
		// Copy mode: allocate per-surface byte buffers and poison output with 0xff before every run.
		surfBufs := make([][]byte, loaded.SurfaceCount)
		for i := 0; i < loaded.SurfaceCount; i++ {
			surfBufs[i] = make([]byte, int(loaded.SurfaceSizes[i]))
		}

		for r := 0; r < totalRuns; r++ {
			isAlt := resolvedAltInputs != nil && (r%2 == 1)
			curInputs := resolvedInputs
			if isAlt {
				curInputs = resolvedAltInputs
			}
			for k, sIdx := range inIndices {
				copy(surfBufs[sIdx], curInputs[k])
			}
			for j := range surfBufs[outIdx] {
				surfBufs[outIdx][j] = 0xff
			}
			t0 := time.Now()
			err := tr.H16Exec(fd, surfBufs, loaded.SurfaceCount, false)
			dt := float64(time.Since(t0).Nanoseconds()) / 1e6
			if err != nil {
				return nil, RunReport{}, fmt.Errorf("DRM_IOCTL_ANE_H16_EXEC (copy run %d): %w", r+1, err)
			}
			if r >= warmup {
				latencies = append(latencies, dt)
			}

			outBytes := make([]byte, outSize)
			copy(outBytes, surfBufs[outIdx])
			if err := recordOutput(r, isAlt, outBytes); err != nil {
				return nil, RunReport{}, err
			}
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

	mode := "copy"
	if opts.InPlace {
		mode = "in-place"
	}
	report := RunReport{
		Device:            caps.Path,
		ABIVersion:        caps.ABIVersion,
		ChipFamily:        caps.ChipFamily,
		Mode:              mode,
		Runs:              runs,
		Warmup:            warmup,
		SurfaceCount:      loaded.SurfaceCount,
		SurfaceSizes:      loaded.SurfaceSizes,
		SurfaceDirections: loaded.SurfaceDirections,
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
