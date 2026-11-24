package ane

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"sort"
	"syscall"
	"unsafe"

	"github.com/niteria/vane/internal/ref"
)

// MockDeviceSpec configures one virtual /dev/accel/accel* node for MockTransport.
type MockDeviceSpec struct {
	ABIVersion uint32
	ChipFamily uint32
	OpenErr    error
	GetCapsErr error
}

type mockFDState struct {
	path     string
	spec     MockDeviceSpec
	loaded   *LoadedProgram
	surfaces [][]byte
}

// MockTransport implements Transport in memory against the Linux accel/ane ABI v3 contract.
type MockTransport struct {
	Devices                map[string]MockDeviceSpec
	ComputeFn              func(inputs [][]byte, call int) []byte
	HostPageSize           int
	nextFD                 int
	openFDs                map[int]*mockFDState
	ObservedOutputPoisonOK []bool
	LoadedSections         map[uint32][]byte
	ExecCalls              int
}

// NewMockTransport creates an in-memory ABI v3 accel/ane transport with 16 KiB default pages.
func NewMockTransport(devices map[string]MockDeviceSpec, compute func([][]byte, int) []byte) *MockTransport {
	return &MockTransport{
		Devices:        devices,
		ComputeFn:      compute,
		HostPageSize:   PageSize16K,
		nextFD:         100,
		openFDs:        map[int]*mockFDState{},
		LoadedSections: map[uint32][]byte{},
	}
}

func (m *MockTransport) ListDevices() ([]string, error) {
	out := make([]string, 0, len(m.Devices))
	for k := range m.Devices {
		out = append(out, k)
	}
	sort.Strings(out)
	return out, nil
}

func (m *MockTransport) Open(path string) (int, error) {
	spec, ok := m.Devices[path]
	if !ok {
		return -1, syscall.ENOENT
	}
	if spec.OpenErr != nil {
		return -1, spec.OpenErr
	}
	fd := m.nextFD
	m.nextFD++
	m.openFDs[fd] = &mockFDState{path: path, spec: spec}
	return fd, nil
}

func (m *MockTransport) Close(fd int) error {
	delete(m.openFDs, fd)
	return nil
}

func (m *MockTransport) PageSize() int {
	return m.HostPageSize
}

func (m *MockTransport) Mmap(fd int, offset int64, length int, prot int, _ int) ([]byte, error) {
	st, ok := m.openFDs[fd]
	if !ok || st.loaded == nil {
		return nil, syscall.EINVAL
	}
	idx := int((offset >> 32) - 1)
	if idx < 0 || idx >= st.loaded.SurfaceCount {
		return nil, syscall.EINVAL
	}
	if st.loaded.SurfaceDirections[idx] == 2 && (prot&syscall.PROT_WRITE) != 0 {
		return nil, syscall.EACCES
	}
	if length > len(st.surfaces[idx]) {
		return nil, syscall.EINVAL
	}
	return st.surfaces[idx][:length], nil
}

func (m *MockTransport) Munmap(_ []byte) error {
	return nil
}

func parseHWXSurfaces(hwx []byte) ([]uint64, []uint32, error) {
	if len(hwx) < 32 {
		return nil, nil, errors.New("truncated hwx")
	}
	ncmds := int(binary.LittleEndian.Uint32(hwx[0x10:]))
	cursor := 32
	var bufRefs []string
	type tdInfo struct {
		dir  uint32
		span uint64
	}
	tensors := map[string]tdInfo{}
	for i := 0; i < ncmds; i++ {
		cmd := binary.LittleEndian.Uint32(hwx[cursor:])
		cmdsize := int(binary.LittleEndian.Uint32(hwx[cursor+4:]))
		if cmd == 0x40 {
			raw := hwx[cursor+24 : cursor+32]
			nul := bytes.IndexByte(raw, 0)
			bufRefs = append(bufRefs, string(raw[:nul]))
		} else if cmd == 4 {
			subtype := binary.LittleEndian.Uint32(hwx[cursor+8:])
			if subtype == 3 {
				dir := binary.LittleEndian.Uint32(hwx[cursor+0x14:])
				n := uint64(binary.LittleEndian.Uint32(hwx[cursor+0x28:]))
				batchStride := binary.LittleEndian.Uint64(hwx[cursor+0x50:])
				shortOff := int(binary.LittleEndian.Uint32(hwx[cursor+0x7c:]))
				rest := hwx[cursor+shortOff : cursor+cmdsize]
				nul := bytes.IndexByte(rest, 0)
				sname := string(rest[:nul])
				tensors[sname] = tdInfo{dir: dir, span: n * batchStride}
			}
		}
		cursor += cmdsize
	}
	sizes := make([]uint64, len(bufRefs))
	dirs := make([]uint32, len(bufRefs))
	for i, nm := range bufRefs {
		td, ok := tensors[nm]
		if !ok {
			return nil, nil, fmt.Errorf("missing tensor for %s", nm)
		}
		sizes[i] = td.span
		dirs[i] = td.dir
	}
	return sizes, dirs, nil
}

func (m *MockTransport) GetCaps(fd int) (GetCapsArgs, error) {
	st, ok := m.openFDs[fd]
	if !ok {
		return GetCapsArgs{}, syscall.EBADF
	}
	if st.spec.GetCapsErr != nil {
		return GetCapsArgs{}, st.spec.GetCapsErr
	}
	return GetCapsArgs{
		Size:       uint32(unsafe.Sizeof(GetCapsArgs{})),
		ABIVersion: st.spec.ABIVersion,
		ChipFamily: st.spec.ChipFamily,
	}, nil
}

func (m *MockTransport) H16Load(fd int, hwx []byte, sections [4]struct {
	Kind uint32
	Data []byte
}) (H16LoadArgs, error) {
	st, ok := m.openFDs[fd]
	if !ok {
		return H16LoadArgs{}, syscall.EBADF
	}
	if st.loaded != nil {
		return H16LoadArgs{}, syscall.EBUSY
	}
	seen := map[uint32]bool{}
	for _, s := range sections {
		seen[s.Kind] = true
		cp := make([]byte, len(s.Data))
		copy(cp, s.Data)
		m.LoadedSections[s.Kind] = cp
	}
	if len(seen) != 4 {
		return H16LoadArgs{}, syscall.EINVAL
	}
	sizes, dirs, err := parseHWXSurfaces(hwx)
	if err != nil {
		return H16LoadArgs{}, syscall.EINVAL
	}
	st.loaded = &LoadedProgram{
		SurfaceCount:      len(sizes),
		SurfaceSizes:      sizes,
		SurfaceDirections: dirs,
	}
	st.surfaces = make([][]byte, len(sizes))
	var out H16LoadArgs
	out.SectionCount = SectionsCount
	out.SurfaceCount = uint32(len(sizes))
	for i, sz := range sizes {
		st.surfaces[i] = make([]byte, ref.AlignUp(int(sz), PageSize16K))
		out.SurfaceSize[i] = sz
		out.SurfaceDirection[i] = dirs[i]
	}
	return out, nil
}

func (m *MockTransport) H16Exec(fd int, bufs [][]byte, surfaceCount int, inPlace bool) error {
	st, ok := m.openFDs[fd]
	if !ok {
		return syscall.EBADF
	}
	if st.loaded == nil {
		return syscall.EINVAL
	}
	if surfaceCount != st.loaded.SurfaceCount {
		return syscall.EINVAL
	}
	m.ExecCalls++
	if inPlace {
		if bufs != nil {
			return syscall.EINVAL
		}
		var inputs [][]byte
		outIdx := -1
		for i, d := range st.loaded.SurfaceDirections {
			if d == 1 {
				cp := make([]byte, st.loaded.SurfaceSizes[i])
				copy(cp, st.surfaces[i][:st.loaded.SurfaceSizes[i]])
				inputs = append(inputs, cp)
			} else if d == 2 {
				outIdx = i
			}
		}
		outSz := int(st.loaded.SurfaceSizes[outIdx])
		outData := m.runCompute(inputs, outSz)
		copy(st.surfaces[outIdx][:outSz], outData)
		return nil
	}

	if len(bufs) != surfaceCount {
		return syscall.EINVAL
	}
	var inputs [][]byte
	var outDst []byte
	for i, raw := range bufs {
		if uint64(len(raw)) != st.loaded.SurfaceSizes[i] {
			return syscall.EINVAL
		}
		if st.loaded.SurfaceDirections[i] == 1 {
			cp := make([]byte, len(raw))
			copy(cp, raw)
			inputs = append(inputs, cp)
		} else {
			poisonOK := true
			for _, v := range raw {
				if v != 0xff {
					poisonOK = false
					break
				}
			}
			m.ObservedOutputPoisonOK = append(m.ObservedOutputPoisonOK, poisonOK)
			outDst = raw
		}
	}
	outData := m.runCompute(inputs, len(outDst))
	copy(outDst, outData)
	return nil
}

func (m *MockTransport) runCompute(inputs [][]byte, outSz int) []byte {
	if m.ComputeFn != nil {
		return m.ComputeFn(inputs, m.ExecCalls)
	}
	out := make([]byte, outSz)
	if len(inputs) > 0 {
		copy(out, inputs[0])
	}
	return out
}
