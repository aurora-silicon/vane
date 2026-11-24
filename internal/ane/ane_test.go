package ane

import (
	"bytes"
	"errors"
	"syscall"
	"testing"
	"unsafe"
)

func TestUapiLayoutAndIoctlNumbers(t *testing.T) {
	if got := unsafe.Sizeof(GetCapsArgs{}); got != 32 {
		t.Fatalf("sizeof(GetCapsArgs) = %d, want 32", got)
	}
	if got := unsafe.Sizeof(H16Section{}); got != 16 {
		t.Fatalf("sizeof(H16Section) = %d, want 16", got)
	}
	if got := unsafe.Sizeof(H16LoadArgs{}); got != 88 {
		t.Fatalf("sizeof(H16LoadArgs) = %d, want 88", got)
	}
	if got := unsafe.Sizeof(H16Buffer{}); got != 16 {
		t.Fatalf("sizeof(H16Buffer) = %d, want 16", got)
	}
	if got := unsafe.Sizeof(H16ExecArgs{}); got != 16 {
		t.Fatalf("sizeof(H16ExecArgs) = %d, want 16", got)
	}

	wantGetCaps := uintptr((3 << 30) | (32 << 16) | (0x64 << 8) | (0x40 + 0x00))
	wantLoad := uintptr((3 << 30) | (88 << 16) | (0x64 << 8) | (0x40 + 0x07))
	wantExec := uintptr((1 << 30) | (16 << 16) | (0x64 << 8) | (0x40 + 0x08))
	if IoctlGetCaps != wantGetCaps {
		t.Fatalf("IoctlGetCaps = 0x%x, want 0x%x", IoctlGetCaps, wantGetCaps)
	}
	if IoctlH16Load != wantLoad {
		t.Fatalf("IoctlH16Load = 0x%x, want 0x%x", IoctlH16Load, wantLoad)
	}
	if IoctlH16Exec != wantExec {
		t.Fatalf("IoctlH16Exec = 0x%x, want 0x%x", IoctlH16Exec, wantExec)
	}
	if MmapOffset(0) != 1<<32 || MmapOffset(3) != 4<<32 {
		t.Fatalf("unexpected MmapOffset values")
	}
}

func TestDiscoverProbeAndMockExecution(t *testing.T) {
	// Empty device list -> ErrNoMatchingDevice
	emptyTr := NewMockTransport(map[string]MockDeviceSpec{}, nil)
	if _, _, err := DiscoverDevice(emptyTr, "", ChipH17); !errors.Is(err, ErrNoMatchingDevice) {
		t.Fatalf("expected ErrNoMatchingDevice, got %v", err)
	}

	// Permission error does not masquerade as ErrNoMatchingDevice
	permTr := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {OpenErr: syscall.EACCES},
	}, nil)
	if _, _, err := DiscoverDevice(permTr, "", ChipH17); err == nil || errors.Is(err, ErrNoMatchingDevice) {
		t.Fatalf("expected EACCES error, got %v", err)
	}

	// Probe skips non-ANE devices and discovers H17 node
	tr := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {GetCapsErr: syscall.ENOTTY},
		"/dev/accel/accel1": {ABIVersion: ABIV3, ChipFamily: ChipH17},
	}, nil)
	devs, err := ProbeDevices(tr, "")
	if err != nil || len(devs) != 1 || devs[0].Path != "/dev/accel/accel1" || devs[0].ChipFamily != ChipH17 {
		t.Fatalf("ProbeDevices unexpected result: devs=%+v err=%v", devs, err)
	}

	// Mock copy + poison + repeatability + in-place page-size check
	fd, caps, err := DiscoverDevice(tr, "", ChipH17)
	if err != nil {
		t.Fatalf("DiscoverDevice failed: %v", err)
	}
	defer tr.Close(fd)

	loaded := LoadedProgram{
		SurfaceCount:      2,
		SurfaceSizes:      []uint64{64, 64},
		SurfaceDirections: []uint32{1, 2},
	}
	tr.openFDs[fd].loaded = &loaded
	tr.openFDs[fd].surfaces = [][]byte{make([]byte, PageSize16K), make([]byte, PageSize16K)}

	inData := make([]byte, 64)
	for i := range inData {
		inData[i] = byte(i + 1)
	}
	outBytes, rep, err := ExecuteProgram(tr, fd, caps, loaded, [][]byte{inData}, ExecOptions{Runs: 3})
	if err != nil {
		t.Fatalf("ExecuteProgram copy mode failed: %v", err)
	}
	if !bytes.Equal(outBytes, inData) || !rep.Repeatable || len(rep.LatenciesMS) != 3 {
		t.Fatalf("unexpected copy mode report: %+v", rep)
	}
	if len(tr.ObservedOutputPoisonOK) != 3 || !tr.ObservedOutputPoisonOK[0] || !tr.ObservedOutputPoisonOK[1] || !tr.ObservedOutputPoisonOK[2] {
		t.Fatalf("expected 0xff output poison before every run, got %v", tr.ObservedOutputPoisonOK)
	}

	// In-place mode succeeds on 16 KiB host page size
	outIP, repIP, err := ExecuteProgram(tr, fd, caps, loaded, [][]byte{inData}, ExecOptions{Runs: 2, InPlace: true})
	if err != nil || !bytes.Equal(outIP, inData) || repIP.Mode != "in-place" {
		t.Fatalf("in-place mode failed: err=%v rep=%+v", err, repIP)
	}

	// In-place mode rejects 4 KiB host page size
	tr.HostPageSize = 4096
	if _, _, err := ExecuteProgram(tr, fd, caps, loaded, [][]byte{inData}, ExecOptions{Runs: 1, InPlace: true}); err == nil {
		t.Fatalf("expected 4 KiB in-place rejection")
	}
	tr.HostPageSize = PageSize16K

	// Non-deterministic repeatability failure detection
	tr.ComputeFn = func(inputs [][]byte, call int) []byte {
		cp := make([]byte, len(inputs[0]))
		copy(cp, inputs[0])
		if call%2 == 0 {
			cp[0] ^= 1
		}
		return cp
	}
	if _, _, err := ExecuteProgram(tr, fd, caps, loaded, [][]byte{inData}, ExecOptions{Runs: 2}); err == nil {
		t.Fatalf("expected non-deterministic output error")
	}

	// LinuxTransport.H16Exec rejects empty buffer slice and empty individual buffers without panicking
	var lt LinuxTransport
	if err := lt.H16Exec(-1, nil, 0, false); err == nil {
		t.Fatalf("expected error on empty bufs in LinuxTransport.H16Exec")
	}
	if err := lt.H16Exec(-1, [][]byte{make([]byte, 64), {}}, 2, false); err == nil {
		t.Fatalf("expected error on zero-length buffer in LinuxTransport.H16Exec")
	}
}

type zeroSizeLoadTransport struct {
	*MockTransport
}

func (z zeroSizeLoadTransport) H16Load(fd int, hwx []byte, sections [4]struct {
	Kind uint32
	Data []byte
}) (H16LoadArgs, error) {
	return H16LoadArgs{
		SurfaceCount:     2,
		SurfaceSize:      [MaxSurfaces]uint64{64, 0},
		SurfaceDirection: [MaxSurfaces]uint32{1, 2},
	}, nil
}

func TestLoadProgramRejectsZeroSurfaceSize(t *testing.T) {
	base := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ABIV3, ChipFamily: ChipH17},
	}, nil)
	tr := zeroSizeLoadTransport{MockTransport: base}
	fd, _, err := DiscoverDevice(tr, "", ChipH17)
	if err != nil {
		t.Fatalf("DiscoverDevice failed: %v", err)
	}
	defer tr.Close(fd)
	secs := map[uint32][]byte{
		SectionGeneric:        make([]byte, SectionMin),
		SectionOperation:      make([]byte, SectionMin),
		SectionProcedure:      make([]byte, SectionMin),
		SectionTextProperties: make([]byte, SectionMin),
	}
	if _, err := LoadProgram(tr, fd, []byte{1, 2, 3, 4}, secs); err == nil {
		t.Fatalf("expected LoadProgram to reject zero SurfaceSize[1]")
	}
}
