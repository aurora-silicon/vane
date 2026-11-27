package ane

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"syscall"
	"testing"
	"unsafe"
)

func TestUapiLayoutAndIoctlNumbers(t *testing.T) {
	if got := unsafe.Sizeof(GetCapsArgs{}); got != 32 {
		t.Fatalf("sizeof(GetCapsArgs) = %d, want 32", got)
	}
	if got := unsafe.Sizeof(BoInitArgs{}); got != 24 {
		t.Fatalf("sizeof(BoInitArgs) = %d, want 24", got)
	}
	if got := unsafe.Sizeof(BoFreeArgs{}); got != 8 {
		t.Fatalf("sizeof(BoFreeArgs) = %d, want 8", got)
	}
	if got := unsafe.Sizeof(SubmitArgs{}); got != 152 {
		t.Fatalf("sizeof(SubmitArgs) = %d, want 152", got)
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
	wantBoInit := uintptr((3 << 30) | (24 << 16) | (0x64 << 8) | (0x40 + 0x01))
	wantBoFree := uintptr((3 << 30) | (8 << 16) | (0x64 << 8) | (0x40 + 0x02))
	wantSubmit := uintptr((3 << 30) | (152 << 16) | (0x64 << 8) | (0x40 + 0x03))
	wantLoad := uintptr((3 << 30) | (88 << 16) | (0x64 << 8) | (0x40 + 0x07))
	wantExec := uintptr((1 << 30) | (16 << 16) | (0x64 << 8) | (0x40 + 0x08))
	if IoctlGetCaps != wantGetCaps {
		t.Fatalf("IoctlGetCaps = 0x%x, want 0x%x", IoctlGetCaps, wantGetCaps)
	}
	if IoctlBoInit != wantBoInit {
		t.Fatalf("IoctlBoInit = 0x%x, want 0x%x", IoctlBoInit, wantBoInit)
	}
	if IoctlBoFree != wantBoFree {
		t.Fatalf("IoctlBoFree = 0x%x, want 0x%x", IoctlBoFree, wantBoFree)
	}
	if IoctlSubmit != wantSubmit {
		t.Fatalf("IoctlSubmit = 0x%x, want 0x%x", IoctlSubmit, wantSubmit)
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

func TestComputeLatencyStatsAndWarmup(t *testing.T) {
	approxEq := func(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

	// Single sample
	mn, md, p95, mean := ComputeLatencyStats([]float64{4.25})
	if !approxEq(mn, 4.25) || !approxEq(md, 4.25) || !approxEq(p95, 4.25) || !approxEq(mean, 4.25) {
		t.Fatalf("single sample stats mismatch: min=%v med=%v p95=%v mean=%v", mn, md, p95, mean)
	}

	// Even count (unsorted input): [10, 2, 8, 4] -> sorted [2, 4, 8, 10]
	// min=2, median=6, p95 at pos=0.95*3=2.85 -> 8 + 0.85*(10-8) = 9.7, mean=6
	mn, md, p95, mean = ComputeLatencyStats([]float64{10, 2, 8, 4})
	if !approxEq(mn, 2.0) || !approxEq(md, 6.0) || !approxEq(p95, 9.7) || !approxEq(mean, 6.0) {
		t.Fatalf("even sample stats mismatch: min=%v med=%v p95=%v mean=%v", mn, md, p95, mean)
	}

	// Odd count 1..20 (21 values: 1..21) -> min=1, median=11, p95 at pos=0.95*20=19 -> 20, mean=11
	vals := make([]float64, 21)
	for i := range vals {
		vals[i] = float64(21 - i)
	}
	mn, md, p95, mean = ComputeLatencyStats(vals)
	if !approxEq(mn, 1.0) || !approxEq(md, 11.0) || !approxEq(p95, 20.0) || !approxEq(mean, 11.0) {
		t.Fatalf("21-sample stats mismatch: min=%v med=%v p95=%v mean=%v", mn, md, p95, mean)
	}

	// Warmup executes warmup+runs total calls, checks determinism on warmup too, and records only timed runs
	calls := 0
	tr := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ABIV3, ChipFamily: ChipH17},
	}, func(inputs [][]byte, _ int) []byte {
		calls++
		cp := make([]byte, len(inputs[0]))
		copy(cp, inputs[0])
		return cp
	})
	fd, caps, err := DiscoverDevice(tr, "", ChipH17)
	if err != nil {
		t.Fatalf("DiscoverDevice: %v", err)
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
	inData[0] = 0x42
	_, rep, err := ExecuteProgram(tr, fd, caps, loaded, [][]byte{inData}, ExecOptions{Runs: 5, Warmup: 3})
	if err != nil {
		t.Fatalf("ExecuteProgram with warmup failed: %v", err)
	}
	if calls != 8 {
		t.Fatalf("expected 8 total executions (3 warmup + 5 timed), got %d", calls)
	}
	if rep.Runs != 5 || rep.Warmup != 3 || len(rep.LatenciesMS) != 5 {
		t.Fatalf("unexpected warmup report counts: runs=%d warmup=%d len(lat)=%d", rep.Runs, rep.Warmup, len(rep.LatenciesMS))
	}
	if !(rep.MinMS <= rep.MedianMS && rep.MedianMS <= rep.P95MS) {
		t.Fatalf("latency ordering violated: min=%v med=%v p95=%v", rep.MinMS, rep.MedianMS, rep.P95MS)
	}

	// Non-determinism between a warmup run and a timed run must fail
	badCalls := 0
	tr.ComputeFn = func(inputs [][]byte, _ int) []byte {
		badCalls++
		cp := make([]byte, len(inputs[0]))
		copy(cp, inputs[0])
		if badCalls == 1 {
			cp[0] ^= 0x01 // corrupt warmup run 1
		}
		return cp
	}
	if _, _, err := ExecuteProgram(tr, fd, caps, loaded, [][]byte{inData}, ExecOptions{Runs: 2, Warmup: 1}); err == nil {
		t.Fatalf("expected non-determinism between warmup and timed run to fail")
	}

	// Negative warmup rejected
	if _, _, err := ExecuteProgram(tr, fd, caps, loaded, [][]byte{inData}, ExecOptions{Runs: 1, Warmup: -1}); err == nil {
		t.Fatalf("expected negative warmup to be rejected")
	}
}

func makeSyntheticAnec(hdr AnecHeader, payload []byte) []byte {
	buf := make([]byte, AnecHeaderSize+len(payload))
	binary.LittleEndian.PutUint64(buf[0x000:0x008], hdr.Size)
	binary.LittleEndian.PutUint32(buf[0x008:0x00c], hdr.TDSize)
	binary.LittleEndian.PutUint32(buf[0x00c:0x010], hdr.TDCount)
	binary.LittleEndian.PutUint64(buf[0x010:0x018], hdr.TskSize)
	binary.LittleEndian.PutUint64(buf[0x018:0x020], hdr.KrnSize)
	binary.LittleEndian.PutUint32(buf[0x020:0x024], hdr.SrcCount)
	binary.LittleEndian.PutUint32(buf[0x024:0x028], hdr.DstCount)
	for i := 0; i < TileCount; i++ {
		binary.LittleEndian.PutUint32(buf[0x028+4*i:], hdr.Tiles[i])
	}
	for i := 0; i < TileCount; i++ {
		for j := 0; j < 6; j++ {
			binary.LittleEndian.PutUint64(buf[0x0a8+(i*6+j)*8:], hdr.NCHW[i][j])
		}
	}
	copy(buf[AnecHeaderSize:], payload)
	return buf
}

func validSyntheticAnecHeader() (AnecHeader, []byte) {
	payload := make([]byte, 0x200)
	// TD0 word 0 with arbitrary high/low nibbles + non-FifoNID byte [23:16] to test PatchBootstrapNID.
	binary.LittleEndian.PutUint32(payload[0:4], 0xfa123456)
	var h AnecHeader
	h.Size = uint64(len(payload))
	h.TDSize = 0x100
	h.TDCount = 1
	h.TskSize = 0x100
	h.KrnSize = 0x100
	h.SrcCount = 1
	h.DstCount = 1
	h.Tiles[0] = 1 // 16 KiB cmd+weights
	h.Tiles[2] = 1 // optional scratch slot
	h.Tiles[4] = 1 // dst0 = 16 KiB
	h.Tiles[5] = 1 // src0 = 16 KiB
	h.NCHW[4] = [6]uint64{1, 4, 8, 16, 1024, 64}
	h.NCHW[5] = [6]uint64{1, 4, 8, 16, 1024, 64}
	return h, payload
}

func TestMultiABIProbeAndFilter(t *testing.T) {
	tr := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {GetCapsErr: syscall.ENOTTY},
		"/dev/accel/accel1": {ABIVersion: ABIV1, ChipFamily: ChipH13},
		"/dev/accel/accel2": {ABIVersion: ABIV2, ChipFamily: ChipH14, SectionSize: 24, BindSize: 24, ExecIOSize: 32},
		"/dev/accel/accel3": {ABIVersion: ABIV3, ChipFamily: ChipH16},
		"/dev/accel/accel4": {ABIVersion: ABIV3, ChipFamily: ChipH17},
	}, nil)

	all, err := ProbeDevices(tr, "")
	if err != nil {
		t.Fatalf("ProbeDevices failed: %v", err)
	}
	if len(all) != 4 {
		t.Fatalf("ProbeDevices returned %d devices, want 4: %+v", len(all), all)
	}
	if all[0].ABIVersion != ABIV1 || all[0].ChipFamily != ChipH13 {
		t.Fatalf("unexpected dev[0]: %+v", all[0])
	}
	if all[1].ABIVersion != ABIV2 || all[1].ChipFamily != ChipH14 || all[1].SectionSize != 24 || all[1].BindSize != 24 || all[1].ExecIOSize != 32 {
		t.Fatalf("unexpected dev[1]: %+v", all[1])
	}
	if all[2].ABIVersion != ABIV3 || all[2].ChipFamily != ChipH16 {
		t.Fatalf("unexpected dev[2]: %+v", all[2])
	}
	if all[3].ABIVersion != ABIV3 || all[3].ChipFamily != ChipH17 {
		t.Fatalf("unexpected dev[3]: %+v", all[3])
	}

	v1Devs, err := ProbeDevicesFilter(tr, "", ABIV1)
	if err != nil || len(v1Devs) != 1 || v1Devs[0].ChipFamily != ChipH13 {
		t.Fatalf("ProbeDevicesFilter(v1) = %+v, err=%v", v1Devs, err)
	}
	v2Devs, err := ProbeDevicesFilter(tr, "", ABIV2)
	if err != nil || len(v2Devs) != 1 || v2Devs[0].ChipFamily != ChipH14 {
		t.Fatalf("ProbeDevicesFilter(v2) = %+v, err=%v", v2Devs, err)
	}
	v3Devs, err := ProbeDevicesFilter(tr, "", ABIV3)
	if err != nil || len(v3Devs) != 2 {
		t.Fatalf("ProbeDevicesFilter(v3) = %+v, err=%v", v3Devs, err)
	}

	fd, caps, err := DiscoverDeviceABI(tr, "", ABIV1, ChipH13)
	if err != nil || caps.ABIVersion != ABIV1 || caps.ChipFamily != ChipH13 || caps.Path != "/dev/accel/accel1" {
		t.Fatalf("DiscoverDeviceABI(v1, H13) = (%d, %+v, %v)", fd, caps, err)
	}
	_ = tr.Close(fd)
}

func TestParseAnecValidation(t *testing.T) {
	baseHdr, basePayload := validSyntheticAnecHeader()
	validBytes := makeSyntheticAnec(baseHdr, basePayload)
	prog, err := ParseAnec(validBytes)
	if err != nil {
		t.Fatalf("ParseAnec valid failed: %v", err)
	}
	if prog.Header.TDCount != 1 || prog.Header.TDSize != 0x100 || prog.Header.TileBytes(4) != PageSize16K || len(prog.Payload) != len(basePayload) {
		t.Fatalf("unexpected parsed AnecProgram: %+v", prog.Header)
	}

	// PatchBootstrapNID preserves low 16 bits and bits [27:24], clears bits [31:28], and sets [23:16] = nid.
	td0 := []byte{0x56, 0x34, 0x12, 0xfa}
	if err := PatchBootstrapNID(td0, FifoNID); err != nil {
		t.Fatalf("PatchBootstrapNID failed: %v", err)
	}
	if got := binary.LittleEndian.Uint32(td0); got != 0x0a403456 {
		t.Fatalf("PatchBootstrapNID got 0x%08x, want 0x0a403456", got)
	}
	if err := PatchBootstrapNID([]byte{1, 2}, FifoNID); err == nil {
		t.Fatalf("expected PatchBootstrapNID error on < 4 bytes")
	}

	// Malformed header cases
	if _, err := ParseAnec(validBytes[:AnecHeaderSize-1]); err == nil {
		t.Fatalf("expected truncated header rejection")
	}
	badCases := []struct {
		name   string
		mutate func(h *AnecHeader, p *[]byte)
	}{
		{"zero_size", func(h *AnecHeader, p *[]byte) { h.Size = 0; *p = nil }},
		{"payload_len_mismatch", func(h *AnecHeader, p *[]byte) { *p = (*p)[:len(*p)-4] }},
		{"td_count_zero", func(h *AnecHeader, _ *[]byte) { h.TDCount = 0 }},
		{"td_count_overflow", func(h *AnecHeader, _ *[]byte) { h.TDCount = MaxTDCount + 1 }},
		{"td_size_too_small", func(h *AnecHeader, _ *[]byte) { h.TDSize = 2 }},
		{"td_size_unaligned", func(h *AnecHeader, _ *[]byte) { h.TDSize = 6 }},
		{"td_size_too_large", func(h *AnecHeader, _ *[]byte) { h.TDSize = MaxTDSize + 4 }},
		{"td_size_gt_payload", func(h *AnecHeader, _ *[]byte) { h.TDSize = 0x300; h.TskSize = 0x300 }},
		{"tsk_size_zero", func(h *AnecHeader, _ *[]byte) { h.TskSize = 0 }},
		{"tsk_size_gt_payload_krn_zero", func(h *AnecHeader, p *[]byte) {
			*p = make([]byte, 0x100)
			h.Size = 0x100
			h.TDCount = 2
			h.TDSize = 0x100
			h.TskSize = 0x200
			h.KrnSize = 0
			h.Tiles[0] = 2
		}},
		{"td_bytes_gt_tsk_size", func(h *AnecHeader, _ *[]byte) { h.TDCount = 2; h.TskSize = 0x100 }},
		{"tiles0_zero", func(h *AnecHeader, _ *[]byte) { h.Tiles[0] = 0 }},
		{"tiles1_nonzero", func(h *AnecHeader, _ *[]byte) { h.Tiles[1] = 1 }},
		{"tsk_size_ge_tile0", func(h *AnecHeader, p *[]byte) {
			*p = make([]byte, PageSize16K)
			h.Size = PageSize16K
			h.TskSize = PageSize16K
			h.KrnSize = 0
		}},
		{"size_gt_tile0", func(h *AnecHeader, p *[]byte) {
			*p = make([]byte, PageSize16K+16)
			h.Size = uint64(len(*p))
		}},
		{"krn_size_overflow", func(h *AnecHeader, _ *[]byte) { h.KrnSize = 0x200 }},
		{"dst_count_zero", func(h *AnecHeader, _ *[]byte) { h.DstCount = 0 }},
		{"slots_overflow_32", func(h *AnecHeader, _ *[]byte) { h.DstCount = 20; h.SrcCount = 10 }},
		{"dst_slot_zero_tile", func(h *AnecHeader, _ *[]byte) { h.Tiles[4] = 0 }},
		{"src_slot_zero_tile", func(h *AnecHeader, _ *[]byte) { h.Tiles[5] = 0 }},
	}

	for _, tc := range badCases {
		t.Run(tc.name, func(t *testing.T) {
			h, p := validSyntheticAnecHeader()
			tc.mutate(&h, &p)
			raw := makeSyntheticAnec(h, p)
			if _, err := ParseAnec(raw); err == nil {
				t.Fatalf("ParseAnec unexpectedly accepted %s", tc.name)
			}
		})
	}
}

func TestExecuteAnecLifecycleAndCleanup(t *testing.T) {
	hdr, payload := validSyntheticAnecHeader()
	prog, err := ParseAnec(makeSyntheticAnec(hdr, payload))
	if err != nil {
		t.Fatalf("ParseAnec: %v", err)
	}

	tr := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ABIV1, ChipFamily: ChipH13},
	}, func(inputs [][]byte, _ int) []byte {
		out := make([]byte, len(inputs[0]))
		for i, b := range inputs[0] {
			out[i] = b ^ 0x5a
		}
		return out
	})

	fd, caps, err := DiscoverDeviceABI(tr, "", ABIV1, ChipH13)
	if err != nil {
		t.Fatalf("DiscoverDeviceABI: %v", err)
	}
	defer tr.Close(fd)

	in1 := make([]byte, PageSize16K)
	in2 := make([]byte, PageSize16K)
	for i := range in1 {
		in1[i] = byte(i & 0xff)
		in2[i] = byte((i + 17) & 0xff)
	}

	// Reject --in-place on ABI v1
	if _, _, err := ExecuteAnec(tr, fd, caps, prog, [][]byte{in1}, ExecOptions{Runs: 1, InPlace: true}); err == nil {
		t.Fatalf("expected --in-place rejection for ABI v1")
	}

	// Full run with warmup=2, runs=4, AltInputs
	outBytes, rep, err := ExecuteAnec(tr, fd, caps, prog, [][]byte{in1}, ExecOptions{
		Warmup:    2,
		Runs:      4,
		AltInputs: [][]byte{in2},
	})
	if err != nil {
		t.Fatalf("ExecuteAnec failed: %v", err)
	}
	if rep.Mode != "abi-v1" || rep.ABIVersion != ABIV1 || rep.ChipFamily != ChipH13 {
		t.Fatalf("unexpected report header: %+v", rep)
	}
	if rep.Warmup != 2 || rep.Runs != 4 || len(rep.LatenciesMS) != 4 || !rep.Repeatable {
		t.Fatalf("unexpected report timings/repeatability: %+v", rep)
	}
	if rep.AltOutputSHA256 == "" || rep.AltOutputSHA256 == rep.OutputSHA256 {
		t.Fatalf("expected distinct AltOutputSHA256, got main=%s alt=%s", rep.OutputSHA256, rep.AltOutputSHA256)
	}
	if tr.SubmitCalls != 6 {
		t.Fatalf("expected 6 SUBMIT calls (2 warmup + 4 timed), got %d", tr.SubmitCalls)
	}
	if len(tr.ObservedOutputPoisonOK) != 6 {
		t.Fatalf("expected 6 poison checks, got %d", len(tr.ObservedOutputPoisonOK))
	}
	for i, ok := range tr.ObservedOutputPoisonOK {
		if !ok {
			t.Fatalf("output BO was not poisoned with 0xff on run %d", i+1)
		}
	}
	// Verify FifoNID was patched into btsp TD0 word 0
	if gotNID := uint8((binary.LittleEndian.Uint32(tr.LastBtspData[0:4]) >> 16) & 0xff); gotNID != FifoNID {
		t.Fatalf("btsp TD0 nid = 0x%02x, want 0x%02x", gotNID, FifoNID)
	}
	// Verify output bytes match ComputeFn(in1)
	for i := range outBytes {
		if outBytes[i] != in1[i]^0x5a {
			t.Fatalf("output byte[%d] = 0x%02x, want 0x%02x", i, outBytes[i], in1[i]^0x5a)
		}
	}
	// Verify zero leaked BO handles
	if tr.ActiveBOCount() != 0 || tr.TotalAllocatedBOs != tr.TotalFreedBOs || tr.TotalAllocatedBOs != 5 {
		t.Fatalf("BO leak check failed: active=%d alloc=%d freed=%d", tr.ActiveBOCount(), tr.TotalAllocatedBOs, tr.TotalFreedBOs)
	}

	// Non-determinism detection
	tr.ComputeFn = func(inputs [][]byte, call int) []byte {
		out := make([]byte, len(inputs[0]))
		copy(out, inputs[0])
		if call%2 == 0 {
			out[0] ^= 0xff
		}
		return out
	}
	if _, _, err := ExecuteAnec(tr, fd, caps, prog, [][]byte{in1}, ExecOptions{Runs: 2}); err == nil {
		t.Fatalf("expected non-determinism error")
	}
	if tr.ActiveBOCount() != 0 || tr.TotalAllocatedBOs != tr.TotalFreedBOs {
		t.Fatalf("BO leak after non-determinism error: active=%d alloc=%d freed=%d", tr.ActiveBOCount(), tr.TotalAllocatedBOs, tr.TotalFreedBOs)
	}

	// Mid-setup BoInit failure must free all previously allocated BOs (zero leaks!)
	trFail := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ABIV1, ChipFamily: ChipH13},
	}, nil)
	trFail.BoInitFailAt = 3
	fdFail, capsFail, err := DiscoverDeviceABI(trFail, "", ABIV1, ChipH13)
	if err != nil {
		t.Fatalf("DiscoverDeviceABI: %v", err)
	}
	defer trFail.Close(fdFail)
	if _, _, err := ExecuteAnec(trFail, fdFail, capsFail, prog, [][]byte{in1}, ExecOptions{Runs: 1}); err == nil {
		t.Fatalf("expected mid-setup BoInit ENOMEM failure")
	}
	if trFail.ActiveBOCount() != 0 || trFail.TotalAllocatedBOs != 2 || trFail.TotalFreedBOs != 2 {
		t.Fatalf("BO leak after mid-setup BoInit failure: active=%d alloc=%d freed=%d",
			trFail.ActiveBOCount(), trFail.TotalAllocatedBOs, trFail.TotalFreedBOs)
	}

	// SUBMIT failure must also free all BOs cleanly
	trSubErr := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ABIV1, ChipFamily: ChipH13},
	}, nil)
	trSubErr.SubmitErr = syscall.ETIMEDOUT
	fdSub, capsSub, err := DiscoverDeviceABI(trSubErr, "", ABIV1, ChipH13)
	if err != nil {
		t.Fatalf("DiscoverDeviceABI: %v", err)
	}
	defer trSubErr.Close(fdSub)
	if _, _, err := ExecuteAnec(trSubErr, fdSub, capsSub, prog, [][]byte{in1}, ExecOptions{Runs: 1}); err == nil {
		t.Fatalf("expected SUBMIT ETIMEDOUT failure")
	}
	if trSubErr.ActiveBOCount() != 0 || trSubErr.TotalAllocatedBOs != trSubErr.TotalFreedBOs {
		t.Fatalf("BO leak after SUBMIT error: active=%d alloc=%d freed=%d",
			trSubErr.ActiveBOCount(), trSubErr.TotalAllocatedBOs, trSubErr.TotalFreedBOs)
	}

	// Mid-setup Mmap failure must unmap prior BOs and free every allocated BO (including the unmapped one)
	trMmapFail := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ABIV1, ChipFamily: ChipH13},
	}, nil)
	trMmapFail.MmapFailAt = 3
	fdMm, capsMm, err := DiscoverDeviceABI(trMmapFail, "", ABIV1, ChipH13)
	if err != nil {
		t.Fatalf("DiscoverDeviceABI: %v", err)
	}
	defer trMmapFail.Close(fdMm)
	if _, _, err := ExecuteAnec(trMmapFail, fdMm, capsMm, prog, [][]byte{in1}, ExecOptions{Runs: 1}); err == nil {
		t.Fatalf("expected mid-setup Mmap ENOMEM failure")
	}
	if trMmapFail.ActiveBOCount() != 0 || trMmapFail.ActiveMappedBOCount() != 0 ||
		trMmapFail.TotalAllocatedBOs != 3 || trMmapFail.TotalFreedBOs != 3 {
		t.Fatalf("BO/mmap leak after mid-setup Mmap failure: active=%d mapped=%d alloc=%d freed=%d",
			trMmapFail.ActiveBOCount(), trMmapFail.ActiveMappedBOCount(), trMmapFail.TotalAllocatedBOs, trMmapFail.TotalFreedBOs)
	}

	// Direct check: BoFree rejects still-mapped BO with EBUSY, then succeeds after Munmap (and ABI v2 Mmap works)
	for _, spec := range []MockDeviceSpec{
		{ABIVersion: ABIV1, ChipFamily: ChipH13},
		{ABIVersion: ABIV2, ChipFamily: ChipH14},
	} {
		trMap := NewMockTransport(map[string]MockDeviceSpec{"/dev/accel/accel0": spec}, nil)
		mfd, _, err := DiscoverDeviceABI(trMap, "", spec.ABIVersion, spec.ChipFamily)
		if err != nil {
			t.Fatalf("DiscoverDeviceABI(v%d): %v", spec.ABIVersion, err)
		}
		bo, err := trMap.BoInit(mfd, PageSize16K)
		if err != nil {
			t.Fatalf("BoInit(v%d): %v", spec.ABIVersion, err)
		}
		mm, err := trMap.Mmap(mfd, int64(bo.Offset), PageSize16K, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
		if err != nil {
			t.Fatalf("Mmap(v%d): %v", spec.ABIVersion, err)
		}
		if err := trMap.BoFree(mfd, bo.Handle); !errors.Is(err, syscall.EBUSY) {
			t.Fatalf("BoFree while mapped (v%d): got %v, want EBUSY", spec.ABIVersion, err)
		}
		if err := trMap.Munmap(mm); err != nil {
			t.Fatalf("Munmap(v%d): %v", spec.ABIVersion, err)
		}
		if err := trMap.BoFree(mfd, bo.Handle); err != nil {
			t.Fatalf("BoFree after Munmap (v%d): %v", spec.ABIVersion, err)
		}
		_ = trMap.Close(mfd)
	}

	// Manually constructed malformed AnecProgram structs must be rejected cleanly without panicking or allocating BOs
	malformedProgs := []struct {
		name string
		prog *AnecProgram
	}{
		{"empty_struct", &AnecProgram{}},
		{"dst_count_zero", func() *AnecProgram {
			h, p := validSyntheticAnecHeader()
			h.DstCount = 0
			return &AnecProgram{Header: h, Payload: p}
		}()},
		{"slots_overflow", func() *AnecProgram {
			h, p := validSyntheticAnecHeader()
			h.DstCount = 20
			h.SrcCount = 10
			return &AnecProgram{Header: h, Payload: p}
		}()},
		{"tiles0_zero", func() *AnecProgram {
			h, p := validSyntheticAnecHeader()
			h.Tiles[0] = 0
			return &AnecProgram{Header: h, Payload: p}
		}()},
		{"tiles1_nonzero", func() *AnecProgram {
			h, p := validSyntheticAnecHeader()
			h.Tiles[1] = 1
			return &AnecProgram{Header: h, Payload: p}
		}()},
		{"td_size_gt_payload", func() *AnecProgram {
			h, p := validSyntheticAnecHeader()
			h.TDSize = 0x400
			h.TskSize = 0x400
			return &AnecProgram{Header: h, Payload: p}
		}()},
		{"tsk_size_gt_payload", func() *AnecProgram {
			h, p := validSyntheticAnecHeader()
			h.KrnSize = 0
			h.TskSize = uint64(len(p)) + 0x100
			return &AnecProgram{Header: h, Payload: p}
		}()},
		{"dst_slot_zero_tile", func() *AnecProgram {
			h, p := validSyntheticAnecHeader()
			h.Tiles[4] = 0
			return &AnecProgram{Header: h, Payload: p}
		}()},
		{"src_slot_zero_tile", func() *AnecProgram {
			h, p := validSyntheticAnecHeader()
			h.Tiles[5] = 0
			return &AnecProgram{Header: h, Payload: p}
		}()},
	}
	for _, tc := range malformedProgs {
		t.Run("manual_malformed_"+tc.name, func(t *testing.T) {
			trBad := NewMockTransport(map[string]MockDeviceSpec{
				"/dev/accel/accel0": {ABIVersion: ABIV1, ChipFamily: ChipH13},
			}, nil)
			bfd, bcaps, err := DiscoverDeviceABI(trBad, "", ABIV1, ChipH13)
			if err != nil {
				t.Fatalf("DiscoverDeviceABI: %v", err)
			}
			defer trBad.Close(bfd)
			if _, _, err := ExecuteAnec(trBad, bfd, bcaps, tc.prog, nil, ExecOptions{Runs: 1}); err == nil {
				t.Fatalf("ExecuteAnec unexpectedly accepted malformed program %s", tc.name)
			}
			if trBad.TotalAllocatedBOs != 0 || trBad.ActiveBOCount() != 0 {
				t.Fatalf("malformed program %s allocated BOs: alloc=%d active=%d", tc.name, trBad.TotalAllocatedBOs, trBad.ActiveBOCount())
			}
		})
	}
}

func TestStrictABIv3SingleOutputValidation(t *testing.T) {
	tr := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ABIV3, ChipFamily: ChipH17},
	}, nil)

	fd, caps, err := DiscoverDevice(tr, "", ChipH17)
	if err != nil {
		t.Fatalf("DiscoverDevice: %v", err)
	}
	defer tr.Close(fd)

	// ABI v3 strictly requires exactly 1 HWX output surface; 2 output surfaces must be rejected.
	loaded := LoadedProgram{
		SurfaceCount:      3,
		SurfaceSizes:      []uint64{64, 64, 64},
		SurfaceDirections: []uint32{1, 2, 2},
	}
	tr.openFDs[fd].loaded = &loaded
	tr.openFDs[fd].surfaces = [][]byte{
		make([]byte, PageSize16K),
		make([]byte, PageSize16K),
		make([]byte, PageSize16K),
	}

	inData := make([]byte, 64)
	if _, _, err := ExecuteProgram(tr, fd, caps, loaded, [][]byte{inData}, ExecOptions{Runs: 1}); err == nil {
		t.Fatalf("expected ExecuteProgram to reject 2 output surfaces under strict ABI v3")
	}
}
