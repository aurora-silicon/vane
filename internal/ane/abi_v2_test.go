package ane

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"unsafe"

	"github.com/aurora-silicon/vane/internal/fp16"
)

func TestABIV2UapiLayoutAndIoctlNumbers(t *testing.T) {
	if got := unsafe.Sizeof(SectionV2{}); got != 24 {
		t.Fatalf("sizeof(SectionV2) = %d, want 24", got)
	}
	if got := unsafe.Sizeof(GenericBindV2{}); got != 24 {
		t.Fatalf("sizeof(GenericBindV2) = %d, want 24", got)
	}
	if got := unsafe.Sizeof(ProgLoadV2{}); got != 32 {
		t.Fatalf("sizeof(ProgLoadV2) = %d, want 32", got)
	}
	if got := unsafe.Sizeof(ProcCreateV2{}); got != 8 {
		t.Fatalf("sizeof(ProcCreateV2) = %d, want 8", got)
	}
	if got := unsafe.Sizeof(ExecIOV2{}); got != 32 {
		t.Fatalf("sizeof(ExecIOV2) = %d, want 32", got)
	}
	if got := unsafe.Sizeof(ExecV2{}); got != 32 {
		t.Fatalf("sizeof(ExecV2) = %d, want 32", got)
	}

	// Verify exact field wire offsets matching uapi/drm/ane_accel.h.
	var sec SectionV2
	if unsafe.Offsetof(sec.ID) != 0x00 || unsafe.Offsetof(sec.BoHandle) != 0x04 ||
		unsafe.Offsetof(sec.Size) != 0x08 || unsafe.Offsetof(sec.Offset) != 0x10 {
		t.Fatalf("SectionV2 field offsets mismatch")
	}
	var gb GenericBindV2
	if unsafe.Offsetof(gb.BufferID) != 0x00 || unsafe.Offsetof(gb.BoHandle) != 0x04 ||
		unsafe.Offsetof(gb.Type) != 0x08 || unsafe.Offsetof(gb.Pad) != 0x0c ||
		unsafe.Offsetof(gb.Size) != 0x10 {
		t.Fatalf("GenericBindV2 field offsets mismatch")
	}
	var pl ProgLoadV2
	if unsafe.Offsetof(pl.SectionsPtr) != 0x00 || unsafe.Offsetof(pl.GenericPtr) != 0x08 ||
		unsafe.Offsetof(pl.SectionCount) != 0x10 || unsafe.Offsetof(pl.GenericCount) != 0x14 ||
		unsafe.Offsetof(pl.ProgIDOut) != 0x18 || unsafe.Offsetof(pl.Pad) != 0x1c {
		t.Fatalf("ProgLoadV2 field offsets mismatch")
	}
	var pc ProcCreateV2
	if unsafe.Offsetof(pc.ProgID) != 0x00 || unsafe.Offsetof(pc.ProcIDOut) != 0x04 {
		t.Fatalf("ProcCreateV2 field offsets mismatch")
	}
	var eio ExecIOV2
	if unsafe.Offsetof(eio.BufferID) != 0x00 || unsafe.Offsetof(eio.BoHandle) != 0x04 ||
		unsafe.Offsetof(eio.Type) != 0x08 || unsafe.Offsetof(eio.Flags) != 0x0c ||
		unsafe.Offsetof(eio.Reserved) != 0x10 || unsafe.Offsetof(eio.Size) != 0x18 {
		t.Fatalf("ExecIOV2 field offsets mismatch")
	}
	var ex ExecV2
	if unsafe.Offsetof(ex.ProgID) != 0x00 || unsafe.Offsetof(ex.ProcID) != 0x04 ||
		unsafe.Offsetof(ex.Priority) != 0x08 || unsafe.Offsetof(ex.TimeoutMS) != 0x0c ||
		unsafe.Offsetof(ex.Count) != 0x10 || unsafe.Offsetof(ex.Pad) != 0x14 ||
		unsafe.Offsetof(ex.IOPtr) != 0x18 {
		t.Fatalf("ExecV2 field offsets mismatch")
	}

	wantProgLoad := uintptr((3 << 30) | (32 << 16) | (0x64 << 8) | (0x40 + 0x04))
	wantProcCreate := uintptr((3 << 30) | (8 << 16) | (0x64 << 8) | (0x40 + 0x05))
	wantExec := uintptr((3 << 30) | (32 << 16) | (0x64 << 8) | (0x40 + 0x06))
	if IoctlProgLoad != wantProgLoad {
		t.Fatalf("IoctlProgLoad = 0x%x, want 0x%x", IoctlProgLoad, wantProgLoad)
	}
	if IoctlProcCreate != wantProcCreate {
		t.Fatalf("IoctlProcCreate = 0x%x, want 0x%x", IoctlProcCreate, wantProcCreate)
	}
	if IoctlExec != wantExec {
		t.Fatalf("IoctlExec = 0x%x, want 0x%x", IoctlExec, wantExec)
	}

	// LinuxTransport must reject invalid zero-pad / count / slice inputs without panicking.
	var lt LinuxTransport
	if _, err := lt.ProgLoad(-1, ProgLoadV2{Pad: 1}, []SectionV2{{ID: 1, BoHandle: 1, Size: 16}}, nil); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("LinuxTransport.ProgLoad nonzero pad: got %v, want EINVAL", err)
	}
	if _, err := lt.ProgLoad(-1, ProgLoadV2{}, nil, nil); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("LinuxTransport.ProgLoad empty sections: got %v, want EINVAL", err)
	}
	if _, err := lt.ProgLoad(-1, ProgLoadV2{}, []SectionV2{{ID: 1, BoHandle: 1, Size: 16}}, []GenericBindV2{{Pad: 1}}); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("LinuxTransport.ProgLoad nonzero generic_bind pad: got %v, want EINVAL", err)
	}
	if err := lt.ExecV2(-1, ExecV2{Pad: 1}, []ExecIOV2{{BufferID: 1, BoHandle: 1, Size: 16}}); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("LinuxTransport.ExecV2 nonzero pad: got %v, want EINVAL", err)
	}
	if err := lt.ExecV2(-1, ExecV2{}, nil); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("LinuxTransport.ExecV2 empty ios: got %v, want EINVAL", err)
	}
	if err := lt.ExecV2(-1, ExecV2{}, []ExecIOV2{{BufferID: 1, BoHandle: 1, Flags: 1, Size: 16}}); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("LinuxTransport.ExecV2 nonzero io flags: got %v, want EINVAL", err)
	}
	if err := lt.ExecV2(-1, ExecV2{}, []ExecIOV2{{BufferID: 1, BoHandle: 1, Reserved: 1, Size: 16}}); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("LinuxTransport.ExecV2 nonzero io reserved: got %v, want EINVAL", err)
	}
}

func writeTestV2BundleDir(t *testing.T, manifest any, secFiles map[string][]byte) string {
	t.Helper()
	dir := t.TempDir()
	for name, content := range secFiles {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, content, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(manifestPath, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return manifestPath
}

func makeValidV2Bundle() *V2Bundle {
	return &V2Bundle{
		Sections: []V2SectionSpec{
			{ID: 1, Offset: 0, Data: []byte("section-one-payload-bytes")},
			{ID: 2, Offset: 0x40, Data: []byte("section-two-offset-payload")},
		},
		GenericBinds: []V2GenericBindSpec{
			{BufferID: 4, Type: 1, Size: 4096},
		},
		IOs: []V2IOSpec{
			{BufferID: 5, Type: 0, Direction: "input", Size: 256},
			{BufferID: 6, Type: 2, Direction: "scratch", Size: 512},
			{BufferID: 4, Type: 1, Direction: "output", Size: 256},
		},
		Priority:  3,
		TimeoutMS: 2500,
	}
}

func TestParseAndValidateV2Bundle(t *testing.T) {
	// 1. Valid manifest with relative section files, generic_binds, explicit priority & timeout.
	mPath := writeTestV2BundleDir(t, map[string]any{
		"sections": []map[string]any{
			{"id": 1, "file": "sec1.bin"},
			{"id": 9, "file": "sub/sec9.bin", "offset": 64},
		},
		"generic_binds": []map[string]any{
			{"buffer_id": 4, "type": 1, "size": 4096},
		},
		"ios": []map[string]any{
			{"buffer_id": 5, "type": 0, "direction": "input", "size": 4096},
			{"buffer_id": 6, "type": 2, "direction": "scratch", "size": 1024},
			{"buffer_id": 4, "type": 1, "direction": "output", "size": 4096},
		},
		"priority":   5,
		"timeout_ms": 4000,
	}, map[string][]byte{
		"sec1.bin":     {1, 2, 3, 4},
		"sub/sec9.bin": {9, 8, 7, 6, 5},
	})
	b, err := LoadV2Bundle(mPath)
	if err != nil {
		t.Fatalf("LoadV2Bundle valid failed: %v", err)
	}
	if len(b.Sections) != 2 || b.Sections[1].Offset != 64 || b.Priority != 5 || b.TimeoutMS != 4000 || len(b.GenericBinds) != 1 {
		t.Fatalf("unexpected parsed V2Bundle: %+v", b)
	}

	// 2. Defaults when optional priority, timeout_ms, and generic_binds are omitted.
	mDef := writeTestV2BundleDir(t, map[string]any{
		"sections": []map[string]any{{"id": 1, "file": "sec1.bin"}},
		"ios":      []map[string]any{{"buffer_id": 4, "type": 1, "direction": "output", "size": 64}},
	}, map[string][]byte{"sec1.bin": {0xaa, 0xbb}})
	bDef, err := LoadV2Bundle(mDef)
	if err != nil {
		t.Fatalf("LoadV2Bundle defaults failed: %v", err)
	}
	if bDef.Priority != DefaultPriorityV2 || bDef.TimeoutMS != 0 || len(bDef.GenericBinds) != 0 {
		t.Fatalf("unexpected defaults: prio=%d timeout=%d binds=%d", bDef.Priority, bDef.TimeoutMS, len(bDef.GenericBinds))
	}

	// 3. Reject malformed JSON, unknown fields, trailing tokens, explicit invalid priority, missing/empty section files.
	tmp := t.TempDir()
	_ = os.WriteFile(filepath.Join(tmp, "sec1.bin"), []byte{1, 2, 3, 4}, 0o644)
	_ = os.WriteFile(filepath.Join(tmp, "empty.bin"), nil, 0o644)

	badJSONs := []struct {
		name string
		raw  string
	}{
		{"invalid_json", `{not-json`},
		{"unknown_field", `{"sections":[{"id":1,"file":"sec1.bin"}],"ios":[{"buffer_id":4,"type":1,"direction":"output","size":64}],"bogus":1}`},
		{"trailing_data", `{"sections":[{"id":1,"file":"sec1.bin"}],"ios":[{"buffer_id":4,"type":1,"direction":"output","size":64}]} {"extra":1}`},
		{"explicit_priority_zero", `{"sections":[{"id":1,"file":"sec1.bin"}],"ios":[{"buffer_id":4,"type":1,"direction":"output","size":64}],"priority":0}`},
		{"explicit_priority_one", `{"sections":[{"id":1,"file":"sec1.bin"}],"ios":[{"buffer_id":4,"type":1,"direction":"output","size":64}],"priority":1}`},
		{"explicit_priority_eight", `{"sections":[{"id":1,"file":"sec1.bin"}],"ios":[{"buffer_id":4,"type":1,"direction":"output","size":64}],"priority":8}`},
		{"timeout_gt_5000", `{"sections":[{"id":1,"file":"sec1.bin"}],"ios":[{"buffer_id":4,"type":1,"direction":"output","size":64}],"timeout_ms":5001}`},
		{"missing_section_file_field", `{"sections":[{"id":1}],"ios":[{"buffer_id":4,"type":1,"direction":"output","size":64}]}`},
		{"nonexistent_section_file", `{"sections":[{"id":1,"file":"missing.bin"}],"ios":[{"buffer_id":4,"type":1,"direction":"output","size":64}]}`},
		{"empty_section_file", `{"sections":[{"id":1,"file":"empty.bin"}],"ios":[{"buffer_id":4,"type":1,"direction":"output","size":64}]}`},
	}
	for _, tc := range badJSONs {
		t.Run("json_"+tc.name, func(t *testing.T) {
			if _, err := ParseV2Bundle([]byte(tc.raw), tmp); err == nil {
				t.Fatalf("ParseV2Bundle unexpectedly accepted %s", tc.name)
			}
		})
	}

	// 4. Programmatic ValidateV2Bundle negative cases.
	if err := ValidateV2Bundle(nil); err == nil {
		t.Fatalf("ValidateV2Bundle(nil) should fail")
	}
	badBundles := []struct {
		name   string
		mutate func(*V2Bundle)
	}{
		{"zero_sections", func(b *V2Bundle) { b.Sections = nil }},
		{"too_many_sections", func(b *V2Bundle) {
			b.Sections = make([]V2SectionSpec, MaxSectionsV2+1)
			for i := range b.Sections {
				b.Sections[i] = V2SectionSpec{ID: uint32(i%MaxSectionsV2) + 1, Data: []byte{1}}
			}
		}},
		{"section_id_zero", func(b *V2Bundle) { b.Sections[0].ID = 0 }},
		{"section_id_ten", func(b *V2Bundle) { b.Sections[0].ID = MaxSectionsV2 + 1 }},
		{"duplicate_section_id", func(b *V2Bundle) { b.Sections[1].ID = b.Sections[0].ID }},
		{"empty_section_data", func(b *V2Bundle) { b.Sections[0].Data = nil }},
		{"section_offset_overflow", func(b *V2Bundle) { b.Sections[0].Offset = ^uint64(0) - 4 }},
		{"too_many_generic_binds", func(b *V2Bundle) {
			b.GenericBinds = make([]V2GenericBindSpec, MaxBindsV2+1)
		}},
		{"zero_ios", func(b *V2Bundle) { b.IOs = nil }},
		{"too_many_ios", func(b *V2Bundle) {
			b.IOs = make([]V2IOSpec, MaxBindsV2+1)
			for i := range b.IOs {
				b.IOs[i] = V2IOSpec{BufferID: uint32(i + 1), Direction: "output", Size: 64}
			}
		}},
		{"duplicate_io_buffer_id", func(b *V2Bundle) { b.IOs[1].BufferID = b.IOs[0].BufferID }},
		{"invalid_io_direction", func(b *V2Bundle) { b.IOs[0].Direction = "inout" }},
		{"zero_io_size", func(b *V2Bundle) { b.IOs[0].Size = 0 }},
		{"io_size_overflow", func(b *V2Bundle) { b.IOs[0].Size = ^uint64(0) }},
		{"no_output_io", func(b *V2Bundle) {
			b.IOs = []V2IOSpec{
				{BufferID: 5, Direction: "input", Size: 64},
				{BufferID: 6, Direction: "scratch", Size: 64},
			}
		}},
		{"priority_too_low", func(b *V2Bundle) { b.Priority = 1 }},
		{"priority_too_high", func(b *V2Bundle) { b.Priority = 8 }},
		{"timeout_too_high", func(b *V2Bundle) { b.TimeoutMS = MaxTimeoutMSV2 + 1 }},
	}
	for _, tc := range badBundles {
		t.Run("bundle_"+tc.name, func(t *testing.T) {
			b := makeValidV2Bundle()
			tc.mutate(b)
			if err := ValidateV2Bundle(b); err == nil {
				t.Fatalf("ValidateV2Bundle unexpectedly accepted %s", tc.name)
			}
		})
	}
}

func TestMockTransportV2KernelRules(t *testing.T) {
	tr := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ABIV1, ChipFamily: ChipH13},
		"/dev/accel/accel1": {ABIVersion: ABIV2, ChipFamily: ChipH14, SectionSize: 24, BindSize: 24, ExecIOSize: 32},
		"/dev/accel/accel2": {ABIVersion: ABIV3, ChipFamily: ChipH17},
	}, nil)

	// Invalid fd -> EBADF
	if _, err := tr.ProgLoad(999, ProgLoadV2{}, []SectionV2{{ID: 1, BoHandle: 1, Size: 16}}, nil); !errors.Is(err, syscall.EBADF) {
		t.Fatalf("ProgLoad bad fd: got %v, want EBADF", err)
	}
	if _, err := tr.ProcCreate(999, ProcCreateV2{ProgID: 1}); !errors.Is(err, syscall.EBADF) {
		t.Fatalf("ProcCreate bad fd: got %v, want EBADF", err)
	}
	if err := tr.ExecV2(999, ExecV2{ProgID: 1, ProcID: 1, Priority: 3}, []ExecIOV2{{BufferID: 1, BoHandle: 1, Size: 16}}); !errors.Is(err, syscall.EBADF) {
		t.Fatalf("ExecV2 bad fd: got %v, want EBADF", err)
	}

	// Wrong ABI version -> EOPNOTSUPP
	fdV1, _, err := DiscoverDeviceABI(tr, "/dev/accel/accel0", ABIV1, ChipH13)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close(fdV1)
	if _, err := tr.ProgLoad(fdV1, ProgLoadV2{}, []SectionV2{{ID: 1, BoHandle: 1, Size: 16}}, nil); !errors.Is(err, syscall.EOPNOTSUPP) {
		t.Fatalf("ProgLoad on v1: got %v, want EOPNOTSUPP", err)
	}
	if _, err := tr.ProcCreate(fdV1, ProcCreateV2{ProgID: 1}); !errors.Is(err, syscall.EOPNOTSUPP) {
		t.Fatalf("ProcCreate on v1: got %v, want EOPNOTSUPP", err)
	}
	if err := tr.ExecV2(fdV1, ExecV2{ProgID: 1, ProcID: 1, Priority: 3}, []ExecIOV2{{BufferID: 1, BoHandle: 1, Size: 16}}); !errors.Is(err, syscall.EOPNOTSUPP) {
		t.Fatalf("ExecV2 on v1: got %v, want EOPNOTSUPP", err)
	}

	fd, _, err := DiscoverDeviceABI(tr, "/dev/accel/accel1", ABIV2, ChipH14)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close(fd)

	secBO, err := tr.BoInit(fd, 128)
	if err != nil {
		t.Fatal(err)
	}
	mm, err := tr.Mmap(fd, int64(secBO.Offset), 128, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	copy(mm, []byte("hello-v2-section"))
	// BoFree while mapped -> EBUSY
	if err := tr.BoFree(fd, secBO.Handle); !errors.Is(err, syscall.EBUSY) {
		t.Fatalf("BoFree while mapped: got %v, want EBUSY", err)
	}
	_ = tr.Munmap(mm)

	validSec := []SectionV2{{ID: 1, BoHandle: secBO.Handle, Size: 16, Offset: 0}}

	// ProgLoad kernel rule rejections
	if _, err := tr.ProgLoad(fd, ProgLoadV2{Pad: 1}, validSec, nil); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("ProgLoad nonzero pad: got %v, want EINVAL", err)
	}
	if _, err := tr.ProgLoad(fd, ProgLoadV2{}, nil, nil); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("ProgLoad 0 sections: got %v, want EINVAL", err)
	}
	if _, err := tr.ProgLoad(fd, ProgLoadV2{SectionCount: 2}, validSec, nil); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("ProgLoad mismatched SectionCount: got %v, want EINVAL", err)
	}
	if _, err := tr.ProgLoad(fd, ProgLoadV2{}, make([]SectionV2, MaxSectionsV2+1), nil); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("ProgLoad >9 sections: got %v, want EINVAL", err)
	}
	if _, err := tr.ProgLoad(fd, ProgLoadV2{}, validSec, make([]GenericBindV2, MaxBindsV2+1)); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("ProgLoad >64 binds: got %v, want EINVAL", err)
	}
	if _, err := tr.ProgLoad(fd, ProgLoadV2{}, validSec, []GenericBindV2{{BufferID: 4, Pad: 1, Size: 64}}); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("ProgLoad generic_bind nonzero pad: got %v, want EINVAL", err)
	}
	badSecs := [][]SectionV2{
		{{ID: 0, BoHandle: secBO.Handle, Size: 16}},
		{{ID: 10, BoHandle: secBO.Handle, Size: 16}},
		{{ID: 1, BoHandle: secBO.Handle, Size: 16}, {ID: 1, BoHandle: secBO.Handle, Size: 16}},
		{{ID: 1, BoHandle: secBO.Handle, Size: 0}},
		{{ID: 1, BoHandle: 9999, Size: 16}},
		{{ID: 1, BoHandle: secBO.Handle, Size: 129, Offset: 0}},
		{{ID: 1, BoHandle: secBO.Handle, Size: 16, Offset: 120}},
		{{ID: 1, BoHandle: secBO.Handle, Size: 16, Offset: ^uint64(0) - 4}},
	}
	for idx, bs := range badSecs {
		if _, err := tr.ProgLoad(fd, ProgLoadV2{}, bs, nil); !errors.Is(err, syscall.EINVAL) {
			t.Fatalf("ProgLoad badSecs[%d]: got %v, want EINVAL", idx, err)
		}
	}

	// Valid ProgLoad + content-addressed idempotency
	p1, err := tr.ProgLoad(fd, ProgLoadV2{}, validSec, nil)
	if err != nil || p1.ProgIDOut == 0 {
		t.Fatalf("ProgLoad valid failed: res=%+v err=%v", p1, err)
	}
	p1Dup, err := tr.ProgLoad(fd, ProgLoadV2{}, validSec, nil)
	if err != nil || p1Dup.ProgIDOut != p1.ProgIDOut {
		t.Fatalf("ProgLoad identical sections should return same prog_id: got %d vs %d (err=%v)", p1Dup.ProgIDOut, p1.ProgIDOut, err)
	}
	p2, err := tr.ProgLoad(fd, ProgLoadV2{}, []SectionV2{{ID: 2, BoHandle: secBO.Handle, Size: 16, Offset: 0}}, nil)
	if err != nil || p2.ProgIDOut == p1.ProgIDOut {
		t.Fatalf("ProgLoad distinct sections should return distinct prog_id: got %d vs %d (err=%v)", p2.ProgIDOut, p1.ProgIDOut, err)
	}

	// ProcCreate rules
	if _, err := tr.ProcCreate(fd, ProcCreateV2{ProgID: 0}); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("ProcCreate prog_id=0: got %v, want EINVAL", err)
	}
	if _, err := tr.ProcCreate(fd, ProcCreateV2{ProgID: 9999}); !errors.Is(err, syscall.EINVAL) {
		t.Fatalf("ProcCreate unknown prog_id: got %v, want EINVAL", err)
	}
	proc1, err := tr.ProcCreate(fd, ProcCreateV2{ProgID: p1.ProgIDOut})
	if err != nil || proc1.ProcIDOut == 0 {
		t.Fatalf("ProcCreate p1 failed: %+v err=%v", proc1, err)
	}
	proc2, err := tr.ProcCreate(fd, ProcCreateV2{ProgID: p2.ProgIDOut})
	if err != nil || proc2.ProcIDOut == proc1.ProcIDOut {
		t.Fatalf("ProcCreate p2 failed: %+v err=%v", proc2, err)
	}

	ioBO, err := tr.BoInit(fd, 64)
	if err != nil {
		t.Fatal(err)
	}
	validIO := []ExecIOV2{{BufferID: 4, BoHandle: ioBO.Handle, Type: 1, Size: 64}}
	validExec := ExecV2{ProgID: p1.ProgIDOut, ProcID: proc1.ProcIDOut, Priority: 3, TimeoutMS: 5000, Count: 1}

	// ExecV2 kernel rule rejections
	badExecCases := []struct {
		name string
		args ExecV2
		ios  []ExecIOV2
	}{
		{"nonzero_pad", ExecV2{ProgID: p1.ProgIDOut, ProcID: proc1.ProcIDOut, Priority: 3, Pad: 1}, validIO},
		{"zero_ios", validExec, nil},
		{"too_many_ios", ExecV2{ProgID: p1.ProgIDOut, ProcID: proc1.ProcIDOut, Priority: 3}, make([]ExecIOV2, MaxBindsV2+1)},
		{"mismatched_count", ExecV2{ProgID: p1.ProgIDOut, ProcID: proc1.ProcIDOut, Priority: 3, Count: 2}, validIO},
		{"priority_1", ExecV2{ProgID: p1.ProgIDOut, ProcID: proc1.ProcIDOut, Priority: 1}, validIO},
		{"priority_8", ExecV2{ProgID: p1.ProgIDOut, ProcID: proc1.ProcIDOut, Priority: 8}, validIO},
		{"timeout_5001", ExecV2{ProgID: p1.ProgIDOut, ProcID: proc1.ProcIDOut, Priority: 3, TimeoutMS: 5001}, validIO},
		{"unknown_prog", ExecV2{ProgID: 999, ProcID: proc1.ProcIDOut, Priority: 3}, validIO},
		{"unknown_proc", ExecV2{ProgID: p1.ProgIDOut, ProcID: 999, Priority: 3}, validIO},
		{"mismatched_proc_prog", ExecV2{ProgID: p1.ProgIDOut, ProcID: proc2.ProcIDOut, Priority: 3}, validIO},
		{"io_nonzero_flags", validExec, []ExecIOV2{{BufferID: 4, BoHandle: ioBO.Handle, Flags: 1, Size: 64}}},
		{"io_nonzero_reserved", validExec, []ExecIOV2{{BufferID: 4, BoHandle: ioBO.Handle, Reserved: 1, Size: 64}}},
		{"io_zero_size", validExec, []ExecIOV2{{BufferID: 4, BoHandle: ioBO.Handle, Size: 0}}},
		{"io_unknown_bo", validExec, []ExecIOV2{{BufferID: 4, BoHandle: 9999, Size: 64}}},
		{"io_size_gt_bo", validExec, []ExecIOV2{{BufferID: 4, BoHandle: ioBO.Handle, Size: 65}}},
	}
	for _, tc := range badExecCases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tr.ExecV2(fd, tc.args, tc.ios); !errors.Is(err, syscall.EINVAL) {
				t.Fatalf("ExecV2 %s: got %v, want EINVAL", tc.name, err)
			}
		})
	}

	// Clean up BOs
	if err := tr.BoFree(fd, ioBO.Handle); err != nil {
		t.Fatalf("BoFree ioBO: %v", err)
	}
	if err := tr.BoFree(fd, secBO.Handle); err != nil {
		t.Fatalf("BoFree secBO: %v", err)
	}
	if tr.ActiveBOCount() != 0 || tr.ActiveMappedBOCount() != 0 {
		t.Fatalf("expected 0 active BOs/maps, got active=%d mapped=%d", tr.ActiveBOCount(), tr.ActiveMappedBOCount())
	}
}

func TestExecuteV2HappyPathAndFeatures(t *testing.T) {
	bundle := makeValidV2Bundle()
	// Add a second output to confirm ALL output BOs are pre-poisoned with 0xff.
	bundle.IOs = append(bundle.IOs, V2IOSpec{BufferID: 7, Type: 1, Direction: "output", Size: 128})

	inMain := make([]byte, 256)
	inAlt := make([]byte, 256)
	for i := range inMain {
		inMain[i] = byte(i)
		inAlt[i] = byte(255 - i)
	}

	tr := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ABIV2, ChipFamily: ChipH14, SectionSize: 24, BindSize: 24, ExecIOSize: 32},
	}, nil)

	poisonChecks := 0
	tr.OnExecV2 = func(args ExecV2, sections map[uint32][]byte, bufs []MockExecV2Buffer, call int) error {
		if args.Priority != bundle.Priority || args.TimeoutMS != bundle.TimeoutMS || args.Pad != 0 || args.Count != 4 {
			t.Errorf("call %d unexpected ExecV2 args: %+v", call, args)
		}
		if !bytes.Equal(sections[1], bundle.Sections[0].Data) {
			t.Errorf("call %d section 1 bytes mismatch", call)
		}
		if !bytes.Equal(sections[2], bundle.Sections[1].Data) {
			t.Errorf("call %d section 2 bytes mismatch (offset copy failed)", call)
		}
		// Check both outputs (bufs[2] and bufs[3]) are 0xff pre-poisoned.
		for _, outIdx := range []int{2, 3} {
			for j, b := range bufs[outIdx].Data {
				if b != 0xff {
					t.Errorf("call %d output buf[%d][%d] = 0x%02x, want 0xff poison", call, outIdx, j, b)
					break
				}
			}
			poisonChecks++
		}
		// Compute deterministic transform of input (bufs[0]) into primary output (bufs[2]).
		for i, b := range bufs[0].Data {
			bufs[2].Data[i] = b ^ 0x3c
		}
		for i := range bufs[3].Data {
			bufs[3].Data[i] = byte(i)
		}
		return nil
	}

	fd, caps, err := DiscoverDeviceABI(tr, "", ABIV2, ChipH14)
	if err != nil {
		t.Fatalf("DiscoverDeviceABI: %v", err)
	}
	defer tr.Close(fd)

	outBytes, rep, err := ExecuteV2(tr, fd, caps, bundle, [][]byte{inMain}, ExecOptions{
		Warmup:    2,
		Runs:      4,
		AltInputs: [][]byte{inAlt},
	})
	if err != nil {
		t.Fatalf("ExecuteV2 failed: %v", err)
	}
	if rep.Mode != "abi-v2" || rep.ABIVersion != ABIV2 || rep.ChipFamily != ChipH14 {
		t.Fatalf("unexpected report header: %+v", rep)
	}
	if rep.Warmup != 2 || rep.Runs != 4 || len(rep.LatenciesMS) != 4 || !rep.Repeatable {
		t.Fatalf("unexpected report timings/repeatability: %+v", rep)
	}
	if rep.SurfaceCount != 4 || rep.OutputSize != 256 {
		t.Fatalf("unexpected surface metadata: %+v", rep)
	}
	if rep.AltOutputSHA256 == "" || rep.AltOutputSHA256 == rep.OutputSHA256 {
		t.Fatalf("expected distinct AltOutputSHA256, main=%s alt=%s", rep.OutputSHA256, rep.AltOutputSHA256)
	}
	if tr.ProgLoadCalls != 1 || tr.ProcCreateCalls != 1 || tr.ExecV2Calls != 6 {
		t.Fatalf("unexpected ioctl counts: progLoad=%d procCreate=%d execV2=%d", tr.ProgLoadCalls, tr.ProcCreateCalls, tr.ExecV2Calls)
	}
	if len(tr.LastProgLoadBinds) != 1 || tr.LastProgLoadBinds[0].BufferID != 4 || tr.LastProgLoadBinds[0].Pad != 0 {
		t.Fatalf("unexpected generic binds passed to ProgLoad: %+v", tr.LastProgLoadBinds)
	}
	if poisonChecks != 12 {
		t.Fatalf("expected 12 output poison checks (6 runs * 2 outputs), got %d", poisonChecks)
	}
	for i, b := range outBytes {
		if b != inMain[i]^0x3c {
			t.Fatalf("output[%d] = 0x%02x, want 0x%02x", i, b, inMain[i]^0x3c)
		}
	}
	// 2 section BOs + 4 I/O BOs = 6 BOs allocated and freed with 0 leaks.
	if tr.ActiveBOCount() != 0 || tr.ActiveMappedBOCount() != 0 || tr.TotalAllocatedBOs != 6 || tr.TotalFreedBOs != 6 {
		t.Fatalf("BO/mmap leak check failed: active=%d mapped=%d alloc=%d freed=%d",
			tr.ActiveBOCount(), tr.ActiveMappedBOCount(), tr.TotalAllocatedBOs, tr.TotalFreedBOs)
	}

	// Default mock path (OnExecV2 == nil, ComputeFn != nil) + Expected FP16 comparison + zero-input bundle.
	fp16Vals := make([]uint16, 32)
	for i := range fp16Vals {
		fp16Vals[i] = fp16.FromFloat64(float64(i) * 0.5)
	}
	expectedBytes := fp16.EncodeLE(fp16Vals)
	trDef := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ABIV2, ChipFamily: ChipH14, SectionSize: 24, BindSize: 24, ExecIOSize: 32},
	}, func(_ [][]byte, _ int) []byte {
		return expectedBytes
	})
	fdDef, capsDef, err := DiscoverDeviceABI(trDef, "", ABIV2, ChipH14)
	if err != nil {
		t.Fatal(err)
	}
	defer trDef.Close(fdDef)
	zeroInBundle := &V2Bundle{
		Sections:  []V2SectionSpec{{ID: 1, Data: []byte{1, 2, 3, 4}}},
		IOs:       []V2IOSpec{{BufferID: 4, Type: 1, Direction: "output", Size: uint64(len(expectedBytes))}},
		Priority:  4,
		TimeoutMS: 0,
	}
	_, repDef, err := ExecuteV2(trDef, fdDef, capsDef, zeroInBundle, nil, ExecOptions{
		Runs:     2,
		Expected: expectedBytes,
		Atol:     1e-3,
		Rtol:     1e-3,
	})
	if err != nil {
		t.Fatalf("ExecuteV2 default mock failed: %v", err)
	}
	if repDef.Comparison == nil || !repDef.Comparison.Passed {
		t.Fatalf("expected FP16 comparison passed, got %+v", repDef.Comparison)
	}
	if len(trDef.ObservedOutputPoisonOK) != 2 || !trDef.ObservedOutputPoisonOK[0] || !trDef.ObservedOutputPoisonOK[1] {
		t.Fatalf("expected 0xff poison OK in default mock path, got %v", trDef.ObservedOutputPoisonOK)
	}
	if trDef.ActiveBOCount() != 0 || trDef.ActiveMappedBOCount() != 0 {
		t.Fatalf("leak in default mock path: active=%d mapped=%d", trDef.ActiveBOCount(), trDef.ActiveMappedBOCount())
	}
}

func TestExecuteV2RejectionsAndLeakFreeCleanup(t *testing.T) {
	bundle := makeValidV2Bundle()
	inData := make([]byte, 256)

	// 1. Wrong caps rejected before allocating any BOs.
	badCaps := []struct {
		name string
		caps DeviceCaps
	}{
		{"wrong_abi_v1", DeviceCaps{Path: "/dev/accel/accel0", ABIVersion: ABIV1, ChipFamily: ChipH13, SectionSize: 24, BindSize: 24, ExecIOSize: 32}},
		{"wrong_abi_v3", DeviceCaps{Path: "/dev/accel/accel0", ABIVersion: ABIV3, ChipFamily: ChipH17, SectionSize: 24, BindSize: 24, ExecIOSize: 32}},
		{"wrong_section_size", DeviceCaps{Path: "/dev/accel/accel0", ABIVersion: ABIV2, ChipFamily: ChipH14, SectionSize: 16, BindSize: 24, ExecIOSize: 32}},
		{"wrong_bind_size", DeviceCaps{Path: "/dev/accel/accel0", ABIVersion: ABIV2, ChipFamily: ChipH14, SectionSize: 24, BindSize: 0, ExecIOSize: 32}},
		{"wrong_exec_io_size", DeviceCaps{Path: "/dev/accel/accel0", ABIVersion: ABIV2, ChipFamily: ChipH14, SectionSize: 24, BindSize: 24, ExecIOSize: 16}},
	}
	for _, tc := range badCaps {
		t.Run("caps_"+tc.name, func(t *testing.T) {
			tr := NewMockTransport(map[string]MockDeviceSpec{
				"/dev/accel/accel0": {ABIVersion: ABIV2, ChipFamily: ChipH14, SectionSize: 24, BindSize: 24, ExecIOSize: 32},
			}, nil)
			fd, _, _ := DiscoverDeviceABI(tr, "", ABIV2, ChipH14)
			defer tr.Close(fd)
			if _, _, err := ExecuteV2(tr, fd, tc.caps, bundle, [][]byte{inData}, ExecOptions{Runs: 1}); err == nil {
				t.Fatalf("expected caps rejection for %s", tc.name)
			}
			if tr.TotalAllocatedBOs != 0 {
				t.Fatalf("expected 0 BOs allocated on bad caps, got %d", tr.TotalAllocatedBOs)
			}
		})
	}

	// 2. Options / input rejections before allocating any BOs.
	trOpt := NewMockTransport(map[string]MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ABIV2, ChipFamily: ChipH14, SectionSize: 24, BindSize: 24, ExecIOSize: 32},
	}, nil)
	fdOpt, capsOpt, _ := DiscoverDeviceABI(trOpt, "", ABIV2, ChipH14)
	defer trOpt.Close(fdOpt)

	optErrors := []struct {
		name   string
		inputs [][]byte
		opts   ExecOptions
	}{
		{"in_place", [][]byte{inData}, ExecOptions{Runs: 1, InPlace: true}},
		{"negative_warmup", [][]byte{inData}, ExecOptions{Runs: 1, Warmup: -1}},
		{"wrong_input_count", [][]byte{inData, inData}, ExecOptions{Runs: 1}},
		{"oversized_input", [][]byte{make([]byte, 257)}, ExecOptions{Runs: 1}},
		{"alt_input_runs_1", [][]byte{inData}, ExecOptions{Runs: 1, AltInputs: [][]byte{inData}}},
		{"alt_input_wrong_count", [][]byte{inData}, ExecOptions{Runs: 2, AltInputs: [][]byte{inData, inData}}},
		{"alt_input_oversized", [][]byte{inData}, ExecOptions{Runs: 2, AltInputs: [][]byte{make([]byte, 257)}}},
	}
	for _, tc := range optErrors {
		t.Run("opts_"+tc.name, func(t *testing.T) {
			if _, _, err := ExecuteV2(trOpt, fdOpt, capsOpt, bundle, tc.inputs, tc.opts); err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
			if trOpt.TotalAllocatedBOs != 0 {
				t.Fatalf("expected 0 BO allocations for %s, got %d", tc.name, trOpt.TotalAllocatedBOs)
			}
		})
	}

	// 3. Sweep every BoInitFailAt (1..5) and MmapFailAt (1..5) across section + I/O BOs -> zero leaks!
	totalBOs := len(bundle.Sections) + len(bundle.IOs) // 2 + 3 = 5
	for failAt := 1; failAt <= totalBOs; failAt++ {
		tr := NewMockTransport(map[string]MockDeviceSpec{
			"/dev/accel/accel0": {ABIVersion: ABIV2, ChipFamily: ChipH14, SectionSize: 24, BindSize: 24, ExecIOSize: 32},
		}, nil)
		tr.BoInitFailAt = failAt
		fd, caps, err := DiscoverDeviceABI(tr, "", ABIV2, ChipH14)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := ExecuteV2(tr, fd, caps, bundle, [][]byte{inData}, ExecOptions{Runs: 1}); err == nil {
			t.Fatalf("BoInitFailAt=%d unexpectedly succeeded", failAt)
		}
		if tr.ActiveBOCount() != 0 || tr.ActiveMappedBOCount() != 0 || tr.TotalAllocatedBOs != failAt-1 || tr.TotalFreedBOs != failAt-1 {
			t.Fatalf("BoInitFailAt=%d leak: active=%d mapped=%d alloc=%d freed=%d",
				failAt, tr.ActiveBOCount(), tr.ActiveMappedBOCount(), tr.TotalAllocatedBOs, tr.TotalFreedBOs)
		}
		_ = tr.Close(fd)
	}

	for failAt := 1; failAt <= totalBOs; failAt++ {
		tr := NewMockTransport(map[string]MockDeviceSpec{
			"/dev/accel/accel0": {ABIVersion: ABIV2, ChipFamily: ChipH14, SectionSize: 24, BindSize: 24, ExecIOSize: 32},
		}, nil)
		tr.MmapFailAt = failAt
		fd, caps, err := DiscoverDeviceABI(tr, "", ABIV2, ChipH14)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := ExecuteV2(tr, fd, caps, bundle, [][]byte{inData}, ExecOptions{Runs: 1}); err == nil {
			t.Fatalf("MmapFailAt=%d unexpectedly succeeded", failAt)
		}
		if tr.ActiveBOCount() != 0 || tr.ActiveMappedBOCount() != 0 || tr.TotalAllocatedBOs != failAt || tr.TotalFreedBOs != failAt {
			t.Fatalf("MmapFailAt=%d leak: active=%d mapped=%d alloc=%d freed=%d",
				failAt, tr.ActiveBOCount(), tr.ActiveMappedBOCount(), tr.TotalAllocatedBOs, tr.TotalFreedBOs)
		}
		_ = tr.Close(fd)
	}

	// 4. ProgLoad / ProcCreate / ExecV2 failure -> leak-free cleanup.
	for _, stage := range []string{"prog_load", "proc_create", "exec_v2", "exec_v2_run2_hook", "nondet_main", "nondet_alt"} {
		t.Run("fail_"+stage, func(t *testing.T) {
			tr := NewMockTransport(map[string]MockDeviceSpec{
				"/dev/accel/accel0": {ABIVersion: ABIV2, ChipFamily: ChipH14, SectionSize: 24, BindSize: 24, ExecIOSize: 32},
			}, nil)
			opts := ExecOptions{Runs: 2}
			switch stage {
			case "prog_load":
				tr.ProgLoadErr = syscall.EIO
			case "proc_create":
				tr.ProcCreateErr = syscall.ENOMEM
			case "exec_v2":
				tr.ExecV2Err = syscall.ETIMEDOUT
			case "exec_v2_run2_hook":
				tr.OnExecV2 = func(_ ExecV2, _ map[uint32][]byte, bufs []MockExecV2Buffer, call int) error {
					if call == 2 {
						return syscall.ETIMEDOUT
					}
					copy(bufs[2].Data, bufs[0].Data)
					return nil
				}
			case "nondet_main":
				tr.OnExecV2 = func(_ ExecV2, _ map[uint32][]byte, bufs []MockExecV2Buffer, call int) error {
					copy(bufs[2].Data, bufs[0].Data)
					bufs[2].Data[0] = byte(call)
					return nil
				}
			case "nondet_alt":
				opts = ExecOptions{Runs: 4, AltInputs: [][]byte{inData}}
				tr.OnExecV2 = func(_ ExecV2, _ map[uint32][]byte, bufs []MockExecV2Buffer, call int) error {
					copy(bufs[2].Data, bufs[0].Data)
					if call == 4 { // 2nd alt run diverges from 1st alt run (call 2)
						bufs[2].Data[0] ^= 0x55
					}
					return nil
				}
			}
			fd, caps, err := DiscoverDeviceABI(tr, "", ABIV2, ChipH14)
			if err != nil {
				t.Fatal(err)
			}
			defer tr.Close(fd)
			if _, _, err := ExecuteV2(tr, fd, caps, bundle, [][]byte{inData}, opts); err == nil {
				t.Fatalf("expected failure in stage %s", stage)
			}
			if tr.ActiveBOCount() != 0 || tr.ActiveMappedBOCount() != 0 || tr.TotalAllocatedBOs != tr.TotalFreedBOs {
				t.Fatalf("leak in stage %s: active=%d mapped=%d alloc=%d freed=%d",
					stage, tr.ActiveBOCount(), tr.ActiveMappedBOCount(), tr.TotalAllocatedBOs, tr.TotalFreedBOs)
			}
		})
	}
}
