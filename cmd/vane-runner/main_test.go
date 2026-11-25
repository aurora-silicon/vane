package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"

	"github.com/aurora-silicon/vane/internal/ane"
	"github.com/aurora-silicon/vane/internal/fp16"
	"github.com/aurora-silicon/vane/internal/ref"
)

func findVaneBin(t *testing.T) string {
	t.Helper()
	if env := os.Getenv("VANE_BIN"); env != "" {
		return env
	}
	_, thisFile, _, _ := runtime.Caller(0)
	repoRoot := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", ".."))
	bin := filepath.Join(repoRoot, ".lake", "build", "bin", "vane")
	if _, err := os.Stat(bin); err != nil {
		t.Fatalf("vane binary not found at %s (set VANE_BIN or run lake build): %v", bin, err)
	}
	return bin
}

func runVane(t *testing.T, vaneBin string, args ...string) string {
	t.Helper()
	cmd := exec.Command(vaneBin, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("vane %v failed: %v\noutput:\n%s", args, err, string(out))
	}
	return string(out)
}

func TestProbeAndRunSkipWithoutDevice(t *testing.T) {
	vaneBin := findVaneBin(t)
	tmp := t.TempDir()
	demoDir := filepath.Join(tmp, "demo")
	runVane(t, vaneBin, "emit-demo", "--target", "h16", "--out-dir", demoDir)

	tr := ane.NewMockTransport(map[string]ane.MockDeviceSpec{}, nil)
	var out, errBuf bytes.Buffer
	if rc := runCLI([]string{"probe"}, tr, &out, &errBuf); rc != ane.ExitSkip {
		t.Fatalf("probe without devices rc=%d, want %d", rc, ane.ExitSkip)
	}
	out.Reset()
	errBuf.Reset()
	rc := runCLI([]string{
		"run",
		"--hwx", filepath.Join(demoDir, "program.hwx"),
		"--sections-dir", demoDir,
	}, tr, &out, &errBuf)
	if rc != ane.ExitSkip {
		t.Fatalf("run without devices rc=%d, want %d", rc, ane.ExitSkip)
	}
}

func TestEmitDemoSectionsMatchAndMockExec(t *testing.T) {
	vaneBin := findVaneBin(t)
	tmp := t.TempDir()
	demoDir := filepath.Join(tmp, "demo_h17")
	runVane(t, vaneBin, "emit-demo", "--target", "h17", "--out-dir", demoDir, "--with-kernel", "--with-scratch")

	valOut := runVane(t, vaneBin, "validate", "--hwx", filepath.Join(demoDir, "program.hwx"), "--sections", demoDir)
	if !strings.Contains(valOut, "status: ok") || !strings.Contains(valOut, "target: h17") {
		t.Fatalf("unexpected vane validate output:\n%s", valOut)
	}

	// Verify pure-Go section builder matches Lean `vane sections` byte-for-byte
	hwxBytes, err := os.ReadFile(filepath.Join(demoDir, "program.hwx"))
	if err != nil {
		t.Fatalf("read hwx: %v", err)
	}
	goSecs, err := ane.BuildSectionsFromHWX(hwxBytes)
	if err != nil {
		t.Fatalf("BuildSectionsFromHWX failed: %v", err)
	}
	for kind, fname := range ane.SectionFilenames {
		leanBytes, err := os.ReadFile(filepath.Join(demoDir, fname))
		if err != nil {
			t.Fatalf("read %s: %v", fname, err)
		}
		if !bytes.Equal(goSecs[kind], leanBytes) {
			t.Fatalf("Go BuildSectionsFromHWX mismatch on %s", fname)
		}
	}

	// Verify OPERATION bindings are sorted ascending: slots 3, 6, 7, 8
	op := goSecs[ane.SectionOperation]
	refCount := binary.LittleEndian.Uint32(op[0x0c:])
	if refCount != 4 {
		t.Fatalf("expected 4 OPERATION bindings, got %d", refCount)
	}
	wantPairs := [][2]uint32{{3, 6}, {6, 5}, {7, 4}, {8, ane.ResourceKernel}}
	for j, want := range wantPairs {
		slot := binary.LittleEndian.Uint32(op[0x10+8*j:])
		tag := binary.LittleEndian.Uint32(op[0x14+8*j:])
		if slot != want[0] || tag != want[1] {
			t.Fatalf("binding %d = (%d, %d), want (%d, %d)", j, slot, tag, want[0], want[1])
		}
	}

	// Execute against mock transport (copy + in-place + JSON report)
	var manifest struct {
		Surfaces []struct {
			Span int `json:"span"`
		} `json:"surfaces"`
	}
	mRaw, _ := os.ReadFile(filepath.Join(demoDir, "manifest.json"))
	if err := json.Unmarshal(mRaw, &manifest); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	inSpan := manifest.Surfaces[0].Span
	inVals := make([]uint16, inSpan/2)
	expVals := make([]uint16, inSpan/2)
	for i := range inVals {
		v := float64(i%17-8) * 0.25
		inVals[i] = fp16.FromFloat64(v)
		if v < 0 {
			v = 0
		}
		expVals[i] = fp16.FromFloat64(v)
	}
	inPath := filepath.Join(tmp, "in0.bin")
	expPath := filepath.Join(tmp, "exp.bin")
	outPath := filepath.Join(tmp, "out.bin")
	jsonPath := filepath.Join(tmp, "report.json")
	_ = os.WriteFile(inPath, fp16.EncodeLE(inVals), 0o644)
	_ = os.WriteFile(expPath, fp16.EncodeLE(expVals), 0o644)

	tr := ane.NewMockTransport(map[string]ane.MockDeviceSpec{
		"/dev/accel/accel0": {GetCapsErr: syscall.ENOTTY},
		"/dev/accel/accel1": {ABIVersion: ane.ABIV3, ChipFamily: ane.ChipH17},
	}, func(inputs [][]byte, _ int) []byte {
		vals, _ := fp16.DecodeLE(inputs[0])
		out := make([]uint16, len(vals))
		for i, v := range vals {
			f := fp16.ToFloat64(v)
			if f < 0 {
				f = 0
			}
			out[i] = fp16.FromFloat64(f)
		}
		return fp16.EncodeLE(out)
	})

	var stdout, stderr bytes.Buffer
	rc := runCLI([]string{
		"run",
		"--hwx", filepath.Join(demoDir, "program.hwx"),
		"--input", inPath,
		"--expected", expPath,
		"--runs", "3",
		"--output-out", outPath,
		"--json-out", jsonPath,
	}, tr, &stdout, &stderr)
	if rc != 0 {
		t.Fatalf("vane-runner run failed rc=%d stderr=%s", rc, stderr.String())
	}
	if len(tr.ObservedOutputPoisonOK) != 3 {
		t.Fatalf("expected 3 poison checks, got %d", len(tr.ObservedOutputPoisonOK))
	}
	gotOut, _ := os.ReadFile(outPath)
	if !bytes.Equal(gotOut, fp16.EncodeLE(expVals)) {
		t.Fatalf("output bytes mismatch")
	}

	// Probe JSON check
	stdout.Reset()
	stderr.Reset()
	if rc := runCLI([]string{"probe", "--json"}, tr, &stdout, &stderr); rc != 0 {
		t.Fatalf("probe --json failed: %s", stderr.String())
	}
	var probed []ane.DeviceCaps
	if err := json.Unmarshal(stdout.Bytes(), &probed); err != nil || len(probed) != 1 || probed[0].ChipFamily != ane.ChipH17 {
		t.Fatalf("unexpected probe JSON: %s (err=%v)", stdout.String(), err)
	}
}

func TestConv1x1EndToEndCLIAndCompare(t *testing.T) {
	vaneBin := findVaneBin(t)

	cases := []struct {
		name   string
		target string
		h, w   int
		cin    int
		cout   int
		bias   bool
		relu   bool
	}{
		{"h17_small_bias_relu", "h17", 8, 16, 16, 16, true, true},
		{"h17_odd_dims", "h17", 7, 13, 9, 11, true, false},
		{"h17_cout1_bias", "h17", 4, 5, 12, 1, true, true},
		{"h17_cin1_nobias", "h17", 5, 9, 1, 17, false, false},
		{"h16_odd_bias_relu", "h16", 7, 13, 9, 11, true, true},
		{"h16_cout1_nobias", "h16", 4, 8, 8, 1, false, true},
	}

	for idx, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmp := t.TempDir()
			vecDir := filepath.Join(tmp, "vec")
			compDir := filepath.Join(tmp, "compiled")

			// 1. Generate deterministic conv1x1 test vectors with vane-runner gen-conv1x1
			genArgs := []string{
				"gen-conv1x1",
				"--h", itoa(tc.h),
				"--w", itoa(tc.w),
				"--cin", itoa(tc.cin),
				"--cout", itoa(tc.cout),
				"--seed", itoa(idx + 100),
				"--out-dir", vecDir,
				"--json",
			}
			if tc.bias {
				genArgs = append(genArgs, "--bias")
			}
			if tc.relu {
				genArgs = append(genArgs, "--relu")
			}
			var stdout, stderr bytes.Buffer
			if rc := runCLI(genArgs, nil, &stdout, &stderr); rc != 0 {
				t.Fatalf("gen-conv1x1 failed rc=%d stderr=%s", rc, stderr.String())
			}

			// 2. Compile with Lean `vane compile-conv1x1`
			compileArgs := []string{
				"compile-conv1x1",
				"--target", tc.target,
				"--h", itoa(tc.h),
				"--w", itoa(tc.w),
				"--cin", itoa(tc.cin),
				"--cout", itoa(tc.cout),
				"--weights", filepath.Join(vecDir, "weights.bin"),
				"--out-dir", compDir,
			}
			if tc.bias {
				compileArgs = append(compileArgs, "--bias", filepath.Join(vecDir, "bias.bin"))
			}
			if tc.relu {
				compileArgs = append(compileArgs, "--relu")
			}
			runVane(t, vaneBin, compileArgs...)

			// 3. Validate with `vane validate --check-tasks`
			valOut := runVane(t, vaneBin, "validate",
				"--hwx", filepath.Join(compDir, "program.hwx"),
				"--sections", compDir,
				"--check-tasks",
			)
			if !strings.Contains(valOut, "status: ok") || !strings.Contains(valOut, "verified_conv_tasks: 1") {
				t.Fatalf("unexpected validate output:\n%s", valOut)
			}

			// 4. Verify Go `BuildSectionsFromHWX` matches Lean `vane sections` byte-for-byte
			hwxBytes, err := os.ReadFile(filepath.Join(compDir, "program.hwx"))
			if err != nil {
				t.Fatalf("read hwx: %v", err)
			}
			goSecs, err := ane.BuildSectionsFromHWX(hwxBytes)
			if err != nil {
				t.Fatalf("BuildSectionsFromHWX failed: %v", err)
			}
			for kind, fname := range ane.SectionFilenames {
				leanBytes, err := os.ReadFile(filepath.Join(compDir, fname))
				if err != nil {
					t.Fatalf("read %s: %v", fname, err)
				}
				if !bytes.Equal(goSecs[kind], leanBytes) {
					t.Fatalf("section %s mismatch between Go and Lean", fname)
				}
			}

			// 5. Verify `compare-conv1x1` passes comparing expected_q1516 vs expected_f64
			// AND explicitly ignores poisoned row padding bytes beyond 2*W!
			actBytes, err := os.ReadFile(filepath.Join(vecDir, "expected_q1516.bin"))
			if err != nil {
				t.Fatalf("read expected_q1516.bin: %v", err)
			}
			rowStride := ref.AlignUp(2*tc.w, 64)
			poisoned := make([]byte, len(actBytes))
			copy(poisoned, actBytes)
			for c := 0; c < tc.cout; c++ {
				for h := 0; h < tc.h; h++ {
					rowOff := (c*tc.h + h) * rowStride
					for p := 2 * tc.w; p < rowStride; p++ {
						poisoned[rowOff+p] = 0xfe // garbage in padding area
					}
				}
			}
			poisonPath := filepath.Join(tmp, "actual_poisoned_pad.bin")
			_ = os.WriteFile(poisonPath, poisoned, 0o644)

			stdout.Reset()
			stderr.Reset()
			rc := runCLI([]string{
				"compare-conv1x1",
				"--manifest", filepath.Join(compDir, "manifest.json"),
				"--actual", poisonPath,
				"--expected", filepath.Join(vecDir, "expected_f64.bin"),
				"--atol", "1e-2",
				"--rtol", "1e-2",
				"--json",
			}, nil, &stdout, &stderr)
			if rc != 0 {
				t.Fatalf("compare-conv1x1 failed rc=%d stdout=%s stderr=%s", rc, stdout.String(), stderr.String())
			}
			var comp map[string]any
			if err := json.Unmarshal(stdout.Bytes(), &comp); err != nil {
				t.Fatalf("parse compare JSON: %v", err)
			}
			if comp["passed"] != true {
				t.Fatalf("expected passed=true, got %v", comp)
			}

			// 6. Verify `vane validate` rejects a corrupted HWX
			corrupt := make([]byte, len(hwxBytes))
			copy(corrupt, hwxBytes)
			corrupt[0] ^= 0xff
			corruptPath := filepath.Join(tmp, "corrupt.hwx")
			_ = os.WriteFile(corruptPath, corrupt, 0o644)
			cmd := exec.Command(vaneBin, "validate", "--hwx", corruptPath)
			if out, err := cmd.CombinedOutput(); err == nil {
				t.Fatalf("expected vane validate to reject corrupted HWX, output:\n%s", string(out))
			}
		})
	}
}

func TestConv3x3EndToEndCLIAndCompare(t *testing.T) {
	vaneBin := findVaneBin(t)

	cases := []struct {
		name   string
		target string
		mode   string
		h, w   int
		cin    int
		cout   int
		stride int
		pad    string
		bias   bool
		relu   bool
	}{
		{"h17_dense_s1_same_bias_relu", "h17", "dense", 8, 16, 16, 16, 1, "same", true, true},
		{"h17_dense_s2_same_even", "h17", "dense", 16, 16, 8, 16, 2, "same", true, false},
		{"h17_dense_s2_same_odd", "h17", "dense", 11, 13, 7, 9, 2, "same", false, true},
		{"h17_dense_s1_valid_cout1", "h17", "dense", 9, 11, 8, 1, 1, "valid", true, true},
		{"h17_dense_s2_valid_prime", "h17", "dense", 13, 17, 5, 11, 2, "valid", true, false},
		{"h17_dw_c1_s1_same", "h17", "depthwise", 8, 8, 1, 1, 1, "same", true, true},
		{"h17_dw_c1_s2_same", "h17", "depthwise", 8, 8, 1, 1, 2, "same", true, false},
		{"h17_dw_c29_s1_valid", "h17", "depthwise", 11, 13, 29, 29, 1, "valid", true, true},
		{"h17_dw_c24_s2_same", "h17", "depthwise", 12, 14, 24, 24, 2, "same", true, false},
		{"h16_dense_s1_same_bias", "h16", "dense", 8, 16, 16, 16, 1, "same", true, true},
		{"h16_dense_s2_same_odd", "h16", "dense", 9, 11, 5, 7, 2, "same", true, false},
		{"h16_dense_s2_valid_cout1", "h16", "dense", 10, 12, 6, 1, 2, "valid", false, true},
		{"h16_dw_c1_s1_same", "h16", "depthwise", 8, 8, 1, 1, 1, "same", true, false},
		{"h16_dw_c16_s2_valid", "h16", "depthwise", 10, 12, 16, 16, 2, "valid", true, true},
	}

	for idx, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmp := t.TempDir()
			vecDir := filepath.Join(tmp, "vec")
			compDir := filepath.Join(tmp, "compiled")

			// 1. Generate deterministic 3x3 test vectors with vane-runner gen-conv3x3
			genArgs := []string{
				"gen-conv3x3",
				"--mode", tc.mode,
				"--h", itoa(tc.h),
				"--w", itoa(tc.w),
				"--cin", itoa(tc.cin),
				"--cout", itoa(tc.cout),
				"--stride", itoa(tc.stride),
				"--pad", tc.pad,
				"--seed", itoa(idx + 200),
				"--out-dir", vecDir,
				"--json",
			}
			if tc.bias {
				genArgs = append(genArgs, "--bias")
			}
			if tc.relu {
				genArgs = append(genArgs, "--relu")
			}
			var stdout, stderr bytes.Buffer
			if rc := runCLI(genArgs, nil, &stdout, &stderr); rc != 0 {
				t.Fatalf("gen-conv3x3 failed rc=%d stderr=%s", rc, stderr.String())
			}
			var genSummary struct {
				Ho int `json:"ho"`
				Wo int `json:"wo"`
			}
			if err := json.Unmarshal(stdout.Bytes(), &genSummary); err != nil {
				t.Fatalf("parse gen-conv3x3 JSON: %v\n%s", err, stdout.String())
			}

			// 2. Compile with Lean `vane compile-conv3x3`
			compileArgs := []string{
				"compile-conv3x3",
				"--target", tc.target,
				"--mode", tc.mode,
				"--h", itoa(tc.h),
				"--w", itoa(tc.w),
				"--cin", itoa(tc.cin),
				"--cout", itoa(tc.cout),
				"--stride", itoa(tc.stride),
				"--pad", tc.pad,
				"--weights", filepath.Join(vecDir, "weights.bin"),
				"--out-dir", compDir,
			}
			if tc.bias {
				compileArgs = append(compileArgs, "--bias", filepath.Join(vecDir, "bias.bin"))
			}
			if tc.relu {
				compileArgs = append(compileArgs, "--relu")
			}
			runVane(t, vaneBin, compileArgs...)

			// 3. Validate with unified `vane validate --check-tasks`
			valOut := runVane(t, vaneBin, "validate",
				"--hwx", filepath.Join(compDir, "program.hwx"),
				"--sections", compDir,
				"--check-tasks",
			)
			if !strings.Contains(valOut, "status: ok") || !strings.Contains(valOut, "verified_conv_tasks: 1") {
				t.Fatalf("unexpected validate output:\n%s", valOut)
			}

			// 4. Verify Go `BuildSectionsFromHWX` matches Lean sections byte-for-byte
			hwxBytes, err := os.ReadFile(filepath.Join(compDir, "program.hwx"))
			if err != nil {
				t.Fatalf("read hwx: %v", err)
			}
			goSecs, err := ane.BuildSectionsFromHWX(hwxBytes)
			if err != nil {
				t.Fatalf("BuildSectionsFromHWX failed: %v", err)
			}
			for kind, fname := range ane.SectionFilenames {
				leanBytes, err := os.ReadFile(filepath.Join(compDir, fname))
				if err != nil {
					t.Fatalf("read %s: %v", fname, err)
				}
				if !bytes.Equal(goSecs[kind], leanBytes) {
					t.Fatalf("section %s mismatch between Go and Lean", fname)
				}
			}

			// 5. Verify `compare-conv` passes on expected_q1516 vs expected_f64 with poisoned padding bytes
			actBytes, err := os.ReadFile(filepath.Join(vecDir, "expected_q1516.bin"))
			if err != nil {
				t.Fatalf("read expected_q1516.bin: %v", err)
			}
			outRowStride := ref.AlignUp(2*genSummary.Wo, 64)
			poisoned := make([]byte, len(actBytes))
			copy(poisoned, actBytes)
			for c := 0; c < tc.cout; c++ {
				for h := 0; h < genSummary.Ho; h++ {
					rowOff := (c*genSummary.Ho + h) * outRowStride
					for p := 2 * genSummary.Wo; p < outRowStride; p++ {
						poisoned[rowOff+p] = 0xfe
					}
				}
			}
			poisonPath := filepath.Join(tmp, "actual_poisoned_pad.bin")
			_ = os.WriteFile(poisonPath, poisoned, 0o644)

			stdout.Reset()
			stderr.Reset()
			rc := runCLI([]string{
				"compare-conv",
				"--manifest", filepath.Join(compDir, "manifest.json"),
				"--actual", poisonPath,
				"--expected", filepath.Join(vecDir, "expected_f64.bin"),
				"--atol", "1e-2",
				"--rtol", "1e-2",
				"--json",
			}, nil, &stdout, &stderr)
			if rc != 0 {
				t.Fatalf("compare-conv failed rc=%d stdout=%s stderr=%s", rc, stdout.String(), stderr.String())
			}
			var comp map[string]any
			if err := json.Unmarshal(stdout.Bytes(), &comp); err != nil {
				t.Fatalf("parse compare JSON: %v", err)
			}
			if comp["passed"] != true {
				t.Fatalf("expected passed=true, got %v", comp)
			}
		})
	}
}

func itoa(n int) string {
	return string(bytes.TrimSpace([]byte(jsonNumber(n))))
}

func jsonNumber(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}

func TestPeEndToEndCLIAndCompare(t *testing.T) {
	vaneBin := findVaneBin(t)

	cases := []struct {
		name      string
		target    string
		op        string
		c, h, w   int
		zc, zh, zw int
		extraGen  []string
		extraComp []string
	}{
		{"h17_abs", "h17", "abs", 4, 8, 16, 0, 0, 0, nil, nil},
		{"h17_square_edge", "h17", "square", 1, 2, 16, 0, 0, 0, []string{"--edge-values"}, nil},
		{"h17_chain3", "h17", "chain", 4, 8, 16, 0, 0, 0,
			[]string{"--in-scale", "1.5", "--bias", "-0.25", "--scale", "0.5"},
			[]string{"--in-scale", "1.5", "--bias", "-0.25", "--scale", "0.5"}},
		{"h17_sdiv", "h17", "sdiv", 4, 8, 16, 0, 0, 0, []string{"--scalar", "4"}, []string{"--scalar", "4"}},
		{"h17_add_bcast_c11", "h17", "add", 4, 8, 16, 4, 1, 1, nil, nil},
		{"h17_rsub_bcast_1hw", "h17", "rsub", 4, 8, 16, 1, 8, 16, nil, nil},
		{"h16_mul_bcast_11w", "h16", "mul", 4, 8, 16, 1, 1, 16, nil, nil},
	}

	for idx, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmp := t.TempDir()
			vecDir := filepath.Join(tmp, "vec")
			compDir := filepath.Join(tmp, "comp")

			genArgs := []string{"gen-pe", "--op", tc.op, "--c", itoa(tc.c), "--h", itoa(tc.h), "--w", itoa(tc.w), "--seed", itoa(idx + 300), "--out-dir", vecDir, "--json"}
			compArgs := []string{"compile-pe", "--target", tc.target, "--op", tc.op, "--c", itoa(tc.c), "--h", itoa(tc.h), "--w", itoa(tc.w), "--out-dir", compDir}
			if tc.zc > 0 {
				genArgs = append(genArgs, "--z-c", itoa(tc.zc), "--z-h", itoa(tc.zh), "--z-w", itoa(tc.zw))
				compArgs = append(compArgs, "--z-c", itoa(tc.zc), "--z-h", itoa(tc.zh), "--z-w", itoa(tc.zw))
			}
			genArgs = append(genArgs, tc.extraGen...)
			compArgs = append(compArgs, tc.extraComp...)

			var stdout, stderr bytes.Buffer
			if rc := runCLI(genArgs, nil, &stdout, &stderr); rc != 0 {
				t.Fatalf("gen-pe failed rc=%d stderr=%s", rc, stderr.String())
			}

			runVane(t, vaneBin, compArgs...)

			valOut := runVane(t, vaneBin, "validate",
				"--hwx", filepath.Join(compDir, "program.hwx"),
				"--sections", compDir,
				"--check-tasks",
			)
			if !strings.Contains(valOut, "status: ok") || !strings.Contains(valOut, "pe=1") {
				t.Fatalf("unexpected PE validate output:\n%s", valOut)
			}

			// Verify Go BuildSectionsFromHWX matches Lean sections byte-for-byte
			hwxBytes, err := os.ReadFile(filepath.Join(compDir, "program.hwx"))
			if err != nil {
				t.Fatalf("read hwx: %v", err)
			}
			goSecs, err := ane.BuildSectionsFromHWX(hwxBytes)
			if err != nil {
				t.Fatalf("BuildSectionsFromHWX: %v", err)
			}
			for kind, fname := range ane.SectionFilenames {
				leanBytes, _ := os.ReadFile(filepath.Join(compDir, fname))
				if !bytes.Equal(goSecs[kind], leanBytes) {
					t.Fatalf("PE section %s mismatch between Go and Lean", fname)
				}
			}

			// compare-surface works on PE manifest
			if len(tc.extraGen) == 0 || tc.extraGen[0] != "--edge-values" {
				stdout.Reset()
				stderr.Reset()
				rc := runCLI([]string{
					"compare-surface",
					"--manifest", filepath.Join(compDir, "manifest.json"),
					"--actual", filepath.Join(vecDir, "expected_rna16.bin"),
					"--expected", filepath.Join(vecDir, "expected_f64.bin"),
					"--atol", "1e-2", "--rtol", "1e-2", "--json",
				}, nil, &stdout, &stderr)
				if rc != 0 {
					t.Fatalf("compare-surface failed rc=%d stdout=%s stderr=%s", rc, stdout.String(), stderr.String())
				}
			}
		})
	}
}

func TestLutAndFusedConvEndToEndCLI(t *testing.T) {
	vaneBin := findVaneBin(t)

	// 1. Standalone LUTs: verify Lean-emitted lut.bin matches Go BuildLutTable byte-for-byte!
	lutCases := []struct {
		name   string
		target string
		act    string
		extra  []string
		tol    string
	}{
		{"h17_relu", "h17", "relu", nil, "1e-3"},
		{"h17_copy", "h17", "copy", nil, "1e-3"},
		{"h17_tanh", "h17", "tanh", nil, "1e-2"},
		{"h17_gelu", "h17", "gelu", nil, "2e-2"},
		{"h17_silu", "h17", "silu", nil, "3e-2"},
		{"h17_exp", "h17", "exp", nil, "2e-1"},
		{"h17_rsqrt_eps", "h17", "rsqrt", []string{"--eps", "0.0625"}, "2e-2"},
		{"h16_hard_sigmoid", "h16", "hard_sigmoid", []string{"--alpha", "0.5", "--beta", "0.25"}, "1e-2"},
		{"h16_scaled_tanh_neg", "h16", "scaled_tanh", []string{"--alpha", "-2.0", "--beta", "0.5"}, "1e-2"},
	}

	for idx, tc := range lutCases {
		t.Run("lut_"+tc.name, func(t *testing.T) {
			tmp := t.TempDir()
			compDir := filepath.Join(tmp, "comp")
			vecDir := filepath.Join(tmp, "vec")

			compArgs := append([]string{"compile-lut", "--target", tc.target, "--act", tc.act, "--c", "4", "--h", "8", "--w", "16", "--out-dir", compDir}, tc.extra...)
			runVane(t, vaneBin, compArgs...)

			valOut := runVane(t, vaneBin, "validate",
				"--hwx", filepath.Join(compDir, "program.hwx"),
				"--sections", compDir,
				"--check-tasks",
			)
			if !strings.Contains(valOut, "status: ok") || !strings.Contains(valOut, "ne_act=1") {
				t.Fatalf("unexpected NE act validate output:\n%s", valOut)
			}

			genArgs := append([]string{"gen-lut", "--act", tc.act, "--c", "4", "--h", "8", "--w", "16", "--seed", itoa(idx + 400), "--out-dir", vecDir, "--json"}, tc.extra...)
			if tc.act != "relu" && tc.act != "copy" {
				genArgs = append(genArgs, "--lut-bin", filepath.Join(compDir, "lut.bin"))
			}
			var stdout, stderr bytes.Buffer
			if rc := runCLI(genArgs, nil, &stdout, &stderr); rc != 0 {
				t.Fatalf("gen-lut failed rc=%d stderr=%s", rc, stderr.String())
			}

			if tc.act != "relu" && tc.act != "copy" {
				stdout.Reset()
				stderr.Reset()
				rc := runCLI([]string{
					"compare-surface",
					"--manifest", filepath.Join(compDir, "manifest.json"),
					"--actual", filepath.Join(vecDir, "expected_lut.bin"),
					"--expected", filepath.Join(vecDir, "expected_f64.bin"),
					"--atol", tc.tol, "--rtol", tc.tol, "--json",
				}, nil, &stdout, &stderr)
				if rc != 0 {
					t.Fatalf("compare-surface LUT failed rc=%d stdout=%s stderr=%s", rc, stdout.String(), stderr.String())
				}
			}
		})
	}

	// 2. Fused Conv1x1 + LUT and Conv3x3 (dense s1, dw s1, dense s2 sparse) + LUT
	t.Run("fused_conv1x1_gelu", func(t *testing.T) {
		tmp := t.TempDir()
		vecDir := filepath.Join(tmp, "vec")
		compDir := filepath.Join(tmp, "comp")
		var stdout, stderr bytes.Buffer
		if rc := runCLI([]string{
			"gen-conv1x1", "--h", "8", "--w", "16", "--cin", "16", "--cout", "16",
			"--bias", "--act", "gelu", "--seed", "501", "--out-dir", vecDir,
		}, nil, &stdout, &stderr); rc != 0 {
			t.Fatalf("gen-conv1x1 --act gelu failed: %s", stderr.String())
		}
		runVane(t, vaneBin, "compile-conv1x1", "--target", "h17", "--h", "8", "--w", "16", "--cin", "16", "--cout", "16",
			"--weights", filepath.Join(vecDir, "weights.bin"),
			"--bias", filepath.Join(vecDir, "bias.bin"),
			"--act", "gelu", "--out-dir", compDir)
		valOut := runVane(t, vaneBin, "validate", "--hwx", filepath.Join(compDir, "program.hwx"), "--sections", compDir, "--check-tasks")
		if !strings.Contains(valOut, "status: ok") || !strings.Contains(valOut, "conv=1") {
			t.Fatalf("unexpected validate output:\n%s", valOut)
		}
		stdout.Reset()
		stderr.Reset()
		if rc := runCLI([]string{
			"compare-conv", "--manifest", filepath.Join(compDir, "manifest.json"),
			"--actual", filepath.Join(vecDir, "expected_lut.bin"),
			"--expected", filepath.Join(vecDir, "expected_f64.bin"),
			"--atol", "2e-2", "--rtol", "2e-2", "--json",
		}, nil, &stdout, &stderr); rc != 0 {
			t.Fatalf("compare-conv fused 1x1 failed: %s %s", stdout.String(), stderr.String())
		}
	})

	t.Run("fused_conv3x3_s2_sparse_tanh", func(t *testing.T) {
		tmp := t.TempDir()
		vecDir := filepath.Join(tmp, "vec")
		compDir := filepath.Join(tmp, "comp")
		var stdout, stderr bytes.Buffer
		if rc := runCLI([]string{
			"gen-conv3x3", "--mode", "dense", "--h", "12", "--w", "14", "--cin", "8", "--cout", "16",
			"--stride", "2", "--pad", "same", "--bias", "--act", "tanh", "--seed", "502", "--out-dir", vecDir,
		}, nil, &stdout, &stderr); rc != 0 {
			t.Fatalf("gen-conv3x3 --act tanh failed: %s", stderr.String())
		}
		runVane(t, vaneBin, "compile-conv3x3", "--target", "h17", "--mode", "dense",
			"--h", "12", "--w", "14", "--cin", "8", "--cout", "16", "--stride", "2", "--pad", "same",
			"--weights", filepath.Join(vecDir, "weights.bin"),
			"--bias", filepath.Join(vecDir, "bias.bin"),
			"--act", "tanh", "--out-dir", compDir)
		valOut := runVane(t, vaneBin, "validate", "--hwx", filepath.Join(compDir, "program.hwx"), "--sections", compDir, "--check-tasks")
		if !strings.Contains(valOut, "status: ok") || !strings.Contains(valOut, "conv=1") {
			t.Fatalf("unexpected validate output:\n%s", valOut)
		}
		// Verify Go-built LUT table matches Lean-emitted lut.bin byte-for-byte
		leanLut, err := os.ReadFile(filepath.Join(compDir, "lut.bin"))
		if err != nil {
			t.Fatalf("read lut.bin: %v", err)
		}
		goLut, _, _, err := ref.BuildLutTable(ref.ActSpec{Name: "tanh"})
		if err != nil {
			t.Fatalf("BuildLutTable: %v", err)
		}
		if !bytes.Equal(leanLut, goLut) {
			t.Fatalf("Lean lut.bin and Go BuildLutTable(tanh) differ!")
		}
		stdout.Reset()
		stderr.Reset()
		if rc := runCLI([]string{
			"compare-conv", "--manifest", filepath.Join(compDir, "manifest.json"),
			"--actual", filepath.Join(vecDir, "expected_lut.bin"),
			"--expected", filepath.Join(vecDir, "expected_f64.bin"),
			"--atol", "2e-2", "--rtol", "2e-2", "--json",
		}, nil, &stdout, &stderr); rc != 0 {
			t.Fatalf("compare-conv fused 3x3 s2 failed: %s %s", stdout.String(), stderr.String())
		}
	})
}

func TestAll23LutTablesMatchLeanByteForByteAndCliGuards(t *testing.T) {
	vaneBin := findVaneBin(t)

	// M4: byte-for-byte cross-check of ALL 23 LUT ops + parameterized variants
	// between `vane compile-lut` (`lut.bin` + manifest) and Go `ref.BuildLutTable`.
	cases := []struct {
		name      string
		spec      ref.ActSpec
		extraArgs []string
	}{
		{"relu6", ref.ActSpec{Name: "relu6"}, nil},
		{"clip", ref.ActSpec{Name: "clip", Lo: -1.5, Hi: 2.5}, []string{"--lo", "-1.5", "--hi", "2.5"}},
		{"leaky_relu", ref.ActSpec{Name: "leaky_relu", Alpha: 0.25}, []string{"--alpha", "0.25"}},
		{"abs", ref.ActSpec{Name: "abs"}, nil},
		{"sign", ref.ActSpec{Name: "sign"}, nil},
		{"threshold", ref.ActSpec{Name: "threshold", Scalar: 0.5}, []string{"--scalar", "0.5"}},
		{"thresholded_relu", ref.ActSpec{Name: "thresholded_relu", Alpha: 1.0}, []string{"--alpha", "1.0"}},
		{"hard_sigmoid_std", ref.ActSpec{Name: "hard_sigmoid", Alpha: 0.2, Beta: 0.5}, []string{"--alpha", "0.2", "--beta", "0.5"}},
		{"hard_sigmoid_shift", ref.ActSpec{Name: "hard_sigmoid", Alpha: 0.5, Beta: 0.25}, []string{"--alpha", "0.5", "--beta", "0.25"}},
		{"clamp_max", ref.ActSpec{Name: "clamp_max", Scalar: -0.5}, []string{"--scalar", "-0.5"}},
		{"clamp_min", ref.ActSpec{Name: "clamp_min", Scalar: 2.0}, []string{"--scalar", "2.0"}},
		{"elu", ref.ActSpec{Name: "elu", Alpha: 1.0}, []string{"--alpha", "1.0"}},
		{"tanh", ref.ActSpec{Name: "tanh"}, nil},
		{"scaled_tanh_pos", ref.ActSpec{Name: "scaled_tanh", Alpha: 2.0, Beta: 0.5}, []string{"--alpha", "2.0", "--beta", "0.5"}},
		{"scaled_tanh_neg", ref.ActSpec{Name: "scaled_tanh", Alpha: -2.0, Beta: -0.5}, []string{"--alpha", "-2.0", "--beta", "-0.5"}},
		{"erf", ref.ActSpec{Name: "erf"}, nil},
		{"sigmoid", ref.ActSpec{Name: "sigmoid"}, nil},
		{"gelu", ref.ActSpec{Name: "gelu"}, nil},
		{"silu", ref.ActSpec{Name: "silu"}, nil},
		{"exp2", ref.ActSpec{Name: "exp2"}, nil},
		{"exp", ref.ActSpec{Name: "exp"}, nil},
		{"sqrt", ref.ActSpec{Name: "sqrt"}, nil},
		{"rsqrt_zero", ref.ActSpec{Name: "rsqrt", Eps: 0.0}, []string{"--eps", "0.0"}},
		{"rsqrt_eps", ref.ActSpec{Name: "rsqrt", Eps: 0.01}, []string{"--eps", "0.01"}},
		{"inverse", ref.ActSpec{Name: "inverse"}, nil},
		{"log2", ref.ActSpec{Name: "log2"}, nil},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmp := t.TempDir()
			compArgs := append([]string{
				"compile-lut", "--target", "h17", "--act", tc.spec.Name,
				"--c", "2", "--h", "4", "--w", "8", "--out-dir", tmp,
			}, tc.extraArgs...)
			runVane(t, vaneBin, compArgs...)

			leanLut, err := os.ReadFile(filepath.Join(tmp, "lut.bin"))
			if err != nil {
				t.Fatalf("read lut.bin: %v", err)
			}
			goLut, goPreScale, goShift, err := ref.BuildLutTable(tc.spec)
			if err != nil {
				t.Fatalf("BuildLutTable(%s): %v", tc.name, err)
			}
			if !bytes.Equal(leanLut, goLut) {
				t.Fatalf("lut.bin mismatch for %s:\nlean=%x\ngo  =%x", tc.name, leanLut, goLut)
			}

			var m struct {
				PreScaleF16   uint16  `json:"pre_scale_f16"`
				InputShiftF16 *uint16 `json:"input_shift_f16"`
			}
			mRaw, err := os.ReadFile(filepath.Join(tmp, "manifest.json"))
			if err != nil {
				t.Fatalf("read manifest.json: %v", err)
			}
			if err := json.Unmarshal(mRaw, &m); err != nil {
				t.Fatalf("parse manifest.json: %v", err)
			}
			if m.PreScaleF16 != goPreScale {
				t.Fatalf("%s pre_scale_f16 mismatch: lean=0x%04x, go=0x%04x", tc.name, m.PreScaleF16, goPreScale)
			}
			if (m.InputShiftF16 == nil) != (goShift == nil) {
				t.Fatalf("%s input_shift_f16 presence mismatch: lean=%v, go=%v", tc.name, m.InputShiftF16, goShift)
			}
			if m.InputShiftF16 != nil && *m.InputShiftF16 != *goShift {
				t.Fatalf("%s input_shift_f16 mismatch: lean=0x%04x, go=0x%04x", tc.name, *m.InputShiftF16, *goShift)
			}
		})
	}

	// N3: gen-pe --op sdiv rejects underflowing fp16 zero and overflowing reciprocal
	t.Run("n3_gen_pe_sdiv_guards", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		tmp := t.TempDir()
		if rc := runCLI([]string{"gen-pe", "--op", "sdiv", "--c", "1", "--h", "2", "--w", "4", "--scalar", "1e-9", "--out-dir", tmp}, nil, &stdout, &stderr); rc == 0 {
			t.Fatalf("expected gen-pe sdiv with 1e-9 (fp16 zero) to fail")
		}
		stderr.Reset()
		if rc := runCLI([]string{"gen-pe", "--op", "sdiv", "--c", "1", "--h", "2", "--w", "4", "--scalar", "1e-6", "--out-dir", tmp}, nil, &stdout, &stderr); rc == 0 {
			t.Fatalf("expected gen-pe sdiv with 1e-6 (reciprocal overflow) to fail")
		}
	})

	// N5: gen-lut --all-fp16 rejects C*H*W != 65536
	t.Run("n5_gen_lut_all_fp16_guard", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		tmp := t.TempDir()
		if rc := runCLI([]string{"gen-lut", "--act", "tanh", "--c", "1", "--h", "8", "--w", "8", "--all-fp16", "--out-dir", tmp}, nil, &stdout, &stderr); rc == 0 {
			t.Fatalf("expected gen-lut --all-fp16 with 64 elements to fail")
		}
		stderr.Reset()
		if rc := runCLI([]string{"gen-lut", "--act", "tanh", "--c", "1", "--h", "256", "--w", "256", "--all-fp16", "--out-dir", tmp}, nil, &stdout, &stderr); rc != 0 {
			t.Fatalf("expected gen-lut --all-fp16 with 65536 elements to succeed, stderr=%s", stderr.String())
		}
	})
}

func TestMultiTaskGraphEndToEndCLI(t *testing.T) {
	vaneBin := findVaneBin(t)

	cases := []struct {
		name           string
		specJSON       string
		wantTasks      int
		wantScratch    bool
		maxScratch     int
		tol            string
	}{
		{
			name: "h17_l2_conv_chain",
			specJSON: `{
				"target": "h17",
				"placement": "l2",
				"inputs": [{"name": "x", "c": 16, "h": 8, "w": 16}],
				"nodes": [
					{"id": "c1", "op": "conv1x1", "inputs": ["x"], "cout": 16, "bias": true, "relu": true},
					{"id": "c2", "op": "conv3x3", "mode": "depthwise", "inputs": ["c1"], "stride": 1, "pad": "same"},
					{"id": "c3", "op": "conv1x1", "inputs": ["c2"], "cout": 16, "bias": true}
				]
			}`,
			wantTasks:   3,
			wantScratch: false,
			tol:         "2e-2",
		},
		{
			name: "h17_dram_conv_chain_reuse",
			specJSON: `{
				"target": "h17",
				"placement": "dram",
				"inputs": [{"name": "x", "c": 16, "h": 16, "w": 16}],
				"nodes": [
					{"id": "c1", "op": "conv1x1", "inputs": ["x"], "cout": 16, "relu": true},
					{"id": "c2", "op": "conv3x3", "mode": "depthwise", "inputs": ["c1"], "stride": 1, "pad": "same"},
					{"id": "c3", "op": "conv3x3", "mode": "dense", "inputs": ["c2"], "cout": 32, "stride": 2, "pad": "same", "relu": true},
					{"id": "c4", "op": "conv1x1", "inputs": ["c3"], "cout": 16}
				]
			}`,
			wantTasks:   4,
			wantScratch: true,
			maxScratch:  49151, // strictly < sum of 3 intermediates (49152B), confirming scratch reuse
			tol:         "3e-2",
		},
		{
			name: "h17_mixed_ne_pe_lut",
			specJSON: `{
				"target": "h17",
				"placement": "auto",
				"inputs": [{"name": "x", "c": 16, "h": 8, "w": 16}],
				"nodes": [
					{"id": "c1", "op": "conv1x1", "inputs": ["x"], "cout": 16, "bias": true, "act": "gelu"},
					{"id": "p1", "op": "pe", "pe_op": "chain", "inputs": ["c1"], "in_scale": 0.5, "bias": 0.125},
					{"id": "l1", "op": "lut", "inputs": ["p1"], "act": "silu"},
					{"id": "c2", "op": "conv1x1", "inputs": ["l1"], "cout": 16}
				]
			}`,
			wantTasks:   4,
			wantScratch: false,
			tol:         "5e-2",
		},
		{
			name: "h17_residual_hybrid",
			specJSON: `{
				"target": "h17",
				"placement": "auto",
				"inputs": [{"name": "x", "c": 16, "h": 8, "w": 16}],
				"nodes": [
					{"id": "c1", "op": "conv3x3", "mode": "dense", "inputs": ["x"], "cout": 16, "stride": 1, "pad": "same", "relu": true, "out_placement": "l2"},
					{"id": "c2", "op": "conv1x1", "inputs": ["c1"], "cout": 16, "out_placement": "dram"},
					{"id": "res", "op": "pe", "pe_op": "add", "inputs": ["c2", "x"]}
				]
			}`,
			wantTasks:   3,
			wantScratch: true,
			tol:         "3e-2",
		},
		{
			name: "h16_two_input_broadcast_residual",
			specJSON: `{
				"target": "h16",
				"placement": "auto",
				"inputs": [
					{"name": "x", "c": 16, "h": 8, "w": 16},
					{"name": "z", "c": 16, "h": 1, "w": 1}
				],
				"nodes": [
					{"id": "c1", "op": "conv1x1", "inputs": ["x"], "cout": 16, "bias": true, "relu": true},
					{"id": "out", "op": "pe", "pe_op": "add", "inputs": ["c1", "z"]}
				]
			}`,
			wantTasks:   2,
			wantScratch: false,
			tol:         "2e-2",
		},
	}

	for idx, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmp := t.TempDir()
			specFile := filepath.Join(tmp, "graph.json")
			vecDir := filepath.Join(tmp, "vec")
			compDir := filepath.Join(tmp, "comp")
			if err := os.WriteFile(specFile, []byte(tc.specJSON), 0o644); err != nil {
				t.Fatalf("write spec: %v", err)
			}

			var stdout, stderr bytes.Buffer
			if rc := runCLI([]string{
				"gen-graph", "--spec", specFile, "--seed", itoa(idx + 700), "--out-dir", vecDir, "--json",
			}, nil, &stdout, &stderr); rc != 0 {
				t.Fatalf("gen-graph failed rc=%d stderr=%s", rc, stderr.String())
			}

			runVane(t, vaneBin, "compile-graph",
				"--spec", filepath.Join(vecDir, "graph_resolved.json"),
				"--out-dir", compDir,
			)

			valOut := runVane(t, vaneBin, "validate",
				"--hwx", filepath.Join(compDir, "program.hwx"),
				"--sections", compDir,
				"--check-tasks",
			)
			if !strings.Contains(valOut, "status: ok") {
				t.Fatalf("unexpected graph validate output:\n%s", valOut)
			}

			// Verify manifest fields
			var man struct {
				TaskCount    int  `json:"task_count"`
				HasScratch   bool `json:"has_scratch"`
				ScratchBytes int  `json:"scratch_bytes"`
			}
			mRaw, err := os.ReadFile(filepath.Join(compDir, "manifest.json"))
			if err != nil {
				t.Fatalf("read manifest: %v", err)
			}
			if err := json.Unmarshal(mRaw, &man); err != nil {
				t.Fatalf("parse manifest: %v", err)
			}
			if man.TaskCount != tc.wantTasks {
				t.Fatalf("task_count = %d, want %d", man.TaskCount, tc.wantTasks)
			}
			if man.HasScratch != tc.wantScratch {
				t.Fatalf("has_scratch = %t, want %t", man.HasScratch, tc.wantScratch)
			}
			if tc.maxScratch > 0 && man.ScratchBytes > tc.maxScratch {
				t.Fatalf("scratch_bytes = %d > max %d (expected liveness reuse)", man.ScratchBytes, tc.maxScratch)
			}

			// Verify Go BuildSectionsFromHWX matches Lean sections byte-for-byte
			hwxBytes, err := os.ReadFile(filepath.Join(compDir, "program.hwx"))
			if err != nil {
				t.Fatalf("read hwx: %v", err)
			}
			goSecs, err := ane.BuildSectionsFromHWX(hwxBytes)
			if err != nil {
				t.Fatalf("BuildSectionsFromHWX: %v", err)
			}
			for kind, fname := range ane.SectionFilenames {
				leanBytes, err := os.ReadFile(filepath.Join(compDir, fname))
				if err != nil {
					t.Fatalf("read %s: %v", fname, err)
				}
				if !bytes.Equal(goSecs[kind], leanBytes) {
					t.Fatalf("graph section %s mismatch between Go and Lean", fname)
				}
			}

			// Verify compare-surface works on graph manifest (Q15.16 vs F64)
			stdout.Reset()
			stderr.Reset()
			if rc := runCLI([]string{
				"compare-surface",
				"--manifest", filepath.Join(compDir, "manifest.json"),
				"--actual", filepath.Join(vecDir, "expected_q1516.bin"),
				"--expected", filepath.Join(vecDir, "expected_f64.bin"),
				"--atol", tc.tol, "--rtol", tc.tol, "--json",
			}, nil, &stdout, &stderr); rc != 0 {
				t.Fatalf("compare-surface on graph failed rc=%d stdout=%s stderr=%s", rc, stdout.String(), stderr.String())
			}
		})
	}
}

func TestRunAltInputMockTransport(t *testing.T) {
	vaneBin := findVaneBin(t)
	tmp := t.TempDir()
	demoDir := filepath.Join(tmp, "demo_alt")
	runVane(t, vaneBin, "emit-demo", "--target", "h17", "--out-dir", demoDir)

	var manifest struct {
		Surfaces []struct {
			Span int `json:"span"`
		} `json:"surfaces"`
	}
	mRaw, _ := os.ReadFile(filepath.Join(demoDir, "manifest.json"))
	if err := json.Unmarshal(mRaw, &manifest); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	inSpan := manifest.Surfaces[0].Span
	inA := make([]uint16, inSpan/2)
	inB := make([]uint16, inSpan/2)
	for i := range inA {
		inA[i] = fp16.FromFloat64(float64(i%7+1) * 0.25)
		inB[i] = fp16.FromFloat64(float64(i%5+2) * 0.5)
	}
	inAPath := filepath.Join(tmp, "inA.bin")
	inBPath := filepath.Join(tmp, "inB.bin")
	_ = os.WriteFile(inAPath, fp16.EncodeLE(inA), 0o644)
	_ = os.WriteFile(inBPath, fp16.EncodeLE(inB), 0o644)

	// 1. Copy mode and in-place mode with --alt-input --runs 4
	for _, inPlace := range []bool{false, true} {
		var observed [][]byte
		tr := ane.NewMockTransport(map[string]ane.MockDeviceSpec{
			"/dev/accel/accel0": {ABIVersion: ane.ABIV3, ChipFamily: ane.ChipH17},
		}, func(inputs [][]byte, _ int) []byte {
			cp := make([]byte, len(inputs[0]))
			copy(cp, inputs[0])
			observed = append(observed, cp)
			// Return deterministic transform of input (XOR low byte with 0x55)
			out := make([]byte, len(inputs[0]))
			for i, b := range inputs[0] {
				out[i] = b ^ 0x55
			}
			return out
		})

		args := []string{
			"run",
			"--hwx", filepath.Join(demoDir, "program.hwx"),
			"--sections-dir", demoDir,
			"--input", inAPath,
			"--alt-input", inBPath,
			"--runs", "4",
			"--json",
		}
		if inPlace {
			args = append(args, "--in-place")
		}
		var stdout, stderr bytes.Buffer
		if rc := runCLI(args, tr, &stdout, &stderr); rc != 0 {
			t.Fatalf("run --alt-input (inPlace=%t) failed rc=%d stderr=%s", inPlace, rc, stderr.String())
		}
		if len(observed) != 4 {
			t.Fatalf("expected 4 runs, got %d", len(observed))
		}
		rawA := fp16.EncodeLE(inA)
		rawB := fp16.EncodeLE(inB)
		if !bytes.Equal(observed[0], rawA) || !bytes.Equal(observed[1], rawB) ||
			!bytes.Equal(observed[2], rawA) || !bytes.Equal(observed[3], rawB) {
			t.Fatalf("inputs did not alternate A, B, A, B (inPlace=%t)", inPlace)
		}
		var rep ane.RunReport
		if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
			t.Fatalf("parse run report: %v", err)
		}
		if !rep.Repeatable || rep.AltOutputSHA256 == "" || rep.OutputSHA256 == rep.AltOutputSHA256 {
			t.Fatalf("unexpected alt-input report: repeatable=%t sha=%s alt_sha=%s",
				rep.Repeatable, rep.OutputSHA256, rep.AltOutputSHA256)
		}
	}

	// 2. Non-deterministic alt-input run (run 4 disagrees with run 2) must fail
	callIdx := 0
	badTr := ane.NewMockTransport(map[string]ane.MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ane.ABIV3, ChipFamily: ane.ChipH17},
	}, func(inputs [][]byte, _ int) []byte {
		callIdx++
		out := make([]byte, len(inputs[0]))
		copy(out, inputs[0])
		if callIdx == 4 {
			out[0] ^= 0xff // corrupt 2nd execution of alt-input
		}
		return out
	})
	var stdout, stderr bytes.Buffer
	rc := runCLI([]string{
		"run",
		"--hwx", filepath.Join(demoDir, "program.hwx"),
		"--sections-dir", demoDir,
		"--input", inAPath,
		"--alt-input", inBPath,
		"--runs", "4",
	}, badTr, &stdout, &stderr)
	if rc == 0 || !strings.Contains(stderr.String(), "non-deterministic alt-input") {
		t.Fatalf("expected non-deterministic alt-input failure, got rc=%d stderr=%s", rc, stderr.String())
	}

	// 3. [N4] --alt-input with --runs 1 must fail cleanly instead of silently skipping alt input
	stdout.Reset()
	stderr.Reset()
	rc = runCLI([]string{
		"run",
		"--hwx", filepath.Join(demoDir, "program.hwx"),
		"--sections-dir", demoDir,
		"--input", inAPath,
		"--alt-input", inBPath,
		"--runs", "1",
	}, badTr, &stdout, &stderr)
	if rc == 0 || !strings.Contains(stderr.String(), "--alt-input requires --runs >= 2") {
		t.Fatalf("expected --alt-input --runs 1 failure, got rc=%d stderr=%s", rc, stderr.String())
	}
}

func TestGraphCliValidationGuards(t *testing.T) {
	vaneBin := findVaneBin(t)
	tmp := t.TempDir()

	// [N2a] gen-graph rejects mismatched explicit cin on conv1x1/conv3x3
	badCinSpec := filepath.Join(tmp, "bad_cin.json")
	_ = os.WriteFile(badCinSpec, []byte(`{
		"target": "h17",
		"inputs": [{"name": "x", "c": 8, "h": 4, "w": 8}],
		"nodes": [{"id": "c1", "op": "conv1x1", "inputs": ["x"], "cin": 16, "cout": 8}]
	}`), 0o644)
	var stdout, stderr bytes.Buffer
	if rc := runCLI([]string{"gen-graph", "--spec", badCinSpec, "--out-dir", filepath.Join(tmp, "out_cin")}, nil, &stdout, &stderr); rc == 0 || !strings.Contains(stderr.String(), "cin 16 != input channels 8") {
		t.Fatalf("expected gen-graph cin mismatch error, got rc=%d stderr=%s", rc, stderr.String())
	}

	// [N2b] gen-graph rejects incompatible PE broadcast dimensions
	badBcastSpec := filepath.Join(tmp, "bad_bcast.json")
	_ = os.WriteFile(badBcastSpec, []byte(`{
		"target": "h17",
		"inputs": [
			{"name": "x", "c": 8, "h": 4, "w": 8},
			{"name": "z", "c": 4, "h": 4, "w": 8}
		],
		"nodes": [{"id": "p1", "op": "pe", "pe_op": "add", "inputs": ["x", "z"]}]
	}`), 0o644)
	stderr.Reset()
	if rc := runCLI([]string{"gen-graph", "--spec", badBcastSpec, "--out-dir", filepath.Join(tmp, "out_bcast")}, nil, &stdout, &stderr); rc == 0 || !strings.Contains(stderr.String(), "incompatible broadcast") {
		t.Fatalf("expected gen-graph broadcast mismatch error, got rc=%d stderr=%s", rc, stderr.String())
	}

	// [N3] vane compile-graph rejects PE scalar/chain float that overflows finite fp16 range
	badF16Spec := filepath.Join(tmp, "bad_f16.json")
	_ = os.WriteFile(badF16Spec, []byte(`{
		"target": "h17",
		"inputs": [{"name": "x", "c": 4, "h": 4, "w": 8}],
		"nodes": [{"id": "p1", "op": "pe", "pe_op": "smul", "inputs": ["x"], "scalar": 100000.0}]
	}`), 0o644)
	cmd := exec.Command(vaneBin, "compile-graph", "--spec", badF16Spec, "--out-dir", filepath.Join(tmp, "out_f16"))
	outBytes, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(outBytes), "overflows finite fp16") {
		t.Fatalf("expected compile-graph fp16 overflow error, got err=%v output=%s", err, string(outBytes))
	}
}

func TestRunWarmupAndLatencyStatsCLI(t *testing.T) {
	vaneBin := findVaneBin(t)
	tmp := t.TempDir()
	demoDir := filepath.Join(tmp, "demo_warmup")
	runVane(t, vaneBin, "emit-demo", "--target", "h17", "--out-dir", demoDir)

	var calls int
	tr := ane.NewMockTransport(map[string]ane.MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ane.ABIV3, ChipFamily: ane.ChipH17},
	}, func(inputs [][]byte, _ int) []byte {
		calls++
		cp := make([]byte, len(inputs[0]))
		copy(cp, inputs[0])
		return cp
	})

	// JSON output includes warmup, min_ms, median_ms, p95_ms, mean_ms and exact timed count
	var stdout, stderr bytes.Buffer
	rc := runCLI([]string{
		"run",
		"--hwx", filepath.Join(demoDir, "program.hwx"),
		"--sections-dir", demoDir,
		"--warmup", "2",
		"--runs", "4",
		"--json",
	}, tr, &stdout, &stderr)
	if rc != 0 {
		t.Fatalf("run --warmup 2 --runs 4 --json failed rc=%d stderr=%s", rc, stderr.String())
	}
	if calls != 6 {
		t.Fatalf("expected 6 total mock calls (2 warmup + 4 timed), got %d", calls)
	}
	var rep ane.RunReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal RunReport: %v", err)
	}
	if rep.Warmup != 2 || rep.Runs != 4 || len(rep.LatenciesMS) != 4 {
		t.Fatalf("unexpected report counts: warmup=%d runs=%d len(latencies)=%d", rep.Warmup, rep.Runs, len(rep.LatenciesMS))
	}
	if rep.MinMS > rep.MedianMS || rep.MedianMS > rep.P95MS {
		t.Fatalf("unexpected stats ordering: min=%v median=%v p95=%v", rep.MinMS, rep.MedianMS, rep.P95MS)
	}

	// Human-readable output prints warmup=, min_ms=, median_ms=, p95_ms=, mean_ms=
	stdout.Reset()
	stderr.Reset()
	rc = runCLI([]string{
		"run",
		"--hwx", filepath.Join(demoDir, "program.hwx"),
		"--sections-dir", demoDir,
		"--warmup", "1",
		"--runs", "3",
	}, tr, &stdout, &stderr)
	if rc != 0 {
		t.Fatalf("run text output failed rc=%d stderr=%s", rc, stderr.String())
	}
	txt := stdout.String()
	for _, want := range []string{"warmup=1", "min_ms=", "median_ms=", "p95_ms=", "mean_ms="} {
		if !strings.Contains(txt, want) {
			t.Fatalf("expected %q in run text output, got: %s", want, txt)
		}
	}

	// Negative --warmup rejected cleanly
	stdout.Reset()
	stderr.Reset()
	if rc := runCLI([]string{
		"run",
		"--hwx", filepath.Join(demoDir, "program.hwx"),
		"--warmup", "-1",
	}, tr, &stdout, &stderr); rc == 0 || !strings.Contains(stderr.String(), "--warmup must be >= 0") {
		t.Fatalf("expected negative --warmup error, got rc=%d stderr=%s", rc, stderr.String())
	}
}

func TestCompileGraphStripHeightCLI(t *testing.T) {
	vaneBin := findVaneBin(t)
	tmp := t.TempDir()
	specFile := filepath.Join(tmp, "graph.json")
	vecDir := filepath.Join(tmp, "vec")
	compDir := filepath.Join(tmp, "comp")

	specJSON := `{
		"target": "h17",
		"placement": "auto",
		"strip_height": 8,
		"inputs": [{"name": "x", "c": 8, "h": 20, "w": 32}],
		"nodes": [
			{"id": "c0", "op": "conv3x3", "mode": "dense", "inputs": ["x"], "cout": 8, "stride": 1, "pad": "same", "relu": true},
			{"id": "c1", "op": "conv3x3", "mode": "dense", "inputs": ["c0"], "cout": 8, "stride": 1, "pad": "same", "relu": true},
			{"id": "c2", "op": "conv1x1", "inputs": ["c1"], "cout": 8}
		]
	}`
	if err := os.WriteFile(specFile, []byte(specJSON), 0o644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	var stdout, stderr bytes.Buffer
	if rc := runCLI([]string{"gen-graph", "--spec", specFile, "--seed", "901", "--out-dir", vecDir, "--json"}, nil, &stdout, &stderr); rc != 0 {
		t.Fatalf("gen-graph failed rc=%d stderr=%s", rc, stderr.String())
	}

	// CLI --strip-height 6 overrides JSON strip_height=8 -> ceil(20/6)=4 strips * 3 nodes = 12 tasks
	compOut := runVane(t, vaneBin, "compile-graph",
		"--spec", filepath.Join(vecDir, "graph_resolved.json"),
		"--strip-height", "6",
		"--out-dir", compDir,
	)
	if !strings.Contains(compOut, "strip_count=4, strip_height=6") {
		t.Fatalf("expected strip summary in compile-graph output, got: %s", compOut)
	}

	valOut := runVane(t, vaneBin, "validate",
		"--hwx", filepath.Join(compDir, "program.hwx"),
		"--sections", compDir,
		"--check-tasks",
	)
	if !strings.Contains(valOut, "status: ok") {
		t.Fatalf("stripped graph validate failed:\n%s", valOut)
	}

	var man struct {
		TaskCount   int `json:"task_count"`
		StripCount  int `json:"strip_count"`
		StripHeight int `json:"strip_height"`
	}
	mRaw, err := os.ReadFile(filepath.Join(compDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(mRaw, &man); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	if man.TaskCount != 12 || man.StripCount != 4 || man.StripHeight != 6 {
		t.Fatalf("unexpected stripped manifest: %+v", man)
	}
}
