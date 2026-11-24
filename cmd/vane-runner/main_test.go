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

	"github.com/niteria/vane/internal/ane"
	"github.com/niteria/vane/internal/fp16"
	"github.com/niteria/vane/internal/ref"
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

func itoa(n int) string {
	return string(bytes.TrimSpace([]byte(jsonNumber(n))))
}

func jsonNumber(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}
