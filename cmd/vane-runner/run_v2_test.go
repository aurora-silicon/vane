package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/aurora-silicon/vane/internal/ane"
	"github.com/aurora-silicon/vane/internal/fp16"
)

func makeTestV2BundleFiles(t *testing.T, bufSize int) (bundlePath string, inBytes, altBytes, expBytes []byte) {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "sec1.bin"), []byte("v2-section-1-data"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "sec2.bin"), []byte("v2-section-2-data"), 0o644); err != nil {
		t.Fatal(err)
	}
	manifest := map[string]any{
		"sections": []map[string]any{
			{"id": 1, "file": "sec1.bin"},
			{"id": 2, "file": "sec2.bin"},
		},
		"generic_binds": []map[string]any{
			{"buffer_id": 4, "type": 1, "size": bufSize},
		},
		"ios": []map[string]any{
			{"buffer_id": 5, "type": 0, "direction": "input", "size": bufSize},
			{"buffer_id": 6, "type": 2, "direction": "scratch", "size": 256},
			{"buffer_id": 4, "type": 1, "direction": "output", "size": bufSize},
		},
		"priority":   3,
		"timeout_ms": 5000,
	}
	raw, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	bundlePath = filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(bundlePath, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	elemCount := bufSize / 2
	inVals := make([]uint16, elemCount)
	altVals := make([]uint16, elemCount)
	expVals := make([]uint16, elemCount)
	for i := range inVals {
		v := float64(i%17-8) * 0.25
		inVals[i] = fp16.FromFloat64(v)
		altVals[i] = fp16.FromFloat64(v + 0.5)
		if v < 0 {
			v = 0
		}
		expVals[i] = fp16.FromFloat64(v)
	}
	return bundlePath, fp16.EncodeLE(inVals), fp16.EncodeLE(altVals), fp16.EncodeLE(expVals)
}

func newReLUMockV2Transport(t *testing.T) *ane.MockTransport {
	t.Helper()
	tr := ane.NewMockTransport(map[string]ane.MockDeviceSpec{
		"/dev/accel/accel0": {ABIVersion: ane.ABIV3, ChipFamily: ane.ChipH17},
		"/dev/accel/accel1": {ABIVersion: ane.ABIV2, ChipFamily: ane.ChipH14, SectionSize: 24, BindSize: 24, ExecIOSize: 32},
	}, nil)
	tr.OnExecV2 = func(_ ane.ExecV2, _ map[uint32][]byte, bufs []ane.MockExecV2Buffer, _ int) error {
		var inData, outData []byte
		for _, b := range bufs {
			switch b.IO.Type {
			case 0:
				inData = b.Data
			case 1:
				outData = b.Data
			}
		}
		for _, v := range outData {
			if v != 0xff {
				t.Errorf("expected 0xff output poison before EXEC, got 0x%02x", v)
				break
			}
		}
		vals, _ := fp16.DecodeLE(inData)
		out := make([]uint16, len(vals))
		for i, v := range vals {
			f := fp16.ToFloat64(v)
			if f < 0 {
				f = 0
			}
			out[i] = fp16.FromFloat64(f)
		}
		copy(outData, fp16.EncodeLE(out))
		return nil
	}
	return tr
}

func TestRunV2CLIHappyPathExpectedJSONAndOverrides(t *testing.T) {
	bundlePath, inBytes, altBytes, expBytes := makeTestV2BundleFiles(t, 128)
	tmp := t.TempDir()
	inPath := filepath.Join(tmp, "in.bin")
	altPath := filepath.Join(tmp, "alt.bin")
	expPath := filepath.Join(tmp, "exp.bin")
	badExpPath := filepath.Join(tmp, "bad_exp.bin")
	outPath := filepath.Join(tmp, "out.bin")
	jsonPath := filepath.Join(tmp, "report.json")

	_ = os.WriteFile(inPath, inBytes, 0o644)
	_ = os.WriteFile(altPath, altBytes, 0o644)
	_ = os.WriteFile(expPath, expBytes, 0o644)
	badExp := make([]byte, len(expBytes))
	for i := range badExp {
		badExp[i] = expBytes[i] ^ 0x7f
	}
	_ = os.WriteFile(badExpPath, badExp, 0o644)

	// 1. Happy path with --json, --json-out, --output-out, --expected (pass), --alt-input, --priority, --timeout-ms.
	tr := newReLUMockV2Transport(t)
	var stdout, stderr bytes.Buffer
	rc := runCLI([]string{
		"run-v2",
		"--bundle", bundlePath,
		"--input", inPath,
		"--alt-input", altPath,
		"--expected", expPath,
		"--output-out", outPath,
		"--json-out", jsonPath,
		"--priority", "6",
		"--timeout-ms", "1500",
		"--runs", "4",
		"--warmup", "1",
		"--max-ulp", "0",
		"--json",
	}, tr, &stdout, &stderr)
	if rc != 0 {
		t.Fatalf("run-v2 failed rc=%d stderr=%s", rc, stderr.String())
	}

	var rep ane.RunReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("parse run-v2 JSON: %v\n%s", err, stdout.String())
	}
	if rep.Mode != "abi-v2" || rep.ABIVersion != ane.ABIV2 || rep.ChipFamily != ane.ChipH14 || rep.Device != "/dev/accel/accel1" {
		t.Fatalf("unexpected report identity: %+v", rep)
	}
	if rep.Runs != 4 || rep.Warmup != 1 || !rep.Repeatable || rep.AltOutputSHA256 == "" || rep.AltOutputSHA256 == rep.OutputSHA256 {
		t.Fatalf("unexpected report timings/hashes: %+v", rep)
	}
	if rep.Comparison == nil || !rep.Comparison.Passed {
		t.Fatalf("expected comparison passed, got %+v", rep.Comparison)
	}
	if tr.LastExecV2Args == nil || tr.LastExecV2Args.Priority != 6 || tr.LastExecV2Args.TimeoutMS != 1500 {
		t.Fatalf("expected CLI overrides priority=6 timeout=1500, got %+v", tr.LastExecV2Args)
	}
	if tr.ExecV2Calls != 5 || tr.ActiveBOCount() != 0 || tr.ActiveMappedBOCount() != 0 || tr.TotalAllocatedBOs != tr.TotalFreedBOs {
		t.Fatalf("expected 5 execs and zero leaks: execs=%d active=%d mapped=%d alloc=%d freed=%d",
			tr.ExecV2Calls, tr.ActiveBOCount(), tr.ActiveMappedBOCount(), tr.TotalAllocatedBOs, tr.TotalFreedBOs)
	}
	gotOut, err := os.ReadFile(outPath)
	if err != nil || !bytes.Equal(gotOut, expBytes) {
		t.Fatalf("--output-out mismatch (err=%v)", err)
	}
	if _, err := os.Stat(jsonPath); err != nil {
		t.Fatalf("--json-out missing: %v", err)
	}

	// 2. Human-readable output format + --expected mismatch returns exit code 1.
	stdout.Reset()
	stderr.Reset()
	rc = runCLI([]string{
		"run-v2",
		"--bundle", bundlePath,
		"--input", inPath,
		"--expected", badExpPath,
		"--runs", "1",
	}, tr, &stdout, &stderr)
	if rc != 1 {
		t.Fatalf("run-v2 expected mismatch rc=%d, want 1", rc)
	}
	text := stdout.String()
	if !strings.Contains(text, "mode=abi-v2") || !strings.Contains(text, "reference: passed=false") {
		t.Fatalf("unexpected text output on mismatch:\n%s", text)
	}
	if tr.ActiveBOCount() != 0 || tr.ActiveMappedBOCount() != 0 {
		t.Fatalf("leak after expected mismatch: active=%d mapped=%d", tr.ActiveBOCount(), tr.ActiveMappedBOCount())
	}
}

func TestRunV2CLISkipAndErrors(t *testing.T) {
	bundlePath, _, _, _ := makeTestV2BundleFiles(t, 64)

	// 1. Skip exit code 4 when no ABI v2 device is present (empty device list or only v1/v3 devices).
	for _, tc := range []struct {
		name string
		devs map[string]ane.MockDeviceSpec
	}{
		{"empty", map[string]ane.MockDeviceSpec{}},
		{"v1_and_v3_only", map[string]ane.MockDeviceSpec{
			"/dev/accel/accel0": {ABIVersion: ane.ABIV1, ChipFamily: ane.ChipH13},
			"/dev/accel/accel1": {ABIVersion: ane.ABIV3, ChipFamily: ane.ChipH17},
		}},
	} {
		t.Run("skip_"+tc.name, func(t *testing.T) {
			tr := ane.NewMockTransport(tc.devs, nil)
			var stdout, stderr bytes.Buffer
			rc := runCLI([]string{"run-v2", "--bundle", bundlePath}, tr, &stdout, &stderr)
			if rc != ane.ExitSkip {
				t.Fatalf("run-v2 (%s) rc=%d, want %d (stderr=%s)", tc.name, rc, ane.ExitSkip, stderr.String())
			}
		})
	}

	// 2. Missing --bundle, malformed manifest, invalid CLI flags -> exit code 1 (even without devices).
	emptyTr := ane.NewMockTransport(map[string]ane.MockDeviceSpec{}, nil)
	tmp := t.TempDir()
	badManifest := filepath.Join(tmp, "bad.json")
	_ = os.WriteFile(badManifest, []byte(`{"sections":[],"ios":[]}`), 0o644)

	errCases := []struct {
		name string
		args []string
	}{
		{"missing_bundle", []string{"run-v2"}},
		{"nonexistent_bundle", []string{"run-v2", "--bundle", filepath.Join(tmp, "nope.json")}},
		{"malformed_manifest", []string{"run-v2", "--bundle", badManifest}},
		{"bad_runs", []string{"run-v2", "--bundle", bundlePath, "--runs", "0"}},
		{"bad_warmup", []string{"run-v2", "--bundle", bundlePath, "--warmup", "-1"}},
		{"bad_priority_low", []string{"run-v2", "--bundle", bundlePath, "--priority", "1"}},
		{"bad_priority_high", []string{"run-v2", "--bundle", bundlePath, "--priority", "8"}},
		{"bad_priority_neg", []string{"run-v2", "--bundle", bundlePath, "--priority", "-1"}},
		{"bad_timeout_high", []string{"run-v2", "--bundle", bundlePath, "--timeout-ms", "5001"}},
		{"bad_timeout_neg", []string{"run-v2", "--bundle", bundlePath, "--timeout-ms", "-1"}},
		{"unexpected_positional", []string{"run-v2", "--bundle", bundlePath, "extra.bin"}},
	}
	for _, tc := range errCases {
		t.Run("err_"+tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			rc := runCLI(tc.args, emptyTr, &stdout, &stderr)
			if rc != 1 {
				t.Fatalf("run-v2 (%s) rc=%d, want 1 (stderr=%s)", tc.name, rc, stderr.String())
			}
			if tc.name == "unexpected_positional" && !strings.Contains(stderr.String(), "unexpected positional arguments") {
				t.Fatalf("expected positional argument error message, got stderr=%q", stderr.String())
			}
		})
	}
}
