package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTempFile(t *testing.T, root, rel, content string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func TestSelftest(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCLI([]string{"selftest"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("selftest exited %d; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "[PASS] lean-guard selftest:") {
		t.Fatalf("unexpected selftest stdout: %q", stdout.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("unexpected selftest stderr: %q", stderr.String())
	}
}

func TestCLIUsage(t *testing.T) {
	cases := [][]string{
		{},
		{"unknown"},
		{"selftest", "."},
		{"forbidden", ".", "extra"},
		{"tcb", ".", "extra"},
	}
	for _, args := range cases {
		var stdout, stderr bytes.Buffer
		code := runCLI(args, &stdout, &stderr)
		if code != 2 {
			t.Errorf("runCLI(%v) = %d, want 2 (stderr=%q)", args, code, stderr.String())
		}
		if stdout.Len() != 0 {
			t.Errorf("runCLI(%v) unexpected stdout: %q", args, stdout.String())
		}
		if !strings.Contains(stderr.String(), "lean-guard forbidden") {
			t.Errorf("runCLI(%v) stderr missing usage: %q", args, stderr.String())
		}
	}
}

func TestRawStringsFailClosed(t *testing.T) {
	cases := []struct {
		src      string
		wantLine string
	}{
		{"def x := r\"hello\"\n", "raw.lean:1:"},
		{"def x := r#\"hello\" unsafe#\n", "raw.lean:1:"},
		{"def x := r##\"hello\"##\n", "raw.lean:1:"},
		{"def x := s!r#\"hello\"#\n", "raw.lean:1:"},
		{"def x := s!\"{r#\"nested\"#}\"\n", "raw.lean:1:"},
		{"-- ok\n\ndef x := r#\"line3\"#\n", "raw.lean:3:"},
		{"def x := (r\"paren\")\n", "raw.lean:1:"},
	}
	for _, tc := range cases {
		_, err := stripLean(tc.src, "raw.lean")
		if err == nil {
			t.Errorf("stripLean(%q) succeeded, want raw-string error", tc.src)
			continue
		}
		if !strings.Contains(err.Error(), tc.wantLine) {
			t.Errorf("stripLean(%q) error = %q, want substring %q", tc.src, err.Error(), tc.wantLine)
		}
		if !strings.Contains(err.Error(), "raw string literal is forbidden") {
			t.Errorf("stripLean(%q) error = %q, want raw string message", tc.src, err.Error())
		}
	}
}

func TestInterpolationsAndTokens(t *testing.T) {
	badCases := []struct {
		src      string
		expected string
		line     string
	}{
		{"def x := s!\"hello {unsafe foo}\"\n", "unsafe", "bad.lean:1:"},
		{"def x := m!\"hello {skipKernelTC}\"\n", "skipKernelTC", "bad.lean:1:"},
		{"def x := f!\"hello {({ a := sorry } : T)}\"\n", "sorry", "bad.lean:1:"},
		{"def x := s! \"space {admit}\"\n", "admit", "bad.lean:1:"},
		{"def x := s! /- comment -/ \"block {bv_decide}\"\n", "bv_decide", "bad.lean:1:"},
		{"def x := s! -- comment\n\"after {native_decide}\"\n", "native_decide", "bad.lean:2:"},
		{"def x := s!\"nested {s!\"inner {unsafe foo}\"}\"\n", "unsafe", "bad.lean:1:"},
		{"def x := s!\"escaped backslash \\\\{unsafe foo}\"\n", "unsafe", "bad.lean:1:"},
		{"def x := s!\"comment {-- }\nunsafe}\"\n", "unsafe", "bad.lean:2:"},
		{"def x := s!\"block {/- } -/\nimplemented_by}\"\n", "implemented_by", "bad.lean:2:"},
		{"def x := s!\"char {'}' ++ unsafe}\"\n", "unsafe", "bad.lean:1:"},
		{"set_option debug.skipKernelTC true\n", "skipKernelTC", "bad.lean:1:"},
		{"@[extern \"foo\"] opaque f : Unit\n", "extern", "bad.lean:1:"},
		{"def c := '\"' ++ unsafe\n", "unsafe", "bad.lean:1:"},
		{"def f (x' y' : Nat) := sorry\n", "sorry", "bad.lean:1:"},
		{"def x := \"multiline\nstring\"\n\ndef y := sorry\n", "sorry", "bad.lean:4:"},
	}
	for _, tc := range badCases {
		hits, err := scanSource(tc.src, "bad.lean")
		if err != nil {
			t.Errorf("scanSource(%q) unexpected error: %v", tc.src, err)
			continue
		}
		want := tc.line + " forbidden `" + tc.expected + "`"
		found := false
		for _, h := range hits {
			if h == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("scanSource(%q) = %v, want %q", tc.src, hits, want)
		}
	}
}

func TestCleanSourcesAndPrimedIdentifiers(t *testing.T) {
	okCases := []string{
		"def x := \"unsafe sorry extern skipKernelTC r#\\\"\"\n",
		"def x := s!\"unsafe \\{sorry} {if b then \"extern\" else \"admit\"}\"\n",
		"def x := \"\\\\r\"\n",
		"def x := \"r\"\n",
		"-- r#\"comment\"# unsafe\n",
		"/- r\"block\" skipKernelTC -/\n",
		"/- outer /- inner -/ still comment sorry -/\n",
		"def foo' := 'a'\ndef bar := foo' + 'a'\n",
		"def x' := \"sorry\"\n",
		"def c1 := '\\''\ndef c2 := '\\n'\ndef c3 := '\\x41'\ndef c4 := '\\u0041'\ndef c5 := '\"'\ndef ok := \"unsafe\"\n",
		"def my_sorry := 1\ndef unsafely := 2\ndef αsorry := 3\ndef sorry₁ := 4\n",
		"def str\"notraw\" := 1\n",
		"def is! \"notinterp {sorry}\" := 1\n",
	}
	for _, sample := range okCases {
		hits, err := scanSource(sample, "ok.lean")
		if err != nil {
			t.Errorf("scanSource(%q) unexpected error: %v", sample, err)
			continue
		}
		if len(hits) != 0 {
			t.Errorf("scanSource(%q) = %v, want no hits", sample, hits)
		}
	}
}

func TestUnterminatedConstructs(t *testing.T) {
	cases := []struct {
		name     string
		src      string
		wantFile string
		wantMsg  string
	}{
		{
			name:     "unterminated block comment",
			src:      "def a := 1\n/- oops\nstill open\n",
			wantFile: "bad.lean:2:",
			wantMsg:  "unterminated block comment",
		},
		{
			name:     "unterminated nested block comment",
			src:      "/- outer\n/- inner -/\nnever closed",
			wantFile: "bad.lean:1:",
			wantMsg:  "unterminated block comment",
		},
		{
			name:     "unterminated string literal",
			src:      "def a := 1\ndef s := \"hello\nworld",
			wantFile: "bad.lean:2:",
			wantMsg:  "unterminated string literal",
		},
		{
			name:     "unterminated string trailing backslash",
			src:      "def s := \"abc\\",
			wantFile: "bad.lean:1:",
			wantMsg:  "unterminated string literal",
		},
		{
			name:     "unterminated string interpolation",
			src:      "def s := s!\"hi {foo\nbar",
			wantFile: "bad.lean:2:",
			wantMsg:  "unterminated string interpolation",
		},
		{
			name:     "unterminated nested brace interpolation",
			src:      "def s := s!\"hi { { a := 1 }\n",
			wantFile: "bad.lean:2:",
			wantMsg:  "unterminated string interpolation",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := stripLean(tc.src, "bad.lean")
			if err == nil {
				t.Fatalf("stripLean(%q) succeeded, want error %q", tc.src, tc.wantMsg)
			}
			if !strings.Contains(err.Error(), tc.wantFile) || !strings.Contains(err.Error(), tc.wantMsg) {
				t.Fatalf("stripLean(%q) error = %q, want %q and %q", tc.src, err.Error(), tc.wantFile, tc.wantMsg)
			}
		})
	}
}

func TestCmdForbidden(t *testing.T) {
	t.Run("empty root fails closed", func(t *testing.T) {
		dir := t.TempDir()
		var stdout, stderr bytes.Buffer
		code := runCLI([]string{"forbidden", dir}, &stdout, &stderr)
		if code != 1 {
			t.Fatalf("code = %d, want 1", code)
		}
		if !strings.Contains(stderr.String(), "no Lean files found") {
			t.Fatalf("stderr = %q, want no Lean files found", stderr.String())
		}
	})

	t.Run("clean project passes", func(t *testing.T) {
		dir := t.TempDir()
		writeTempFile(t, dir, "Main.lean", "def main : IO Unit := pure ()\n")
		writeTempFile(t, dir, "Vane.lean", "import Vane.Binary\n")
		writeTempFile(t, dir, "Vane/Binary.lean", "-- sorry in comment\ndef x := \"unsafe\"\n")
		writeTempFile(t, dir, "Test/Main.lean", "def t := s!\"ok \\{admit}\"\n")

		var stdout, stderr bytes.Buffer
		code := runCLI([]string{"forbidden", dir}, &stdout, &stderr)
		if code != 0 {
			t.Fatalf("code = %d, want 0; stderr=%q", code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "[PASS] forbidden-scan: 4 Lean files free of") {
			t.Fatalf("stdout = %q", stdout.String())
		}
	})

	t.Run("hits report exact relpath and line", func(t *testing.T) {
		dir := t.TempDir()
		writeTempFile(t, dir, "Vane/Sub/Bad.lean", "def ok := 1\n-- comment\ndef bad := sorry\n")
		writeTempFile(t, dir, "Test/Other.lean", "def f := s!\"x {unsafe 1}\"\n")

		var stdout, stderr bytes.Buffer
		code := runCLI([]string{"forbidden", dir}, &stdout, &stderr)
		if code != 1 {
			t.Fatalf("code = %d, want 1", code)
		}
		errOut := stderr.String()
		if !strings.Contains(errOut, "[FAIL] forbidden-scan: Vane/Sub/Bad.lean:3: forbidden `sorry`") {
			t.Errorf("stderr missing Vane/Sub/Bad.lean hit: %q", errOut)
		}
		if !strings.Contains(errOut, "[FAIL] forbidden-scan: Test/Other.lean:1: forbidden `unsafe`") {
			t.Errorf("stderr missing Test/Other.lean hit: %q", errOut)
		}
	})

	t.Run("lexer error fails closed", func(t *testing.T) {
		dir := t.TempDir()
		writeTempFile(t, dir, "Vane/Raw.lean", "\ndef x := r#\"bad\"#\n")

		var stdout, stderr bytes.Buffer
		code := runCLI([]string{"forbidden", dir}, &stdout, &stderr)
		if code != 1 {
			t.Fatalf("code = %d, want 1", code)
		}
		if !strings.Contains(stderr.String(), "Vane/Raw.lean:2: raw string literal is forbidden") {
			t.Fatalf("stderr = %q", stderr.String())
		}
	})
}

func TestCmdTCB(t *testing.T) {
	t.Run("pass with public/private/meta/import all syntax", func(t *testing.T) {
		dir := t.TempDir()
		writeTempFile(t, dir, "Vane/Verify.lean", strings.Join([]string{
			"-- import Vane.Graph",
			"/- import Vane.Program -/",
			"public import Vane.Binary",
			"private meta import all Vane.Hwx Vane.Register",
			"meta import Vane.TaskStream",
			"import all Vane.Reach",
		}, "\n")+"\n")
		writeTempFile(t, dir, "Vane/Binary.lean", "def b := 1\n")
		writeTempFile(t, dir, "Vane/Hwx.lean", "import Vane.Binary\n")
		writeTempFile(t, dir, "Vane/Register.lean", "import Vane.Binary\n")
		writeTempFile(t, dir, "Vane/TaskStream.lean", "import Vane.Register\n")
		writeTempFile(t, dir, "Vane/Reach.lean", "def r := 1\n")

		var stdout, stderr bytes.Buffer
		code := runCLI([]string{"tcb", dir}, &stdout, &stderr)
		if code != 0 {
			t.Fatalf("code = %d, want 0; stderr=%q", code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "[PASS] tcb-check: Vane.Verify imports Vane.Binary, Vane.Hwx, Vane.Register, Vane.TaskStream, Vane.Reach; import closure (5 modules) stays inside the TCB allowlist") {
			t.Fatalf("unexpected stdout: %q", stdout.String())
		}
	})

	t.Run("empty imports fail closed", func(t *testing.T) {
		dir := t.TempDir()
		writeTempFile(t, dir, "Vane/Verify.lean", "-- no imports here\ndef x := 1\n")

		var stdout, stderr bytes.Buffer
		code := runCLI([]string{"tcb", dir}, &stdout, &stderr)
		if code != 1 {
			t.Fatalf("code = %d, want 1", code)
		}
		if !strings.Contains(stderr.String(), "no imports found in Vane.Verify") {
			t.Fatalf("stderr = %q", stderr.String())
		}
	})

	t.Run("direct out-of-TCB import rejected", func(t *testing.T) {
		dir := t.TempDir()
		writeTempFile(t, dir, "Vane/Verify.lean", "import Vane.Binary\nimport Vane.Graph\n")
		writeTempFile(t, dir, "Vane/Binary.lean", "def b := 1\n")

		var stdout, stderr bytes.Buffer
		code := runCLI([]string{"tcb", dir}, &stdout, &stderr)
		if code != 1 {
			t.Fatalf("code = %d, want 1", code)
		}
		errOut := stderr.String()
		if !strings.Contains(errOut, "Vane.Verify imports Vane.Graph, which is outside the verifier TCB") {
			t.Errorf("stderr = %q", errOut)
		}
		if !strings.Contains(errOut, "allowed modules: Vane.Binary, Vane.Hwx, Vane.Reach, Vane.Register, Vane.TaskStream") {
			t.Errorf("stderr missing allowed modules: %q", errOut)
		}
	})

	t.Run("transitive out-of-TCB import rejected", func(t *testing.T) {
		dir := t.TempDir()
		writeTempFile(t, dir, "Vane/Verify.lean", "import Vane.Hwx\n")
		writeTempFile(t, dir, "Vane/Hwx.lean", "import Vane.Binary\n")
		writeTempFile(t, dir, "Vane/Binary.lean", "public import Vane.Conv\n")

		var stdout, stderr bytes.Buffer
		code := runCLI([]string{"tcb", dir}, &stdout, &stderr)
		if code != 1 {
			t.Fatalf("code = %d, want 1", code)
		}
		if !strings.Contains(stderr.String(), "Vane.Binary imports Vane.Conv, which is outside the verifier TCB") {
			t.Fatalf("stderr = %q", stderr.String())
		}
	})

	t.Run("missing module file fails closed", func(t *testing.T) {
		dir := t.TempDir()
		writeTempFile(t, dir, "Vane/Verify.lean", "import Vane.Binary\n")

		var stdout, stderr bytes.Buffer
		code := runCLI([]string{"tcb", dir}, &stdout, &stderr)
		if code != 1 {
			t.Fatalf("code = %d, want 1", code)
		}
		if !strings.Contains(stderr.String(), "[FAIL] tcb-check:") {
			t.Fatalf("stderr = %q", stderr.String())
		}
	})

	t.Run("missing Vane.Verify fails closed", func(t *testing.T) {
		dir := t.TempDir()
		var stdout, stderr bytes.Buffer
		code := runCLI([]string{"tcb", dir}, &stdout, &stderr)
		if code != 1 {
			t.Fatalf("code = %d, want 1", code)
		}
		if !strings.Contains(stderr.String(), "[FAIL] tcb-check:") {
			t.Fatalf("stderr = %q", stderr.String())
		}
	})

	t.Run("lexer error in TCB module fails closed", func(t *testing.T) {
		dir := t.TempDir()
		writeTempFile(t, dir, "Vane/Verify.lean", "import Vane.Binary\n")
		writeTempFile(t, dir, "Vane/Binary.lean", "/- unterminated\n")

		var stdout, stderr bytes.Buffer
		code := runCLI([]string{"tcb", dir}, &stdout, &stderr)
		if code != 1 {
			t.Fatalf("code = %d, want 1", code)
		}
		if !strings.Contains(stderr.String(), "Vane/Binary.lean:1: unterminated block comment") {
			t.Fatalf("stderr = %q", stderr.String())
		}
	})
}

func TestRepoRootChecks(t *testing.T) {
	root := filepath.Join("..", "..")

	var stdout, stderr bytes.Buffer
	if code := runCLI([]string{"forbidden", root}, &stdout, &stderr); code != 0 {
		t.Fatalf("forbidden on repo root exited %d; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "[PASS] forbidden-scan:") {
		t.Fatalf("unexpected forbidden stdout: %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	if code := runCLI([]string{"tcb", root}, &stdout, &stderr); code != 0 {
		t.Fatalf("tcb on repo root exited %d; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "[PASS] tcb-check:") {
		t.Fatalf("unexpected tcb stdout: %q", stdout.String())
	}
}
