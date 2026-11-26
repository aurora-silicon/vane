// Command lean-guard provides source-level guards for the Lean 4 codebase:
//
//	lean-guard forbidden [ROOT]   reject proof and trust shortcuts in Lean sources
//	lean-guard tcb [ROOT]         keep Vane/Verify.lean's import closure inside the TCB
//	lean-guard selftest           run built-in lexer and scanner self-tests
//
// Both checks inspect code only: comments (-- ..., nested /- ... -/) and the
// literal text of strings and characters are blanked out first, while {...}
// interpolation expressions inside s!"..." / m!"..." / f!"..." remain scanned
// as code and raw string literals (r", r#", ...) fail closed.
package main

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

const usageText = `Source-level guards for the Lean code, run by ` + "`nix flake check`" + `.

  lean-guard forbidden [ROOT]   reject proof and trust shortcuts in Lean sources
  lean-guard tcb [ROOT]         keep Vane/Verify.lean's import closure inside the TCB
  lean-guard selftest           run built-in lexer and scanner self-tests

Both commands look at code only: comments (` + "`-- ...`" + `, nested ` + "`/- ... -/`" + `) and the
literal text of strings and characters are blanked out first, while ` + "`{...}`" + `
interpolation expressions inside ` + "`s!\"...\"`" + ` / ` + "`m!\"...\"`" + ` / ` + "`f!\"...\"`" + ` remain scanned
as code and raw string literals (` + "`r\"`" + `, ` + "`r#\"`" + `, ...) fail closed.
`

var forbiddenTokens = []string{
	"sorry",
	"admit",
	"native_decide",
	"implemented_by",
	"bv_decide",
	"unsafe",
	"skipKernelTC",
	"extern",
}

const tcbRoot = "Vane.Verify"

var tcbAllowlist = map[string]bool{
	"Vane.Binary":     true,
	"Vane.TaskStream": true,
	"Vane.Hwx":        true,
	"Vane.Register":   true,
	"Vane.Reach":      true,
}

var importRe = regexp.MustCompile(`^\s*(?:public\s+|private\s+)?(?:meta\s+)?import\s+(?:all\s+)?(.*)$`)

func lineOf(runes []rune, pos int) int {
	line := 1
	for i := 0; i < pos && i < len(runes); i++ {
		if runes[i] == '\n' {
			line++
		}
	}
	return line
}

func isWordChar(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsNumber(r) || r == '_'
}

func isWordIdentChar(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsNumber(r) || r == '_' || r == '\''
}

func isIdentChar(r rune) bool {
	return (r >= 'a' && r <= 'z') ||
		(r >= 'A' && r <= 'Z') ||
		(r >= '0' && r <= '9') ||
		r == '_' || r == '\'' || r == '!' || r == '?' || r == '.'
}

func isHexDigit(r rune) bool {
	return (r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')
}

func hasPrefixAt(runes []rune, i int, a, b rune) bool {
	return i+1 < len(runes) && runes[i] == a && runes[i+1] == b
}

func isRawStringOpener(runes []rune, i int) bool {
	if runes[i] != 'r' {
		return false
	}
	if i > 0 && isWordIdentChar(runes[i-1]) {
		return false
	}
	j := i + 1
	for j < len(runes) && runes[j] == '#' {
		j++
	}
	return j < len(runes) && runes[j] == '"'
}

func isInterpPrefix(runes []rune, i int) bool {
	n := len(runes)
	if i+1 >= n || (runes[i] != 's' && runes[i] != 'm' && runes[i] != 'f') || runes[i+1] != '!' {
		return false
	}
	if i > 0 && isWordIdentChar(runes[i-1]) {
		return false
	}
	if i+2 >= n {
		return false
	}
	nxt := runes[i+2]
	return nxt == '"' ||
		unicode.IsSpace(nxt) ||
		hasPrefixAt(runes, i+2, '/', '-') ||
		hasPrefixAt(runes, i+2, '-', '-')
}

// matchCharLit matches a Lean character literal starting at a single quote:
// regular char ('x'), simple escape ('\n'), hex escape ('\x41'), or unicode escape ('\u0041').
func matchCharLit(runes []rune, i int) (int, bool) {
	n := len(runes)
	if i >= n || runes[i] != '\'' {
		return 0, false
	}
	if i+1 < n && runes[i+1] == '\\' {
		if i+5 < n && runes[i+2] == 'x' && isHexDigit(runes[i+3]) && isHexDigit(runes[i+4]) && runes[i+5] == '\'' {
			return i + 6, true
		}
		if i+7 < n && runes[i+2] == 'u' &&
			isHexDigit(runes[i+3]) && isHexDigit(runes[i+4]) &&
			isHexDigit(runes[i+5]) && isHexDigit(runes[i+6]) &&
			runes[i+7] == '\'' {
			return i + 8, true
		}
		if i+3 < n && runes[i+2] != '\n' && runes[i+3] == '\'' {
			return i + 4, true
		}
		return 0, false
	}
	if i+2 < n && runes[i+1] != '\\' && runes[i+1] != '\'' && runes[i+1] != '\n' && runes[i+2] == '\'' {
		return i + 3, true
	}
	return 0, false
}

func scanString(runes []rune, i int, out *[]rune, filename string, interpolated bool) (int, error) {
	n := len(runes)
	start := i
	i++ // consume opening '"'
	*out = append(*out, '"')
	for i < n {
		c := runes[i]
		if c == '"' {
			*out = append(*out, '"')
			return i + 1, nil
		}
		if c == '\\' {
			if i+1 >= n {
				return 0, fmt.Errorf("%s:%d: unterminated string literal", filename, lineOf(runes, start))
			}
			if runes[i+1] == '\n' {
				*out = append(*out, '\n')
			}
			i += 2
		} else if interpolated && c == '{' {
			*out = append(*out, '"', ' ', '{', ' ')
			var err error
			i, err = scanCode(runes, i+1, out, filename, true)
			if err != nil {
				return 0, err
			}
			*out = append(*out, ' ', '}', ' ', '"')
		} else {
			if c == '\n' {
				*out = append(*out, '\n')
			}
			i++
		}
	}
	return 0, fmt.Errorf("%s:%d: unterminated string literal", filename, lineOf(runes, start))
}

func scanCode(runes []rune, i int, out *[]rune, filename string, stopOnRbrace bool) (int, error) {
	n := len(runes)
	braceDepth := 0
	pendingInterp := false
	for i < n {
		c := runes[i]
		if stopOnRbrace {
			if c == '{' {
				braceDepth++
				pendingInterp = false
				*out = append(*out, c)
				i++
				continue
			}
			if c == '}' {
				if braceDepth == 0 {
					return i + 1, nil
				}
				braceDepth--
				pendingInterp = false
				*out = append(*out, c)
				i++
				continue
			}
		}

		if hasPrefixAt(runes, i, '/', '-') {
			start := i
			depth := 1
			i += 2
			for i < n && depth > 0 {
				if hasPrefixAt(runes, i, '/', '-') {
					depth++
					i += 2
				} else if hasPrefixAt(runes, i, '-', '/') {
					depth--
					i += 2
				} else {
					if runes[i] == '\n' {
						*out = append(*out, '\n')
					}
					i++
				}
			}
			if depth > 0 {
				return 0, fmt.Errorf("%s:%d: unterminated block comment", filename, lineOf(runes, start))
			}
			*out = append(*out, ' ')
		} else if hasPrefixAt(runes, i, '-', '-') {
			for i < n && runes[i] != '\n' {
				i++
			}
		} else if unicode.IsSpace(c) {
			*out = append(*out, c)
			i++
		} else if isRawStringOpener(runes, i) {
			return 0, fmt.Errorf("%s:%d: raw string literal is forbidden by lean-guard", filename, lineOf(runes, i))
		} else if isInterpPrefix(runes, i) {
			*out = append(*out, runes[i], runes[i+1])
			i += 2
			pendingInterp = true
		} else if c == '"' {
			var err error
			i, err = scanString(runes, i, out, filename, pendingInterp)
			if err != nil {
				return 0, err
			}
			pendingInterp = false
		} else if c == '\'' && (len(*out) == 0 || !isIdentChar((*out)[len(*out)-1])) {
			pendingInterp = false
			if nextIdx, ok := matchCharLit(runes, i); ok {
				i = nextIdx
				*out = append(*out, '\'', ' ', '\'')
			} else {
				*out = append(*out, c)
				i++
			}
		} else {
			pendingInterp = false
			*out = append(*out, c)
			i++
		}
	}

	if stopOnRbrace {
		return 0, fmt.Errorf("%s:%d: unterminated string interpolation", filename, lineOf(runes, n))
	}
	return i, nil
}

// stripLean blanks out comments and string/char literal text while scanning
// {...} interpolations as code and rejecting raw string literals fail-closed.
func stripLean(src, filename string) (string, error) {
	runes := []rune(src)
	out := make([]rune, 0, len(runes))
	if _, err := scanCode(runes, 0, &out, filename, false); err != nil {
		return "", err
	}
	return string(out), nil
}

func containsWord(line, word string) bool {
	offset := 0
	for {
		idx := strings.Index(line[offset:], word)
		if idx < 0 {
			return false
		}
		start := offset + idx
		end := start + len(word)
		leftOK := true
		if start > 0 {
			prev, _ := utf8.DecodeLastRuneInString(line[:start])
			leftOK = !isWordChar(prev)
		}
		rightOK := true
		if end < len(line) {
			next, _ := utf8.DecodeRuneInString(line[end:])
			rightOK = !isWordChar(next)
		}
		if leftOK && rightOK {
			return true
		}
		offset = start + 1
	}
}

func scanSource(src, filename string) ([]string, error) {
	code, err := stripLean(src, filename)
	if err != nil {
		return nil, err
	}
	var hits []string
	lines := strings.Split(code, "\n")
	for i, line := range lines {
		lineno := i + 1
		for _, tok := range forbiddenTokens {
			if containsWord(line, tok) {
				hits = append(hits, fmt.Sprintf("%s:%d: forbidden `%s`", filename, lineno, tok))
			}
		}
	}
	return hits, nil
}

func isRegularFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular()
}

func leanFiles(root string) ([]string, error) {
	var files []string
	for _, top := range []string{"Main.lean", "Vane.lean"} {
		p := filepath.Join(root, top)
		if isRegularFile(p) {
			files = append(files, p)
		}
	}
	for _, dirName := range []string{"Vane", "Test"} {
		dirPath := filepath.Join(root, dirName)
		if _, err := os.Stat(dirPath); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, err
		}
		var dirFiles []string
		err := filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if !d.IsDir() && strings.HasSuffix(d.Name(), ".lean") && isRegularFile(path) {
				dirFiles = append(dirFiles, path)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
		sort.Slice(dirFiles, func(i, j int) bool {
			return filepath.ToSlash(dirFiles[i]) < filepath.ToSlash(dirFiles[j])
		})
		files = append(files, dirFiles...)
	}
	return files, nil
}

func readUTF8File(path, rel string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if !utf8.Valid(data) {
		return "", fmt.Errorf("%s: invalid UTF-8", rel)
	}
	return string(data), nil
}

func cmdForbidden(root string, stdout, stderr io.Writer) int {
	files, err := leanFiles(root)
	if err != nil {
		fmt.Fprintf(stderr, "[FAIL] forbidden-scan: %v\n", err)
		return 1
	}
	var hits []string
	scanned := 0
	for _, f := range files {
		rel, err := filepath.Rel(root, f)
		if err != nil {
			fmt.Fprintf(stderr, "[FAIL] forbidden-scan: %v\n", err)
			return 1
		}
		relPosix := filepath.ToSlash(rel)
		scanned++
		src, err := readUTF8File(f, relPosix)
		if err != nil {
			fmt.Fprintf(stderr, "[FAIL] forbidden-scan: %v\n", err)
			return 1
		}
		fileHits, err := scanSource(src, relPosix)
		if err != nil {
			fmt.Fprintf(stderr, "[FAIL] forbidden-scan: %v\n", err)
			return 1
		}
		hits = append(hits, fileHits...)
	}
	if scanned == 0 {
		fmt.Fprintln(stderr, "[FAIL] forbidden-scan: no Lean files found")
		return 1
	}
	for _, h := range hits {
		fmt.Fprintf(stderr, "[FAIL] forbidden-scan: %s\n", h)
	}
	if len(hits) > 0 {
		return 1
	}
	fmt.Fprintf(stdout, "[PASS] forbidden-scan: %d Lean files free of %s\n", scanned, strings.Join(forbiddenTokens, ", "))
	return 0
}

func importsOf(root, module string) ([]string, error) {
	rel := strings.ReplaceAll(module, ".", "/") + ".lean"
	path := filepath.Join(root, filepath.FromSlash(rel))
	src, err := readUTF8File(path, rel)
	if err != nil {
		return nil, err
	}
	code, err := stripLean(src, rel)
	if err != nil {
		return nil, err
	}
	var mods []string
	for _, line := range strings.Split(code, "\n") {
		line = strings.TrimSuffix(line, "\r")
		m := importRe.FindStringSubmatch(line)
		if m != nil {
			mods = append(mods, strings.Fields(m[1])...)
		}
	}
	return mods, nil
}

func allowedModulesSorted() []string {
	allowed := make([]string, 0, len(tcbAllowlist))
	for m := range tcbAllowlist {
		allowed = append(allowed, m)
	}
	sort.Strings(allowed)
	return allowed
}

func cmdTCB(root string, stdout, stderr io.Writer) int {
	direct, err := importsOf(root, tcbRoot)
	if err != nil {
		fmt.Fprintf(stderr, "[FAIL] tcb-check: %v\n", err)
		return 1
	}
	if len(direct) == 0 {
		fmt.Fprintf(stderr, "[FAIL] tcb-check: no imports found in %s; refusing to pass an empty check\n", tcbRoot)
		return 1
	}

	type edge struct {
		parent string
		mod    string
	}
	var errs []string
	seen := make(map[string]bool)
	stack := make([]edge, 0, len(direct))
	for _, m := range direct {
		stack = append(stack, edge{parent: tcbRoot, mod: m})
	}
	for len(stack) > 0 {
		e := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[e.mod] {
			continue
		}
		seen[e.mod] = true
		if !tcbAllowlist[e.mod] {
			errs = append(errs, fmt.Sprintf("%s imports %s, which is outside the verifier TCB", e.parent, e.mod))
			continue
		}
		subImports, err := importsOf(root, e.mod)
		if err != nil {
			fmt.Fprintf(stderr, "[FAIL] tcb-check: %v\n", err)
			return 1
		}
		for _, m := range subImports {
			stack = append(stack, edge{parent: e.mod, mod: m})
		}
	}

	sort.Strings(errs)
	for _, e := range errs {
		fmt.Fprintf(stderr, "[FAIL] tcb-check: %s\n", e)
	}
	if len(errs) > 0 {
		fmt.Fprintf(stderr, "[FAIL] tcb-check: allowed modules: %s\n", strings.Join(allowedModulesSorted(), ", "))
		return 1
	}
	fmt.Fprintf(stdout, "[PASS] tcb-check: %s imports %s; import closure (%d modules) stays inside the TCB allowlist\n",
		tcbRoot, strings.Join(direct, ", "), len(seen))
	return 0
}

func cmdSelftest(stdout, stderr io.Writer) int {
	// Raw strings must fail closed with filename:line.
	rawSamples := []string{
		"def x := r\"hello\"\n",
		"def x := r#\"hello\" unsafe#\n",
		"def x := r##\"hello\"##\n",
		"def x := s!r#\"hello\"#\n",
		"def x := s!\"{r#\"nested\"#}\"\n",
	}
	for _, sample := range rawSamples {
		_, err := stripLean(sample, "raw.lean")
		if err == nil {
			fmt.Fprintf(stderr, "[FAIL] lean-guard selftest: expected raw string failure for %q\n", sample)
			return 1
		}
		if !strings.Contains(err.Error(), "raw.lean:1:") {
			fmt.Fprintf(stderr, "[FAIL] lean-guard selftest: expected file:line in error, got: %v\n", err)
			return 1
		}
	}

	// Interpolation code inside s!/m!/f! must be scanned, including nested braces/strings.
	badSamples := []struct {
		src      string
		expected string
	}{
		{"def x := s!\"hello {unsafe foo}\"\n", "unsafe"},
		{"def x := m!\"hello {skipKernelTC}\"\n", "skipKernelTC"},
		{"def x := f!\"hello {({ a := sorry } : T)}\"\n", "sorry"},
		{"def x := s! \"space {admit}\"\n", "admit"},
		{"def x := s!\"nested {s!\"inner {unsafe foo}\"}\"\n", "unsafe"},
		{"def x := s!\"escaped backslash \\\\{unsafe foo}\"\n", "unsafe"},
		{"def x := s!\"comment {-- }\nunsafe}\"\n", "unsafe"},
		{"def x := s!\"char {'}' ++ unsafe}\"\n", "unsafe"},
		{"set_option debug.skipKernelTC true\n", "skipKernelTC"},
		{"@[extern \"foo\"] opaque f : Unit\n", "extern"},
	}
	for _, tc := range badSamples {
		hits, err := scanSource(tc.src, "bad.lean")
		if err != nil {
			fmt.Fprintf(stderr, "[FAIL] lean-guard selftest: unexpected error for %q: %v\n", tc.src, err)
			return 1
		}
		needle := "`" + tc.expected + "`"
		found := false
		for _, h := range hits {
			if strings.Contains(h, needle) {
				found = true
				break
			}
		}
		if !found {
			fmt.Fprintf(stderr, "[FAIL] lean-guard selftest: expected `%s` hit for %q, got %v\n", tc.expected, tc.src, hits)
			return 1
		}
	}

	// Literal string text and escaped braces must remain clean.
	okSamples := []string{
		"def x := \"unsafe sorry extern skipKernelTC r#\\\"\"\n",
		"def x := s!\"unsafe \\{sorry} {if b then \"extern\" else \"admit\"}\"\n",
		"def x := \"\\\\r\"\n",
		"def x := \"r\"\n",
		"-- r#\"comment\"# unsafe\n",
		"/- r\"block\" skipKernelTC -/\n",
	}
	for _, sample := range okSamples {
		hits, err := scanSource(sample, "ok.lean")
		if err != nil {
			fmt.Fprintf(stderr, "[FAIL] lean-guard selftest: unexpected error for %q: %v\n", sample, err)
			return 1
		}
		if len(hits) > 0 {
			fmt.Fprintf(stderr, "[FAIL] lean-guard selftest: unexpected hits for %q: %v\n", sample, hits)
			return 1
		}
	}

	fmt.Fprintln(stdout, "[PASS] lean-guard selftest: raw-string fail-closed, interpolation scanning, and forbidden tokens verified")
	return 0
}

func runCLI(args []string, stdout, stderr io.Writer) int {
	if len(args) == 1 && args[0] == "selftest" {
		return cmdSelftest(stdout, stderr)
	}
	if (len(args) != 1 && len(args) != 2) || (args[0] != "forbidden" && args[0] != "tcb") {
		fmt.Fprint(stderr, usageText)
		return 2
	}
	root := "."
	if len(args) == 2 {
		root = args[1]
	}
	if args[0] == "forbidden" {
		return cmdForbidden(root, stdout, stderr)
	}
	return cmdTCB(root, stdout, stderr)
}

func main() {
	os.Exit(runCLI(os.Args[1:], os.Stdout, os.Stderr))
}
