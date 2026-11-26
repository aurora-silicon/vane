#!/usr/bin/env python3
"""Source-level guards for the Lean code, run by `nix flake check`.

  lean-guard.py forbidden [ROOT]   reject proof and trust shortcuts in Lean sources
  lean-guard.py tcb [ROOT]         keep Vane/Verify.lean's import closure inside the TCB
  lean-guard.py selftest           run built-in lexer and scanner self-tests

Both commands look at code only: comments (`-- ...`, nested `/- ... -/`) and the
literal text of strings and characters are blanked out first, while `{...}`
interpolation expressions inside `s!"..."` / `m!"..."` / `f!"..."` remain scanned
as code and raw string literals (`r"`, `r#"`, ...) fail closed.
"""

import pathlib
import re
import sys

FORBIDDEN = [
    ("sorry", re.compile(r"\bsorry\b")),
    ("admit", re.compile(r"\badmit\b")),
    ("native_decide", re.compile(r"\bnative_decide\b")),
    ("implemented_by", re.compile(r"\bimplemented_by\b")),
    ("bv_decide", re.compile(r"\bbv_decide\b")),
    ("unsafe", re.compile(r"\bunsafe\b")),
    ("skipKernelTC", re.compile(r"\bskipKernelTC\b")),
    ("extern", re.compile(r"\bextern\b")),
]

TCB_ROOT = "Vane.Verify"
TCB_ALLOWLIST = {
    "Vane.Binary",
    "Vane.TaskStream",
    "Vane.Hwx",
    "Vane.Register",
    "Vane.Reach",
}

IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'!?.")
CHAR_LIT_RE = re.compile(r"'(\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)|[^\\'\n])'")
RAW_STR_TAIL_RE = re.compile(r'#*\"')


def _line_of(src: str, pos: int) -> int:
    return src.count("\n", 0, pos) + 1


def _is_word_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch in "_'"


def _is_raw_string_opener(src: str, i: int) -> bool:
    if src[i] != "r":
        return False
    if i > 0 and _is_word_ident_char(src[i - 1]):
        return False
    return RAW_STR_TAIL_RE.match(src, i + 1) is not None


def _is_interp_prefix(src: str, i: int) -> bool:
    n = len(src)
    if i + 1 >= n or src[i] not in ("s", "m", "f") or src[i + 1] != "!":
        return False
    if i > 0 and _is_word_ident_char(src[i - 1]):
        return False
    if i + 2 >= n:
        return False
    nxt = src[i + 2]
    return (
        nxt == '"'
        or nxt.isspace()
        or src.startswith("/-", i + 2)
        or src.startswith("--", i + 2)
    )


def _scan_string(src: str, i: int, out: list[str], filename: str, interpolated: bool) -> int:
    n = len(src)
    start = i
    i += 1  # consume opening '"'
    out.append('"')
    while i < n:
        c = src[i]
        if c == '"':
            out.append('"')
            return i + 1
        if c == "\\":
            if i + 1 >= n:
                raise ValueError(f"{filename}:{_line_of(src, start)}: unterminated string literal")
            if src[i + 1] == "\n":
                out.append("\n")
            i += 2
        elif interpolated and c == "{":
            out.append('" { ')
            i = _scan_code(src, i + 1, out, filename, stop_on_rbrace=True)
            out.append(' } "')
        else:
            if c == "\n":
                out.append("\n")
            i += 1
    raise ValueError(f"{filename}:{_line_of(src, start)}: unterminated string literal")


def _scan_code(src: str, i: int, out: list[str], filename: str, stop_on_rbrace: bool) -> int:
    n = len(src)
    brace_depth = 0
    pending_interp = False
    while i < n:
        c = src[i]
        if stop_on_rbrace:
            if c == "{":
                brace_depth += 1
                pending_interp = False
                out.append(c)
                i += 1
                continue
            if c == "}":
                if brace_depth == 0:
                    return i + 1
                brace_depth -= 1
                pending_interp = False
                out.append(c)
                i += 1
                continue

        if src.startswith("/-", i):
            start = i
            depth, i = 1, i + 2
            while i < n and depth:
                if src.startswith("/-", i):
                    depth, i = depth + 1, i + 2
                elif src.startswith("-/", i):
                    depth, i = depth - 1, i + 2
                else:
                    if src[i] == "\n":
                        out.append("\n")
                    i += 1
            if depth:
                raise ValueError(f"{filename}:{_line_of(src, start)}: unterminated block comment")
            out.append(" ")
        elif src.startswith("--", i):
            while i < n and src[i] != "\n":
                i += 1
        elif c.isspace():
            out.append(c)
            i += 1
        elif _is_raw_string_opener(src, i):
            raise ValueError(f"{filename}:{_line_of(src, i)}: raw string literal is forbidden by lean-guard")
        elif _is_interp_prefix(src, i):
            out.append(src[i : i + 2])
            i += 2
            pending_interp = True
        elif c == '"':
            i = _scan_string(src, i, out, filename, interpolated=pending_interp)
            pending_interp = False
        elif c == "'" and (not out or out[-1][-1:] not in IDENT_CHARS):
            pending_interp = False
            m = CHAR_LIT_RE.match(src, i)
            if m:
                i = m.end()
                out.append("' '")
            else:
                out.append(c)
                i += 1
        else:
            pending_interp = False
            out.append(c)
            i += 1

    if stop_on_rbrace:
        raise ValueError(f"{filename}:{_line_of(src, n)}: unterminated string interpolation")
    return i


def strip_lean(src: str, filename: str = "<input>") -> str:
    """Blank out comments and string/char literal text while scanning `{...}`
    interpolations as code and rejecting raw string literals fail-closed."""
    out: list[str] = []
    _scan_code(src, 0, out, filename, stop_on_rbrace=False)
    return "".join(out)


def scan_source(src: str, filename: str = "<input>") -> list[str]:
    code = strip_lean(src, filename)
    hits = []
    for lineno, line in enumerate(code.splitlines(), 1):
        for name, pat in FORBIDDEN:
            if pat.search(line):
                hits.append(f"{filename}:{lineno}: forbidden `{name}`")
    return hits


def lean_files(root: pathlib.Path):
    files = [root / "Main.lean", root / "Vane.lean"]
    for d in ("Vane", "Test"):
        files += sorted((root / d).rglob("*.lean"))
    return [f for f in files if f.is_file()]


def cmd_forbidden(root: pathlib.Path) -> int:
    hits, scanned = [], 0
    for f in lean_files(root):
        rel = f.relative_to(root).as_posix()
        scanned += 1
        try:
            hits.extend(scan_source(f.read_text(encoding="utf-8"), rel))
        except ValueError as err:
            print(f"[FAIL] forbidden-scan: {err}", file=sys.stderr)
            return 1
    if scanned == 0:
        print("[FAIL] forbidden-scan: no Lean files found", file=sys.stderr)
        return 1
    for h in hits:
        print(f"[FAIL] forbidden-scan: {h}", file=sys.stderr)
    if hits:
        return 1
    tokens = ", ".join(name for name, _ in FORBIDDEN)
    print(f"[PASS] forbidden-scan: {scanned} Lean files free of {tokens}")
    return 0


IMPORT_RE = re.compile(r"^\s*(?:public\s+|private\s+)?(?:meta\s+)?import\s+(?:all\s+)?(.*)$")


def imports_of(root: pathlib.Path, module: str):
    rel = module.replace(".", "/") + ".lean"
    path = root / rel
    code = strip_lean(path.read_text(encoding="utf-8"), rel)
    mods = []
    for line in code.splitlines():
        m = IMPORT_RE.match(line)
        if m:
            mods += m.group(1).split()
    return mods


def cmd_tcb(root: pathlib.Path) -> int:
    try:
        direct = imports_of(root, TCB_ROOT)
    except ValueError as err:
        print(f"[FAIL] tcb-check: {err}", file=sys.stderr)
        return 1
    if not direct:
        print(f"[FAIL] tcb-check: no imports found in {TCB_ROOT}; refusing to pass an empty check",
              file=sys.stderr)
        return 1
    errors, seen = [], set()
    stack = [(TCB_ROOT, m) for m in direct]
    while stack:
        parent, mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        if mod not in TCB_ALLOWLIST:
            errors.append(f"{parent} imports {mod}, which is outside the verifier TCB")
            continue
        try:
            sub_imports = imports_of(root, mod)
        except ValueError as err:
            print(f"[FAIL] tcb-check: {err}", file=sys.stderr)
            return 1
        stack += [(mod, m) for m in sub_imports]
    for e in sorted(errors):
        print(f"[FAIL] tcb-check: {e}", file=sys.stderr)
    if errors:
        print(f"[FAIL] tcb-check: allowed modules: {', '.join(sorted(TCB_ALLOWLIST))}", file=sys.stderr)
        return 1
    print(f"[PASS] tcb-check: {TCB_ROOT} imports {', '.join(direct)}; "
          f"import closure ({len(seen)} modules) stays inside the TCB allowlist")
    return 0


def cmd_selftest() -> int:
    # Raw strings must fail closed with filename:line.
    for sample in (
        'def x := r"hello"\n',
        'def x := r#"hello" unsafe#\n',
        'def x := r##"hello"##\n',
        'def x := s!r#"hello"#\n',
        'def x := s!"{r#"nested"#}"\n',
    ):
        try:
            strip_lean(sample, "raw.lean")
        except ValueError as err:
            assert "raw.lean:1:" in str(err), f"expected file:line in error, got: {err}"
        else:
            raise AssertionError(f"expected raw string failure for {sample!r}")

    # Interpolation code inside s!/m!/f! must be scanned, including nested braces/strings.
    for sample, expected in (
        ('def x := s!"hello {unsafe foo}"\n', "unsafe"),
        ('def x := m!"hello {skipKernelTC}"\n', "skipKernelTC"),
        ('def x := f!"hello {({ a := sorry } : T)}"\n', "sorry"),
        ('def x := s! "space {admit}"\n', "admit"),
        ('def x := s!"nested {s!"inner {unsafe foo}"}"\n', "unsafe"),
        ('def x := s!"escaped backslash \\\\{unsafe foo}"\n', "unsafe"),
        ('def x := s!"comment {-- }\nunsafe}"\n', "unsafe"),
        ('def x := s!"char {\'}\' ++ unsafe}"\n', "unsafe"),
        ('set_option debug.skipKernelTC true\n', "skipKernelTC"),
        ('@[extern "foo"] opaque f : Unit\n', "extern"),
    ):
        hits = scan_source(sample, "bad.lean")
        assert any(f"`{expected}`" in h for h in hits), f"expected `{expected}` hit for {sample!r}, got {hits}"

    # Literal string text and escaped braces must remain clean.
    for sample in (
        'def x := "unsafe sorry extern skipKernelTC r#\\""\n',
        'def x := s!"unsafe \\{sorry} {if b then "extern" else "admit"}"\n',
        'def x := "\\\\r"\n',
        'def x := "r"\n',
        '-- r#"comment"# unsafe\n',
        '/- r"block" skipKernelTC -/\n',
    ):
        hits = scan_source(sample, "ok.lean")
        assert not hits, f"unexpected hits for {sample!r}: {hits}"

    print("[PASS] lean-guard selftest: raw-string fail-closed, interpolation scanning, and forbidden tokens verified")
    return 0


def main(argv) -> int:
    if len(argv) == 2 and argv[1] == "selftest":
        return cmd_selftest()
    if len(argv) not in (2, 3) or argv[1] not in ("forbidden", "tcb"):
        print(__doc__, file=sys.stderr)
        return 2
    root = pathlib.Path(argv[2] if len(argv) == 3 else ".")
    return cmd_forbidden(root) if argv[1] == "forbidden" else cmd_tcb(root)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
