#!/usr/bin/env python3
"""Source-level guards for the Lean code, run by `nix flake check`.

  lean-guard.py forbidden ROOT   reject proof and trust shortcuts in Lean sources
  lean-guard.py tcb ROOT         keep Vane/Verify.lean's import closure inside the TCB

Both commands look at code only: comments (`-- ...`, nested `/- ... -/`) and the
contents of string and character literals are blanked out first.
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
]

# Files the forbidden-token scan skips: the audit tool names the tokens it hunts for.
SCAN_SKIP = {"Test/Audit.lean"}

TCB_ROOT = "Vane.Verify"
TCB_ALLOWLIST = {
    "Vane.Binary",
    "Vane.TaskStream",
    "Vane.Hwx",
    "Vane.Register",
    "Vane.Reach",
}

IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'!?.")


def strip_lean(src: str) -> str:
    """Blank out comments and literal contents, keeping newlines so line numbers hold."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("/-", i):
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
                raise ValueError("unterminated block comment")
            out.append(" ")
        elif src.startswith("--", i):
            while i < n and src[i] != "\n":
                i += 1
        elif c == '"':
            i += 1
            while i < n and src[i] != '"':
                if src[i] == "\\":
                    i += 1
                elif src[i] == "\n":
                    out.append("\n")
                i += 1
            if i >= n:
                raise ValueError("unterminated string literal")
            i += 1
            out.append('""')
        elif c == "'" and (not out or out[-1][-1:] not in IDENT_CHARS):
            m = re.match(r"'(\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)|[^\\'\n])'", src[i:])
            if m:
                i += m.end()
                out.append("' '")
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def lean_files(root: pathlib.Path):
    files = [root / "Main.lean", root / "Vane.lean"]
    for d in ("Vane", "Test"):
        files += sorted((root / d).rglob("*.lean"))
    return [f for f in files if f.is_file()]


def cmd_forbidden(root: pathlib.Path) -> int:
    hits, scanned = [], 0
    for f in lean_files(root):
        rel = f.relative_to(root).as_posix()
        if rel in SCAN_SKIP:
            continue
        scanned += 1
        code = strip_lean(f.read_text(encoding="utf-8"))
        for lineno, line in enumerate(code.splitlines(), 1):
            for name, pat in FORBIDDEN:
                if pat.search(line):
                    hits.append(f"{rel}:{lineno}: forbidden `{name}`")
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
    path = root / (module.replace(".", "/") + ".lean")
    code = strip_lean(path.read_text(encoding="utf-8"))
    mods = []
    for line in code.splitlines():
        m = IMPORT_RE.match(line)
        if m:
            mods += m.group(1).split()
    return mods


def cmd_tcb(root: pathlib.Path) -> int:
    direct = imports_of(root, TCB_ROOT)
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
        stack += [(mod, m) for m in imports_of(root, mod)]
    for e in sorted(errors):
        print(f"[FAIL] tcb-check: {e}", file=sys.stderr)
    if errors:
        print(f"[FAIL] tcb-check: allowed modules: {', '.join(sorted(TCB_ALLOWLIST))}", file=sys.stderr)
        return 1
    print(f"[PASS] tcb-check: {TCB_ROOT} imports {', '.join(direct)}; "
          f"import closure ({len(seen)} modules) stays inside the TCB allowlist")
    return 0


def main(argv) -> int:
    if len(argv) != 3 or argv[1] not in ("forbidden", "tcb"):
        print(__doc__, file=sys.stderr)
        return 2
    root = pathlib.Path(argv[2])
    return cmd_forbidden(root) if argv[1] == "forbidden" else cmd_tcb(root)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
