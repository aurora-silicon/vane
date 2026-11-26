# Agent and contributor rules: vane

**Rulebook:** follow the Aurora Silicon rulebook, https://github.com/aurora-silicon/agent-guidelines/blob/main/AGENTS.md. It applies to every agent and contributor, whoever runs them. This file adds the repo specifics.

## Repo notes

- Default branch: `main`. Open pull requests against it from `feat/`, `fix/`, `chore/`, or `wip/` branches.
- Reviews and merges: Core and repo code owners (https://github.com/aurora-silicon/agent-guidelines/blob/main/policy/access-tiers.md).
- License: MIT (`LICENSE`).

## Build and test

```sh
nix flake check
nix build .#vane .#vane-runner .#vane-runner-arm64
```

`nixpkgs` is pinned in `flake.lock`; update it deliberately with `nix flake update`, in its own commit.

`nix flake check` runs:

- `lean-tests`: `vane-test`, including the cross-target smoke matrix over `Target.all`.
- `proof-audit`: `vane-audit` loads the compiled `.olean`s and checks that every `Vane.*` constant uses only `propext`, `Classical.choice` and `Quot.sound`, declares no `axiom`, `unsafe`, `implemented_by` or `extern`, and that `Vane.Proofs` still has at least 166 theorems; `scripts/lean-guard.py forbidden` rejects `sorry`, `admit`, `native_decide`, `implemented_by`, `bv_decide`, `unsafe`, `skipKernelTC` and `extern` in Lean code; and `leanchecker Vane Vane.Proofs` replays every declaration through the Lean kernel.
- `tcb-check`: `scripts/lean-guard.py tcb` keeps the import closure of `Vane/Verify.lean` inside `Vane.Binary`, `Vane.TaskStream`, `Vane.Hwx`, `Vane.Register` and `Vane.Reach`.
- `go-lint`: `gofmt`, `go vet`, and no `require` in `go.mod` / no `go.sum`.
- `go-tests`: `go test -race -v ./...` against the built `vane`.

Lean warnings are errors (`moreLeanArgs` in `lakefile.toml`).

Without Nix (Lean 4 + Go 1.22+, Python 3):

```sh
lake build && lake exe vane-test
LEAN_PATH=.lake/build/lib/lean lake exe vane-audit
LEAN_PATH=.lake/build/lib/lean leanchecker Vane Vane.Proofs
python3 scripts/lean-guard.py forbidden . && python3 scripts/lean-guard.py tcb .
gofmt -l . && go vet ./... && go test -race -v ./...
```

Every commit in a pull request needs a `Signed-off-by: Name <email>` trailer (`git commit -s`); the `dco` CI job checks it.

## Repo specifics

- `Vane/Verify.lean` is the independent static safety verifier (TCB); keep it independent from lowering/codegen modules (`Vane/Conv.lean`, `Vane/Graph.lean`, `Vane/Program.lean`).
- `Vane/Proofs.lean` must remain kernel-clean with zero `sorry`, `admit`, `axiom`, `native_decide`, or `implemented_by` shortcuts.
- Every new verifier rejection rule must ship with both a positive test and a dedicated negative tamper test in `Test/Main.lean`.
- Changes touching HWX emission, register lowering, tiling/legality, `Vane/Verify.lean`, or `cmd/vane-runner` require on-device Linux `accel/ane` verification (with clean `dmesg`) at the PR head SHA.
