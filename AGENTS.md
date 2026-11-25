# Agent and contributor rules: vane

**Rulebook:** follow the Aurora Silicon rulebook, https://github.com/aurora-silicon/agent-guidelines/blob/main/AGENTS.md. It applies to every agent and contributor, whoever runs them. This file adds the repo specifics.

## Repo notes

- Default branch: `main`. Open pull requests against it from `feat/`, `fix/`, `chore/`, or `wip/` branches.
- Reviews and merges: Core and repo code owners (https://github.com/aurora-silicon/agent-guidelines/blob/main/policy/access-tiers.md).
- License: MIT (`LICENSE`).

## Build and test

```sh
nix flake check --no-write-lock-file
nix build --no-write-lock-file .#vane .#vane-runner .#vane-runner-arm64
```

Without Nix (Lean 4 + Go 1.22+):

```sh
lake build && lake exe vane-test
go test -v ./...
```

## Repo specifics

- `Vane/Verify.lean` is the independent static safety verifier (TCB); keep it independent from lowering/codegen modules (`Vane/Conv.lean`, `Vane/Graph.lean`, `Vane/Program.lean`).
- `Vane/Proofs.lean` must remain kernel-clean with zero `sorry`, `admit`, `axiom`, `native_decide`, or `implemented_by` shortcuts.
- Every new verifier rejection rule must ship with both a positive test and a dedicated negative tamper test in `Test/Main.lean`.
- Changes touching HWX emission, register lowering, tiling/legality, `Vane/Verify.lean`, or `cmd/vane-runner` require on-device Linux `accel/ane` verification (with clean `dmesg`) at the PR head SHA.
