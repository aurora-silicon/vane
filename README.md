# vane

A Lean 4 compiler and safety verifier targeting Neural Engine (ANE) hardware on Linux via the `accel/ane` kernel driver, paired with a zero-dependency Go device runner and reference harness.

## Goals

- **Hardware-verified compilation:** Emit ANE task descriptors and HWX program containers from an explicit, width-checked machine register model verified directly on Linux hardware (default kernel command line, no extra kernel parameters).
- **Safety by construction:** Statically verify bitfield bounds, tiling legality, 2 MiB L2 SRAM footprints, kernel chunk-slot invariants, alignment, and DMA surface extents before emitting binaries.
- **Reproducible on-device testing:** Every supported operator ships with independent float64 and Q15.16 reference checks and a static Linux `arm64` device runner (`vane-runner`).

## Building & Testing

### Lean 4 Compiler & Verifier

```bash
lake build
lake exe vane-test
```

### Go Device Runner & Reference Harness

```bash
go test -v ./...
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -o vane-runner ./cmd/vane-runner
```

### Nix Flake

```bash
nix build .#vane
nix build .#vane-runner
nix build .#vane-runner-arm64
nix flake check
```

## CLI Quick Reference

### `vane` (Lean 4 Compiler & Static Verifier)

```bash
# Compile a single-task fp16 1x1 convolution (H16 or H17)
vane compile-conv1x1 \
  --target h17 \
  --h 16 --w 32 --cin 64 --cout 64 \
  --weights weights.bin \
  --bias bias.bin \
  --relu \
  --out-dir out/

# Validate an HWX container, LOAD sections, and decoded task legality
vane validate --hwx out/program.hwx --sections out/ --check-tasks

# Derive the 4 host-built LOAD sections from an HWX container
vane sections --hwx out/program.hwx --out-dir sections/

# Emit a minimal demo container
vane emit-demo --target h17 --out-dir demo/ --with-kernel --with-scratch
```

### `vane-runner` (Go Device Runner & Reference Harness)

```bash
# Probe available Linux accel/ane ABI v3 devices
vane-runner probe --json

# Generate deterministic fp16 1x1 conv inputs, weights, optional bias, and reference outputs
vane-runner gen-conv1x1 --h 16 --w 32 --cin 64 --cout 64 --bias --relu --seed 1 --out-dir vec/

# Execute an HWX program on Linux accel/ane (copy mode or 16 KiB --in-place mmap mode)
vane-runner run \
  --hwx out/program.hwx \
  --sections-dir out/ \
  --input vec/input.bin \
  --output-out actual.bin \
  --runs 3

# Compare strided fp16 outputs using manifest.json geometry (ignoring row padding)
vane-runner compare-conv1x1 \
  --manifest out/manifest.json \
  --actual actual.bin \
  --expected vec/expected_f64.bin \
  --atol 1e-2 --rtol 1e-2 --json
```

## Status

Milestone 1 (single-task fp16 1×1 convolution compiler, explicit register schema, independent tile chooser, static task verifier, and Go tooling). See [ROADMAP.md](ROADMAP.md) for upcoming milestones.
