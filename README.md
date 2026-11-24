# vane

A Lean 4 compiler and safety verifier targeting Neural Engine (ANE) hardware on Linux via the `accel/ane` kernel driver, paired with a zero-dependency Go device runner and reference harness.

## Goals

- **Hardware-verified compilation:** Emit ANE task descriptors and HWX program containers from an explicit, width-checked machine register model verified directly on Linux hardware (default kernel command line, no extra kernel parameters).
- **Safety by construction:** Statically verify bitfield bounds, tiling legality, 2 MiB L2 SRAM footprints, PE anti-hang channel strides, 128-byte LUT table invariants, kernel chunk-slot limits, alignment, and DMA surface extents before emitting binaries.
- **Reproducible on-device testing:** Every supported operator ships with independent float64, Q15.16, RNA16, and 128-byte LUT reference models and a static Linux `arm64` device runner (`vane-runner`).

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
# Compile a single-task fp16 1x1 convolution with optional ReLU or fused 128-byte LUT activation
vane compile-conv1x1 \
  --target h17 \
  --h 16 --w 32 --cin 64 --cout 64 \
  --weights weights.bin \
  --bias bias.bin \
  --act gelu \
  --out-dir out1x1/

# Compile a single-task fp16 dense or depthwise 3x3 convolution (stride 1 or 2, same or valid)
# with optional ReLU or fused 128-byte LUT activation
vane compile-conv3x3 \
  --target h17 \
  --mode dense \
  --h 16 --w 32 --cin 64 --cout 64 \
  --stride 1 --pad same \
  --weights weights.bin \
  --bias bias.bin \
  --act silu \
  --out-dir out3x3/

# Compile a Planar Engine (PE) 1-source unary/scalar/chain or 2-source broadcasted binary op
vane compile-pe \
  --target h17 \
  --op add \
  --h 8 --w 16 --c 4 \
  --z-h 1 --z-w 1 --z-c 4 \
  --out-dir out_pe/

# Compile a standalone Neural Engine activation (relu, copy, or any of 23 mathematically generated LUTs)
vane compile-lut \
  --target h17 \
  --act tanh \
  --h 16 --w 32 --c 8 \
  --out-dir out_lut/

# Validate an HWX container, LOAD sections, and decoded conv / NE activation / PE task legality
vane validate --hwx out3x3/program.hwx --sections out3x3/ --check-tasks

# Derive the 4 host-built LOAD sections from an HWX container
vane sections --hwx out3x3/program.hwx --out-dir sections/

# Emit a minimal demo container
vane emit-demo --target h17 --out-dir demo/ --with-kernel --with-scratch
```

### `vane-runner` (Go Device Runner & Reference Harness)

```bash
# Probe available Linux accel/ane ABI v3 devices
vane-runner probe --json

# Generate deterministic fp16 1x1 or 3x3 conv test vectors (with optional --relu or --act)
vane-runner gen-conv1x1 --h 16 --w 32 --cin 64 --cout 64 --bias --act gelu --seed 1 --out-dir vec1x1/
vane-runner gen-conv3x3 \
  --mode dense --h 16 --w 32 --cin 64 --cout 64 \
  --stride 1 --pad same --bias --act silu --seed 1 --out-dir vec3x3/

# Generate deterministic PE test vectors (1-source or 2-source with optional broadcast / --edge-values)
vane-runner gen-pe --op add --h 8 --w 16 --c 4 --z-h 1 --z-w 1 --z-c 4 --seed 1 --out-dir vec_pe/

# Generate deterministic or exhaustive (--all-fp16) standalone LUT test vectors
vane-runner gen-lut --act tanh --h 16 --w 32 --c 8 --lut-bin out_lut/lut.bin --seed 1 --out-dir vec_lut/

# Execute an HWX program on Linux accel/ane (copy mode or 16 KiB --in-place mmap mode)
vane-runner run \
  --hwx out3x3/program.hwx \
  --sections-dir out3x3/ \
  --input vec3x3/input.bin \
  --output-out actual.bin \
  --runs 3

# Compare strided fp16 outputs using manifest.json geometry (works on conv, PE, and LUT manifests)
vane-runner compare-surface \
  --manifest out3x3/manifest.json \
  --actual actual.bin \
  --expected vec3x3/expected_f64.bin \
  --atol 1e-2 --rtol 1e-2 --json
```

## Status

Milestone 3 (Planar Engine 1-source and 2-source broadcasted elementwise operations, standalone Neural Engine activations with 23 mathematically generated 128-byte LUT tables, fused 1×1 and 3×3 dense/depthwise/stride-2 conv+LUT compilation, unified independent static safety verifier, kernel-clean Lean 4 proofs, and Go reference/runner tooling). See [ROADMAP.md](ROADMAP.md) for upcoming milestones.
