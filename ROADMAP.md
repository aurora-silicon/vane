# Roadmap

`vane` is built in small vertical slices. Each milestone adds a self-contained capability backed by unit tests, static safety checks, and Linux on-device tests running through `accel/ane` on the default kernel command line.

## Milestone 0 — Project skeleton and device test harness
- Lean 4 / Lake build and Nix flake checks.
- Minimal program container emitter and Linux `accel/ane` test runner (`ioctls`, buffer setup, multi-run repeatability check, reference comparison).
- Independent float64 / PyTorch reference harness for verifying numerical outputs.

## Milestone 1 — Single-task fp16 1×1 convolution
- Explicit machine register schema and bitfield encoder (no implicit omission shortcuts).
- Single-task fp16 pointwise convolution (with optional bias and ReLU) and legal tile chooser.
- Static field-width and basic task-legality checks.
- Hardware verification across representative `(H, W, Cin, Cout)` shapes on H17 (T8140) and H16 (T8132).

## Milestone 2 — Dense and depthwise 3×3 convolution
- Spatial kernel layout, padding, stride, and channel/tile constraints for dense and depthwise 3×3 convolutions.
- Independent legal tiling chooser respecting hardware work-unit, output-group, and L2 footprint bounds.
- Hardware verification across shapes, boundary dimensions, and multi-run determinism checks.

## Milestone 3 — Planar Engine (PE) elementwise ops and activations
- Two-source elementwise ops (`add`, `sub`, `mul`, `max`, `min`, broadcast forms) and one-source linear/scale/bias chains.
- Hardware-verified LUT activation tables and fused activations.
- Hardware verification including edge/special fp16 values.

## Milestone 4 — Multi-task graphs, L2 chaining, and static safety verifier
- Multi-task scheduling with explicit L2/DRAM surface allocation, staging geometry, and synchronization (wait bits, dependency ordering, rings).
- Static safety checker verifying L2 footprint bounds, memory-hazard freedom, and cross-engine ordering on every emitted program.
- Hardware verification on multi-layer chains and residual blocks (including stress/repeatability runs).

## Milestone 5 — End-to-end workloads and broader operator coverage
- End-to-end camera-denoise graph compilation and on-device benchmarking (1080p / 720p / VGA) on the default kernel command line.
- Matmul / linear layers, pooling, views/reshapes, normalization, and softmax.
- Extended SoC coverage (H16, H17, and newer generations as Linux driver support lands).
