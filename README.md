# vane

A Lean 4 compiler and safety verifier targeting Apple Silicon Neural Engine (ANE) hardware on Linux via the `accel/ane` kernel driver.

## Goals

- **Hardware-verified compilation:** Emit ANE task descriptors and program containers from an explicit, documented machine model verified directly on Linux hardware (default kernel command line, no extra kernel parameters).
- **Safety by construction:** Verify task footprints, L2 capacity, alignment, and cross-task synchronization before execution so invalid or hanging schedules are rejected statically.
- **Reproducible on-device testing:** Every supported operator and optimization ships with an independent reference check and on-device Linux tests.

## Status

Early scaffolding. See [ROADMAP.md](ROADMAP.md) for planned milestones.
