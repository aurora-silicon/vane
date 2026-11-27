import Vane.Conv

namespace Vane.Conv1x1

open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Tiling
open Vane.Program
open Vane.Verify
open Vane.Conv

export Vane.Program (textConstBytes dmaLineBytes)
export Vane.Conv (ConvActivation KernelChunkSlot PackedKernel compileConv1x1Quant compileLinearQuant)
export Vane.Verify (isNeConvTask verifyConv1x1Task)

/-- Complete compilation artifact for a single-task or output-channel-split fp16 1x1 convolution. -/
structure CompiledConv1x1 where
  target : Target
  shape : Conv1x1Shape
  tiling : Tiling
  slices : Array Nat := #[0]
  hasBias : Bool
  relu : Bool
  activation : ConvActivation := .none
  lutBytes : Option ByteArray := none
  direct : Bool
  l2UseBytes : Nat
  kernelBytes : Nat
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections

/-- Pack little-endian fp16 weights `(Cout, Cin)` row-major and optional bias `(Cout,)` into `__KERN_0`. -/
def packKernelChunks
    (target : Target)
    (shape : Conv1x1Shape)
    (tiling : Tiling)
    (weights : ByteArray)
    (bias : Option ByteArray)
    (lutPayload? : Option Vane.Lut.LutPayload := none) : Except String PackedKernel :=
  packConvKernelChunks target shape.toConvSpec tiling weights bias lutPayload?

/-- Verify all recognized tasks decoded from a validated HWX binary. -/
def verifyHwxTasks (hwxBytes : ByteArray) (info : HwxInfo) (requireConvTask : Bool := true) : Except String Nat :=
  verifyConvHwxTasks hwxBytes info (requireConvTask := requireConvTask)

/-- Compile a fp16 1x1 convolution into a statically verified HWX binary and its 4 host-built LOAD sections.
    Automatically splits `Cout` across multiple NE tasks when single-task channel or 64 KiB chunk limits
    are exceeded, while preserving bit-for-bit parity on all single-task shapes. -/
def compileConv1x1
    (target : Target)
    (shape : Conv1x1Shape)
    (weights : ByteArray)
    (bias : Option ByteArray := none)
    (relu : Bool := false)
    (tilingOverride : Option Tiling := none)
    (act : ConvActivation := .none) : Except String CompiledConv1x1 := do
  let compiled ← compileConv1x1Split target shape weights bias relu tilingOverride act
  pure {
    target := compiled.target
    shape := shape
    tiling := compiled.tilings[0]!
    slices := compiled.slices
    hasBias := compiled.hasBias
    relu := compiled.relu
    activation := compiled.activation
    lutBytes := compiled.lutBytes
    direct := compiled.direct
    l2UseBytes := compiled.l2UseBytes
    kernelBytes := compiled.kernelBytes
    hwx := compiled.hwx
    info := compiled.info
    sections := compiled.sections
  }

/-- Compile a `linear` layer `[1, K, B, M] -> [1, N, B, M]` (or 2D `[M, K] -> [M, N]` with `batch = 1`)
    by lowering onto single-task or auto-output-split `conv1x1`. -/
def compileLinear
    (target : Target)
    (m k n : Nat)
    (weights : ByteArray)
    (bias : Option ByteArray := none)
    (batch : Nat := 1)
    (relu : Bool := false)
    (act : ConvActivation := .none) : Except String CompiledConv1x1 :=
  compileConv1x1 target { h := batch, w := m, cin := k, cout := n } weights bias relu none act

end Vane.Conv1x1
