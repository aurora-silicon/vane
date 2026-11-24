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
export Vane.Conv (KernelChunkSlot PackedKernel)
export Vane.Verify (isNeConvTask verifyConv1x1Task)

/-- Complete compilation artifact for a single-task fp16 1x1 convolution. -/
structure CompiledConv1x1 where
  target : Target
  shape : Conv1x1Shape
  tiling : Tiling
  hasBias : Bool
  relu : Bool
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
    (bias : Option ByteArray) : Except String PackedKernel :=
  packConvKernelChunks target shape.toConvSpec tiling weights bias

/-- Verify all NE conv tasks decoded from a validated HWX binary. -/
def verifyHwxTasks (hwxBytes : ByteArray) (info : HwxInfo) (requireConvTask : Bool := true) : Except String Nat :=
  verifyConvHwxTasks hwxBytes info (requireConvTask := requireConvTask)

/-- Compile a single-task fp16 1x1 convolution into a statically verified HWX binary
    and its 4 host-built LOAD sections. -/
def compileConv1x1
    (target : Target)
    (shape : Conv1x1Shape)
    (weights : ByteArray)
    (bias : Option ByteArray := none)
    (relu : Bool := false)
    (tilingOverride : Option Tiling := none) : Except String CompiledConv1x1 := do
  let compiled ← compileConv target shape.toConvSpec weights bias relu tilingOverride
  pure {
    target := compiled.target
    shape := shape
    tiling := compiled.tiling
    hasBias := compiled.hasBias
    relu := compiled.relu
    direct := compiled.direct
    l2UseBytes := compiled.l2UseBytes
    kernelBytes := compiled.kernelBytes
    hwx := compiled.hwx
    info := compiled.info
    sections := compiled.sections
  }

end Vane.Conv1x1
