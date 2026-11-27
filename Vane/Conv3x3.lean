import Vane.Conv

namespace Vane.Conv3x3

open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Tiling
open Vane.Verify
open Vane.Conv

export Vane.Conv (ConvActivation stride2TapSlots isNonZeroFp16 maxSparseBodyBytes encodeSparseStream decodeSparseStream)
export Vane.Verify (verifyConv3x3Task verifyConvTask verifyConvHwxTasks)

/-- Complete compilation artifact for a single-task fp16 3x3 convolution (dense or depthwise). -/
structure CompiledConv3x3 where
  target : Target
  shape : Conv3x3Shape
  tiling : Tiling
  sparse : Bool
  wide3x3 : Bool
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

/-- Compute `neCommon` bit 27 (`wide3x3`) for a 3x3 convolution shape and tiling. -/
def computeWide3x3 (target : Target) (sh : Conv3x3Shape) (t : Tiling) : Bool :=
  computeWide3x3Conv target sh.toConvSpec t

/-- Pack fp16 3x3 weights (`(Cout, Cin, 3, 3)` for dense or `(C, 1, 3, 3)` for depthwise)
    and optional bias `(Cout,)` into `__KERN_0` chunk slots. -/
def packConv3x3KernelChunks
    (target : Target)
    (sh : Conv3x3Shape)
    (tiling : Tiling)
    (weights : ByteArray)
    (bias : Option ByteArray)
    (lutPayload? : Option Vane.Lut.LutPayload := none) : Except String PackedKernel :=
  packConvKernelChunks target sh.toConvSpec tiling weights bias lutPayload?

/-- Compile a single-task fp16 3x3 convolution (dense or depthwise, stride 1 or 2,
    same or valid padding) into a statically verified HWX binary and 4 LOAD sections. -/
def compileConv3x3
    (target : Target)
    (shape : Conv3x3Shape)
    (weights : ByteArray)
    (bias : Option ByteArray := none)
    (relu : Bool := false)
    (tilingOverride : Option Tiling := none)
    (act : ConvActivation := .none) : Except String CompiledConv3x3 := do
  let compiled ← compileConv target shape.toConvSpec weights bias relu tilingOverride act
  pure {
    target := compiled.target
    shape := shape
    tiling := compiled.tiling
    sparse := compiled.sparse
    wide3x3 := compiled.wide3x3
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

/-- Compile a single-task quantized/palettized dense 3x3 convolution (stride 1, same or valid padding)
    into a statically verified HWX binary and 4 LOAD sections. -/
def compileConv3x3DenseQuant
    (target : Target)
    (shape : Conv3x3Shape)
    (quant : Vane.Quant.QuantWeightSpec)
    (bias : Option ByteArray := none)
    (relu : Bool := false)
    (tilingOverride : Option Tiling := none)
    (act : ConvActivation := .none) : Except String CompiledConv3x3 := do
  if shape.mode != .dense then
    throw s!"compileConv3x3DenseQuant: quantized 3x3 convolution only supports dense mode, got {shape.mode.name}"
  if shape.stride != 1 then
    throw s!"compileConv3x3DenseQuant: quantized 3x3 convolution only supports stride=1, got stride={shape.stride}"
  let compiled ← compileConvQuant target shape.toConvSpec quant bias relu tilingOverride act
  pure {
    target := compiled.target
    shape := shape
    tiling := compiled.tiling
    sparse := compiled.sparse
    wide3x3 := compiled.wide3x3
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

end Vane.Conv3x3
