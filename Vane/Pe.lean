import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register
import Vane.Tiling
import Vane.Program
import Vane.Verify

namespace Vane.Pe

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Tiling
open Vane.Program
open Vane.Verify

/-- Single-batch 3D tensor shape `(C, H, W)` for Planar Engine operations. -/
structure TensorDims where
  c : Nat
  h : Nat
  w : Nat
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Two-source runtime Planar Engine binary opcode. -/
inductive PeBinOp where
  | add
  | sub
  | mul
  | max
  | min
  deriving Repr, BEq, DecidableEq, Inhabited

namespace PeBinOp

def name : PeBinOp → String
  | .add => "add"
  | .sub => "sub"
  | .mul => "mul"
  | .max => "max"
  | .min => "min"

def regValue : PeBinOp → UInt32
  | .add => peOpAddValue
  | .sub => peOpSubValue
  | .mul => peOpMulValue
  | .max => peOpMaxValue
  | .min => peOpMinValue

def parse? : String → Option PeBinOp
  | "add" => some .add
  | "sub" => some .sub
  | "mul" => some .mul
  | "max" => some .max
  | "min" => some .min
  | _ => none

end PeBinOp

/-- Planar Engine operation specification.
    For `binary op src1Dims src2Dims rev`:
    - Surface 0 (`x`, BAR slot 6) has shape `src1Dims`.
    - Surface 1 (`z`, BAR slot 7) has shape `src2Dims`.
    - When `rev = false`, hardware `src1` reads slot 6 (`x`) and `src2` reads slot 7 (`z`): `x op z`.
    - When `rev = true`, hardware `src1` reads slot 7 (`z`) and `src2` reads slot 6 (`x`): `z op x`. -/
inductive PeOp where
  | abs
  | square
  | chain (inScale bias scale : Option UInt16)
  | binary (op : PeBinOp) (src1Dims src2Dims : TensorDims) (rev : Bool := false)
  deriving Repr, BEq, DecidableEq, Inhabited

namespace PeOp

/-- `alpha * x + beta` via 1-source PE chain (`in_scale = alpha`, `bias = beta`). -/
def linearActivation (alpha beta : UInt16) : PeOp :=
  .chain (some alpha) (some beta) none

/-- `x + c` via 1-source PE chain (`bias = c`). -/
def scalarAdd (c : UInt16) : PeOp :=
  .chain none (some c) none

/-- `x - c` via 1-source PE chain (`bias = -c`). -/
def scalarSub (c : UInt16) : PeOp :=
  .chain none (some (negF16CanonicalZero c)) none

/-- `c - x` via 1-source PE chain (`in_scale = -1.0`, `bias = c`). -/
def scalarRSub (c : UInt16) : PeOp :=
  .chain (some f16NegOne) (some c) none

/-- `x * c` via 1-source PE chain (`scale = c`). -/
def scalarMul (c : UInt16) : PeOp :=
  .chain none none (some c)

/-- `x / c` via 1-source PE chain (`scale = fp16(1.0 / c)`), rejecting zero or NaN divisors. -/
def scalarDiv (c : UInt16) : Except String PeOp := do
  if (c &&& 0x7fff) == 0 then
    throw "PE scalarDiv: divisor must be non-zero"
  if f16IsNaN c then
    throw "PE scalarDiv: divisor must not be NaN"
  let inv := floatToF16CanonicalZero (1.0 / f16ToFloat c)
  pure (.chain none none (some inv))

def isTwoSource : PeOp → Bool
  | .binary _ _ _ _ => true
  | _ => false

def opName : PeOp → String
  | .abs => "abs"
  | .square => "square"
  | .chain _ _ _ => "chain"
  | .binary op _ _ rev => if rev && op == .sub then "rsub" else op.name

end PeOp

/-- Tiling parameters `(k, tileHeight)` for a single-task PE operation (`T = 2^k`). -/
structure PeTiling where
  k : Nat
  tileHeight : Nat
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Complete compilation artifact for a single-task Planar Engine operation. -/
structure CompiledPe where
  target : Target
  op : PeOp
  outDims : TensorDims
  tiling : PeTiling
  direct : Bool
  l2UseBytes : Nat
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections

/-- Maximum legal strip-width exponent `k`: `8` for 2-source, `9` for 1-source. -/
def peMaxK (twoSource : Bool) : Nat :=
  if twoSource then 8 else 9

/-- Strip width `T = 2^k`. -/
def peStripWidth (k : Nat) : Nat :=
  (1 : Nat) <<< k

/-- Tile height shift field `m = min(maxK - k, ceil_log2(th))` in `0x0044[7:4]`. -/
def peTileM (twoSource : Bool) (k th : Nat) : Nat :=
  min (peMaxK twoSource - k) (ceilLog2 th)

/-- Validate `TensorDims` bounds (`1..65535` on `C, H, W`). -/
def validateTensorDims (d : TensorDims) (label : String) : Except String Unit := do
  if d.c == 0 || d.c > 65535 then
    throw s!"PE {label}: C={d.c} must be in 1..65535"
  if d.h == 0 || d.h > 65535 then
    throw s!"PE {label}: H={d.h} must be in 1..65535"
  if d.w == 0 || d.w > 65535 then
    throw s!"PE {label}: W={d.w} must be in 1..65535"
  pure ()

/-- Validate broadcast compatibility of `src` into `out` (each axis `1` or equal to `out`). -/
def validateBroadcastAxis (srcDim outDim : Nat) (label axis : String) : Except String Unit := do
  if srcDim != 1 && srcDim != outDim then
    throw s!"PE {label}: {axis}={srcDim} cannot broadcast to output {axis}={outDim}"
  pure ()

/-- Compute and validate the broadcasted output shape of two PE input shapes. -/
def broadcastDims (s1 s2 : TensorDims) : Except String TensorDims := do
  validateTensorDims s1 "src1"
  validateTensorDims s2 "src2"
  let out : TensorDims := {
    c := max s1.c s2.c
    h := max s1.h s2.h
    w := max s1.w s2.w
  }
  validateBroadcastAxis s1.c out.c "src1" "C"
  validateBroadcastAxis s1.h out.h "src1" "H"
  validateBroadcastAxis s1.w out.w "src1" "W"
  validateBroadcastAxis s2.c out.c "src2" "C"
  validateBroadcastAxis s2.h out.h "src2" "H"
  validateBroadcastAxis s2.w out.w "src2" "W"
  pure out

/-- Whether PE result staging operates in direct mode (`true` on H17; staged on H16 iff
    `k == 5 && align_up(W, 8) > 32 && lastStrip == 24`). -/
def peIsDirect (target : Target) (outW k : Nat) : Bool :=
  match target with
  | .h17 => true
  | .h16 =>
    let r := alignUp outW 8
    let lastStrip := if r == 0 then 0 else r - 32 * ((r - 1) / 32)
    !(k == 5 && r > 32 && lastStrip == 24)

/-- Computed L2 SRAM staging geometry for a PE task. -/
structure PeL2Layout where
  stripW : Nat
  row1 : Nat
  u1 : Nat
  size1 : Nat
  whole2 : Bool
  src2Base : Nat
  row2 : Nat
  u2 : Nat
  size2 : Nat
  direct : Bool
  resBase : Nat
  drow : Nat
  du : Nat
  resL2Bytes : Nat
  l2UseBytes : Nat
  deriving Repr, BEq, Inhabited

/-- Compute exact L2 SRAM layout for a PE task given hardware `src1` and optional `src2`. -/
def computePeL2Layout
    (target : Target)
    (outDims hwSrc1 : TensorDims)
    (hwSrc2? : Option TensorDims)
    (t : PeTiling) : PeL2Layout :=
  let stripW := peStripWidth t.k
  let row1 := 2 * min (alignUp hwSrc1.w 8) stripW
  let u1 := hwSrc1.c * row1
  let size1 := (if hwSrc1.h > 1 then t.tileHeight else 1) * u1
  let (whole2, src2Base, row2, u2, size2, stop) :=
    match hwSrc2? with
    | none => (false, 0, 0, 0, 0, size1)
    | some hwSrc2 =>
      let w2 := hwSrc2.c == 1 && hwSrc2.h == 1 && hwSrc2.w > 1 && outDims.h > 1 && t.tileHeight < outDims.h
      let r2 := 2 * (if w2 then alignUp hwSrc2.w 8 else min (alignUp hwSrc2.w 8) stripW)
      let u2Val := hwSrc2.c * r2
      let b2 := alignUp size1 dmaLineBytes
      let sz2 := (if hwSrc2.h > 1 then t.tileHeight else 1) * u2Val
      (w2, b2, r2, u2Val, sz2, b2 + sz2)
  let direct := peIsDirect target outDims.w t.k
  let resBase := alignUp stop dmaLineBytes
  let drow := 2 * min (alignUp outDims.w 8) stripW
  let du := outDims.c * drow
  let resL2 := if direct then 0 else t.tileHeight * du
  {
    stripW := stripW
    row1 := row1
    u1 := u1
    size1 := size1
    whole2 := whole2
    src2Base := src2Base
    row2 := row2
    u2 := u2
    size2 := size2
    direct := direct
    resBase := resBase
    drow := drow
    du := du
    resL2Bytes := resL2
    l2UseBytes := resBase + resL2
  }

/-- Independent hard safety legality validator for PE tiling `(k, tileHeight)`.
    Enforces dimension bounds, broadcast compatibility, `k` range, `1 <= th <= out.h`,
    2 MiB L2 capacity, and the critical `l2ChanStride >= 2 * min(src.w, T)` anti-hang rule. -/
def validatePeTiling
    (target : Target)
    (outDims hwSrc1 : TensorDims)
    (hwSrc2? : Option TensorDims)
    (t : PeTiling) : Except String Unit := do
  validateTensorDims outDims "out"
  validateTensorDims hwSrc1 "src1"
  let twoSource := hwSrc2?.isSome
  if let some hwSrc2 := hwSrc2? then
    let expectedOut ← broadcastDims hwSrc1 hwSrc2
    if expectedOut != outDims then
      throw s!"PE out dims {repr outDims} != broadcasted dims {repr expectedOut}"
  else
    if hwSrc1 != outDims then
      throw s!"PE 1-source dims {repr hwSrc1} != out dims {repr outDims}"

  let maxK := peMaxK twoSource
  if t.k < 4 || t.k > maxK then
    throw s!"illegal PE tiling: k={t.k} out of range 4..{maxK}"
  if t.tileHeight == 0 || t.tileHeight > outDims.h then
    throw s!"illegal PE tiling: tileHeight={t.tileHeight} out of range 1..{outDims.h}"

  let l2 := computePeL2Layout target outDims hwSrc1 hwSrc2? t
  let minChs1 := 2 * min hwSrc1.w l2.stripW
  if hwSrc1.c > 1 && l2.row1 < minChs1 then
    throw s!"illegal PE tiling (anti-hang): src1ChanStride {l2.row1} < 2 * min(W1={hwSrc1.w}, T={l2.stripW}) = {minChs1}"
  if let some hwSrc2 := hwSrc2? then
    let minChs2 := 2 * min hwSrc2.w l2.stripW
    if hwSrc2.c > 1 && l2.row2 < minChs2 then
      throw s!"illegal PE tiling (anti-hang): src2ChanStride {l2.row2} < 2 * min(W2={hwSrc2.w}, T={l2.stripW}) = {minChs2}"
  if l2.l2UseBytes > l2CapacityBytes then
    throw s!"illegal PE tiling: L2 footprint {l2.l2UseBytes}B exceeds {l2CapacityBytes}B (2 MiB)"
  pure ()

/-- Compute maximal legal `tileHeight in 1..outDims.h` in O(1) for a fixed `k`. -/
private def maxPeTileHeightForK
    (target : Target)
    (outDims hwSrc1 : TensorDims)
    (hwSrc2? : Option TensorDims)
    (k : Nat) : Nat := Id.run do
  -- First try full height (th = out.h, where whole2 = false)
  let fullL2 := computePeL2Layout target outDims hwSrc1 hwSrc2? { k := k, tileHeight := outDims.h }
  if fullL2.l2UseBytes <= l2CapacityBytes then
    return outDims.h
  if outDims.h <= 1 then
    return 0

  -- Solve linearly for th in 1..(out.h - 1) where whole2 may be active
  let stripW := peStripWidth k
  let row1 := 2 * min (alignUp hwSrc1.w 8) stripW
  let u1 := hwSrc1.c * row1
  let (fixedBytes, rowCostSrc2) :=
    match hwSrc2? with
    | none => (if hwSrc1.h > 1 then 0 else u1, 0)
    | some hwSrc2 =>
      let w2 := hwSrc2.c == 1 && hwSrc2.h == 1 && hwSrc2.w > 1 && outDims.h > 1
      let r2 := 2 * (if w2 then alignUp hwSrc2.w 8 else min (alignUp hwSrc2.w 8) stripW)
      let u2 := hwSrc2.c * r2
      let f1 := if hwSrc1.h > 1 then 0 else u1
      let f2 := if hwSrc2.h > 1 then 0 else u2
      (f1 + f2, if hwSrc2.h > 1 then u2 else 0)
  let direct := peIsDirect target outDims.w k
  let drow := 2 * min (alignUp outDims.w 8) stripW
  let du := outDims.c * drow
  let rowCost := (if hwSrc1.h > 1 then u1 else 0) + rowCostSrc2 + (if direct then 0 else du)

  let thUpper : Nat :=
    if fixedBytes >= l2CapacityBytes then 0
    else if rowCost == 0 then outDims.h - 1
    else min (outDims.h - 1) ((l2CapacityBytes - fixedBytes) / rowCost)

  if thUpper == 0 then
    return 0

  -- Alignment slack is at most 2 * 63 = 126 bytes; refine down by up to 8 rows
  for dec in [:8] do
    if thUpper > dec then
      let th := thUpper - dec
      if th >= 1 then
        let l2 := computePeL2Layout target outDims hwSrc1 hwSrc2? { k := k, tileHeight := th }
        if l2.l2UseBytes <= l2CapacityBytes then
          return th
  return 0

/-- Deterministic O(1) tile chooser for PE tasks.
    Iterates `k` from `maxK` down to `4`, solves maximal `th` in O(1), ranks candidates by
    `(direct, min stripCount, max T*th, min T)`, and validates the winner via `validatePeTiling`. -/
def choosePeTiling
    (target : Target)
    (outDims hwSrc1 : TensorDims)
    (hwSrc2? : Option TensorDims) : Except String PeTiling := do
  validateTensorDims outDims "out"
  validateTensorDims hwSrc1 "src1"
  let twoSource := hwSrc2?.isSome
  if let some hwSrc2 := hwSrc2? then
    let expectedOut ← broadcastDims hwSrc1 hwSrc2
    if expectedOut != outDims then
      throw s!"PE out dims {repr outDims} != broadcasted dims {repr expectedOut}"
  else
    if hwSrc1 != outDims then
      throw s!"PE 1-source dims {repr hwSrc1} != out dims {repr outDims}"

  let maxK := peMaxK twoSource
  let outWAligned := alignUp outDims.w 8
  let mut best? : Option (PeTiling × Bool × Nat × Nat × Nat) := none

  -- Iterate k from maxK down to 4
  for step in [:maxK - 3] do
    let k := maxK - step
    let th := maxPeTileHeightForK target outDims hwSrc1 hwSrc2? k
    if th >= 1 then
      let cand : PeTiling := { k := k, tileHeight := th }
      match validatePeTiling target outDims hwSrc1 hwSrc2? cand with
      | .error _ => pure ()
      | .ok () =>
        let stripW := peStripWidth k
        let direct := peIsDirect target outDims.w k
        let strips := ceilDiv outWAligned stripW * ceilDiv outDims.h th
        let area := stripW * th
        let isBetter := match best? with
          | none => true
          | some (_, bDirect, bStrips, bArea, bStripW) =>
            if direct != bDirect then direct
            else if strips != bStrips then strips < bStrips
            else if area != bArea then area > bArea
            else stripW < bStripW
        if isBetter then
          best? := some (cand, direct, strips, area, stripW)

  match best? with
  | some (winner, _, _, _, _) =>
    validatePeTiling target outDims hwSrc1 hwSrc2? winner
    pure winner
  | none =>
    throw s!"no legal {target.name} PE tiling found for out={repr outDims}, src1={repr hwSrc1}"

/-- Resolve hardware `(outDims, hwSrc1, hwSrc2?, hwSrc1Bar, hwSrc2Bar?)` from `(op, primaryDims)`. -/
def resolvePeGeometry (op : PeOp) (primaryDims : TensorDims) :
    Except String (TensorDims × TensorDims × Option TensorDims × UInt8 × Option UInt8) := do
  match op with
  | .abs | .square =>
    validateTensorDims primaryDims "input"
    pure (primaryDims, primaryDims, none, 6, none)
  | .chain inScale bias scale =>
    validateTensorDims primaryDims "input"
    if inScale.isNone && bias.isNone && scale.isNone then
      throw "PE chain requires at least one of inScale, bias, or scale"
    pure (primaryDims, primaryDims, none, 6, none)
  | .binary _ xDims zDims rev =>
    let outDims ← broadcastDims xDims zDims
    if rev then
      pure (outDims, zDims, some xDims, 7, some 6)
    else
      pure (outDims, xDims, some zDims, 6, some 7)

/-- Lower a single-task Planar Engine op into a `TaskDesc` (`ene = 0`, no `__KERN_0`). -/
def lowerPeTask
    (target : Target)
    (op : PeOp)
    (outDims hwSrc1 : TensorDims)
    (hwSrc2? : Option TensorDims)
    (hwSrc1Bar : UInt8)
    (hwSrc2Bar? : Option UInt8)
    (tiling : PeTiling)
    (taskId : Nat := 0) : Except String TaskDesc := do
  validatePeTiling target outDims hwSrc1 hwSrc2? tiling
  let twoSource := hwSrc2?.isSome
  let l2 := computePeL2Layout target outDims hwSrc1 hwSrc2? tiling
  let m := peTileM twoSource tiling.k tiling.tileHeight

  let wU32 ← natToU32Checked outDims.w "W"
  let hU32 ← natToU32Checked outDims.h "H"
  let cU32 ← natToU32Checked outDims.c "C"
  let thU32 ← natToU32Checked tiling.tileHeight "TH"
  let kU32 ← natToU32Checked tiling.k "peTile.k"
  let mU32 ← natToU32Checked m "peTile.m"
  let taskIdU32 ← natToU32Checked taskId "taskId"

  let wWin ← encodeWord (dim16Layout "Win") #[("val", wU32)]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", hU32)]
  let wWout ← encodeWord (dim16Layout "Wout") #[("val", wU32)]
  let wHout ← encodeWord (dim16Layout "Hout") #[("val", hU32)]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", thU32)]
  let wPeCommon ← encodeWord peCommonLayout #[
    ("mode", if twoSource then peCommonTwoSrcValue else peCommonOneSrcValue)
  ]
  let wPeTile ← encodeWord peTileLayout #[("k", kU32), ("m", mU32)]

  let wSrc1Cfg ← encodeWord src1CfgLayout #[("mode", 2), ("P", 1)]
  let wResCfg ← encodeWord resCfgLayout #[
    ("mode", 2),
    ("direct", if l2.direct then 1 else 0),
    ("P", 1)
  ]
  let wSrc1Fmt ← encodeWord (dmaFmtLayout "src1Fmt") #[("format", 0x002031), ("P", 1)]
  let wResFmt ← encodeWord (dmaFmtLayout "resFmt") #[("format", 0x302031), ("P", 1)]
  let wOut2 ← encodeWord out2Layout #[("form", 0xe0), ("task", taskIdU32)]

  let row1U32 ← natToU32Checked l2.row1 "src1ChanStride"
  let u1U32 ← natToU32Checked l2.u1 "src1RowStride"
  let resBaseU32 ← natToU32Checked l2.resBase "resBase"

  let src1DmaRow := alignUp (2 * hwSrc1.w) dmaLineBytes
  let src1DmaPlane := hwSrc1.h * src1DmaRow
  let src1DmaRowU32 ← natToU32Checked src1DmaRow "src1DmaRowStride"
  let src1DmaPlaneU32 ← natToU32Checked src1DmaPlane "src1DmaPlaneStride"

  let outDmaRow := alignUp (2 * outDims.w) dmaLineBytes
  let outDmaPlane := outDims.h * outDmaRow
  let outDmaRowU32 ← natToU32Checked outDmaRow "resDmaRowStride"
  let outDmaPlaneU32 ← natToU32Checked outDmaPlane "resDmaPlaneStride"

  let mut plain : Array RegWrite := #[
    -- Common block
    { byteAddr := regWin, value := wWin },
    { byteAddr := regHin, value := wHin },
    { byteAddr := regCin, value := cU32 },
    { byteAddr := regWout, value := wWout },
    { byteAddr := regHout, value := wHout },
    { byteAddr := regCout, value := cU32 },
    { byteAddr := regBatchN, value := 1 },
    { byteAddr := regTH, value := wTH },
    { byteAddr := regPeCommon, value := wPeCommon },
    { byteAddr := regPeTile, value := wPeTile },
    { byteAddr := regTargetCfg, value := 0 },
    -- L2 staging (src1 & result)
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
    { byteAddr := regSrc2Cfg, value := if twoSource then src2CfgEnabledValue else src2CfgValue },
    { byteAddr := regSrc1Base, value := 0 },
    { byteAddr := regSrc1ChanStride, value := row1U32 },
    { byteAddr := regSrc1RowStride, value := u1U32 },
    { byteAddr := regSrc1RowBytesA, value := u1U32 },
    { byteAddr := regSrc1RowBytesB, value := u1U32 },
    { byteAddr := regResCfg, value := wResCfg },
    { byteAddr := regResBase, value := resBaseU32 },
    -- Tile DMA (src1)
    { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
    { byteAddr := regSrc1DmaRowStride, value := src1DmaRowU32 },
    { byteAddr := regSrc1DmaPlaneStride, value := src1DmaPlaneU32 },
    { byteAddr := regSrc1Fmt, value := wSrc1Fmt },
    -- Result DMA
    { byteAddr := regResDmaRowStride, value := outDmaRowU32 },
    { byteAddr := regResDmaPlaneStride, value := outDmaPlaneU32 },
    { byteAddr := regResFmt, value := wResFmt },
    -- Kernel stage (no kernel)
    { byteAddr := regKernelStage, value := kernelStageNoKernValue },
    -- Output stage
    { byteAddr := regOut0, value := out0Value },
    { byteAddr := regOut1, value := out1Value },
    { byteAddr := regOut2, value := wOut2 },
    { byteAddr := regOut3, value := out3Value },
    { byteAddr := regOut4, value := out4Value },
    { byteAddr := regOut5, value := out5Value }
  ]

  -- Staged result L2 registers (H16 staged edge case only)
  if !l2.direct then
    let drowU32 ← natToU32Checked l2.drow "resChanStride"
    let duU32 ← natToU32Checked l2.du "resRowStride"
    plain := plain ++ #[
      { byteAddr := regResChanStride, value := drowU32 },
      { byteAddr := regResRowStride, value := duU32 },
      { byteAddr := regResRowBytesA, value := duU32 },
      { byteAddr := regResRowBytesB, value := duU32 }
    ]

  -- Source 2 L2, DMA, and broadcast flags (2-source only)
  if let some hwSrc2 := hwSrc2? then
    let wPeBcast ← encodeWord peBcastLayout #[
      ("src1W", if hwSrc1.w != outDims.w then 1 else 0),
      ("src1H", if hwSrc1.h != outDims.h then 1 else 0),
      ("src1C", if hwSrc1.c != outDims.c then 1 else 0),
      ("src2W", if hwSrc2.w != outDims.w then 1 else 0),
      ("src2H", if hwSrc2.h != outDims.h then 1 else 0),
      ("src2C", if hwSrc2.c != outDims.c then 1 else 0)
    ]
    let src2BaseU32 ← natToU32Checked l2.src2Base "src2Base"
    let row2U32 ← natToU32Checked l2.row2 "src2ChanStride"
    let u2U32 ← natToU32Checked l2.u2 "src2RowStride"
    let src2DmaRow := alignUp (2 * hwSrc2.w) dmaLineBytes
    let src2DmaPlane := hwSrc2.h * src2DmaRow
    let src2DmaRowU32 ← natToU32Checked src2DmaRow "src2DmaRowStride"
    let src2DmaPlaneU32 ← natToU32Checked src2DmaPlane "src2DmaPlaneStride"
    let wSrc2Fmt ← encodeWord (dmaFmtLayout "src2Fmt") #[("format", 0x002031), ("P", 1)]
    plain := plain ++ #[
      { byteAddr := regPeBcast, value := wPeBcast },
      { byteAddr := regSrc2Base, value := src2BaseU32 },
      { byteAddr := regSrc2ChanStride, value := row2U32 },
      { byteAddr := regSrc2RowStride, value := u2U32 },
      { byteAddr := regSrc2RowBytesA, value := u2U32 },
      { byteAddr := regSrc2RowBytesB, value := u2U32 },
      { byteAddr := regSrc2DmaCfg, value := src1DmaCfgValue },
      { byteAddr := regSrc2DmaRowStride, value := src2DmaRowU32 },
      { byteAddr := regSrc2DmaPlaneStride, value := src2DmaPlaneU32 },
      { byteAddr := regSrc2Fmt, value := wSrc2Fmt }
    ]

  -- PE arithmetic block (0x4500..0x4510)
  match op with
  | .abs =>
    plain := plain.push { byteAddr := regPeOp, value := peOpAbsValue }
  | .square =>
    plain := plain.push { byteAddr := regPeOp, value := peOpSquareValue }
  | .chain inScale bias scale =>
    if let some b := bias then
      plain := plain.push { byteAddr := regPeBias, value := f16ToF32Bits b }
    if let some s := scale then
      plain := plain.push { byteAddr := regPeScale, value := f16ToF32Bits s }
    if let some ins := inScale then
      plain := plain.push { byteAddr := regPeInScale, value := f16ToF32Bits ins }
  | .binary binOp _ _ _ =>
    plain := plain.push { byteAddr := regPeOp, value := binOp.regValue }

  -- BAR-tagged records
  let wSrcDep ← encodeWord src1DepLayout #[("code", 0x21), ("task", taskIdU32)]
  let wResDep ← encodeWord resDepLayout #[
    ("code", resDepCode target),
    ("task", taskIdU32),
    ("direct", if l2.direct then 1 else 0)
  ]
  let resBarSlot : UInt8 := if twoSource then 8 else 7
  let mut barRecs : Array BarRegWrite := #[
    { baseByteAddr := regSrc1BaseLo, barSlot := hwSrc1Bar, values := #[0, 0] }
  ]
  if let some s2Bar := hwSrc2Bar? then
    barRecs := barRecs.push { baseByteAddr := regSrc2BaseLo, barSlot := s2Bar, values := #[0, 0] }
  barRecs := barRecs ++ #[
    { baseByteAddr := regResBaseLo, barSlot := resBarSlot, values := #[0, 0] },
    { baseByteAddr := regSrc1Dep, barSlot := 4, values := #[wSrcDep] }
  ]
  if twoSource then
    barRecs := barRecs.push { baseByteAddr := regSrc2Dep, barSlot := 4, values := #[wSrcDep] }
  barRecs := barRecs.push { baseByteAddr := regResDep, barSlot := 4, values := #[wResDep] }

  pure {
    taskId := taskId
    logNE := 0
    eneOverride := some 0
    header4 := taskHeaderWord4Const
    plainWrites := plain
    barWrites := barRecs
    kernelRelocs := #[]
  }

/-- Compile a single-task Planar Engine operation into a statically verified HWX binary
    and its 4 host-built Linux LOAD sections. -/
def compilePe
    (target : Target)
    (op : PeOp)
    (dims : TensorDims)
    (tilingOverride : Option PeTiling := none) : Except String CompiledPe := do
  let (outDims, hwSrc1, hwSrc2?, hwSrc1Bar, hwSrc2Bar?) ← resolvePeGeometry op dims
  let tiling ← match tilingOverride with
    | some t => validatePeTiling target outDims hwSrc1 hwSrc2? t; pure t
    | none => choosePeTiling target outDims hwSrc1 hwSrc2?

  let taskDesc ← lowerPeTask target op outDims hwSrc1 hwSrc2? hwSrc1Bar hwSrc2Bar? tiling

  let surfaces ← match op with
    | .binary _ xDims zDims _ =>
      let sX ← makeFp16Surface target "x" "input_x" .input xDims.c xDims.h xDims.w
      let sZ ← makeFp16Surface target "z" "input_z" .input zDims.c zDims.h zDims.w
      let sY ← makeFp16Surface target "y" "output_y" .output outDims.c outDims.h outDims.w
      pure #[sX, sZ, sY]
    | _ =>
      let sX ← makeFp16Surface target "x" "input_x" .input dims.c dims.h dims.w
      let sY ← makeFp16Surface target "y" "output_y" .output outDims.c outDims.h outDims.w
      pure #[sX, sY]

  let emitted ← emitProgram {
    target := target
    surfaces := surfaces
    tasks := #[taskDesc]
    kernel := none
    scratch := none
    constants := zeroBytes textConstBytes
    ident := some "vane"
  }

  let _ ← verifyHwxTasks emitted.hwx emitted.info (requireKnownTask := true)
  let l2 := computePeL2Layout target outDims hwSrc1 hwSrc2? tiling

  pure {
    target := target
    op := op
    outDims := outDims
    tiling := tiling
    direct := l2.direct
    l2UseBytes := l2.l2UseBytes
    hwx := emitted.hwx
    info := emitted.info
    sections := emitted.sections
  }

end Vane.Pe
