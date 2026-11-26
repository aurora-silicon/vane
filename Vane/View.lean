import Vane.Binary
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register
import Vane.Alloc
import Vane.Tiling
import Vane.Program
import Vane.Verify

namespace Vane.View

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Alloc
open Vane.Tiling
open Vane.Program
open Vane.Verify

-- ============================================================================
-- 4D Tensor Shapes, Pure Partition Helpers & Group Code
-- ============================================================================

/-- 4D ANE fp16 tensor shape `[N, C, H, W]`. -/
structure Shape4D where
  n : Nat := 1
  c : Nat
  h : Nat
  w : Nat
  deriving Repr, BEq, DecidableEq, Inhabited

namespace Shape4D

def toArray (s : Shape4D) : Array Nat :=
  #[s.n, s.c, s.h, s.w]

def ofArray (a : Array Nat) : Except String Shape4D := do
  if a.size != 4 then
    throw s!"expected 4D shape [N, C, H, W], got {a.size} dims"
  pure { n := a[0]!, c := a[1]!, h := a[2]!, w := a[3]! }

def axis (s : Shape4D) : Nat → Nat
  | 0 => s.n
  | 1 => s.c
  | 2 => s.h
  | 3 => s.w
  | _ => 0

def withAxis (s : Shape4D) (ax val : Nat) : Shape4D :=
  match ax with
  | 0 => { s with n := val }
  | 1 => { s with c := val }
  | 2 => { s with h := val }
  | 3 => { s with w := val }
  | _ => s

def numElems (s : Shape4D) : Nat :=
  s.n * s.c * s.h * s.w

def rowStride (s : Shape4D) : Nat :=
  alignUp (2 * s.w) dmaLineBytes

def planeStride (s : Shape4D) : Nat :=
  s.h * s.rowStride

def batchStride (s : Shape4D) : Nat :=
  s.c * s.planeStride

def span (s : Shape4D) : Nat :=
  s.n * s.batchStride

/-- ANE `[N, C, H, W]` fp16 DRAM strides `#[batchStride, planeStride, rowStride, 2]`. -/
def dramStrides (s : Shape4D) : Array Nat :=
  #[s.batchStride, s.planeStride, s.rowStride, 2]

def validate (s : Shape4D) (label : String := "shape") : Except String Unit := do
  if s.n == 0 || s.n > 65535 then
    throw s!"{label}: N={s.n} must be in 1..65535"
  if s.c == 0 || s.c > maxU32Nat then
    throw s!"{label}: C={s.c} must be in 1..{maxU32Nat}"
  if s.h == 0 || s.h > 65535 then
    throw s!"{label}: H={s.h} must be in 1..65535"
  if s.w == 0 || s.w > 65535 then
    throw s!"{label}: W={s.w} must be in 1..65535"
  pure ()

end Shape4D

/-- `groupCode(c)` for T-mode (`transposeCW = true`) NE view copy tasks:
    `1` for `c <= 16`, `2` for `17..32`, `4` for `33..64`, `8` for `c >= 65`.
    Matches `min(8, 2 ^ floor(log2(2 * ceil(c / 16) - 1)))` for all `c >= 1`. -/
def groupCode (c : Nat) : Nat :=
  if c <= 16 then 1
  else if c <= 32 then 2
  else if c <= 64 then 4
  else 8

/-- `log2(groupCode c)` in `0..3`, used for `neTile.ocg` when `transposeCW = true`. -/
def groupCodeLog2 (c : Nat) : Nat :=
  if c <= 16 then 0
  else if c <= 32 then 1
  else if c <= 64 then 2
  else 3

/-- Auxiliary prefix-start accumulator for slice/concat/channel-split partitions. -/
def prefixStartsAux (acc : Nat) : List Nat → List Nat
  | [] => []
  | sz :: rest => acc :: prefixStartsAux (acc + sz) rest

/-- Prefix start offsets `[0, s_0, s_0 + s_1, ...]` for a list of slice sizes. -/
def prefixStarts (sizes : List Nat) : List Nat :=
  prefixStartsAux 0 sizes

/-- Total sum of slice sizes. -/
def sumSizes : List Nat → Nat
  | [] => 0
  | sz :: rest => sz + sumSizes rest

/-- Convert a list of slice sizes starting at `acc` into half-open `[start, start + sz)` intervals. -/
def sliceIntervalsAux (acc : Nat) : List Nat → List (Nat × Nat)
  | [] => []
  | sz :: rest => (acc, acc + sz) :: sliceIntervalsAux (acc + sz) rest

/-- Contiguous half-open `[start_i, start_i + sz_i)` partition intervals starting at `0`. -/
def sliceIntervals (sizes : List Nat) : List (Nat × Nat) :=
  sliceIntervalsAux 0 sizes

-- ============================================================================
-- Abstract Copy Descriptor & View Plan
-- ============================================================================

/-- Descriptor for a single weightless NE copy / view task (`0x4904 = 0x00103c0c`). -/
structure ViewCopy where
  n : Nat
  c : Nat
  h : Nat
  w : Nat
  d : Nat := 1
  inOff : Nat := 0
  inRowStride : Nat := 0
  inPlaneStride : Nat := 0
  inDStride : Nat := 0
  inNStride : Nat := 0
  outOff : Nat := 0
  outRowStride : Nat := 0
  outPlaneStride : Nat := 0
  outDStride : Nat := 0
  outNStride : Nat := 0
  transposeCW : Bool := false
  inBuf : Nat := 0
  outBuf : Nat := 0
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Complete lowered view plan consisting of 1 or more `ViewCopy` tasks. -/
structure ViewPlan where
  opName : String
  inShapes : Array Shape4D
  outShape : Shape4D
  copies : Array ViewCopy
  deriving Repr, BEq, Inhabited

/-- Placement & tiling plan for one `ViewCopy` task (usable standalone or in multi-task graphs). -/
structure ViewTaskPlan where
  tiling : Tiling
  direct : Bool
  srcStageBase : Nat
  resStageBase : Nat
  l2TaskPeak : Nat
  deriving Repr, BEq, Inhabited

/-- Complete compiled artifact for a standalone data-movement view program. -/
structure CompiledView where
  target : Target
  plan : ViewPlan
  taskPlans : Array ViewTaskPlan
  l2PeakBytes : Nat
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections

-- ============================================================================
-- No-Copy 64B-Padded DRAM Stride Derivation for Reshape (`viewStrides`)
-- ============================================================================

/-- Compute DRAM byte strides `[s0, s1, s2, s3]` for viewing 4D fp16 tensor `x` as 4D tensor `y`
    without copying, respecting ANE's 64-byte-padded row stride on `x[3]`.
    Returns `some s` iff non-1 axes of `x` and `y` partition into matching-product groups where
    every merged `x` group is contiguous in DRAM (`xs[k] == x[k+1] * xs[k+1]`). -/
def viewStrides (x y : Array Nat) : Option (Array Nat) := Id.run do
  if x.size != 4 || y.size != 4 then
    return none
  for d in x do
    if d == 0 then return none
  for d in y do
    if d == 0 then return none
  let prodX := x[0]! * x[1]! * x[2]! * x[3]!
  let prodY := y[0]! * y[1]! * y[2]! * y[3]!
  if prodX != prodY then
    return none

  let rowX := alignUp (2 * x[3]!) dmaLineBytes
  let planeX := x[2]! * rowX
  let batchX := x[1]! * planeX
  let xs : Array Nat := #[batchX, planeX, rowX, 2]

  if prodX == 1 then
    return some #[batchX, planeX, rowX, 2]

  let mut xNon1 : Array (Nat × Nat × Nat) := #[]
  for i in [:4] do
    if x[i]! > 1 then
      xNon1 := xNon1.push (i, x[i]!, xs[i]!)

  let mut yNon1 : Array (Nat × Nat) := #[]
  for i in [:4] do
    if y[i]! > 1 then
      yNon1 := yNon1.push (i, y[i]!)

  let mut outSt : Array Nat := #[0, 0, 0, 2]
  let mut xi : Nat := 0
  let mut yi : Nat := 0

  while xi < xNon1.size && yi < yNon1.size do
    let mut xEnd := xi
    let mut yEnd := yi
    let mut px := xNon1[xi]!.2.1
    let mut py := yNon1[yi]!.2

    while px != py do
      if px < py then
        xEnd := xEnd + 1
        if xEnd >= xNon1.size then
          return none
        let prevSt := xNon1[xEnd - 1]!.2.2
        let (_, curDim, curSt) := xNon1[xEnd]!
        if prevSt != curDim * curSt then
          return none
        px := px * curDim
      else
        yEnd := yEnd + 1
        if yEnd >= yNon1.size then
          return none
        py := py * yNon1[yEnd]!.2

    let mut curSt := xNon1[xEnd]!.2.2
    let yGroupLen := yEnd - yi + 1
    for step in [:yGroupLen] do
      let k := yEnd - step
      let (yIdx, yDim) := yNon1[k]!
      outSt := outSt.set! yIdx curSt
      curSt := curSt * yDim

    xi := xEnd + 1
    yi := yEnd + 1

  if xi != xNon1.size || yi != yNon1.size then
    return none

  -- Fill unused (size-1) axis strides with safe 64B-aligned values (2 for W)
  if y[3]! == 1 then
    outSt := outSt.set! 3 2
  for step in [:3] do
    let idx := 2 - step
    if y[idx]! == 1 then
      let nextMin := y[idx + 1]! * outSt[idx + 1]!
      outSt := outSt.set! idx (max dmaLineBytes (alignUp nextMin dmaLineBytes))

  return some outSt

-- ============================================================================
-- View Operation Planners (transpose, slice, concat, split, reshape)
-- ============================================================================

/-- Plan a 4D `transpose(x, perm)` view copy.
    - W-preserving (`perm[3] == 3` or both `x.w == 1 && y.w == 1`): plain copy (`transposeCW = false`).
    - W-moving (`C <-> W` or any axis moving to `W`): T-mode (`transposeCW = true`), enforcing
      the Linux kernel `ane_hx_dma_tile` destination bound at compile time. -/
def planTranspose (x : Shape4D) (perm : Array Nat) : Except String ViewPlan := do
  x.validate "transpose input"
  if perm.size != 4 then
    throw s!"transpose: perm must have length 4, got {perm.size}"
  let mut seen : Array Bool := Array.replicate 4 false
  for p in perm do
    if p >= 4 then
      throw s!"transpose: perm axis {p} out of range 0..3"
    if seen[p]! then
      throw s!"transpose: duplicate axis {p} in perm"
    seen := seen.set! p true

  let xArr := x.toArray
  let yArr : Array Nat := #[xArr[perm[0]!]!, xArr[perm[1]!]!, xArr[perm[2]!]!, xArr[perm[3]!]!]
  let y ← Shape4D.ofArray yArr
  y.validate "transpose output"
  let xs := x.dramStrides
  let ys := y.dramStrides

  if perm[3]! == 3 || (x.w == 1 && y.w == 1) then
    let inStAt := fun (i : Nat) =>
      let ax := perm[i]!
      if yArr[i]! <= 1 then 0
      else xs[ax]!
    let cp : ViewCopy := {
      n := y.n, c := y.c, h := y.h, w := y.w, d := 1
      inOff := 0
      inNStride := inStAt 0
      inPlaneStride := inStAt 1
      inRowStride := inStAt 2
      outOff := 0
      outNStride := if y.n > 1 then ys[0]! else 0
      outPlaneStride := if y.c > 1 then ys[1]! else 0
      outRowStride := if y.h > 1 then ys[2]! else 0
      transposeCW := false
      inBuf := 0, outBuf := 0
    }
    pure { opName := "transpose", inShapes := #[x], outShape := y, copies := #[cp] }
  else
    -- W-moving T-mode (`transposeCW = true`)
    let aC := perm[3]!
    let posOf := fun (ax : Nat) => Id.run do
      for i in [:4] do
        if perm[i]! == ax then return i
      return 0
    let rem : Array Nat := (#[0, 1, 2] : Array Nat).filter (fun a => a != aC)
    let aN := rem[0]!
    let aH := rem[1]!

    let tn := xArr[aN]!
    let tc := xArr[aC]!
    let th := xArr[aH]!
    let tw := xArr[3]!

    let inNSt := if tn > 1 then xs[aN]! else 0
    let inPSt := if tc > 1 then xs[aC]! else 0
    let inRSt := if th > 1 then xs[aH]! else 0

    let outNSt := if tn > 1 then ys[posOf aN]! else 0
    let outPSt := if tw > 1 then ys[posOf 3]! else 0
    let outRSt := if th > 1 then ys[posOf aH]! else 0

    -- Enforce Linux kernel `ane_hx_dma_tile` destination bound (uses Cout=tc with outPlaneStride=outPSt)
    let kernelOutRaw := 2 * tw + (th - 1) * outRSt + (tc - 1) * outPSt + (tn - 1) * outNSt
    let kernelOutEnd := alignUp kernelOutRaw dmaLineBytes
    if kernelOutEnd > y.span then
      throw s!"transpose T-mode ({x.n}x{x.c}x{x.h}x{x.w} -> {y.n}x{y.c}x{y.h}x{y.w}) exceeds Linux kernel ane_hx_dma_tile destination bound ({kernelOutEnd}B > {y.span}B; requires C={tc} <= W={tw} or sufficient output span)"

    let cp : ViewCopy := {
      n := tn, c := tc, h := th, w := tw, d := 1
      inOff := 0
      inNStride := inNSt
      inPlaneStride := inPSt
      inRowStride := inRSt
      outOff := 0
      outNStride := outNSt
      outPlaneStride := outPSt
      outRowStride := outRSt
      transposeCW := true
      inBuf := 0, outBuf := 0
    }
    pure { opName := "transpose", inShapes := #[x], outShape := y, copies := #[cp] }

/-- Plan a 4D `slice(x, begin, end, step)` view copy.
    Kernel-clean legality: `step[3] == 1` and `begin[3] % 32 == 0` (64B-aligned `0x4d10`, zero `0x4d84`). -/
def planSlice (x : Shape4D) (beginIdx endIdx step : Array Nat) : Except String ViewPlan := do
  x.validate "slice input"
  if beginIdx.size != 4 || endIdx.size != 4 || step.size != 4 then
    throw "slice: begin, end, and step must each have length 4"
  let xArr := x.toArray
  let mut yArr : Array Nat := Array.mkEmpty 4
  for i in [:4] do
    let b := beginIdx[i]!
    let e := endIdx[i]!
    let s := step[i]!
    if s == 0 then
      throw s!"slice: step[{i}] must be >= 1, got 0"
    if b >= e || e > xArr[i]! then
      throw s!"slice: axis {i} range [{b}, {e}) invalid for dim {xArr[i]!}"
    yArr := yArr.push (ceilDiv (e - b) s)
  if step[3]! != 1 then
    throw s!"slice: W-axis step must be 1 (got step[3]={step[3]!})"
  if beginIdx[3]! % 32 != 0 then
    throw s!"slice: W-axis begin offset {beginIdx[3]!} must be a multiple of 32 (64-byte DMA alignment without 0x4d84)"

  let y ← Shape4D.ofArray yArr
  y.validate "slice output"
  let xs := x.dramStrides
  let ys := y.dramStrides

  let inOff := beginIdx[0]! * xs[0]! + beginIdx[1]! * xs[1]! + beginIdx[2]! * xs[2]! + 2 * beginIdx[3]!
  let cp : ViewCopy := {
    n := y.n, c := y.c, h := y.h, w := y.w, d := 1
    inOff := inOff
    inNStride := if y.n > 1 then xs[0]! * step[0]! else 0
    inPlaneStride := if y.c > 1 then xs[1]! * step[1]! else 0
    inRowStride := if y.h > 1 then xs[2]! * step[2]! else 0
    outOff := 0
    outNStride := if y.n > 1 then ys[0]! else 0
    outPlaneStride := if y.c > 1 then ys[1]! else 0
    outRowStride := if y.h > 1 then ys[2]! else 0
    transposeCW := false
    inBuf := 0, outBuf := 0
  }
  pure { opName := "slice", inShapes := #[x], outShape := y, copies := #[cp] }

/-- Plan a multi-input `concat(inShapes, axis)` (`axis in {1, 2, 3}`).
    Emits `m` plain copy tasks writing disjoint slices into the shared output buffer. -/
def planConcat (inShapes : Array Shape4D) (axis : Nat) : Except String ViewPlan := do
  if inShapes.size < 2 then
    throw s!"concat: requires at least 2 inputs, got {inShapes.size}"
  if axis != 1 && axis != 2 && axis != 3 then
    throw s!"concat: axis must be 1 (C), 2 (H), or 3 (W), got {axis}"
  let base := inShapes[0]!
  base.validate "concat input[0]"
  let mut sizes : List Nat := []
  for i in [:inShapes.size] do
    let s := inShapes[i]!
    s.validate s!"concat input[{i}]"
    for ax in [:4] do
      if ax != axis && s.axis ax != base.axis ax then
        throw s!"concat: input[{i}] axis {ax} dim {s.axis ax} != input[0] dim {base.axis ax}"
    sizes := sizes ++ [s.axis axis]

  let starts := prefixStarts sizes
  let total := sumSizes sizes
  let y := base.withAxis axis total
  y.validate "concat output"
  let ys := y.dramStrides

  if axis == 3 then
    for i in [:starts.length] do
      let st := starts[i]!
      if st % 32 != 0 then
        throw s!"concat along W: input[{i}] start offset {st} is not a multiple of 32 (64-byte DMA alignment)"

  let mut copies : Array ViewCopy := Array.mkEmpty inShapes.size
  for i in [:inShapes.size] do
    let s := inShapes[i]!
    let xs := s.dramStrides
    let st := starts[i]!
    let outOff :=
      match axis with
      | 1 => st * ys[1]!
      | 2 => st * ys[2]!
      | _ => 2 * st
    copies := copies.push {
      n := s.n, c := s.c, h := s.h, w := s.w, d := 1
      inOff := 0
      inNStride := if s.n > 1 then xs[0]! else 0
      inPlaneStride := if s.c > 1 then xs[1]! else 0
      inRowStride := if s.h > 1 then xs[2]! else 0
      outOff := outOff
      outNStride := if s.n > 1 then ys[0]! else 0
      outPlaneStride := if s.c > 1 then ys[1]! else 0
      outRowStride := if s.h > 1 then ys[2]! else 0
      transposeCW := false
      inBuf := i, outBuf := 0
    }
  pure { opName := "concat", inShapes := inShapes, outShape := y, copies := copies }

/-- Plan extracting output partition `part` of `split(x, axis, sizes)` (`axis in {1, 2, 3}`). -/
def planSplitPart (x : Shape4D) (axis : Nat) (sizes : Array Nat) (part : Nat) : Except String ViewPlan := do
  x.validate "split input"
  if axis != 1 && axis != 2 && axis != 3 then
    throw s!"split: axis must be 1 (C), 2 (H), or 3 (W), got {axis}"
  if sizes.isEmpty then
    throw "split: sizes must be non-empty"
  for i in [:sizes.size] do
    if sizes[i]! == 0 then
      throw s!"split: sizes[{i}] must be >= 1"
  let total := sumSizes sizes.toList
  if total != x.axis axis then
    throw s!"split: sum(sizes)={total} != input axis {axis} dim {x.axis axis}"
  if part >= sizes.size then
    throw s!"split: part index {part} out of range 0..{sizes.size - 1}"

  let starts := prefixStarts sizes.toList
  if axis == 3 then
    for i in [:starts.length] do
      let st := starts[i]!
      if st % 32 != 0 then
        throw s!"split along W: part {i} start offset {st} is not a multiple of 32 (64-byte DMA alignment)"

  let st := starts[part]!
  let sz := sizes[part]!
  let y := x.withAxis axis sz
  y.validate "split output"
  let xs := x.dramStrides
  let ys := y.dramStrides
  let inOff :=
    match axis with
    | 1 => st * xs[1]!
    | 2 => st * xs[2]!
    | _ => 2 * st

  let cp : ViewCopy := {
    n := y.n, c := y.c, h := y.h, w := y.w, d := 1
    inOff := inOff
    inNStride := if y.n > 1 then xs[0]! else 0
    inPlaneStride := if y.c > 1 then xs[1]! else 0
    inRowStride := if y.h > 1 then xs[2]! else 0
    outOff := 0
    outNStride := if y.n > 1 then ys[0]! else 0
    outPlaneStride := if y.c > 1 then ys[1]! else 0
    outRowStride := if y.h > 1 then ys[2]! else 0
    transposeCW := false
    inBuf := 0, outBuf := 0
  }
  pure { opName := "split", inShapes := #[x], outShape := y, copies := #[cp] }

/-- Plan a no-copy strided `reshape(x, y)` via `viewStrides`. -/
def planReshape (x y : Shape4D) : Except String ViewPlan := do
  x.validate "reshape input"
  y.validate "reshape output"
  if x.numElems != y.numElems then
    throw s!"reshape: element count mismatch ({x.numElems} != {y.numElems})"
  let some s := viewStrides x.toArray y.toArray
    | throw s!"reshape ({x.n}x{x.c}x{x.h}x{x.w} -> {y.n}x{y.c}x{y.h}x{y.w}): non-contiguous across 64-byte row padding"
  if s[0]! % dmaLineBytes != 0 || s[1]! % dmaLineBytes != 0 || s[2]! % dmaLineBytes != 0 then
    throw s!"reshape ({x.n}x{x.c}x{x.h}x{x.w} -> {y.n}x{y.c}x{y.h}x{y.w}): derived input strides [{s[0]!}, {s[1]!}, {s[2]!}] are not 64-byte aligned"
  if y.w > 1 && s[3]! != 2 then
    throw s!"reshape ({x.n}x{x.c}x{x.h}x{x.w} -> {y.n}x{y.c}x{y.h}x{y.w}): innermost W stride {s[3]!} != 2"

  let ys := y.dramStrides
  let cp : ViewCopy := {
    n := y.n, c := y.c, h := y.h, w := y.w, d := 1
    inOff := 0
    inNStride := if y.n > 1 then s[0]! else 0
    inPlaneStride := if y.c > 1 then s[1]! else 0
    inRowStride := if y.h > 1 then s[2]! else 0
    outOff := 0
    outNStride := if y.n > 1 then ys[0]! else 0
    outPlaneStride := if y.c > 1 then ys[1]! else 0
    outRowStride := if y.h > 1 then ys[2]! else 0
    transposeCW := false
    inBuf := 0, outBuf := 0
  }
  pure { opName := "reshape", inShapes := #[x], outShape := y, copies := #[cp] }

-- ============================================================================
-- Tiling Chooser & Multi-Task Register Lowering for `ViewCopy`
-- ============================================================================

/-- Direct-mode predicate for weightless NE view copy tasks. -/
def viewIsDirect (target : Target) (cp : ViewCopy) (t : Tiling) : Bool :=
  if cp.transposeCW then
    !(9 <= cp.c && cp.c <= 16 && t.shPref == 0 && cp.w < 32)
  else
    target.profile.directThreshold t.fat t.shMax >= min 32 cp.w

/-- Source L2 channel-row stride in bytes (`2 * min(alignUp(w, 8), max(T, 32))`). -/
def viewSrcChanStride (cp : ViewCopy) (t : Tiling) : Nat :=
  let stripW := (128 <<< t.fat) >>> t.shPref
  2 * min (alignUp cp.w 8) (max stripW 32)

/-- Source L2 row stride in bytes (`d * c * srcChanStride + 16 * srcSkew`). -/
def viewSrcRowStride (cp : ViewCopy) (t : Tiling) : Nat :=
  cp.d * cp.c * viewSrcChanStride cp t + 16 * t.srcSkew

/-- Result L2 channel-row stride in bytes (staged mode). -/
def viewResChanStride (cp : ViewCopy) (t : Tiling) : Nat :=
  let stripW := (128 <<< t.fat) >>> t.shPref
  let rw := if cp.transposeCW then cp.c else cp.w
  2 * min (alignUp rw 8) (max stripW 32)

/-- Result L2 row stride in bytes (staged mode). -/
def viewResRowStride (cp : ViewCopy) (t : Tiling) : Nat :=
  let rch := if cp.transposeCW then cp.w else cp.c
  rch * viewResChanStride cp t + 16 * t.resSkew

/-- Result L2 footprint in bytes (direct or staged mode). -/
def viewResL2Bytes (target : Target) (cp : ViewCopy) (t : Tiling) : Nat :=
  if viewIsDirect target cp t then
    ((1 : Nat) <<< (t.f + t.ocg + t.fat)) * 512
  else
    min t.tileHeight cp.h * viewResRowStride cp t

/-- Source L2 footprint in bytes. -/
def viewSrcL2Bytes (cp : ViewCopy) (t : Tiling) : Nat :=
  min t.tileHeight cp.h * viewSrcRowStride cp t

private def maxEndRanges (rs : List Range) : Nat :=
  rs.foldl (fun acc r => max acc r.hi) 0

private def viewStageBasesForTh
    (target : Target) (cp : ViewCopy) (probe : Tiling)
    (occupied : List Range) (th : Nat) : Option (Nat × Nat × Nat × Bool) :=
  let cand : Tiling := { probe with tileHeight := th }
  let direct := viewIsDirect target cp cand
  let srcSz := viewSrcL2Bytes cp cand
  let resSz := viewResL2Bytes target cp cand
  let srcAligned := alignUp srcSz dmaLineBytes
  let resAligned := alignUp resSz dmaLineBytes
  match findFreeOffset occupied srcAligned l2CapacityBytes with
  | none => none
  | some sb =>
    let occ2 := { base := sb, bytes := srcAligned } :: occupied
    match findFreeOffset occ2 resAligned l2CapacityBytes with
    | none => none
    | some rb =>
      let peak := max (maxEndRanges occupied) (max (sb + srcSz) (rb + resSz))
      some (sb, rb, peak, direct)

/-- Budget-aware view tiling chooser (supports both standalone `occupied = []` and multi-task graphs). -/
def chooseViewTaskPlan
    (target : Target) (cp : ViewCopy) (occupied : List Range := []) : Except String ViewTaskPlan := do
  if cp.n == 0 || cp.c == 0 || cp.h == 0 || cp.w == 0 || cp.d == 0 then
    throw s!"view copy: dimensions (N={cp.n}, C={cp.c}, H={cp.h}, D={cp.d}, W={cp.w}) must all be >= 1"
  let ocg := if cp.transposeCW then groupCodeLog2 cp.c else 0
  let fats : Array Nat := #[0, 1]
  -- Prefer direct=true first, then fallback to staged mode if needed
  for wantDirect in #[true, false] do
    for fat in fats do
      let logNE := fat
      for shPref in [:5] do
        let stripW := (128 <<< fat) >>> shPref
        if stripW >= 8 && stripW <= 256 then
          let probe : Tiling := {
            logNE := logNE
            f := 0
            fat := fat
            ocg := ocg
            shPref := shPref
            shMax := shPref
            tileHeight := 1
          }
          if viewIsDirect target cp probe == wantDirect then
            if let some (sb, rb, pk, dir) := viewStageBasesForTh target cp probe occupied cp.h then
              return {
                tiling := { probe with tileHeight := cp.h }
                direct := dir
                srcStageBase := sb
                resStageBase := rb
                l2TaskPeak := pk
              }
            if cp.h > 1 then
              let mut lo : Nat := 1
              let mut hi : Nat := cp.h - 1
              let mut best : Option (Nat × Nat × Nat × Nat × Bool) := none
              while lo <= hi do
                let mid := (lo + hi) / 2
                match viewStageBasesForTh target cp probe occupied mid with
                | some (sb, rb, pk, dir) =>
                  best := some (mid, sb, rb, pk, dir)
                  lo := mid + 1
                | none =>
                  if mid == 0 then break
                  hi := mid - 1
              if let some (th, sb, rb, pk, dir) := best then
                return {
                  tiling := { probe with tileHeight := th }
                  direct := dir
                  srcStageBase := sb
                  resStageBase := rb
                  l2TaskPeak := pk
                }
  throw s!"no legal {target.name} NE view tiling fits 2 MiB L2 for (N={cp.n}, C={cp.c}, H={cp.h}, D={cp.d}, W={cp.w}, transposeCW={cp.transposeCW})"

/-- Choose a legal standalone `Tiling` for `ViewCopy` within 2 MiB L2 SRAM. -/
def chooseViewTiling (target : Target) (cp : ViewCopy) : Except String Tiling := do
  let tp ← chooseViewTaskPlan target cp []
  pure tp.tiling

/-- Lower one `ViewCopy` task into a multi-task `TaskDesc` with explicit width-checked registers
    strictly within `ane_hx_dma_modelled`. -/
def buildViewTaskMulti
    (target : Target)
    (cp : ViewCopy)
    (vp : ViewTaskPlan)
    (taskId totalTasks : Nat)
    (inBarSlot outBarSlot : UInt8)
    (outIsScratch : Bool)
    (sync : TaskSyncInfo)
    (liveL2Mask : UInt32 := 0) : Except String TaskDesc := do
  let isFirst := taskId == 0
  let isLast := taskId + 1 == totalTasks
  let t := vp.tiling
  let code := if cp.transposeCW then groupCode cp.c else 1
  let wideGroup : UInt32 := if cp.transposeCW && code >= 2 then 1 else 0

  let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked cp.w "Win")]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked cp.h "Hin")]
  let cU32 ← natToU32Checked cp.c "Cin"
  let wBatchN ← encodeWord (dim16Layout "BatchN") #[("val", ← natToU32Checked cp.n "BatchN")]
  let wDw ← encodeWord dwGroupsLayout #[("perGroupIn", ← natToU32Checked code "groupCode")]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked t.tileHeight "TH")]
  let wNeCommon ← encodeWord neViewCommonLayout #[
    ("kind", 0),
    ("shPref", ← natToU32Checked t.shPref "shPref"),
    ("shMin", ← natToU32Checked t.shPref "shMin"),
    ("shMax", ← natToU32Checked t.shMax "shMax"),
    ("F", ← natToU32Checked t.f "F"),
    ("transposeOut", if cp.transposeCW then 1 else 0),
    ("transposeWideGroup", wideGroup)
  ]
  let wNeTile ← encodeWord neTileLayout #[
    ("ocg", ← natToU32Checked t.ocg "ocg"),
    ("fat", ← natToU32Checked t.fat "fat"),
    ("WU", ← natToU32Checked t.wu "WU")
  ]
  let wSrc1Cfg ← encodeWord (portCfgMultiLayout "src1Cfg") #[
    ("mode", portModeDma),
    ("handoffNe", 0),
    ("handoffPe", 0),
    ("waitNe4", if isFirst then 1 else 0),
    ("waitNe5", if isFirst || sync.src1WaitNe5 then 1 else 0),
    ("dmaFlag", 1),
    ("P", 1),
    ("waitPe20", if isFirst then 1 else 0),
    ("waitPe22", if isFirst || sync.src1WaitPe22 then 1 else 0)
  ]
  let wResCfg ← encodeWord resCfgMultiLayout #[
    ("mode", portModeDma),
    ("direct", if vp.direct then 1 else 0),
    ("waitNe4", if isFirst then 1 else 0),
    ("waitNe5", if isFirst then 1 else 0),
    ("dmaFlag", 1),
    ("P", 1),
    ("waitPe20", if isFirst then 1 else 0),
    ("waitPe22", if isFirst then 1 else 0)
  ]

  let srcChs := viewSrcChanStride cp t
  let u := cp.c * srcChs
  let srcRow := viewSrcRowStride cp t

  let mut plain : Array RegWrite := #[
    { byteAddr := regWin, value := wWin },
    { byteAddr := regHin, value := wHin },
    { byteAddr := regCin, value := cU32 },
    { byteAddr := regWout, value := wWin },
    { byteAddr := regHout, value := wHin },
    { byteAddr := regCout, value := cU32 },
    { byteAddr := regBatchN, value := wBatchN },
    { byteAddr := regDwGroups, value := wDw },
    { byteAddr := regTH, value := wTH },
    { byteAddr := regNeCommon, value := wNeCommon },
    { byteAddr := regNeTile, value := wNeTile },
    { byteAddr := regTargetCfg, value := targetCfgValue target },
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
    { byteAddr := regSrc1Base, value := ← natToU32Checked vp.srcStageBase "src1Base" },
    { byteAddr := regSrc1ChanStride, value := ← natToU32Checked srcChs "src1ChanStride" },
    { byteAddr := regSrc1RowStride, value := ← natToU32Checked srcRow "src1RowStride" },
    { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked u "src1RowBytesA" },
    { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked (cp.d * u) "src1RowBytesB" },
    { byteAddr := regResCfg, value := wResCfg },
    { byteAddr := regResBase, value := ← natToU32Checked vp.resStageBase "resBase" },
    { byteAddr := regNeOp, value := neOpViewCopy },
    { byteAddr := regNeScale, value := neScaleOne },
    { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
    { byteAddr := regSrc1Fmt, value := srcFmtWordP1 },
    { byteAddr := regResFmt, value := resFmtWordP1 }
  ]

  if cp.d > 1 then
    let wD ← encodeWord (dim16Layout "Din") #[("val", ← natToU32Checked cp.d "Din")]
    plain := plain ++ #[
      { byteAddr := regDin, value := wD },
      { byteAddr := regDout, value := wD }
    ]

  if isFirst then
    plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }

  if !vp.direct then
    let resChs := viewResChanStride cp t
    let rch := if cp.transposeCW then cp.w else cp.c
    let resRowBytes := rch * resChs
    let resRow := viewResRowStride cp t
    plain := plain ++ #[
      { byteAddr := regResChanStride, value := ← natToU32Checked resChs "resChanStride" },
      { byteAddr := regResRowStride, value := ← natToU32Checked resRow "resRowStride" },
      { byteAddr := regResRowBytesA, value := ← natToU32Checked resRowBytes "resRowBytesA" },
      { byteAddr := regResRowBytesB, value := ← natToU32Checked resRowBytes "resRowBytesB" }
    ]

  -- Input DMA strides
  if cp.h > 1 then
    plain := plain.push { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked cp.inRowStride "src1DmaRowStride" }
  if cp.c > 1 then
    plain := plain.push { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked cp.inPlaneStride "src1DmaPlaneStride" }
  if cp.d > 1 then
    plain := plain.push { byteAddr := regSrc1DmaDStride, value := ← natToU32Checked cp.inDStride "src1DmaDStride" }
  if cp.n > 1 then
    plain := plain.push { byteAddr := regSrc1DmaNStride, value := ← natToU32Checked cp.inNStride "src1DmaNStride" }

  -- Output DMA strides
  let resPlaneDim := if cp.transposeCW then cp.w else cp.c
  if cp.h > 1 then
    plain := plain.push { byteAddr := regResDmaRowStride, value := ← natToU32Checked cp.outRowStride "resDmaRowStride" }
  if resPlaneDim > 1 then
    plain := plain.push { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked cp.outPlaneStride "resDmaPlaneStride" }
  if cp.d > 1 then
    plain := plain.push { byteAddr := regResDmaDStride, value := ← natToU32Checked cp.outDStride "resDmaDStride" }
  if cp.n > 1 then
    plain := plain.push { byteAddr := regResDmaNStride, value := ← natToU32Checked cp.outNStride "resDmaNStride" }

  let taskIdU32 ← natToU32Checked taskId "taskId"
  if isLast then
    let wOut2 ← encodeWord out2Layout #[("form", out2FormValue), ("task", taskIdU32)]
    plain := plain ++ #[
      { byteAddr := regKernelStage, value := kernelStageNoKernValue },
      { byteAddr := regOut0, value := out0Value },
      { byteAddr := regOut1, value := out1Value },
      { byteAddr := regOut2, value := wOut2 },
      { byteAddr := regOut3, value := out3Value },
      { byteAddr := regOut4, value := out4Value },
      { byteAddr := regOut5, value := out5Value }
    ]

  let wSrc1Dep ← encodeWord (srcDepMultiLayout "src1Dep") #[
    ("code", srcDepCodeDma),
    ("task", taskIdU32),
    ("depDist", ← natToU32Checked sync.src1DepDist "src1DepDist"),
    ("depKind", ← natToU32Checked sync.src1DepKind "src1DepKind")
  ]
  let resDepC := if outIsScratch then resDepCode target ||| resDepScratchFlags else resDepCode target
  let wResDep ← encodeWord resDepLayout #[
    ("code", resDepC),
    ("task", taskIdU32),
    ("direct", if vp.direct then 1 else 0)
  ]
  let inOffLo ← natToU32Checked (cp.inOff &&& 0xffffffff) "src1BaseLo"
  let inOffHi ← natToU32Checked (cp.inOff >>> 32) "src1BaseHi"
  let outOffLo ← natToU32Checked (cp.outOff &&& 0xffffffff) "resBaseLo"
  let outOffHi ← natToU32Checked (cp.outOff >>> 32) "resBaseHi"

  let barRecs : Array BarRegWrite := #[
    { baseByteAddr := regSrc1BaseLo, barSlot := inBarSlot, values := #[inOffLo, inOffHi] },
    { baseByteAddr := regResBaseLo, barSlot := outBarSlot, values := #[outOffLo, outOffHi] },
    { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrc1Dep] },
    { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
  ]

  pure {
    taskId := taskId
    logNE := t.logNE
    eneOverride := some (t.logNE + 1)
    isFirst := isFirst
    isLast := isLast
    liveL2Mask := liveL2Mask
    plainWrites := plain
    barWrites := barRecs
    kernelRelocs := #[]
  }

/-- Compile a standalone `ViewPlan` (`transpose`, `slice`, `concat`, `split`, `reshape`) into
    a statically verified HWX container and its 4 Linux LOAD sections. -/
def compileView (target : Target) (plan : ViewPlan) : Except String CompiledView := do
  if plan.inShapes.isEmpty || plan.inShapes.size > 3 then
    throw s!"compileView ({plan.opName}): input surface count {plan.inShapes.size} must be in 1..3"
  if plan.copies.isEmpty || plan.copies.size > 256 then
    throw s!"compileView ({plan.opName}): task count {plan.copies.size} must be in 1..256"

  let nTasks := plan.copies.size
  let outBarSlot : UInt8 := barSlotFirstSurface + plan.inShapes.size.toUInt8

  let mut taskPlans : Array ViewTaskPlan := Array.mkEmpty nTasks
  let mut taskDescs : Array TaskDesc := Array.mkEmpty nTasks
  let mut peakL2 : Nat := 0

  for i in [:nTasks] do
    let cp := plan.copies[i]!
    if cp.inBuf >= plan.inShapes.size then
      throw s!"compileView ({plan.opName}): copy[{i}] inBuf={cp.inBuf} >= {plan.inShapes.size}"
    let vp ← chooseViewTaskPlan target cp []
    peakL2 := max peakL2 vp.l2TaskPeak
    taskPlans := taskPlans.push vp
    let inBarSlot : UInt8 := barSlotFirstSurface + cp.inBuf.toUInt8
    -- On multi-task views (concat), waitNe5 orders both L2 srcStageAhead and any interleaved H/W DRAM writes.
    let sync : TaskSyncInfo := {
      src1WaitNe5 := i > 0
      hasSrc1Dma := true
    }
    let td ← buildViewTaskMulti target cp vp i nTasks inBarSlot outBarSlot false sync 0
    taskDescs := taskDescs.push td

  let mut surfaces : Array SurfaceSpec := Array.mkEmpty (plan.inShapes.size + 1)
  for i in [:plan.inShapes.size] do
    let s := plan.inShapes[i]!
    let sn := if plan.inShapes.size == 1 then "x" else s!"in{i}"
    let sym := if plan.inShapes.size == 1 then "input_x" else s!"input_{i}"
    surfaces := surfaces.push (← makeFp16Surface target sn sym .input s.c s.h s.w s.n)
  surfaces := surfaces.push (← makeFp16Surface target "y" "output_y" .output
    plan.outShape.c plan.outShape.h plan.outShape.w plan.outShape.n)

  let emitted ← emitProgram {
    target := target
    surfaces := surfaces
    tasks := taskDescs
    kernel := none
    scratch := none
    constants := zeroBytes textConstBytes
    ident := some "vane"
  }

  let summary ← verifyHwxTasksSummary emitted.hwx emitted.info (requireKnownTask := true)
  if summary.neView != nTasks || summary.total != nTasks then
    throw s!"internal error: verifier checked {summary.neView}/{summary.total} NE view tasks != {nTasks}"

  pure {
    target := target
    plan := plan
    taskPlans := taskPlans
    l2PeakBytes := peakL2
    hwx := emitted.hwx
    info := emitted.info
    sections := emitted.sections
  }

end Vane.View
