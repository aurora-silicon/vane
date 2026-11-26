import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register
import Vane.Alloc
import Vane.Tiling
import Vane.Program
import Vane.Verify
import Vane.Pe

namespace Vane.Pool

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Alloc
open Vane.Tiling
open Vane.Program
open Vane.Verify
open Vane.Pe

-- ============================================================================
-- Pooling & Reduction Specifications
-- ============================================================================

/-- 2D spatial pooling operation kind (`max` or `avg`). -/
inductive PoolKind where
  | max
  | avg
  deriving Repr, BEq, DecidableEq, Inhabited

namespace PoolKind

def name : PoolKind → String
  | .max => "max"
  | .avg => "avg"

def opName : PoolKind → String
  | .max => "max_pool"
  | .avg => "avg_pool"

def parse? (s : String) : Option PoolKind :=
  match s.toLower with
  | "max" | "max_pool" => some .max
  | "avg" | "avg_pool" => some .avg
  | _ => none

end PoolKind

/-- Padding mode for 2D spatial pooling (`valid`, `same`, or `custom pt pb pl pr`). -/
inductive PoolPad where
  | valid
  | same
  | custom (pt pb pl pr : Nat)
  deriving Repr, BEq, DecidableEq, Inhabited

namespace PoolPad

def name : PoolPad → String
  | .valid => "valid"
  | .same => "same"
  | .custom pt pb pl pr => s!"custom({pt},{pb},{pl},{pr})"

end PoolPad

/-- Single-batch fp16 2D pooling specification. -/
structure PoolSpec where
  target : Target
  op : PoolKind
  dims : TensorDims
  kh : Nat
  kw : Nat
  sh : Nat := 1
  sw : Nat := 1
  pad : PoolPad := .valid
  excludePad : Bool := true
  deriving Repr, BEq, DecidableEq

/-- Resolved per-axis pooling geometry (`out`, `padBefore`, `padAfter`, `used`, `sEff`). -/
structure PoolAxisGeom where
  inp : Nat
  k : Nat
  s : Nat
  out : Nat
  padBefore : Nat
  padAfter : Nat
  used : Nat
  sEff : Nat
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Selected hardware backend for a `PoolSpec`. -/
inductive PoolBackend where
  | pePool
  | nePool
  | peReduceGlobal
  deriving Repr, BEq, DecidableEq, Inhabited

namespace PoolBackend

def name : PoolBackend → String
  | .pePool => "pe_pool"
  | .nePool => "ne_pool"
  | .peReduceGlobal => "pe_reduce"

def engine : PoolBackend → Engine
  | .pePool => .pe
  | .nePool => .ne
  | .peReduceGlobal => .pe

end PoolBackend

/-- All 10 supported tensor reduction operations. -/
inductive ReduceOp where
  | sum
  | mean
  | max
  | min
  | l1
  | sumSquare
  | l2
  | invSum
  | rsqrtSumSquare
  | rsqrtMeanSquare
  deriving Repr, BEq, DecidableEq, Inhabited

namespace ReduceOp

def name : ReduceOp → String
  | .sum => "sum"
  | .mean => "mean"
  | .max => "max"
  | .min => "min"
  | .l1 => "l1"
  | .sumSquare => "sum_square"
  | .l2 => "l2"
  | .invSum => "inv_sum"
  | .rsqrtSumSquare => "rsqrt_sum_square"
  | .rsqrtMeanSquare => "rsqrt_mean_square"

def parse? (s : String) : Option ReduceOp :=
  match s.toLower with
  | "sum" => some .sum
  | "mean" => some .mean
  | "max" => some .max
  | "min" => some .min
  | "l1" => some .l1
  | "sum_square" => some .sumSquare
  | "l2" => some .l2
  | "inv_sum" => some .invSum
  | "rsqrt_sum_square" => some .rsqrtSumSquare
  | "rsqrt_mean_square" => some .rsqrtMeanSquare
  | _ => none

def isNeChannelSupported : ReduceOp → Bool
  | .max | .min | .sum | .mean => true
  | _ => false

def usesEps : ReduceOp → Bool
  | .rsqrtSumSquare | .rsqrtMeanSquare => true
  | _ => false

def usesScale : ReduceOp → Bool
  | .mean | .rsqrtMeanSquare => true
  | _ => false

/-- Encode `0x4500` reduction opcode word via `peReduceOpLayout`. -/
def peOpWord (op : ReduceOp) : Except String UInt32 :=
  let (reducer, preOp, postOp) : UInt32 × UInt32 × UInt32 := match op with
    | .sum => (0, 0, 0)
    | .mean => (0, 0, 0)
    | .max => (2, 0, 0)
    | .min => (3, 0, 0)
    | .l1 => (0, 0x1c, 0)
    | .sumSquare => (0, 1, 0)
    | .l2 => (0, 1, 3)
    | .invSum => (0, 0, 1)
    | .rsqrtSumSquare => (0, 1, 2)
    | .rsqrtMeanSquare => (0, 1, 2)
  encodeWord peReduceOpLayout #[
    ("reducer", reducer),
    ("preOp", preOp),
    ("postOp", postOp)
  ]

def cmpFlag? : ReduceOp → Option UInt32
  | .max => some cmpFlagMaxValue
  | .min => some cmpFlagMinValue
  | _ => none

end ReduceOp

/-- Logical axes reduced on a `[C, H, W]` tensor (`keep_dims = true`). -/
structure ReduceAxes where
  reduceC : Bool := false
  reduceH : Bool := false
  reduceW : Bool := false
  deriving Repr, BEq, DecidableEq, Inhabited

namespace ReduceAxes

def isEmpty (a : ReduceAxes) : Bool :=
  !a.reduceC && !a.reduceH && !a.reduceW

def name (a : ReduceAxes) : String :=
  (if a.reduceC then "c" else "") ++
  (if a.reduceH then "h" else "") ++
  (if a.reduceW then "w" else "")

def parse? (s : String) : Option ReduceAxes :=
  match s.toLower with
  | "w" => some { reduceW := true }
  | "h" => some { reduceH := true }
  | "c" => some { reduceC := true }
  | "hw" | "wh" => some { reduceH := true, reduceW := true }
  | "cw" | "wc" => some { reduceC := true, reduceW := true }
  | "ch" | "hc" => some { reduceC := true, reduceH := true }
  | "chw" | "cwh" | "hcw" | "hwc" | "wch" | "whc" =>
    some { reduceC := true, reduceH := true, reduceW := true }
  | _ => none

end ReduceAxes

/-- Single-batch fp16 tensor reduction specification. -/
structure ReduceSpec where
  target : Target
  op : ReduceOp
  dims : TensorDims
  axes : ReduceAxes
  eps : UInt16 := 0
  deriving Repr, BEq, DecidableEq

/-- Selected hardware backend for a `ReduceSpec`. -/
inductive ReduceBackend where
  | peReduce
  | neChannelReduce
  deriving Repr, BEq, DecidableEq, Inhabited

namespace ReduceBackend

def name : ReduceBackend → String
  | .peReduce => "pe_reduce"
  | .neChannelReduce => "ne_channel_reduce"

def engine : ReduceBackend → Engine
  | .peReduce => .pe
  | .neChannelReduce => .ne

end ReduceBackend

-- ============================================================================
-- Common Window Geometry & 4D PE Axis Remap (pure, kernel-reducible for proofs)
-- ============================================================================

/-- Resolve and validate per-axis 2D pooling geometry. -/
def poolAxisGeom (inp k s : Nat) (pad : PoolPad) (isHeight : Bool) : Except String PoolAxisGeom := do
  let axisLabel := if isHeight then "H" else "W"
  if inp == 0 || inp > 65535 then
    throw s!"pool {axisLabel}: input extent {inp} must be in 1..65535"
  if k == 0 then
    throw s!"pool {axisLabel}: kernel size must be >= 1"
  if s != 1 && s != 2 then
    throw s!"pool {axisLabel}: stride {s} must be 1 or 2"
  let (out, padBefore, padAfter) ← match pad with
    | .same =>
      let o := ceilDiv inp s
      let padTotal := (o - 1) * s + k - inp
      let pb := padTotal / 2
      let pa := padTotal - pb
      pure (o, pb, pa)
    | .valid =>
      if inp < k then
        throw s!"pool {axisLabel} (valid): input extent {inp} < kernel size {k}"
      let o := (inp - k) / s + 1
      pure (o, 0, 0)
    | .custom pt pb pl pr =>
      let pBefore := if isHeight then pt else pl
      let pAfter := if isHeight then pb else pr
      let maxPad := if isHeight then 63 else 31
      if pBefore > maxPad then
        throw s!"pool {axisLabel} (custom): padBefore={pBefore} exceeds hardware max {maxPad}"
      if pBefore >= k then
        throw s!"pool {axisLabel} (custom): padBefore={pBefore} must be < kernel size {k}"
      if pAfter >= k then
        throw s!"pool {axisLabel} (custom): padAfter={pAfter} must be < kernel size {k}"
      if inp + pBefore + pAfter < k then
        throw s!"pool {axisLabel} (custom): inp({inp}) + padBefore({pBefore}) + padAfter({pAfter}) < k({k})"
      let o := (inp + pBefore + pAfter - k) / s + 1
      if (o - 1) * s >= inp + pBefore then
        throw s!"pool {axisLabel} (custom): last output window does not overlap any input pixel"
      pure (o, pBefore, pAfter)
  let maxPad := if isHeight then 63 else 31
  if padBefore > maxPad then
    throw s!"pool {axisLabel}: padBefore={padBefore} exceeds hardware max {maxPad}"
  if padBefore >= k then
    throw s!"pool {axisLabel}: padBefore={padBefore} >= kernel size {k}"
  let used := min inp ((out - 1) * s + k - padBefore)
  let sEff := if out == 1 && k == inp && inp > 1 && inp + padBefore + padAfter <= s then 1 else s
  pure {
    inp := inp
    k := k
    s := s
    out := out
    padBefore := padBefore
    padAfter := padAfter
    used := used
    sEff := sEff
  }

/-- Input rows buffered for output row tile `i` of a 2D pooling task. -/
def poolTileRows (usedH kh sh pt oh th : Nat) (isNe : Bool) (i : Nat) : Nat :=
  let o0 := i * th
  let o1 := min oh (o0 + th)
  if o1 <= o0 then 0
  else
    let ke := if isNe then alignUp kh sh else max kh sh
    let lo := o0 * sh - pt
    let hi := min usedH ((o1 - 1) * sh + ke - pt)
    hi - lo

/-- Recursive maximum of `poolTileRows` over tiles `0 .. n - 1` (kernel-reducible for formal proofs). -/
def poolInRowsAux (usedH kh sh pt oh th : Nat) (isNe : Bool) : Nat → Nat
  | 0 => 0
  | n + 1 => max (poolInRowsAux usedH kh sh pt oh th isNe n)
                 (poolTileRows usedH kh sh pt oh th isNe n)

/-- Maximum input rows buffered per row tile of height `th` (`1 <= th <= oh`). -/
def poolInRows (usedH kh sh pt oh th : Nat) (isNe : Bool) : Nat :=
  if th == 0 || oh == 0 then 0
  else poolInRowsAux usedH kh sh pt oh th isNe (ceilDiv oh th)

/-- L2 source channel-row pixels per strip for 2D window pooling. -/
def poolStripPx (usedW kw sw pl stripW ow : Nat) : Nat :=
  let r := alignUp usedW 8
  if ow <= stripW then r
  else
    let tin := stripW * sw
    let left := pl > 0
    let right := kw - pl > sw
    let extraL := if left then 8 else 0
    let l := max 24 (min 32 (r - 2 * tin))
    let extraR := if right then max l (32 - extraL) else 0
    min r (tin + extraL + extraR)

/-- Direct-mode condition for PE window pooling (`0x003c = 0x10`). -/
def pePoolDirect (target : Target) (ow k : Nat) : Bool :=
  let stripW := (1 : Nat) <<< k
  let multi := ow > stripW
  match target with
  | .h17 => !multi || stripW >= 32
  | .h16 => !multi || stripW >= 64

/-- Logical source axis mapped onto one of the 4 PE reduction dimensions `(N, C, H, D)`. -/
inductive PeRedLogAxis where
  | none
  | logC
  | logH
  deriving Repr, BEq, DecidableEq, Inhabited

/-- 4D PE reduction axis remap from logical `[C, H, W]` with flags `(rc, rh)` (`W` reduced). -/
structure PeReduceAxisMap where
  sN : Nat
  sC : Nat
  sH : Nat
  sD : Nat
  axN : PeRedLogAxis
  axC : PeRedLogAxis
  axH : PeRedLogAxis
  axD : PeRedLogAxis
  outC : Nat
  outH : Nat
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Pure 4D PE reduction axis mapping preserving `sN * sC * sH * sD = c * h`. -/
def peReduceAxes (c h : Nat) (rc rh : Bool) : PeReduceAxisMap :=
  match rc, rh with
  | false, false =>
    if c == 1 then
      { sN := 1, sC := h, sH := 1, sD := 1,
        axN := .none, axC := .logH, axH := .none, axD := .none,
        outC := 1, outH := h }
    else
      { sN := h, sC := c, sH := 1, sD := 1,
        axN := .logH, axC := .logC, axH := .none, axD := .none,
        outC := c, outH := h }
  | false, true =>
    { sN := 1, sC := c, sH := h, sD := 1,
      axN := .none, axC := .logC, axH := .logH, axD := .none,
      outC := c, outH := 1 }
  | true, false =>
    { sN := 1, sC := h, sH := c, sD := 1,
      axN := .none, axC := .logH, axH := .logC, axD := .none,
      outC := 1, outH := h }
  | true, true =>
    if c > h then
      { sN := 1, sC := 1, sH := c, sD := h,
        axN := .none, axC := .none, axH := .logC, axD := .logH,
        outC := 1, outH := 1 }
    else
      { sN := 1, sC := 1, sH := h, sD := c,
        axN := .none, axC := .none, axH := .logH, axD := .logC,
        outC := 1, outH := 1 }

/-- Resolve backend and validate geometry for a `PoolSpec`. -/
def resolvePoolBackend (spec : PoolSpec) : Except String (PoolBackend × PoolAxisGeom × PoolAxisGeom) := do
  validateTensorDims spec.dims "pool input"
  let gh ← poolAxisGeom spec.dims.h spec.kh spec.sh spec.pad true
  let gw ← poolAxisGeom spec.dims.w spec.kw spec.sw spec.pad false
  let zeroPad := gh.padBefore == 0 && gh.padAfter == 0 && gw.padBefore == 0 && gw.padAfter == 0
  let isGlobal := spec.kh == spec.dims.h && spec.kw == spec.dims.w && zeroPad && gh.out == 1 && gw.out == 1
  if isGlobal && (spec.kh > 5 || spec.kw > 5) then
    return (.peReduceGlobal, gh, gw)
  if spec.kh <= 5 && spec.kw <= 5 then
    return (.pePool, gh, gw)
  if spec.kh > 15 || spec.kw > 15 then
    throw s!"pool kernel ({spec.kh}x{spec.kw}) exceeds weightless NE maximum 15x15 (and is not a global valid pool)"
  if spec.op == .avg && spec.excludePad && !zeroPad then
    throw s!"weightless NE avg_pool ({spec.kh}x{spec.kw}) does not support exclude_pad=true with non-zero padding"
  if spec.op == .max && spec.sh == 1 && spec.sw == 1 && max spec.kh spec.kw >= 9 then
    throw s!"weightless NE stride-1 max_pool does not support max(kh, kw)={max spec.kh spec.kw} >= 9"
  if spec.op == .avg && spec.sh == 1 && spec.sw == 1 && max spec.kh spec.kw >= 9 && max spec.kh spec.kw % 2 == 1 then
    throw s!"weightless NE stride-1 avg_pool does not support odd max(kh, kw)={max spec.kh spec.kw} >= 9"
  pure (.nePool, gh, gw)

/-- Validate `ReduceSpec` and select hardware backend (`peReduce` or `neChannelReduce`) and output shape. -/
def resolveReduceBackend (spec : ReduceSpec) : Except String (ReduceBackend × TensorDims) := do
  validateTensorDims spec.dims "reduce input"
  if spec.axes.isEmpty then
    throw "reduce: at least one axis (c, h, or w) must be reduced"
  if spec.op.usesEps then
    if !f16IsFinite spec.eps then
      throw s!"reduce {spec.op.name}: eps must be finite fp16"
    if (spec.eps &&& 0x8000) != 0 && (spec.eps &&& 0x7fff) != 0 then
      throw s!"reduce {spec.op.name}: eps must be >= 0"
  let outDims : TensorDims := {
    c := if spec.axes.reduceC then 1 else spec.dims.c
    h := if spec.axes.reduceH then 1 else spec.dims.h
    w := if spec.axes.reduceW then 1 else spec.dims.w
  }
  if spec.axes.reduceW || spec.dims.w == 1 then
    pure (.peReduce, outDims)
  else if spec.axes.reduceC && !spec.axes.reduceH && !spec.axes.reduceW then
    if spec.op.isNeChannelSupported then
      pure (.neChannelReduce, outDims)
    else
      throw s!"reduce {spec.op.name} over axis C (with W={spec.dims.w} > 1) is unsupported on weightless NE (supported: max, min, sum, mean)"
  else
    throw s!"reduce over axes '{spec.axes.name}' with W={spec.dims.w} > 1 requires a future transpose lowering"

-- ============================================================================
-- Tiling Structures & O(1) Deterministic Tile Choosers
-- ============================================================================

/-- Tiling plan for `pePool` (`0x003c = 0x10`). -/
structure PePoolPlan where
  k : Nat
  tileHeight : Nat
  direct : Bool
  srcStageBase : Nat
  resStageBase : Nat
  l2TaskPeak : Nat
  deriving Repr, BEq, Inhabited

/-- Tiling plan for `peReduce` (`0x003c = 0x50`). -/
structure PeReducePlan where
  k : Nat
  tileHeight : Nat
  direct : Bool
  srcStageBase : Nat
  resStageBase : Nat
  l2TaskPeak : Nat
  deriving Repr, BEq, Inhabited

/-- Tiling plan for weightless `nePool` or `neChannelReduce`. -/
structure NeWeightlessPlan where
  logNE : Nat := 4
  f : Nat := 4
  fat : Nat := 0
  shPref : Nat
  shMax : Nat
  tileHeight : Nat
  direct : Bool
  srcStageBase : Nat
  resStageBase : Nat
  l2TaskPeak : Nat
  deriving Repr, BEq, Inhabited

def NeWeightlessPlan.wu (p : NeWeightlessPlan) : Nat :=
  p.logNE - p.f - p.fat

def NeWeightlessPlan.stripWidth (p : NeWeightlessPlan) : Nat :=
  (128 <<< p.fat) >>> p.shPref

private def maxEndOf (rs : List Range) : Nat :=
  rs.foldl (fun acc r => max acc r.hi) 0

private def placeTwoTemps (occupied : List Range) (srcSz resSz : Nat) (srcIsL2 resIsL2 : Bool) :
    Option (Nat × Nat × Nat) := Id.run do
  let mut occ := occupied
  let mut sb := 0
  let mut rb := 0
  let mut peak := maxEndOf occupied
  if !srcIsL2 then
    let sz := alignUp srcSz dmaLineBytes
    match findFreeOffset occ sz l2CapacityBytes with
    | none => return none
    | some off =>
      sb := off
      peak := max peak (off + sz)
      occ := { base := off, bytes := sz } :: occ
  if !resIsL2 then
    let sz := alignUp resSz dmaLineBytes
    match findFreeOffset occ sz l2CapacityBytes with
    | none => return none
    | some off =>
      rb := off
      peak := max peak (off + sz)
  some (sb, rb, peak)

/-- Budget-aware O(1) tile chooser for `pePool` respecting `occupied` live L2 ranges. -/
def choosePePoolTilingForL2
    (target : Target) (c : Nat) (gh gw : PoolAxisGeom)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) : Except String PePoolPlan := do
  let owAligned := alignUp gw.out 8
  let mut best? : Option (PePoolPlan × Nat × Nat × Nat) := none

  for step in [:5] do
    let k := 9 - step
    let stripW := (1 : Nat) <<< k
    let direct := if resIsL2 then false else pePoolDirect target gw.out k
    let px := poolStripPx gw.used gw.k gw.sEff gw.padBefore stripW gw.out
    let row := 2 * px
    let u := c * row
    let drow := 2 * min owAligned (max stripW 32)
    let du := c * drow
    let tryTh := fun (th : Nat) =>
      let srcSz := if srcIsL2 then 0 else alignUp (poolInRows gh.used gh.k gh.sEff gh.padBefore gh.out th false * u) dmaLineBytes
      let resSz :=
        if resIsL2 then 0
        else if direct then peDirectResBytes c drow th
        else alignUp (th * du) dmaLineBytes
      placeTwoTemps occupied srcSz resSz srcIsL2 resIsL2

    let found? : Option (Nat × Nat × Nat × Nat) := Id.run do
      if let some (sb, rb, pk) := tryTh gh.out then
        return some (gh.out, sb, rb, pk)
      if gh.out <= 1 then
        return none
      let mut lo := 1
      let mut hi := gh.out - 1
      let mut b : Option (Nat × Nat × Nat × Nat) := none
      while lo <= hi do
        let mid := (lo + hi) / 2
        match tryTh mid with
        | some (sb, rb, pk) =>
          b := some (mid, sb, rb, pk)
          lo := mid + 1
        | none =>
          if mid == 0 then break
          hi := mid - 1
      return b

    if let some (th, sb, rb, pk) := found? then
      let strips := ceilDiv owAligned stripW * ceilDiv gh.out th
      let area := stripW * th
      let plan : PePoolPlan := {
        k := k, tileHeight := th, direct := direct,
        srcStageBase := sb, resStageBase := rb, l2TaskPeak := pk
      }
      let isBetter := match best? with
        | none => true
        | some (bp, bStrips, bArea, bStripW) =>
          if strips != bStrips then strips < bStrips
          else if plan.direct != bp.direct then plan.direct
          else if area != bArea then area > bArea
          else stripW < bStripW
      if isBetter then
        best? := some (plan, strips, area, stripW)

  match best? with
  | some (winner, _, _, _) => pure winner
  | none => throw s!"no legal {target.name} PE pool tiling fits available L2 budget (C={c}, out={gh.out}x{gw.out})"

/-- Standalone O(1) tile chooser for `pePool`. -/
def choosePePoolTiling (target : Target) (c : Nat) (gh gw : PoolAxisGeom) : Except String PePoolPlan :=
  choosePePoolTilingForL2 target c gh gw false false []

/-- Budget-aware O(1) tile chooser for `peReduce` respecting `occupied` live L2 ranges. -/
def choosePeReduceTilingForL2
    (target : Target) (w : Nat) (ax : PeReduceAxisMap)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) : Except String PeReducePlan := do
  let wAligned := alignUp w 8
  let resSz := if resIsL2 then 0 else 2048
  if srcIsL2 then
    let k := min 9 (max 5 (ceilLog2 wAligned))
    match placeTwoTemps occupied 0 resSz true resIsL2 with
    | some (_, rb, pk) =>
      return { k := k, tileHeight := ax.sH, direct := !resIsL2, srcStageBase := 0, resStageBase := rb, l2TaskPeak := pk }
    | none =>
      throw s!"no legal {target.name} PE reduce L2 result staging fits available L2 budget"

  let mut best? : Option (PeReducePlan × Nat × Nat × Nat) := none
  for step in [:5] do
    let k := 9 - step
    let stripW := (1 : Nat) <<< k
    let px := min wAligned (max stripW 32)
    let row := 2 * px
    let u := ax.sC * ax.sD * row
    let tryTh := fun (th : Nat) =>
      let srcSz := alignUp (th * u) dmaLineBytes
      placeTwoTemps occupied srcSz resSz false resIsL2

    let found? : Option (Nat × Nat × Nat × Nat) := Id.run do
      if let some (sb, rb, pk) := tryTh ax.sH then
        return some (ax.sH, sb, rb, pk)
      if ax.sH <= 1 then
        return none
      let mut lo := 1
      let mut hi := ax.sH - 1
      let mut b : Option (Nat × Nat × Nat × Nat) := none
      while lo <= hi do
        let mid := (lo + hi) / 2
        match tryTh mid with
        | some (sb, rb, pk) =>
          b := some (mid, sb, rb, pk)
          lo := mid + 1
        | none =>
          if mid == 0 then break
          hi := mid - 1
      return b

    if let some (th, sb, rb, pk) := found? then
      let strips := ceilDiv wAligned stripW * ceilDiv ax.sH th
      let area := stripW * th
      let plan : PeReducePlan := {
        k := k, tileHeight := th, direct := !resIsL2, srcStageBase := sb, resStageBase := rb, l2TaskPeak := pk
      }
      let isBetter := match best? with
        | none => true
        | some (_, bStrips, bArea, bStripW) =>
          if strips != bStrips then strips < bStrips
          else if area != bArea then area > bArea
          else stripW < bStripW
      if isBetter then
        best? := some (plan, strips, area, stripW)

  match best? with
  | some (winner, _, _, _) => pure winner
  | none => throw s!"no legal {target.name} PE reduce tiling fits available L2 budget (sC={ax.sC}, sH={ax.sH}, sD={ax.sD}, W={w})"

/-- Standalone O(1) tile chooser for `peReduce`. -/
def choosePeReduceTiling (target : Target) (w : Nat) (ax : PeReduceAxisMap) : Except String PeReducePlan :=
  choosePeReduceTilingForL2 target w ax false false []

/-- Budget-aware O(1) tile chooser for weightless `nePool` respecting `occupied` live L2 ranges. -/
def chooseNePoolTilingForL2
    (target : Target) (c : Nat) (gh gw : PoolAxisGeom)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) : Except String NeWeightlessPlan := do
  let owAligned := alignUp gw.out 8
  let resSz := if resIsL2 then 0 else ((1 : Nat) <<< 4) * 512
  let mut best? : Option (NeWeightlessPlan × Nat × Nat × Nat) := none

  for shP in [:3] do
    let stripW := 128 >>> shP
    let px := poolStripPx gw.used gw.k gw.sEff gw.padBefore stripW gw.out
    let row := 2 * px
    let u := c * row
    let tryTh := fun (th : Nat) =>
      let srcSz := if srcIsL2 then 0 else alignUp (poolInRows gh.used gh.k gh.sEff gh.padBefore gh.out th true * u) dmaLineBytes
      placeTwoTemps occupied srcSz resSz srcIsL2 resIsL2

    let found? : Option (Nat × Nat × Nat × Nat) := Id.run do
      if let some (sb, rb, pk) := tryTh gh.out then
        return some (gh.out, sb, rb, pk)
      if gh.out <= 1 then
        return none
      let mut lo := 1
      let mut hi := gh.out - 1
      let mut b : Option (Nat × Nat × Nat × Nat) := none
      while lo <= hi do
        let mid := (lo + hi) / 2
        match tryTh mid with
        | some (sb, rb, pk) =>
          b := some (mid, sb, rb, pk)
          lo := mid + 1
        | none =>
          if mid == 0 then break
          hi := mid - 1
      return b

    if let some (th, sb, rb, pk) := found? then
      let strips := ceilDiv owAligned stripW * ceilDiv gh.out th
      let area := stripW * th
      let plan : NeWeightlessPlan := {
        logNE := 4, f := 4, fat := 0, shPref := shP, shMax := shP,
        tileHeight := th, direct := !resIsL2,
        srcStageBase := sb, resStageBase := rb, l2TaskPeak := pk
      }
      let isBetter := match best? with
        | none => true
        | some (_, bStrips, bArea, bStripW) =>
          if strips != bStrips then strips < bStrips
          else if area != bArea then area > bArea
          else stripW < bStripW
      if isBetter then
        best? := some (plan, strips, area, stripW)

  match best? with
  | some (winner, _, _, _) => pure winner
  | none => throw s!"no legal {target.name} NE pool tiling fits available L2 budget (C={c}, out={gh.out}x{gw.out})"

/-- Standalone O(1) tile chooser for weightless `nePool`. -/
def chooseNePoolTiling (target : Target) (c : Nat) (gh gw : PoolAxisGeom) : Except String NeWeightlessPlan :=
  chooseNePoolTilingForL2 target c gh gw false false []

/-- Budget-aware O(1) tile chooser for weightless `neChannelReduce` respecting `occupied` live L2 ranges. -/
def chooseNeChannelReduceTilingForL2
    (target : Target) (dims : TensorDims)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) : Except String NeWeightlessPlan := do
  let wAligned := alignUp dims.w 8
  let resSz := if resIsL2 then 0 else ((1 : Nat) <<< 4) * 512
  let mut best? : Option (NeWeightlessPlan × Nat × Nat × Nat) := none

  for shP in [:3] do
    let stripW := 128 >>> shP
    let px := min wAligned stripW
    let row := 2 * px
    let u := dims.c * row
    let tryTh := fun (th : Nat) =>
      let srcSz := if srcIsL2 then 0 else alignUp (th * u) dmaLineBytes
      placeTwoTemps occupied srcSz resSz srcIsL2 resIsL2

    let found? : Option (Nat × Nat × Nat × Nat) := Id.run do
      if let some (sb, rb, pk) := tryTh dims.h then
        return some (dims.h, sb, rb, pk)
      if dims.h <= 1 then
        return none
      let mut lo := 1
      let mut hi := dims.h - 1
      let mut b : Option (Nat × Nat × Nat × Nat) := none
      while lo <= hi do
        let mid := (lo + hi) / 2
        match tryTh mid with
        | some (sb, rb, pk) =>
          b := some (mid, sb, rb, pk)
          lo := mid + 1
        | none =>
          if mid == 0 then break
          hi := mid - 1
      return b

    if let some (th, sb, rb, pk) := found? then
      let strips := ceilDiv wAligned stripW * ceilDiv dims.h th
      let area := stripW * th
      let plan : NeWeightlessPlan := {
        logNE := 4, f := 4, fat := 0, shPref := shP, shMax := shP,
        tileHeight := th, direct := !resIsL2,
        srcStageBase := sb, resStageBase := rb, l2TaskPeak := pk
      }
      let isBetter := match best? with
        | none => true
        | some (_, bStrips, bArea, bStripW) =>
          if strips != bStrips then strips < bStrips
          else if area != bArea then area > bArea
          else stripW < bStripW
      if isBetter then
        best? := some (plan, strips, area, stripW)

  match best? with
  | some (winner, _, _, _) => pure winner
  | none => throw s!"no legal {target.name} NE channel reduce tiling fits available L2 budget (dims={repr dims})"

/-- Standalone O(1) tile chooser for weightless `neChannelReduce`. -/
def chooseNeChannelReduceTiling (target : Target) (dims : TensorDims) : Except String NeWeightlessPlan :=
  chooseNeChannelReduceTilingForL2 target dims false false []

-- ============================================================================
-- Multi-Task Builders (shared by single-task compilers & Vane/Graph.lean)
-- ============================================================================

/-- Endpoint storage location passed to multi-task pool/reduce task builders. -/
inductive PoolPortLoc where
  | surface (barSlot : UInt8)
  | dram (offset : Nat)
  | l2 (base : Nat)
  deriving Repr, BEq, Inhabited

private def encodePortSrcCfg
    (isFirst isL2 waitNe5 waitPe22 handoffNe handoffPe : Bool) : Except String UInt32 :=
  encodeWord (portCfgMultiLayout "src1Cfg") #[
    ("mode", if isL2 then portModeL2 else portModeDma),
    ("handoffNe", if !isFirst && isL2 && handoffNe then 1 else 0),
    ("handoffPe", if !isFirst && isL2 && handoffPe then 1 else 0),
    ("waitNe4", if isFirst then 1 else 0),
    ("waitNe5", if isFirst || waitNe5 then 1 else 0),
    ("dmaFlag", if isL2 then 0 else 1),
    ("P", 1),
    ("waitPe20", if isFirst then 1 else 0),
    ("waitPe22", if isFirst || waitPe22 then 1 else 0)
  ]

private def encodePortResCfg (isFirst isL2 direct : Bool) : Except String UInt32 :=
  encodeWord resCfgMultiLayout #[
    ("mode", if isL2 then portModeL2 else portModeDma),
    ("direct", if !isL2 && direct then 1 else 0),
    ("waitNe4", if isFirst then 1 else 0),
    ("waitNe5", if isFirst then 1 else 0),
    ("dmaFlag", 1),
    ("P", 1),
    ("waitPe20", if isFirst then 1 else 0),
    ("waitPe22", if isFirst then 1 else 0)
  ]

private def encodePoolSrcDep (taskId depKind depDist : Nat) : Except String UInt32 := do
  encodeWord (srcDepMultiLayout "src1Dep") #[
    ("code", srcDepCodeDma),
    ("task", ← natToU32Checked taskId "src1Dep.task"),
    ("depDist", ← natToU32Checked depDist "src1Dep.depDist"),
    ("depKind", ← natToU32Checked depKind "src1Dep.depKind")
  ]

private def poolOutputStageWrites (taskId : Nat) : Except String (Array RegWrite) := do
  let wOut2 ← encodeWord out2Layout #[("form", out2FormValue), ("task", ← natToU32Checked taskId "out2.task")]
  pure #[
    { byteAddr := regOut0, value := out0Value },
    { byteAddr := regOut1, value := out1Value },
    { byteAddr := regOut2, value := wOut2 },
    { byteAddr := regOut3, value := out3Value },
    { byteAddr := regOut4, value := out4Value },
    { byteAddr := regOut5, value := out5Value }
  ]

/-- Build a multi-task `TaskDesc` for PE Window Pooling (`0x003c = 0x10`, `ene = 0`). -/
def buildPePoolTaskMulti
    (target : Target) (op : PoolKind) (excludePad : Bool)
    (dims : TensorDims) (gh gw : PoolAxisGeom) (plan : PePoolPlan)
    (taskId totalTasks : Nat) (srcLoc resLoc : PoolPortLoc)
    (sync : TaskSyncInfo) (liveL2Mask : UInt32 := 0) : Except String TaskDesc := do
  let isFirst := taskId == 0
  let isLast := taskId + 1 == totalTasks
  let srcIsL2 := match srcLoc with | .l2 _ => true | _ => false
  let resIsL2 := match resLoc with | .l2 _ => true | _ => false
  let taskIdU32 ← natToU32Checked taskId "taskId"
  let stripW := (1 : Nat) <<< plan.k
  let m := min (9 - plan.k) (ceilLog2 plan.tileHeight)

  let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked gw.used "Win")]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked gh.used "Hin")]
  let cU32 ← natToU32Checked dims.c "Cin"
  let wWout ← encodeWord (dim16Layout "Wout") #[("val", ← natToU32Checked gw.out "Wout")]
  let wHout ← encodeWord (dim16Layout "Hout") #[("val", ← natToU32Checked gh.out "Hout")]
  let wConvCfg ← encodeWord convCfgLayout #[
    ("kw", ← natToU32Checked gw.k "kw"), ("kh", ← natToU32Checked gh.k "kh"),
    ("sx", ← natToU32Checked gw.sEff "sx"), ("sy", ← natToU32Checked gh.sEff "sy"),
    ("pl", ← natToU32Checked gw.padBefore "pl"), ("pt", ← natToU32Checked gh.padBefore "pt"),
    ("hx", 1), ("vy", 1)
  ]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked plan.tileHeight "TH")]
  let wPeCommon ← encodeWord peCommonLayout #[("mode", peCommonPoolValue)]
  let wPeTile ← encodeWord peTileLayout #[
    ("k", ← natToU32Checked plan.k "k"), ("m", ← natToU32Checked m "m")
  ]
  let wSrc1Cfg ← encodePortSrcCfg isFirst srcIsL2 sync.src1WaitNe5 sync.src1WaitPe22 sync.src1HandoffNe sync.src1HandoffPe
  let wResCfg ← encodePortResCfg isFirst resIsL2 plan.direct

  let mut plain : Array RegWrite := #[
    { byteAddr := regWin, value := wWin },
    { byteAddr := regHin, value := wHin },
    { byteAddr := regCin, value := cU32 },
    { byteAddr := regWout, value := wWout },
    { byteAddr := regHout, value := wHout },
    { byteAddr := regCout, value := cU32 },
    { byteAddr := regBatchN, value := 1 },
    { byteAddr := regConvCfg, value := wConvCfg },
    { byteAddr := regTH, value := wTH },
    { byteAddr := regPeCommon, value := wPeCommon },
    { byteAddr := regPeTile, value := wPeTile },
    { byteAddr := regTargetCfg, value := 0 },
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
    { byteAddr := regResCfg, value := wResCfg }
  ]
  if isFirst then
    plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }

  match op with
  | .max =>
    plain := plain ++ #[
      { byteAddr := regCmpFlags, value := cmpFlagMaxValue },
      { byteAddr := regPeOp, value := 0x00000002 }
    ]
  | .avg =>
    plain := plain.push { byteAddr := regPeOp, value := if excludePad then 0x00000000 else 0x00000001 }

  let mut barRecs : Array BarRegWrite := #[]
  match srcLoc with
  | .l2 base =>
    let l2Chs := 2 * alignUp dims.w 8
    let l2Row := dims.c * l2Chs
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := ← natToU32Checked l2Row "src1RowStride" }
    ]
  | loc =>
    let px := poolStripPx gw.used gw.k gw.sEff gw.padBefore stripW gw.out
    let row := 2 * px
    let u := dims.c * row
    let dmaRow := alignUp (2 * dims.w) dmaLineBytes
    let dmaPlane := dims.h * dmaRow
    let (slot, off) := match loc with | .surface s => (s, 0) | .dram o => (barSlotScratch, o) | .l2 _ => (0, 0)
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked plan.srcStageBase "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked row "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := ← natToU32Checked u "src1RowStride" },
      { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked u "src1RowBytesA" },
      { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked u "src1RowBytesB" },
      { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
      { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked dmaRow "src1DmaRowStride" },
      { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked dmaPlane "src1DmaPlaneStride" },
      { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
    ]
    let wSrcDep ← encodePoolSrcDep taskId sync.src1DepKind sync.src1DepDist
    barRecs := barRecs ++ #[
      { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[← natToU32Checked off "src1BaseLo", 0] },
      { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrcDep] }
    ]

  match resLoc with
  | .l2 base =>
    let drow := 2 * alignUp gw.out 8
    let du := dims.c * drow
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
      { byteAddr := regResChanStride, value := ← natToU32Checked drow "resChanStride" },
      { byteAddr := regResRowStride, value := ← natToU32Checked du "resRowStride" }
    ]
  | loc =>
    let dmaRow := alignUp (2 * gw.out) dmaLineBytes
    let dmaPlane := gh.out * dmaRow
    let (slot, off, depCode) := match loc with
      | .surface s => (s, 0, resDepCode target)
      | .dram o => (barSlotScratch, o, resDepCode target ||| resDepScratchFlags)
      | .l2 _ => (0, 0, resDepCode target)
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked plan.resStageBase "resBase" },
      { byteAddr := regResDmaRowStride, value := ← natToU32Checked dmaRow "resDmaRowStride" },
      { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked dmaPlane "resDmaPlaneStride" },
      { byteAddr := regResFmt, value := resFmtWordP1 }
    ]
    if !plan.direct then
      let drow := 2 * min (alignUp gw.out 8) (max stripW 32)
      let du := dims.c * drow
      plain := plain ++ #[
        { byteAddr := regResChanStride, value := ← natToU32Checked drow "resChanStride" },
        { byteAddr := regResRowStride, value := ← natToU32Checked du "resRowStride" },
        { byteAddr := regResRowBytesA, value := ← natToU32Checked du "resRowBytesA" },
        { byteAddr := regResRowBytesB, value := ← natToU32Checked du "resRowBytesB" }
      ]
    let wResDep ← encodeWord resDepLayout #[
      ("code", depCode), ("task", taskIdU32), ("direct", if plan.direct then 1 else 0)
    ]
    barRecs := barRecs ++ #[
      { baseByteAddr := regResBaseLo, barSlot := slot, values := #[← natToU32Checked off "resBaseLo", 0] },
      { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
    ]

  if isLast then
    plain := plain.push { byteAddr := regKernelStage, value := kernelStageNoKernValue }
    plain := plain ++ (← poolOutputStageWrites taskId)

  pure {
    taskId := taskId, logNE := 0, eneOverride := some 0,
    isFirst := isFirst, isLast := isLast, liveL2Mask := liveL2Mask,
    plainWrites := plain, barWrites := barRecs, kernelRelocs := #[]
  }

/-- Build a multi-task `TaskDesc` for PE Tensor Reduction (`0x003c = 0x50`, `ene = 0`). -/
def buildPeReduceTaskMulti
    (target : Target) (op : ReduceOp) (axes : ReduceAxes) (eps : UInt16)
    (dims : TensorDims) (plan : PeReducePlan)
    (taskId totalTasks : Nat) (srcLoc resLoc : PoolPortLoc)
    (sync : TaskSyncInfo) (liveL2Mask : UInt32 := 0) : Except String TaskDesc := do
  let isFirst := taskId == 0
  let isLast := taskId + 1 == totalTasks
  let srcIsL2 := match srcLoc with | .l2 _ => true | _ => false
  let resIsL2 := match resLoc with | .l2 _ => true | _ => false
  let taskIdU32 ← natToU32Checked taskId "taskId"
  let ax := peReduceAxes dims.c dims.h axes.reduceC axes.reduceH
  let stripW := (1 : Nat) <<< plan.k
  let m := min (9 - plan.k) (ceilLog2 plan.tileHeight)
  let count := (if axes.reduceC then dims.c else 1) * (if axes.reduceH then dims.h else 1) * dims.w

  let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked dims.w "Win")]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked ax.sH "Hin")]
  let sCU32 ← natToU32Checked ax.sC "Cin"
  let wDin ← encodeWord (dim16Layout "Din") #[("val", ← natToU32Checked ax.sD "Din")]
  let wBatchN ← encodeWord (dim16Layout "BatchN") #[("val", ← natToU32Checked ax.sN "BatchN")]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked plan.tileHeight "TH")]
  let wPeCommon ← encodeWord peCommonLayout #[("mode", peCommonReduceValue)]
  let wPeTile ← encodeWord peTileLayout #[
    ("k", ← natToU32Checked plan.k "k"), ("m", ← natToU32Checked m "m")
  ]
  let wSrc1Cfg ← encodePortSrcCfg isFirst srcIsL2 sync.src1WaitNe5 sync.src1WaitPe22 sync.src1HandoffNe sync.src1HandoffPe
  let wResCfg ← encodePortResCfg isFirst resIsL2 (!resIsL2)
  let wPeOp ← op.peOpWord

  let mut plain : Array RegWrite := #[
    { byteAddr := regWin, value := wWin },
    { byteAddr := regHin, value := wHin },
    { byteAddr := regCin, value := sCU32 },
    { byteAddr := regDin, value := wDin },
    { byteAddr := regWout, value := wWin },
    { byteAddr := regHout, value := wHin },
    { byteAddr := regCout, value := sCU32 },
    { byteAddr := regDout, value := wDin },
    { byteAddr := regBatchN, value := wBatchN },
    { byteAddr := regTH, value := wTH },
    { byteAddr := regPeCommon, value := wPeCommon },
    { byteAddr := regPeTile, value := wPeTile },
    { byteAddr := regTargetCfg, value := 0 },
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
    { byteAddr := regResCfg, value := wResCfg },
    { byteAddr := regPeOp, value := wPeOp }
  ]
  if isFirst then
    plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }
  if let some cf := op.cmpFlag? then
    plain := plain.push { byteAddr := regCmpFlags, value := cf }
  if op.usesEps && (eps &&& 0x7fff) != 0 then
    plain := plain.push { byteAddr := regPeEps, value := f16ToF32Bits eps }
  if op.usesScale then
    let invCount := floatToF32Bits (1.0 / Float.ofNat count)
    plain := plain.push { byteAddr := regPeRedScale, value := invCount }

  let mut barRecs : Array BarRegWrite := #[]
  match srcLoc with
  | .l2 base =>
    let inCh := 2 * alignUp dims.w 8
    let inRow := dims.c * inCh
    let l2StrideOf := fun
      | PeRedLogAxis.logC => inCh
      | PeRedLogAxis.logH => inRow
      | PeRedLogAxis.none => 0
    plain := plain.push { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" }
    if ax.sC > 1 then
      plain := plain.push { byteAddr := regSrc1ChanStride, value := ← natToU32Checked (l2StrideOf ax.axC) "src1ChanStride" }
    if ax.sH > 1 then
      plain := plain.push { byteAddr := regSrc1RowStride, value := ← natToU32Checked (l2StrideOf ax.axH) "src1RowStride" }
    if ax.sD > 1 then
      plain := plain.push { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked (l2StrideOf ax.axD) "src1RowBytesA" }
    if ax.sN > 1 then
      plain := plain.push { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked (l2StrideOf ax.axN) "src1RowBytesB" }
  | loc =>
    let px := min (alignUp dims.w 8) (max stripW 32)
    let row := 2 * px
    let u := ax.sC * ax.sD * row
    let dStrideL2 := if ax.sD > 1 then ax.sC * row else u
    let rs := alignUp (2 * dims.w) dmaLineBytes
    let dmaStrideOf := fun
      | PeRedLogAxis.logC => dims.h * rs
      | PeRedLogAxis.logH => rs
      | PeRedLogAxis.none => 0
    let (slot, off) := match loc with | .surface s => (s, 0) | .dram o => (barSlotScratch, o) | .l2 _ => (0, 0)
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked plan.srcStageBase "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked row "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := ← natToU32Checked u "src1RowStride" },
      { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked dStrideL2 "src1RowBytesA" },
      { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked u "src1RowBytesB" },
      { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
      { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
    ]
    if ax.sH > 1 then
      plain := plain.push { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked (dmaStrideOf ax.axH) "src1DmaRowStride" }
    if ax.sC > 1 then
      plain := plain.push { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked (dmaStrideOf ax.axC) "src1DmaPlaneStride" }
    if ax.sD > 1 then
      plain := plain.push { byteAddr := regSrc1DmaDStride, value := ← natToU32Checked (dmaStrideOf ax.axD) "src1DmaDStride" }
    if ax.sN > 1 then
      plain := plain.push { byteAddr := regSrc1DmaNStride, value := ← natToU32Checked (dmaStrideOf ax.axN) "src1DmaNStride" }
    let wSrcDep ← encodePoolSrcDep taskId sync.src1DepKind sync.src1DepDist
    barRecs := barRecs ++ #[
      { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[← natToU32Checked off "src1BaseLo", 0] },
      { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrcDep] }
    ]

  match resLoc with
  | .l2 base =>
    plain := plain.push { byteAddr := regResBase, value := ← natToU32Checked base "resBase" }
    if ax.sC > 1 then
      plain := plain.push { byteAddr := regResChanStride, value := 16 }
    if ax.sN > 1 then
      plain := plain.push { byteAddr := regResRowBytesB, value := ← natToU32Checked (ax.sC * 16) "resRowBytesB" }
  | loc =>
    let outDmaStrideOf := fun
      | PeRedLogAxis.logC => ax.outH * dmaLineBytes
      | PeRedLogAxis.logH => dmaLineBytes
      | PeRedLogAxis.none => 0
    let (slot, off, depCode) := match loc with
      | .surface s => (s, 0, resDepCode target)
      | .dram o => (barSlotScratch, o, resDepCode target ||| resDepScratchFlags)
      | .l2 _ => (0, 0, resDepCode target)
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked plan.resStageBase "resBase" },
      { byteAddr := regResFmt, value := resFmtWordP1 }
    ]
    if ax.sC > 1 then
      plain := plain.push { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked (outDmaStrideOf ax.axC) "resDmaPlaneStride" }
    if ax.sN > 1 then
      plain := plain.push { byteAddr := regResDmaNStride, value := ← natToU32Checked (outDmaStrideOf ax.axN) "resDmaNStride" }
    let wResDep ← encodeWord resDepLayout #[
      ("code", depCode), ("task", taskIdU32), ("direct", 1)
    ]
    barRecs := barRecs ++ #[
      { baseByteAddr := regResBaseLo, barSlot := slot, values := #[← natToU32Checked off "resBaseLo", 0] },
      { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
    ]

  if isLast then
    plain := plain.push { byteAddr := regKernelStage, value := kernelStageNoKernValue }
    plain := plain ++ (← poolOutputStageWrites taskId)

  pure {
    taskId := taskId, logNE := 0, eneOverride := some 0,
    isFirst := isFirst, isLast := isLast, liveL2Mask := liveL2Mask,
    plainWrites := plain, barWrites := barRecs, kernelRelocs := #[]
  }

/-- Build a multi-task `TaskDesc` for Weightless NE Window Pooling (`ene = logNE + 1`, no `__KERN_0`). -/
def buildNePoolTaskMulti
    (target : Target) (op : PoolKind)
    (dims : TensorDims) (gh gw : PoolAxisGeom) (plan : NeWeightlessPlan)
    (taskId totalTasks : Nat) (srcLoc resLoc : PoolPortLoc)
    (sync : TaskSyncInfo) (liveL2Mask : UInt32 := 0) : Except String TaskDesc := do
  let isFirst := taskId == 0
  let isLast := taskId + 1 == totalTasks
  let srcIsL2 := match srcLoc with | .l2 _ => true | _ => false
  let resIsL2 := match resLoc with | .l2 _ => true | _ => false
  let taskIdU32 ← natToU32Checked taskId "taskId"
  let stripW := plan.stripWidth

  let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked gw.used "Win")]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked gh.used "Hin")]
  let cU32 ← natToU32Checked dims.c "Cin"
  let wWout ← encodeWord (dim16Layout "Wout") #[("val", ← natToU32Checked gw.out "Wout")]
  let wHout ← encodeWord (dim16Layout "Hout") #[("val", ← natToU32Checked gh.out "Hout")]
  let wConvCfg ← encodeWord convCfgLayout #[
    ("kw", ← natToU32Checked gw.k "kw"), ("kh", ← natToU32Checked gh.k "kh"),
    ("sx", ← natToU32Checked gw.sEff "sx"), ("sy", ← natToU32Checked gh.sEff "sy"),
    ("pl", ← natToU32Checked gw.padBefore "pl"), ("pt", ← natToU32Checked gh.padBefore "pt"),
    ("hx", 1), ("vy", 1)
  ]
  let wDwGroups ← encodeWord dwGroupsLayout #[("perGroupIn", 1)]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked plan.tileHeight "TH")]
  let wNeCommon ← encodeWord neCommonLayout #[
    ("kind", 0),
    ("shPref", ← natToU32Checked plan.shPref "shPref"),
    ("shMin", ← natToU32Checked plan.shPref "shMin"),
    ("shMax", ← natToU32Checked plan.shMax "shMax"),
    ("F", ← natToU32Checked plan.f "F"),
    ("wide3x3", 0)
  ]
  let wNeTile ← encodeWord neTileLayout #[
    ("ocg", 3), ("fat", ← natToU32Checked plan.fat "fat"), ("WU", ← natToU32Checked plan.wu "WU")
  ]
  let wSrc1Cfg ← encodePortSrcCfg isFirst srcIsL2 sync.src1WaitNe5 sync.src1WaitPe22 sync.src1HandoffNe sync.src1HandoffPe
  let wResCfg ← encodePortResCfg isFirst resIsL2 plan.direct

  let mut plain : Array RegWrite := #[
    { byteAddr := regWin, value := wWin },
    { byteAddr := regHin, value := wHin },
    { byteAddr := regCin, value := cU32 },
    { byteAddr := regWout, value := wWout },
    { byteAddr := regHout, value := wHout },
    { byteAddr := regCout, value := cU32 },
    { byteAddr := regBatchN, value := 1 },
    { byteAddr := regConvCfg, value := wConvCfg },
    { byteAddr := regDwGroups, value := wDwGroups },
    { byteAddr := regTH, value := wTH },
    { byteAddr := regNeCommon, value := wNeCommon },
    { byteAddr := regNeTile, value := wNeTile },
    { byteAddr := regTargetCfg, value := targetCfgValue target },
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
    { byteAddr := regResCfg, value := wResCfg }
  ]
  if isFirst then
    plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }

  match op with
  | .max =>
    plain := plain ++ #[
      { byteAddr := regCmpFlags, value := cmpFlagMaxValue },
      { byteAddr := regNeOp, value := nePoolMaxOpValue },
      { byteAddr := regNeScale, value := fp16OneBits }
    ]
  | .avg =>
    let invArea := floatToF16CanonicalZero (1.0 / Float.ofNat (gh.k * gw.k))
    plain := plain ++ #[
      { byteAddr := regNeActShiftEn, value := 1 },
      { byteAddr := regNeOp, value := nePoolSumOpValue },
      { byteAddr := regNeScale, value := invArea.toUInt32 }
    ]

  let mut barRecs : Array BarRegWrite := #[]
  match srcLoc with
  | .l2 base =>
    let l2Chs := 2 * alignUp dims.w 8
    let l2Row := dims.c * l2Chs
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := ← natToU32Checked l2Row "src1RowStride" }
    ]
  | loc =>
    let px := poolStripPx gw.used gw.k gw.sEff gw.padBefore stripW gw.out
    let row := 2 * px
    let u := dims.c * row
    let dmaRow := alignUp (2 * dims.w) dmaLineBytes
    let dmaPlane := dims.h * dmaRow
    let (slot, off) := match loc with | .surface s => (s, 0) | .dram o => (barSlotScratch, o) | .l2 _ => (0, 0)
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked plan.srcStageBase "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked row "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := ← natToU32Checked u "src1RowStride" },
      { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked u "src1RowBytesA" },
      { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked u "src1RowBytesB" },
      { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
      { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked dmaRow "src1DmaRowStride" },
      { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked dmaPlane "src1DmaPlaneStride" },
      { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
    ]
    let wSrcDep ← encodePoolSrcDep taskId sync.src1DepKind sync.src1DepDist
    barRecs := barRecs ++ #[
      { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[← natToU32Checked off "src1BaseLo", 0] },
      { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrcDep] }
    ]

  match resLoc with
  | .l2 base =>
    let drow := 2 * alignUp gw.out 8
    let du := dims.c * drow
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
      { byteAddr := regResChanStride, value := ← natToU32Checked drow "resChanStride" },
      { byteAddr := regResRowStride, value := ← natToU32Checked du "resRowStride" }
    ]
  | loc =>
    let dmaRow := alignUp (2 * gw.out) dmaLineBytes
    let dmaPlane := gh.out * dmaRow
    let (slot, off, depCode) := match loc with
      | .surface s => (s, 0, resDepCode target)
      | .dram o => (barSlotScratch, o, resDepCode target ||| resDepScratchFlags)
      | .l2 _ => (0, 0, resDepCode target)
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked plan.resStageBase "resBase" },
      { byteAddr := regResDmaRowStride, value := ← natToU32Checked dmaRow "resDmaRowStride" },
      { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked dmaPlane "resDmaPlaneStride" },
      { byteAddr := regResFmt, value := resFmtWordP1 }
    ]
    if !plan.direct then
      let drow := 2 * min (alignUp gw.out 8) (max stripW 32)
      let du := dims.c * drow
      plain := plain ++ #[
        { byteAddr := regResChanStride, value := ← natToU32Checked drow "resChanStride" },
        { byteAddr := regResRowStride, value := ← natToU32Checked du "resRowStride" },
        { byteAddr := regResRowBytesA, value := ← natToU32Checked du "resRowBytesA" },
        { byteAddr := regResRowBytesB, value := ← natToU32Checked du "resRowBytesB" }
      ]
    let wResDep ← encodeWord resDepLayout #[
      ("code", depCode), ("task", taskIdU32), ("direct", if plan.direct then 1 else 0)
    ]
    barRecs := barRecs ++ #[
      { baseByteAddr := regResBaseLo, barSlot := slot, values := #[← natToU32Checked off "resBaseLo", 0] },
      { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
    ]

  if isLast then
    plain := plain.push { byteAddr := regKernelStage, value := kernelStageNoKernValue }
    plain := plain ++ (← poolOutputStageWrites taskId)

  pure {
    taskId := taskId, logNE := plan.logNE, eneOverride := some (plan.logNE + 1),
    isFirst := isFirst, isLast := isLast, liveL2Mask := liveL2Mask,
    plainWrites := plain, barWrites := barRecs, kernelRelocs := #[]
  }

/-- Build a multi-task `TaskDesc` for Weightless NE Channel Reduction (`Cout = 1`, no `__KERN_0`). -/
def buildNeChannelReduceTaskMulti
    (target : Target) (op : ReduceOp) (dims : TensorDims) (plan : NeWeightlessPlan)
    (taskId totalTasks : Nat) (srcLoc resLoc : PoolPortLoc)
    (sync : TaskSyncInfo) (liveL2Mask : UInt32 := 0) : Except String TaskDesc := do
  let isFirst := taskId == 0
  let isLast := taskId + 1 == totalTasks
  let srcIsL2 := match srcLoc with | .l2 _ => true | _ => false
  let resIsL2 := match resLoc with | .l2 _ => true | _ => false
  let taskIdU32 ← natToU32Checked taskId "taskId"
  let stripW := plan.stripWidth

  let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked dims.w "Win")]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked dims.h "Hin")]
  let cU32 ← natToU32Checked dims.c "Cin"
  let wDwGroups ← encodeWord neChanReduceGroupsLayout #[("chanCount", cU32)]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked plan.tileHeight "TH")]
  let wNeCommon ← encodeWord neCommonLayout #[
    ("kind", 0),
    ("shPref", ← natToU32Checked plan.shPref "shPref"),
    ("shMin", ← natToU32Checked plan.shPref "shMin"),
    ("shMax", ← natToU32Checked plan.shMax "shMax"),
    ("F", ← natToU32Checked plan.f "F"),
    ("wide3x3", 0)
  ]
  let wNeTile ← encodeWord neTileLayout #[
    ("ocg", 3), ("fat", ← natToU32Checked plan.fat "fat"), ("WU", ← natToU32Checked plan.wu "WU")
  ]
  let wSrc1Cfg ← encodePortSrcCfg isFirst srcIsL2 sync.src1WaitNe5 sync.src1WaitPe22 sync.src1HandoffNe sync.src1HandoffPe
  let wResCfg ← encodePortResCfg isFirst resIsL2 plan.direct

  let mut plain : Array RegWrite := #[
    { byteAddr := regWin, value := wWin },
    { byteAddr := regHin, value := wHin },
    { byteAddr := regCin, value := cU32 },
    { byteAddr := regWout, value := wWin },
    { byteAddr := regHout, value := wHin },
    { byteAddr := regCout, value := 1 },
    { byteAddr := regBatchN, value := 1 },
    { byteAddr := regDwGroups, value := wDwGroups },
    { byteAddr := regTH, value := wTH },
    { byteAddr := regNeCommon, value := wNeCommon },
    { byteAddr := regNeTile, value := wNeTile },
    { byteAddr := regTargetCfg, value := targetCfgValue target },
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
    { byteAddr := regResCfg, value := wResCfg }
  ]
  if isFirst then
    plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }

  match op with
  | .max =>
    plain := plain ++ #[
      { byteAddr := regCmpFlags, value := cmpFlagMaxValue },
      { byteAddr := regNeOp, value := nePoolMaxOpValue },
      { byteAddr := regNeScale, value := fp16OneBits }
    ]
  | .min =>
    plain := plain ++ #[
      { byteAddr := regCmpFlags, value := cmpFlagMinValue },
      { byteAddr := regNeOp, value := nePoolMinOpValue },
      { byteAddr := regNeScale, value := fp16OneBits }
    ]
  | .sum =>
    plain := plain ++ #[
      { byteAddr := regNeActShiftEn, value := 1 },
      { byteAddr := regNeOp, value := nePoolSumOpValue },
      { byteAddr := regNeScale, value := fp16OneBits }
    ]
  | .mean =>
    let invC := floatToF16CanonicalZero (1.0 / Float.ofNat dims.c)
    plain := plain ++ #[
      { byteAddr := regNeActShiftEn, value := 1 },
      { byteAddr := regNeOp, value := nePoolSumOpValue },
      { byteAddr := regNeScale, value := invC.toUInt32 }
    ]
  | other =>
    throw s!"weightless NE channel reduce does not support op '{other.name}'"

  let mut barRecs : Array BarRegWrite := #[]
  match srcLoc with
  | .l2 base =>
    let l2Chs := 2 * alignUp dims.w 8
    let l2Row := dims.c * l2Chs
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := ← natToU32Checked l2Row "src1RowStride" }
    ]
  | loc =>
    let px := min (alignUp dims.w 8) stripW
    let row := 2 * px
    let u := dims.c * row
    let dmaRow := alignUp (2 * dims.w) dmaLineBytes
    let dmaPlane := dims.h * dmaRow
    let (slot, off) := match loc with | .surface s => (s, 0) | .dram o => (barSlotScratch, o) | .l2 _ => (0, 0)
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked plan.srcStageBase "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked row "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := ← natToU32Checked u "src1RowStride" },
      { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked u "src1RowBytesA" },
      { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked u "src1RowBytesB" },
      { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
      { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
    ]
    if dims.h > 1 then
      plain := plain.push { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked dmaRow "src1DmaRowStride" }
    if dims.c > 1 then
      plain := plain.push { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked dmaPlane "src1DmaPlaneStride" }
    let wSrcDep ← encodePoolSrcDep taskId sync.src1DepKind sync.src1DepDist
    barRecs := barRecs ++ #[
      { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[← natToU32Checked off "src1BaseLo", 0] },
      { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrcDep] }
    ]

  match resLoc with
  | .l2 base =>
    let drow := 2 * alignUp dims.w 8
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
      { byteAddr := regResChanStride, value := ← natToU32Checked drow "resChanStride" },
      { byteAddr := regResRowStride, value := ← natToU32Checked drow "resRowStride" }
    ]
  | loc =>
    let dmaRow := alignUp (2 * dims.w) dmaLineBytes
    let (slot, off, depCode) := match loc with
      | .surface s => (s, 0, resDepCode target)
      | .dram o => (barSlotScratch, o, resDepCode target ||| resDepScratchFlags)
      | .l2 _ => (0, 0, resDepCode target)
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked plan.resStageBase "resBase" },
      { byteAddr := regResFmt, value := resFmtWordP1 }
    ]
    if dims.h > 1 then
      plain := plain.push { byteAddr := regResDmaRowStride, value := ← natToU32Checked dmaRow "resDmaRowStride" }
    if !plan.direct then
      let drow := 2 * min (alignUp dims.w 8) (max stripW 32)
      plain := plain ++ #[
        { byteAddr := regResChanStride, value := ← natToU32Checked drow "resChanStride" },
        { byteAddr := regResRowStride, value := ← natToU32Checked drow "resRowStride" },
        { byteAddr := regResRowBytesA, value := ← natToU32Checked drow "resRowBytesA" },
        { byteAddr := regResRowBytesB, value := ← natToU32Checked drow "resRowBytesB" }
      ]
    let wResDep ← encodeWord resDepLayout #[
      ("code", depCode), ("task", taskIdU32), ("direct", if plan.direct then 1 else 0)
    ]
    barRecs := barRecs ++ #[
      { baseByteAddr := regResBaseLo, barSlot := slot, values := #[← natToU32Checked off "resBaseLo", 0] },
      { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
    ]

  if isLast then
    plain := plain.push { byteAddr := regKernelStage, value := kernelStageNoKernValue }
    plain := plain ++ (← poolOutputStageWrites taskId)

  pure {
    taskId := taskId, logNE := plan.logNE, eneOverride := some (plan.logNE + 1),
    isFirst := isFirst, isLast := isLast, liveL2Mask := liveL2Mask,
    plainWrites := plain, barWrites := barRecs, kernelRelocs := #[]
  }

-- ============================================================================
-- Single-Task Compilers (verified by Vane.Verify.verifyHwxTasksSummary)
-- ============================================================================

/-- Complete compilation artifact for a single-task 2D pooling operation. -/
structure CompiledPool where
  spec : PoolSpec
  backend : PoolBackend
  outDims : TensorDims
  direct : Bool
  l2UseBytes : Nat
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections

/-- Complete compilation artifact for a single-task tensor reduction operation. -/
structure CompiledReduce where
  spec : ReduceSpec
  backend : ReduceBackend
  outDims : TensorDims
  direct : Bool
  l2UseBytes : Nat
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections

/-- Compile a single-task 2D pooling operation into a statically verified HWX binary
    and its 4 host-built Linux LOAD sections. -/
def compilePool (spec : PoolSpec) : Except String CompiledPool := do
  let (backend, gh, gw) ← resolvePoolBackend spec
  let outDims : TensorDims := { c := spec.dims.c, h := gh.out, w := gw.out }
  let noSync : TaskSyncInfo := { hasSrc1Dma := true }
  let (taskDesc, direct, l2Peak) ← match backend with
    | .pePool =>
      let plan ← choosePePoolTiling spec.target spec.dims.c gh gw
      let td ← buildPePoolTaskMulti spec.target spec.op spec.excludePad spec.dims gh gw plan
        0 1 (.surface barSlotFirstSurface) (.surface (barSlotFirstSurface + 1)) noSync
      pure (td, plan.direct, plan.l2TaskPeak)
    | .nePool =>
      let plan ← chooseNePoolTiling spec.target spec.dims.c gh gw
      let td ← buildNePoolTaskMulti spec.target spec.op spec.dims gh gw plan
        0 1 (.surface barSlotFirstSurface) (.surface (barSlotFirstSurface + 1)) noSync
      pure (td, plan.direct, plan.l2TaskPeak)
    | .peReduceGlobal =>
      let redOp : ReduceOp := match spec.op with | .max => .max | .avg => .mean
      let axes : ReduceAxes := { reduceH := true, reduceW := true }
      let ax := peReduceAxes spec.dims.c spec.dims.h false true
      let plan ← choosePeReduceTiling spec.target spec.dims.w ax
      let td ← buildPeReduceTaskMulti spec.target redOp axes 0 spec.dims plan
        0 1 (.surface barSlotFirstSurface) (.surface (barSlotFirstSurface + 1)) noSync
      pure (td, true, plan.l2TaskPeak)

  let sX ← makeFp16Surface spec.target "x" "input_x" .input spec.dims.c spec.dims.h spec.dims.w
  let sY ← makeFp16Surface spec.target "y" "output_y" .output outDims.c outDims.h outDims.w
  let emitted ← emitProgram {
    target := spec.target
    surfaces := #[sX, sY]
    tasks := #[taskDesc]
    kernel := none
    scratch := none
    constants := zeroBytes textConstBytes
    ident := some "vane"
  }
  let summary ← verifyHwxTasksSummary emitted.hwx emitted.info (requireKnownTask := true)
  if summary.total != 1 then
    throw s!"internal error: verifier checked {summary.total} tasks != 1"
  pure {
    spec := spec
    backend := backend
    outDims := outDims
    direct := direct
    l2UseBytes := l2Peak
    hwx := emitted.hwx
    info := emitted.info
    sections := emitted.sections
  }

/-- Compile a single-task tensor reduction operation into a statically verified HWX binary
    and its 4 host-built Linux LOAD sections. -/
def compileReduce (spec : ReduceSpec) : Except String CompiledReduce := do
  let (backend, outDims) ← resolveReduceBackend spec
  let noSync : TaskSyncInfo := { hasSrc1Dma := true }
  let (taskDesc, direct, l2Peak) ← match backend with
    | .peReduce =>
      let ax := peReduceAxes spec.dims.c spec.dims.h spec.axes.reduceC spec.axes.reduceH
      let plan ← choosePeReduceTiling spec.target spec.dims.w ax
      let td ← buildPeReduceTaskMulti spec.target spec.op spec.axes spec.eps spec.dims plan
        0 1 (.surface barSlotFirstSurface) (.surface (barSlotFirstSurface + 1)) noSync
      pure (td, true, plan.l2TaskPeak)
    | .neChannelReduce =>
      let plan ← chooseNeChannelReduceTiling spec.target spec.dims
      let td ← buildNeChannelReduceTaskMulti spec.target spec.op spec.dims plan
        0 1 (.surface barSlotFirstSurface) (.surface (barSlotFirstSurface + 1)) noSync
      pure (td, plan.direct, plan.l2TaskPeak)

  let sX ← makeFp16Surface spec.target "x" "input_x" .input spec.dims.c spec.dims.h spec.dims.w
  let sY ← makeFp16Surface spec.target "y" "output_y" .output outDims.c outDims.h outDims.w
  let emitted ← emitProgram {
    target := spec.target
    surfaces := #[sX, sY]
    tasks := #[taskDesc]
    kernel := none
    scratch := none
    constants := zeroBytes textConstBytes
    ident := some "vane"
  }
  let summary ← verifyHwxTasksSummary emitted.hwx emitted.info (requireKnownTask := true)
  if summary.total != 1 then
    throw s!"internal error: verifier checked {summary.total} tasks != 1"
  pure {
    spec := spec
    backend := backend
    outDims := outDims
    direct := direct
    l2UseBytes := l2Peak
    hwx := emitted.hwx
    info := emitted.info
    sections := emitted.sections
  }

end Vane.Pool
