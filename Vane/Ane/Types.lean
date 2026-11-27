import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Tiling
import Vane.Quant
import Vane.Pe
import Vane.Lut
import Vane.Pool
import Vane.Conv
import Vane.View
import Vane.Matmul
import Vane.IR.Graph

namespace Vane.Ane

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Tiling
open Vane.Quant
open Vane.Pe
open Vane.Lut
open Vane.Pool
open Vane.Conv
open Vane.View
open Vane.Matmul
open Vane.IR

-- ============================================================================
-- Shared Backend Placement & Strip Policy Modes
-- ============================================================================

/-- Placement strategy for intermediate edges between tasks. -/
inductive EdgePlacementMode where
  | auto
  | l2
  | dram
  deriving Repr, BEq, DecidableEq, Inhabited

namespace EdgePlacementMode

def name : EdgePlacementMode → String
  | .auto => "auto"
  | .l2 => "l2"
  | .dram => "dram"

def parse? (s : String) : Option EdgePlacementMode :=
  match s.toLower with
  | "auto" => some .auto
  | "l2" => some .l2
  | "dram" => some .dram
  | _ => none

end EdgePlacementMode

/-- Spatial H-strip / W-strip rolling cascade mode. -/
inductive StripMode where
  | off
  | auto
  | fixed (height : Nat)
  deriving Repr, BEq, DecidableEq, Inhabited

namespace StripMode

def name : StripMode → String
  | .off => "off"
  | .auto => "auto"
  | .fixed h => toString h

def parse? (s : String) : Option StripMode :=
  match s.toLower with
  | "off" => some .off
  | "auto" => some .auto
  | raw =>
    match raw.toNat? with
    | some n => if n >= 1 then some (.fixed n) else none
    | none => none

end StripMode

-- ============================================================================
-- L2 Value-Semantic ANE Engine IR (`SSAGraph AneOp`)
-- ============================================================================

/-- Convolution weight payload absorbed at L2 (`fp16` or native M8A `quant`). -/
inductive WeightData where
  | fp16 (bytes : ByteArray)
  | quant (spec : QuantWeightSpec)
  deriving BEq, Inhabited

namespace WeightData

def pretty : WeightData → String
  | .fp16 bs => s!"fp16({bs.size}B)"
  | .quant q => s!"quant({q.format.name},{q.codes.size}B)"

def byteSize : WeightData → Nat
  | .fp16 bs => bs.size
  | .quant q => q.codes.size

end WeightData

/-- Convolution grouping mode in `AneOp.neConv`. -/
inductive ConvMode where
  | dense
  | depthwise
  | grouped (groups : Nat)
  deriving Repr, BEq, DecidableEq, Inhabited

namespace ConvMode

def name : ConvMode → String
  | .dense => "dense"
  | .depthwise => "depthwise"
  | .grouped g => s!"grouped({g})"

def toConv3x3Mode? : ConvMode → Option Conv3x3Mode
  | .dense => some .dense
  | .depthwise => some .depthwise
  | .grouped 1 => some .dense
  | .grouped _ => none

end ConvMode

/-- Optional L2 tiling hint computed for `AneOp.neConv`. -/
structure ConvTilePlan where
  logNE : Nat := 0
  f : Nat := 0
  ocg : Nat := 0
  fat : Nat := 0
  shPref : Nat := 0
  shMax : Nat := 0
  tileHeight : Nat := 0
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Sub-family of weightless / LUT-only NE tasks (`neAct`). -/
inductive NeActKind where
  | act (spec : NeActSpec)
  | channelReduce (op : ReduceOp)
  | nePool (op : PoolKind) (kh kw sh sw : Nat) (pad : PoolPad) (excludePad : Bool := true)
  deriving Repr, BEq, Inhabited

namespace NeActKind

def pretty : NeActKind → String
  | .act s => s.name
  | .channelReduce op => s!"channel_reduce.{op.name}"
  | .nePool op kh kw sh sw pad _ => s!"ne_{op.opName}(k={kh}x{kw},s={sh}x{sw},pad={pad.name})"

end NeActKind

/-- Optional L2 tiling hint computed for `AneOp.neAct`. -/
structure NeActTilePlan where
  logNE : Nat := 0
  f : Nat := 0
  fat : Nat := 0
  shPref : Nat := 0
  shMax : Nat := 0
  tileHeight : Nat := 0
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Sub-family of Planar Engine tasks (`pe`). -/
inductive PeMachineOp where
  | unary (op : PeOp)
  | binary (op : PeBinOp) (rev : Bool := false)
  | pool (op : PoolKind) (kh kw sh sw : Nat) (pad : PoolPad) (excludePad : Bool := true)
  | reduce (op : ReduceOp) (axes : ReduceAxes) (eps : UInt16 := 0)
  deriving Repr, BEq, Inhabited

namespace PeMachineOp

def pretty : PeMachineOp → String
  | .unary op => s!"pe1.{op.opName}"
  | .binary op rev => if rev && op == .sub then "pe2.rsub" else s!"pe2.{op.name}"
  | .pool op kh kw sh sw pad _ => s!"pe_{op.opName}(k={kh}x{kw},s={sh}x{sw},pad={pad.name})"
  | .reduce op axes _ => s!"pe_reduce.{op.name}(axes={axes.name})"

end PeMachineOp

/-- Optional L2 tiling hint computed for `AneOp.pe`. -/
structure PeTilePlan where
  k : Nat := 0
  tileHeight : Nat := 0
  deriving Repr, BEq, DecidableEq, Inhabited

/-- General strided 4D/batched DMA view descriptor (`dmaView`).
    Expresses transpose, slice, concat, split, reshape, halo row stitching,
    and zero-copy aliasable channel/spatial subviews. -/
structure DmaViewSpec where
  inShape : Shape4D
  outShape : Shape4D
  srcBaseOff : Nat := 0
  srcStrides : Array Nat := #[0, 0, 0, 2]
  dstBaseOff : Nat := 0
  dstStrides : Array Nat := #[0, 0, 0, 2]
  transposeCW : Bool := false
  batchN : Nat := 1
  d : Nat := 1
  /-- Non-none for multi-input concat (including SOK channel-split joins). -/
  concatAxis? : Option Nat := none
  concatInShapes : Array Shape4D := #[]
  /-- If `true`, L3 `MemoryPlan` may lower this view as a zero-copy `BufferStorage.subview`. -/
  aliasable : Bool := false
  /-- Minimum DRAM scratch bytes required when this view writes a T-mode transpose scratch tensor. -/
  minDramBytes : Nat := 0
  /-- Explicit `ViewCopy` override when lowering directly to `neView`. -/
  explicitCopy? : Option ViewCopy := none
  deriving Repr, BEq, Inhabited

namespace DmaViewSpec

def toViewCopy (d : DmaViewSpec) : ViewCopy :=
  match d.explicitCopy? with
  | some cp => cp
  | none =>
    let srcN := d.srcStrides.getD 0 0
    let srcP := d.srcStrides.getD 1 0
    let srcR := d.srcStrides.getD 2 0
    let dstN := d.dstStrides.getD 0 0
    let dstP := d.dstStrides.getD 1 0
    let dstR := d.dstStrides.getD 2 0
    let (c, h, w) :=
      if d.transposeCW then (d.inShape.c, d.inShape.h, d.inShape.w)
      else (d.outShape.c, d.outShape.h, d.outShape.w)
    {
      n := d.batchN
      c := c
      h := h
      w := w
      d := d.d
      inOff := d.srcBaseOff
      inRowStride := srcR
      inPlaneStride := srcP
      inDStride := 0
      inNStride := srcN
      outOff := d.dstBaseOff
      outRowStride := dstR
      outPlaneStride := dstP
      outDStride := 0
      outNStride := dstN
      transposeCW := d.transposeCW
    }

def pretty (d : DmaViewSpec) : String :=
  match d.concatAxis? with
  | some ax =>
    let al := if d.aliasable then ",alias" else ""
    s!"concat(axis={ax},out={d.outShape.toArray.toList}{al})"
  | none =>
    let al := if d.aliasable then ",alias" else ""
    let tr := if d.transposeCW then ",trCW" else ""
    s!"view({d.inShape.toArray.toList}->{d.outShape.toArray.toList},off={d.srcBaseOff}->{d.dstBaseOff}{tr}{al})"

end DmaViewSpec

/-- Build a `DmaViewSpec` from a single-copy `ViewPlan`. -/
def dmaViewSpecFromPlan (inSh : Shape4D) (vp : ViewPlan) (aliasable : Bool := false) : DmaViewSpec :=
  let cp := vp.copies.getD 0 default
  {
    inShape := inSh
    outShape := vp.outShape
    srcBaseOff := cp.inOff
    srcStrides := #[cp.inNStride, cp.inPlaneStride, cp.inRowStride, 2]
    dstBaseOff := cp.outOff
    dstStrides := #[cp.outNStride, cp.outPlaneStride, cp.outRowStride, 2]
    transposeCW := cp.transposeCW
    batchN := cp.n
    d := cp.d
    aliasable := aliasable
    explicitCopy? := some cp
  }

/-- Build a `DmaViewSpec` for a multi-input `concat(axis)`. -/
def dmaViewSpecFromConcat (inShapes : Array Shape4D) (axis : Nat) (outShape : Shape4D)
    (aliasable : Bool := false) : DmaViewSpec :=
  {
    inShape := inShapes.getD 0 outShape
    outShape := outShape
    srcStrides := (inShapes.getD 0 outShape).dramStrides
    dstStrides := outShape.dramStrides
    concatAxis? := some axis
    concatInShapes := inShapes
    aliasable := aliasable
  }

/-- The 5 unified value-semantic ANE hardware engine task families (L2 `AneIR`). -/
inductive AneOp where
  | neConv
      (kh kw sh sw : Nat)
      (pad : PadMode)
      (mode : ConvMode)
      (cin cout : Nat)
      (weight : WeightData)
      (bias : Option ByteArray := none)
      (act : ConvActivation := .none)
      (tiling : Option ConvTilePlan := none)
      (weightGroupId : Nat := 0)
      (opLabel : String := "conv")
  | neAct
      (kind : NeActKind)
      (tiling : Option NeActTilePlan := none)
      (weightGroupId : Nat := 0)
  | neMatmul
      (heads m k n : Nat)
      (transposeA transposeB : Bool := false)
      (scaleBits : UInt16 := f16PosOne)
      (mOffset : Nat := 0)
      (totalM : Nat := 0)
  | dmaView
      (desc : DmaViewSpec)
      (opLabel : String := "dma_view")
  | pe
      (op : PeMachineOp)
      (tiling : Option PeTilePlan := none)
  deriving BEq, Inhabited

namespace AneOp

def pretty : AneOp → String
  | .neConv kh kw sh sw pad mode cin cout w b act _ wg lbl =>
    let bStr := if b.isSome then "+bias" else ""
    let aStr := if act == .none then "" else s!"+{act.name}"
    s!"ane.ne_conv[{lbl}]({kh}x{kw},s={sh}x{sw},pad={pad.name},{mode.name},{cin}->{cout},{w.pretty}{bStr}{aStr},wg={wg})"
  | .neAct kind _ wg =>
    s!"ane.ne_act({kind.pretty},wg={wg})"
  | .neMatmul heads m k n trA trB sc mOff totM =>
    let slStr := if totM > m then s!",mOff={mOff}/{totM}" else ""
    s!"ane.ne_matmul(H={heads},M={m},K={k},N={n},trA={trA},trB={trB},scale={f16ToFloat sc}{slStr})"
  | .dmaView desc lbl =>
    s!"ane.dma_view[{lbl}]({desc.pretty})"
  | .pe op _ =>
    s!"ane.pe({op.pretty})"

def usesKernel : AneOp → Bool
  | .neConv .. => true
  | .neAct (.act (.lut _)) _ _ => true
  | _ => false

def isNeCompute : AneOp → Bool
  | .neConv .. => true
  | .neAct (.act (.lut _)) _ _ => true
  | .neMatmul .. => true
  | _ => false

def opKindLabel : AneOp → String
  | .neConv kh kw _ _ _ _ _ _ _ _ _ _ _ lbl =>
    if !lbl.isEmpty then lbl
    else if kh == 1 && kw == 1 then "conv1x1"
    else s!"conv{kh}x{kw}"
  | .neAct kind _ _ =>
    match kind with
    | .act s => s!"lut_{s.name}"
    | .channelReduce op => s!"reduce_{op.name}"
    | .nePool op .. => op.opName
  | .neMatmul .. => "matmul"
  | .dmaView _ lbl => if lbl.isEmpty then "dma_view" else lbl
  | .pe op _ =>
    match op with
    | .unary u => s!"pe_{u.opName}"
    | .binary b rev => if rev && b == .sub then "pe_rsub" else s!"pe_{b.name}"
    | .pool p .. => p.opName
    | .reduce r .. => s!"reduce_{r.name}"

end AneOp

/-- Optional horizontal row-band window `(rowOffset, fullH)` on a network surface. -/
structure AneSurfaceWindow where
  rowOffset : Nat := 0
  fullH : Nat := 0
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Per-node L2 side metadata (placement override, logical name, surface windows for strips). -/
structure AneNodeMeta where
  logicalId : String := ""
  outPlacement : Option EdgePlacementMode := none
  src1Win : AneSurfaceWindow := {}
  src2Win : AneSurfaceWindow := {}
  resWin : AneSurfaceWindow := {}
  /-- For multi-strip outputs, which graph output index `0..outputs.size-1` this strip writes to (if any). -/
  stripOutputIdx? : Option Nat := none
  deriving Repr, BEq, Inhabited

/-- Complete value-semantic L2 `AneIR` program (`SSAGraph AneOp` + I/O/cascade metadata). -/
structure AneProgram where
  graph : SSAGraph AneOp
  inputShortNames : Array String := #[]
  outputNames : Array String := #[]
  outputShortNames : Array String := #[]
  defaultPlacement : EdgePlacementMode := .auto
  nodeMeta : Array AneNodeMeta := #[]
  /-- Optional input→output buffer donation pairs `(inputSurfIdx, outputIdx)`. -/
  donations : Array (Nat × Nat) := #[]
  stripCount : Nat := 1
  stripHeight : Nat := 0
  stripAxis : Nat := 2
  weightCount : Nat := 0
  weightBytes : Nat := 0
  /-- Logical shapes of graph outputs (for ONNX static shape validation). -/
  outputLogicalShapes : Array (Array Nat) := #[]
  deriving BEq, Inhabited

namespace AneProgram

def getNodeMeta (p : AneProgram) (idx : Nat) : AneNodeMeta :=
  p.nodeMeta.getD idx {}

def dumpAne (p : AneProgram) : String :=
  let header :=
    if p.stripCount > 1 || p.stripHeight > 0 then
      let axNm := if p.stripAxis == 3 then "W" else "H"
      s!"// AneIR: target={p.graph.target.name}, placement={p.defaultPlacement.name}, cascade={axNm}(strips={p.stripCount}, size={p.stripHeight})\n"
    else
      s!"// AneIR: target={p.graph.target.name}, placement={p.defaultPlacement.name}\n"
  header ++ p.graph.pretty AneOp.pretty

end AneProgram

/-- Convert a 4D `TensorType` (`#[N, C, H, W]`) to `Shape4D`. -/
def tensorTypeToShape4D (ty : TensorType) (ctx : String := "tensor") : Except String Shape4D :=
  Shape4D.ofArray ty.shape |>.mapError fun e => s!"{ctx}: {e}"

/-- Convert a `Shape4D` to a 4D fp16 `TensorType`. -/
def shape4DToTensorType (s : Shape4D) : TensorType :=
  { shape := s.toArray, dtype := .fp16 }

/-- Convert a 4D `Shape4D` to `TensorDims` (dropping N=1). -/
def shape4DToDims (s : Shape4D) : TensorDims :=
  { c := s.c, h := s.h, w := s.w }

/-- L2-resident byte footprint of a 4D tensor `[1, C, H, W]` (`alignUp(H * C * 2 * alignUp(W, 8), 64)`). -/
def l2ResidentBytes4D (s : Shape4D) : Nat :=
  let chs := 2 * alignUp s.w 8
  let row := s.c * chs
  alignUp (s.h * row) 64

/-- DRAM scratch byte footprint of a 4D tensor `[N, C, H, W]` (64B-aligned). -/
def dramScratchBytes4D (s : Shape4D) : Nat :=
  alignUp s.span 64

end Vane.Ane
