import Vane.IR.Rewrite
import Vane.IR.Passes.HighFuse
import Vane.View

namespace Vane.IR

open Vane.Binary
open Vane.Float16
open Vane.View

-- ============================================================================
-- Pure ONNX / TensorIR Validation Helpers (Shared with Frontend & Formal Proofs)
-- ============================================================================

/-- Validate a `LayerNormalization` `Scale` or `B` tensor shape against the normalized suffix `X.shape[axis:]`.
    For `normAxis == 1` (with `H=W=1` on 4D or 2D `[1, C]`): accepts scalar all-1s, `[C]`,
    `[C, 1, 1]`, `[1, C, 1, 1]` (4D), or `[1, C]` (2D).
    For `normAxis == 3` (last axis `W` on 4D or `K` on 3D): accepts scalar all-1s, `[W]`,
    or `[1, ..., 1, W]` up to `inRank`. -/
def isValidLayerNormParamShape (dims : Array Nat) (normAxis normDim inRank : Nat) : Bool :=
  if dims.size > inRank then
    false
  else if dims.all (· == 1) then
    true
  else if normAxis == 1 then
    dims == #[normDim] ||
    (inRank == 4 && (dims == #[normDim, 1, 1] || dims == #[1, normDim, 1, 1])) ||
    (inRank == 2 && dims == #[1, normDim])
  else
    dims.size >= 1 && dims[dims.size - 1]! == normDim &&
    (dims.extract 0 (dims.size - 1)).all (· == 1)

private def arrayUniformFloat (arr : Array Float) : Option Float :=
  if arr.isEmpty then none
  else
    let v0 := arr[0]!
    if arr.all (fun v => v == v0) then some v0 else none

private def validateFiniteFp16ScalarIR (v : Float) (ctx : String) : Except String UInt16 := do
  if v.isNaN || v.isInf then
    throw s!"{ctx}: scalar {v} must be finite"
  let h := floatToF16CanonicalZero v
  if !f16IsFinite h then
    throw s!"{ctx}: scalar {v} overflows finite fp16 range"
  pure h

-- ============================================================================
-- Pass 5: Decompose `SSAGraph HighOp → Except String (SSAGraph CoreOp)`
-- ============================================================================

private def lookupAlias (aliases : Array (ValueId × ValueId)) (v : ValueId) : ValueId := Id.run do
  let mut cur := v
  for _ in [:256] do
    match aliases.find? fun p => p.1 == cur with
    | some (_, nxt) =>
      if nxt == cur then return cur
      cur := nxt
    | none => return cur
  return cur

private def lookupConstInCore (nodes : Array (IRNode CoreOp)) (v : ValueId) :
    Option (TensorType × ConstPayload) :=
  nodes.findSome? fun nd =>
    match nd.op, nd.outputs.toList with
    | .const p, [(vid, ty)] => if vid == v then some (ty, p) else none
    | _, _ => none

private def lookupTypeInCore (inputs : Array (ValueId × String × TensorType))
    (nodes : Array (IRNode CoreOp)) (v : ValueId) : Option TensorType :=
  match inputs.find? (fun e => e.1 == v) with
  | some (_, _, ty) => some ty
  | none =>
    nodes.findSome? fun nd =>
      (nd.outputs.find? fun o => o.1 == v).map Prod.snd

/-- Decompose every high-level composite `HighOp` (`batchNorm`, `layerNorm`, `rmsNorm`,
    `softmax`, `gelu`, `split`, `pow`) into strictly closed `CoreOp` primitives. -/
def decomposeGraph (g : SSAGraph HighOp) (allowViewOps : Bool := true) :
    Except String (SSAGraph CoreOp × Nat) := do
  let mut outNodes : Array (IRNode CoreOp) := Array.mkEmpty g.nodes.size
  let mut aliases : Array (ValueId × ValueId) := #[]
  let mut nextId := g.nextValueId
  let mut elidedCount : Nat := 0

  let getType := fun (v : ValueId) (nodes : Array (IRNode CoreOp)) (ctx : String) => do
    let some ty := lookupTypeInCore g.inputs nodes v
      | throw s!"{ctx}: unknown SSA value %{v}"
    pure ty

  for nd in g.nodes do
    let inps := nd.inputs.map (lookupAlias aliases)
    let ctx := if nd.name.isEmpty then "decompose" else s!"node '{nd.name}'"

    match nd.op with
    | .core c =>
      outNodes := outNodes.push {
        name := nd.name
        op := c
        inputs := inps
        outputs := nd.outputs
      }

    | .gelu .exact =>
      if inps.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: Gelu expects 1 input and 1 output"
      outNodes := outNodes.push {
        name := nd.name
        op := .lut (.lut .gelu)
        inputs := inps
        outputs := nd.outputs
      }

    | .pow kind =>
      if inps.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: Pow expects 1 input and 1 output"
      let xId := inps[0]!
      let outId := nd.outputs[0]!.1
      match kind with
      | .identity =>
        aliases := aliases.push (outId, xId)
        elidedCount := elidedCount + 1
      | .square =>
        outNodes := outNodes.push {
          name := nd.name
          op := .elemBinary .mul
          inputs := #[xId, xId]
          outputs := nd.outputs
        }
      | .sqrt =>
        outNodes := outNodes.push {
          name := nd.name
          op := .lut (.lut .sqrt)
          inputs := #[xId]
          outputs := nd.outputs
        }
      | .rsqrt =>
        outNodes := outNodes.push {
          name := nd.name
          op := .lut (.lut (.rsqrt 0.0))
          inputs := #[xId]
          outputs := nd.outputs
        }
      | .reciprocal =>
        outNodes := outNodes.push {
          name := nd.name
          op := .lut (.lut .inverse)
          inputs := #[xId]
          outputs := nd.outputs
        }

    | .batchNorm eps =>
      if inps.size < 5 || nd.outputs.size < 1 then
        throw s!"{ctx}: BatchNormalization requires 5 inputs and 1 output"
      let xId := inps[0]!
      let xTy ← getType xId outNodes ctx
      if xTy.shape.size != 4 && xTy.shape.size != 2 then
        throw s!"{ctx}: BatchNormalization requires 4D [1,C,H,W] or 2D [1,C] input, got rank {xTy.shape.size} ({xTy.shape.toList})"
      let c := xTy.shape[1]!
      let readConstVec := fun (slot : Nat) (label : String) => do
        let some (_, p) := lookupConstInCore outNodes inps[slot]!
          | throw s!"{ctx}: BatchNormalization {label} must be a constant initializer"
        let some vs := p.toFloat64?
          | throw s!"{ctx}: BatchNormalization {label} must be float"
        pure vs
      let scale ← readConstVec 1 "scale"
      let bVec ← readConstVec 2 "B"
      let mean ← readConstVec 3 "mean"
      let variance ← readConstVec 4 "var"
      let (factor, biasF64) ← computeBNAffineF64 scale bVec mean variance eps c ctx
      let mut diagW : Array Float := Array.replicate (c * c) 0.0
      for ch in [:c] do
        diagW := diagW.set! (ch * c + ch) factor[ch]!
      let wId := nextId
      let bId := nextId + 1
      nextId := nextId + 2
      outNodes := outNodes.push {
        name := s!"{nd.name}_w"
        op := .const (.f32 diagW)
        inputs := #[]
        outputs := #[(wId, { shape := #[c, c, 1, 1], dtype := .f32 })]
      }
      outNodes := outNodes.push {
        name := s!"{nd.name}_b"
        op := .const (.f32 biasF64)
        inputs := #[]
        outputs := #[(bId, { shape := #[c], dtype := .f32 })]
      }
      outNodes := outNodes.push {
        name := nd.name
        op := .conv 1 1 1 1 #[0, 0, 0, 0] 1 true .none
        inputs := #[xId, wId, bId]
        outputs := #[nd.outputs[0]!]
      }

    | .softmax axis legacyOpset =>
      if inps.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: Softmax expects 1 input and 1 output"
      let xId := inps[0]!
      let xTy ← getType xId outNodes ctx
      let rank := xTy.shape.size
      let ax := if axis < 0 then axis + (rank : Int) else axis
      let redAxes : Array Nat ←
        if legacyOpset then
          if rank == 3 then
            if ax == 2 then pure #[2]
            else throw s!"{ctx}: 3D Softmax (legacy opset) axis {axis} not supported (expected -1 or 2)"
          else if rank == 4 then
            match ax with
            | 0 | 1 => pure #[1, 2, 3]
            | 2 => pure #[2, 3]
            | 3 => pure #[3]
            | _ => throw s!"{ctx}: Softmax (legacy opset) axis {axis} out of range [0,3]"
          else if rank == 2 then
            if ax == 1 then pure #[1]
            else throw s!"{ctx}: 2D Softmax axis {axis} not supported"
          else
            throw s!"{ctx}: Softmax unsupported rank {rank}"
        else
          if rank == 3 then
            if ax == 2 then pure #[2]
            else throw s!"{ctx}: 3D Softmax axis {axis} not supported (expected -1 or 2)"
          else if rank == 4 then
            match ax with
            | 1 => pure #[1]
            | 2 => pure #[2]
            | 3 => pure #[3]
            | _ => throw s!"{ctx}: Softmax axis {axis} not in \{1,2,3}"
          else if rank == 2 then
            if ax == 1 then pure #[1]
            else throw s!"{ctx}: 2D Softmax axis {axis} not supported"
          else
            throw s!"{ctx}: Softmax unsupported rank {rank}"

      let redShape := Id.run do
        let mut sh := xTy.shape
        for a in redAxes do
          if a < sh.size then sh := sh.set! a 1
        return sh
      let redTy : TensorType := { shape := redShape, dtype := xTy.dtype }
      let maxId := nextId
      let subId := nextId + 1
      let expId := nextId + 2
      nextId := nextId + 3

      outNodes := outNodes.push {
        name := s!"{nd.name}_max"
        op := .reduce .max redAxes true 0
        inputs := #[xId]
        outputs := #[(maxId, redTy)]
      }
      outNodes := outNodes.push {
        name := s!"{nd.name}_sub"
        op := .elemBinary .sub
        inputs := #[xId, maxId]
        outputs := #[(subId, xTy)]
      }
      outNodes := outNodes.push {
        name := s!"{nd.name}_exp"
        op := .lut (.lut .exp)
        inputs := #[subId]
        outputs := #[(expId, xTy)]
      }

      -- Channel-only reduction with W > 1 uses NE channel sum + LUT inverse
      let isChannelOnlyWithW := rank == 4 && redAxes == #[1] && xTy.shape[3]! > 1
      let invSumId ←
        if isChannelOnlyWithW then
          let sumId := nextId
          let invId := nextId + 1
          nextId := nextId + 2
          outNodes := outNodes.push {
            name := s!"{nd.name}_sum"
            op := .reduce .sum redAxes true 0
            inputs := #[expId]
            outputs := #[(sumId, redTy)]
          }
          outNodes := outNodes.push {
            name := s!"{nd.name}_invsum"
            op := .lut (.lut .inverse)
            inputs := #[sumId]
            outputs := #[(invId, redTy)]
          }
          pure invId
        else
          let invId := nextId
          nextId := nextId + 1
          outNodes := outNodes.push {
            name := s!"{nd.name}_invsum"
            op := .reduce .invSum redAxes true 0
            inputs := #[expId]
            outputs := #[(invId, redTy)]
          }
          pure invId

      outNodes := outNodes.push {
        name := nd.name
        op := .elemBinary .mul
        inputs := #[expId, invSumId]
        outputs := #[nd.outputs[0]!]
      }

    | .layerNorm axis eps hasScale hasBias =>
      if inps.size < 1 || nd.outputs.size < 1 then
        throw s!"{ctx}: LayerNormalization requires at least 1 input and 1 output"
      let xId := inps[0]!
      let xTy ← getType xId outNodes ctx
      let rank := xTy.shape.size
      let ax := if axis < 0 then axis + (rank : Int) else axis
      let (redAxes, redShape, normAxis, normDim) ←
        if rank == 3 then
          if ax == 2 then
            pure (#[2], #[xTy.shape[0]!, xTy.shape[1]!, 1], 3, xTy.shape[2]!)
          else
            throw s!"{ctx}: 3D LayerNormalization axis {axis} not supported (expected -1 or 2)"
        else if rank == 2 then
          if ax == 1 then
            pure (#[1], #[1, 1], 1, xTy.shape[1]!)
          else
            throw s!"{ctx}: 2D LayerNormalization axis {axis} not supported (expected -1 or 1)"
        else if rank == 4 then
          match ax with
          | 1 =>
            if xTy.shape[2]! != 1 || xTy.shape[3]! != 1 then
              throw s!"{ctx}: 4D LayerNormalization with axis=1 normalizes over [C,H,W]={xTy.shape.extract 1 4 |>.toList}, which is only supported when H=1 and W=1 (got H={xTy.shape[2]!}, W={xTy.shape[3]!}); use axis=-1 (W) or reshape/transpose so the normalized axis is last"
            pure (#[1], #[1, 1, 1, 1], 1, xTy.shape[1]!)
          | 3 =>
            pure (#[3], #[1, xTy.shape[1]!, xTy.shape[2]!, 1], 3, xTy.shape[3]!)
          | _ =>
            throw s!"{ctx}: LayerNormalization axis {axis} not supported (supported: C [axis=1 with H=W=1] or W [axis=3/-1])"
        else
          throw s!"{ctx}: LayerNormalization requires 4D, 3D, or 2D input, got rank {rank}"

      if eps.isNaN || eps.isInf || eps < 0.0 then
        throw s!"{ctx}: LayerNormalization invalid epsilon {eps}"
      let epsBits ← validateFiniteFp16ScalarIR eps ctx
      let redTy : TensorType := { shape := redShape, dtype := xTy.dtype }

      let meanId := if nd.outputs.size >= 2 then nd.outputs[1]!.1 else nextId
      if nd.outputs.size < 2 then nextId := nextId + 1
      let diffId := nextId
      let sqId := nextId + 1
      let varId := nextId + 2
      let varEpsId := nextId + 3
      nextId := nextId + 4
      let invStdId := if nd.outputs.size >= 3 then nd.outputs[2]!.1 else nextId
      if nd.outputs.size < 3 then nextId := nextId + 1

      outNodes := outNodes.push {
        name := s!"{nd.name}_mean"
        op := .reduce .mean redAxes true 0
        inputs := #[xId]
        outputs := #[(meanId, redTy)]
      }
      outNodes := outNodes.push {
        name := s!"{nd.name}_diff"
        op := .elemBinary .sub
        inputs := #[xId, meanId]
        outputs := #[(diffId, xTy)]
      }
      outNodes := outNodes.push {
        name := s!"{nd.name}_sq"
        op := .elemBinary .mul
        inputs := #[diffId, diffId]
        outputs := #[(sqId, xTy)]
      }
      outNodes := outNodes.push {
        name := s!"{nd.name}_var"
        op := .reduce .mean redAxes true 0
        inputs := #[sqId]
        outputs := #[(varId, redTy)]
      }
      outNodes := outNodes.push {
        name := s!"{nd.name}_vareps"
        op := .elemUnary (.scalarAdd epsBits)
        inputs := #[varId]
        outputs := #[(varEpsId, redTy)]
      }
      outNodes := outNodes.push {
        name := s!"{nd.name}_invstd"
        op := .lut (.lut (.rsqrt 0.0))
        inputs := #[varEpsId]
        outputs := #[(invStdId, redTy)]
      }

      let yOut := nd.outputs[0]!
      if !hasScale then
        outNodes := outNodes.push {
          name := nd.name
          op := .elemBinary .mul
          inputs := #[diffId, invStdId]
          outputs := #[yOut]
        }
      else
        if inps.size < 2 then
          throw s!"{ctx}: LayerNormalization missing Scale input"
        let scaleId := inps[1]!
        let biasId? := if hasBias && inps.size >= 3 then some inps[2]! else none
        match lookupConstInCore outNodes scaleId with
        | some (scTy, scPay) =>
          if !isValidLayerNormParamShape scTy.shape normAxis normDim rank then
            throw s!"{ctx}: LayerNormalization Scale shape {scTy.shape.toList} incompatible with normalized suffix X.shape[{ax}:] (normDim={normDim})"
          let some scVals := scPay.toFloat64?
            | throw s!"{ctx}: LayerNormalization Scale must be float"
          if scVals.size != 1 && scVals.size != normDim then
            throw s!"{ctx}: LayerNormalization Scale length {scVals.size} incompatible with normalized dim {normDim}"
          let biasVals? ← match biasId? with
            | none => pure none
            | some bId =>
              match lookupConstInCore outNodes bId with
              | some (bTy, bPay) =>
                if !isValidLayerNormParamShape bTy.shape normAxis normDim rank then
                  throw s!"{ctx}: LayerNormalization B shape {bTy.shape.toList} incompatible with normalized suffix X.shape[{ax}:] (normDim={normDim})"
                let some bv := bPay.toFloat64?
                  | throw s!"{ctx}: LayerNormalization B must be float"
                if bv.size != 1 && bv.size != normDim then
                  throw s!"{ctx}: LayerNormalization B length {bv.size} incompatible with normalized dim {normDim}"
                pure (some bv)
              | none =>
                throw s!"{ctx}: LayerNormalization bias must be constant when Scale is constant"

          let scUnif? := arrayUniformFloat scVals
          let bUnif? := match biasVals? with
            | none => some 0.0
            | some bv => arrayUniformFloat bv

          match scUnif?, bUnif? with
          | some sVal, some bVal =>
            if sVal == 1.0 && bVal == 0.0 then
              outNodes := outNodes.push {
                name := nd.name
                op := .elemBinary .mul
                inputs := #[diffId, invStdId]
                outputs := #[yOut]
              }
            else
              let normId := nextId
              nextId := nextId + 1
              let sBits ← validateFiniteFp16ScalarIR sVal ctx
              let bBits ← validateFiniteFp16ScalarIR bVal ctx
              let uop :=
                if bVal == 0.0 then ElemUnaryOp.scalarMul sBits
                else if sVal == 1.0 then ElemUnaryOp.scalarAdd bBits
                else ElemUnaryOp.linearAct sBits bBits
              outNodes := outNodes.push {
                name := s!"{nd.name}_norm"
                op := .elemBinary .mul
                inputs := #[diffId, invStdId]
                outputs := #[(normId, xTy)]
              }
              outNodes := outNodes.push {
                name := nd.name
                op := .elemUnary uop
                inputs := #[normId]
                outputs := #[yOut]
              }
          | _, _ =>
            let fullSc := if scVals.size == 1 then Array.replicate normDim scVals[0]! else scVals
            let fullB? := biasVals?.map fun bv => if bv.size == 1 then Array.replicate normDim bv[0]! else bv
            let mut diagW : Array Float := Array.replicate (normDim * normDim) 0.0
            for k in [:normDim] do
              diagW := diagW.set! (k * normDim + k) fullSc[k]!
            let normId := nextId
            let wId := nextId + 1
            nextId := nextId + 2
            outNodes := outNodes.push {
              name := s!"{nd.name}_norm"
              op := .elemBinary .mul
              inputs := #[diffId, invStdId]
              outputs := #[(normId, xTy)]
            }
            outNodes := outNodes.push {
              name := s!"{nd.name}_w"
              op := .const (.f32 diagW)
              inputs := #[]
              outputs := #[(wId, { shape := #[normDim, normDim, 1, 1], dtype := .f32 })]
            }
            let (convInps, hasB) ← match fullB? with
              | some fb =>
                let bId := nextId
                nextId := nextId + 1
                outNodes := outNodes.push {
                  name := s!"{nd.name}_b"
                  op := .const (.f32 fb)
                  inputs := #[]
                  outputs := #[(bId, { shape := #[normDim], dtype := .f32 })]
                }
                pure (#[normId, wId, bId], true)
              | none => pure (#[normId, wId], false)
            if normAxis == 1 then
              outNodes := outNodes.push {
                name := nd.name
                op := .conv 1 1 1 1 #[0, 0, 0, 0] 1 hasB .none
                inputs := convInps
                outputs := #[yOut]
              }
            else
              if !allowViewOps then
                throw s!"{ctx}: LayerNormalization along W with non-uniform constant scale/bias requires --allow-view-ops=true (or runtime scale/bias inputs)"
              if rank != 4 || xTy.shape[2]! != 1 || xTy.shape[1]! != xTy.shape[3]! || normDim > 128 then
                throw s!"{ctx}: LayerNormalization along W with non-uniform constant scale/bias requires C==W<=128 and H==1 (got {xTy.shape.toList}); use runtime scale/bias inputs or normalize along C"
              let tr1Id := nextId
              let affId := nextId + 1
              nextId := nextId + 2
              outNodes := outNodes.push {
                name := s!"{nd.name}_tr1"
                op := .transpose #[0, 3, 2, 1]
                inputs := #[normId]
                outputs := #[(tr1Id, xTy)]
              }
              let affInps := convInps.set! 0 tr1Id
              outNodes := outNodes.push {
                name := s!"{nd.name}_aff"
                op := .conv 1 1 1 1 #[0, 0, 0, 0] 1 hasB .none
                inputs := affInps
                outputs := #[(affId, xTy)]
              }
              outNodes := outNodes.push {
                name := nd.name
                op := .transpose #[0, 3, 2, 1]
                inputs := #[affId]
                outputs := #[yOut]
              }
        | none =>
          -- Runtime Scale (and optional runtime Bias)
          let scTy ← getType scaleId outNodes ctx
          if !isValidLayerNormParamShape scTy.shape normAxis normDim rank then
            throw s!"{ctx}: LayerNormalization runtime Scale shape {scTy.shape.toList} incompatible with normalized suffix X.shape[{ax}:] (normDim={normDim})"
          let normId := nextId
          nextId := nextId + 1
          outNodes := outNodes.push {
            name := s!"{nd.name}_norm"
            op := .elemBinary .mul
            inputs := #[diffId, invStdId]
            outputs := #[(normId, xTy)]
          }
          match biasId? with
          | none =>
            outNodes := outNodes.push {
              name := nd.name
              op := .elemBinary .mul
              inputs := #[normId, scaleId]
              outputs := #[yOut]
            }
          | some bId =>
            let bTy ← getType bId outNodes ctx
            if !isValidLayerNormParamShape bTy.shape normAxis normDim rank then
              throw s!"{ctx}: LayerNormalization runtime B shape {bTy.shape.toList} incompatible with normalized suffix X.shape[{ax}:] (normDim={normDim})"
            let scaledId := nextId
            nextId := nextId + 1
            outNodes := outNodes.push {
              name := s!"{nd.name}_scaled"
              op := .elemBinary .mul
              inputs := #[normId, scaleId]
              outputs := #[(scaledId, xTy)]
            }
            outNodes := outNodes.push {
              name := nd.name
              op := .elemBinary .add
              inputs := #[scaledId, bId]
              outputs := #[yOut]
            }

    | .rmsNorm axis eps hasScale =>
      if inps.size < 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: RMSNorm expects at least 1 input and 1 output"
      let xId := inps[0]!
      let xTy ← getType xId outNodes ctx
      let rank := xTy.shape.size
      let ax := if axis < 0 then axis + (rank : Int) else axis
      if ax < 0 || ax >= (rank : Int) then
        throw s!"{ctx}: RMSNorm axis {axis} out of range for rank {rank}"
      let axNat := ax.toNat
      if eps.isNaN || eps.isInf || eps < 0.0 then
        throw s!"{ctx}: RMSNorm invalid epsilon {eps}"
      let epsBits ← validateFiniteFp16ScalarIR eps ctx
      let redShape := xTy.shape.set! axNat 1
      let redTy : TensorType := { shape := redShape, dtype := xTy.dtype }
      let invRmsId := nextId
      nextId := nextId + 1
      outNodes := outNodes.push {
        name := s!"{nd.name}_invrms"
        op := .reduce .rsqrtMeanSquare #[axNat] true epsBits
        inputs := #[xId]
        outputs := #[(invRmsId, redTy)]
      }
      let yOut := nd.outputs[0]!
      if !hasScale || inps.size < 2 then
        outNodes := outNodes.push {
          name := nd.name
          op := .elemBinary .mul
          inputs := #[xId, invRmsId]
          outputs := #[yOut]
        }
      else
        let scaleId := inps[1]!
        let scTy ← getType scaleId outNodes ctx
        let normDim := xTy.shape[axNat]!
        let scShapeOk : Bool :=
          if axNat == 1 || axNat == rank - 1 then
            isValidLayerNormParamShape scTy.shape axNat normDim rank
          else
            (scTy.shape.size <= rank && scTy.shape.all (· == 1)) ||
            (scTy.shape.size == rank && scTy.shape[axNat]! == normDim &&
              (Array.ofFn (n := rank) fun i => i.val == axNat || scTy.shape[i.val]! == 1).all id)
        if !scShapeOk then
          throw s!"{ctx}: RMSNorm Scale shape {scTy.shape.toList} incompatible with normalized axis {axNat} (normDim={normDim}, input rank {rank})"
        match lookupConstInCore outNodes scaleId with
        | some (_, scPay) =>
          let some scVals := scPay.toFloat64?
            | throw s!"{ctx}: RMSNorm Scale constant must be float"
          match arrayUniformFloat scVals with
          | some sVal =>
            if sVal == 1.0 then
              outNodes := outNodes.push {
                name := nd.name
                op := .elemBinary .mul
                inputs := #[xId, invRmsId]
                outputs := #[yOut]
              }
            else
              let normId := nextId
              nextId := nextId + 1
              let sBits ← validateFiniteFp16ScalarIR sVal ctx
              outNodes := outNodes.push {
                name := s!"{nd.name}_norm"
                op := .elemBinary .mul
                inputs := #[xId, invRmsId]
                outputs := #[(normId, xTy)]
              }
              outNodes := outNodes.push {
                name := nd.name
                op := .elemUnary (.scalarMul sBits)
                inputs := #[normId]
                outputs := #[yOut]
              }
          | none =>
            let normId := nextId
            nextId := nextId + 1
            outNodes := outNodes.push {
              name := s!"{nd.name}_norm"
              op := .elemBinary .mul
              inputs := #[xId, invRmsId]
              outputs := #[(normId, xTy)]
            }
            outNodes := outNodes.push {
              name := nd.name
              op := .elemBinary .mul
              inputs := #[normId, scaleId]
              outputs := #[yOut]
            }
        | none =>
          let normId := nextId
          nextId := nextId + 1
          outNodes := outNodes.push {
            name := s!"{nd.name}_norm"
            op := .elemBinary .mul
            inputs := #[xId, invRmsId]
            outputs := #[(normId, xTy)]
          }
          outNodes := outNodes.push {
            name := nd.name
            op := .elemBinary .mul
            inputs := #[normId, scaleId]
            outputs := #[yOut]
          }

    | .split axis sizes =>
      if inps.size != 1 then
        throw s!"{ctx}: Split expects 1 data input"
      if sizes.size != nd.outputs.size then
        throw s!"{ctx}: Split sizes length {sizes.size} != outputs count {nd.outputs.size}"
      let xId := inps[0]!
      let xTy ← getType xId outNodes ctx
      let rank := xTy.shape.size
      if axis >= rank then
        throw s!"{ctx}: Split axis {axis} out of range for rank {rank}"
      let dim := xTy.shape[axis]!
      let total := sizes.foldl (· + ·) 0
      if total != dim then
        throw s!"{ctx}: Split sum(sizes)={total} != axis {axis} dim {dim}"
      if sizes.size == 1 then
        aliases := aliases.push (nd.outputs[0]!.1, xId)
        elidedCount := elidedCount + 1
      else
        if !allowViewOps then
          throw s!"{ctx}: non-no-op view op Split (axis={axis}, sizes={sizes.toList}) requires --allow-view-ops=true"
        let mut curStart : Nat := 0
        for i in [:sizes.size] do
          let sz := sizes[i]!
          let beg := (Array.replicate rank 0).set! axis curStart
          let en := xTy.shape.set! axis (curStart + sz)
          let step := Array.replicate rank 1
          outNodes := outNodes.push {
            name := s!"{nd.name}_p{i}"
            op := .slice beg en step
            inputs := #[xId]
            outputs := #[nd.outputs[i]!]
          }
          curStart := curStart + sz

  let newOutputs := g.outputs.map (lookupAlias aliases)
  pure ({
    target := g.target
    inputs := g.inputs
    nodes := outNodes
    outputs := newOutputs
  }, elidedCount)

end Vane.IR
