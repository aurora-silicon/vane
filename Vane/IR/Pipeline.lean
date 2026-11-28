import Vane.IR.Passes.DCE
import Vane.IR.Passes.ViewSimplify
import Vane.IR.Passes.ConstFold
import Vane.IR.Passes.HighFuse
import Vane.IR.Passes.Decompose
import Vane.IR.Passes.CoreFuse
import Vane.IR.Passes.DecomposeConv
import Vane.IR.Passes.Quantize

namespace Vane.IR

open Vane.Quant

-- ============================================================================
-- Standard L1 TensorIR Pass Pipeline (`HighOp → CoreOp`)
-- ============================================================================

/-- Configuration flags for the L1 `TensorIR` optimization & lowering pipeline. -/
structure PassConfig where
  enableFusion : Bool := true
  allowViewOps : Bool := true
  quantize : Option QuantFormat := none
  viewSimplifyFuel : Nat := 0
  deriving BEq, Inhabited

/-- Summary statistics collected across the L1 pass pipeline. -/
structure PipelineStats where
  fusedBnCount : Nat := 0
  fusedActCount : Nat := 0
  fusedMatmulCount : Nat := 0
  elidedCount : Nat := 0
  quantizedConvCount : Nat := 0
  deriving Repr, BEq, Inhabited

/-- Result of running `runStandardPipeline` on an `SSAGraph HighOp`. -/
structure PipelineResult where
  highGraph : SSAGraph HighOp
  coreGraph : SSAGraph CoreOp
  stats : PipelineStats
  deriving BEq

/-- Run the complete 7-pass L1 `TensorIR` pipeline:
    1. Validate input `SSAGraph HighOp`
    2. High-level `DCE` → `ViewSimplify` → `ConstFold` → `HighFuse` (if enabled)
    3. `Decompose`: `SSAGraph HighOp → SSAGraph CoreOp`
    4. Core-level `DCE` → `ViewSimplify` → `ConstFold` → `CoreFuse` (if enabled)
    5. `Quantize` (QDQ + optional `--quantize` PTQ) → `DCE` → `ViewSimplify` → `ConstFold`
    6. Validate final `SSAGraph CoreOp` -/
def runStandardPipeline (g : SSAGraph HighOp) (cfg : PassConfig := {}) :
    Except String PipelineResult := do
  validateGraph g

  -- L1a: HighOp passes
  let g1 := dce g
  let g2 := viewSimplify g1 cfg.viewSimplifyFuel
  let g3 ← constFold g2
  let (g4, bnCnt, hiActCnt) ←
    if cfg.enableFusion then highFuse g3
    else pure (g3, 0, 0)

  -- L1a -> L1b: Decompose HighOp -> CoreOp
  let (c0, decElided) ← decomposeGraph g4 cfg.allowViewOps

  -- L1b: CoreOp passes
  let c1 := viewSimplify (dce c0) cfg.viewSimplifyFuel
  let c2 ← constFold c1
  let (c3, coreActCnt, mmCnt) ←
    if cfg.enableFusion then coreFuse c2
    else pure (c2, 0, 0)
  let c3b ← decomposeConvPass c3
  let (c4, qCnt) ← quantizePass c3b cfg.quantize
  let c5 ← constFold (viewSimplify (dce c4) cfg.viewSimplifyFuel)

  validateGraph c5

  pure {
    highGraph := g
    coreGraph := c5
    stats := {
      fusedBnCount := bnCnt
      fusedActCount := hiActCnt + coreActCnt
      fusedMatmulCount := mmCnt
      elidedCount := decElided
      quantizedConvCount := qCnt
    }
  }

end Vane.IR
