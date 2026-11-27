import Vane.Binary
import Vane.TaskStream

namespace Vane.Tiling

open Vane.Binary
open Vane.TaskStream

/-- Hardware L2 SRAM capacity in bytes (2 MiB). -/
def l2CapacityBytes : Nat := 0x200000

/-- Hardware per-chunk kernel size limit in bytes (64 KiB). -/
def maxKernelChunkBytes : Nat := 0x10000

/-- Mode for single-task fp16 convolution (`dense` or `depthwise`). -/
inductive Conv3x3Mode where
  | dense
  | depthwise
  deriving Repr, BEq, DecidableEq, Inhabited

namespace Conv3x3Mode

def name : Conv3x3Mode → String
  | .dense => "dense"
  | .depthwise => "depthwise"

def parse? : String → Option Conv3x3Mode
  | "dense" => some .dense
  | "depthwise" => some .depthwise
  | _ => none

end Conv3x3Mode

/-- Padding mode for single-task and stripped fp16 3x3 convolution (`same`, `valid`, or `strip pt pb`). -/
inductive PadMode where
  | same
  | valid
  | strip (padTop padBottom : Nat)
  deriving Repr, BEq, DecidableEq, Inhabited

namespace PadMode

def name : PadMode → String
  | .same => "same"
  | .valid => "valid"
  | .strip pt pb => s!"strip({pt},{pb})"

def parse? : String → Option PadMode
  | "same" => some .same
  | "valid" => some .valid
  | _ => none

end PadMode

/-- Spatial and channel shape for a single-task fp16 1x1 convolution. -/
structure Conv1x1Shape where
  h : Nat
  w : Nat
  cin : Nat
  cout : Nat
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Shape and parameters for a single-task fp16 3x3 convolution. -/
structure Conv3x3Shape where
  mode : Conv3x3Mode
  h : Nat
  w : Nat
  cin : Nat
  cout : Nat
  stride : Nat
  pad : PadMode
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Unified specification for a single-task fp16 NE convolution (`kernelSize in {1, 3}`). -/
structure ConvSpec where
  kernelSize : Nat
  mode : Conv3x3Mode := .dense
  h : Nat
  w : Nat
  cin : Nat
  cout : Nat
  stride : Nat := 1
  pad : PadMode := .same
  deriving Repr, BEq, DecidableEq, Inhabited

def Conv1x1Shape.toConvSpec (sh : Conv1x1Shape) : ConvSpec := {
  kernelSize := 1
  mode := .dense
  h := sh.h
  w := sh.w
  cin := sh.cin
  cout := sh.cout
  stride := 1
  pad := .same
}

def Conv3x3Shape.toConvSpec (sh : Conv3x3Shape) : ConvSpec := {
  kernelSize := 3
  mode := sh.mode
  h := sh.h
  w := sh.w
  cin := sh.cin
  cout := sh.cout
  stride := sh.stride
  pad := sh.pad
}

/-- Tiling configuration for a single-task fp16 NE convolution. -/
structure Tiling where
  logNE : Nat
  f : Nat
  fat : Nat
  ocg : Nat
  shPref : Nat
  shMax : Nat
  tileHeight : Nat
  srcSkew : Nat := 0
  resSkew : Nat := 0
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Ceiling base-2 logarithm: `ceilLog2 1 = 0`, `ceilLog2 n = floor(log2(n - 1)) + 1` for `n > 1`. -/
def ceilLog2 (n : Nat) : Nat :=
  if n <= 1 then 0 else Nat.log2 (n - 1) + 1

/-- Ceiling integer division for positive denominator `b > 0`. -/
def ceilDiv (a b : Nat) : Nat :=
  if b == 0 then 0 else (a + b - 1) / b

/-- Work-unit split exponent `WU = logNE - F - fat` (valid when `F + fat <= logNE`). -/
def Tiling.wu (t : Tiling) : Nat :=
  t.logNE - t.f - t.fat

/-- Preferred work-unit strip width in pixels: `T = (128 << fat) >> shPref`. -/
def stripWidth (t : Tiling) : Nat :=
  (128 <<< t.fat) >>> t.shPref

/-- Compute the output channel blocks `(o0, count)` assigned to group `g` (`0 <= g < 2^F`)
    under standard round blocking. -/
def groupOutputBlocks (cout f ocg g : Nat) : Array (Nat × Nat) :=
  let groups := (1 : Nat) <<< f
  let blk := (1 : Nat) <<< ocg
  let round := groups * blk
  let full := cout / round
  let rem := cout % round
  let fullBlocks : Array (Nat × Nat) :=
    Array.ofFn (n := full) fun r => (r.val * round + g * blk, blk)
  if rem > 0 then
    let base := rem / groups
    let extra := rem % groups
    let lastSize := base + (if g < extra then 1 else 0)
    let lastStart := g * base + min g extra
    if lastSize > 0 then
      fullBlocks.push (full * round + lastStart, lastSize)
    else
      fullBlocks
  else
    fullBlocks

/-- Total output channels assigned to group `g` by `groupOutputBlocks`. -/
def groupOutputCount (cout f ocg g : Nat) : Nat :=
  (groupOutputBlocks cout f ocg g).foldl (fun acc (_, n) => acc + n) 0

-- ============================================================================
-- Unified ConvSpec spatial geometry
-- ============================================================================

namespace ConvSpec

/-- Effective hardware mode: 1x1 and single-channel depthwise (`C == 1`) run as dense. -/
def effMode (sh : ConvSpec) : Conv3x3Mode :=
  if sh.kernelSize == 1 || sh.mode == .dense || sh.cin == 1 then .dense else .depthwise

/-- Whether stride-2 zero-skipping sparse kernel encoding is used. -/
def isSparse (sh : ConvSpec) : Bool :=
  sh.kernelSize == 3 && sh.stride == 2

/-- Input channels per output kernel (`1` for true depthwise, `Cin` for dense). -/
def cinW (sh : ConvSpec) : Nat :=
  if sh.effMode == .depthwise then 1 else sh.cin

/-- Spatial taps per input channel (`1` for 1x1, `9` for 3x3). -/
def taps (sh : ConvSpec) : Nat :=
  sh.kernelSize * sh.kernelSize

/-- Output spatial height `Ho`. -/
def outH (sh : ConvSpec) : Nat :=
  if sh.kernelSize == 1 then sh.h
  else if sh.stride == 0 then 0
  else
    match sh.pad with
    | .same => ceilDiv sh.h sh.stride
    | .valid => if sh.h < 3 then 0 else (sh.h - 3) / sh.stride + 1
    | .strip pt pb =>
      if sh.stride != 1 || pt > 1 || pb > 1 || sh.h + pt + pb < 3 then 0
      else sh.h + pt + pb - 2

/-- Output spatial width `Wo`. -/
def outW (sh : ConvSpec) : Nat :=
  if sh.kernelSize == 1 then sh.w
  else if sh.stride == 0 then 0
  else
    match sh.pad with
    | .same => ceilDiv sh.w sh.stride
    | .valid => if sh.w < 3 then 0 else (sh.w - 3) / sh.stride + 1
    | .strip _ _ => if sh.stride != 1 then 0 else sh.w

/-- Top padding `pt` (`0` for 1x1 and `valid`, floor of half total padding for 3x3 `same`, explicit `pt` for `strip`). -/
def padTop (sh : ConvSpec) : Nat :=
  if sh.kernelSize == 1 then 0
  else
    match sh.pad with
    | .valid => 0
    | .same =>
      let ho := sh.outH
      if ho == 0 then 0 else ((ho - 1) * sh.stride + 3 - sh.h) / 2
    | .strip pt _ => pt

/-- Left padding `pl` (`0` for 1x1 and `valid`, floor of half total padding for 3x3 `same`, `1` for `strip`). -/
def padLeft (sh : ConvSpec) : Nat :=
  if sh.kernelSize == 1 then 0
  else
    match sh.pad with
    | .valid => 0
    | .same =>
      let wo := sh.outW
      if wo == 0 then 0 else ((wo - 1) * sh.stride + 3 - sh.w) / 2
    | .strip _ _ => 1

/-- Phase-rounded kernel extent `rk` (`1` for 1x1, `align_up(3, stride)` for 3x3). -/
def rk (sh : ConvSpec) : Nat :=
  if sh.kernelSize == 1 then 1 else alignUp 3 sh.stride

/-- Cropped input width `cropW`. -/
def cropW (sh : ConvSpec) : Nat :=
  if sh.kernelSize == 1 then sh.w
  else
    let wo := sh.outW
    if wo == 0 then 0 else min sh.w ((wo - 1) * sh.stride + sh.rk - sh.padLeft)

/-- Cropped input height `cropH`. -/
def cropH (sh : ConvSpec) : Nat :=
  if sh.kernelSize == 1 then sh.h
  else
    let ho := sh.outH
    if ho == 0 then 0 else min sh.h ((ho - 1) * sh.stride + sh.rk - sh.padTop)

/-- Register `0x0004` (`Win`) value: `W` for 1x1 or `s = 1`, `cropW` for 3x3 `s = 2`. -/
def winReg (sh : ConvSpec) : Nat :=
  if sh.kernelSize == 1 || sh.stride == 1 then sh.w else sh.cropW

/-- Register `0x0008` (`Hin`) value: `H` for 1x1 or `s = 1`, `cropH` for 3x3 `s = 2`. -/
def hinReg (sh : ConvSpec) : Nat :=
  if sh.kernelSize == 1 || sh.stride == 1 then sh.h else sh.cropH

end ConvSpec

-- ============================================================================
-- Unified block assignment, L2 staging geometry, and width-cover functions
-- ============================================================================

/-- Effective `ocg` (`0` for true depthwise, `t.ocg` for dense). -/
def effOcgConv (sh : ConvSpec) (t : Tiling) : Nat :=
  if sh.effMode == .depthwise then 0 else t.ocg

/-- Output channel blocks `(o0, n)` assigned to group `g` (`0 <= g < 2^F`).
    True depthwise with `F == 4` assigns `o % 16 == g` as `(o, 1)` blocks in ascending order;
    all other cases use the standard round-blocking formula with `(Cout, F, effOcg)`. -/
def groupOutputBlocksConv (sh : ConvSpec) (t : Tiling) (g : Nat) : Array (Nat × Nat) :=
  if sh.effMode == .depthwise && t.f == 4 then
    Id.run do
      let mut blocks : Array (Nat × Nat) := #[]
      let mut o := g
      while o < sh.cout do
        blocks := blocks.push (o, 1)
        o := o + 16
      return blocks
  else
    groupOutputBlocks sh.cout t.f (effOcgConv sh t) g

/-- Compute `maxBlockSlots` across all output blocks of all groups `g in 0..2^F-1` for stride-1 3x3:
    `blockSlots(n) = Cin * 9 * align_up(n, 2)`. -/
def maxBlockSlotsConv (sh : ConvSpec) (t : Tiling) : Nat := Id.run do
  let groups := (1 : Nat) <<< t.f
  let mut best := 0
  for g in [:groups] do
    let blocks := groupOutputBlocksConv sh t g
    for (_, n) in blocks do
      let slots := sh.cin * 9 * alignUp n 2
      if slots > best then
        best := slots
  return best

/-- Input rows read by output row tile `tileIdx` (`0 <= tileIdx < ceil(Ho / th)`) for 3x3 conv. -/
def tileInRows3x3 (h s pt ho th tileIdx : Nat) : Nat :=
  let rk := alignUp 3 s
  let y0 := tileIdx * th
  let y1 := min ho (y0 + th)
  if y1 <= y0 then 0
  else
    let hi := min h ((y1 - 1) * s + rk - pt)
    let lo := y0 * s - pt
    hi - lo

/-- Maximum input rows buffered per row tile in O(1) time (exact max over tiles `0`, `1`, `nTiles - 1`). -/
def inRows3x3Raw (h s pt ho th : Nat) : Nat :=
  if th == 0 || ho == 0 then 0
  else
    let nTiles := ceilDiv ho th
    let r0 := tileInRows3x3 h s pt ho th 0
    if nTiles <= 1 then r0
    else
      let r1 := tileInRows3x3 h s pt ho th 1
      if nTiles == 2 then max r0 r1
      else
        let rLast := tileInRows3x3 h s pt ho th (nTiles - 1)
        max (max r0 r1) rLast

/-- Source strip width `srcPx` in pixels for 1x1 or 3x3 conv. Never narrow below this formula. -/
def srcPxConv (sh : ConvSpec) (t : Tiling) : Nat :=
  let stripW := stripWidth t
  if sh.kernelSize == 1 then
    min (alignUp sh.w 8) (max stripW 32)
  else
    let ti := sh.stride * stripW
    let r := alignUp sh.winReg 8
    let halo : Int := if sh.padLeft > 0 then 8 else 0
    let flr : Int := if ti > 8 then ((min ti 32 : Nat) : Int) else 24
    let lookahead : Int := halo + min ((r : Int) - 2 * ((max ti 16 : Nat) : Int)) 32
    let extra : Nat := (max flr lookahead).toNat
    min r (ti + extra)

/-- Bytes per channel row in L2 source staging (`chs = 2 * srcPx`). -/
def chsBytesConv (sh : ConvSpec) (t : Tiling) : Nat :=
  2 * srcPxConv sh t

/-- Source row stride in L2 SRAM (`Cin * chs + 16 * srcSkew`). -/
def srcRowStrideConv (sh : ConvSpec) (t : Tiling) : Nat :=
  sh.cin * chsBytesConv sh t + 16 * t.srcSkew

/-- Maximum input rows buffered in L2 per output row tile of height `th`. -/
def inRowsConv (sh : ConvSpec) (th : Nat) : Nat :=
  if sh.kernelSize == 1 then min th sh.h
  else inRows3x3Raw sh.h sh.stride sh.padTop sh.outH th

/-- Whether result staging operates in direct mode (`!staged`), evaluated on `Wo`. -/
def isDirectConv (target : Target) (sh : ConvSpec) (t : Tiling) : Bool :=
  target.profile.directThreshold t.fat t.shMax >= min 32 sh.outW

/-- Bytes per channel row in L2 result staging (`dchs = 2 * min(align_up(Wo, 8), max(T, 32))`). -/
def dchsBytesConv (sh : ConvSpec) (t : Tiling) : Nat :=
  2 * min (alignUp sh.outW 8) (max (stripWidth t) 32)

/-- Result row stride in L2 SRAM (staged mode): `Cout * dchs + 16 * resSkew`. -/
def resRowStrideConv (sh : ConvSpec) (t : Tiling) : Nat :=
  sh.cout * dchsBytesConv sh t + 16 * t.resSkew

/-- Bytes of L2 SRAM occupied by source staging. -/
def srcL2BytesConv (sh : ConvSpec) (t : Tiling) : Nat :=
  inRowsConv sh t.tileHeight * srcRowStrideConv sh t

/-- 64-byte aligned L2 SRAM byte offset of result staging (`alignUp srcL2Bytes 64`). -/
def resBaseBytesConv (sh : ConvSpec) (t : Tiling) : Nat :=
  alignUp (srcL2BytesConv sh t) 64

/-- Bytes of L2 SRAM occupied by result staging. -/
def resL2BytesConv (target : Target) (sh : ConvSpec) (t : Tiling) : Nat :=
  if isDirectConv target sh t then
    ((1 : Nat) <<< (t.f + effOcgConv sh t + t.fat)) * 512
  else
    min t.tileHeight sh.outH * resRowStrideConv sh t

/-- Total L2 SRAM footprint (`alignUp srcL2Bytes 64 + resL2Bytes`). -/
def l2UseBytesConv (target : Target) (sh : ConvSpec) (t : Tiling) : Nat :=
  resBaseBytesConv sh t + resL2BytesConv target sh t

/-- Smallest `i in 0..5` such that `wd % (base >> i) == 0`, or `6` if none. -/
def firstDiv (wd base : Nat) : Nat := Id.run do
  for i in [:6] do
    let d := base >>> i
    if d > 0 && wd % d == 0 then
      return i
  return 6

/-- Width-cover exponent for aligned width `wAligned`, strip width `stripW`, and `base = 128 << fat`.
    Equivalent to `min(4, max_{x = 0, T, 2T, ... < W'} firstDiv(min(T, W' - x), base))`. -/
def cover (wAligned stripW base : Nat) : Nat :=
  if stripW == 0 || wAligned == 0 then
    6
  else if wAligned <= stripW then
    min 4 (firstDiv wAligned base)
  else
    let fullDiv := firstDiv stripW base
    let rem := wAligned % stripW
    let mx := if rem == 0 then fullDiv else max fullDiv (firstDiv rem base)
    min 4 mx

/-- Minimum cover exponent across alignment multiples `{8, 16, 32, 64}`. -/
def coverMin (w stripW fat : Nat) : Nat :=
  let wuBase : Nat := (128 : Nat) <<< fat
  let c8 := cover (alignUp w 8) stripW wuBase
  let c16 := cover (alignUp w 16) stripW wuBase
  let c32 := cover (alignUp w 32) stripW wuBase
  let c64 := cover (alignUp w 64) stripW wuBase
  min (min c8 c16) (min c32 c64)

-- ============================================================================
-- Unified Safety Validator vs Conservative Chooser Policy
-- ============================================================================

/-- Validate basic NE conv shape bounds and kernel/mode/padding constraints. -/
def validateConvShape (sh : ConvSpec) : Except String Unit := do
  if sh.kernelSize != 1 && sh.kernelSize != 3 then
    throw s!"invalid kernelSize={sh.kernelSize}: must be 1 or 3"
  if sh.h == 0 || sh.h > 65535 then
    throw s!"invalid H={sh.h}: must be in 1..65535"
  if sh.w == 0 || sh.w > 65535 then
    throw s!"invalid W={sh.w}: must be in 1..65535"
  if sh.cin == 0 || sh.cin > maxU32Nat then
    throw s!"invalid Cin={sh.cin}: must be in 1..{maxU32Nat}"
  if sh.cout == 0 || sh.cout > maxU32Nat then
    throw s!"invalid Cout={sh.cout}: must be in 1..{maxU32Nat}"
  if sh.kernelSize == 1 then
    if sh.mode != .dense || sh.stride != 1 || sh.pad != .same then
      throw s!"invalid 1x1 conv spec: requires dense mode, stride=1, pad=same"
  else
    if sh.stride != 1 && sh.stride != 2 then
      throw s!"invalid stride={sh.stride}: must be 1 or 2"
    if sh.mode == .depthwise && sh.cin != sh.cout then
      throw s!"invalid depthwise shape: Cin={sh.cin} != Cout={sh.cout}"
    match sh.pad with
    | .valid =>
      if sh.h < 3 || sh.w < 3 then
        throw s!"invalid valid-padded 3x3 shape ({sh.h}x{sh.w}): requires H >= 3 and W >= 3"
    | .strip pt pb =>
      if sh.stride != 1 then
        throw s!"invalid strip-padded 3x3 stride={sh.stride}: requires stride=1"
      if pt > 1 || pb > 1 then
        throw s!"invalid strip padding (pt={pt}, pb={pb}): requires pt <= 1 and pb <= 1"
      if sh.h + pt + pb < 3 then
        throw s!"invalid strip-padded 3x3 height H={sh.h} with (pt={pt}, pb={pb}): requires H + pt + pb >= 3"
    | .same => pure ()
    if sh.outH == 0 || sh.outW == 0 then
      throw s!"invalid 3x3 output shape ({sh.outH}x{sh.outW}): Ho and Wo must be >= 1"
  pure ()

/-- Compute the 64B-aligned `__KERN_0` dense chunk byte size for output group `g`. -/
def convDenseGroupChunkBytes
    (sh : ConvSpec) (t : Tiling)
    (extraChunkBytes : Nat := 0) (hasBiasInKernel : Bool := false) (g : Nat) : Nat :=
  let wordsPerOut := sh.cinW * sh.taps + (if hasBiasInKernel then 1 else 0)
  let blocks := groupOutputBlocksConv sh t g
  let groupOuts : Nat := blocks.foldl (fun (acc : Nat) (_, n) => acc + n) 0
  alignUp (extraChunkBytes + 2 * groupOuts * wordsPerOut) 64

/-- Total recursive helper checking every group `g ∈ [g0, groups)` against `maxKernelChunkBytes`. -/
def checkConvDenseChunks64KLoop
    (sh : ConvSpec) (t : Tiling)
    (extraChunkBytes : Nat) (hasBiasInKernel : Bool)
    (groups g : Nat) : Except String Unit :=
  if g < groups then
    let blocks := groupOutputBlocksConv sh t g
    let groupOuts : Nat := blocks.foldl (fun (acc : Nat) (_, n) => acc + n) 0
    if groupOuts > 0 then
      let chunkBytes := convDenseGroupChunkBytes sh t extraChunkBytes hasBiasInKernel g
      if chunkBytes > maxKernelChunkBytes then
        .error s!"illegal tiling: group {g} weight chunk {chunkBytes}B exceeds {maxKernelChunkBytes}B (64 KiB)"
      else
        checkConvDenseChunks64KLoop sh t extraChunkBytes hasBiasInKernel groups (g + 1)
    else
      checkConvDenseChunks64KLoop sh t extraChunkBytes hasBiasInKernel groups (g + 1)
  else
    .ok ()
termination_by groups - g

/-- Check stride-1 dense per-group kernel chunk size cap (`<= 64 KiB`), accounting for
    optional prepended LUT overhead (`extraChunkBytes`) and kernel bias words. -/
def checkConvDenseChunks64K
    (sh : ConvSpec)
    (t : Tiling)
    (extraChunkBytes : Nat := 0)
    (hasBiasInKernel : Bool := false) : Except String Unit :=
  if sh.stride == 1 then
    checkConvDenseChunks64KLoop sh t extraChunkBytes hasBiasInKernel ((1 : Nat) <<< t.f) 0
  else
    .ok ()

/-- Hardware `ocg` safety bounds for single-task NE conv. -/
def validateConvOcgSafety (target : Target) (sh : ConvSpec) (t : Tiling) : Except String Unit :=
  match sh.effMode with
  | .depthwise =>
    if t.ocg != 0 then
      .error s!"illegal tiling: depthwise mode requires ocg=0, got ocg={t.ocg}"
    else .ok ()
  | .dense =>
    let maxOcg :=
      if sh.kernelSize == 3 && sh.stride == 1 then
        target.profile.dense3x3Stride1MaxOcg
      else
        4
    if t.ocg > maxOcg then
      .error s!"illegal tiling: {target.name} ocg={t.ocg} exceeds maximum {maxOcg}"
    else .ok ()

/-- Hardware output channel coverage bounds for single-task NE conv. -/
def validateConvCoverageSafety (target : Target) (sh : ConvSpec) (t : Tiling) : Except String Unit :=
  match sh.effMode with
  | .depthwise => .ok ()
  | .dense =>
    let prof := target.profile
    if sh.kernelSize == 1 then
      let maxCout := prof.conv1x1MaxCout t.f t.ocg
      if sh.cout > maxCout then
        .error s!"illegal tiling: Cout={sh.cout} exceeds {prof.name} coverage {maxCout} (F={t.f}, ocg={t.ocg})"
      else .ok ()
    else
      let singleCap := (1 : Nat) <<< (t.f + t.ocg)
      if sh.stride == 2 then
        if sh.cout > singleCap then
          .error s!"illegal tiling: stride-2 dense Cout={sh.cout} exceeds single-round coverage {singleCap} (F={t.f}, ocg={t.ocg})"
        else .ok ()
      else
        let multiRoundOk := prof.dense3x3Stride1MultiRoundF4Ocg3 && t.f == 4 && t.ocg == 3
        if sh.cout > singleCap && !multiRoundOk then
          .error s!"illegal tiling: {prof.name} stride-1 dense Cout={sh.cout} exceeds coverage {singleCap} (F={t.f}, ocg={t.ocg})"
        else .ok ()

/-- Hard silicon safety validator for single-task fp16 NE convolutions (1x1 and 3x3).
    Enforces only hardware-required invariants (register field limits, WU <= 3, ocg bounds,
    channel coverage, strip width, shPref <= shMax, tileHeight >= 1, L2 <= 2 MiB, and
    stride-1 per-group chunk size <= 64 KiB), omitting chooser-only policy rules. -/
def validateConvSafety
    (target : Target) (sh : ConvSpec) (t : Tiling)
    (extraChunkBytes : Nat := 0) (hasBiasInKernel : Bool := false) : Except String Unit :=
  validateConvShape sh >>= fun () =>
  if t.logNE > 4 then
    .error s!"illegal tiling: logNE={t.logNE} exceeds 4"
  else if t.fat > 1 then
    .error s!"illegal tiling: fat={t.fat} must be 0 or 1"
  else if t.f + t.fat > t.logNE then
    .error s!"illegal tiling: F({t.f}) + fat({t.fat}) > logNE({t.logNE})"
  else if t.wu > 3 then
    .error s!"illegal tiling: WU={t.wu} exceeds 2-bit limit 3 (logNE={t.logNE}, F={t.f}, fat={t.fat})"
  else
    validateConvOcgSafety target sh t >>= fun () =>
    validateConvCoverageSafety target sh t >>= fun () =>
    let stripW := stripWidth t
    if stripW < 8 || stripW > 256 then
      .error s!"illegal tiling: strip width T={stripW} outside [8, 256]"
    else if t.shPref > t.shMax then
      .error s!"illegal tiling: shifts shPref={t.shPref} > shMax={t.shMax}"
    else if t.shMax > 4 then
      .error s!"illegal tiling: shMax={t.shMax} exceeds maximum 4"
    else if t.tileHeight == 0 then
      .error s!"illegal tiling: tileHeight=0 must be >= 1"
    else if l2UseBytesConv target sh t > l2CapacityBytes then
      .error s!"illegal tiling: L2 footprint {l2UseBytesConv target sh t} bytes exceeds {l2CapacityBytes} bytes (2 MiB)"
    else
      checkConvDenseChunks64K sh t extraChunkBytes hasBiasInKernel

/-- Conservative chooser policy constraints layered on top of `validateConvSafety`. -/
def validateConvChooserPolicy (sh : ConvSpec) (t : Tiling) : Except String Unit := do
  let eff := sh.effMode
  let ho := sh.outH
  let wo := sh.outW

  -- Group bound (no empty groups beyond ceil_log2(Cout); depthwise C >= 16 fixes F = 4)
  let maxF := min 4 (ceilLog2 sh.cout)
  if t.f > maxF then
    throw s!"chooser policy: F={t.f} exceeds min(4, ceil_log2(Cout={sh.cout}))={maxF}"
  if eff == .depthwise && sh.cout >= 16 && t.f != 4 then
    throw s!"chooser policy: depthwise with C={sh.cout} >= 16 requires F=4, got F={t.f}"

  -- Width cover rule on output width Wo
  let stripW := stripWidth t
  let cMin := coverMin wo stripW t.fat
  if cMin > t.shMax then
    throw s!"chooser policy: coverMin(Wo={wo}, T={stripW}, fat={t.fat})={cMin} > shMax={t.shMax}"

  -- Tile height bounded by output height Ho
  if t.tileHeight > ho then
    throw s!"chooser policy: tileHeight={t.tileHeight} exceeds Ho={ho}"

  pure ()

/-- Full tiling legality check combining silicon safety (`validateConvSafety`) and
    conservative chooser policy (`validateConvChooserPolicy`). -/
def validateConvTiling
    (target : Target) (sh : ConvSpec) (t : Tiling)
    (extraChunkBytes : Nat := 0) (hasBiasInKernel : Bool := false) : Except String Unit := do
  validateConvSafety target sh t extraChunkBytes hasBiasInKernel
  validateConvChooserPolicy sh t

-- ============================================================================
-- Unified Independent Deterministic O(1)-Height Tile Chooser
-- ============================================================================

/-- Choose legal `ocg` for `(target, sh, f)`. -/
def chooseConvOcg? (target : Target) (sh : ConvSpec) (f : Nat) : Option Nat :=
  match sh.effMode with
  | .depthwise => some 0
  | .dense =>
    let prof := target.profile
    let need := ceilDiv sh.cout ((1 : Nat) <<< f)
    let ocg1 := ceilLog2 need
    if sh.kernelSize == 1 then
      if ocg1 <= 4 then
        some ocg1
      else if sh.cout <= prof.conv1x1MaxCout f 4 then
        some 4
      else
        none
    else if sh.stride == 1 then
      if ocg1 <= prof.dense3x3Stride1MaxOcg then
        some ocg1
      else if prof.dense3x3Stride1MultiRoundF4Ocg3 && f == 4 then
        some 3
      else
        none
    else
      if ocg1 <= 4 then some ocg1 else none

/-- Compute maximal legal `tileHeight` in O(1) for a candidate conv tiling configuration. -/
def maxTileHeightConv (target : Target) (sh : ConvSpec) (probe : Tiling) : Nat := Id.run do
  let ho := sh.outH
  if ho == 0 then
    return 0
  let srcRow := srcRowStrideConv sh probe
  let direct := isDirectConv target sh probe

  if sh.kernelSize == 1 then
    let thBound1 : Nat :=
      if direct then
        let resBytes := ((1 : Nat) <<< (probe.f + effOcgConv sh probe + probe.fat)) * 512
        if resBytes >= l2CapacityBytes || srcRow == 0 then 0
        else min ho ((l2CapacityBytes - resBytes) / srcRow)
      else
        let perRow := srcRow + resRowStrideConv sh probe
        if perRow == 0 then 0
        else min ho (l2CapacityBytes / perRow)
    for dec in [:4] do
      if thBound1 > dec then
        let th := thBound1 - dec
        if th >= 1 then
          let cand : Tiling := { probe with tileHeight := th }
          if l2UseBytesConv target sh cand <= l2CapacityBytes then
            return th
    return 0

  -- 3x3 O(1) maximal tile height:
  -- Step 1: test th = Ho directly
  let fullCand : Tiling := { probe with tileHeight := ho }
  if l2UseBytesConv target sh fullCand <= l2CapacityBytes then
    return ho

  -- Step 2: compute exact tight upper bound thBound using inRows(th) >= (th - 1) * s + rk - pt
  let s := sh.stride
  let rk := sh.rk
  let pt := sh.padTop
  let thBound : Nat :=
    if direct then
      let resBytes := ((1 : Nat) <<< (probe.f + effOcgConv sh probe + probe.fat)) * 512
      if resBytes >= l2CapacityBytes || srcRow == 0 then 0
      else
        let maxInRows := (l2CapacityBytes - resBytes) / srcRow
        if maxInRows + pt < rk then 0
        else min ho ((maxInRows + pt - rk) / s + 1)
    else
      let resRow := resRowStrideConv sh probe
      let fixedSrc := (rk - pt - s) * srcRow
      let perTh := s * srcRow + resRow
      if fixedSrc >= l2CapacityBytes || perTh == 0 then 0
      else min ho ((l2CapacityBytes - fixedSrc) / perTh)

  if thBound == 0 then
    return 0

  -- Step 3: refine over at most {thBound, thBound - 1, thBound - 2, thBound - 3}
  for dec in [:4] do
    if thBound > dec then
      let th := thBound - dec
      if th >= 1 then
        let cand : Tiling := { probe with tileHeight := th }
        if l2UseBytesConv target sh cand <= l2CapacityBytes then
          return th
  return 0

/-- Return the first `some y` produced by `f` over `xs`. -/
def findFirstSome {α β : Type} : List α → (α → Option β) → Option β
  | [], _ => none
  | x :: rest, f =>
    match f x with
    | some y => some y
    | none => findFirstSome rest f

/-- Ascending list of natural numbers `[lo, hi)`. -/
def natRange (lo hi : Nat) : List Nat :=
  if _ : lo < hi then
    lo :: natRange (lo + 1) hi
  else
    []
termination_by hi - lo

/-- Deterministic independent tile chooser for single-task fp16 NE convolutions (1x1 and 3x3).
    Computes maximal legal `tileHeight` in O(1) per candidate and validates through `validateConvTiling`.
    Pass `extraChunkBytes := 128` when fusing a 128-byte LUT into every kernel chunk. -/
def chooseConvTiling
    (target : Target) (sh : ConvSpec)
    (extraChunkBytes : Nat := 0) (hasBiasInKernel : Bool := false) : Except String Tiling :=
  validateConvShape sh >>= fun () =>
  let eff := sh.effMode
  let wo := sh.outW
  let maxF := min 4 (ceilLog2 sh.cout)
  let fCandidates : List Nat :=
    if eff == .depthwise && sh.cout >= 16 then
      [4]
    else
      (natRange 0 (maxF + 1)).map (fun step => maxF - step)
  let logNEs : List Nat := [4, 3, 2, 1, 0]
  let fats : List Nat := [0, 1]
  let chosen? : Option Tiling :=
    findFirstSome logNEs fun logNE =>
      findFirstSome fats fun fat =>
        findFirstSome fCandidates fun f =>
          if f + fat <= logNE && (logNE - f - fat) <= 3 then
            match chooseConvOcg? target sh f with
            | none => none
            | some ocg =>
              findFirstSome (natRange 0 5) fun shPref =>
                let stripW := (128 <<< fat) >>> shPref
                if stripW >= 8 && stripW <= 256 then
                  let cMin := coverMin wo stripW fat
                  let shMaxStart := max shPref cMin
                  if shMaxStart <= 4 then
                    findFirstSome (natRange shMaxStart 5) fun shMax =>
                      let probe : Tiling := {
                        logNE := logNE
                        f := f
                        fat := fat
                        ocg := ocg
                        shPref := shPref
                        shMax := shMax
                        tileHeight := 1
                        srcSkew := 0
                        resSkew := 0
                      }
                      let maxTh := maxTileHeightConv target sh probe
                      if maxTh >= 1 then
                        let cand : Tiling := { probe with tileHeight := maxTh }
                        match validateConvTiling target sh cand extraChunkBytes hasBiasInKernel with
                        | .ok () => some cand
                        | .error _ => none
                      else
                        none
                  else
                    none
                else
                  none
          else
            none
  match chosen? with
  | some cand => .ok cand
  | none =>
    .error s!"no legal {target.name} tiling found for {sh.kernelSize}x{sh.kernelSize} {sh.mode.name} conv shape (H={sh.h}, W={sh.w}, Cin={sh.cin}, Cout={sh.cout}, s={sh.stride}, pad={sh.pad.name})"

-- ============================================================================
-- Compatibility wrappers for Conv1x1Shape and Conv3x3Shape callers
-- ============================================================================

def srcPx (sh : Conv1x1Shape) (t : Tiling) : Nat := srcPxConv sh.toConvSpec t
def chsBytes (sh : Conv1x1Shape) (t : Tiling) : Nat := chsBytesConv sh.toConvSpec t
def dchsBytes (sh : Conv1x1Shape) (t : Tiling) : Nat := dchsBytesConv sh.toConvSpec t
def srcRowStride (sh : Conv1x1Shape) (t : Tiling) : Nat := srcRowStrideConv sh.toConvSpec t
def resRowStride (sh : Conv1x1Shape) (t : Tiling) : Nat := resRowStrideConv sh.toConvSpec t
def isDirect (target : Target) (sh : Conv1x1Shape) (t : Tiling) : Bool := isDirectConv target sh.toConvSpec t
def srcL2Bytes (sh : Conv1x1Shape) (t : Tiling) : Nat := srcL2BytesConv sh.toConvSpec t
def resL2Bytes (target : Target) (sh : Conv1x1Shape) (t : Tiling) : Nat := resL2BytesConv target sh.toConvSpec t
def l2UseBytes (target : Target) (sh : Conv1x1Shape) (t : Tiling) : Nat := l2UseBytesConv target sh.toConvSpec t
def validateShape (sh : Conv1x1Shape) : Except String Unit := validateConvShape sh.toConvSpec
def validateTiling (target : Target) (sh : Conv1x1Shape) (t : Tiling) : Except String Unit := validateConvTiling target sh.toConvSpec t
def chooseTiling (target : Target) (sh : Conv1x1Shape) : Except String Tiling := chooseConvTiling target sh.toConvSpec

namespace Conv3x3Shape
def effMode (sh : Conv3x3Shape) : Conv3x3Mode := sh.toConvSpec.effMode
def isSparse (sh : Conv3x3Shape) : Bool := sh.toConvSpec.isSparse
def outH (sh : Conv3x3Shape) : Nat := sh.toConvSpec.outH
def outW (sh : Conv3x3Shape) : Nat := sh.toConvSpec.outW
def padTop (sh : Conv3x3Shape) : Nat := sh.toConvSpec.padTop
def padLeft (sh : Conv3x3Shape) : Nat := sh.toConvSpec.padLeft
def rk (sh : Conv3x3Shape) : Nat := sh.toConvSpec.rk
def cropW (sh : Conv3x3Shape) : Nat := sh.toConvSpec.cropW
def cropH (sh : Conv3x3Shape) : Nat := sh.toConvSpec.cropH
def winReg (sh : Conv3x3Shape) : Nat := sh.toConvSpec.winReg
def hinReg (sh : Conv3x3Shape) : Nat := sh.toConvSpec.hinReg
end Conv3x3Shape

def effOcg (sh : Conv3x3Shape) (t : Tiling) : Nat := effOcgConv sh.toConvSpec t
def groupOutputBlocks3x3 (sh : Conv3x3Shape) (t : Tiling) (g : Nat) : Array (Nat × Nat) := groupOutputBlocksConv sh.toConvSpec t g
def maxBlockSlots3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat := maxBlockSlotsConv sh.toConvSpec t
def srcPx3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat := srcPxConv sh.toConvSpec t
def chsBytes3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat := chsBytesConv sh.toConvSpec t
def srcRowStride3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat := srcRowStrideConv sh.toConvSpec t
def inRows3x3 (sh : Conv3x3Shape) (th : Nat) : Nat := inRowsConv sh.toConvSpec th
def isDirect3x3 (target : Target) (sh : Conv3x3Shape) (t : Tiling) : Bool := isDirectConv target sh.toConvSpec t
def dchsBytes3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat := dchsBytesConv sh.toConvSpec t
def resRowStride3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat := resRowStrideConv sh.toConvSpec t
def srcL2Bytes3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat := srcL2BytesConv sh.toConvSpec t
def resL2Bytes3x3 (target : Target) (sh : Conv3x3Shape) (t : Tiling) : Nat := resL2BytesConv target sh.toConvSpec t
def l2UseBytes3x3 (target : Target) (sh : Conv3x3Shape) (t : Tiling) : Nat := l2UseBytesConv target sh.toConvSpec t
def validateConv3x3Shape (sh : Conv3x3Shape) : Except String Unit := validateConvShape sh.toConvSpec
def validateConv3x3Tiling (target : Target) (sh : Conv3x3Shape) (t : Tiling) : Except String Unit := validateConvTiling target sh.toConvSpec t
def chooseConv3x3Tiling (target : Target) (sh : Conv3x3Shape) : Except String Tiling := chooseConvTiling target sh.toConvSpec

end Vane.Tiling
