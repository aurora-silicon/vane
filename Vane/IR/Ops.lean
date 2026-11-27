import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Quant
import Vane.Conv
import Vane.Lut
import Vane.Pool
import Vane.Pe

namespace Vane.IR

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Quant
open Vane.Conv
open Vane.Lut
open Vane.Pool
open Vane.Pe

-- ============================================================================
-- Value IDs, Data Types, Pure Logical Tensor Types, and Constant Payloads
-- ============================================================================

/-- Dense SSA value identifier. Graph inputs occupy `0 .. inputs.size - 1`;
    node outputs follow in strict topological order. -/
abbrev ValueId := Nat

/-- Logical scalar/weight data type in L1 `TensorIR`. -/
inductive DType where
  | fp16
  | f32
  | i8
  | u8
  | i4
  | u4
  | pal (bits : Nat)
  deriving Repr, BEq, DecidableEq, Inhabited

namespace DType

def name : DType → String
  | .fp16 => "fp16"
  | .f32 => "f32"
  | .i8 => "i8"
  | .u8 => "u8"
  | .i4 => "i4"
  | .u4 => "u4"
  | .pal b => s!"pal{b}"

def ofQuantFormat : QuantFormat → DType
  | .int8 => .i8
  | .uint8 => .u8
  | .int4 => .i4
  | .uint4 => .u4
  | .pal b => .pal b

def toQuantFormat? : DType → Option QuantFormat
  | .i8 => some .int8
  | .u8 => some .uint8
  | .i4 => some .int4
  | .u4 => some .uint4
  | .pal b => some (.pal b)
  | _ => none

end DType

/-- Pure rank-N logical tensor type in L1 `TensorIR` (no ANE-specific layout tags). -/
structure TensorType where
  shape : Array Nat
  dtype : DType := .fp16
  deriving Repr, BEq, DecidableEq, Inhabited

namespace TensorType

def rank (t : TensorType) : Nat := t.shape.size

def numElems (t : TensorType) : Nat :=
  t.shape.foldl (· * ·) 1

def hasPositiveDims (t : TensorType) : Bool :=
  t.shape.all (· > 0)

def pretty (t : TensorType) : String :=
  let dims := String.intercalate "x" (t.shape.toList.map toString)
  s!"{t.dtype.name}[{dims}]"

end TensorType

/-- Constant tensor payload stored in a first-class SSA `CoreOp.const` node. -/
inductive ConstPayload where
  | fp16 (bytes : ByteArray)
  | f32 (vals : Array Float)
  | quant (spec : QuantWeightSpec)
  deriving BEq, Inhabited

namespace ConstPayload

def dtype : ConstPayload → DType
  | .fp16 _ => .fp16
  | .f32 _ => .f32
  | .quant q => DType.ofQuantFormat q.format

/-- Extract finite float64 values from an `fp16` or `f32` constant payload. -/
def toFloat64? : ConstPayload → Option (Array Float)
  | .f32 vals => some vals
  | .fp16 bytes =>
    if bytes.size % 2 != 0 then none
    else
      let n := bytes.size / 2
      some (Array.ofFn (n := n) fun i =>
        let lo := bytes.data.getD (2 * i.val) 0
        let hi := bytes.data.getD (2 * i.val + 1) 0
        let bits : UInt16 := lo.toUInt16 ||| (hi.toUInt16 <<< 8)
        f16ToFloat bits)
  | .quant _ => none

/-- Convert an `fp16` or `f32` constant payload into packed little-endian fp16 bytes. -/
def toFp16Bytes (p : ConstPayload) (ctx : String := "const") : Except String ByteArray :=
  match p with
  | .fp16 bytes => do
    if bytes.size % 2 != 0 then
      throw s!"{ctx}: fp16 payload byte size {bytes.size} is not a multiple of 2"
    pure bytes
  | .f32 vals => do
    let mut out := ByteArray.emptyWithCapacity (2 * vals.size)
    for i in [:vals.size] do
      let f := vals[i]!
      if f.isNaN || f.isInf then
        throw s!"{ctx}: non-finite value at index {i} ({f})"
      let h := floatToF16 f
      if !f16IsFinite h then
        throw s!"{ctx}: value {f} at index {i} overflows fp16"
      out := pushU16LE out h
    pure out
  | .quant q =>
    throw s!"{ctx}: cannot extract raw fp16 bytes from quantized '{q.format.name}' payload without dequantization"

/-- Byte/element summary for IR dumps. -/
def pretty : ConstPayload → String
  | .fp16 bs => s!"fp16({bs.size}B)"
  | .f32 vs => s!"f32({vs.size} elems)"
  | .quant q => s!"quant({q.format.name}, {q.codes.size} codes)"

end ConstPayload

-- ============================================================================
-- Elementwise & High-Level Helper Classifiers
-- ============================================================================

/-- Elementwise unary operations in `CoreOp`. -/
inductive ElemUnaryOp where
  | abs
  | neg
  | square
  | log
  | softplus
  | scalarAdd (valBits : UInt16)
  | scalarSub (valBits : UInt16)
  | scalarRSub (valBits : UInt16)
  | scalarMul (valBits : UInt16)
  | scalarDiv (valBits : UInt16)
  | linearAct (alphaBits betaBits : UInt16)
  deriving Repr, BEq, DecidableEq, Inhabited

namespace ElemUnaryOp

def name : ElemUnaryOp → String
  | .abs => "abs"
  | .neg => "neg"
  | .square => "square"
  | .log => "log"
  | .softplus => "softplus"
  | .scalarAdd b => s!"sadd({f16ToFloat b})"
  | .scalarSub b => s!"ssub({f16ToFloat b})"
  | .scalarRSub b => s!"srsub({f16ToFloat b})"
  | .scalarMul b => s!"smul({f16ToFloat b})"
  | .scalarDiv b => s!"sdiv({f16ToFloat b})"
  | .linearAct a b => s!"linear_act({f16ToFloat a},{f16ToFloat b})"

end ElemUnaryOp

/-- Elementwise binary operations in `CoreOp`. -/
inductive ElemBinaryOp where
  | add
  | sub
  | mul
  | div
  | max
  | min
  deriving Repr, BEq, DecidableEq, Inhabited

namespace ElemBinaryOp

def name : ElemBinaryOp → String
  | .add => "add"
  | .sub => "sub"
  | .mul => "mul"
  | .div => "div"
  | .max => "max"
  | .min => "min"

end ElemBinaryOp

/-- Supported constant scalar exponent classifications for `HighOp.pow`. -/
inductive PowKind where
  | square      -- exp == 2.0  -> mul(x, x)
  | identity    -- exp == 1.0  -> elided
  | sqrt        -- exp == 0.5  -> lut(sqrt)
  | rsqrt       -- exp == -0.5 -> lut(rsqrt)
  | reciprocal  -- exp == -1.0 -> lut(inverse)
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Exact IEEE-754 binary64 bit pattern for each supported `PowKind` exponent. -/
def powKindToF64Bits : PowKind → UInt64
  | .square     => 0x4000000000000000 --  2.0
  | .identity   => 0x3ff0000000000000 --  1.0
  | .sqrt       => 0x3fe0000000000000 --  0.5
  | .rsqrt      => 0xbfe0000000000000 -- -0.5
  | .reciprocal => 0xbff0000000000000 -- -1.0

/-- Classify an IEEE-754 binary64 exponent bit pattern into a supported `PowKind`. -/
def classifyPowBits (bits : UInt64) : Option PowKind :=
  if bits == 0x4000000000000000 then some .square
  else if bits == 0x3ff0000000000000 then some .identity
  else if bits == 0x3fe0000000000000 then some .sqrt
  else if bits == 0xbfe0000000000000 then some .rsqrt
  else if bits == 0xbff0000000000000 then some .reciprocal
  else none

/-- Validate and classify a floating-point `Pow` scalar exponent. -/
def classifyPowExponent (expVal : Float) (ctx : String) : Except String PowKind :=
  match classifyPowBits expVal.toBits with
  | some k => .ok k
  | none => .error s!"{ctx}: unsupported Pow exponent {expVal} (supported: 2.0, 1.0, 0.5, -0.5, -1.0)"

/-- Validated `Gelu` `approximate` attribute mode (only exact erf GELU is supported). -/
inductive GeluApprox where
  | exact       -- "" or "none"
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Validate the ONNX `Gelu` `approximate` attribute value.
    Rejects `"tanh"` and any unknown value instead of silently approximating. -/
def classifyGeluApprox (raw : String) : Except String GeluApprox :=
  match raw with
  | "" | "none" => .ok .exact
  | "tanh" => .error "Gelu: approximate='tanh' is not supported (only exact erf GELU with approximate='' or 'none' is supported)"
  | other => .error s!"Gelu: unsupported approximate='{other}' (expected '' or 'none')"

-- ============================================================================
-- Closed 13-Op CoreOp & HighOp Dialect Hierarchy
-- ============================================================================

/-- Closed 13-op L1b core tensor algebra dialect.
    Weights and biases are first-class SSA `const` values; quantization is
    explicit via `dequantize`. -/
inductive CoreOp where
  | const (payload : ConstPayload)
  | dequantize (format : QuantFormat)
  | conv (kh kw sh sw : Nat) (pads : Array Nat) (groups : Nat) (hasBias : Bool) (act : ConvActivation)
  | matmul (transposeA transposeB : Bool) (scaleBits : UInt16)
  | elemUnary (op : ElemUnaryOp)
  | elemBinary (op : ElemBinaryOp)
  | lut (act : NeActSpec)
  | pool (kind : PoolKind) (kh kw sh sw : Nat) (pad : PoolPad) (excludePad : Bool)
  | reduce (op : ReduceOp) (axes : Array Nat) (keepDims : Bool) (eps : UInt16)
  | reshape (targetShape : Array Nat)
  | transpose (perm : Array Nat)
  | slice (beginIdx endIdx step : Array Nat)
  | concat (axis : Nat)
  deriving BEq, Inhabited

namespace CoreOp

def pretty : CoreOp → String
  | .const p => s!"const({p.pretty})"
  | .dequantize fmt => s!"dequantize({fmt.name})"
  | .conv kh kw sh sw pads groups hasBias act =>
    s!"conv(k={kh}x{kw}, s={sh}x{sw}, pads={pads.toList}, g={groups}, bias={hasBias}, act={act.name})"
  | .matmul trA trB sc =>
    s!"matmul(trA={trA}, trB={trB}, scale={f16ToFloat sc})"
  | .elemUnary op => s!"elem_unary({op.name})"
  | .elemBinary op => s!"elem_binary({op.name})"
  | .lut act => s!"lut({act.name})"
  | .pool kind kh kw sh sw pad excl =>
    s!"pool({kind.name}, k={kh}x{kw}, s={sh}x{sw}, pad={pad.name}, excl={excl})"
  | .reduce op axes keepDims eps =>
    s!"reduce({op.name}, axes={axes.toList}, keepdims={keepDims}, eps={f16ToFloat eps})"
  | .reshape sh => s!"reshape({sh.toList})"
  | .transpose perm => s!"transpose({perm.toList})"
  | .slice beg en st => s!"slice(begin={beg.toList}, end={en.toList}, step={st.toList})"
  | .concat ax => s!"concat(axis={ax})"

end CoreOp

/-- L1a high-level frontend dialect: wraps `CoreOp` without duplicating constructors,
    adding composite ops lowered to `CoreOp` by `Passes/Decompose.lean`. -/
inductive HighOp where
  | core (op : CoreOp)
  | batchNorm (eps : Float)
  | layerNorm (axis : Int) (eps : Float) (hasScale hasBias : Bool)
  | rmsNorm (axis : Int) (eps : Float) (hasScale : Bool)
  | softmax (axis : Int) (legacyOpset : Bool := false)
  | gelu (approx : GeluApprox)
  | split (axis : Nat) (sizes : Array Nat)
  | pow (kind : PowKind)
  deriving BEq, Inhabited

namespace HighOp

def pretty : HighOp → String
  | .core op => op.pretty
  | .batchNorm eps => s!"high.batch_norm(eps={eps})"
  | .layerNorm ax eps hs hb => s!"high.layer_norm(axis={ax}, eps={eps}, scale={hs}, bias={hb})"
  | .rmsNorm ax eps hs => s!"high.rms_norm(axis={ax}, eps={eps}, scale={hs})"
  | .softmax ax leg => s!"high.softmax(axis={ax}, legacy={leg})"
  | .gelu _ => "high.gelu(exact)"
  | .split ax szs => s!"high.split(axis={ax}, sizes={szs.toList})"
  | .pow k => s!"high.pow({reprStr k})"

end HighOp

/-- Typeclass exposing the closed `CoreOp` sub-algebra inside any IR dialect (`HighOp`, `CoreOp`,
    and future `AneOp`) with round-trip laws, enabling polymorphic `ViewSimplify` and `ConstFold`. -/
class HasCoreOps (Op : Type) where
  asCore? : Op → Option CoreOp
  ofCore : CoreOp → Op
  asCore_ofCore : ∀ op : CoreOp, asCore? (ofCore op) = some op
  ofCore_asCore : ∀ (o : Op) (c : CoreOp), asCore? o = some c → ofCore c = o

instance : HasCoreOps CoreOp where
  asCore? := some
  ofCore := id
  asCore_ofCore _ := rfl
  ofCore_asCore _ _ h := (Option.some.inj h).symm

instance : HasCoreOps HighOp where
  asCore?
    | .core c => some c
    | _ => none
  ofCore := .core
  asCore_ofCore _ := rfl
  ofCore_asCore o c h := by
    cases o with
    | core c' =>
      have hc : c' = c := Option.some.inj h
      rw [hc]
    | batchNorm _ => cases h
    | layerNorm _ _ _ _ => cases h
    | rmsNorm _ _ _ => cases h
    | softmax _ _ => cases h
    | gelu _ => cases h
    | split _ _ => cases h
    | pow _ => cases h

-- ============================================================================
-- Pure Permutation Algebra (Kernel-Clean Helpers for ViewSimplify & Proofs)
-- ============================================================================

/-- Boolean nodup check on lists (decidable counterpart to `List.Nodup`). -/
def listNodupB {α : Type} [BEq α] : List α → Bool
  | [] => true
  | x :: xs => !(xs.elem x) && listNodupB xs

/-- Identity permutation `[0, 1, ..., n - 1]`. -/
def idPerm (n : Nat) : Array Nat :=
  Array.ofFn (n := n) fun i => i.val

/-- Check whether `p` is the identity permutation `[0, 1, ..., p.size - 1]`. -/
def isIdentityPerm (p : Array Nat) : Bool :=
  p == idPerm p.size

/-- Compose two permutations: `transpose(transpose(x, p1), p2) = transpose(x, composePerm p1 p2)`.
    Output axis `k` pulls from `p2[k]` of the intermediate tensor, which pulled from `p1[p2[k]]` of `x`. -/
def composePerm (p1 p2 : Array Nat) : Array Nat :=
  p2.map fun idx => p1.getD idx 0

/-- Prop-level valid permutation: every entry is in `[0, p.size)` and all entries are distinct. -/
def ValidPerm (p : Array Nat) : Prop :=
  (∀ i (hi : i < p.size), p[i] < p.size) ∧ p.toList.Nodup

/-- Boolean check for `ValidPerm`. -/
def isValidPerm (p : Array Nat) : Bool :=
  p.all (· < p.size) && listNodupB p.toList

/-- First index of `target` in `l`, or `0` if absent. -/
def findIdxNat (l : List Nat) (target : Nat) : Nat :=
  match l with
  | [] => 0
  | x :: xs => if x == target then 0 else 1 + findIdxNat xs target

/-- Inverse permutation `invPerm p`: for valid `p`, `p[invPerm p[a]] = a`. -/
def invPerm (p : Array Nat) : Array Nat :=
  Array.ofFn (n := p.size) fun a => findIdxNat p.toList a.val

end Vane.IR
