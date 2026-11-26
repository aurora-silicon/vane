import Vane.Binary
import Vane.Float16
import Vane.Frontend.SafeTensors

namespace Vane.Frontend.Onnx

open Vane.Binary
open Vane.Float16
open Vane.Frontend.SafeTensors

-- ============================================================================
-- Protobuf Wire Format Constants & Low-Level Decoder / Encoder
-- ============================================================================

def wireVarint  : UInt8 := 0
def wireFixed64 : UInt8 := 1
def wireBytes   : UInt8 := 2
def wireFixed32 : UInt8 := 5

/-- Maximum nested message depth permitted during protobuf decoding. -/
def maxProtoDepth : Nat := 32

-- ONNX TensorProto.DataType enum values per onnx.proto3.
def onnxUndefined : Nat := 0
def onnxFloat     : Nat := 1
def onnxUint8     : Nat := 2
def onnxInt8      : Nat := 3
def onnxUint16    : Nat := 4
def onnxInt16     : Nat := 5
def onnxInt32     : Nat := 6
def onnxInt64     : Nat := 7
def onnxString    : Nat := 8
def onnxBool      : Nat := 9
def onnxFloat16   : Nat := 10
def onnxDouble    : Nat := 11
def onnxUint32    : Nat := 12
def onnxUint64    : Nat := 13
def onnxBFloat16  : Nat := 16

-- ONNX AttributeProto.AttributeType enum values per onnx.proto3.
def attrUndefined : Nat := 0
def attrFloat     : Nat := 1
def attrInt       : Nat := 2
def attrString    : Nat := 3
def attrTensor    : Nat := 4
def attrGraph     : Nat := 5
def attrFloats    : Nat := 6
def attrInts      : Nat := 7
def attrStrings   : Nat := 8

-- ONNX TensorProto.DataLocation enum values per onnx.proto3.
def onnxDataDefault  : Nat := 0
def onnxDataExternal : Nat := 1

/-- Mutable cursor over a protobuf wire byte buffer. -/
structure ProtoCursor where
  buf : ByteArray
  pos : Nat := 0
  deriving BEq, Inhabited

def ProtoCursor.done (c : ProtoCursor) : Bool :=
  c.pos >= c.buf.size

def ProtoCursor.remaining (c : ProtoCursor) : Nat :=
  c.buf.size - c.pos

/-- Decode a base-128 unsigned 64-bit protobuf varint (at most 10 bytes). -/
def readVarint (c : ProtoCursor) : Except String (UInt64 × ProtoCursor) := do
  let mut pos := c.pos
  let mut val : UInt64 := 0
  for i in [:10] do
    if pos >= c.buf.size then
      throw s!"protobuf truncated varint at offset {pos}"
    let b := c.buf.get! pos
    pos := pos + 1
    let shift : UInt64 := (7 * i).toUInt64
    if i == 9 && b > 0x01 then
      throw s!"protobuf varint overflow at offset {pos - 1}"
    val := val ||| ((b &&& 0x7f).toUInt64 <<< shift)
    if (b &&& 0x80) == 0 then
      return (val, { c with pos := pos })
  throw s!"protobuf varint exceeds 10 bytes at offset {pos}"

/-- Read and validate the next protobuf field tag `(fieldNum, wireType)`.
    Rejects wire types 3, 4, 6, 7 and invalid field numbers (`0` or `> 2^29 - 1`). -/
def nextTag (c : ProtoCursor) : Except String (Nat × UInt8 × ProtoCursor) := do
  let (tag, c') ← readVarint c
  let wt : UInt8 := (tag &&& 0x07).toUInt8
  let fn : Nat := (tag >>> 3).toNat
  if fn == 0 || fn > ((1 : Nat) <<< 29) - 1 then
    throw s!"protobuf invalid field number {fn}"
  if wt != wireVarint && wt != wireFixed64 && wt != wireBytes && wt != wireFixed32 then
    throw s!"protobuf unsupported wire type {wt.toNat} for field {fn}"
  pure (fn, wt, c')

def readFixed32 (c : ProtoCursor) : Except String (UInt32 × ProtoCursor) := do
  if c.remaining < 4 then
    throw s!"protobuf truncated fixed32 at offset {c.pos}"
  let v ← readU32LE c.buf c.pos
  pure (v, { c with pos := c.pos + 4 })

def readFloat32 (c : ProtoCursor) : Except String (Float × ProtoCursor) := do
  let (u, c') ← readFixed32 c
  pure (f32BitsToFloat u, c')

def readFixed64 (c : ProtoCursor) : Except String (UInt64 × ProtoCursor) := do
  if c.remaining < 8 then
    throw s!"protobuf truncated fixed64 at offset {c.pos}"
  let v ← readU64LE c.buf c.pos
  pure (v, { c with pos := c.pos + 8 })

def readFloat64 (c : ProtoCursor) : Except String (Float × ProtoCursor) := do
  let (u, c') ← readFixed64 c
  pure (Float.ofBits u, c')

def readProtoBytes (c : ProtoCursor) : Except String (ByteArray × ProtoCursor) := do
  let (n64, c1) ← readVarint c
  let n := n64.toNat
  if n > c1.remaining then
    throw s!"protobuf length-delimited field size {n} exceeds remaining {c1.remaining} bytes"
  let out := c1.buf.extract c1.pos (c1.pos + n)
  pure (out, { c1 with pos := c1.pos + n })

def readProtoString (c : ProtoCursor) : Except String (String × ProtoCursor) := do
  let (bs, c1) ← readProtoBytes c
  match String.fromUTF8? bs with
  | some s => pure (s, c1)
  | none => throw s!"protobuf string at offset {c.pos} is not valid UTF-8"

def skipField (c : ProtoCursor) (wt : UInt8) : Except String ProtoCursor :=
  if wt == wireVarint then do
    let (_, c') ← readVarint c
    pure c'
  else if wt == wireFixed64 then do
    let (_, c') ← readFixed64 c
    pure c'
  else if wt == wireBytes then do
    let (_, c') ← readProtoBytes c
    pure c'
  else if wt == wireFixed32 then do
    let (_, c') ← readFixed32 c
    pure c'
  else
    throw s!"protobuf cannot skip unsupported wire type {wt.toNat}"

def readPackedInt64 (bs : ByteArray) : Except String (Array Int) := do
  let mut cur : ProtoCursor := { buf := bs, pos := 0 }
  let mut out : Array Int := #[]
  for _ in [:bs.size] do
    if cur.done then break
    let (v, cur') ← readVarint cur
    out := out.push (u64ToInt64 v)
    cur := cur'
  pure out

def readPackedInt32 (bs : ByteArray) : Except String (Array Int) := do
  let mut cur : ProtoCursor := { buf := bs, pos := 0 }
  let mut out : Array Int := #[]
  for _ in [:bs.size] do
    if cur.done then break
    let (v, cur') ← readVarint cur
    out := out.push (u32ToInt32 (v &&& 0xffffffff).toUInt32)
    cur := cur'
  pure out

def readPackedFloat32 (bs : ByteArray) : Except String (Array Float) := do
  if bs.size % 4 != 0 then
    throw s!"protobuf packed float32 length {bs.size} is not a multiple of 4"
  let n := bs.size / 4
  let mut out : Array Float := Array.mkEmpty n
  for i in [:n] do
    let u ← readU32LE bs (4 * i)
    out := out.push (f32BitsToFloat u)
  pure out

def readPackedFloat64 (bs : ByteArray) : Except String (Array Float) := do
  if bs.size % 8 != 0 then
    throw s!"protobuf packed float64 length {bs.size} is not a multiple of 8"
  let n := bs.size / 8
  let mut out : Array Float := Array.mkEmpty n
  for i in [:n] do
    let u ← readU64LE bs (8 * i)
    out := out.push (Float.ofBits u)
  pure out

-- ============================================================================
-- ONNX AST Data Structures
-- ============================================================================

structure OnnxDim where
  hasValue : Bool := false
  value : Int := 0
  param : String := ""
  deriving Repr, BEq, Inhabited

structure OnnxValueInfo where
  name : String := ""
  elemType : Nat := 0
  hasShape : Bool := false
  dims : Array OnnxDim := #[]
  deriving Repr, BEq, Inhabited

structure OnnxTensor where
  name : String := ""
  dims : Array Int := #[]
  dataType : Nat := 0
  floatData : Array Float := #[]
  int32Data : Array Int := #[]
  int64Data : Array Int := #[]
  doubleData : Array Float := #[]
  rawData : ByteArray := ByteArray.empty
  dataLocation : Nat := onnxDataDefault
  deriving BEq, Inhabited

structure OnnxAttribute where
  name : String := ""
  attrType : Nat := 0
  f : Float := 0.0
  i : Int := 0
  s : ByteArray := ByteArray.empty
  t : Option OnnxTensor := none
  floats : Array Float := #[]
  ints : Array Int := #[]
  strings : Array ByteArray := #[]
  deriving BEq, Inhabited

structure OnnxNode where
  inputs : Array String := #[]
  outputs : Array String := #[]
  name : String := ""
  opType : String := ""
  domain : String := ""
  attributes : Array OnnxAttribute := #[]
  deriving BEq, Inhabited

structure OnnxOpsetImport where
  domain : String := ""
  version : Int := 0
  deriving Repr, BEq, Inhabited

structure OnnxGraph where
  name : String := ""
  nodes : Array OnnxNode := #[]
  initializers : Array OnnxTensor := #[]
  inputs : Array OnnxValueInfo := #[]
  outputs : Array OnnxValueInfo := #[]
  deriving BEq, Inhabited

structure OnnxModel where
  irVersion : Int := 0
  producerName : String := ""
  opsetImports : Array OnnxOpsetImport := #[]
  graph : OnnxGraph := {}
  hasGraph : Bool := false
  deriving BEq, Inhabited

-- ============================================================================
-- ONNX AST Helpers & Tensor Value Extractors
-- ============================================================================

namespace OnnxModel

def defaultOpsetVersion (m : OnnxModel) : Int :=
  match m.opsetImports.find? fun op => op.domain == "" || op.domain == "ai.onnx" with
  | some op => op.version
  | none => 0

end OnnxModel

namespace OnnxNode

def attr? (nd : OnnxNode) (name : String) : Option OnnxAttribute :=
  nd.attributes.find? fun a => a.name == name

def attrInt (nd : OnnxNode) (name : String) (defaultVal : Int) : Int :=
  match nd.attr? name with
  | some a => a.i
  | none => defaultVal

def attrFloat (nd : OnnxNode) (name : String) (defaultVal : Float) : Float :=
  match nd.attr? name with
  | some a => a.f
  | none => defaultVal

def attrInts? (nd : OnnxNode) (name : String) : Option (Array Int) :=
  (nd.attr? name).map fun a => a.ints

def attrFloats? (nd : OnnxNode) (name : String) : Option (Array Float) :=
  (nd.attr? name).map fun a => a.floats

def attrString (nd : OnnxNode) (name : String) (defaultVal : String) : String :=
  match nd.attr? name with
  | some a => (String.fromUTF8? a.s).getD defaultVal
  | none => defaultVal

def attrStringChecked (nd : OnnxNode) (name : String) (defaultVal : String) : Except String String :=
  match nd.attr? name with
  | some a =>
    match String.fromUTF8? a.s with
    | some s => pure s
    | none => throw s!"attribute '{name}' contains invalid UTF-8 bytes"
  | none => pure defaultVal

def attrTensor? (nd : OnnxNode) (name : String) : Option OnnxTensor :=
  match nd.attr? name with
  | some a => a.t
  | none => none

end OnnxNode

/-- Convert a `Float` array to finite IEEE-754 binary16 values, rejecting `NaN`/`Inf`/overflow. -/
def float64ArrayToFp16 (vals : Array Float) (name : String) : Except String (Array UInt16) := do
  let mut out : Array UInt16 := Array.mkEmpty vals.size
  for i in [:vals.size] do
    let f := vals[i]!
    if f.isNaN || f.isInf then
      throw s!"tensor '{name}': non-finite value at index {i} ({f})"
    let h := floatToF16 f
    if !f16IsFinite h then
      throw s!"tensor '{name}': value {f} at index {i} overflows fp16"
    out := out.push h
  pure out

namespace OnnxTensor

def dimsToNat (t : OnnxTensor) : Except String (Array Nat) := do
  let mut out : Array Nat := Array.mkEmpty t.dims.size
  for i in [:t.dims.size] do
    let d := t.dims[i]!
    if d < 0 then
      throw s!"tensor '{t.name}': negative dim[{i}]={d}"
    out := out.push d.toNat
  pure out

def numElements (t : OnnxTensor) : Except String Nat := do
  let ds ← t.dimsToNat
  shapeProductChecked ds s!"tensor '{t.name}'"

/-- Extract finite float64 values from a `FLOAT`, `DOUBLE`, `FLOAT16`, or `BFLOAT16` `OnnxTensor`. -/
def toFloat64 (t : OnnxTensor) : Except String (Array Float) := do
  if t.dataLocation == onnxDataExternal then
    throw s!"tensor '{t.name}' uses external data_location=EXTERNAL (provide --safetensors override or embedded weights)"
  let nElems ← t.numElements
  if t.dataType == onnxFloat then
    if t.rawData.size > 0 then
      let want := nElems * 4
      if t.rawData.size != want then
        throw s!"tensor '{t.name}' (FLOAT): raw_data length {t.rawData.size} != expected {want}"
      let mut out : Array Float := Array.mkEmpty nElems
      for i in [:nElems] do
        let u ← readU32LE t.rawData (4 * i)
        let f := f32BitsToFloat u
        if f.isNaN || f.isInf then
          throw s!"tensor '{t.name}' (FLOAT): non-finite value at index {i}"
        out := out.push f
      pure out
    else
      if t.floatData.size != nElems then
        throw s!"tensor '{t.name}' (FLOAT): float_data length {t.floatData.size} != expected {nElems}"
      let mut out : Array Float := Array.mkEmpty nElems
      for i in [:nElems] do
        let f := t.floatData[i]!
        if f.isNaN || f.isInf then
          throw s!"tensor '{t.name}' (FLOAT): non-finite value at index {i}"
        out := out.push f
      pure out
  else if t.dataType == onnxDouble then
    if t.rawData.size > 0 then
      let want := nElems * 8
      if t.rawData.size != want then
        throw s!"tensor '{t.name}' (DOUBLE): raw_data length {t.rawData.size} != expected {want}"
      let mut out : Array Float := Array.mkEmpty nElems
      for i in [:nElems] do
        let u ← readU64LE t.rawData (8 * i)
        let f := Float.ofBits u
        if f.isNaN || f.isInf then
          throw s!"tensor '{t.name}' (DOUBLE): non-finite value at index {i}"
        out := out.push f
      pure out
    else
      if t.doubleData.size != nElems then
        throw s!"tensor '{t.name}' (DOUBLE): double_data length {t.doubleData.size} != expected {nElems}"
      let mut out : Array Float := Array.mkEmpty nElems
      for i in [:nElems] do
        let f := t.doubleData[i]!
        if f.isNaN || f.isInf then
          throw s!"tensor '{t.name}' (DOUBLE): non-finite value at index {i}"
        out := out.push f
      pure out
  else if t.dataType == onnxFloat16 then
    if t.rawData.size > 0 then
      let want := nElems * 2
      if t.rawData.size != want then
        throw s!"tensor '{t.name}' (FLOAT16): raw_data length {t.rawData.size} != expected {want}"
      let mut out : Array Float := Array.mkEmpty nElems
      for i in [:nElems] do
        let u ← readU16LE t.rawData (2 * i)
        if !f16IsFinite u then
          throw s!"tensor '{t.name}' (FLOAT16): non-finite value at index {i} ({toHex u.toNat})"
        out := out.push (f16ToFloat u)
      pure out
    else
      -- Per ONNX spec, FLOAT16 may be stored in the low 16 bits of int32_data.
      if t.int32Data.size != nElems then
        throw s!"tensor '{t.name}' (FLOAT16): int32_data length {t.int32Data.size} != expected {nElems}"
      let mut out : Array Float := Array.mkEmpty nElems
      for i in [:nElems] do
        let u : UInt16 := (int32ToU32 t.int32Data[i]! &&& 0xffff).toUInt16
        if !f16IsFinite u then
          throw s!"tensor '{t.name}' (FLOAT16): non-finite value at index {i} ({toHex u.toNat})"
        out := out.push (f16ToFloat u)
      pure out
  else if t.dataType == onnxBFloat16 then
    if t.rawData.size > 0 then
      let want := nElems * 2
      if t.rawData.size != want then
        throw s!"tensor '{t.name}' (BFLOAT16): raw_data length {t.rawData.size} != expected {want}"
      let mut out : Array Float := Array.mkEmpty nElems
      for i in [:nElems] do
        let b ← readU16LE t.rawData (2 * i)
        let f := bf16ToFloat b
        if f.isNaN || f.isInf then
          throw s!"tensor '{t.name}' (BFLOAT16): non-finite value at index {i} ({toHex b.toNat})"
        out := out.push f
      pure out
    else
      if t.int32Data.size != nElems then
        throw s!"tensor '{t.name}' (BFLOAT16): int32_data length {t.int32Data.size} != expected {nElems}"
      let mut out : Array Float := Array.mkEmpty nElems
      for i in [:nElems] do
        let b : UInt16 := (int32ToU32 t.int32Data[i]! &&& 0xffff).toUInt16
        let f := bf16ToFloat b
        if f.isNaN || f.isInf then
          throw s!"tensor '{t.name}' (BFLOAT16): non-finite value at index {i} ({toHex b.toNat})"
        out := out.push f
      pure out
  else
    throw s!"tensor '{t.name}': unsupported float data_type {t.dataType}"

/-- Extract integer values from an `INT64`, `INT32`, or `BOOL` `OnnxTensor`. -/
def toInt64 (t : OnnxTensor) : Except String (Array Int) := do
  if t.dataLocation == onnxDataExternal then
    throw s!"tensor '{t.name}' uses external data_location=EXTERNAL"
  let nElems ← t.numElements
  if t.dataType == onnxInt64 then
    if t.rawData.size > 0 then
      let want := nElems * 8
      if t.rawData.size != want then
        throw s!"tensor '{t.name}' (INT64): raw_data length {t.rawData.size} != expected {want}"
      let mut out : Array Int := Array.mkEmpty nElems
      for i in [:nElems] do
        let u ← readU64LE t.rawData (8 * i)
        out := out.push (u64ToInt64 u)
      pure out
    else
      if t.int64Data.size != nElems then
        throw s!"tensor '{t.name}' (INT64): int64_data length {t.int64Data.size} != expected {nElems}"
      pure t.int64Data
  else if t.dataType == onnxInt32 then
    if t.rawData.size > 0 then
      let want := nElems * 4
      if t.rawData.size != want then
        throw s!"tensor '{t.name}' (INT32): raw_data length {t.rawData.size} != expected {want}"
      let mut out : Array Int := Array.mkEmpty nElems
      for i in [:nElems] do
        let u ← readU32LE t.rawData (4 * i)
        out := out.push (u32ToInt32 u)
      pure out
    else
      if t.int32Data.size != nElems then
        throw s!"tensor '{t.name}' (INT32): int32_data length {t.int32Data.size} != expected {nElems}"
      pure t.int32Data
  else if t.dataType == onnxBool then
    if t.rawData.size > 0 then
      if t.rawData.size != nElems then
        throw s!"tensor '{t.name}' (BOOL): raw_data length {t.rawData.size} != expected {nElems}"
      let mut out : Array Int := Array.mkEmpty nElems
      for i in [:nElems] do
        let b ← readU8 t.rawData i
        out := out.push (if b != 0 then 1 else 0)
      pure out
    else
      if t.int32Data.size != nElems then
        throw s!"tensor '{t.name}' (BOOL): int32_data length {t.int32Data.size} != expected {nElems}"
      pure (t.int32Data.map fun v => if v != 0 then 1 else 0)
  else
    throw s!"tensor '{t.name}': unsupported integer data_type {t.dataType}"

/-- Convert a float `OnnxTensor` to finite IEEE-754 binary16 bit patterns. -/
def toFp16 (t : OnnxTensor) : Except String (Array UInt16) := do
  let vals ← t.toFloat64
  float64ArrayToFp16 vals t.name

end OnnxTensor

-- ============================================================================
-- Fuel-Bounded ONNX Proto3 Message Decoders (Zero `partial def`, Depth Cap 32)
-- ============================================================================

private def checkDepth (remDepth : Nat) : Except String Nat :=
  match remDepth with
  | 0 => throw s!"protobuf recursion depth limit {maxProtoDepth} exceeded"
  | d + 1 => pure d

private def decodeDimensionProto (buf : ByteArray) : Except String OnnxDim := do
  let mut cur : ProtoCursor := { buf := buf, pos := 0 }
  let mut d : OnnxDim := {}
  for _ in [:buf.size] do
    if cur.done then break
    let (fn, wt, cur1) ← nextTag cur
    match fn with
    | 1 => -- dim_value (int64)
      if wt != wireVarint then throw s!"Dimension.dim_value: unexpected wire type {wt.toNat}"
      let (v, cur2) ← readVarint cur1
      d := { d with hasValue := true, value := u64ToInt64 v }
      cur := cur2
    | 2 => -- dim_param (string)
      if wt != wireBytes then throw s!"Dimension.dim_param: unexpected wire type {wt.toNat}"
      let (s, cur2) ← readProtoString cur1
      d := { d with param := s }
      cur := cur2
    | _ =>
      cur ← skipField cur1 wt
  pure d

private def decodeTensorShapeProto (buf : ByteArray) (remDepth : Nat) : Except String (Array OnnxDim) := do
  let mut cur : ProtoCursor := { buf := buf, pos := 0 }
  let mut dims : Array OnnxDim := #[]
  for _ in [:buf.size] do
    if cur.done then break
    let (fn, wt, cur1) ← nextTag cur
    match fn with
    | 1 => -- dim (repeated Dimension)
      if wt != wireBytes then throw s!"TensorShapeProto.dim: unexpected wire type {wt.toNat}"
      let (sub, cur2) ← readProtoBytes cur1
      let _ ← checkDepth remDepth
      let dim ← decodeDimensionProto sub
      dims := dims.push dim
      cur := cur2
    | _ =>
      cur ← skipField cur1 wt
  pure dims

private def decodeTypeTensorProto (buf : ByteArray) (remDepth : Nat) (vi : OnnxValueInfo) : Except String OnnxValueInfo := do
  let mut cur : ProtoCursor := { buf := buf, pos := 0 }
  let mut out := vi
  for _ in [:buf.size] do
    if cur.done then break
    let (fn, wt, cur1) ← nextTag cur
    match fn with
    | 1 => -- elem_type
      if wt != wireVarint then throw s!"TypeProto.Tensor.elem_type: unexpected wire type {wt.toNat}"
      let (v, cur2) ← readVarint cur1
      out := { out with elemType := v.toNat }
      cur := cur2
    | 2 => -- shape (TensorShapeProto)
      if wt != wireBytes then throw s!"TypeProto.Tensor.shape: unexpected wire type {wt.toNat}"
      let (sub, cur2) ← readProtoBytes cur1
      let dNext ← checkDepth remDepth
      let dims ← decodeTensorShapeProto sub dNext
      out := { out with hasShape := true, dims := dims }
      cur := cur2
    | _ =>
      cur ← skipField cur1 wt
  pure out

private def decodeTypeProto (buf : ByteArray) (remDepth : Nat) (vi : OnnxValueInfo) : Except String OnnxValueInfo := do
  let mut cur : ProtoCursor := { buf := buf, pos := 0 }
  let mut out := vi
  for _ in [:buf.size] do
    if cur.done then break
    let (fn, wt, cur1) ← nextTag cur
    match fn with
    | 1 => -- tensor_type (TypeProto.Tensor)
      if wt != wireBytes then throw s!"TypeProto.tensor_type: unexpected wire type {wt.toNat}"
      let (sub, cur2) ← readProtoBytes cur1
      let dNext ← checkDepth remDepth
      out ← decodeTypeTensorProto sub dNext out
      cur := cur2
    | _ =>
      cur ← skipField cur1 wt
  pure out

private def decodeValueInfoProto (buf : ByteArray) (remDepth : Nat) : Except String OnnxValueInfo := do
  let mut cur : ProtoCursor := { buf := buf, pos := 0 }
  let mut vi : OnnxValueInfo := {}
  for _ in [:buf.size] do
    if cur.done then break
    let (fn, wt, cur1) ← nextTag cur
    match fn with
    | 1 => -- name
      if wt != wireBytes then throw s!"ValueInfoProto.name: unexpected wire type {wt.toNat}"
      let (s, cur2) ← readProtoString cur1
      vi := { vi with name := s }
      cur := cur2
    | 2 => -- type (TypeProto)
      if wt != wireBytes then throw s!"ValueInfoProto.type: unexpected wire type {wt.toNat}"
      let (sub, cur2) ← readProtoBytes cur1
      let dNext ← checkDepth remDepth
      vi ← decodeTypeProto sub dNext vi
      cur := cur2
    | _ =>
      cur ← skipField cur1 wt
  pure vi

private def decodeTensorProto (buf : ByteArray) : Except String OnnxTensor := do
  let mut cur : ProtoCursor := { buf := buf, pos := 0 }
  let mut t : OnnxTensor := {}
  for _ in [:buf.size] do
    if cur.done then break
    let (fn, wt, cur1) ← nextTag cur
    match fn with
    | 1 => -- dims (repeated int64, packed or unpacked)
      if wt == wireVarint then
        let (v, cur2) ← readVarint cur1
        t := { t with dims := t.dims.push (u64ToInt64 v) }
        cur := cur2
      else if wt == wireBytes then
        let (sub, cur2) ← readProtoBytes cur1
        let vs ← readPackedInt64 sub
        t := { t with dims := t.dims ++ vs }
        cur := cur2
      else
        throw s!"TensorProto.dims: unexpected wire type {wt.toNat}"
    | 2 => -- data_type
      if wt != wireVarint then throw s!"TensorProto.data_type: unexpected wire type {wt.toNat}"
      let (v, cur2) ← readVarint cur1
      t := { t with dataType := v.toNat }
      cur := cur2
    | 4 => -- float_data (packed or unpacked)
      if wt == wireFixed32 then
        let (f, cur2) ← readFloat32 cur1
        t := { t with floatData := t.floatData.push f }
        cur := cur2
      else if wt == wireBytes then
        let (sub, cur2) ← readProtoBytes cur1
        let fs ← readPackedFloat32 sub
        t := { t with floatData := t.floatData ++ fs }
        cur := cur2
      else
        throw s!"TensorProto.float_data: unexpected wire type {wt.toNat}"
    | 5 => -- int32_data (packed or unpacked)
      if wt == wireVarint then
        let (v, cur2) ← readVarint cur1
        t := { t with int32Data := t.int32Data.push (u32ToInt32 (v &&& 0xffffffff).toUInt32) }
        cur := cur2
      else if wt == wireBytes then
        let (sub, cur2) ← readProtoBytes cur1
        let vs ← readPackedInt32 sub
        t := { t with int32Data := t.int32Data ++ vs }
        cur := cur2
      else
        throw s!"TensorProto.int32_data: unexpected wire type {wt.toNat}"
    | 7 => -- int64_data (packed or unpacked)
      if wt == wireVarint then
        let (v, cur2) ← readVarint cur1
        t := { t with int64Data := t.int64Data.push (u64ToInt64 v) }
        cur := cur2
      else if wt == wireBytes then
        let (sub, cur2) ← readProtoBytes cur1
        let vs ← readPackedInt64 sub
        t := { t with int64Data := t.int64Data ++ vs }
        cur := cur2
      else
        throw s!"TensorProto.int64_data: unexpected wire type {wt.toNat}"
    | 8 => -- name
      if wt != wireBytes then throw s!"TensorProto.name: unexpected wire type {wt.toNat}"
      let (s, cur2) ← readProtoString cur1
      t := { t with name := s }
      cur := cur2
    | 9 => -- raw_data
      if wt != wireBytes then throw s!"TensorProto.raw_data: unexpected wire type {wt.toNat}"
      let (bs, cur2) ← readProtoBytes cur1
      t := { t with rawData := bs }
      cur := cur2
    | 10 => -- double_data (packed or unpacked)
      if wt == wireFixed64 then
        let (f, cur2) ← readFloat64 cur1
        t := { t with doubleData := t.doubleData.push f }
        cur := cur2
      else if wt == wireBytes then
        let (sub, cur2) ← readProtoBytes cur1
        let fs ← readPackedFloat64 sub
        t := { t with doubleData := t.doubleData ++ fs }
        cur := cur2
      else
        throw s!"TensorProto.double_data: unexpected wire type {wt.toNat}"
    | 14 => -- data_location
      if wt != wireVarint then throw s!"TensorProto.data_location: unexpected wire type {wt.toNat}"
      let (v, cur2) ← readVarint cur1
      t := { t with dataLocation := v.toNat }
      cur := cur2
    | _ =>
      cur ← skipField cur1 wt
  pure t

private def decodeOpsetImport (buf : ByteArray) : Except String OnnxOpsetImport := do
  let mut cur : ProtoCursor := { buf := buf, pos := 0 }
  let mut op : OnnxOpsetImport := {}
  for _ in [:buf.size] do
    if cur.done then break
    let (fn, wt, cur1) ← nextTag cur
    match fn with
    | 1 => -- domain
      if wt != wireBytes then throw s!"OperatorSetIdProto.domain: unexpected wire type {wt.toNat}"
      let (s, cur2) ← readProtoString cur1
      op := { op with domain := s }
      cur := cur2
    | 2 => -- version
      if wt != wireVarint then throw s!"OperatorSetIdProto.version: unexpected wire type {wt.toNat}"
      let (v, cur2) ← readVarint cur1
      op := { op with version := u64ToInt64 v }
      cur := cur2
    | _ =>
      cur ← skipField cur1 wt
  pure op

/-! Mutually recursive fuel-bounded decoders for `GraphProto`, `NodeProto`, and `AttributeProto`
    so even nested subgraph attributes (`AttributeProto.g` / `graphs`) enforce the exact depth cap `32`
    with zero `partial def`. -/
mutual

private def decodeGraphProto (buf : ByteArray) : Nat → Except String OnnxGraph
  | 0 => throw s!"protobuf recursion depth limit {maxProtoDepth} exceeded"
  | d + 1 => do
    let mut cur : ProtoCursor := { buf := buf, pos := 0 }
    let mut g : OnnxGraph := {}
    for _ in [:buf.size] do
      if cur.done then break
      let (fn, wt, cur1) ← nextTag cur
      match fn with
      | 1 => -- node (repeated NodeProto)
        if wt != wireBytes then throw s!"GraphProto.node: unexpected wire type {wt.toNat}"
        let (sub, cur2) ← readProtoBytes cur1
        let nd ← decodeNodeProto sub d
        g := { g with nodes := g.nodes.push nd }
        cur := cur2
      | 2 => -- name
        if wt != wireBytes then throw s!"GraphProto.name: unexpected wire type {wt.toNat}"
        let (s, cur2) ← readProtoString cur1
        g := { g with name := s }
        cur := cur2
      | 5 => -- initializer (repeated TensorProto)
        if wt != wireBytes then throw s!"GraphProto.initializer: unexpected wire type {wt.toNat}"
        let (sub, cur2) ← readProtoBytes cur1
        let _ ← checkDepth d
        let t ← decodeTensorProto sub
        g := { g with initializers := g.initializers.push t }
        cur := cur2
      | 11 => -- input (repeated ValueInfoProto)
        if wt != wireBytes then throw s!"GraphProto.input: unexpected wire type {wt.toNat}"
        let (sub, cur2) ← readProtoBytes cur1
        let dNext ← checkDepth d
        let vi ← decodeValueInfoProto sub dNext
        g := { g with inputs := g.inputs.push vi }
        cur := cur2
      | 12 => -- output (repeated ValueInfoProto)
        if wt != wireBytes then throw s!"GraphProto.output: unexpected wire type {wt.toNat}"
        let (sub, cur2) ← readProtoBytes cur1
        let dNext ← checkDepth d
        let vi ← decodeValueInfoProto sub dNext
        g := { g with outputs := g.outputs.push vi }
        cur := cur2
      | _ =>
        cur ← skipField cur1 wt
    pure g

private def decodeNodeProto (buf : ByteArray) : Nat → Except String OnnxNode
  | 0 => throw s!"protobuf recursion depth limit {maxProtoDepth} exceeded"
  | d + 1 => do
    let mut cur : ProtoCursor := { buf := buf, pos := 0 }
    let mut nd : OnnxNode := {}
    for _ in [:buf.size] do
      if cur.done then break
      let (fn, wt, cur1) ← nextTag cur
      match fn with
      | 1 => -- input
        if wt != wireBytes then throw s!"NodeProto.input: unexpected wire type {wt.toNat}"
        let (s, cur2) ← readProtoString cur1
        nd := { nd with inputs := nd.inputs.push s }
        cur := cur2
      | 2 => -- output
        if wt != wireBytes then throw s!"NodeProto.output: unexpected wire type {wt.toNat}"
        let (s, cur2) ← readProtoString cur1
        nd := { nd with outputs := nd.outputs.push s }
        cur := cur2
      | 3 => -- name
        if wt != wireBytes then throw s!"NodeProto.name: unexpected wire type {wt.toNat}"
        let (s, cur2) ← readProtoString cur1
        nd := { nd with name := s }
        cur := cur2
      | 4 => -- op_type
        if wt != wireBytes then throw s!"NodeProto.op_type: unexpected wire type {wt.toNat}"
        let (s, cur2) ← readProtoString cur1
        nd := { nd with opType := s }
        cur := cur2
      | 5 => -- attribute
        if wt != wireBytes then throw s!"NodeProto.attribute: unexpected wire type {wt.toNat}"
        let (sub, cur2) ← readProtoBytes cur1
        let attr ← decodeAttributeProto sub d
        nd := { nd with attributes := nd.attributes.push attr }
        cur := cur2
      | 7 => -- domain
        if wt != wireBytes then throw s!"NodeProto.domain: unexpected wire type {wt.toNat}"
        let (s, cur2) ← readProtoString cur1
        nd := { nd with domain := s }
        cur := cur2
      | _ =>
        cur ← skipField cur1 wt
    pure nd

private def decodeAttributeProto (buf : ByteArray) : Nat → Except String OnnxAttribute
  | 0 => throw s!"protobuf recursion depth limit {maxProtoDepth} exceeded"
  | d + 1 => do
    let mut cur : ProtoCursor := { buf := buf, pos := 0 }
    let mut a : OnnxAttribute := {}
    for _ in [:buf.size] do
      if cur.done then break
      let (fn, wt, cur1) ← nextTag cur
      match fn with
      | 1 => -- name
        if wt != wireBytes then throw s!"AttributeProto.name: unexpected wire type {wt.toNat}"
        let (s, cur2) ← readProtoString cur1
        a := { a with name := s }
        cur := cur2
      | 2 => -- f (float)
        if wt != wireFixed32 then throw s!"AttributeProto.f: unexpected wire type {wt.toNat}"
        let (f, cur2) ← readFloat32 cur1
        a := { a with f := f }
        cur := cur2
      | 3 => -- i (int64)
        if wt != wireVarint then throw s!"AttributeProto.i: unexpected wire type {wt.toNat}"
        let (v, cur2) ← readVarint cur1
        a := { a with i := u64ToInt64 v }
        cur := cur2
      | 4 => -- s (bytes)
        if wt != wireBytes then throw s!"AttributeProto.s: unexpected wire type {wt.toNat}"
        let (bs, cur2) ← readProtoBytes cur1
        a := { a with s := bs }
        cur := cur2
      | 5 => -- t (TensorProto)
        if wt != wireBytes then throw s!"AttributeProto.t: unexpected wire type {wt.toNat}"
        let (sub, cur2) ← readProtoBytes cur1
        let _ ← checkDepth d
        let t ← decodeTensorProto sub
        a := { a with t := some t }
        cur := cur2
      | 6 => -- g (GraphProto): validate nested depth even when unused
        if wt != wireBytes then throw s!"AttributeProto.g: unexpected wire type {wt.toNat}"
        let (sub, cur2) ← readProtoBytes cur1
        let _ ← decodeGraphProto sub d
        cur := cur2
      | 7 => -- floats (packed or unpacked)
        if wt == wireFixed32 then
          let (f, cur2) ← readFloat32 cur1
          a := { a with floats := a.floats.push f }
          cur := cur2
        else if wt == wireBytes then
          let (sub, cur2) ← readProtoBytes cur1
          let fs ← readPackedFloat32 sub
          a := { a with floats := a.floats ++ fs }
          cur := cur2
        else
          throw s!"AttributeProto.floats: unexpected wire type {wt.toNat}"
      | 8 => -- ints (packed or unpacked)
        if wt == wireVarint then
          let (v, cur2) ← readVarint cur1
          a := { a with ints := a.ints.push (u64ToInt64 v) }
          cur := cur2
        else if wt == wireBytes then
          let (sub, cur2) ← readProtoBytes cur1
          let vs ← readPackedInt64 sub
          a := { a with ints := a.ints ++ vs }
          cur := cur2
        else
          throw s!"AttributeProto.ints: unexpected wire type {wt.toNat}"
      | 9 => -- strings
        if wt != wireBytes then throw s!"AttributeProto.strings: unexpected wire type {wt.toNat}"
        let (bs, cur2) ← readProtoBytes cur1
        a := { a with strings := a.strings.push bs }
        cur := cur2
      | 11 => -- graphs (repeated GraphProto): validate nested depth
        if wt != wireBytes then throw s!"AttributeProto.graphs: unexpected wire type {wt.toNat}"
        let (sub, cur2) ← readProtoBytes cur1
        let _ ← decodeGraphProto sub d
        cur := cur2
      | 20 => -- type
        if wt != wireVarint then throw s!"AttributeProto.type: unexpected wire type {wt.toNat}"
        let (v, cur2) ← readVarint cur1
        a := { a with attrType := v.toNat }
        cur := cur2
      | _ =>
        cur ← skipField cur1 wt
    pure a

end

/-- Decode a binary ONNX `ModelProto` buffer. -/
def parseOnnx (data : ByteArray) : Except String OnnxModel := do
  if data.isEmpty then
    throw "onnx buffer is empty"
  let mut cur : ProtoCursor := { buf := data, pos := 0 }
  let mut m : OnnxModel := {}
  for _ in [:data.size] do
    if cur.done then break
    let (fn, wt, cur1) ← nextTag cur
    match fn with
    | 1 => -- ir_version (int64)
      if wt != wireVarint then throw s!"ModelProto.ir_version: unexpected wire type {wt.toNat}"
      let (v, cur2) ← readVarint cur1
      m := { m with irVersion := u64ToInt64 v }
      cur := cur2
    | 2 => -- producer_name (string)
      if wt != wireBytes then throw s!"ModelProto.producer_name: unexpected wire type {wt.toNat}"
      let (s, cur2) ← readProtoString cur1
      m := { m with producerName := s }
      cur := cur2
    | 7 => -- graph (GraphProto)
      if wt != wireBytes then throw s!"ModelProto.graph: unexpected wire type {wt.toNat}"
      let (sub, cur2) ← readProtoBytes cur1
      let g ← decodeGraphProto sub maxProtoDepth
      m := { m with graph := g, hasGraph := true }
      cur := cur2
    | 8 => -- opset_import (repeated OperatorSetIdProto)
      if wt != wireBytes then throw s!"ModelProto.opset_import: unexpected wire type {wt.toNat}"
      let (sub, cur2) ← readProtoBytes cur1
      let op ← decodeOpsetImport sub
      m := { m with opsetImports := m.opsetImports.push op }
      cur := cur2
    | _ =>
      cur ← skipField cur1 wt
  if !m.hasGraph then
    throw "ModelProto: missing graph field"
  pure m

/-- Read and decode a binary `.onnx` file from disk. -/
def loadOnnx (path : System.FilePath) : IO (Except String OnnxModel) := do
  let data ← try
    IO.FS.readBinFile path
  catch e =>
    return .error s!"read onnx '{path}': {e}"
  return parseOnnx data

-- ============================================================================
-- Pure `ProtoBuilder` Encoder (for Self-Contained Binary `.onnx` Tests)
-- ============================================================================

structure ProtoBuilder where
  buf : ByteArray := ByteArray.empty
  deriving BEq, Inhabited

namespace ProtoBuilder

def empty : ProtoBuilder := { buf := ByteArray.empty }

def appendVarint (b : ProtoBuilder) (v0 : UInt64) : ProtoBuilder := Id.run do
  let mut out := b.buf
  let mut v := v0
  for _ in [:10] do
    if v < 0x80 then
      out := out.push v.toUInt8
      return { buf := out }
    out := out.push ((v &&& 0x7f).toUInt8 ||| 0x80)
    v := v >>> 7
  return { buf := out }

def appendTag (b : ProtoBuilder) (fieldNum : Nat) (wireType : UInt8) : ProtoBuilder :=
  b.appendVarint ((fieldNum.toUInt64 <<< 3) ||| wireType.toUInt64)

def writeVarint (b : ProtoBuilder) (fieldNum : Nat) (v : UInt64) : ProtoBuilder :=
  (b.appendTag fieldNum wireVarint).appendVarint v

def writeInt64 (b : ProtoBuilder) (fieldNum : Nat) (v : Int) : ProtoBuilder :=
  b.writeVarint fieldNum (int64ToU64 v)

def writeFloat32 (b : ProtoBuilder) (fieldNum : Nat) (v : Float) : ProtoBuilder :=
  let b1 := b.appendTag fieldNum wireFixed32
  { buf := pushU32LE b1.buf (floatToF32Bits v) }

def writeFloat64 (b : ProtoBuilder) (fieldNum : Nat) (v : Float) : ProtoBuilder :=
  let b1 := b.appendTag fieldNum wireFixed64
  { buf := pushU64LE b1.buf v.toBits }

def writeBytes (b : ProtoBuilder) (fieldNum : Nat) (data : ByteArray) : ProtoBuilder :=
  let b1 := (b.appendTag fieldNum wireBytes).appendVarint data.size.toUInt64
  { buf := b1.buf ++ data }

def writeString (b : ProtoBuilder) (fieldNum : Nat) (s : String) : ProtoBuilder :=
  b.writeBytes fieldNum s.toUTF8

def writeMessage (b : ProtoBuilder) (fieldNum : Nat) (sub : ProtoBuilder) : ProtoBuilder :=
  b.writeBytes fieldNum sub.buf

def writePackedInt64 (b : ProtoBuilder) (fieldNum : Nat) (vals : Array Int) : ProtoBuilder :=
  let sub := vals.foldl (fun acc v => acc.appendVarint (int64ToU64 v)) ProtoBuilder.empty
  b.writeBytes fieldNum sub.buf

def writePackedInt32 (b : ProtoBuilder) (fieldNum : Nat) (vals : Array Int) : ProtoBuilder :=
  let sub := vals.foldl (fun acc v => acc.appendVarint (int64ToU64 v)) ProtoBuilder.empty
  b.writeBytes fieldNum sub.buf

def writePackedFloat32 (b : ProtoBuilder) (fieldNum : Nat) (vals : Array Float) : ProtoBuilder := Id.run do
  let mut raw := ByteArray.emptyWithCapacity (4 * vals.size)
  for v in vals do
    raw := pushU32LE raw (floatToF32Bits v)
  return b.writeBytes fieldNum raw

def writePackedFloat64 (b : ProtoBuilder) (fieldNum : Nat) (vals : Array Float) : ProtoBuilder := Id.run do
  let mut raw := ByteArray.emptyWithCapacity (8 * vals.size)
  for v in vals do
    raw := pushU64LE raw v.toBits
  return b.writeBytes fieldNum raw

def bytes (b : ProtoBuilder) : ByteArray :=
  b.buf

end ProtoBuilder

end Vane.Frontend.Onnx
