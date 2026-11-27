import Lean.Data.Json
import Vane.Binary
import Vane.Float16

namespace Vane.Frontend.SafeTensors

open Lean (Json)
open Vane.Binary
open Vane.Float16

/-- Maximum allowed SafeTensors JSON header size in bytes (100 MiB). -/
def maxHeaderBytes : Nat := 100 * 1024 * 1024

/-- Options controlling SafeTensors parsing validation. -/
structure SafeTensorsOptions where
  /-- Reject overlapping non-empty tensor spans in the data section. -/
  strict : Bool := false
  deriving Repr, BEq, Inhabited

/-- One tensor descriptor entry parsed from a SafeTensors header. -/
structure SafeTensorInfo where
  name : String
  dtype : String
  shape : Array Nat
  beginOff : Nat
  endOff : Nat
  byteLen : Nat
  deriving Repr, BEq, Inhabited

/-- A validated SafeTensors archive held in memory. -/
structure SafeTensorsFile where
  metadata : Array (String × String)
  tensors : Array SafeTensorInfo
  dataBase : Nat
  raw : ByteArray
  deriving BEq, Inhabited

/-- Input descriptor for encoding a `.safetensors` buffer with `writeSafeTensors`. -/
structure SafeTensorWrite where
  name : String
  dtype : String
  shape : Array Nat
  data : ByteArray
  deriving BEq, Inhabited

/-- Record of one tensor exported to an fp16 `.bin` file. -/
structure ExportedTensor where
  name : String
  dtype : String
  shape : Array Nat
  file : String
  outBytes : Nat
  deriving Repr, BEq, Inhabited

/-- Half-open tensor byte span `[beginOff, endOff)` relative to the data section start. -/
structure TensorSpan where
  name : String
  beginOff : Nat
  endOff : Nat
  deriving Repr, BEq, Inhabited

/-- Element width in bytes for a recognized SafeTensors dtype string. -/
def dtypeBytes? (dtype : String) : Option Nat :=
  match dtype with
  | "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" => some 1
  | "F16" | "BF16" | "I16" | "U16" => some 2
  | "F32" | "I32" | "U32" => some 4
  | "F64" | "I64" | "U64" => some 8
  | _ => none

/-- Pure product of shape dimensions (scalar `[]` has 1 element). -/
def shapeProduct : List Nat → Nat
  | [] => 1
  | d :: rest => d * shapeProduct rest

/-- Overflow-checked element count product of a tensor shape (bounded by `maxU64Nat`). -/
def shapeProductChecked (shape : Array Nat) (label : String) : Except String Nat := do
  let mut prod : Nat := 1
  for i in [:shape.size] do
    let d := shape[i]!
    let next := prod * d
    if next > maxU64Nat then
      throw s!"{label}: shape element count overflow at dim[{i}]={d}"
    prod := next
  pure prod

/-- Check that span `[beginOff, endOff)` is ordered and fits inside `fileSize` after `dataBase`. -/
def validSpan (dataBase fileSize : Nat) (s : TensorSpan) : Bool :=
  decide (s.beginOff ≤ s.endOff ∧ dataBase + s.endOff ≤ fileSize)

/-- Check that two half-open byte spans `[a.beginOff, a.endOff)` and `[b.beginOff, b.endOff)` do not overlap. -/
def spansDisjoint (a b : TensorSpan) : Bool :=
  decide (a.endOff ≤ b.beginOff ∨ b.endOff ≤ a.beginOff)

/-- Check that a list of spans sorted by `beginOff` has no adjacent overlaps (`a.endOff <= b.beginOff`). -/
def checkSortedSpansNonOverlapping : List TensorSpan → Bool
  | [] => true
  | [_] => true
  | a :: b :: rest =>
    decide (a.endOff ≤ b.beginOff) && checkSortedSpansNonOverlapping (b :: rest)

/-- Find the first adjacent overlapping pair in a sorted span list for error diagnostics. -/
def findSortedSpanOverlap : List TensorSpan → Option (TensorSpan × TensorSpan)
  | [] => none
  | [_] => none
  | a :: b :: rest =>
    if a.endOff > b.beginOff then some (a, b)
    else findSortedSpanOverlap (b :: rest)

/-- Decode a little-endian signed 32-bit two's-complement integer as `Int`. -/
def u32ToInt32 (u : UInt32) : Int :=
  if u >= ((1 : UInt32) <<< 31) then
    (u.toNat : Int) - ((1 : Nat) <<< 32 : Int)
  else
    (u.toNat : Int)

/-- Decode a little-endian signed 64-bit two's-complement integer as `Int`. -/
def u64ToInt64 (u : UInt64) : Int :=
  if u >= ((1 : UInt64) <<< 63) then
    (u.toNat : Int) - ((1 : Nat) <<< 64 : Int)
  else
    (u.toNat : Int)

/-- Encode a signed `Int` into 64-bit two's-complement `UInt64`. -/
def int64ToU64 (v : Int) : UInt64 :=
  (v.emod ((1 : Nat) <<< 64 : Int)).toNat.toUInt64

/-- Encode a signed `Int` into 32-bit two's-complement `UInt32`. -/
def int32ToU32 (v : Int) : UInt32 :=
  (v.emod ((1 : Nat) <<< 32 : Int)).toNat.toUInt32

private def jsonObjField? (j : Json) (key : String) : Option Json :=
  match j.getObjVal? key with
  | .ok v => some v
  | .error _ => none

private def parseNonNegIntJson (j : Json) (ctx : String) : Except String Nat :=
  match j.getInt? with
  | .ok v =>
    if v < 0 then throw s!"{ctx}: negative integer {v}"
    else pure v.toNat
  | .error e => throw s!"{ctx}: expected integer ({e})"

/-- Parse and validate a binary `.safetensors` buffer. -/
def parseSafeTensors (data : ByteArray) (opts : SafeTensorsOptions := {}) : Except String SafeTensorsFile := do
  if data.size < 8 then
    throw s!"safetensors truncated: need at least 8 bytes, got {data.size}"
  let headerLen := (← readU64LE data 0).toNat
  if headerLen == 0 then
    throw "safetensors header_len must be > 0"
  if headerLen > maxHeaderBytes then
    throw s!"safetensors header_len {headerLen} exceeds limit {maxHeaderBytes}"
  let dataBase := 8 + headerLen
  if dataBase > data.size then
    throw s!"safetensors header_len {headerLen} exceeds remaining file size {data.size - 8}"

  let headerBytes := data.extract 8 dataBase
  let some headerStr := String.fromUTF8? headerBytes
    | throw "safetensors header is not valid UTF-8"
  let root ← match Json.parse headerStr with
    | .ok j => pure j
    | .error e => throw s!"safetensors invalid JSON header: {e}"
  let obj ← match root.getObj? with
    | .ok m => pure m
    | .error _ => throw "safetensors JSON header must be an object"

  let entries := obj.toArray.qsort fun a b => a.1 < b.1
  let mut metadata : Array (String × String) := #[]
  let mut tensors : Array SafeTensorInfo := #[]
  let mut spans : Array TensorSpan := #[]

  for (name, val) in entries do
    if name == "__metadata__" then
      if val.isNull then
        pure ()
      else
        let metaObj ← match val.getObj? with
          | .ok m => pure m
          | .error _ => throw "safetensors __metadata__ must be a JSON object"
        let metaEntries := metaObj.toArray.qsort fun a b => a.1 < b.1
        for (mk, mv) in metaEntries do
          let s := match mv.getStr? with
            | .ok str => str
            | .error _ => mv.compress
          metadata := metadata.push (mk, s)
    else
      let ctx := s!"safetensors tensor '{name}'"
      let _ ← match val.getObj? with
        | .ok m => pure m
        | .error _ => throw s!"{ctx}: descriptor must be a JSON object"
      let dtype ← match jsonObjField? val "dtype" with
        | none => throw s!"{ctx}: missing dtype"
        | some dj =>
          match dj.getStr? with
          | .ok s => if s.isEmpty then throw s!"{ctx}: empty dtype" else pure s
          | .error _ => throw s!"{ctx}: dtype must be a string"
      let some elemBytes := dtypeBytes? dtype
        | throw s!"{ctx}: unsupported dtype '{dtype}'"
      let shapeArr ← match jsonObjField? val "shape" with
        | none => throw s!"{ctx}: missing shape array"
        | some sj =>
          match sj.getArr? with
          | .ok a => pure a
          | .error _ => throw s!"{ctx}: shape must be an array"
      let mut shape : Array Nat := Array.mkEmpty shapeArr.size
      for idx in [:shapeArr.size] do
        let d ← parseNonNegIntJson shapeArr[idx]! s!"{ctx} shape[{idx}]"
        shape := shape.push d

      let offsetsArr ← match jsonObjField? val "data_offsets" with
        | none => throw s!"{ctx}: missing data_offsets"
        | some oj =>
          match oj.getArr? with
          | .ok a => pure a
          | .error _ => throw s!"{ctx}: data_offsets must be an array"
      if offsetsArr.size != 2 then
        throw s!"{ctx}: data_offsets must have 2 elements, got {offsetsArr.size}"
      let beginOff ← parseNonNegIntJson offsetsArr[0]! s!"{ctx} data_offsets[0]"
      let endOff ← parseNonNegIntJson offsetsArr[1]! s!"{ctx} data_offsets[1]"
      if beginOff > endOff then
        throw s!"{ctx}: data_offsets begin {beginOff} > end {endOff}"
      if dataBase + endOff > data.size then
        throw s!"{ctx}: data_offsets end {endOff} exceeds data section size {data.size - dataBase}"

      let numElems ← shapeProductChecked shape ctx
      let expectedBytes := numElems * elemBytes
      if expectedBytes > maxU64Nat then
        throw s!"{ctx}: byte size overflow ({numElems} * {elemBytes})"
      let actualBytes := endOff - beginOff
      if actualBytes != expectedBytes then
        throw s!"{ctx}: byte span {actualBytes} != shape {shape.toList} * {dtype} ({expectedBytes})"

      let info : SafeTensorInfo := {
        name := name
        dtype := dtype
        shape := shape
        beginOff := beginOff
        endOff := endOff
        byteLen := actualBytes
      }
      tensors := tensors.push info
      if actualBytes > 0 then
        spans := spans.push { name := name, beginOff := beginOff, endOff := endOff }

  if opts.strict && spans.size > 1 then
    let sortedSpans := spans.qsort fun a b =>
      if a.beginOff != b.beginOff then a.beginOff < b.beginOff else a.endOff < b.endOff
    if !checkSortedSpansNonOverlapping sortedSpans.toList then
      match findSortedSpanOverlap sortedSpans.toList with
      | some (a, b) =>
        throw s!"safetensors strict mode: overlapping spans between '{a.name}' [{a.beginOff},{a.endOff}) and '{b.name}' [{b.beginOff},{b.endOff})"
      | none =>
        throw "safetensors strict mode: overlapping tensor spans"

  pure {
    metadata := metadata
    tensors := tensors
    dataBase := dataBase
    raw := data
  }

/-- Read and parse a `.safetensors` file from disk. -/
def loadSafeTensors (path : System.FilePath) (opts : SafeTensorsOptions := {}) : IO (Except String SafeTensorsFile) := do
  let data ← try
    IO.FS.readBinFile path
  catch e =>
    return .error s!"read safetensors '{path}': {e}"
  return parseSafeTensors data opts

namespace SafeTensorsFile

/-- Look up tensor metadata by name. -/
def find? (sf : SafeTensorsFile) (name : String) : Option SafeTensorInfo :=
  sf.tensors.find? fun t => t.name == name

/-- Return the raw little-endian payload slice for tensor `name`. -/
def rawBytes (sf : SafeTensorsFile) (name : String) : Except String ByteArray := do
  let some info := sf.find? name
    | throw s!"safetensors: tensor '{name}' not found"
  let start := sf.dataBase + info.beginOff
  let stop := sf.dataBase + info.endOff
  if stop > sf.raw.size || start > stop then
    throw s!"safetensors: tensor '{name}' slice [{start}, {stop}) out of bounds ({sf.raw.size})"
  pure (sf.raw.extract start stop)

/-- Decode a float tensor (`F16`, `BF16`, `F32`, or `F64`) to finite binary64 `Float` values.
    Rejects `NaN` and `Inf`. -/
def toFloat64 (sf : SafeTensorsFile) (name : String) : Except String (Array Float) := do
  let some info := sf.find? name
    | throw s!"safetensors: tensor '{name}' not found"
  let raw ← sf.rawBytes name
  match info.dtype with
  | "F16" =>
    let n := raw.size / 2
    let mut out : Array Float := Array.mkEmpty n
    for i in [:n] do
      let u ← readU16LE raw (2 * i)
      if !f16IsFinite u then
        throw s!"safetensors tensor '{name}': non-finite F16 element at index {i} ({toHex u.toNat})"
      out := out.push (f16ToFloat u)
    pure out
  | "BF16" =>
    let n := raw.size / 2
    let mut out : Array Float := Array.mkEmpty n
    for i in [:n] do
      let b ← readU16LE raw (2 * i)
      let f := bf16ToFloat b
      if f.isNaN || f.isInf then
        throw s!"safetensors tensor '{name}': non-finite BF16 element at index {i} ({toHex b.toNat})"
      out := out.push f
    pure out
  | "F32" =>
    let n := raw.size / 4
    let mut out : Array Float := Array.mkEmpty n
    for i in [:n] do
      let u ← readU32LE raw (4 * i)
      let f := f32BitsToFloat u
      if f.isNaN || f.isInf then
        throw s!"safetensors tensor '{name}': non-finite F32 element at index {i} ({toHex u.toNat})"
      out := out.push f
    pure out
  | "F64" =>
    let n := raw.size / 8
    let mut out : Array Float := Array.mkEmpty n
    for i in [:n] do
      let u ← readU64LE raw (8 * i)
      let f := Float.ofBits u
      if f.isNaN || f.isInf then
        throw s!"safetensors tensor '{name}': non-finite F64 element at index {i} ({toHex u.toNat})"
      out := out.push f
    pure out
  | other =>
    throw s!"safetensors tensor '{name}': dtype '{other}' cannot be decoded as float64"

/-- Convert a float tensor (`F16`, `BF16`, `F32`, or `F64`) to finite IEEE-754 binary16 bit patterns.
    Rejects `NaN`, `Inf`, and values overflowing the finite fp16 range `[-65504, 65504]`. -/
def toFp16 (sf : SafeTensorsFile) (name : String) : Except String (Array UInt16) := do
  let some info := sf.find? name
    | throw s!"safetensors: tensor '{name}' not found"
  let raw ← sf.rawBytes name
  match info.dtype with
  | "F16" =>
    let n := raw.size / 2
    let mut out : Array UInt16 := Array.mkEmpty n
    for i in [:n] do
      let u ← readU16LE raw (2 * i)
      if !f16IsFinite u then
        throw s!"safetensors tensor '{name}': non-finite F16 element at index {i} ({toHex u.toNat})"
      out := out.push u
    pure out
  | "BF16" | "F32" | "F64" =>
    let vals ← sf.toFloat64 name
    let mut out : Array UInt16 := Array.mkEmpty vals.size
    for i in [:vals.size] do
      let f := vals[i]!
      let h := floatToF16 f
      if !f16IsFinite h then
        throw s!"safetensors tensor '{name}': {info.dtype} element at index {i} ({f}) overflows fp16"
      out := out.push h
    pure out
  | other =>
    throw s!"safetensors tensor '{name}': dtype '{other}' cannot be converted to fp16 (supported: F16, BF16, F32, F64)"

/-- Convert a float tensor (`F16`, `BF16`, `F32`, or `F64`) to little-endian fp16 bytes. -/
def toFp16Bytes (sf : SafeTensorsFile) (name : String) : Except String ByteArray := do
  let vals ← sf.toFp16 name
  let mut out := ByteArray.emptyWithCapacity (2 * vals.size)
  for h in vals do
    out := pushU16LE out h
  pure out

/-- Decode an integer or boolean tensor (`I64`, `I32`, or `BOOL`) into `Int` values. -/
def toInt64 (sf : SafeTensorsFile) (name : String) : Except String (Array Int) := do
  let some info := sf.find? name
    | throw s!"safetensors: tensor '{name}' not found"
  let raw ← sf.rawBytes name
  match info.dtype with
  | "I64" =>
    let n := raw.size / 8
    let mut out : Array Int := Array.mkEmpty n
    for i in [:n] do
      let u ← readU64LE raw (8 * i)
      out := out.push (u64ToInt64 u)
    pure out
  | "I32" =>
    let n := raw.size / 4
    let mut out : Array Int := Array.mkEmpty n
    for i in [:n] do
      let u ← readU32LE raw (4 * i)
      out := out.push (u32ToInt32 u)
    pure out
  | "BOOL" =>
    let mut out : Array Int := Array.mkEmpty raw.size
    for i in [:raw.size] do
      let b ← readU8 raw i
      out := out.push (if b != 0 then 1 else 0)
    pure out
  | other =>
    throw s!"safetensors tensor '{name}': dtype '{other}' cannot be decoded as int64"

/-- Decode a `BOOL` tensor into `Bool` values. -/
def toBool (sf : SafeTensorsFile) (name : String) : Except String (Array Bool) := do
  let some info := sf.find? name
    | throw s!"safetensors: tensor '{name}' not found"
  if info.dtype != "BOOL" then
    throw s!"safetensors tensor '{name}': dtype '{info.dtype}' is not BOOL"
  let raw ← sf.rawBytes name
  let mut out : Array Bool := Array.mkEmpty raw.size
  for i in [:raw.size] do
    let b ← readU8 raw i
    out := out.push (b != 0)
  pure out

private def sanitizeFilenameBase (name : String) : String :=
  let cleaned := String.ofList <| name.toList.map fun c =>
    if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
       c == '_' || c == '-' || c == '.' then c else '_'
  let chars := cleaned.toList
  let trimmedFront := chars.dropWhile fun c => c == '.' || c == '_'
  let trimmed := (trimmedFront.reverse.dropWhile fun c => c == '.' || c == '_').reverse
  let s := String.ofList trimmed
  if s.isEmpty then "tensor" else s

/-- Export every float tensor (`F16`, `BF16`, `F32`, `F64`) to little-endian fp16 `.bin` files in `outDir`.
    Non-float tensors (`I64`, `I32`, `BOOL`, etc.) are skipped. -/
def exportFp16Dir (sf : SafeTensorsFile) (outDir : System.FilePath) : IO (Except String (Array ExportedTensor)) := do
  try
    IO.FS.createDirAll outDir
  catch e =>
    return .error s!"create export dir '{outDir}': {e}"

  let mut usedFiles : Array String := #[]
  let mut exported : Array ExportedTensor := #[]
  for info in sf.tensors do
    if info.dtype == "F16" || info.dtype == "BF16" || info.dtype == "F32" || info.dtype == "F64" then
      let buf ← match sf.toFp16Bytes info.name with
        | .ok b => pure b
        | .error e => return .error e
      let base := sanitizeFilenameBase info.name
      let mut fname := s!"{base}.bin"
      if usedFiles.contains fname then
        for suffix in [2 : sf.tensors.size + 3] do
          let cand := s!"{base}_{suffix}.bin"
          if !usedFiles.contains cand then
            fname := cand
            break
      usedFiles := usedFiles.push fname
      try
        IO.FS.writeBinFile (outDir / fname) buf
      catch e =>
        return .error s!"write '{fname}': {e}"
      exported := exported.push {
        name := info.name
        dtype := info.dtype
        shape := info.shape
        file := fname
        outBytes := buf.size
      }
  return .ok exported

end SafeTensorsFile

/-- Encode a deterministic `.safetensors` binary buffer from tensor payloads and optional `__metadata__`. -/
def writeSafeTensors (tensors : Array SafeTensorWrite) (metadata : Array (String × String) := #[]) : Except String ByteArray := do
  let sorted := tensors.qsort fun a b => a.name < b.name
  let mut seen : Array String := #[]
  let mut dataBuf := ByteArray.empty
  let mut headerFields : List (String × Json) := []

  if !metadata.isEmpty then
    let sortedMeta := metadata.qsort fun a b => a.1 < b.1
    let metaFields := sortedMeta.toList.map fun (k, v) => (k, Json.str v)
    headerFields := headerFields ++ [("__metadata__", Json.mkObj metaFields)]

  for t in sorted do
    if t.name.isEmpty then
      throw "safetensors write: empty tensor name"
    if t.name == "__metadata__" then
      throw "safetensors write: tensor name cannot be __metadata__"
    if seen.contains t.name then
      throw s!"safetensors write: duplicate tensor name '{t.name}'"
    seen := seen.push t.name

    let some elemBytes := dtypeBytes? t.dtype
      | throw s!"safetensors write: unsupported dtype '{t.dtype}' for '{t.name}'"
    let numElems ← shapeProductChecked t.shape s!"safetensors write '{t.name}'"
    let wantBytes := numElems * elemBytes
    if wantBytes > maxU64Nat then
      throw s!"safetensors write '{t.name}': byte size overflow"
    if t.data.size != wantBytes then
      throw s!"safetensors write '{t.name}': data length {t.data.size} != expected {wantBytes}"

    let beginOff := dataBuf.size
    dataBuf := dataBuf ++ t.data
    let endOff := dataBuf.size

    let shapeJson := Json.arr (t.shape.map fun (d : Nat) => Json.num d)
    let offsJson := Json.arr #[Json.num beginOff, Json.num endOff]
    let entryJson := Json.mkObj [
      ("dtype", Json.str t.dtype),
      ("shape", shapeJson),
      ("data_offsets", offsJson)
    ]
    headerFields := headerFields ++ [(t.name, entryJson)]

  let headerStr := (Json.mkObj headerFields).compress
  let headerBytes := headerStr.toUTF8
  if headerBytes.size == 0 || headerBytes.size > maxHeaderBytes then
    throw s!"safetensors write: header size {headerBytes.size} out of bounds (1..{maxHeaderBytes})"
  let headerLenU64 ← natToU64Checked headerBytes.size "safetensors header_len"
  let mut out := ByteArray.emptyWithCapacity (8 + headerBytes.size + dataBuf.size)
  out := pushU64LE out headerLenU64
  out := out ++ headerBytes
  out := out ++ dataBuf
  pure out

end Vane.Frontend.SafeTensors
