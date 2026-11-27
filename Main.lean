import Lean.Data.Json
import Vane

open Lean (Json)
open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Tiling
open Vane.Quant
open Vane.Verify
open Vane.Pe
open Vane.Lut
open Vane.Pool
open Vane.Conv
open Vane.Conv1x1
open Vane.Conv3x3
open Vane.View
open Vane.Matmul
open Vane.Graph
open Vane.Frontend.SafeTensors
open Vane.Frontend.Onnx
open Vane.Frontend.Lower

private def usage : String :=
  "Usage:\n" ++
  "  vane validate --hwx <path> [--sections <dir>] [--check-tasks]\n" ++
  "  vane sections --hwx <path> --out-dir <dir>\n" ++
  "  vane emit-demo --target <h16|h17> --out-dir <dir> [--with-kernel] [--with-scratch]\n" ++
  "  vane compile-conv1x1 --target <h16|h17> --h <H> --w <W> --cin <Cin> --cout <Cout>\n" ++
  "                       --weights <path> [--bias <path>] [--relu | --act <name> [act params]]\n" ++
  "                       [--weight-format <fp16|int8|uint8|int4|uint4|pal1|pal2|pal4|pal6|pal8>]\n" ++
  "                       [--scales <path>] [--zero-points <path>] [--palette <path>] --out-dir <dir>\n" ++
  "  vane compile-linear --target <h16|h17> --m <M> --k <K> --n <N> [--batch <B>]\n" ++
  "                      --weights <path> [--bias <path>] [--relu | --act <name> [act params]]\n" ++
  "                      [--weight-format <fp16|int8|uint8|int4|uint4|pal1|pal2|pal4|pal6|pal8>]\n" ++
  "                      [--scales <path>] [--zero-points <path>] [--palette <path>] --out-dir <dir>\n" ++
  "  vane compile-conv3x3 --target <h16|h17> --mode <dense|depthwise>\n" ++
  "                       --h <H> --w <W> --cin <Cin> --cout <Cout>\n" ++
  "                       --stride <1|2> --pad <same|valid>\n" ++
  "                       --weights <path> [--bias <path>] [--relu | --act <name> [act params]]\n" ++
  "                       [--weight-format <fp16|int8|uint8|int4|uint4|pal1|pal2|pal4|pal6|pal8>]\n" ++
  "                       [--scales <path>] [--zero-points <path>] [--palette <path>] --out-dir <dir>\n" ++
  "  vane compile-pe --target <h16|h17> --op <abs|square|linear|chain|sadd|ssub|srsub|smul|sdiv|add|sub|rsub|mul|max|min>\n" ++
  "                  --h <H> --w <W> --c <C> [--z-h <ZH> --z-w <ZW> --z-c <ZC>]\n" ++
  "                  [--alpha F] [--beta F] [--in-scale F] [--bias F] [--scale F] [--scalar F]\n" ++
  "                  --out-dir <dir>\n" ++
  "  vane compile-lut --target <h16|h17> --act <relu|copy|relu6|clip|leaky_relu|abs|sign|threshold|thresholded_relu|hard_sigmoid|clamp_max|clamp_min|elu|tanh|scaled_tanh|erf|sigmoid|gelu|silu|exp2|exp|sqrt|rsqrt|inverse|log2>\n" ++
  "                   --h <H> --w <W> --c <C>\n" ++
  "                   [--alpha F] [--beta F] [--lo F] [--hi F] [--eps F] [--scalar F]\n" ++
  "                   --out-dir <dir>\n" ++
  "  vane compile-pool --target <h16|h17> --op <max|avg> --h <H> --w <W> --c <C>\n" ++
  "                    --kh <KH> --kw <KW> [--sh <SH>] [--sw <SW>] [--pad <valid|same|custom>]\n" ++
  "                    [--pt N --pb N --pl N --pr N] [--exclude-pad <true|false>] --out-dir <dir>\n" ++
  "  vane compile-reduce --target <h16|h17> --op <sum|mean|max|min|l1|sum_square|l2|inv_sum|rsqrt_sum_square|rsqrt_mean_square>\n" ++
  "                      --h <H> --w <W> --c <C> --axes <w|h|c|hw|cw|ch|chw> [--eps F] --out-dir <dir>\n" ++
  "  vane compile-view --target <h16|h17> --op <transpose|slice|concat|split|reshape> [op flags] --out-dir <dir>\n" ++
  "  vane compile-matmul --target <h16|h17> [--b <B>] --m <M> --k <K> --n <N> [--transpose-a] [--transpose-b] [--scale <F>] --out-dir <dir>\n" ++
  "  vane compile-graph --spec <graph.json> [--strip-height <off|auto|N>] --out-dir <dir>\n" ++
  "  vane compile-onnx --target <h16|h17> --model <path> [--safetensors <path>] [--input-shape NAME=C,H,W]...\n" ++
  "                    [--placement auto|l2|dram] [--no-fuse] [--allow-view-ops=true|false]\n" ++
  "                    [--quantize <int8|int4|pal4|pal2>] [--dump-ir[=high|core]]\n" ++
  "                    [--emit-graph-dir <dir>] --out-dir <dir>\n" ++
  "  vane inspect-safetensors --file <path> [--export-dir <dir>] [--strict] [--json]\n"

private def requireArgValue (flag : String) (rest : List String) : Except String (String × List String) :=
  match rest with
  | v :: tl => pure (v, tl)
  | [] => throw s!"missing value for {flag}"

private def parsePosNat (flag s : String) : Except String Nat :=
  match s.toNat? with
  | some n =>
    if n == 0 then throw s!"{flag} must be >= 1, got 0" else pure n
  | none => throw s!"invalid positive integer '{s}' for {flag}"

private def parseNonNegNat (flag s : String) : Except String Nat :=
  match s.toNat? with
  | some n => pure n
  | none => throw s!"invalid non-negative integer '{s}' for {flag}"

private def parseBoolArg (flag s : String) : Except String Bool :=
  match s.toLower with
  | "true" | "1" => pure true
  | "false" | "0" => pure false
  | _ => throw s!"invalid boolean '{s}' for {flag} (expected true or false)"

private def parseFloatArg (flag s : String) : Except String Float :=
  match parseFloat? s with
  | some f =>
    if f.isNaN || f.isInf then
      throw s!"{flag} must be a finite floating-point number, got '{s}'"
    else
      pure f
  | none => throw s!"invalid floating-point value '{s}' for {flag}"

private def parseF16Arg (flag s : String) : Except String UInt16 := do
  let f ← parseFloatArg flag s
  pure (floatToF16CanonicalZero f)

private structure ActParamFlags where
  alpha : Option Float := none
  beta : Option Float := none
  lo : Option Float := none
  hi : Option Float := none
  eps : Option Float := none
  scalar : Option Float := none

private def parseLutSpec (actName : String) (p : ActParamFlags) : Except String LutSpec :=
  match actName.toLower with
  | "relu6" => pure .relu6
  | "clip" =>
    match p.lo, p.hi with
    | some lo, some hi => pure (.clip lo hi)
    | _, _ => throw "clip activation requires both --lo/--act-lo and --hi/--act-hi"
  | "leaky_relu" => pure (.leakyRelu (p.alpha.getD 0.01))
  | "abs" => pure .abs
  | "sign" => pure .sign
  | "threshold" =>
    let a := p.scalar.orElse (fun () => p.alpha) |>.orElse (fun () => p.lo) |>.getD 0.0
    pure (.threshold a)
  | "thresholded_relu" =>
    let a := p.alpha.orElse (fun () => p.scalar) |>.orElse (fun () => p.lo) |>.getD 1.0
    pure (.thresholdedRelu a)
  | "hard_sigmoid" => pure (.hardSigmoid (p.alpha.getD 0.2) (p.beta.getD 0.5))
  | "clamp_max" =>
    let v := p.scalar.orElse (fun () => p.lo) |>.orElse (fun () => p.alpha) |>.getD 0.0
    pure (.clampMax v)
  | "clamp_min" =>
    let v := p.scalar.orElse (fun () => p.hi) |>.orElse (fun () => p.alpha) |>.getD 0.0
    pure (.clampMin v)
  | "elu" => pure (.elu (p.alpha.getD 1.0))
  | "tanh" => pure .tanh
  | "scaled_tanh" => pure (.scaledTanh (p.alpha.getD 1.0) (p.beta.getD 1.0))
  | "erf" => pure .erf
  | "sigmoid" => pure .sigmoid
  | "gelu" => pure .gelu
  | "silu" => pure .silu
  | "exp2" => pure .exp2
  | "exp" => pure .exp
  | "sqrt" => pure .sqrt
  | "rsqrt" => pure (.rsqrt (p.eps.getD 0.0))
  | "inverse" => pure .inverse
  | "log2" => pure .log2
  | other => throw s!"unknown LUT activation '{other}'"

private def parseNeActSpec (actName : String) (p : ActParamFlags) : Except String NeActSpec :=
  match actName.toLower with
  | "relu" => pure .relu
  | "copy" => pure .copy
  | other => NeActSpec.lut <$> parseLutSpec other p

private def parseConvAct (reluFlag : Bool) (actName? : Option String) (p : ActParamFlags) : Except String ConvActivation :=
  match actName? with
  | none => pure (if reluFlag then .relu else .none)
  | some raw =>
    let low := raw.toLower
    if low == "none" then
      if reluFlag then throw "cannot combine --relu with --act none"
      else pure .none
    else if low == "relu" then
      pure .relu
    else if reluFlag then
      throw s!"cannot combine --relu with --act {raw}"
    else
      ConvActivation.lut <$> parseLutSpec low p

private def readF32ArrayLE (bs : ByteArray) (label : String) : Except String (Array Float) := do
  if bs.size % 4 != 0 then
    throw s!"{label}: byte size {bs.size} is not a multiple of 4 (raw LE float32)"
  let n := bs.size / 4
  let mut out : Array Float := Array.mkEmpty n
  for i in [:n] do
    let u ← readU32LE bs (4 * i)
    out := out.push (f32BitsToFloat u)
  pure out

private def readU16ArrayLE (bs : ByteArray) (label : String) : Except String (Array UInt16) := do
  if bs.size % 2 != 0 then
    throw s!"{label}: byte size {bs.size} is not a multiple of 2 (raw LE uint16)"
  let n := bs.size / 2
  let mut out : Array UInt16 := Array.mkEmpty n
  for i in [:n] do
    let u ← readU16LE bs (2 * i)
    out := out.push u
  pure out

private def loadQuantWeightSpecIO
    (fmtStr : String)
    (codesBytes : ByteArray)
    (scalesPath? zpPath? palPath? : Option System.FilePath)
    (ctx : String) : IO (Except String (Option QuantWeightSpec)) := do
  let low := fmtStr.toLower
  if low == "fp16" then
    if scalesPath?.isSome || zpPath?.isSome || palPath?.isSome then
      return .error s!"{ctx}: --scales, --zero-points, and --palette cannot be used with fp16 weights"
    return .ok none
  let some fmt := QuantFormat.parse? low
    | return .error s!"{ctx}: unknown weight format '{fmtStr}' (expected fp16, int8, uint8, int4, uint4, pal1, pal2, pal4, pal6, or pal8)"
  if fmt.isInteger then
    if palPath?.isSome then
      return .error s!"{ctx}: integer weight format '{fmt.name}' cannot specify a palette"
    let some sPath := scalesPath?
      | return .error s!"{ctx}: integer weight format '{fmt.name}' requires scales"
    let scalesBytes ← try IO.FS.readBinFile sPath catch e =>
      return .error s!"{ctx}: failed to read scales '{sPath}': {e}"
    let scales ← match readF32ArrayLE scalesBytes s!"{ctx} scales" with
      | .ok arr => pure arr
      | .error e => return .error e
    let zp? ← match zpPath? with
      | none => pure none
      | some zPath =>
        let zBytes ← try IO.FS.readBinFile zPath catch e =>
          return .error s!"{ctx}: failed to read zero-points '{zPath}': {e}"
        pure (some zBytes.data)
    return .ok (some {
      format := fmt
      codes := codesBytes
      scales := scales
      zeroPoints := zp?
      palette := #[]
    })
  else
    if scalesPath?.isSome || zpPath?.isSome then
      return .error s!"{ctx}: palette weight format '{fmt.name}' cannot specify scales or zero-points"
    let some pPath := palPath?
      | return .error s!"{ctx}: palette weight format '{fmt.name}' requires a palette file"
    let palBytes ← try IO.FS.readBinFile pPath catch e =>
      return .error s!"{ctx}: failed to read palette '{pPath}': {e}"
    let pal ← match readU16ArrayLE palBytes s!"{ctx} palette" with
      | .ok arr => pure arr
      | .error e => return .error e
    return .ok (some {
      format := fmt
      codes := codesBytes
      scales := #[]
      zeroPoints := none
      palette := pal
    })

private def compiledSplitToConv1x1 (c : CompiledConv1x1Split) : CompiledConv1x1 := {
  target := c.target
  shape := c.shape
  tiling := c.tilings[0]!
  slices := c.slices
  hasBias := c.hasBias
  relu := c.relu
  activation := c.activation
  lutBytes := c.lutBytes
  direct := c.direct
  l2UseBytes := c.l2UseBytes
  kernelBytes := c.kernelBytes
  hwx := c.hwx
  info := c.info
  sections := c.sections
}

private def formatBarSlots (slots : Array UInt8) : String :=
  let parts := slots.toList.map fun s => toString s.toNat
  "[" ++ String.intercalate ", " parts ++ "]"

private def escapeJsonString (s : String) : String :=
  s.foldl (fun acc c =>
    match c with
    | '"' => acc ++ "\\\""
    | '\\' => acc ++ "\\\\"
    | '\n' => acc ++ "\\n"
    | '\r' => acc ++ "\\r"
    | '\t' => acc ++ "\\t"
    | _ => acc.push c
  ) ""

private def buildDemoProgram (target : Target) (withKernel withScratch : Bool) : Except String HwxProgram := do
  let sIn := SurfaceSpec.packed "in0" "input_0" .input 1 16 8 8
  let sOut := SurfaceSpec.packed "out0" "output_0" .output 1 16 8 8

  let mut recs : Array RegRecord := #[
    .sequential 0x0000 (some 6) #[0x00000001, 0x00000002],
    .sequential 0x1040 (some 7) #[0x00000010, 0x00000020],
    .bitmask 0x1140 0x0005 #[0x00000100, 0x00000200, 0x00000300]
  ]

  let mut kernelPayload : Option ByteArray := none
  let mut kernelRecIdx : Option Nat := none
  if withKernel then
    let mut kb := zeroBytes 256
    kb ← putU32LE kb 0 0x12345678
    kb ← putU32LE kb 64 0x9abcdef0
    kernelPayload := some kb
    kernelRecIdx := some recs.size
    recs := recs.push (.sequential 0x1340 (some 8) #[0x00000040, 0x00000000])

  let mut scratchSpec : Option ScratchSpec := none
  if withScratch then
    scratchSpec := some { size := 0x4000, usable := 0x2000 }
    recs := recs.push (.sequential 0x1240 (some 3) #[0x00000000])

  let task := Task.mk9 recs
  let stream : TaskStream := { prologue := 1, tasks := #[task] }

  let mut relocs : Array RelocationSpec := #[]
  if let some kRecIdx := kernelRecIdx then
    let enc ← encodeTaskStream target stream
    match enc.payloadSites.find? (fun (ti, ri, pi, _) => ti == 0 && ri == kRecIdx && pi == 0) with
    | some (_, _, _, off) => relocs := #[{ patchOffset := off.toUInt32 }]
    | none => throw "internal error: failed to locate kernel relocation payload offset"

  pure {
    target := target
    surfaces := #[sIn, sOut]
    constants := zeroBytes 32
    kernel := kernelPayload
    scratch := scratchSpec
    taskStream := stream
    relocations := relocs
    ident := some "vane"
  }

private def renderSurfacesJson (surfaces : Array ValidatedSurface) : String :=
  let surfaceItems := surfaces.toList.map fun s =>
    "    {\n" ++
    s!"      \"short_name\": \"{escapeJsonString s.shortName}\",\n" ++
    s!"      \"symbol_name\": \"{escapeJsonString s.symbolName}\",\n" ++
    s!"      \"direction\": \"{s.direction.name}\",\n" ++
    s!"      \"shape\": [{s.n}, {s.channels}, {s.height}, {s.width}],\n" ++
    s!"      \"strides\": [{s.batchStride}, {s.planeStride}, {s.rowStride}],\n" ++
    s!"      \"span\": {s.span},\n" ++
    s!"      \"section_size\": {s.sect.size},\n" ++
    s!"      \"alloc_size\": {s.vmSize}\n" ++
    "    }"
  String.intercalate ",\n" surfaceItems

private def renderManifestJson (info : HwxInfo) (secs : LoadSections) : String :=
  "{\n" ++
  s!"  \"target\": \"{info.target.name}\",\n" ++
  "  \"hwx\": \"program.hwx\",\n" ++
  "  \"sections\": {\n" ++
  "    \"generic\": {\"file\": \"generic.bin\", \"size\": " ++ toString secs.generic.size ++ "},\n" ++
  "    \"operation\": {\"file\": \"operation.bin\", \"size\": " ++ toString secs.operation.size ++ "},\n" ++
  "    \"procedure\": {\"file\": \"procedure.bin\", \"size\": " ++ toString secs.procedure.size ++ "},\n" ++
  "    \"text_properties\": {\"file\": \"text-properties.bin\", \"size\": " ++ toString secs.textProperties.size ++ "}\n" ++
  "  },\n" ++
  s!"  \"task_count\": {info.taskStream.taskCount},\n" ++
  s!"  \"record_count\": {info.records},\n" ++
  s!"  \"relocation_count\": {info.relocCount},\n" ++
  s!"  \"has_kernel\": {if info.kernelSection.isSome then "true" else "false"},\n" ++
  s!"  \"has_scratch\": {if info.scratchSection.isSome then "true" else "false"},\n" ++
  "  \"surfaces\": [\n" ++
  renderSurfacesJson info.surfaces ++ "\n" ++
  "  ]\n" ++
  "}\n"

private def formatNatListJson (xs : Array Nat) : String :=
  "[" ++ String.intercalate ", " (xs.toList.map toString) ++ "]"

private def renderConv1x1ManifestJson (compiled : CompiledConv1x1) (opLabel : String := "conv1x1")
    (weightFormat : String := "fp16") : String :=
  let info := compiled.info
  let secs := compiled.sections
  let sh := compiled.shape
  let t := compiled.tiling
  let lutLine := if compiled.lutBytes.isSome then "  \"lut_file\": \"lut.bin\",\n" else ""
  let wfLine := if weightFormat != "fp16" then s!"  \"weight_format\": \"{escapeJsonString weightFormat}\",\n" else ""
  "{\n" ++
  s!"  \"op\": \"{escapeJsonString opLabel}\",\n" ++
  wfLine ++
  s!"  \"target\": \"{compiled.target.name}\",\n" ++
  "  \"hwx\": \"program.hwx\",\n" ++
  "  \"sections\": {\n" ++
  "    \"generic\": {\"file\": \"generic.bin\", \"size\": " ++ toString secs.generic.size ++ "},\n" ++
  "    \"operation\": {\"file\": \"operation.bin\", \"size\": " ++ toString secs.operation.size ++ "},\n" ++
  "    \"procedure\": {\"file\": \"procedure.bin\", \"size\": " ++ toString secs.procedure.size ++ "},\n" ++
  "    \"text_properties\": {\"file\": \"text-properties.bin\", \"size\": " ++ toString secs.textProperties.size ++ "}\n" ++
  "  },\n" ++
  "  \"shape\": {\n" ++
  s!"    \"h\": {sh.h},\n" ++
  s!"    \"w\": {sh.w},\n" ++
  s!"    \"cin\": {sh.cin},\n" ++
  s!"    \"cout\": {sh.cout}\n" ++
  "  },\n" ++
  s!"  \"slices\": {formatNatListJson compiled.slices},\n" ++
  s!"  \"has_bias\": {if compiled.hasBias then "true" else "false"},\n" ++
  s!"  \"relu\": {if compiled.relu then "true" else "false"},\n" ++
  s!"  \"activation\": \"{escapeJsonString compiled.activation.name}\",\n" ++
  lutLine ++
  "  \"tiling\": {\n" ++
  s!"    \"log_ne\": {t.logNE},\n" ++
  s!"    \"f\": {t.f},\n" ++
  s!"    \"fat\": {t.fat},\n" ++
  s!"    \"ocg\": {t.ocg},\n" ++
  s!"    \"wu\": {t.wu},\n" ++
  s!"    \"sh_pref\": {t.shPref},\n" ++
  s!"    \"sh_max\": {t.shMax},\n" ++
  s!"    \"tile_height\": {t.tileHeight},\n" ++
  s!"    \"src_skew\": {t.srcSkew},\n" ++
  s!"    \"res_skew\": {t.resSkew}\n" ++
  "  },\n" ++
  s!"  \"direct\": {if compiled.direct then "true" else "false"},\n" ++
  s!"  \"l2_use_bytes\": {compiled.l2UseBytes},\n" ++
  s!"  \"kernel_bytes\": {compiled.kernelBytes},\n" ++
  s!"  \"task_count\": {info.taskStream.taskCount},\n" ++
  s!"  \"record_count\": {info.records},\n" ++
  s!"  \"relocation_count\": {info.relocCount},\n" ++
  s!"  \"has_kernel\": {if info.kernelSection.isSome then "true" else "false"},\n" ++
  s!"  \"has_scratch\": {if info.scratchSection.isSome then "true" else "false"},\n" ++
  "  \"surfaces\": [\n" ++
  renderSurfacesJson info.surfaces ++ "\n" ++
  "  ]\n" ++
  "}\n"

private def renderViewManifestJson (compiled : CompiledView) : String :=
  let info := compiled.info
  let secs := compiled.sections
  let pl := compiled.plan
  let outS := pl.outShape
  let copyItems := pl.copies.toList.map fun cp =>
    s!"    \{\"n\": {cp.n}, \"c\": {cp.c}, \"h\": {cp.h}, \"w\": {cp.w}, \"in_off\": {cp.inOff}, \"out_off\": {cp.outOff}, \"transpose_cw\": {if cp.transposeCW then "true" else "false"}}"
  "{\n" ++
  "  \"op\": \"view\",\n" ++
  s!"  \"view_op\": \"{escapeJsonString pl.opName}\",\n" ++
  s!"  \"target\": \"{compiled.target.name}\",\n" ++
  "  \"hwx\": \"program.hwx\",\n" ++
  "  \"sections\": {\n" ++
  "    \"generic\": {\"file\": \"generic.bin\", \"size\": " ++ toString secs.generic.size ++ "},\n" ++
  "    \"operation\": {\"file\": \"operation.bin\", \"size\": " ++ toString secs.operation.size ++ "},\n" ++
  "    \"procedure\": {\"file\": \"procedure.bin\", \"size\": " ++ toString secs.procedure.size ++ "},\n" ++
  "    \"text_properties\": {\"file\": \"text-properties.bin\", \"size\": " ++ toString secs.textProperties.size ++ "}\n" ++
  "  },\n" ++
  s!"  \"out_shape\": [{outS.n}, {outS.c}, {outS.h}, {outS.w}],\n" ++
  "  \"copies\": [\n" ++
  String.intercalate ",\n" copyItems ++ "\n" ++
  "  ],\n" ++
  s!"  \"l2_peak_bytes\": {compiled.l2PeakBytes},\n" ++
  s!"  \"task_count\": {info.taskStream.taskCount},\n" ++
  s!"  \"record_count\": {info.records},\n" ++
  s!"  \"relocation_count\": {info.relocCount},\n" ++
  s!"  \"has_kernel\": {if info.kernelSection.isSome then "true" else "false"},\n" ++
  s!"  \"has_scratch\": {if info.scratchSection.isSome then "true" else "false"},\n" ++
  "  \"surfaces\": [\n" ++
  renderSurfacesJson info.surfaces ++ "\n" ++
  "  ]\n" ++
  "}\n"

private def renderConv3x3ManifestJson (compiled : CompiledConv3x3) (weightFormat : String := "fp16") : String :=
  let info := compiled.info
  let secs := compiled.sections
  let sh := compiled.shape
  let t := compiled.tiling
  let lutLine := if compiled.lutBytes.isSome then "  \"lut_file\": \"lut.bin\",\n" else ""
  let wfLine := if weightFormat != "fp16" then s!"  \"weight_format\": \"{escapeJsonString weightFormat}\",\n" else ""
  "{\n" ++
  "  \"op\": \"conv3x3\",\n" ++
  wfLine ++
  s!"  \"mode\": \"{sh.mode.name}\",\n" ++
  s!"  \"effective_mode\": \"{sh.effMode.name}\",\n" ++
  s!"  \"target\": \"{compiled.target.name}\",\n" ++
  "  \"hwx\": \"program.hwx\",\n" ++
  "  \"sections\": {\n" ++
  "    \"generic\": {\"file\": \"generic.bin\", \"size\": " ++ toString secs.generic.size ++ "},\n" ++
  "    \"operation\": {\"file\": \"operation.bin\", \"size\": " ++ toString secs.operation.size ++ "},\n" ++
  "    \"procedure\": {\"file\": \"procedure.bin\", \"size\": " ++ toString secs.procedure.size ++ "},\n" ++
  "    \"text_properties\": {\"file\": \"text-properties.bin\", \"size\": " ++ toString secs.textProperties.size ++ "}\n" ++
  "  },\n" ++
  "  \"shape\": {\n" ++
  s!"    \"h\": {sh.h},\n" ++
  s!"    \"w\": {sh.w},\n" ++
  s!"    \"cin\": {sh.cin},\n" ++
  s!"    \"cout\": {sh.cout},\n" ++
  s!"    \"ho\": {sh.outH},\n" ++
  s!"    \"wo\": {sh.outW},\n" ++
  s!"    \"stride\": {sh.stride},\n" ++
  s!"    \"pad\": \"{sh.pad.name}\",\n" ++
  s!"    \"pt\": {sh.padTop},\n" ++
  s!"    \"pl\": {sh.padLeft},\n" ++
  s!"    \"crop_h\": {sh.cropH},\n" ++
  s!"    \"crop_w\": {sh.cropW}\n" ++
  "  },\n" ++
  s!"  \"sparse\": {if compiled.sparse then "true" else "false"},\n" ++
  s!"  \"wide3x3\": {if compiled.wide3x3 then "true" else "false"},\n" ++
  s!"  \"has_bias\": {if compiled.hasBias then "true" else "false"},\n" ++
  s!"  \"relu\": {if compiled.relu then "true" else "false"},\n" ++
  s!"  \"activation\": \"{escapeJsonString compiled.activation.name}\",\n" ++
  lutLine ++
  "  \"tiling\": {\n" ++
  s!"    \"log_ne\": {t.logNE},\n" ++
  s!"    \"f\": {t.f},\n" ++
  s!"    \"fat\": {t.fat},\n" ++
  s!"    \"ocg\": {effOcg sh t},\n" ++
  s!"    \"wu\": {t.wu},\n" ++
  s!"    \"sh_pref\": {t.shPref},\n" ++
  s!"    \"sh_max\": {t.shMax},\n" ++
  s!"    \"tile_height\": {t.tileHeight},\n" ++
  s!"    \"src_skew\": {t.srcSkew},\n" ++
  s!"    \"res_skew\": {t.resSkew}\n" ++
  "  },\n" ++
  s!"  \"direct\": {if compiled.direct then "true" else "false"},\n" ++
  s!"  \"l2_use_bytes\": {compiled.l2UseBytes},\n" ++
  s!"  \"kernel_bytes\": {compiled.kernelBytes},\n" ++
  s!"  \"task_count\": {info.taskStream.taskCount},\n" ++
  s!"  \"record_count\": {info.records},\n" ++
  s!"  \"relocation_count\": {info.relocCount},\n" ++
  s!"  \"has_kernel\": {if info.kernelSection.isSome then "true" else "false"},\n" ++
  s!"  \"has_scratch\": {if info.scratchSection.isSome then "true" else "false"},\n" ++
  "  \"surfaces\": [\n" ++
  renderSurfacesJson info.surfaces ++ "\n" ++
  "  ]\n" ++
  "}\n"

private def renderPeManifestJson (compiled : CompiledPe) (peOpLabel : String) : String :=
  let info := compiled.info
  let secs := compiled.sections
  let t := compiled.tiling
  let m := peTileM compiled.op.isTwoSource t.k t.tileHeight
  "{\n" ++
  "  \"op\": \"pe\",\n" ++
  s!"  \"pe_op\": \"{escapeJsonString peOpLabel}\",\n" ++
  s!"  \"target\": \"{compiled.target.name}\",\n" ++
  "  \"hwx\": \"program.hwx\",\n" ++
  "  \"sections\": {\n" ++
  "    \"generic\": {\"file\": \"generic.bin\", \"size\": " ++ toString secs.generic.size ++ "},\n" ++
  "    \"operation\": {\"file\": \"operation.bin\", \"size\": " ++ toString secs.operation.size ++ "},\n" ++
  "    \"procedure\": {\"file\": \"procedure.bin\", \"size\": " ++ toString secs.procedure.size ++ "},\n" ++
  "    \"text_properties\": {\"file\": \"text-properties.bin\", \"size\": " ++ toString secs.textProperties.size ++ "}\n" ++
  "  },\n" ++
  "  \"tiling\": {\n" ++
  s!"    \"k\": {t.k},\n" ++
  s!"    \"m\": {m},\n" ++
  s!"    \"strip_width\": {peStripWidth t.k},\n" ++
  s!"    \"tile_height\": {t.tileHeight}\n" ++
  "  },\n" ++
  s!"  \"direct\": {if compiled.direct then "true" else "false"},\n" ++
  s!"  \"l2_use_bytes\": {compiled.l2UseBytes},\n" ++
  s!"  \"task_count\": {info.taskStream.taskCount},\n" ++
  s!"  \"record_count\": {info.records},\n" ++
  s!"  \"relocation_count\": {info.relocCount},\n" ++
  s!"  \"has_kernel\": {if info.kernelSection.isSome then "true" else "false"},\n" ++
  s!"  \"has_scratch\": {if info.scratchSection.isSome then "true" else "false"},\n" ++
  "  \"surfaces\": [\n" ++
  renderSurfacesJson info.surfaces ++ "\n" ++
  "  ]\n" ++
  "}\n"

private def renderNeActManifestJson (compiled : CompiledNeAct) : String :=
  let info := compiled.info
  let secs := compiled.sections
  let t := compiled.tiling
  let lutExtra := match compiled.lutPayload with
    | none => ""
    | some lp =>
      let shiftLine := match lp.inputShiftBits with
        | some s => s!"  \"input_shift_f16\": {s.toNat},\n"
        | none => ""
      "  \"lut_file\": \"lut.bin\",\n" ++
      s!"  \"pre_scale_f16\": {lp.preScaleBits.toNat},\n" ++
      shiftLine
  "{\n" ++
  "  \"op\": \"ne_act\",\n" ++
  s!"  \"act\": \"{escapeJsonString compiled.act.name}\",\n" ++
  s!"  \"target\": \"{compiled.target.name}\",\n" ++
  "  \"hwx\": \"program.hwx\",\n" ++
  "  \"sections\": {\n" ++
  "    \"generic\": {\"file\": \"generic.bin\", \"size\": " ++ toString secs.generic.size ++ "},\n" ++
  "    \"operation\": {\"file\": \"operation.bin\", \"size\": " ++ toString secs.operation.size ++ "},\n" ++
  "    \"procedure\": {\"file\": \"procedure.bin\", \"size\": " ++ toString secs.procedure.size ++ "},\n" ++
  "    \"text_properties\": {\"file\": \"text-properties.bin\", \"size\": " ++ toString secs.textProperties.size ++ "}\n" ++
  "  },\n" ++
  lutExtra ++
  "  \"tiling\": {\n" ++
  s!"    \"log_ne\": {t.logNE},\n" ++
  s!"    \"f\": {t.f},\n" ++
  s!"    \"fat\": {t.fat},\n" ++
  s!"    \"wu\": {t.wu},\n" ++
  s!"    \"sh_pref\": {t.shPref},\n" ++
  s!"    \"sh_max\": {t.shMax},\n" ++
  s!"    \"tile_height\": {t.tileHeight}\n" ++
  "  },\n" ++
  s!"  \"direct\": {if compiled.direct then "true" else "false"},\n" ++
  s!"  \"l2_use_bytes\": {compiled.l2UseBytes},\n" ++
  s!"  \"task_count\": {info.taskStream.taskCount},\n" ++
  s!"  \"record_count\": {info.records},\n" ++
  s!"  \"relocation_count\": {info.relocCount},\n" ++
  s!"  \"has_kernel\": {if info.kernelSection.isSome then "true" else "false"},\n" ++
  s!"  \"has_scratch\": {if info.scratchSection.isSome then "true" else "false"},\n" ++
  "  \"surfaces\": [\n" ++
  renderSurfacesJson info.surfaces ++ "\n" ++
  "  ]\n" ++
  "}\n"

private def cmdValidate (args : List String) : IO UInt32 := do
  let mut hwxPath : Option String := none
  let mut sectionsDir : Option String := none
  let mut checkTasks := false
  let mut rest := args
  while !rest.isEmpty do
    match rest with
    | "--hwx" :: tl =>
      match requireArgValue "--hwx" tl with
      | .ok (v, tl') => hwxPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--sections" :: tl =>
      match requireArgValue "--sections" tl with
      | .ok (v, tl') => sectionsDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--check-tasks" :: tl =>
      checkTasks := true; rest := tl
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for validate"
      IO.eprintln usage
      return 1
    | [] => break

  let some path := hwxPath
    | do IO.eprintln "error: --hwx <path> is required"; return 1

  let hwxBytes ← try
    IO.FS.readBinFile path
  catch e =>
    IO.eprintln s!"error: failed to read '{path}': {e}"
    return 1

  let info ← match validateHwx hwxBytes with
    | .ok i => pure i
    | .error e =>
      IO.eprintln s!"HWX validation failed: {e}"
      return 1

  let summary ← match verifyHwxTasksSummary hwxBytes info (requireKnownTask := checkTasks) with
    | .ok s => pure s
    | .error e =>
      IO.eprintln s!"task-legality validation failed: {e}"
      return 1

  let mut sectionSummary : Option String := none
  if let some sdir := sectionsDir then
    let dir := System.FilePath.mk sdir
    let readSec := fun (fname : String) => IO.FS.readBinFile (dir / fname)
    let secs ← try
      let g ← readSec "generic.bin"
      let o ← readSec "operation.bin"
      let p ← readSec "procedure.bin"
      let t ← readSec "text-properties.bin"
      pure ({ generic := g, operation := o, procedure := p, textProperties := t } : LoadSections)
    catch e =>
      IO.eprintln s!"error: failed to read sections from '{sdir}': {e}"
      return 1
    match validateSectionsWithInfo info secs with
    | .ok () =>
      sectionSummary := some (
        s!"sections: generic={secs.generic.size}B operation={secs.operation.size}B " ++
        s!"procedure={secs.procedure.size}B text-properties={secs.textProperties.size}B")
    | .error e =>
      IO.eprintln s!"LOAD section validation failed: {e}"
      return 1

  let surfDesc := info.surfaces.toList.map fun s =>
    s!"{s.shortName}({s.direction.name} {s.n}x{s.channels}x{s.height}x{s.width} span={s.span}B)"
  IO.println s!"status: ok"
  IO.println s!"target: {info.target.name}"
  IO.println s!"surfaces ({info.surfaces.size}): {String.intercalate ", " surfDesc}"
  IO.println s!"tasks: {info.taskStream.taskCount} (records={info.records}, text={info.textSection.size}B, relocs={info.relocCount})"
  if summary.total > 0 then
    IO.println s!"verified_tasks: {summary.total} (conv={summary.neConv}, ne_act={summary.neAct}, pe={summary.peOp}, pe_pool={summary.pePool}, pe_reduce={summary.peReduce}, ne_pool={summary.nePool}, ne_chan_reduce={summary.neChannelReduce}, ne_view={summary.neView}, ne_matmul={summary.neMatmul})"
    if summary.neConv > 0 then
      IO.println s!"verified_conv_tasks: {summary.neConv}"
  IO.println s!"bar_refs: {formatBarSlots info.taskStream.barSlots}"
  IO.println s!"kernel: {match info.kernelSection with | some k => s!"{k.size}B" | none => "none"}, scratch: {match info.scratchSection with | some sc => s!"{sc.size}B" | none => "none"}"
  if let some s := sectionSummary then
    IO.println s
  return 0

private def cmdSections (args : List String) : IO UInt32 := do
  let mut hwxPath : Option String := none
  let mut outDir : Option String := none
  let mut rest := args
  while !rest.isEmpty do
    match rest with
    | "--hwx" :: tl =>
      match requireArgValue "--hwx" tl with
      | .ok (v, tl') => hwxPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for sections"
      IO.eprintln usage
      return 1
    | [] => break

  let some path := hwxPath
    | do IO.eprintln "error: --hwx <path> is required"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let hwxBytes ← try
    IO.FS.readBinFile path
  catch e =>
    IO.eprintln s!"error: failed to read '{path}': {e}"
    return 1

  let secs ← match buildSections hwxBytes with
    | .ok s => pure s
    | .error e =>
      IO.eprintln s!"failed to build LOAD sections: {e}"
      return 1

  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "generic.bin") secs.generic
    IO.FS.writeBinFile (dir / "operation.bin") secs.operation
    IO.FS.writeBinFile (dir / "procedure.bin") secs.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") secs.textProperties
  catch e =>
    IO.eprintln s!"error: failed to write sections to '{dirStr}': {e}"
    return 1

  IO.println s!"wrote 4 sections to {dirStr} (generic={secs.generic.size}B, operation={secs.operation.size}B, procedure={secs.procedure.size}B, text-properties={secs.textProperties.size}B)"
  return 0

private def cmdEmitDemo (args : List String) : IO UInt32 := do
  let mut targetStr : Option String := none
  let mut outDir : Option String := none
  let mut withKernel := false
  let mut withScratch := false
  let mut rest := args
  while !rest.isEmpty do
    match rest with
    | "--target" :: tl =>
      match requireArgValue "--target" tl with
      | .ok (v, tl') => targetStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--with-kernel" :: tl =>
      withKernel := true; rest := tl
    | "--with-scratch" :: tl =>
      withScratch := true; rest := tl
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for emit-demo"
      IO.eprintln usage
      return 1
    | [] => break

  let some tStr := targetStr
    | do IO.eprintln "error: --target <h16|h17> is required"; return 1
  let some target := Target.parse? tStr
    | do IO.eprintln s!"error: unknown target '{tStr}' (expected h16 or h17)"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let prog ← match buildDemoProgram target withKernel withScratch with
    | .ok p => pure p
    | .error e => IO.eprintln s!"error: failed to build demo program: {e}"; return 1

  let hwxBytes ← match emitHwx prog with
    | .ok b => pure b
    | .error e => IO.eprintln s!"error: failed to emit HWX: {e}"; return 1

  let info ← match validateHwx hwxBytes with
    | .ok i => pure i
    | .error e => IO.eprintln s!"error: emitted HWX failed self-validation: {e}"; return 1

  let secs ← match buildSectionsFromInfo info with
    | .ok s => pure s
    | .error e => IO.eprintln s!"error: failed to build sections: {e}"; return 1

  match validateSectionsWithInfo info secs with
  | .ok () => pure ()
  | .error e => IO.eprintln s!"error: built sections failed self-validation: {e}"; return 1

  let manifest := renderManifestJson info secs
  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "program.hwx") hwxBytes
    IO.FS.writeBinFile (dir / "generic.bin") secs.generic
    IO.FS.writeBinFile (dir / "operation.bin") secs.operation
    IO.FS.writeBinFile (dir / "procedure.bin") secs.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") secs.textProperties
    IO.FS.writeFile (dir / "manifest.json") manifest
  catch e =>
    IO.eprintln s!"error: failed to write demo files to '{dirStr}': {e}"
    return 1

  IO.println s!"emitted {target.name} demo container ({hwxBytes.size}B) and 4 sections to {dirStr}"
  return 0

private def cmdCompileConv1x1 (args : List String) : IO UInt32 := do
  let mut targetStr : Option String := none
  let mut hStr : Option String := none
  let mut wStr : Option String := none
  let mut cinStr : Option String := none
  let mut coutStr : Option String := none
  let mut weightsPath : Option String := none
  let mut biasPath : Option String := none
  let mut weightFormatStr : String := "fp16"
  let mut scalesPath : Option String := none
  let mut zeroPointsPath : Option String := none
  let mut palettePath : Option String := none
  let mut relu := false
  let mut actStr : Option String := none
  let mut actParams : ActParamFlags := {}
  let mut outDir : Option String := none
  let mut rest := args

  while !rest.isEmpty do
    match rest with
    | "--target" :: tl =>
      match requireArgValue "--target" tl with
      | .ok (v, tl') => targetStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--h" :: tl =>
      match requireArgValue "--h" tl with
      | .ok (v, tl') => hStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--w" :: tl =>
      match requireArgValue "--w" tl with
      | .ok (v, tl') => wStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--cin" :: tl =>
      match requireArgValue "--cin" tl with
      | .ok (v, tl') => cinStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--cout" :: tl =>
      match requireArgValue "--cout" tl with
      | .ok (v, tl') => coutStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--weights" :: tl =>
      match requireArgValue "--weights" tl with
      | .ok (v, tl') => weightsPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--bias" :: tl =>
      match requireArgValue "--bias" tl with
      | .ok (v, tl') => biasPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--weight-format" :: tl =>
      match requireArgValue "--weight-format" tl with
      | .ok (v, tl') => weightFormatStr := v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--scales" :: tl =>
      match requireArgValue "--scales" tl with
      | .ok (v, tl') => scalesPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--zero-points" :: tl =>
      match requireArgValue "--zero-points" tl with
      | .ok (v, tl') => zeroPointsPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--palette" :: tl =>
      match requireArgValue "--palette" tl with
      | .ok (v, tl') => palettePath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--relu" :: tl =>
      relu := true; rest := tl
    | "--act" :: tl =>
      match requireArgValue "--act" tl with
      | .ok (v, tl') => actStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-alpha" :: tl =>
      match requireArgValue "--act-alpha" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-alpha" v with
        | .ok f => actParams := { actParams with alpha := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-beta" :: tl =>
      match requireArgValue "--act-beta" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-beta" v with
        | .ok f => actParams := { actParams with beta := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-lo" :: tl =>
      match requireArgValue "--act-lo" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-lo" v with
        | .ok f => actParams := { actParams with lo := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-hi" :: tl =>
      match requireArgValue "--act-hi" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-hi" v with
        | .ok f => actParams := { actParams with hi := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-eps" :: tl =>
      match requireArgValue "--act-eps" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-eps" v with
        | .ok f => actParams := { actParams with eps := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-scalar" :: tl =>
      match requireArgValue "--act-scalar" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-scalar" v with
        | .ok f => actParams := { actParams with scalar := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for compile-conv1x1"
      IO.eprintln usage
      return 1
    | [] => break

  let some tStr := targetStr
    | do IO.eprintln "error: --target <h16|h17> is required"; return 1
  let some target := Target.parse? tStr
    | do IO.eprintln s!"error: unknown target '{tStr}' (expected h16 or h17)"; return 1
  let some hRaw := hStr
    | do IO.eprintln "error: --h <H> is required"; return 1
  let some wRaw := wStr
    | do IO.eprintln "error: --w <W> is required"; return 1
  let some cinRaw := cinStr
    | do IO.eprintln "error: --cin <Cin> is required"; return 1
  let some coutRaw := coutStr
    | do IO.eprintln "error: --cout <Cout> is required"; return 1
  let some wPath := weightsPath
    | do IO.eprintln "error: --weights <path> is required"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let act ← match parseConvAct relu actStr actParams with
    | .ok a => pure a
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let shape ← match (do
      let h ← parsePosNat "--h" hRaw
      let w ← parsePosNat "--w" wRaw
      let cin ← parsePosNat "--cin" cinRaw
      let cout ← parsePosNat "--cout" coutRaw
      pure ({ h := h, w := w, cin := cin, cout := cout } : Conv1x1Shape)) with
    | .ok sh => pure sh
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let weightsBytes ← try
    IO.FS.readBinFile wPath
  catch e =>
    IO.eprintln s!"error: failed to read weights '{wPath}': {e}"
    return 1

  let biasBytes? ← match biasPath with
    | none => pure none
    | some bPath =>
      try
        let bs ← IO.FS.readBinFile bPath
        pure (some bs)
      catch e =>
        IO.eprintln s!"error: failed to read bias '{bPath}': {e}"
        return 1

  let quantSpec? ← match ← loadQuantWeightSpecIO weightFormatStr weightsBytes
      (scalesPath.map System.FilePath.mk) (zeroPointsPath.map System.FilePath.mk)
      (palettePath.map System.FilePath.mk) "compile-conv1x1" with
    | .ok q? => pure q?
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let (compiled, wfName) ← match quantSpec? with
    | none =>
      match compileConv1x1 target shape weightsBytes biasBytes? false none act with
      | .ok c => pure (c, "fp16")
      | .error e => IO.eprintln s!"error: compile-conv1x1 failed: {e}"; return 1
    | some q =>
      match compileConv1x1Quant target shape q biasBytes? false none act with
      | .ok c => pure (compiledSplitToConv1x1 c, q.format.name)
      | .error e => IO.eprintln s!"error: compile-conv1x1 failed: {e}"; return 1

  let manifest := renderConv1x1ManifestJson compiled "conv1x1" wfName
  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "program.hwx") compiled.hwx
    IO.FS.writeBinFile (dir / "generic.bin") compiled.sections.generic
    IO.FS.writeBinFile (dir / "operation.bin") compiled.sections.operation
    IO.FS.writeBinFile (dir / "procedure.bin") compiled.sections.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") compiled.sections.textProperties
    if let some lutBs := compiled.lutBytes then
      IO.FS.writeBinFile (dir / "lut.bin") lutBs
    IO.FS.writeFile (dir / "manifest.json") manifest
  catch e =>
    IO.eprintln s!"error: failed to write compiled outputs to '{dirStr}': {e}"
    return 1

  let t := compiled.tiling
  let splitSuffix :=
    if compiled.slices.size > 1 then s!" slices={formatNatListJson compiled.slices}" else ""
  IO.println (
    s!"compiled {target.name} conv1x1 ({shape.h}x{shape.w}, {shape.cin}->{shape.cout}, " ++
    s!"bias={compiled.hasBias}, act={compiled.activation.name}) to {dirStr}: " ++
    s!"hwx={compiled.hwx.size}B kernel={compiled.kernelBytes}B l2={compiled.l2UseBytes}B " ++
    s!"direct={compiled.direct}{splitSuffix} tiling=(logNE={t.logNE},F={t.f},fat={t.fat},ocg={t.ocg},WU={t.wu},sh={t.shPref}..{t.shMax},TH={t.tileHeight})"
  )
  return 0

private def cmdCompileLinear (args : List String) : IO UInt32 := do
  let mut targetStr : Option String := none
  let mut mStr : Option String := none
  let mut kStr : Option String := none
  let mut nStr : Option String := none
  let mut batchStr : Option String := none
  let mut weightsPath : Option String := none
  let mut biasPath : Option String := none
  let mut weightFormatStr : String := "fp16"
  let mut scalesPath : Option String := none
  let mut zeroPointsPath : Option String := none
  let mut palettePath : Option String := none
  let mut relu := false
  let mut actStr : Option String := none
  let mut actParams : ActParamFlags := {}
  let mut outDir : Option String := none
  let mut rest := args

  while !rest.isEmpty do
    match rest with
    | "--target" :: tl =>
      match requireArgValue "--target" tl with
      | .ok (v, tl') => targetStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--m" :: tl =>
      match requireArgValue "--m" tl with
      | .ok (v, tl') => mStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--k" :: tl =>
      match requireArgValue "--k" tl with
      | .ok (v, tl') => kStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--n" :: tl =>
      match requireArgValue "--n" tl with
      | .ok (v, tl') => nStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--batch" :: tl =>
      match requireArgValue "--batch" tl with
      | .ok (v, tl') => batchStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--weights" :: tl =>
      match requireArgValue "--weights" tl with
      | .ok (v, tl') => weightsPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--bias" :: tl =>
      match requireArgValue "--bias" tl with
      | .ok (v, tl') => biasPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--weight-format" :: tl =>
      match requireArgValue "--weight-format" tl with
      | .ok (v, tl') => weightFormatStr := v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--scales" :: tl =>
      match requireArgValue "--scales" tl with
      | .ok (v, tl') => scalesPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--zero-points" :: tl =>
      match requireArgValue "--zero-points" tl with
      | .ok (v, tl') => zeroPointsPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--palette" :: tl =>
      match requireArgValue "--palette" tl with
      | .ok (v, tl') => palettePath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--relu" :: tl =>
      relu := true; rest := tl
    | "--act" :: tl =>
      match requireArgValue "--act" tl with
      | .ok (v, tl') => actStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-alpha" :: tl =>
      match requireArgValue "--act-alpha" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-alpha" v with
        | .ok f => actParams := { actParams with alpha := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-beta" :: tl =>
      match requireArgValue "--act-beta" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-beta" v with
        | .ok f => actParams := { actParams with beta := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-lo" :: tl =>
      match requireArgValue "--act-lo" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-lo" v with
        | .ok f => actParams := { actParams with lo := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-hi" :: tl =>
      match requireArgValue "--act-hi" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-hi" v with
        | .ok f => actParams := { actParams with hi := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-eps" :: tl =>
      match requireArgValue "--act-eps" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-eps" v with
        | .ok f => actParams := { actParams with eps := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-scalar" :: tl =>
      match requireArgValue "--act-scalar" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-scalar" v with
        | .ok f => actParams := { actParams with scalar := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for compile-linear"
      IO.eprintln usage
      return 1
    | [] => break

  let some tStr := targetStr
    | do IO.eprintln "error: --target <h16|h17> is required"; return 1
  let some target := Target.parse? tStr
    | do IO.eprintln s!"error: unknown target '{tStr}' (expected h16 or h17)"; return 1
  let some mRaw := mStr
    | do IO.eprintln "error: --m <M> is required"; return 1
  let some kRaw := kStr
    | do IO.eprintln "error: --k <K> is required"; return 1
  let some nRaw := nStr
    | do IO.eprintln "error: --n <N> is required"; return 1
  let some wPath := weightsPath
    | do IO.eprintln "error: --weights <path> is required"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let act ← match parseConvAct relu actStr actParams with
    | .ok a => pure a
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let (m, k, n, batch) ← match (do
      let m ← parsePosNat "--m" mRaw
      let k ← parsePosNat "--k" kRaw
      let n ← parsePosNat "--n" nRaw
      let b ← match batchStr with | some s => parsePosNat "--batch" s | none => pure 1
      pure (m, k, n, b)) with
    | .ok v => pure v
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let weightsBytes ← try
    IO.FS.readBinFile wPath
  catch e =>
    IO.eprintln s!"error: failed to read weights '{wPath}': {e}"
    return 1

  let biasBytes? ← match biasPath with
    | none => pure none
    | some bPath =>
      try some <$> IO.FS.readBinFile bPath
      catch e =>
        IO.eprintln s!"error: failed to read bias '{bPath}': {e}"
        return 1

  let quantSpec? ← match ← loadQuantWeightSpecIO weightFormatStr weightsBytes
      (scalesPath.map System.FilePath.mk) (zeroPointsPath.map System.FilePath.mk)
      (palettePath.map System.FilePath.mk) "compile-linear" with
    | .ok q? => pure q?
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let (compiled, wfName) ← match quantSpec? with
    | none =>
      match compileLinear target m k n weightsBytes biasBytes? batch false act with
      | .ok c => pure (c, "fp16")
      | .error e => IO.eprintln s!"error: compile-linear failed: {e}"; return 1
    | some q =>
      match compileLinearQuant target m k n q biasBytes? batch false act with
      | .ok c => pure (compiledSplitToConv1x1 c, q.format.name)
      | .error e => IO.eprintln s!"error: compile-linear failed: {e}"; return 1

  let manifest := renderConv1x1ManifestJson compiled "linear" wfName
  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "program.hwx") compiled.hwx
    IO.FS.writeBinFile (dir / "generic.bin") compiled.sections.generic
    IO.FS.writeBinFile (dir / "operation.bin") compiled.sections.operation
    IO.FS.writeBinFile (dir / "procedure.bin") compiled.sections.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") compiled.sections.textProperties
    if let some lutBs := compiled.lutBytes then
      IO.FS.writeBinFile (dir / "lut.bin") lutBs
    IO.FS.writeFile (dir / "manifest.json") manifest
  catch e =>
    IO.eprintln s!"error: failed to write compiled outputs to '{dirStr}': {e}"
    return 1

  IO.println (
    s!"compiled {target.name} linear (batch={batch}, M={m}, K={k}, N={n}, " ++
    s!"slices={formatNatListJson compiled.slices}, bias={compiled.hasBias}, act={compiled.activation.name}) to {dirStr}: " ++
    s!"hwx={compiled.hwx.size}B kernel={compiled.kernelBytes}B l2={compiled.l2UseBytes}B"
  )
  return 0

private def cmdCompileConv3x3 (args : List String) : IO UInt32 := do
  let mut targetStr : Option String := none
  let mut modeStr : Option String := none
  let mut hStr : Option String := none
  let mut wStr : Option String := none
  let mut cinStr : Option String := none
  let mut coutStr : Option String := none
  let mut strideStr : Option String := none
  let mut padStr : Option String := none
  let mut weightsPath : Option String := none
  let mut biasPath : Option String := none
  let mut weightFormatStr : String := "fp16"
  let mut scalesPath : Option String := none
  let mut zeroPointsPath : Option String := none
  let mut palettePath : Option String := none
  let mut relu := false
  let mut actStr : Option String := none
  let mut actParams : ActParamFlags := {}
  let mut outDir : Option String := none
  let mut rest := args

  while !rest.isEmpty do
    match rest with
    | "--target" :: tl =>
      match requireArgValue "--target" tl with
      | .ok (v, tl') => targetStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--mode" :: tl =>
      match requireArgValue "--mode" tl with
      | .ok (v, tl') => modeStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--h" :: tl =>
      match requireArgValue "--h" tl with
      | .ok (v, tl') => hStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--w" :: tl =>
      match requireArgValue "--w" tl with
      | .ok (v, tl') => wStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--cin" :: tl =>
      match requireArgValue "--cin" tl with
      | .ok (v, tl') => cinStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--cout" :: tl =>
      match requireArgValue "--cout" tl with
      | .ok (v, tl') => coutStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--stride" :: tl =>
      match requireArgValue "--stride" tl with
      | .ok (v, tl') => strideStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--pad" :: tl =>
      match requireArgValue "--pad" tl with
      | .ok (v, tl') => padStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--weights" :: tl =>
      match requireArgValue "--weights" tl with
      | .ok (v, tl') => weightsPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--bias" :: tl =>
      match requireArgValue "--bias" tl with
      | .ok (v, tl') => biasPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--weight-format" :: tl =>
      match requireArgValue "--weight-format" tl with
      | .ok (v, tl') => weightFormatStr := v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--scales" :: tl =>
      match requireArgValue "--scales" tl with
      | .ok (v, tl') => scalesPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--zero-points" :: tl =>
      match requireArgValue "--zero-points" tl with
      | .ok (v, tl') => zeroPointsPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--palette" :: tl =>
      match requireArgValue "--palette" tl with
      | .ok (v, tl') => palettePath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--relu" :: tl =>
      relu := true; rest := tl
    | "--act" :: tl =>
      match requireArgValue "--act" tl with
      | .ok (v, tl') => actStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-alpha" :: tl =>
      match requireArgValue "--act-alpha" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-alpha" v with
        | .ok f => actParams := { actParams with alpha := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-beta" :: tl =>
      match requireArgValue "--act-beta" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-beta" v with
        | .ok f => actParams := { actParams with beta := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-lo" :: tl =>
      match requireArgValue "--act-lo" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-lo" v with
        | .ok f => actParams := { actParams with lo := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-hi" :: tl =>
      match requireArgValue "--act-hi" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-hi" v with
        | .ok f => actParams := { actParams with hi := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-eps" :: tl =>
      match requireArgValue "--act-eps" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-eps" v with
        | .ok f => actParams := { actParams with eps := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act-scalar" :: tl =>
      match requireArgValue "--act-scalar" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--act-scalar" v with
        | .ok f => actParams := { actParams with scalar := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for compile-conv3x3"
      IO.eprintln usage
      return 1
    | [] => break

  let some tStr := targetStr
    | do IO.eprintln "error: --target <h16|h17> is required"; return 1
  let some target := Target.parse? tStr
    | do IO.eprintln s!"error: unknown target '{tStr}' (expected h16 or h17)"; return 1
  let some mStr := modeStr
    | do IO.eprintln "error: --mode <dense|depthwise> is required"; return 1
  let some mode := Conv3x3Mode.parse? mStr
    | do IO.eprintln s!"error: unknown mode '{mStr}' (expected dense or depthwise)"; return 1
  let some hRaw := hStr
    | do IO.eprintln "error: --h <H> is required"; return 1
  let some wRaw := wStr
    | do IO.eprintln "error: --w <W> is required"; return 1
  let some cinRaw := cinStr
    | do IO.eprintln "error: --cin <Cin> is required"; return 1
  let some coutRaw := coutStr
    | do IO.eprintln "error: --cout <Cout> is required"; return 1
  let some sRaw := strideStr
    | do IO.eprintln "error: --stride <1|2> is required"; return 1
  let some pStr := padStr
    | do IO.eprintln "error: --pad <same|valid> is required"; return 1
  let some padMode := PadMode.parse? pStr
    | do IO.eprintln s!"error: unknown pad '{pStr}' (expected same or valid)"; return 1
  let some wPath := weightsPath
    | do IO.eprintln "error: --weights <path> is required"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let act ← match parseConvAct relu actStr actParams with
    | .ok a => pure a
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let shape ← match (do
      let h ← parsePosNat "--h" hRaw
      let w ← parsePosNat "--w" wRaw
      let cin ← parsePosNat "--cin" cinRaw
      let cout ← parsePosNat "--cout" coutRaw
      let stride ← parsePosNat "--stride" sRaw
      pure ({
        mode := mode
        h := h
        w := w
        cin := cin
        cout := cout
        stride := stride
        pad := padMode
      } : Conv3x3Shape)) with
    | .ok sh => pure sh
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let weightsBytes ← try
    IO.FS.readBinFile wPath
  catch e =>
    IO.eprintln s!"error: failed to read weights '{wPath}': {e}"
    return 1

  let biasBytes? ← match biasPath with
    | none => pure none
    | some bPath =>
      try
        let bs ← IO.FS.readBinFile bPath
        pure (some bs)
      catch e =>
        IO.eprintln s!"error: failed to read bias '{bPath}': {e}"
        return 1

  let quantSpec? ← match ← loadQuantWeightSpecIO weightFormatStr weightsBytes
      (scalesPath.map System.FilePath.mk) (zeroPointsPath.map System.FilePath.mk)
      (palettePath.map System.FilePath.mk) "compile-conv3x3" with
    | .ok q? => pure q?
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let (compiled, wfName) ← match quantSpec? with
    | none =>
      match compileConv3x3 target shape weightsBytes biasBytes? false none act with
      | .ok c => pure (c, "fp16")
      | .error e => IO.eprintln s!"error: compile-conv3x3 failed: {e}"; return 1
    | some q =>
      match compileConv3x3DenseQuant target shape q biasBytes? false none act with
      | .ok c => pure (c, q.format.name)
      | .error e => IO.eprintln s!"error: compile-conv3x3 failed: {e}"; return 1

  let manifest := renderConv3x3ManifestJson compiled wfName
  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "program.hwx") compiled.hwx
    IO.FS.writeBinFile (dir / "generic.bin") compiled.sections.generic
    IO.FS.writeBinFile (dir / "operation.bin") compiled.sections.operation
    IO.FS.writeBinFile (dir / "procedure.bin") compiled.sections.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") compiled.sections.textProperties
    if let some lutBs := compiled.lutBytes then
      IO.FS.writeBinFile (dir / "lut.bin") lutBs
    IO.FS.writeFile (dir / "manifest.json") manifest
  catch e =>
    IO.eprintln s!"error: failed to write compiled outputs to '{dirStr}': {e}"
    return 1

  let t := compiled.tiling
  IO.println (
    s!"compiled {target.name} conv3x3 {shape.mode.name} ({shape.h}x{shape.w}->{shape.outH}x{shape.outW}, " ++
    s!"{shape.cin}->{shape.cout}, s={shape.stride}, pad={shape.pad.name}, sparse={compiled.sparse}, " ++
    s!"wide3x3={compiled.wide3x3}, bias={compiled.hasBias}, act={compiled.activation.name}) to {dirStr}: " ++
    s!"hwx={compiled.hwx.size}B kernel={compiled.kernelBytes}B l2={compiled.l2UseBytes}B " ++
    s!"direct={compiled.direct} tiling=(logNE={t.logNE},F={t.f},fat={t.fat},ocg={effOcg shape t},WU={t.wu},sh={t.shPref}..{t.shMax},TH={t.tileHeight})"
  )
  return 0

private def cmdCompilePe (args : List String) : IO UInt32 := do
  let mut targetStr : Option String := none
  let mut opStr : Option String := none
  let mut hStr : Option String := none
  let mut wStr : Option String := none
  let mut cStr : Option String := none
  let mut zhStr : Option String := none
  let mut zwStr : Option String := none
  let mut zcStr : Option String := none
  let mut alphaStr : Option String := none
  let mut betaStr : Option String := none
  let mut inScaleStr : Option String := none
  let mut biasStr : Option String := none
  let mut scaleStr : Option String := none
  let mut scalarStr : Option String := none
  let mut outDir : Option String := none
  let mut rest := args

  while !rest.isEmpty do
    match rest with
    | "--target" :: tl =>
      match requireArgValue "--target" tl with
      | .ok (v, tl') => targetStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--op" :: tl =>
      match requireArgValue "--op" tl with
      | .ok (v, tl') => opStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--h" :: tl =>
      match requireArgValue "--h" tl with
      | .ok (v, tl') => hStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--w" :: tl =>
      match requireArgValue "--w" tl with
      | .ok (v, tl') => wStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--c" :: tl =>
      match requireArgValue "--c" tl with
      | .ok (v, tl') => cStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--z-h" :: tl =>
      match requireArgValue "--z-h" tl with
      | .ok (v, tl') => zhStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--z-w" :: tl =>
      match requireArgValue "--z-w" tl with
      | .ok (v, tl') => zwStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--z-c" :: tl =>
      match requireArgValue "--z-c" tl with
      | .ok (v, tl') => zcStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--alpha" :: tl =>
      match requireArgValue "--alpha" tl with
      | .ok (v, tl') => alphaStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--beta" :: tl =>
      match requireArgValue "--beta" tl with
      | .ok (v, tl') => betaStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--in-scale" :: tl =>
      match requireArgValue "--in-scale" tl with
      | .ok (v, tl') => inScaleStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--bias" :: tl =>
      match requireArgValue "--bias" tl with
      | .ok (v, tl') => biasStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--scale" :: tl =>
      match requireArgValue "--scale" tl with
      | .ok (v, tl') => scaleStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--scalar" :: tl =>
      match requireArgValue "--scalar" tl with
      | .ok (v, tl') => scalarStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for compile-pe"
      IO.eprintln usage
      return 1
    | [] => break

  let some tStr := targetStr
    | do IO.eprintln "error: --target <h16|h17> is required"; return 1
  let some target := Target.parse? tStr
    | do IO.eprintln s!"error: unknown target '{tStr}' (expected h16 or h17)"; return 1
  let some opRaw := opStr
    | do IO.eprintln "error: --op <name> is required"; return 1
  let some hRaw := hStr
    | do IO.eprintln "error: --h <H> is required"; return 1
  let some wRaw := wStr
    | do IO.eprintln "error: --w <W> is required"; return 1
  let some cRaw := cStr
    | do IO.eprintln "error: --c <C> is required"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let dims ← match (do
      let h ← parsePosNat "--h" hRaw
      let w ← parsePosNat "--w" wRaw
      let c ← parsePosNat "--c" cRaw
      pure ({ c := c, h := h, w := w } : TensorDims)) with
    | .ok d => pure d
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let zDims ← match (do
      let zh ← match zhStr with | some s => parsePosNat "--z-h" s | none => pure dims.h
      let zw ← match zwStr with | some s => parsePosNat "--z-w" s | none => pure dims.w
      let zc ← match zcStr with | some s => parsePosNat "--z-c" s | none => pure dims.c
      pure ({ c := zc, h := zh, w := zw } : TensorDims)) with
    | .ok d => pure d
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let opLow := opRaw.toLower
  let peOp ← match (do
      match opLow with
      | "abs" => pure PeOp.abs
      | "square" => pure PeOp.square
      | "linear" =>
        let a ← match alphaStr with | some s => parseF16Arg "--alpha" s | none => pure f16PosOne
        let b ← match betaStr with | some s => parseF16Arg "--beta" s | none => pure f16PosZero
        pure (PeOp.linearActivation a b)
      | "chain" =>
        let ins ← match inScaleStr with | some s => some <$> parseF16Arg "--in-scale" s | none => pure none
        let b ← match biasStr with | some s => some <$> parseF16Arg "--bias" s | none => pure none
        let sc ← match scaleStr with | some s => some <$> parseF16Arg "--scale" s | none => pure none
        pure (PeOp.chain ins b sc)
      | "sadd" =>
        let some s := scalarStr | throw "sadd requires --scalar F"
        pure (PeOp.scalarAdd (← parseF16Arg "--scalar" s))
      | "ssub" =>
        let some s := scalarStr | throw "ssub requires --scalar F"
        pure (PeOp.scalarSub (← parseF16Arg "--scalar" s))
      | "srsub" =>
        let some s := scalarStr | throw "srsub requires --scalar F"
        pure (PeOp.scalarRSub (← parseF16Arg "--scalar" s))
      | "smul" =>
        let some s := scalarStr | throw "smul requires --scalar F"
        pure (PeOp.scalarMul (← parseF16Arg "--scalar" s))
      | "sdiv" =>
        let some s := scalarStr | throw "sdiv requires --scalar F"
        PeOp.scalarDiv (← parseF16Arg "--scalar" s)
      | "add" => pure (.binary .add dims zDims false)
      | "sub" => pure (.binary .sub dims zDims false)
      | "rsub" => pure (.binary .sub dims zDims true)
      | "mul" => pure (.binary .mul dims zDims false)
      | "max" => pure (.binary .max dims zDims false)
      | "min" => pure (.binary .min dims zDims false)
      | other => throw s!"unknown PE op '{other}'") with
    | .ok o => pure o
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let compiled ← match compilePe target peOp dims with
    | .ok c => pure c
    | .error e => IO.eprintln s!"error: compile-pe failed: {e}"; return 1

  let manifest := renderPeManifestJson compiled opLow
  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "program.hwx") compiled.hwx
    IO.FS.writeBinFile (dir / "generic.bin") compiled.sections.generic
    IO.FS.writeBinFile (dir / "operation.bin") compiled.sections.operation
    IO.FS.writeBinFile (dir / "procedure.bin") compiled.sections.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") compiled.sections.textProperties
    IO.FS.writeFile (dir / "manifest.json") manifest
  catch e =>
    IO.eprintln s!"error: failed to write PE outputs to '{dirStr}': {e}"
    return 1

  let t := compiled.tiling
  IO.println (
    s!"compiled {target.name} pe {opLow} (out={compiled.outDims.c}x{compiled.outDims.h}x{compiled.outDims.w}) " ++
    s!"to {dirStr}: hwx={compiled.hwx.size}B l2={compiled.l2UseBytes}B direct={compiled.direct} " ++
    s!"tiling=(k={t.k},T={peStripWidth t.k},TH={t.tileHeight})"
  )
  return 0

private def cmdCompileLut (args : List String) : IO UInt32 := do
  let mut targetStr : Option String := none
  let mut actStr : Option String := none
  let mut hStr : Option String := none
  let mut wStr : Option String := none
  let mut cStr : Option String := none
  let mut params : ActParamFlags := {}
  let mut outDir : Option String := none
  let mut rest := args

  while !rest.isEmpty do
    match rest with
    | "--target" :: tl =>
      match requireArgValue "--target" tl with
      | .ok (v, tl') => targetStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--act" :: tl =>
      match requireArgValue "--act" tl with
      | .ok (v, tl') => actStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--h" :: tl =>
      match requireArgValue "--h" tl with
      | .ok (v, tl') => hStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--w" :: tl =>
      match requireArgValue "--w" tl with
      | .ok (v, tl') => wStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--c" :: tl =>
      match requireArgValue "--c" tl with
      | .ok (v, tl') => cStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--alpha" :: tl =>
      match requireArgValue "--alpha" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--alpha" v with
        | .ok f => params := { params with alpha := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--beta" :: tl =>
      match requireArgValue "--beta" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--beta" v with
        | .ok f => params := { params with beta := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--lo" :: tl =>
      match requireArgValue "--lo" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--lo" v with
        | .ok f => params := { params with lo := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--hi" :: tl =>
      match requireArgValue "--hi" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--hi" v with
        | .ok f => params := { params with hi := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--eps" :: tl =>
      match requireArgValue "--eps" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--eps" v with
        | .ok f => params := { params with eps := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--scalar" :: tl =>
      match requireArgValue "--scalar" tl with
      | .ok (v, tl') =>
        match parseFloatArg "--scalar" v with
        | .ok f => params := { params with scalar := some f }; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for compile-lut"
      IO.eprintln usage
      return 1
    | [] => break

  let some tStr := targetStr
    | do IO.eprintln "error: --target <h16|h17> is required"; return 1
  let some target := Target.parse? tStr
    | do IO.eprintln s!"error: unknown target '{tStr}' (expected h16 or h17)"; return 1
  let some aStr := actStr
    | do IO.eprintln "error: --act <name> is required"; return 1
  let some hRaw := hStr
    | do IO.eprintln "error: --h <H> is required"; return 1
  let some wRaw := wStr
    | do IO.eprintln "error: --w <W> is required"; return 1
  let some cRaw := cStr
    | do IO.eprintln "error: --c <C> is required"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let act ← match parseNeActSpec aStr params with
    | .ok a => pure a
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let (c, h, w) ← match (do
      let h ← parsePosNat "--h" hRaw
      let w ← parsePosNat "--w" wRaw
      let c ← parsePosNat "--c" cRaw
      pure (c, h, w)) with
    | .ok dims => pure dims
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let compiled ← match compileNeAct target act c h w with
    | .ok c => pure c
    | .error e => IO.eprintln s!"error: compile-lut failed: {e}"; return 1

  let manifest := renderNeActManifestJson compiled
  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "program.hwx") compiled.hwx
    IO.FS.writeBinFile (dir / "generic.bin") compiled.sections.generic
    IO.FS.writeBinFile (dir / "operation.bin") compiled.sections.operation
    IO.FS.writeBinFile (dir / "procedure.bin") compiled.sections.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") compiled.sections.textProperties
    if let some lp := compiled.lutPayload then
      IO.FS.writeBinFile (dir / "lut.bin") lp.bytes
    IO.FS.writeFile (dir / "manifest.json") manifest
  catch e =>
    IO.eprintln s!"error: failed to write LUT outputs to '{dirStr}': {e}"
    return 1

  let t := compiled.tiling
  IO.println (
    s!"compiled {target.name} ne_act {compiled.act.name} ({c}x{h}x{w}) " ++
    s!"to {dirStr}: hwx={compiled.hwx.size}B l2={compiled.l2UseBytes}B direct={compiled.direct} " ++
    s!"tiling=(logNE={t.logNE},F={t.f},fat={t.fat},WU={t.wu},sh={t.shPref}..{t.shMax},TH={t.tileHeight})"
  )
  return 0

private def renderPoolManifestJson (compiled : CompiledPool) : String :=
  let info := compiled.info
  let secs := compiled.sections
  let sp := compiled.spec
  "{\n" ++
  s!"  \"op\": \"{sp.op.opName}\",\n" ++
  s!"  \"backend\": \"{compiled.backend.name}\",\n" ++
  s!"  \"target\": \"{sp.target.name}\",\n" ++
  "  \"hwx\": \"program.hwx\",\n" ++
  "  \"sections\": {\n" ++
  "    \"generic\": {\"file\": \"generic.bin\", \"size\": " ++ toString secs.generic.size ++ "},\n" ++
  "    \"operation\": {\"file\": \"operation.bin\", \"size\": " ++ toString secs.operation.size ++ "},\n" ++
  "    \"procedure\": {\"file\": \"procedure.bin\", \"size\": " ++ toString secs.procedure.size ++ "},\n" ++
  "    \"text_properties\": {\"file\": \"text-properties.bin\", \"size\": " ++ toString secs.textProperties.size ++ "}\n" ++
  "  },\n" ++
  "  \"pool\": {\n" ++
  s!"    \"kh\": {sp.kh}, \"kw\": {sp.kw}, \"sh\": {sp.sh}, \"sw\": {sp.sw},\n" ++
  s!"    \"pad\": \"{escapeJsonString sp.pad.name}\",\n" ++
  s!"    \"exclude_pad\": {if sp.excludePad then "true" else "false"}\n" ++
  "  },\n" ++
  s!"  \"direct\": {if compiled.direct then "true" else "false"},\n" ++
  s!"  \"l2_use_bytes\": {compiled.l2UseBytes},\n" ++
  s!"  \"task_count\": {info.taskStream.taskCount},\n" ++
  s!"  \"record_count\": {info.records},\n" ++
  s!"  \"relocation_count\": {info.relocCount},\n" ++
  s!"  \"has_kernel\": {if info.kernelSection.isSome then "true" else "false"},\n" ++
  s!"  \"has_scratch\": {if info.scratchSection.isSome then "true" else "false"},\n" ++
  "  \"surfaces\": [\n" ++
  renderSurfacesJson info.surfaces ++ "\n" ++
  "  ]\n" ++
  "}\n"

private def renderReduceManifestJson (compiled : CompiledReduce) : String :=
  let info := compiled.info
  let secs := compiled.sections
  let sp := compiled.spec
  "{\n" ++
  "  \"op\": \"reduce\",\n" ++
  s!"  \"reduce_op\": \"{sp.op.name}\",\n" ++
  s!"  \"axes\": \"{sp.axes.name}\",\n" ++
  s!"  \"backend\": \"{compiled.backend.name}\",\n" ++
  s!"  \"target\": \"{sp.target.name}\",\n" ++
  "  \"hwx\": \"program.hwx\",\n" ++
  "  \"sections\": {\n" ++
  "    \"generic\": {\"file\": \"generic.bin\", \"size\": " ++ toString secs.generic.size ++ "},\n" ++
  "    \"operation\": {\"file\": \"operation.bin\", \"size\": " ++ toString secs.operation.size ++ "},\n" ++
  "    \"procedure\": {\"file\": \"procedure.bin\", \"size\": " ++ toString secs.procedure.size ++ "},\n" ++
  "    \"text_properties\": {\"file\": \"text-properties.bin\", \"size\": " ++ toString secs.textProperties.size ++ "}\n" ++
  "  },\n" ++
  s!"  \"eps_f16\": {sp.eps.toNat},\n" ++
  s!"  \"direct\": {if compiled.direct then "true" else "false"},\n" ++
  s!"  \"l2_use_bytes\": {compiled.l2UseBytes},\n" ++
  s!"  \"task_count\": {info.taskStream.taskCount},\n" ++
  s!"  \"record_count\": {info.records},\n" ++
  s!"  \"relocation_count\": {info.relocCount},\n" ++
  s!"  \"has_kernel\": {if info.kernelSection.isSome then "true" else "false"},\n" ++
  s!"  \"has_scratch\": {if info.scratchSection.isSome then "true" else "false"},\n" ++
  "  \"surfaces\": [\n" ++
  renderSurfacesJson info.surfaces ++ "\n" ++
  "  ]\n" ++
  "}\n"

private def cmdCompilePool (args : List String) : IO UInt32 := do
  let mut targetStr : Option String := none
  let mut opStr : Option String := none
  let mut hStr : Option String := none
  let mut wStr : Option String := none
  let mut cStr : Option String := none
  let mut khStr : Option String := none
  let mut kwStr : Option String := none
  let mut shStr : Option String := none
  let mut swStr : Option String := none
  let mut padStr : String := "valid"
  let mut ptStr : Option String := none
  let mut pbStr : Option String := none
  let mut plStr : Option String := none
  let mut prStr : Option String := none
  let mut exclStr : Option String := none
  let mut outDir : Option String := none
  let mut rest := args

  while !rest.isEmpty do
    match rest with
    | "--target" :: tl =>
      match requireArgValue "--target" tl with
      | .ok (v, tl') => targetStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--op" :: tl =>
      match requireArgValue "--op" tl with
      | .ok (v, tl') => opStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--h" :: tl =>
      match requireArgValue "--h" tl with
      | .ok (v, tl') => hStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--w" :: tl =>
      match requireArgValue "--w" tl with
      | .ok (v, tl') => wStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--c" :: tl =>
      match requireArgValue "--c" tl with
      | .ok (v, tl') => cStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--kh" :: tl =>
      match requireArgValue "--kh" tl with
      | .ok (v, tl') => khStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--kw" :: tl =>
      match requireArgValue "--kw" tl with
      | .ok (v, tl') => kwStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--sh" :: tl =>
      match requireArgValue "--sh" tl with
      | .ok (v, tl') => shStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--sw" :: tl =>
      match requireArgValue "--sw" tl with
      | .ok (v, tl') => swStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--pad" :: tl =>
      match requireArgValue "--pad" tl with
      | .ok (v, tl') => padStr := v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--pt" :: tl =>
      match requireArgValue "--pt" tl with
      | .ok (v, tl') => ptStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--pb" :: tl =>
      match requireArgValue "--pb" tl with
      | .ok (v, tl') => pbStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--pl" :: tl =>
      match requireArgValue "--pl" tl with
      | .ok (v, tl') => plStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--pr" :: tl =>
      match requireArgValue "--pr" tl with
      | .ok (v, tl') => prStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--exclude-pad" :: tl =>
      match requireArgValue "--exclude-pad" tl with
      | .ok (v, tl') => exclStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for compile-pool"
      IO.eprintln usage
      return 1
    | [] => break

  let some tStr := targetStr
    | do IO.eprintln "error: --target <h16|h17> is required"; return 1
  let some target := Target.parse? tStr
    | do IO.eprintln s!"error: unknown target '{tStr}' (expected h16 or h17)"; return 1
  let some oStr := opStr
    | do IO.eprintln "error: --op <max|avg> is required"; return 1
  let some poolOp := PoolKind.parse? oStr
    | do IO.eprintln s!"error: unknown pool op '{oStr}' (expected max or avg)"; return 1
  let some hRaw := hStr
    | do IO.eprintln "error: --h <H> is required"; return 1
  let some wRaw := wStr
    | do IO.eprintln "error: --w <W> is required"; return 1
  let some cRaw := cStr
    | do IO.eprintln "error: --c <C> is required"; return 1
  let some khRaw := khStr
    | do IO.eprintln "error: --kh <KH> is required"; return 1
  let some kwRaw := kwStr
    | do IO.eprintln "error: --kw <KW> is required"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let spec ← match (do
      let h ← parsePosNat "--h" hRaw
      let w ← parsePosNat "--w" wRaw
      let c ← parsePosNat "--c" cRaw
      let kh ← parsePosNat "--kh" khRaw
      let kw ← parsePosNat "--kw" kwRaw
      let sh ← match shStr with | some s => parsePosNat "--sh" s | none => pure 1
      let sw ← match swStr with | some s => parsePosNat "--sw" s | none => pure 1
      let excl ← match exclStr with | some s => parseBoolArg "--exclude-pad" s | none => pure true
      let pad ← match padStr.toLower with
        | "valid" => pure PoolPad.valid
        | "same" => pure PoolPad.same
        | "custom" =>
          let pt ← match ptStr with | some s => parseNonNegNat "--pt" s | none => pure 0
          let pb ← match pbStr with | some s => parseNonNegNat "--pb" s | none => pure 0
          let pl ← match plStr with | some s => parseNonNegNat "--pl" s | none => pure 0
          let pr ← match prStr with | some s => parseNonNegNat "--pr" s | none => pure 0
          pure (PoolPad.custom pt pb pl pr)
        | other => throw s!"unknown --pad '{other}' (expected valid, same, or custom)"
      pure ({
        target := target, op := poolOp, dims := { c := c, h := h, w := w },
        kh := kh, kw := kw, sh := sh, sw := sw, pad := pad, excludePad := excl
      } : PoolSpec)) with
    | .ok s => pure s
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let compiled ← match compilePool spec with
    | .ok c => pure c
    | .error e => IO.eprintln s!"error: compile-pool failed: {e}"; return 1

  let manifest := renderPoolManifestJson compiled
  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "program.hwx") compiled.hwx
    IO.FS.writeBinFile (dir / "generic.bin") compiled.sections.generic
    IO.FS.writeBinFile (dir / "operation.bin") compiled.sections.operation
    IO.FS.writeBinFile (dir / "procedure.bin") compiled.sections.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") compiled.sections.textProperties
    IO.FS.writeFile (dir / "manifest.json") manifest
  catch e =>
    IO.eprintln s!"error: failed to write pool outputs to '{dirStr}': {e}"
    return 1

  IO.println (
    s!"compiled {target.name} {spec.op.opName} ({spec.dims.c}x{spec.dims.h}x{spec.dims.w} -> " ++
    s!"{compiled.outDims.c}x{compiled.outDims.h}x{compiled.outDims.w}, k={spec.kh}x{spec.kw}, " ++
    s!"s={spec.sh}x{spec.sw}, pad={spec.pad.name}, backend={compiled.backend.name}) to {dirStr}: " ++
    s!"hwx={compiled.hwx.size}B l2={compiled.l2UseBytes}B direct={compiled.direct}"
  )
  return 0

private def cmdCompileReduce (args : List String) : IO UInt32 := do
  let mut targetStr : Option String := none
  let mut opStr : Option String := none
  let mut hStr : Option String := none
  let mut wStr : Option String := none
  let mut cStr : Option String := none
  let mut axesStr : Option String := none
  let mut epsStr : Option String := none
  let mut outDir : Option String := none
  let mut rest := args

  while !rest.isEmpty do
    match rest with
    | "--target" :: tl =>
      match requireArgValue "--target" tl with
      | .ok (v, tl') => targetStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--op" :: tl =>
      match requireArgValue "--op" tl with
      | .ok (v, tl') => opStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--h" :: tl =>
      match requireArgValue "--h" tl with
      | .ok (v, tl') => hStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--w" :: tl =>
      match requireArgValue "--w" tl with
      | .ok (v, tl') => wStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--c" :: tl =>
      match requireArgValue "--c" tl with
      | .ok (v, tl') => cStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--axes" :: tl =>
      match requireArgValue "--axes" tl with
      | .ok (v, tl') => axesStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--eps" :: tl =>
      match requireArgValue "--eps" tl with
      | .ok (v, tl') => epsStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for compile-reduce"
      IO.eprintln usage
      return 1
    | [] => break

  let some tStr := targetStr
    | do IO.eprintln "error: --target <h16|h17> is required"; return 1
  let some target := Target.parse? tStr
    | do IO.eprintln s!"error: unknown target '{tStr}' (expected h16 or h17)"; return 1
  let some oStr := opStr
    | do IO.eprintln "error: --op <reduce_op> is required"; return 1
  let some redOp := ReduceOp.parse? oStr
    | do IO.eprintln s!"error: unknown reduce op '{oStr}'"; return 1
  let some hRaw := hStr
    | do IO.eprintln "error: --h <H> is required"; return 1
  let some wRaw := wStr
    | do IO.eprintln "error: --w <W> is required"; return 1
  let some cRaw := cStr
    | do IO.eprintln "error: --c <C> is required"; return 1
  let some axRaw := axesStr
    | do IO.eprintln "error: --axes <w|h|c|hw|cw|ch|chw> is required"; return 1
  let some axes := ReduceAxes.parse? axRaw
    | do IO.eprintln s!"error: invalid --axes '{axRaw}' (expected w, h, c, hw, cw, ch, or chw)"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let spec ← match (do
      let h ← parsePosNat "--h" hRaw
      let w ← parsePosNat "--w" wRaw
      let c ← parsePosNat "--c" cRaw
      let eps ← match epsStr with
        | some s =>
          let f ← parseFloatArg "--eps" s
          if f < 0.0 then
            throw s!"--eps must be >= 0, got '{s}'"
          let bits := floatToF16CanonicalZero f
          if !f16IsFinite bits then
            throw s!"--eps value {f} overflows finite fp16 range"
          pure bits
        | none => pure 0
      pure ({
        target := target, op := redOp, dims := { c := c, h := h, w := w },
        axes := axes, eps := eps
      } : ReduceSpec)) with
    | .ok s => pure s
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let compiled ← match compileReduce spec with
    | .ok c => pure c
    | .error e => IO.eprintln s!"error: compile-reduce failed: {e}"; return 1

  let manifest := renderReduceManifestJson compiled
  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "program.hwx") compiled.hwx
    IO.FS.writeBinFile (dir / "generic.bin") compiled.sections.generic
    IO.FS.writeBinFile (dir / "operation.bin") compiled.sections.operation
    IO.FS.writeBinFile (dir / "procedure.bin") compiled.sections.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") compiled.sections.textProperties
    IO.FS.writeFile (dir / "manifest.json") manifest
  catch e =>
    IO.eprintln s!"error: failed to write reduce outputs to '{dirStr}': {e}"
    return 1

  IO.println (
    s!"compiled {target.name} reduce {spec.op.name} axes={spec.axes.name} ({spec.dims.c}x{spec.dims.h}x{spec.dims.w} -> " ++
    s!"{compiled.outDims.c}x{compiled.outDims.h}x{compiled.outDims.w}, backend={compiled.backend.name}) to {dirStr}: " ++
    s!"hwx={compiled.hwx.size}B l2={compiled.l2UseBytes}B direct={compiled.direct}"
  )
  return 0

private def parseNatCsv (flag raw : String) (allowZero : Bool := true) : Except String (Array Nat) := do
  let parts := raw.splitOn ","
  if parts.isEmpty || (parts.length == 1 && parts[0]!.trimAscii.isEmpty) then
    throw s!"{flag}: expected comma-separated integers, got '{raw}'"
  let mut out : Array Nat := Array.mkEmpty parts.length
  for p in parts do
    let s := p.trimAscii.toString
    match s.toNat? with
    | none => throw s!"{flag}: invalid integer '{s}' in '{raw}'"
    | some v =>
      if !allowZero && v == 0 then
        throw s!"{flag}: value must be >= 1 in '{raw}'"
      out := out.push v
  pure out

private def parseViewShapeToken (tok : String) : Except String Shape4D := do
  let parts := (tok.trimAscii.toString).splitOn "x"
  if parts.length == 3 then
    let h ← parsePosNat "--shapes (H)" parts[0]!
    let w ← parsePosNat "--shapes (W)" parts[1]!
    let c ← parsePosNat "--shapes (C)" parts[2]!
    pure { n := 1, c := c, h := h, w := w }
  else if parts.length == 4 then
    let n ← parsePosNat "--shapes (N)" parts[0]!
    let c ← parsePosNat "--shapes (C)" parts[1]!
    let h ← parsePosNat "--shapes (H)" parts[2]!
    let w ← parsePosNat "--shapes (W)" parts[3]!
    pure { n := n, c := c, h := h, w := w }
  else
    throw s!"invalid shape '{tok}' (expected HxWxC or NxCxHxW)"

private def cmdCompileView (args : List String) : IO UInt32 := do
  let mut targetStr : Option String := none
  let mut opStr : Option String := none
  let mut nStr : Option String := none
  let mut cStr : Option String := none
  let mut hStr : Option String := none
  let mut wStr : Option String := none
  let mut permStr : Option String := none
  let mut beginStr : Option String := none
  let mut endStr : Option String := none
  let mut stepStr : Option String := none
  let mut axisStr : Option String := none
  let mut shapesStr : Option String := none
  let mut sizesStr : Option String := none
  let mut partStr : Option String := none
  let mut outShapeStr : Option String := none
  let mut outDir : Option String := none
  let mut rest := args

  while !rest.isEmpty do
    match rest with
    | "--target" :: tl =>
      match requireArgValue "--target" tl with
      | .ok (v, tl') => targetStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--op" :: tl =>
      match requireArgValue "--op" tl with
      | .ok (v, tl') => opStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--n" :: tl =>
      match requireArgValue "--n" tl with
      | .ok (v, tl') => nStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--c" :: tl =>
      match requireArgValue "--c" tl with
      | .ok (v, tl') => cStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--h" :: tl =>
      match requireArgValue "--h" tl with
      | .ok (v, tl') => hStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--w" :: tl =>
      match requireArgValue "--w" tl with
      | .ok (v, tl') => wStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--perm" :: tl =>
      match requireArgValue "--perm" tl with
      | .ok (v, tl') => permStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--begin" :: tl =>
      match requireArgValue "--begin" tl with
      | .ok (v, tl') => beginStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--end" :: tl =>
      match requireArgValue "--end" tl with
      | .ok (v, tl') => endStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--step" :: tl =>
      match requireArgValue "--step" tl with
      | .ok (v, tl') => stepStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--axis" :: tl =>
      match requireArgValue "--axis" tl with
      | .ok (v, tl') => axisStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--shapes" :: tl =>
      match requireArgValue "--shapes" tl with
      | .ok (v, tl') => shapesStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--sizes" :: tl =>
      match requireArgValue "--sizes" tl with
      | .ok (v, tl') => sizesStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--part" :: tl =>
      match requireArgValue "--part" tl with
      | .ok (v, tl') => partStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-shape" :: tl =>
      match requireArgValue "--out-shape" tl with
      | .ok (v, tl') => outShapeStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for compile-view"
      IO.eprintln usage
      return 1
    | [] => break

  let some tStr := targetStr
    | do IO.eprintln "error: --target <h16|h17> is required"; return 1
  let some target := Target.parse? tStr
    | do IO.eprintln s!"error: unknown target '{tStr}' (expected h16 or h17)"; return 1
  let some opRaw := opStr
    | do IO.eprintln "error: --op <transpose|slice|concat|split|reshape> is required"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let parseInShape := fun () => do
    let some hRaw := hStr | throw "--h <H> is required"
    let some wRaw := wStr | throw "--w <W> is required"
    let some cRaw := cStr | throw "--c <C> is required"
    let n ← match nStr with | some s => parsePosNat "--n" s | none => pure 1
    let c ← parsePosNat "--c" cRaw
    let h ← parsePosNat "--h" hRaw
    let w ← parsePosNat "--w" wRaw
    pure ({ n := n, c := c, h := h, w := w } : Shape4D)

  let plan ← match (do
      match opRaw.toLower with
      | "transpose" =>
        let sh ← parseInShape ()
        let some pRaw := permStr | throw "transpose requires --perm p0,p1,p2,p3"
        let perm ← parseNatCsv "--perm" pRaw true
        planTranspose sh perm
      | "slice" =>
        let sh ← parseInShape ()
        let some bRaw := beginStr | throw "slice requires --begin b0,b1,b2,b3"
        let some eRaw := endStr | throw "slice requires --end e0,e1,e2,e3"
        let beg ← parseNatCsv "--begin" bRaw true
        let en ← parseNatCsv "--end" eRaw true
        let st ← match stepStr with
          | some sRaw => parseNatCsv "--step" sRaw false
          | none => pure #[1, 1, 1, 1]
        planSlice sh beg en st
      | "concat" =>
        let some axRaw := axisStr | throw "concat requires --axis <1|2|3>"
        let ax ← parseNonNegNat "--axis" axRaw
        let some shRaw := shapesStr | throw "concat requires --shapes H0xW0xC0,H1xW1xC1,..."
        let toks := shRaw.splitOn ","
        let mut shs : Array Shape4D := Array.mkEmpty toks.length
        for tok in toks do
          shs := shs.push (← parseViewShapeToken tok)
        planConcat shs ax
      | "split" =>
        let sh ← parseInShape ()
        let some axRaw := axisStr | throw "split requires --axis <1|2|3>"
        let ax ← parseNonNegNat "--axis" axRaw
        let some szRaw := sizesStr | throw "split requires --sizes s0,s1,..."
        let sizes ← parseNatCsv "--sizes" szRaw false
        let some pRaw := partStr | throw "split requires --part <idx>"
        let part ← parseNonNegNat "--part" pRaw
        planSplitPart sh ax sizes part
      | "reshape" =>
        let sh ← parseInShape ()
        let some oRaw := outShapeStr | throw "reshape requires --out-shape No,Co,Ho,Wo"
        let oArr ← parseNatCsv "--out-shape" oRaw false
        let oSh ← Shape4D.ofArray oArr
        planReshape sh oSh
      | other => throw s!"unknown view op '{other}' (expected transpose, slice, concat, split, or reshape)") with
    | .ok p => pure p
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let compiled ← match compileView target plan with
    | .ok c => pure c
    | .error e => IO.eprintln s!"error: compile-view failed: {e}"; return 1

  let manifest := renderViewManifestJson compiled
  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "program.hwx") compiled.hwx
    IO.FS.writeBinFile (dir / "generic.bin") compiled.sections.generic
    IO.FS.writeBinFile (dir / "operation.bin") compiled.sections.operation
    IO.FS.writeBinFile (dir / "procedure.bin") compiled.sections.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") compiled.sections.textProperties
    IO.FS.writeFile (dir / "manifest.json") manifest
  catch e =>
    IO.eprintln s!"error: failed to write view outputs to '{dirStr}': {e}"
    return 1

  let os := compiled.plan.outShape
  IO.println (
    s!"compiled {target.name} view {compiled.plan.opName} (out={os.n}x{os.c}x{os.h}x{os.w}, " ++
    s!"tasks={compiled.plan.copies.size}) to {dirStr}: hwx={compiled.hwx.size}B l2_peak={compiled.l2PeakBytes}B"
  )
  return 0

private def renderMatmulManifestJson (compiled : CompiledMatmul) : String :=
  let info := compiled.info
  let secs := compiled.sections
  let sp := compiled.spec
  "{\n" ++
  "  \"op\": \"matmul\",\n" ++
  s!"  \"target\": \"{compiled.target.name}\",\n" ++
  "  \"hwx\": \"program.hwx\",\n" ++
  "  \"sections\": {\n" ++
  "    \"generic\": {\"file\": \"generic.bin\", \"size\": " ++ toString secs.generic.size ++ "},\n" ++
  "    \"operation\": {\"file\": \"operation.bin\", \"size\": " ++ toString secs.operation.size ++ "},\n" ++
  "    \"procedure\": {\"file\": \"procedure.bin\", \"size\": " ++ toString secs.procedure.size ++ "},\n" ++
  "    \"text_properties\": {\"file\": \"text-properties.bin\", \"size\": " ++ toString secs.textProperties.size ++ "}\n" ++
  "  },\n" ++
  "  \"matmul\": {\n" ++
  s!"    \"b\": {sp.b}, \"m\": {sp.m}, \"k\": {sp.k}, \"n\": {sp.n},\n" ++
  s!"    \"transpose_a\": {if sp.transposeA then "true" else "false"},\n" ++
  s!"    \"transpose_b\": {if sp.transposeB then "true" else "false"},\n" ++
  s!"    \"scale_f16\": {sp.scaleBits.toNat}\n" ++
  "  },\n" ++
  s!"  \"direct\": {if compiled.direct then "true" else "false"},\n" ++
  s!"  \"l2_peak_bytes\": {compiled.l2PeakBytes},\n" ++
  s!"  \"scratch_bytes\": {compiled.scratchBytes},\n" ++
  s!"  \"task_count\": {compiled.taskCount},\n" ++
  s!"  \"record_count\": {info.records},\n" ++
  s!"  \"relocation_count\": {info.relocCount},\n" ++
  s!"  \"has_kernel\": {if info.kernelSection.isSome then "true" else "false"},\n" ++
  s!"  \"has_scratch\": {if info.scratchSection.isSome then "true" else "false"},\n" ++
  "  \"surfaces\": [\n" ++
  renderSurfacesJson info.surfaces ++ "\n" ++
  "  ]\n" ++
  "}\n"

private def cmdCompileMatmul (args : List String) : IO UInt32 := do
  let mut targetStr : Option String := none
  let mut bStr : Option String := none
  let mut mStr : Option String := none
  let mut kStr : Option String := none
  let mut nStr : Option String := none
  let mut transposeA := false
  let mut transposeB := false
  let mut scaleStr : Option String := none
  let mut outDir : Option String := none
  let mut rest := args

  while !rest.isEmpty do
    match rest with
    | "--target" :: tl =>
      match requireArgValue "--target" tl with
      | .ok (v, tl') => targetStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--b" :: tl =>
      match requireArgValue "--b" tl with
      | .ok (v, tl') => bStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--m" :: tl =>
      match requireArgValue "--m" tl with
      | .ok (v, tl') => mStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--k" :: tl =>
      match requireArgValue "--k" tl with
      | .ok (v, tl') => kStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--n" :: tl =>
      match requireArgValue "--n" tl with
      | .ok (v, tl') => nStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--transpose-a" :: tl =>
      transposeA := true; rest := tl
    | "--transpose-b" :: tl =>
      transposeB := true; rest := tl
    | "--scale" :: tl =>
      match requireArgValue "--scale" tl with
      | .ok (v, tl') => scaleStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl | "--out" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for compile-matmul"
      IO.eprintln usage
      return 1
    | [] => break

  let some tStr := targetStr
    | do IO.eprintln "error: --target <h16|h17> is required"; return 1
  let some target := Target.parse? tStr
    | do IO.eprintln s!"error: unknown target '{tStr}' (expected h16 or h17)"; return 1
  let some mRaw := mStr
    | do IO.eprintln "error: --m <M> is required"; return 1
  let some kRaw := kStr
    | do IO.eprintln "error: --k <K> is required"; return 1
  let some nRaw := nStr
    | do IO.eprintln "error: --n <N> is required"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let spec ← match (do
      let b ← match bStr with | some s => parsePosNat "--b" s | none => pure 1
      let m ← parsePosNat "--m" mRaw
      let k ← parsePosNat "--k" kRaw
      let n ← parsePosNat "--n" nRaw
      let scaleBits ← match scaleStr with
        | none => pure f16PosOne
        | some s =>
          let bits ← parseF16Arg "--scale" s
          if !f16IsFinite bits then
            throw s!"--scale '{s}' overflows finite fp16 range"
          pure bits
      pure ({
        b := b, m := m, k := k, n := n,
        transposeA := transposeA, transposeB := transposeB, scaleBits := scaleBits
      } : MatmulSpec)) with
    | .ok s => pure s
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let compiled ← match compileMatmul target spec with
    | .ok c => pure c
    | .error e => IO.eprintln s!"error: compile-matmul failed: {e}"; return 1

  let manifest := renderMatmulManifestJson compiled
  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "program.hwx") compiled.hwx
    IO.FS.writeBinFile (dir / "generic.bin") compiled.sections.generic
    IO.FS.writeBinFile (dir / "operation.bin") compiled.sections.operation
    IO.FS.writeBinFile (dir / "procedure.bin") compiled.sections.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") compiled.sections.textProperties
    IO.FS.writeFile (dir / "manifest.json") manifest
  catch e =>
    IO.eprintln s!"error: failed to write matmul outputs to '{dirStr}': {e}"
    return 1

  IO.println (
    s!"compiled {target.name} matmul (B={spec.b}, M={spec.m}, K={spec.k}, N={spec.n}, " ++
    s!"trA={spec.transposeA}, trB={spec.transposeB}, tasks={compiled.taskCount}) to {dirStr}: " ++
    s!"hwx={compiled.hwx.size}B scratch={compiled.scratchBytes}B l2_peak={compiled.l2PeakBytes}B"
  )
  return 0

private def jsonObjField? (j : Json) (key : String) : Option Json :=
  match j.getObjVal? key with
  | .ok v => if v.isNull then none else some v
  | .error _ => none

private def jsonReqStr (j : Json) (key ctx : String) : Except String String :=
  match j.getObjValAs? String key with
  | .ok s => pure s
  | .error e => throw s!"{ctx}: missing or invalid string field '{key}' ({e})"

private def jsonOptStr (j : Json) (key ctx : String) : Except String (Option String) :=
  match jsonObjField? j key with
  | none => pure none
  | some v =>
    match v.getStr? with
    | .ok s => pure (some s)
    | .error e => throw s!"{ctx}: invalid string field '{key}' ({e})"

private def jsonReqPosNat (j : Json) (key ctx : String) : Except String Nat :=
  match j.getObjValAs? Nat key with
  | .ok n => if n == 0 then throw s!"{ctx}: field '{key}' must be >= 1" else pure n
  | .error e => throw s!"{ctx}: missing or invalid positive integer '{key}' ({e})"

private def jsonOptPosNat (j : Json) (key ctx : String) : Except String (Option Nat) :=
  match jsonObjField? j key with
  | none => pure none
  | some v =>
    match v.getNat? with
    | .ok n => if n == 0 then throw s!"{ctx}: field '{key}' must be >= 1" else pure (some n)
    | .error e => throw s!"{ctx}: invalid positive integer '{key}' ({e})"

private def jsonOptNat (j : Json) (key ctx : String) : Except String (Option Nat) :=
  match jsonObjField? j key with
  | none => pure none
  | some v =>
    match v.getNat? with
    | .ok n => pure (some n)
    | .error e => throw s!"{ctx}: invalid non-negative integer '{key}' ({e})"

private def jsonOptBoolDefault (j : Json) (key ctx : String) (defaultVal : Bool) : Except String Bool :=
  match jsonObjField? j key with
  | none => pure defaultVal
  | some v =>
    match v.getBool? with
    | .ok b => pure b
    | .error e => throw s!"{ctx}: invalid boolean field '{key}' ({e})"

private def jsonOptFloat (j : Json) (key ctx : String) : Except String (Option Float) :=
  match jsonObjField? j key with
  | none => pure none
  | some v =>
    match v.getNum? with
    | .ok num =>
      let f := num.toFloat
      if f.isNaN || f.isInf then
        throw s!"{ctx}: field '{key}' must be a finite number"
      else
        pure (some f)
    | .error e => throw s!"{ctx}: invalid numeric field '{key}' ({e})"

private def jsonOptBool (j : Json) (key ctx : String) : Except String Bool :=
  match jsonObjField? j key with
  | none => pure false
  | some v =>
    match v.getBool? with
    | .ok b => pure b
    | .error e => throw s!"{ctx}: invalid boolean field '{key}' ({e})"

private def jsonReqStrArray (j : Json) (key ctx : String) : Except String (Array String) :=
  match j.getObjValAs? (Array Json) key with
  | .error e => throw s!"{ctx}: missing or invalid array '{key}' ({e})"
  | .ok arr => do
    let mut out : Array String := Array.mkEmpty arr.size
    for elem in arr do
      match elem.getStr? with
      | .ok s => out := out.push s
      | .error e => throw s!"{ctx}: element in '{key}' must be a string ({e})"
    pure out

private def jsonReqNatArray (j : Json) (key ctx : String) : Except String (Array Nat) :=
  match j.getObjValAs? (Array Json) key with
  | .error e => throw s!"{ctx}: missing or invalid integer array '{key}' ({e})"
  | .ok arr => do
    let mut out : Array Nat := Array.mkEmpty arr.size
    for elem in arr do
      match elem.getNat? with
      | .ok v => out := out.push v
      | .error e => throw s!"{ctx}: element in '{key}' must be a non-negative integer ({e})"
    pure out

private def jsonOptNatArray (j : Json) (key ctx : String) : Except String (Option (Array Nat)) :=
  match jsonObjField? j key with
  | none => pure none
  | some _ => some <$> jsonReqNatArray j key ctx

private def parseJsonActField (j : Json) (ctx : String) : Except String (Option (String × ActParamFlags)) := do
  match jsonObjField? j "act" with
  | none => pure none
  | some v =>
    match v.getStr? with
    | .ok s => pure (some (s, {}))
    | .error _ =>
      let nm ← jsonReqStr v "name" s!"{ctx}.act"
      let params : ActParamFlags := {
        alpha := ← jsonOptFloat v "alpha" s!"{ctx}.act"
        beta := ← jsonOptFloat v "beta" s!"{ctx}.act"
        lo := ← jsonOptFloat v "lo" s!"{ctx}.act"
        hi := ← jsonOptFloat v "hi" s!"{ctx}.act"
        eps := ← jsonOptFloat v "eps" s!"{ctx}.act"
        scalar := ← jsonOptFloat v "scalar" s!"{ctx}.act"
      }
      pure (some (nm, params))

private def resolveSpecRelPath (specDir : System.FilePath) (raw : String) : System.FilePath :=
  let p := System.FilePath.mk raw
  if p.isAbsolute then p else specDir / p

private structure JsonQuantPaths where
  format : String
  scales? : Option String := none
  zeroPoints? : Option String := none
  palette? : Option String := none

private def parseJsonQuantField (j : Json) (ctx : String) : Except String (Option JsonQuantPaths) := do
  match jsonObjField? j "quant" with
  | none => pure none
  | some qj =>
    let fmt ← jsonReqStr qj "format" s!"{ctx}.quant"
    let sc? ← jsonOptStr qj "scales" s!"{ctx}.quant"
    let zp? ← jsonOptStr qj "zero_points" s!"{ctx}.quant"
    let pal? ← jsonOptStr qj "palette" s!"{ctx}.quant"
    pure (some { format := fmt, scales? := sc?, zeroPoints? := zp?, palette? := pal? })

private def renderGraphManifestJson (compiled : CompiledGraph) : String :=
  let info := compiled.info
  let secs := compiled.sections
  let nodeItems := compiled.nodes.toList.map fun nd =>
    let engStr := match nd.engine with | .ne => "ne" | .pe => "pe"
    "    {\n" ++
    s!"      \"id\": \"{escapeJsonString nd.id}\",\n" ++
    s!"      \"task_id\": {nd.taskId},\n" ++
    s!"      \"engine\": \"{engStr}\",\n" ++
    s!"      \"op\": \"{escapeJsonString nd.opKind}\",\n" ++
    s!"      \"shape\": [1, {nd.outDims.c}, {nd.outDims.h}, {nd.outDims.w}],\n" ++
    s!"      \"out_placement\": \"{nd.outLocation.placementName}\",\n" ++
    s!"      \"direct\": {if nd.direct then "true" else "false"},\n" ++
    s!"      \"l2_use_bytes\": {nd.l2UseBytes}\n" ++
    "    }"
  "{\n" ++
  "  \"op\": \"graph\",\n" ++
  s!"  \"target\": \"{compiled.target.name}\",\n" ++
  "  \"hwx\": \"program.hwx\",\n" ++
  "  \"sections\": {\n" ++
  "    \"generic\": {\"file\": \"generic.bin\", \"size\": " ++ toString secs.generic.size ++ "},\n" ++
  "    \"operation\": {\"file\": \"operation.bin\", \"size\": " ++ toString secs.operation.size ++ "},\n" ++
  "    \"procedure\": {\"file\": \"procedure.bin\", \"size\": " ++ toString secs.procedure.size ++ "},\n" ++
  "    \"text_properties\": {\"file\": \"text-properties.bin\", \"size\": " ++ toString secs.textProperties.size ++ "}\n" ++
  "  },\n" ++
  s!"  \"task_count\": {compiled.taskCount},\n" ++
  s!"  \"record_count\": {info.records},\n" ++
  s!"  \"relocation_count\": {info.relocCount},\n" ++
  s!"  \"has_kernel\": {if info.kernelSection.isSome then "true" else "false"},\n" ++
  s!"  \"has_scratch\": {if info.scratchSection.isSome then "true" else "false"},\n" ++
  s!"  \"kernel_bytes\": {compiled.kernelBytes},\n" ++
  s!"  \"scratch_bytes\": {compiled.scratchBytes},\n" ++
  s!"  \"l2_peak_bytes\": {compiled.l2PeakBytes},\n" ++
  s!"  \"strip_count\": {compiled.stripCount},\n" ++
  s!"  \"strip_height\": {compiled.stripHeight},\n" ++
  "  \"nodes\": [\n" ++
  String.intercalate ",\n" nodeItems ++ "\n" ++
  "  ],\n" ++
  "  \"surfaces\": [\n" ++
  renderSurfacesJson info.surfaces ++ "\n" ++
  "  ]\n" ++
  "}\n"

private def cmdCompileGraph (args : List String) : IO UInt32 := do
  let mut specPath : Option String := none
  let mut outDir : Option String := none
  let mut stripFlag : Option StripMode := none
  let mut rest := args
  while !rest.isEmpty do
    match rest with
    | "--spec" :: tl =>
      match requireArgValue "--spec" tl with
      | .ok (v, tl') => specPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--strip-height" :: tl =>
      match requireArgValue "--strip-height" tl with
      | .ok (v, tl') =>
        match StripMode.parse? v with
        | some m => stripFlag := some m; rest := tl'
        | none => IO.eprintln s!"error: invalid --strip-height '{v}' (expected off, auto, or positive integer)"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for compile-graph"
      IO.eprintln usage
      return 1
    | [] => break

  let some sPathStr := specPath
    | do IO.eprintln "error: --spec <graph.json> is required"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let specFilePath := System.FilePath.mk sPathStr
  let specDir := specFilePath.parent.getD (System.FilePath.mk ".")

  let rawJson ← try
    IO.FS.readFile specFilePath
  catch e =>
    IO.eprintln s!"error: failed to read graph spec '{sPathStr}': {e}"
    return 1

  let root ← match Json.parse rawJson with
    | .ok j => pure j
    | .error e => IO.eprintln s!"error: invalid JSON in '{sPathStr}': {e}"; return 1

  let target ← match (do
      let tStr ← jsonReqStr root "target" "graph"
      match Target.parse? tStr with
      | some t => pure t
      | none => throw s!"unknown target '{tStr}' (expected h16 or h17)") with
    | .ok t => pure t
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let defaultPlacement ← match (do
      match ← jsonOptStr root "placement" "graph" with
      | none => pure EdgePlacementMode.auto
      | some pStr =>
        match EdgePlacementMode.parse? pStr with
        | some m => pure m
        | none => throw s!"unknown placement '{pStr}' (expected auto, l2, or dram)") with
    | .ok p => pure p
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let jsonStripMode : Option StripMode ← match ((do
      match jsonObjField? root "strip_height" with
      | none => pure none
      | some v =>
        match v.getStr? with
        | .ok s =>
          match StripMode.parse? s with
          | some m => pure (some m)
          | none => throw s!"invalid strip_height '{s}' (expected \"off\", \"auto\", or positive integer)"
        | .error _ =>
          match v.getNat? with
          | .ok n =>
            if n == 0 then throw "strip_height must be >= 1"
            else pure (some (.fixed n))
          | .error e => throw s!"invalid strip_height field ({e})") : Except String (Option StripMode)) with
    | .ok m => pure m
    | .error e => IO.eprintln s!"error: {e}"; return 1
  let stripMode := stripFlag.getD (jsonStripMode.getD .off)

  let inputsArr ← match root.getObjValAs? (Array Json) "inputs" with
    | .ok a => pure a
    | .error e => IO.eprintln s!"error: graph missing 'inputs' array ({e})"; return 1

  let mut graphInputs : Array GraphInput := #[]
  let mut knownShapes : Array (String × TensorDims) := #[]
  for idx in [:inputsArr.size] do
    let ij := inputsArr[idx]!
    let ctx := s!"inputs[{idx}]"
    let gi ← match (do
        let nm ← jsonReqStr ij "name" ctx
        let c ← jsonReqPosNat ij "c" ctx
        let h ← jsonReqPosNat ij "h" ctx
        let w ← jsonReqPosNat ij "w" ctx
        let sn := (← jsonOptStr ij "short_name" ctx).getD ""
        pure ({ name := nm, shape := { c := c, h := h, w := w }, shortName := sn } : GraphInput)) with
      | .ok g => pure g
      | .error e => IO.eprintln s!"error: {e}"; return 1
    graphInputs := graphInputs.push gi
    knownShapes := knownShapes.push (gi.name, gi.shape)

  let nodesArr ← match root.getObjValAs? (Array Json) "nodes" with
    | .ok a => pure a
    | .error e => IO.eprintln s!"error: graph missing 'nodes' array ({e})"; return 1

  let mut graphNodes : Array GraphNode := #[]
  for idx in [:nodesArr.size] do
    let nj := nodesArr[idx]!
    let ctx := s!"nodes[{idx}]"
    let (ndId, ndOpStr, ndInputs, ndOutPlace?) ← match (do
        let id ← jsonReqStr nj "id" ctx
        let opStr ← jsonReqStr nj "op" ctx
        let inps ← jsonReqStrArray nj "inputs" ctx
        let op? ← match ← jsonOptStr nj "out_placement" ctx with
          | none => pure none
          | some pStr =>
            match EdgePlacementMode.parse? pStr with
            | some m => pure (some m)
            | none => throw s!"{ctx}: unknown out_placement '{pStr}'"
        pure (id, opStr.toLower, inps, op?)) with
      | .ok v => pure v
      | .error e => IO.eprintln s!"error: {e}"; return 1

    let lookupShape := fun (nm : String) =>
      knownShapes.find? (fun (k, _) => k == nm) |>.map Prod.snd

    let (nodeOp, inferredDims) ← match ndOpStr with
      | "conv1x1" | "linear" =>
        let isLinear := ndOpStr == "linear"
        let parsed ← match (do
            if ndInputs.size != 1 then throw s!"{ctx} ({ndOpStr}): expected 1 input"
            let some inDims := lookupShape ndInputs[0]!
              | throw s!"{ctx} ({ndOpStr}): unknown input '{ndInputs[0]!}'"
            let cin ← match ← jsonOptPosNat nj "cin" ctx with
              | some v => pure v
              | none =>
                match ← jsonOptPosNat nj "k" ctx with
                | some v => pure v
                | none => pure inDims.c
            let cout ← match ← jsonOptPosNat nj "cout" ctx with
              | some v => pure v
              | none =>
                match ← jsonOptPosNat nj "n" ctx with
                | some v => pure v
                | none => throw s!"{ctx} ({ndOpStr}): missing required field 'cout' (or 'n')"
            let wRel ← jsonReqStr nj "weights" ctx
            let bRel? ← jsonOptStr nj "bias" ctx
            let reluFlag ← jsonOptBool nj "relu" ctx
            let actInfo? ← parseJsonActField nj ctx
            let act ← match actInfo? with
              | none => parseConvAct reluFlag none {}
              | some (nm, p) => parseConvAct reluFlag (some nm) p
            let quantPaths? ← parseJsonQuantField nj ctx
            pure (cin, cout, wRel, bRel?, act, quantPaths?, { c := cout, h := inDims.h, w := inDims.w : TensorDims })) with
          | .ok v => pure v
          | .error e => IO.eprintln s!"error: {e}"; return 1
        let (cin, cout, wRel, bRel?, act, quantPaths?, outDims) := parsed
        let wPath := resolveSpecRelPath specDir wRel
        let weightsBytes ← try IO.FS.readBinFile wPath catch e =>
          IO.eprintln s!"error: {ctx}: failed to read weights '{wPath}': {e}"; return 1
        let biasBytes? ← match bRel? with
          | none => pure none
          | some bRel =>
            let bPath := resolveSpecRelPath specDir bRel
            try some <$> IO.FS.readBinFile bPath catch e =>
              IO.eprintln s!"error: {ctx}: failed to read bias '{bPath}': {e}"; return 1
        let quantSpec? ← match quantPaths? with
          | none => pure none
          | some qp =>
            match ← loadQuantWeightSpecIO qp.format weightsBytes
                (qp.scales?.map (resolveSpecRelPath specDir))
                (qp.zeroPoints?.map (resolveSpecRelPath specDir))
                (qp.palette?.map (resolveSpecRelPath specDir)) ctx with
            | .ok q? => pure q?
            | .error e => IO.eprintln s!"error: {e}"; return 1
        let op := if isLinear then NodeOp.linear cin cout weightsBytes biasBytes? act quantSpec?
                  else NodeOp.conv1x1 cin cout weightsBytes biasBytes? act quantSpec?
        pure (op, outDims)

      | "conv3x3" =>
        let parsed ← match (do
            if ndInputs.size != 1 then throw s!"{ctx} (conv3x3): expected 1 input"
            let some inDims := lookupShape ndInputs[0]!
              | throw s!"{ctx} (conv3x3): unknown input '{ndInputs[0]!}'"
            let modeStr := (← jsonOptStr nj "mode" ctx).getD "dense"
            let some mode := Conv3x3Mode.parse? modeStr
              | throw s!"{ctx}: unknown conv3x3 mode '{modeStr}'"
            let cin := (← jsonOptPosNat nj "cin" ctx).getD inDims.c
            let cout ← match ← jsonOptPosNat nj "cout" ctx with
              | some co => pure co
              | none => if mode == Conv3x3Mode.depthwise then pure cin else throw s!"{ctx}: missing required field 'cout'"
            let stride := (← jsonOptPosNat nj "stride" ctx).getD 1
            let padStr := (← jsonOptStr nj "pad" ctx).getD "same"
            let some padMode := PadMode.parse? padStr
              | throw s!"{ctx}: unknown conv3x3 pad '{padStr}'"
            let wRel ← jsonReqStr nj "weights" ctx
            let bRel? ← jsonOptStr nj "bias" ctx
            let reluFlag ← jsonOptBool nj "relu" ctx
            let actInfo? ← parseJsonActField nj ctx
            let act ← match actInfo? with
              | none => parseConvAct reluFlag none {}
              | some (nm, p) => parseConvAct reluFlag (some nm) p
            let quantPaths? ← parseJsonQuantField nj ctx
            let sh : ConvSpec := {
              kernelSize := 3, mode := mode, h := inDims.h, w := inDims.w,
              cin := cin, cout := cout, stride := stride, pad := padMode
            }
            validateConvShape sh
            pure (mode, cin, cout, stride, padMode, wRel, bRel?, act, quantPaths?, { c := cout, h := sh.outH, w := sh.outW : TensorDims })) with
          | .ok v => pure v
          | .error e => IO.eprintln s!"error: {e}"; return 1
        let (mode, cin, cout, stride, padMode, wRel, bRel?, act, quantPaths?, outDims) := parsed
        let wPath := resolveSpecRelPath specDir wRel
        let weightsBytes ← try IO.FS.readBinFile wPath catch e =>
          IO.eprintln s!"error: {ctx}: failed to read weights '{wPath}': {e}"; return 1
        let biasBytes? ← match bRel? with
          | none => pure none
          | some bRel =>
            let bPath := resolveSpecRelPath specDir bRel
            try some <$> IO.FS.readBinFile bPath catch e =>
              IO.eprintln s!"error: {ctx}: failed to read bias '{bPath}': {e}"; return 1
        let quantSpec? ← match quantPaths? with
          | none => pure none
          | some qp =>
            match ← loadQuantWeightSpecIO qp.format weightsBytes
                (qp.scales?.map (resolveSpecRelPath specDir))
                (qp.zeroPoints?.map (resolveSpecRelPath specDir))
                (qp.palette?.map (resolveSpecRelPath specDir)) ctx with
            | .ok q? => pure q?
            | .error e => IO.eprintln s!"error: {e}"; return 1
        pure (NodeOp.conv3x3 mode cin cout stride padMode weightsBytes biasBytes? act quantSpec?, outDims)

      | "pe" =>
        let parsed ← match (do
            let peOpStr := (← jsonReqStr nj "pe_op" ctx).toLower
            let optF16 := fun (key : String) => do
              match ← jsonOptFloat nj key ctx with
              | none => pure none
              | some f =>
                let bits := floatToF16CanonicalZero f
                if !f16IsFinite bits then
                  throw s!"{ctx}: '{key}' value {f} overflows finite fp16 range"
                pure (some bits)
            match peOpStr with
            | "abs" | "square" | "linear" | "chain" | "sadd" | "ssub" | "srsub" | "smul" | "sdiv" =>
              if ndInputs.size != 1 then throw s!"{ctx} (pe {peOpStr}): expected 1 input"
              let some inDims := lookupShape ndInputs[0]!
                | throw s!"{ctx}: unknown input '{ndInputs[0]!}'"
              let op ← match peOpStr with
                | "abs" => pure PeOp.abs
                | "square" => pure PeOp.square
                | "linear" =>
                  let a := (← optF16 "alpha").getD f16PosOne
                  let b := (← optF16 "beta").getD f16PosZero
                  pure (PeOp.linearActivation a b)
                | "chain" =>
                  let ins ← optF16 "in_scale"
                  let b ← optF16 "bias"
                  let sc ← optF16 "scale"
                  pure (PeOp.chain ins b sc)
                | "sadd" =>
                  let some s ← optF16 "scalar" | throw s!"{ctx}: sadd requires 'scalar'"
                  pure (PeOp.scalarAdd s)
                | "ssub" =>
                  let some s ← optF16 "scalar" | throw s!"{ctx}: ssub requires 'scalar'"
                  pure (PeOp.scalarSub s)
                | "srsub" =>
                  let some s ← optF16 "scalar" | throw s!"{ctx}: srsub requires 'scalar'"
                  pure (PeOp.scalarRSub s)
                | "smul" =>
                  let some s ← optF16 "scalar" | throw s!"{ctx}: smul requires 'scalar'"
                  pure (PeOp.scalarMul s)
                | "sdiv" =>
                  let some s ← optF16 "scalar" | throw s!"{ctx}: sdiv requires 'scalar'"
                  PeOp.scalarDiv s
                | _ => throw s!"{ctx}: unsupported PE op '{peOpStr}'"
              pure (NodeOp.pe1 op, inDims)
            | "add" | "sub" | "rsub" | "mul" | "max" | "min" =>
              if ndInputs.size != 2 then throw s!"{ctx} (pe {peOpStr}): expected 2 inputs"
              let some d1 := lookupShape ndInputs[0]!
                | throw s!"{ctx}: unknown input '{ndInputs[0]!}'"
              let some d2 := lookupShape ndInputs[1]!
                | throw s!"{ctx}: unknown input '{ndInputs[1]!}'"
              let outDims ← broadcastDims d1 d2
              let (binOp, rev) := match peOpStr with
                | "add" => (PeBinOp.add, false)
                | "sub" => (PeBinOp.sub, false)
                | "rsub" => (PeBinOp.sub, true)
                | "mul" => (PeBinOp.mul, false)
                | "max" => (PeBinOp.max, false)
                | _ => (PeBinOp.min, false)
              pure (NodeOp.pe2 binOp rev, outDims)
            | other => throw s!"{ctx}: unknown pe_op '{other}'") with
          | .ok v => pure v
          | .error e => IO.eprintln s!"error: {e}"; return 1
        pure parsed

      | "lut" =>
        let parsed ← match (do
            if ndInputs.size != 1 then throw s!"{ctx} (lut): expected 1 input"
            let some inDims := lookupShape ndInputs[0]!
              | throw s!"{ctx}: unknown input '{ndInputs[0]!}'"
            let some (actName, params) ← parseJsonActField nj ctx
              | throw s!"{ctx} (lut): missing required 'act' field"
            let neAct ← parseNeActSpec actName params
            pure (NodeOp.lut neAct, inDims)) with
          | .ok v => pure v
          | .error e => IO.eprintln s!"error: {e}"; return 1
        pure parsed

      | "max_pool" | "avg_pool" =>
        let poolKind : PoolKind := if ndOpStr == "max_pool" then .max else .avg
        let parsed ← match (do
            if ndInputs.size != 1 then throw s!"{ctx} ({ndOpStr}): expected 1 input"
            let some inDims := lookupShape ndInputs[0]!
              | throw s!"{ctx}: unknown input '{ndInputs[0]!}'"
            let kh ← jsonReqPosNat nj "kh" ctx
            let kw ← jsonReqPosNat nj "kw" ctx
            let sh := (← jsonOptPosNat nj "sh" ctx).getD 1
            let sw := (← jsonOptPosNat nj "sw" ctx).getD 1
            let padStr := ((← jsonOptStr nj "pad" ctx).getD "valid").toLower
            let pad ← match padStr with
              | "valid" => pure PoolPad.valid
              | "same" => pure PoolPad.same
              | "custom" =>
                let pt := (← jsonOptNat nj "pt" ctx).getD 0
                let pb := (← jsonOptNat nj "pb" ctx).getD 0
                let pl := (← jsonOptNat nj "pl" ctx).getD 0
                let pr := (← jsonOptNat nj "pr" ctx).getD 0
                pure (PoolPad.custom pt pb pl pr)
              | other => throw s!"{ctx} ({ndOpStr}): unknown pad '{other}' (expected valid, same, or custom)"
            let excl ← jsonOptBoolDefault nj "exclude_pad" ctx true
            let pspec : PoolSpec := {
              target := target, op := poolKind, dims := inDims,
              kh := kh, kw := kw, sh := sh, sw := sw, pad := pad, excludePad := excl
            }
            let (_, gh, gw) ← resolvePoolBackend pspec
            let outDims : TensorDims := { c := inDims.c, h := gh.out, w := gw.out }
            pure (NodeOp.pool poolKind kh kw sh sw pad excl, outDims)) with
          | .ok v => pure v
          | .error e => IO.eprintln s!"error: {e}"; return 1
        pure parsed

      | "reduce" =>
        let parsed ← match (do
            if ndInputs.size != 1 then throw s!"{ctx} (reduce): expected 1 input"
            let some inDims := lookupShape ndInputs[0]!
              | throw s!"{ctx}: unknown input '{ndInputs[0]!}'"
            let redOpStr ← jsonReqStr nj "reduce_op" ctx
            let some redOp := ReduceOp.parse? redOpStr
              | throw s!"{ctx} (reduce): unknown reduce_op '{redOpStr}'"
            let axesStr ← jsonReqStr nj "axes" ctx
            let some axes := ReduceAxes.parse? axesStr
              | throw s!"{ctx} (reduce): invalid axes '{axesStr}' (expected w, h, c, hw, cw, ch, or chw)"
            let eps : UInt16 ← match ← jsonOptFloat nj "eps" ctx with
              | none => pure 0
              | some f =>
                if f < 0.0 then
                  throw s!"{ctx} (reduce): eps must be >= 0, got {f}"
                let bits := floatToF16CanonicalZero f
                if !f16IsFinite bits then
                  throw s!"{ctx} (reduce): eps value {f} overflows finite fp16 range"
                pure bits
            let rspec : ReduceSpec := {
              target := target, op := redOp, dims := inDims, axes := axes, eps := eps
            }
            let (_, outDims) ← resolveReduceBackend rspec
            pure (NodeOp.reduce redOp axes eps, outDims)) with
          | .ok v => pure v
          | .error e => IO.eprintln s!"error: {e}"; return 1
        pure parsed

      | "transpose" =>
        let parsed ← match (do
            if ndInputs.size != 1 then throw s!"{ctx} (transpose): expected 1 input"
            let some inDims := lookupShape ndInputs[0]!
              | throw s!"{ctx}: unknown input '{ndInputs[0]!}'"
            let perm ← jsonReqNatArray nj "perm" ctx
            let vp ← planTranspose { n := 1, c := inDims.c, h := inDims.h, w := inDims.w } perm
            if vp.outShape.n != 1 then
              throw s!"{ctx} (transpose): graph output requires N=1, got {vp.outShape.n}"
            pure (NodeOp.transpose perm, { c := vp.outShape.c, h := vp.outShape.h, w := vp.outShape.w : TensorDims })) with
          | .ok v => pure v
          | .error e => IO.eprintln s!"error: {e}"; return 1
        pure parsed

      | "slice" =>
        let parsed ← match (do
            if ndInputs.size != 1 then throw s!"{ctx} (slice): expected 1 input"
            let some inDims := lookupShape ndInputs[0]!
              | throw s!"{ctx}: unknown input '{ndInputs[0]!}'"
            let beg ← jsonReqNatArray nj "begin" ctx
            let en ← jsonReqNatArray nj "end" ctx
            let st := (← jsonOptNatArray nj "step" ctx).getD #[1, 1, 1, 1]
            let vp ← planSlice { n := 1, c := inDims.c, h := inDims.h, w := inDims.w } beg en st
            if vp.outShape.n != 1 then
              throw s!"{ctx} (slice): graph output requires N=1, got {vp.outShape.n}"
            pure (NodeOp.slice beg en st, { c := vp.outShape.c, h := vp.outShape.h, w := vp.outShape.w : TensorDims })) with
          | .ok v => pure v
          | .error e => IO.eprintln s!"error: {e}"; return 1
        pure parsed

      | "concat" =>
        let parsed ← match (do
            if ndInputs.size < 2 then throw s!"{ctx} (concat): expected >= 2 inputs"
            let ax ← match ← jsonOptNat nj "axis" ctx with
              | some v => pure v
              | none => throw s!"{ctx} (concat): missing required field 'axis'"
            let mut inShapes4 : Array Shape4D := Array.mkEmpty ndInputs.size
            for inpNm in ndInputs do
              let some d := lookupShape inpNm
                | throw s!"{ctx} (concat): unknown input '{inpNm}'"
              inShapes4 := inShapes4.push { n := 1, c := d.c, h := d.h, w := d.w }
            let vp ← planConcat inShapes4 ax
            pure (NodeOp.concat ax, { c := vp.outShape.c, h := vp.outShape.h, w := vp.outShape.w : TensorDims })) with
          | .ok v => pure v
          | .error e => IO.eprintln s!"error: {e}"; return 1
        pure parsed

      | "reshape" =>
        let parsed ← match (do
            if ndInputs.size != 1 then throw s!"{ctx} (reshape): expected 1 input"
            let some inDims := lookupShape ndInputs[0]!
              | throw s!"{ctx}: unknown input '{ndInputs[0]!}'"
            let oArr ← jsonReqNatArray nj "out_shape" ctx
            let targetDims : TensorDims ←
              if oArr.size == 4 then
                if oArr[0]! != 1 then
                  throw s!"{ctx} (reshape): graph output requires N=1, got {oArr[0]!}"
                pure { c := oArr[1]!, h := oArr[2]!, w := oArr[3]! }
              else if oArr.size == 3 then
                pure { c := oArr[0]!, h := oArr[1]!, w := oArr[2]! }
              else
                throw s!"{ctx} (reshape): out_shape must have length 4 ([1,C,H,W]) or 3 ([C,H,W]), got {oArr.size}"
            let vp ← planReshape { n := 1, c := inDims.c, h := inDims.h, w := inDims.w }
              { n := 1, c := targetDims.c, h := targetDims.h, w := targetDims.w }
            pure (NodeOp.reshape targetDims, { c := vp.outShape.c, h := vp.outShape.h, w := vp.outShape.w : TensorDims })) with
          | .ok v => pure v
          | .error e => IO.eprintln s!"error: {e}"; return 1
        pure parsed

      | "matmul" =>
        let parsed ← match (do
            if ndInputs.size != 2 then throw s!"{ctx} (matmul): expected 2 inputs [a, b]"
            let some da := lookupShape ndInputs[0]!
              | throw s!"{ctx} (matmul): unknown input '{ndInputs[0]!}'"
            let some db := lookupShape ndInputs[1]!
              | throw s!"{ctx} (matmul): unknown input '{ndInputs[1]!}'"
            let heads := (← jsonOptPosNat nj "heads" ctx).getD 1
            let trA ← jsonOptBool nj "transpose_a" ctx
            let trB ← jsonOptBool nj "transpose_b" ctx
            let scaleBits ← match ← jsonOptFloat nj "scale" ctx with
              | none => pure f16PosOne
              | some f =>
                let bits := floatToF16CanonicalZero f
                if !f16IsFinite bits then
                  throw s!"{ctx} (matmul): scale value {f} overflows finite fp16 range"
                pure bits
            if da.h != 1 || db.h != 1 then
              throw s!"{ctx} (matmul): requires H=1 for both inputs (got A.h={da.h}, B.h={db.h})"
            if da.c % heads != 0 || db.c % heads != 0 then
              throw s!"{ctx} (matmul): input channels ({da.c}, {db.c}) not divisible by heads={heads}"
            let (m, ka) := if trA then (da.w, da.c / heads) else (da.c / heads, da.w)
            let (kb, n) := if trB then (db.w, db.c / heads) else (db.c / heads, db.w)
            if ka != kb then
              throw s!"{ctx} (matmul): inner dimension mismatch A.K={ka} != B.K={kb}"
            validateMatmulSpec {
              b := heads, m := m, k := ka, n := n,
              transposeA := trA, transposeB := trB, scaleBits := scaleBits
            }
            pure (NodeOp.matmul heads trA trB scaleBits, { c := heads * m, h := 1, w := n : TensorDims })) with
          | .ok v => pure v
          | .error e => IO.eprintln s!"error: {e}"; return 1
        pure parsed

      | other =>
        IO.eprintln s!"error: {ctx}: unsupported node op '{other}' (expected conv1x1, linear, conv3x3, pe, lut, max_pool, avg_pool, reduce, transpose, slice, concat, reshape, or matmul)"
        return 1

    graphNodes := graphNodes.push {
      id := ndId
      inputs := ndInputs
      op := nodeOp
      outPlacement := ndOutPlace?
    }
    knownShapes := knownShapes.push (ndId, inferredDims)

  let graphSpec : GraphSpec := {
    target := target
    inputs := graphInputs
    nodes := graphNodes
    defaultPlacement := defaultPlacement
    stripMode := stripMode
  }

  let compiled ← match compileGraph graphSpec with
    | .ok c => pure c
    | .error e => IO.eprintln s!"error: compile-graph failed: {e}"; return 1

  let manifest := renderGraphManifestJson compiled
  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "program.hwx") compiled.hwx
    IO.FS.writeBinFile (dir / "generic.bin") compiled.sections.generic
    IO.FS.writeBinFile (dir / "operation.bin") compiled.sections.operation
    IO.FS.writeBinFile (dir / "procedure.bin") compiled.sections.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") compiled.sections.textProperties
    IO.FS.writeFile (dir / "manifest.json") manifest
  catch e =>
    IO.eprintln s!"error: failed to write graph outputs to '{dirStr}': {e}"
    return 1

  let stripInfo :=
    if compiled.stripCount > 1 || compiled.stripHeight > 0 then
      s!", strip_count={compiled.stripCount}, strip_height={compiled.stripHeight}"
    else ""
  IO.println (
    s!"compiled {target.name} graph ({compiled.taskCount} tasks, placement={defaultPlacement.name}{stripInfo}) " ++
    s!"to {dirStr}: hwx={compiled.hwx.size}B kernel={compiled.kernelBytes}B " ++
    s!"scratch={compiled.scratchBytes}B l2_peak={compiled.l2PeakBytes}B"
  )
  return 0

private def cmdCompileOnnx (args : List String) : IO UInt32 := do
  let mut targetStr : Option String := none
  let mut modelPath : Option String := none
  let mut sfPath : Option String := none
  let mut inputShapes : Array (String × TensorDims) := #[]
  let mut placement : EdgePlacementMode := .auto
  let mut stripMode : StripMode := .off
  let mut noFuse : Bool := false
  let mut allowViewOps : Bool := true
  let mut quantizeFmt : Option QuantFormat := none
  let mut dumpIrMode : Option String := none
  let mut emitGraphDirPath : Option String := none
  let mut outDir : Option String := none
  let mut rest := args

  while !rest.isEmpty do
    match rest with
    | "--target" :: tl =>
      match requireArgValue "--target" tl with
      | .ok (v, tl') => targetStr := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--model" :: tl =>
      match requireArgValue "--model" tl with
      | .ok (v, tl') => modelPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--safetensors" :: tl =>
      match requireArgValue "--safetensors" tl with
      | .ok (v, tl') => sfPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--input-shape" :: tl =>
      match requireArgValue "--input-shape" tl with
      | .ok (v, tl') =>
        match parseInputShapeOverride v with
        | .ok entry => inputShapes := inputShapes.push entry; rest := tl'
        | .error e => IO.eprintln s!"error: {e}"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--placement" :: tl =>
      match requireArgValue "--placement" tl with
      | .ok (v, tl') =>
        match EdgePlacementMode.parse? v with
        | some p => placement := p; rest := tl'
        | none => IO.eprintln s!"error: unknown placement '{v}' (expected auto, l2, or dram)"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--strip-height" :: tl =>
      match requireArgValue "--strip-height" tl with
      | .ok (v, tl') =>
        match StripMode.parse? v with
        | some m => stripMode := m; rest := tl'
        | none => IO.eprintln s!"error: invalid --strip-height '{v}' (expected off, auto, or positive integer)"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--no-fuse" :: tl =>
      noFuse := true; rest := tl
    | "--quantize" :: tl =>
      match requireArgValue "--quantize" tl with
      | .ok (v, tl') =>
        match Vane.IR.parsePTQFormat? v with
        | some fmt => quantizeFmt := some fmt; rest := tl'
        | none => IO.eprintln s!"error: invalid --quantize '{v}' (expected int8, int4, pal4, or pal2)"; return 1
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--dump-ir" :: tl =>
      dumpIrMode := some "both"; rest := tl
    | "--dump-ir=high" :: tl =>
      dumpIrMode := some "high"; rest := tl
    | "--dump-ir=core" :: tl =>
      dumpIrMode := some "core"; rest := tl
    | "--allow-view-ops=true" :: tl =>
      allowViewOps := true; rest := tl
    | "--allow-view-ops=false" :: tl =>
      allowViewOps := false; rest := tl
    | "--allow-view-ops" :: tl =>
      match tl with
      | v :: tl' =>
        if v.toLower == "true" || v.toLower == "false" || v == "1" || v == "0" then
          match parseBoolArg "--allow-view-ops" v with
          | .ok b => allowViewOps := b; rest := tl'
          | .error e => IO.eprintln s!"error: {e}"; return 1
        else
          allowViewOps := true; rest := tl
      | [] =>
        allowViewOps := true; rest := []
    | "--emit-graph-dir" :: tl =>
      match requireArgValue "--emit-graph-dir" tl with
      | .ok (v, tl') => emitGraphDirPath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--out-dir" :: tl =>
      match requireArgValue "--out-dir" tl with
      | .ok (v, tl') => outDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | flag :: _ =>
      if flag.startsWith "--allow-view-ops=" then
        let val := (flag.drop 17).toString
        match parseBoolArg "--allow-view-ops" val with
        | .ok b => allowViewOps := b; rest := rest.tail
        | .error e => IO.eprintln s!"error: {e}"; return 1
      else if flag.startsWith "--dump-ir=" then
        let val := (flag.drop 10).toString.toLower
        if val == "high" || val == "core" then
          dumpIrMode := some val; rest := rest.tail
        else
          IO.eprintln s!"error: invalid --dump-ir '{val}' (expected high or core)"; return 1
      else if flag.startsWith "--quantize=" then
        let val := (flag.drop 11).toString
        match Vane.IR.parsePTQFormat? val with
        | some fmt => quantizeFmt := some fmt; rest := rest.tail
        | none => IO.eprintln s!"error: invalid --quantize '{val}' (expected int8, int4, pal4, or pal2)"; return 1
      else
        IO.eprintln s!"error: unknown argument '{flag}' for compile-onnx"
        IO.eprintln usage
        return 1
    | [] => break

  let some tStr := targetStr
    | do IO.eprintln "error: --target <h16|h17> is required"; return 1
  let some target := Target.parse? tStr
    | do IO.eprintln s!"error: unknown target '{tStr}' (expected h16 or h17)"; return 1
  let some mPath := modelPath
    | do IO.eprintln "error: --model <path> is required"; return 1
  let some dirStr := outDir
    | do IO.eprintln "error: --out-dir <dir> is required"; return 1

  let sf? ← match sfPath with
    | none => pure none
    | some sp =>
      match ← loadSafeTensors (System.FilePath.mk sp) with
      | .ok sf => pure (some sf)
      | .error e => IO.eprintln s!"error: {e}"; return 1

  let model ← match ← loadOnnx (System.FilePath.mk mPath) with
    | .ok m => pure m
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let lowered ← match lowerOnnx model {
      target := target
      placement := placement
      stripMode := stripMode
      inputShapes := inputShapes
      safeTensors := sf?
      noFuse := noFuse
      allowViewOps := allowViewOps
      quantize := quantizeFmt
    } with
    | .ok r => pure r
    | .error e => IO.eprintln s!"error: compile-onnx lowering failed: {e}"; return 1

  if let some mode := dumpIrMode then
    if mode == "high" || mode == "both" then
      IO.println "=== TensorIR (HighOp) ==="
      IO.println (Vane.IR.SSAGraph.dumpHigh lowered.highGraph)
    if mode == "core" || mode == "both" then
      IO.println "=== TensorIR (CoreOp) ==="
      IO.println (Vane.IR.SSAGraph.dumpCore lowered.coreGraph)

  for w in lowered.summary.warnings do
    IO.eprintln w

  if let some gDir := emitGraphDirPath then
    match ← emitGraphDir (System.FilePath.mk gDir) lowered with
    | .ok () => pure ()
    | .error e => IO.eprintln s!"error: --emit-graph-dir failed: {e}"; return 1

  let compiled ← match compileGraph lowered.spec with
    | .ok c => pure c
    | .error e => IO.eprintln s!"error: compile-onnx compilation failed: {e}"; return 1

  let manifest := renderGraphManifestJson compiled
  let dir := System.FilePath.mk dirStr
  try
    IO.FS.createDirAll dir
    IO.FS.writeBinFile (dir / "program.hwx") compiled.hwx
    IO.FS.writeBinFile (dir / "generic.bin") compiled.sections.generic
    IO.FS.writeBinFile (dir / "operation.bin") compiled.sections.operation
    IO.FS.writeBinFile (dir / "procedure.bin") compiled.sections.procedure
    IO.FS.writeBinFile (dir / "text-properties.bin") compiled.sections.textProperties
    IO.FS.writeFile (dir / "manifest.json") manifest
  catch e =>
    IO.eprintln s!"error: failed to write compiled outputs to '{dirStr}': {e}"
    return 1

  let sm := lowered.summary
  let outS := sm.outputShape
  IO.println (
    s!"compiled {target.name} onnx ({compiled.taskCount} tasks, nodes={sm.nodeCount}, " ++
    s!"fused_bn={sm.fusedBnCount}, fused_act={sm.fusedActCount}, fused_mm={sm.fusedMatmulCount}, " ++
    s!"elided={sm.elidedCount}, out=[{outS.n},{outS.c},{outS.h},{outS.w}]) to {dirStr}: " ++
    s!"hwx={compiled.hwx.size}B kernel={compiled.kernelBytes}B " ++
    s!"scratch={compiled.scratchBytes}B l2_peak={compiled.l2PeakBytes}B"
  )
  return 0

private def cmdInspectSafeTensors (args : List String) : IO UInt32 := do
  let mut filePath : Option String := none
  let mut exportDir : Option String := none
  let mut strict : Bool := false
  let mut asJson : Bool := false
  let mut rest := args

  while !rest.isEmpty do
    match rest with
    | "--file" :: tl =>
      match requireArgValue "--file" tl with
      | .ok (v, tl') => filePath := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--export-dir" :: tl =>
      match requireArgValue "--export-dir" tl with
      | .ok (v, tl') => exportDir := some v; rest := tl'
      | .error e => IO.eprintln s!"error: {e}"; return 1
    | "--strict" :: tl =>
      strict := true; rest := tl
    | "--json" :: tl =>
      asJson := true; rest := tl
    | flag :: _ =>
      IO.eprintln s!"error: unknown argument '{flag}' for inspect-safetensors"
      IO.eprintln usage
      return 1
    | [] => break

  let some fPath := filePath
    | do IO.eprintln "error: --file <path> is required"; return 1

  let sf ← match ← loadSafeTensors (System.FilePath.mk fPath) { strict := strict } with
    | .ok s => pure s
    | .error e => IO.eprintln s!"error: {e}"; return 1

  let exported ← match exportDir with
    | none => pure #[]
    | some eDir =>
      match ← sf.exportFp16Dir (System.FilePath.mk eDir) with
      | .ok ex => pure ex
      | .error e => IO.eprintln s!"error: {e}"; return 1

  let totalBytes : Nat := sf.tensors.foldl (fun acc t => acc + t.byteLen) 0

  if asJson then
    let tensorJsons := sf.tensors.map fun t =>
      Json.mkObj [
        ("name", Json.str t.name),
        ("dtype", Json.str t.dtype),
        ("shape", Json.arr (t.shape.map fun (d : Nat) => Json.num d)),
        ("data_offsets", Json.arr #[Json.num t.beginOff, Json.num t.endOff]),
        ("byte_len", Json.num t.byteLen)
      ]
    let metaFields := if sf.metadata.isEmpty then [] else
      [("metadata", Json.mkObj (sf.metadata.toList.map fun (k, v) => (k, Json.str v)))]
    let exportFields := if exported.isEmpty then [] else
      [("exported", Json.arr (exported.map fun e => Json.mkObj [
        ("name", Json.str e.name),
        ("dtype", Json.str e.dtype),
        ("shape", Json.arr (e.shape.map fun (d : Nat) => Json.num d)),
        ("file", Json.str e.file),
        ("out_bytes", Json.num e.outBytes)
      ]))]
    let rep := Json.mkObj ([
      ("file", Json.str fPath),
      ("tensor_count", Json.num sf.tensors.size),
      ("total_bytes", Json.num totalBytes)
    ] ++ metaFields ++ [("tensors", Json.arr tensorJsons)] ++ exportFields)
    IO.println rep.pretty
    return 0

  IO.println s!"safetensors: {fPath} ({sf.tensors.size} tensors, {totalBytes} bytes)"
  for info in sf.tensors do
    IO.println s!"  {info.name} dtype={info.dtype} shape={info.shape.toList} bytes={info.byteLen}"
  if !exported.isEmpty then
    IO.println s!"exported {exported.size} fp16 tensor files to {exportDir.getD ""}"
  return 0

def main (args : List String) : IO UInt32 := do
  match args with
  | "validate" :: rest => cmdValidate rest
  | "sections" :: rest => cmdSections rest
  | "emit-demo" :: rest => cmdEmitDemo rest
  | "compile-conv1x1" :: rest => cmdCompileConv1x1 rest
  | "compile-linear" :: rest => cmdCompileLinear rest
  | "compile-conv3x3" :: rest => cmdCompileConv3x3 rest
  | "compile-pe" :: rest => cmdCompilePe rest
  | "compile-lut" :: rest => cmdCompileLut rest
  | "compile-pool" :: rest => cmdCompilePool rest
  | "compile-reduce" :: rest => cmdCompileReduce rest
  | "compile-view" :: rest => cmdCompileView rest
  | "compile-matmul" :: rest => cmdCompileMatmul rest
  | "compile-graph" :: rest => cmdCompileGraph rest
  | "compile-onnx" :: rest => cmdCompileOnnx rest
  | "inspect-safetensors" :: rest => cmdInspectSafeTensors rest
  | "--help" :: _ | "-h" :: _ | [] =>
    IO.println usage
    return 0
  | cmd :: _ =>
    IO.eprintln s!"error: unknown subcommand '{cmd}'"
    IO.eprintln usage
    return 1
