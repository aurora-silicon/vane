import Vane

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Tiling
open Vane.Verify
open Vane.Pe
open Vane.Lut
open Vane.Conv
open Vane.Conv1x1
open Vane.Conv3x3

private def usage : String :=
  "Usage:\n" ++
  "  vane validate --hwx <path> [--sections <dir>] [--check-tasks]\n" ++
  "  vane sections --hwx <path> --out-dir <dir>\n" ++
  "  vane emit-demo --target <h16|h17> --out-dir <dir> [--with-kernel] [--with-scratch]\n" ++
  "  vane compile-conv1x1 --target <h16|h17> --h <H> --w <W> --cin <Cin> --cout <Cout>\n" ++
  "                       --weights <path> [--bias <path>] [--relu | --act <name> [act params]] --out-dir <dir>\n" ++
  "  vane compile-conv3x3 --target <h16|h17> --mode <dense|depthwise>\n" ++
  "                       --h <H> --w <W> --cin <Cin> --cout <Cout>\n" ++
  "                       --stride <1|2> --pad <same|valid>\n" ++
  "                       --weights <path> [--bias <path>] [--relu | --act <name> [act params]] --out-dir <dir>\n" ++
  "  vane compile-pe --target <h16|h17> --op <abs|square|linear|chain|sadd|ssub|srsub|smul|sdiv|add|sub|rsub|mul|max|min>\n" ++
  "                  --h <H> --w <W> --c <C> [--z-h <ZH> --z-w <ZW> --z-c <ZC>]\n" ++
  "                  [--alpha F] [--beta F] [--in-scale F] [--bias F] [--scale F] [--scalar F]\n" ++
  "                  --out-dir <dir>\n" ++
  "  vane compile-lut --target <h16|h17> --act <relu|copy|relu6|clip|leaky_relu|abs|sign|threshold|thresholded_relu|hard_sigmoid|clamp_max|clamp_min|elu|tanh|scaled_tanh|erf|sigmoid|gelu|silu|exp2|exp|sqrt|rsqrt|inverse|log2>\n" ++
  "                   --h <H> --w <W> --c <C>\n" ++
  "                   [--alpha F] [--beta F] [--lo F] [--hi F] [--eps F] [--scalar F]\n" ++
  "                   --out-dir <dir>\n"

private def requireArgValue (flag : String) (rest : List String) : Except String (String × List String) :=
  match rest with
  | v :: tl => pure (v, tl)
  | [] => throw s!"missing value for {flag}"

private def parsePosNat (flag s : String) : Except String Nat :=
  match s.toNat? with
  | some n =>
    if n == 0 then throw s!"{flag} must be >= 1, got 0" else pure n
  | none => throw s!"invalid positive integer '{s}' for {flag}"

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

private def renderConv1x1ManifestJson (compiled : CompiledConv1x1) : String :=
  let info := compiled.info
  let secs := compiled.sections
  let sh := compiled.shape
  let t := compiled.tiling
  let lutLine := if compiled.lutBytes.isSome then "  \"lut_file\": \"lut.bin\",\n" else ""
  "{\n" ++
  "  \"op\": \"conv1x1\",\n" ++
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

private def renderConv3x3ManifestJson (compiled : CompiledConv3x3) : String :=
  let info := compiled.info
  let secs := compiled.sections
  let sh := compiled.shape
  let t := compiled.tiling
  let lutLine := if compiled.lutBytes.isSome then "  \"lut_file\": \"lut.bin\",\n" else ""
  "{\n" ++
  "  \"op\": \"conv3x3\",\n" ++
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
    IO.println s!"verified_tasks: {summary.total} (conv={summary.neConv}, ne_act={summary.neAct}, pe={summary.peOp})"
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

  let compiled ← match compileConv1x1 target shape weightsBytes biasBytes? false none act with
    | .ok c => pure c
    | .error e =>
      IO.eprintln s!"error: compile-conv1x1 failed: {e}"
      return 1

  let manifest := renderConv1x1ManifestJson compiled
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
    s!"compiled {target.name} conv1x1 ({shape.h}x{shape.w}, {shape.cin}->{shape.cout}, " ++
    s!"bias={compiled.hasBias}, act={compiled.activation.name}) to {dirStr}: " ++
    s!"hwx={compiled.hwx.size}B kernel={compiled.kernelBytes}B l2={compiled.l2UseBytes}B " ++
    s!"direct={compiled.direct} tiling=(logNE={t.logNE},F={t.f},fat={t.fat},ocg={t.ocg},WU={t.wu},sh={t.shPref}..{t.shMax},TH={t.tileHeight})"
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

  let compiled ← match compileConv3x3 target shape weightsBytes biasBytes? false none act with
    | .ok c => pure c
    | .error e =>
      IO.eprintln s!"error: compile-conv3x3 failed: {e}"
      return 1

  let manifest := renderConv3x3ManifestJson compiled
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

def main (args : List String) : IO UInt32 := do
  match args with
  | "validate" :: rest => cmdValidate rest
  | "sections" :: rest => cmdSections rest
  | "emit-demo" :: rest => cmdEmitDemo rest
  | "compile-conv1x1" :: rest => cmdCompileConv1x1 rest
  | "compile-conv3x3" :: rest => cmdCompileConv3x3 rest
  | "compile-pe" :: rest => cmdCompilePe rest
  | "compile-lut" :: rest => cmdCompileLut rest
  | "--help" :: _ | "-h" :: _ | [] =>
    IO.println usage
    return 0
  | cmd :: _ =>
    IO.eprintln s!"error: unknown subcommand '{cmd}'"
    IO.eprintln usage
    return 1
