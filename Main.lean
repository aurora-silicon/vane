import Vane

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections

private def usage : String :=
  "Usage:\n" ++
  "  vane validate --hwx <path> [--sections <dir>]\n" ++
  "  vane sections --hwx <path> --out-dir <dir>\n" ++
  "  vane emit-demo --target <h16|h17> --out-dir <dir> [--with-kernel] [--with-scratch]\n"

private def requireArgValue (flag : String) (rest : List String) : Except String (String × List String) :=
  match rest with
  | v :: tl => pure (v, tl)
  | [] => throw s!"missing value for {flag}"

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
  -- Two surfaces: input x0 (slot 6), output y0 (slot 7).
  -- If withKernel, __KERN_0 lands at slot 8. If withScratch, __DATA lands at slot 3.
  let sIn := SurfaceSpec.packed "in0" "input_0" .input 1 16 8 8
  let sOut := SurfaceSpec.packed "out0" "output_0" .output 1 16 8 8

  let mut recs : Array RegRecord := #[
    -- Sequential record referencing input surface BAR slot 6 (block 0: word 0x0000..0x0003)
    .sequential 0x0000 (some 6) #[0x00000001, 0x00000002],
    -- Sequential record referencing output surface BAR slot 7 (block 1: word 0x1040..0x1041)
    .sequential 0x1040 (some 7) #[0x00000010, 0x00000020],
    -- Bitmask record (block 3: word 0x1140 + mask 0b101 => 0x1140, 0x1141, 0x1143)
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
    -- Sequential record referencing kernel BAR slot 8 (block 5: word 0x1340), payload word 0 = addend 0x40
    recs := recs.push (.sequential 0x1340 (some 8) #[0x00000040, 0x00000000])

  let mut scratchSpec : Option ScratchSpec := none
  if withScratch then
    scratchSpec := some { size := 0x4000, usable := 0x2000 }
    -- Sequential record referencing scratch BAR slot 3 (block 4: word 0x1240)
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

private def renderManifestJson (info : HwxInfo) (secs : LoadSections) : String :=
  let surfaceItems := info.surfaces.toList.map fun s =>
    "    {\n" ++
    s!"      \"short_name\": \"{escapeJsonString s.shortName}\",\n" ++
    s!"      \"symbol_name\": \"{escapeJsonString s.symbolName}\",\n" ++
    s!"      \"direction\": \"{s.direction.name}\",\n" ++
    s!"      \"shape\": [{s.n}, {s.channels}, {s.height}, {s.width}],\n" ++
    s!"      \"strides\": [{s.batchStride}, {s.planeStride}, {s.rowStride}],\n" ++
    s!"      \"span\": {s.span},\n" ++
    s!"      \"alloc_size\": {s.vmSize}\n" ++
    "    }"
  let surfacesJson := String.intercalate ",\n" surfaceItems
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
  surfacesJson ++ "\n" ++
  "  ]\n" ++
  "}\n"

private def cmdValidate (args : List String) : IO UInt32 := do
  let mut hwxPath : Option String := none
  let mut sectionsDir : Option String := none
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
  IO.println s!"bar_refs: {formatBarSlots info.taskStream.barSlots}"
  IO.println s!"kernel: {match info.kernelSection with | some k => s!"{k.size}B" | none => "none"}, scratch: {match info.scratchSection with | some sc => s!"{sc.size}B" | none => "none"}"
  if let some summary := sectionSummary then
    IO.println summary
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

def main (args : List String) : IO UInt32 := do
  match args with
  | "validate" :: rest => cmdValidate rest
  | "sections" :: rest => cmdSections rest
  | "emit-demo" :: rest => cmdEmitDemo rest
  | "--help" :: _ | "-h" :: _ | [] =>
    IO.println usage
    return 0
  | cmd :: _ =>
    IO.eprintln s!"error: unknown subcommand '{cmd}'"
    IO.eprintln usage
    return 1
