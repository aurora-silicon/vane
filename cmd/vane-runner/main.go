// Command vane-runner provides Linux accel/ane multi-ABI device probing, ABI v3 HWX
// and ABI v1 .anec execution, deterministic conv/PE/LUT test vector generation, and
// strided surface comparison.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/aurora-silicon/vane/internal/ane"
	"github.com/aurora-silicon/vane/internal/fp16"
	"github.com/aurora-silicon/vane/internal/ref"
)

type stringSliceFlag []string

func (s *stringSliceFlag) String() string {
	return strings.Join(*s, ",")
}

func (s *stringSliceFlag) Set(v string) error {
	*s = append(*s, v)
	return nil
}

type optFloatFlag struct {
	set bool
	val float64
}

func (f *optFloatFlag) String() string {
	if !f.set {
		return ""
	}
	return strconv.FormatFloat(f.val, 'g', -1, 64)
}

func (f *optFloatFlag) Set(s string) error {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return err
	}
	f.set = true
	f.val = v
	return nil
}

func usage(w io.Writer) {
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  vane-runner probe [--abi N] [--device PATH] [--json]")
	fmt.Fprintln(w, "  vane-runner run --hwx PATH [--sections-dir DIR] [--input PATH]... [--alt-input PATH]... [--output-out PATH]")
	fmt.Fprintln(w, "                  [--expected PATH] [--runs N] [--warmup N] [--in-place] [--atol F] [--rtol F]")
	fmt.Fprintln(w, "                  [--max-ulp N] [--device PATH] [--json] [--json-out PATH]")
	fmt.Fprintln(w, "  vane-runner run-anec --anec PATH [--input PATH]... [--alt-input PATH]... [--output-out PATH]")
	fmt.Fprintln(w, "                       [--expected PATH] [--runs N] [--warmup N] [--atol F] [--rtol F]")
	fmt.Fprintln(w, "                       [--max-ulp N] [--device PATH] [--json] [--json-out PATH]")
	fmt.Fprintln(w, "  vane-runner gen-conv1x1 --h H --w W --cin C --cout O [--bias] [--relu | --act NAME [act flags]] [--lut-bin PATH] [--seed N] --out-dir DIR [--json]")
	fmt.Fprintln(w, "  vane-runner gen-conv3x3 --mode <dense|depthwise> --h H --w W --cin C --cout O --stride <1|2> --pad <same|valid> [--bias] [--relu | --act NAME [act flags]] [--lut-bin PATH] [--seed N] --out-dir DIR [--json]")
	fmt.Fprintln(w, "  vane-runner gen-pe --op <abs|square|linear|chain|sadd|ssub|srsub|smul|sdiv|add|sub|rsub|mul|max|min> --h H --w W --c C [--z-h ZH --z-w ZW --z-c ZC] [--alpha F] [--beta F] [--in-scale F] [--bias F] [--scale F] [--scalar F] [--edge-values] [--seed N] --out-dir DIR [--json]")
	fmt.Fprintln(w, "  vane-runner gen-lut --act NAME --h H --w W --c C [--alpha F] [--beta F] [--lo F] [--hi F] [--eps F] [--scalar F] [--all-fp16] [--lut-bin PATH] [--seed N] --out-dir DIR [--json]")
	fmt.Fprintln(w, "  vane-runner gen-pool --op <max|avg> --h H --w W --c C --kh KH --kw KW [--sh SH] [--sw SW] [--pad <valid|same|custom>] [--pt N --pb N --pl N --pr N] [--exclude-pad] [--seed N] --out-dir DIR [--json]")
	fmt.Fprintln(w, "  vane-runner gen-reduce --op <sum|mean|max|min|l1|sum_square|l2|inv_sum|rsqrt_sum_square|rsqrt_mean_square> --h H --w W --c C --axes <w|h|c|hw|cw|ch|chw> [--eps F] [--seed N] --out-dir DIR [--json]")
	fmt.Fprintln(w, "  vane-runner gen-graph --spec PATH --out-dir DIR [--seed N] [--json]")
	fmt.Fprintln(w, "  vane-runner compare-conv --manifest PATH --actual PATH --expected PATH [--atol F] [--rtol F] [--max-ulp N] [--json]")
	fmt.Fprintln(w, "  vane-runner compare-conv1x1 --manifest PATH --actual PATH --expected PATH [--atol F] [--rtol F] [--max-ulp N] [--json]")
	fmt.Fprintln(w, "  vane-runner compare-surface --manifest PATH --actual PATH --expected PATH [--atol F] [--rtol F] [--max-ulp N] [--json]")
}

func runCLI(args []string, tr ane.Transport, stdout, stderr io.Writer) int {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		usage(stdout)
		return 0
	}
	switch args[0] {
	case "probe":
		return cmdProbe(args[1:], tr, stdout, stderr)
	case "run":
		return cmdRun(args[1:], tr, stdout, stderr)
	case "run-anec":
		return cmdRunAnec(args[1:], tr, stdout, stderr)
	case "gen-conv1x1":
		return cmdGenConv1x1(args[1:], stdout, stderr)
	case "gen-conv3x3":
		return cmdGenConv3x3(args[1:], stdout, stderr)
	case "gen-pe":
		return cmdGenPe(args[1:], stdout, stderr)
	case "gen-lut":
		return cmdGenLut(args[1:], stdout, stderr)
	case "gen-pool":
		return cmdGenPool(args[1:], stdout, stderr)
	case "gen-reduce":
		return cmdGenReduce(args[1:], stdout, stderr)
	case "gen-graph":
		return cmdGenGraph(args[1:], stdout, stderr)
	case "compare-conv":
		return cmdCompareConv("compare-conv", args[1:], stdout, stderr)
	case "compare-conv1x1":
		return cmdCompareConv("compare-conv1x1", args[1:], stdout, stderr)
	case "compare-surface":
		return cmdCompareConv("compare-surface", args[1:], stdout, stderr)
	default:
		fmt.Fprintf(stderr, "error: unknown subcommand %q\n", args[0])
		usage(stderr)
		return 1
	}
}

func buildActSpec(name string, alpha, beta, lo, hi, eps, scalar optFloatFlag) (ref.ActSpec, error) {
	low := strings.ToLower(name)
	spec := ref.ActSpec{Name: low}
	switch low {
	case "", "none", "relu", "copy", "relu6", "abs", "sign", "tanh", "erf", "sigmoid", "gelu", "silu", "exp2", "exp", "sqrt", "inverse", "log2":
		return spec, nil
	case "clip":
		if !lo.set || !hi.set {
			return spec, fmt.Errorf("clip requires both --lo/--act-lo and --hi/--act-hi")
		}
		spec.Lo, spec.Hi = lo.val, hi.val
		return spec, nil
	case "leaky_relu":
		spec.Alpha = 0.01
		if alpha.set {
			spec.Alpha = alpha.val
		}
		return spec, nil
	case "threshold":
		if scalar.set {
			spec.Scalar = scalar.val
		} else if alpha.set {
			spec.Scalar = alpha.val
		} else if lo.set {
			spec.Scalar = lo.val
		}
		return spec, nil
	case "thresholded_relu":
		spec.Alpha = 1.0
		if alpha.set {
			spec.Alpha = alpha.val
		} else if scalar.set {
			spec.Alpha = scalar.val
		} else if lo.set {
			spec.Alpha = lo.val
		}
		return spec, nil
	case "hard_sigmoid":
		spec.Alpha, spec.Beta = 0.2, 0.5
		if alpha.set {
			spec.Alpha = alpha.val
		}
		if beta.set {
			spec.Beta = beta.val
		}
		return spec, nil
	case "clamp_max":
		if scalar.set {
			spec.Scalar = scalar.val
		} else if lo.set {
			spec.Scalar = lo.val
		} else if alpha.set {
			spec.Scalar = alpha.val
		}
		return spec, nil
	case "clamp_min":
		if scalar.set {
			spec.Scalar = scalar.val
		} else if hi.set {
			spec.Scalar = hi.val
		} else if alpha.set {
			spec.Scalar = alpha.val
		}
		return spec, nil
	case "elu":
		spec.Alpha = 1.0
		if alpha.set {
			spec.Alpha = alpha.val
		}
		return spec, nil
	case "scaled_tanh":
		spec.Alpha, spec.Beta = 1.0, 1.0
		if alpha.set {
			spec.Alpha = alpha.val
		}
		if beta.set {
			spec.Beta = beta.val
		}
		return spec, nil
	case "rsqrt":
		if eps.set {
			spec.Eps = eps.val
		}
		return spec, nil
	default:
		return spec, fmt.Errorf("unknown activation %q", name)
	}
}

func resolveConvActivation(relu bool, actName string, alpha, beta, lo, hi, eps, scalar optFloatFlag) (ref.ActSpec, bool, error) {
	low := strings.ToLower(actName)
	if low == "" {
		if relu {
			return ref.ActSpec{Name: "relu"}, false, nil
		}
		return ref.ActSpec{Name: "none"}, false, nil
	}
	if low == "none" {
		if relu {
			return ref.ActSpec{}, false, fmt.Errorf("cannot combine --relu with --act none")
		}
		return ref.ActSpec{Name: "none"}, false, nil
	}
	if low == "relu" {
		return ref.ActSpec{Name: "relu"}, false, nil
	}
	if relu {
		return ref.ActSpec{}, false, fmt.Errorf("cannot combine --relu with --act %s", actName)
	}
	spec, err := buildActSpec(low, alpha, beta, lo, hi, eps, scalar)
	if err != nil {
		return ref.ActSpec{}, false, err
	}
	return spec, true, nil
}

func loadOrBuildLut(spec ref.ActSpec, lutBinPath string) ([]byte, uint16, *uint16, error) {
	tableBytes, preScale, shift, err := ref.BuildLutTable(spec)
	if err != nil {
		return nil, 0, nil, err
	}
	if lutBinPath != "" {
		b, err := os.ReadFile(lutBinPath)
		if err != nil {
			return nil, 0, nil, fmt.Errorf("read --lut-bin %q: %w", lutBinPath, err)
		}
		if len(b) != 128 {
			return nil, 0, nil, fmt.Errorf("--lut-bin %q size %d != 128 bytes", lutBinPath, len(b))
		}
		tableBytes = b
	}
	return tableBytes, preScale, shift, nil
}

func cmdProbe(args []string, tr ane.Transport, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("probe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	abiFilter := fs.Uint("abi", 0, "Filter by ANE ABI version (0 = all, or 1/2/3)")
	device := fs.String("device", "", "Explicit /dev/accel/accel* device path")
	asJSON := fs.Bool("json", false, "Emit JSON output")
	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *abiFilter > 3 {
		fmt.Fprintf(stderr, "error: --abi must be 0 (all) or 1..3, got %d\n", *abiFilter)
		return 1
	}

	devs, err := ane.ProbeDevicesFilter(tr, *device, uint32(*abiFilter))
	if err != nil {
		if errors.Is(err, ane.ErrNoMatchingDevice) {
			fmt.Fprintf(stderr, "skip: %v\n", err)
			return ane.ExitSkip
		}
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if len(devs) == 0 {
		if *asJSON {
			fmt.Fprintln(stdout, "[]")
		}
		if *abiFilter != 0 {
			fmt.Fprintf(stderr, "skip: no matching accel/ane ABI v%d device found\n", *abiFilter)
		} else {
			fmt.Fprintln(stderr, "skip: no matching accel/ane device found")
		}
		return ane.ExitSkip
	}
	if *asJSON {
		enc, _ := json.MarshalIndent(devs, "", "  ")
		fmt.Fprintln(stdout, string(enc))
		return 0
	}
	for _, d := range devs {
		if d.SectionSize != 0 || d.BindSize != 0 || d.ExecIOSize != 0 {
			fmt.Fprintf(stdout, "device=%s abi=v%d chip=H%d caps_size=%d section_size=%d bind_size=%d exec_io_size=%d\n",
				d.Path, d.ABIVersion, d.ChipFamily, d.Size, d.SectionSize, d.BindSize, d.ExecIOSize)
		} else {
			fmt.Fprintf(stdout, "device=%s abi=v%d chip=H%d caps_size=%d\n",
				d.Path, d.ABIVersion, d.ChipFamily, d.Size)
		}
	}
	return 0
}

func cmdRun(args []string, tr ane.Transport, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	hwxPath := fs.String("hwx", "", "Path to HWX program file")
	sectionsDir := fs.String("sections-dir", "", "Directory containing the 4 host-built LOAD sections")
	var inputPaths stringSliceFlag
	fs.Var(&inputPaths, "input", "Input binary file (repeatable in HWX input surface order)")
	var altInputPaths stringSliceFlag
	fs.Var(&altInputPaths, "alt-input", "Alternate input binary file used on odd runs when --runs > 1 (repeatable)")
	outputOut := fs.String("output-out", "", "Write output surface bytes to this path")
	expectedPath := fs.String("expected", "", "Expected output fp16 binary for direct byte comparison")
	runs := fs.Int("runs", 1, "Number of consecutive timed runs for repeatability verification")
	warmup := fs.Int("warmup", 0, "Number of unmeasured warmup runs before timed --runs (checked for determinism)")
	inPlace := fs.Bool("in-place", false, "Use mmap in-place execution mode (requires 16 KiB pages)")
	atol := fs.Float64("atol", 1e-3, "Absolute tolerance for reference comparison")
	rtol := fs.Float64("rtol", 1e-3, "Relative tolerance for reference comparison")
	maxULPVal := fs.Int("max-ulp", -1, "Maximum allowed fp16 ULP error (negative disables)")
	device := fs.String("device", "", "Explicit /dev/accel/accel* device path")
	asJSON := fs.Bool("json", false, "Emit JSON report to stdout")
	jsonOut := fs.String("json-out", "", "Write JSON report to this path")

	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *hwxPath == "" {
		fmt.Fprintln(stderr, "error: --hwx PATH is required")
		return 1
	}
	if *runs < 1 {
		fmt.Fprintf(stderr, "error: --runs must be >= 1, got %d\n", *runs)
		return 1
	}
	if *warmup < 0 {
		fmt.Fprintf(stderr, "error: --warmup must be >= 0, got %d\n", *warmup)
		return 1
	}

	hwxBytes, err := os.ReadFile(*hwxPath)
	if err != nil {
		fmt.Fprintf(stderr, "error: read HWX %q: %v\n", *hwxPath, err)
		return 1
	}
	expectedChip, err := ane.ExpectedChipFromHWX(hwxBytes)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	var sections map[uint32][]byte
	if *sectionsDir != "" {
		sections, err = ane.LoadSectionsFromDir(*sectionsDir)
	} else {
		sections, err = ane.BuildSectionsFromHWX(hwxBytes)
	}
	if err != nil {
		fmt.Fprintf(stderr, "error: load/build sections: %v\n", err)
		return 1
	}

	var inputPayloads [][]byte
	for _, p := range inputPaths {
		b, err := os.ReadFile(p)
		if err != nil {
			fmt.Fprintf(stderr, "error: read input %q: %v\n", p, err)
			return 1
		}
		inputPayloads = append(inputPayloads, b)
	}

	var altInputPayloads [][]byte
	for _, p := range altInputPaths {
		b, err := os.ReadFile(p)
		if err != nil {
			fmt.Fprintf(stderr, "error: read alt-input %q: %v\n", p, err)
			return 1
		}
		altInputPayloads = append(altInputPayloads, b)
	}

	var expectedBytes []byte
	if *expectedPath != "" {
		expectedBytes, err = os.ReadFile(*expectedPath)
		if err != nil {
			fmt.Fprintf(stderr, "error: read expected %q: %v\n", *expectedPath, err)
			return 1
		}
	}

	var maxULP *int
	if *maxULPVal >= 0 {
		v := *maxULPVal
		maxULP = &v
	}

	fd, caps, err := ane.DiscoverDevice(tr, *device, expectedChip)
	if err != nil {
		if errors.Is(err, ane.ErrNoMatchingDevice) {
			fmt.Fprintf(stderr, "skip: %v\n", err)
			return ane.ExitSkip
		}
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	defer tr.Close(fd)

	loaded, err := ane.LoadProgram(tr, fd, hwxBytes, sections)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	outBytes, report, err := ane.ExecuteProgram(tr, fd, caps, loaded, inputPayloads, ane.ExecOptions{
		Runs:      *runs,
		Warmup:    *warmup,
		InPlace:   *inPlace,
		AltInputs: altInputPayloads,
		Expected:  expectedBytes,
		Atol:      *atol,
		Rtol:      *rtol,
		MaxULP:    maxULP,
	})
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	if *outputOut != "" {
		if err := os.MkdirAll(filepath.Dir(*outputOut), 0o755); err != nil {
			fmt.Fprintf(stderr, "error: create output dir: %v\n", err)
			return 1
		}
		if err := os.WriteFile(*outputOut, outBytes, 0o644); err != nil {
			fmt.Fprintf(stderr, "error: write output %q: %v\n", *outputOut, err)
			return 1
		}
	}

	if *jsonOut != "" {
		if err := os.MkdirAll(filepath.Dir(*jsonOut), 0o755); err != nil {
			fmt.Fprintf(stderr, "error: create json-out dir: %v\n", err)
			return 1
		}
		raw, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "error: marshal report: %v\n", err)
			return 1
		}
		if err := os.WriteFile(*jsonOut, append(raw, '\n'), 0o644); err != nil {
			fmt.Fprintf(stderr, "error: write json-out %q: %v\n", *jsonOut, err)
			return 1
		}
	}

	if *asJSON {
		raw, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "error: marshal report: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
	} else {
		fmt.Fprintf(stdout, "device=%s abi=v%d chip=H%d mode=%s runs=%d",
			report.Device, report.ABIVersion, report.ChipFamily, report.Mode, report.Runs)
		if report.Warmup > 0 {
			fmt.Fprintf(stdout, " warmup=%d", report.Warmup)
		}
		fmt.Fprintf(stdout, " min_ms=%.3f median_ms=%.3f p95_ms=%.3f mean_ms=%.3f sha256=%s",
			report.MinMS, report.MedianMS, report.P95MS, report.MeanMS, report.OutputSHA256)
		if report.AltOutputSHA256 != "" {
			fmt.Fprintf(stdout, " alt_sha256=%s", report.AltOutputSHA256)
		}
		fmt.Fprintln(stdout)
		if report.Comparison != nil {
			c := report.Comparison
			fmt.Fprintf(stdout, "reference: passed=%t max_abs=%.6g max_rel=%.6g rmse=%.6g max_ulp=%d exact=%d/%d\n",
				c.Passed, c.MaxAbsErr, c.MaxRelErr, c.RMSE, c.MaxULPErr, c.ExactFP16Count, c.TotalElements)
		}
	}

	if report.Comparison != nil && !report.Comparison.Passed {
		return 1
	}
	return 0
}

func cmdRunAnec(args []string, tr ane.Transport, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("run-anec", flag.ContinueOnError)
	fs.SetOutput(stderr)
	anecPath := fs.String("anec", "", "Path to ABI v1 .anec program file")
	var inputPaths stringSliceFlag
	fs.Var(&inputPaths, "input", "Input binary file (repeatable in .anec input surface order)")
	var altInputPaths stringSliceFlag
	fs.Var(&altInputPaths, "alt-input", "Alternate input binary file used on odd runs when --runs > 1 (repeatable)")
	outputOut := fs.String("output-out", "", "Write output surface bytes to this path")
	expectedPath := fs.String("expected", "", "Expected output fp16 binary for direct byte comparison")
	runs := fs.Int("runs", 1, "Number of consecutive timed runs for repeatability verification")
	warmup := fs.Int("warmup", 0, "Number of unmeasured warmup runs before timed --runs (checked for determinism)")
	atol := fs.Float64("atol", 1e-3, "Absolute tolerance for reference comparison")
	rtol := fs.Float64("rtol", 1e-3, "Relative tolerance for reference comparison")
	maxULPVal := fs.Int("max-ulp", -1, "Maximum allowed fp16 ULP error (negative disables)")
	device := fs.String("device", "", "Explicit /dev/accel/accel* device path")
	asJSON := fs.Bool("json", false, "Emit JSON report to stdout")
	jsonOut := fs.String("json-out", "", "Write JSON report to this path")

	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *anecPath == "" {
		fmt.Fprintln(stderr, "error: --anec PATH is required")
		return 1
	}
	if *runs < 1 {
		fmt.Fprintf(stderr, "error: --runs must be >= 1, got %d\n", *runs)
		return 1
	}
	if *warmup < 0 {
		fmt.Fprintf(stderr, "error: --warmup must be >= 0, got %d\n", *warmup)
		return 1
	}

	anecBytes, err := os.ReadFile(*anecPath)
	if err != nil {
		fmt.Fprintf(stderr, "error: read .anec %q: %v\n", *anecPath, err)
		return 1
	}
	prog, err := ane.ParseAnec(anecBytes)
	if err != nil {
		fmt.Fprintf(stderr, "error: parse .anec %q: %v\n", *anecPath, err)
		return 1
	}

	var inputPayloads [][]byte
	for _, p := range inputPaths {
		b, err := os.ReadFile(p)
		if err != nil {
			fmt.Fprintf(stderr, "error: read input %q: %v\n", p, err)
			return 1
		}
		inputPayloads = append(inputPayloads, b)
	}

	var altInputPayloads [][]byte
	for _, p := range altInputPaths {
		b, err := os.ReadFile(p)
		if err != nil {
			fmt.Fprintf(stderr, "error: read alt-input %q: %v\n", p, err)
			return 1
		}
		altInputPayloads = append(altInputPayloads, b)
	}

	var expectedBytes []byte
	if *expectedPath != "" {
		expectedBytes, err = os.ReadFile(*expectedPath)
		if err != nil {
			fmt.Fprintf(stderr, "error: read expected %q: %v\n", *expectedPath, err)
			return 1
		}
	}

	var maxULP *int
	if *maxULPVal >= 0 {
		v := *maxULPVal
		maxULP = &v
	}

	fd, caps, err := ane.DiscoverDeviceABI(tr, *device, ane.ABIV1, ane.ChipH13)
	if err != nil {
		if errors.Is(err, ane.ErrNoMatchingDevice) {
			fmt.Fprintf(stderr, "skip: %v\n", err)
			return ane.ExitSkip
		}
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	defer tr.Close(fd)

	outBytes, report, err := ane.ExecuteAnec(tr, fd, caps, prog, inputPayloads, ane.ExecOptions{
		Runs:      *runs,
		Warmup:    *warmup,
		AltInputs: altInputPayloads,
		Expected:  expectedBytes,
		Atol:      *atol,
		Rtol:      *rtol,
		MaxULP:    maxULP,
	})
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	if *outputOut != "" {
		if err := os.MkdirAll(filepath.Dir(*outputOut), 0o755); err != nil {
			fmt.Fprintf(stderr, "error: create output dir: %v\n", err)
			return 1
		}
		if err := os.WriteFile(*outputOut, outBytes, 0o644); err != nil {
			fmt.Fprintf(stderr, "error: write output %q: %v\n", *outputOut, err)
			return 1
		}
	}

	if *jsonOut != "" {
		if err := os.MkdirAll(filepath.Dir(*jsonOut), 0o755); err != nil {
			fmt.Fprintf(stderr, "error: create json-out dir: %v\n", err)
			return 1
		}
		raw, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "error: marshal report: %v\n", err)
			return 1
		}
		if err := os.WriteFile(*jsonOut, append(raw, '\n'), 0o644); err != nil {
			fmt.Fprintf(stderr, "error: write json-out %q: %v\n", *jsonOut, err)
			return 1
		}
	}

	if *asJSON {
		raw, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "error: marshal report: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
	} else {
		fmt.Fprintf(stdout, "device=%s abi=v%d chip=H%d mode=%s runs=%d",
			report.Device, report.ABIVersion, report.ChipFamily, report.Mode, report.Runs)
		if report.Warmup > 0 {
			fmt.Fprintf(stdout, " warmup=%d", report.Warmup)
		}
		fmt.Fprintf(stdout, " min_ms=%.3f median_ms=%.3f p95_ms=%.3f mean_ms=%.3f sha256=%s",
			report.MinMS, report.MedianMS, report.P95MS, report.MeanMS, report.OutputSHA256)
		if report.AltOutputSHA256 != "" {
			fmt.Fprintf(stdout, " alt_sha256=%s", report.AltOutputSHA256)
		}
		fmt.Fprintln(stdout)
		if report.Comparison != nil {
			c := report.Comparison
			fmt.Fprintf(stdout, "reference: passed=%t max_abs=%.6g max_rel=%.6g rmse=%.6g max_ulp=%d exact=%d/%d\n",
				c.Passed, c.MaxAbsErr, c.MaxRelErr, c.RMSE, c.MaxULPErr, c.ExactFP16Count, c.TotalElements)
		}
	}

	if report.Comparison != nil && !report.Comparison.Passed {
		return 1
	}
	return 0
}

func cmdGenConv1x1(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("gen-conv1x1", flag.ContinueOnError)
	fs.SetOutput(stderr)
	h := fs.Int("h", 0, "Spatial height H")
	w := fs.Int("w", 0, "Spatial width W")
	cin := fs.Int("cin", 0, "Input channels Cin")
	cout := fs.Int("cout", 0, "Output channels Cout")
	withBias := fs.Bool("bias", false, "Generate bias vector")
	relu := fs.Bool("relu", false, "Apply ReLU to expected output")
	actName := fs.String("act", "", "Optional fused activation name (relu or any LUT op)")
	var actAlpha, actBeta, actLo, actHi, actEps, actScalar optFloatFlag
	fs.Var(&actAlpha, "act-alpha", "Activation alpha parameter")
	fs.Var(&actBeta, "act-beta", "Activation beta parameter")
	fs.Var(&actLo, "act-lo", "Activation lo parameter")
	fs.Var(&actHi, "act-hi", "Activation hi parameter")
	fs.Var(&actEps, "act-eps", "Activation eps parameter")
	fs.Var(&actScalar, "act-scalar", "Activation scalar parameter")
	lutBin := fs.String("lut-bin", "", "Optional 128-byte LUT binary emitted by vane")
	seed := fs.Uint64("seed", 1, "Deterministic PRNG seed")
	outDir := fs.String("out-dir", "", "Output directory for generated test vectors")
	asJSON := fs.Bool("json", false, "Emit JSON summary")

	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *h <= 0 || *w <= 0 || *cin <= 0 || *cout <= 0 {
		fmt.Fprintln(stderr, "error: --h, --w, --cin, and --cout must all be positive integers")
		return 1
	}
	if *outDir == "" {
		fmt.Fprintln(stderr, "error: --out-dir DIR is required")
		return 1
	}

	actSpec, isLut, err := resolveConvActivation(*relu, *actName, actAlpha, actBeta, actLo, actHi, actEps, actScalar)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	rng := rand.New(rand.NewPCG(*seed, *seed^0x9e3779b97f4a7c15))

	x, err := ref.NewTensor(1, *cin, *h, *w)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	for i := range x.Data {
		x.Data[i] = fp16.FromFloat64(rng.Float64()*2.0 - 1.0)
	}

	wScale := math.Min(0.5, 2.0/math.Sqrt(float64(*cin)))
	weights := make([]uint16, (*cout)*(*cin))
	for i := range weights {
		weights[i] = fp16.FromFloat64((rng.Float64()*2.0 - 1.0) * wScale)
	}

	var bias []uint16
	if *withBias {
		bias = make([]uint16, *cout)
		for i := range bias {
			bias[i] = fp16.FromFloat64(rng.Float64() - 0.5)
		}
	}

	var yF64, yQ1516 ref.Tensor
	if isLut {
		tableBytes, preScale, shift, err := loadOrBuildLut(actSpec, *lutBin)
		if err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
		yF64, err = ref.Conv2d1x1Act(x, weights, *cout, bias, actSpec)
		if err != nil {
			fmt.Fprintf(stderr, "error: float64 reference failed: %v\n", err)
			return 1
		}
		yQ1516, err = ref.Conv2d1x1Q1516Lut(x, weights, *cout, bias, tableBytes, preScale, shift)
		if err != nil {
			fmt.Fprintf(stderr, "error: Q15.16+LUT reference failed: %v\n", err)
			return 1
		}
	} else {
		effRelu := actSpec.Name == "relu"
		yF64, err = ref.Conv2d1x1(x, weights, *cout, bias, effRelu)
		if err != nil {
			fmt.Fprintf(stderr, "error: float64 reference failed: %v\n", err)
			return 1
		}
		yQ1516, err = ref.Conv2d1x1Q1516(x, weights, *cout, bias, effRelu)
		if err != nil {
			fmt.Fprintf(stderr, "error: Q15.16 reference failed: %v\n", err)
			return 1
		}
	}

	rowStride := ref.AlignUp(2*(*w), 64)
	planeStride := (*h) * rowStride
	inBatchStride := (*cin) * planeStride
	outBatchStride := (*cout) * planeStride

	inSurface, err := ref.PackSurface(x, inBatchStride, planeStride, rowStride, inBatchStride)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack input surface: %v\n", err)
		return 1
	}
	expF64Surface, err := ref.PackSurface(yF64, outBatchStride, planeStride, rowStride, outBatchStride)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack expected_f64 surface: %v\n", err)
		return 1
	}
	expQSurface, err := ref.PackSurface(yQ1516, outBatchStride, planeStride, rowStride, outBatchStride)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack expected_q1516 surface: %v\n", err)
		return 1
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "error: create out-dir: %v\n", err)
		return 1
	}
	writeFile := func(name string, data []byte) error {
		return os.WriteFile(filepath.Join(*outDir, name), data, 0o644)
	}
	if err := writeFile("input.bin", inSurface); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if err := writeFile("weights.bin", fp16.EncodeLE(weights)); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if *withBias {
		if err := writeFile("bias.bin", fp16.EncodeLE(bias)); err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
	}
	if err := writeFile("expected_f64.bin", expF64Surface); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if err := writeFile("expected_q1516.bin", expQSurface); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if isLut {
		if err := writeFile("expected_lut.bin", expQSurface); err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
	}

	summary := map[string]any{
		"h":             *h,
		"w":             *w,
		"cin":           *cin,
		"cout":          *cout,
		"bias":          *withBias,
		"relu":          actSpec.Name == "relu",
		"activation":    actSpec.Name,
		"seed":          *seed,
		"row_stride":    rowStride,
		"plane_stride":  planeStride,
		"in_bytes":      len(inSurface),
		"out_bytes":     len(expF64Surface),
		"weights_bytes": len(weights) * 2,
		"bias_bytes":    len(bias) * 2,
		"out_dir":       *outDir,
	}
	if *asJSON {
		raw, _ := json.MarshalIndent(summary, "", "  ")
		fmt.Fprintln(stdout, string(raw))
	} else {
		fmt.Fprintf(stdout, "generated conv1x1 (%dx%d, %d->%d, bias=%t, act=%s, seed=%d) in %s (in=%dB out=%dB weights=%dB)\n",
			*h, *w, *cin, *cout, *withBias, actSpec.Name, *seed, *outDir, len(inSurface), len(expF64Surface), len(weights)*2)
	}
	return 0
}

func cmdGenConv3x3(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("gen-conv3x3", flag.ContinueOnError)
	fs.SetOutput(stderr)
	mode := fs.String("mode", "", "Convolution mode: dense or depthwise")
	h := fs.Int("h", 0, "Spatial height H")
	w := fs.Int("w", 0, "Spatial width W")
	cin := fs.Int("cin", 0, "Input channels Cin")
	cout := fs.Int("cout", 0, "Output channels Cout")
	stride := fs.Int("stride", 1, "Spatial stride (1 or 2)")
	pad := fs.String("pad", "", "Padding mode: same or valid")
	withBias := fs.Bool("bias", false, "Generate bias vector")
	relu := fs.Bool("relu", false, "Apply ReLU to expected output")
	actName := fs.String("act", "", "Optional fused activation name (relu or any LUT op)")
	var actAlpha, actBeta, actLo, actHi, actEps, actScalar optFloatFlag
	fs.Var(&actAlpha, "act-alpha", "Activation alpha parameter")
	fs.Var(&actBeta, "act-beta", "Activation beta parameter")
	fs.Var(&actLo, "act-lo", "Activation lo parameter")
	fs.Var(&actHi, "act-hi", "Activation hi parameter")
	fs.Var(&actEps, "act-eps", "Activation eps parameter")
	fs.Var(&actScalar, "act-scalar", "Activation scalar parameter")
	lutBin := fs.String("lut-bin", "", "Optional 128-byte LUT binary emitted by vane")
	seed := fs.Uint64("seed", 1, "Deterministic PRNG seed")
	outDir := fs.String("out-dir", "", "Output directory for generated test vectors")
	asJSON := fs.Bool("json", false, "Emit JSON summary")

	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *mode != "dense" && *mode != "depthwise" {
		fmt.Fprintf(stderr, "error: --mode must be dense or depthwise, got %q\n", *mode)
		return 1
	}
	if *h <= 0 || *w <= 0 || *cin <= 0 || *cout <= 0 {
		fmt.Fprintln(stderr, "error: --h, --w, --cin, and --cout must all be positive integers")
		return 1
	}
	if *mode == "depthwise" && *cin != *cout {
		fmt.Fprintf(stderr, "error: depthwise mode requires --cin == --cout (got %d vs %d)\n", *cin, *cout)
		return 1
	}
	if *stride != 1 && *stride != 2 {
		fmt.Fprintf(stderr, "error: --stride must be 1 or 2, got %d\n", *stride)
		return 1
	}
	if *pad != "same" && *pad != "valid" {
		fmt.Fprintf(stderr, "error: --pad must be same or valid, got %q\n", *pad)
		return 1
	}
	if *outDir == "" {
		fmt.Fprintln(stderr, "error: --out-dir DIR is required")
		return 1
	}

	actSpec, isLut, err := resolveConvActivation(*relu, *actName, actAlpha, actBeta, actLo, actHi, actEps, actScalar)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	ho, wo, padTop, padLeft, err := ref.Conv3x3Geometry(*h, *w, *stride, *pad)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	groups := 1
	cinPerGroup := *cin
	if *mode == "depthwise" {
		groups = *cin
		cinPerGroup = 1
	}

	rng := rand.New(rand.NewPCG(*seed, *seed^0x9e3779b97f4a7c15))

	x, err := ref.NewTensor(1, *cin, *h, *w)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	for i := range x.Data {
		x.Data[i] = fp16.FromFloat64(rng.Float64()*2.0 - 1.0)
	}

	wScale := math.Min(0.5, 2.0/math.Sqrt(float64(9*cinPerGroup)))
	weights := make([]uint16, (*cout)*cinPerGroup*9)
	for i := range weights {
		weights[i] = fp16.FromFloat64((rng.Float64()*2.0 - 1.0) * wScale)
	}

	var bias []uint16
	if *withBias {
		bias = make([]uint16, *cout)
		for i := range bias {
			bias[i] = fp16.FromFloat64(rng.Float64() - 0.5)
		}
	}

	var yF64, yQ1516 ref.Tensor
	if isLut {
		tableBytes, preScale, shift, err := loadOrBuildLut(actSpec, *lutBin)
		if err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
		yF64, err = ref.Conv2d3x3PadAct(x, weights, *cout, bias, *stride, *pad, groups, actSpec)
		if err != nil {
			fmt.Fprintf(stderr, "error: float64 reference failed: %v\n", err)
			return 1
		}
		yQ1516, err = ref.Conv2d3x3Q1516Lut(x, weights, *cout, bias, *stride, *pad, groups, tableBytes, preScale, shift)
		if err != nil {
			fmt.Fprintf(stderr, "error: Q15.16+LUT reference failed: %v\n", err)
			return 1
		}
	} else {
		effRelu := actSpec.Name == "relu"
		yF64, err = ref.Conv2d3x3Pad(x, weights, *cout, bias, *stride, *pad, groups, effRelu)
		if err != nil {
			fmt.Fprintf(stderr, "error: float64 reference failed: %v\n", err)
			return 1
		}
		yQ1516, err = ref.Conv2d3x3Q1516(x, weights, *cout, bias, *stride, *pad, groups, effRelu)
		if err != nil {
			fmt.Fprintf(stderr, "error: Q15.16 reference failed: %v\n", err)
			return 1
		}
	}

	inRowStride := ref.AlignUp(2*(*w), 64)
	inPlaneStride := (*h) * inRowStride
	inBatchStride := (*cin) * inPlaneStride

	outRowStride := ref.AlignUp(2*wo, 64)
	outPlaneStride := ho * outRowStride
	outBatchStride := (*cout) * outPlaneStride

	inSurface, err := ref.PackSurface(x, inBatchStride, inPlaneStride, inRowStride, inBatchStride)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack input surface: %v\n", err)
		return 1
	}
	expF64Surface, err := ref.PackSurface(yF64, outBatchStride, outPlaneStride, outRowStride, outBatchStride)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack expected_f64 surface: %v\n", err)
		return 1
	}
	expQSurface, err := ref.PackSurface(yQ1516, outBatchStride, outPlaneStride, outRowStride, outBatchStride)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack expected_q1516 surface: %v\n", err)
		return 1
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "error: create out-dir: %v\n", err)
		return 1
	}
	writeFile := func(name string, data []byte) error {
		return os.WriteFile(filepath.Join(*outDir, name), data, 0o644)
	}
	if err := writeFile("input.bin", inSurface); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if err := writeFile("weights.bin", fp16.EncodeLE(weights)); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if *withBias {
		if err := writeFile("bias.bin", fp16.EncodeLE(bias)); err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
	}
	if err := writeFile("expected_f64.bin", expF64Surface); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if err := writeFile("expected_q1516.bin", expQSurface); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if isLut {
		if err := writeFile("expected_lut.bin", expQSurface); err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
	}

	summary := map[string]any{
		"mode":             *mode,
		"h":                *h,
		"w":                *w,
		"cin":              *cin,
		"cout":             *cout,
		"ho":               ho,
		"wo":               wo,
		"stride":           *stride,
		"pad":              *pad,
		"pt":               padTop,
		"pl":               padLeft,
		"bias":             *withBias,
		"relu":             actSpec.Name == "relu",
		"activation":       actSpec.Name,
		"seed":             *seed,
		"in_row_stride":    inRowStride,
		"in_plane_stride":  inPlaneStride,
		"out_row_stride":   outRowStride,
		"out_plane_stride": outPlaneStride,
		"in_bytes":         len(inSurface),
		"out_bytes":        len(expF64Surface),
		"weights_bytes":    len(weights) * 2,
		"bias_bytes":       len(bias) * 2,
		"out_dir":          *outDir,
	}
	if *asJSON {
		raw, _ := json.MarshalIndent(summary, "", "  ")
		fmt.Fprintln(stdout, string(raw))
	} else {
		fmt.Fprintf(stdout, "generated conv3x3 %s (%dx%d->%dx%d, %d->%d, s=%d, pad=%s, bias=%t, act=%s, seed=%d) in %s (in=%dB out=%dB weights=%dB)\n",
			*mode, *h, *w, ho, wo, *cin, *cout, *stride, *pad, *withBias, actSpec.Name, *seed, *outDir, len(inSurface), len(expF64Surface), len(weights)*2)
	}
	return 0
}

var peEdgeValues = []uint16{
	0x0000, 0x8000, // +0, -0
	0x0001, 0x8001, // +-min subnormal
	0x03ff, 0x83ff, // +-max subnormal
	0x0400, 0x8400, // +-min normal
	0x3800, 0xb800, // +-0.5
	0x3c00, 0xbc00, // +-1.0
	0x3c01, 0xbc01, // +-(1 + 2^-10)
	0x3e00, 0xbe00, // +-1.5
	0x4000, 0xc000, // +-2.0
	0x4200, 0xc200, // +-3.0
	0x4900, 0xc900, // +-10.0
	0x5640, 0xd640, // +-100.0
	0x7bff, 0xfbff, // +-65504
	0x7c00, 0xfc00, // +-inf
	0x7d00, 0xfd00, // +-sNaN
	0x7e00, 0x7fff, // qNaN, max NaN
}

func packTensor64B(t ref.Tensor) ([]byte, error) {
	rs := ref.AlignUp(2*t.W, 64)
	ps := t.H * rs
	bs := t.C * ps
	return ref.PackSurface(t, bs, ps, rs, bs)
}

func cmdGenPe(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("gen-pe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	opStr := fs.String("op", "", "PE op: abs|square|linear|chain|sadd|ssub|srsub|smul|sdiv|add|sub|rsub|mul|max|min")
	h := fs.Int("h", 0, "Primary height H")
	w := fs.Int("w", 0, "Primary width W")
	c := fs.Int("c", 0, "Primary channels C")
	zh := fs.Int("z-h", 0, "Secondary input height ZH (defaults to --h)")
	zw := fs.Int("z-w", 0, "Secondary input width ZW (defaults to --w)")
	zc := fs.Int("z-c", 0, "Secondary input channels ZC (defaults to --c)")
	var alpha, beta, inScale, bias, scale, scalar optFloatFlag
	fs.Var(&alpha, "alpha", "linear alpha")
	fs.Var(&beta, "beta", "linear beta")
	fs.Var(&inScale, "in-scale", "chain in_scale")
	fs.Var(&bias, "bias", "chain bias")
	fs.Var(&scale, "scale", "chain scale")
	fs.Var(&scalar, "scalar", "scalar constant for sadd/ssub/srsub/smul/sdiv")
	edgeValues := fs.Bool("edge-values", false, "Populate input with special/edge fp16 bit patterns")
	seed := fs.Uint64("seed", 1, "Deterministic PRNG seed")
	outDir := fs.String("out-dir", "", "Output directory")
	asJSON := fs.Bool("json", false, "Emit JSON summary")

	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *opStr == "" || *h <= 0 || *w <= 0 || *c <= 0 || *outDir == "" {
		fmt.Fprintln(stderr, "error: --op, --h, --w, --c, and --out-dir are required")
		return 1
	}

	op := strings.ToLower(*opStr)
	isTwoSource := op == "add" || op == "sub" || op == "rsub" || op == "mul" || op == "max" || op == "min"

	rng := rand.New(rand.NewPCG(*seed, *seed^0x517cc1b727220a95))
	x, err := ref.NewTensor(1, *c, *h, *w)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	for i := range x.Data {
		if *edgeValues {
			x.Data[i] = peEdgeValues[i%len(peEdgeValues)]
		} else {
			x.Data[i] = fp16.FromFloat64(rng.Float64()*4.0 - 2.0)
		}
	}

	var z ref.Tensor
	var yRNA, yF64 ref.Tensor
	if isTwoSource {
		zC, zH, zW := *c, *h, *w
		if *zc > 0 {
			zC = *zc
		}
		if *zh > 0 {
			zH = *zh
		}
		if *zw > 0 {
			zW = *zw
		}
		z, err = ref.NewTensor(1, zC, zH, zW)
		if err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
		for i := range z.Data {
			if *edgeValues {
				z.Data[i] = peEdgeValues[(i*7+3)%len(peEdgeValues)]
			} else {
				z.Data[i] = fp16.FromFloat64(rng.Float64()*4.0 - 2.0)
			}
		}
		yRNA, err = ref.PeBinary(op, x, z, false)
		if err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
		yF64, err = ref.PeBinaryF64(op, x, z, false)
		if err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
	} else {
		f16Ptr := func(v float64) *uint16 {
			h := fp16.FromFloat64(v)
			if h == fp16.NegZero {
				h = fp16.PosZero
			}
			return &h
		}
		switch op {
		case "abs", "square":
			yRNA, err = ref.PeUnary(op, x)
			if err != nil {
				fmt.Fprintf(stderr, "error: %v\n", err)
				return 1
			}
			yF64, _ = ref.NewTensor(x.N, x.C, x.H, x.W)
			for i, v := range x.Data {
				f := fp16.ToFloat64(v)
				if op == "abs" {
					yF64.Data[i] = fp16.FromFloat64(math.Abs(f))
				} else {
					yF64.Data[i] = fp16.FromFloat64(f * f)
				}
			}
		case "linear":
			aVal, bVal := 1.0, 0.0
			if alpha.set {
				aVal = alpha.val
			}
			if beta.set {
				bVal = beta.val
			}
			yRNA, err = ref.PeChain(x, f16Ptr(aVal), f16Ptr(bVal), nil)
			if err != nil {
				fmt.Fprintf(stderr, "error: %v\n", err)
				return 1
			}
			yF64, _ = ref.PeChainF64(x, f16Ptr(aVal), f16Ptr(bVal), nil)
		case "chain":
			var insP, bP, sP *uint16
			if inScale.set {
				insP = f16Ptr(inScale.val)
			}
			if bias.set {
				bP = f16Ptr(bias.val)
			}
			if scale.set {
				sP = f16Ptr(scale.val)
			}
			yRNA, err = ref.PeChain(x, insP, bP, sP)
			if err != nil {
				fmt.Fprintf(stderr, "error: %v\n", err)
				return 1
			}
			yF64, _ = ref.PeChainF64(x, insP, bP, sP)
		case "sadd":
			if !scalar.set {
				fmt.Fprintln(stderr, "error: sadd requires --scalar F")
				return 1
			}
			yRNA, _ = ref.PeChain(x, nil, f16Ptr(scalar.val), nil)
			yF64, _ = ref.PeChainF64(x, nil, f16Ptr(scalar.val), nil)
		case "ssub":
			if !scalar.set {
				fmt.Fprintln(stderr, "error: ssub requires --scalar F")
				return 1
			}
			yRNA, _ = ref.PeChain(x, nil, f16Ptr(-scalar.val), nil)
			yF64, _ = ref.PeChainF64(x, nil, f16Ptr(-scalar.val), nil)
		case "srsub":
			if !scalar.set {
				fmt.Fprintln(stderr, "error: srsub requires --scalar F")
				return 1
			}
			yRNA, _ = ref.PeChain(x, f16Ptr(-1.0), f16Ptr(scalar.val), nil)
			yF64, _ = ref.PeChainF64(x, f16Ptr(-1.0), f16Ptr(scalar.val), nil)
		case "smul":
			if !scalar.set {
				fmt.Fprintln(stderr, "error: smul requires --scalar F")
				return 1
			}
			yRNA, _ = ref.PeChain(x, nil, nil, f16Ptr(scalar.val))
			yF64, _ = ref.PeChainF64(x, nil, nil, f16Ptr(scalar.val))
		case "sdiv":
			if !scalar.set || math.IsNaN(scalar.val) || math.IsInf(scalar.val, 0) {
				fmt.Fprintln(stderr, "error: sdiv requires finite non-zero --scalar F")
				return 1
			}
			sBits := fp16.FromFloat64(scalar.val)
			if !fp16.IsFinite(sBits) || (sBits&0x7fff) == 0 {
				fmt.Fprintln(stderr, "error: sdiv requires finite non-zero fp16 --scalar F")
				return 1
			}
			inv := 1.0 / fp16.ToFloat64(sBits)
			invBits := fp16.FromFloat64(inv)
			if !fp16.IsFinite(invBits) {
				fmt.Fprintln(stderr, "error: sdiv reciprocal overflows fp16 finite range")
				return 1
			}
			yRNA, _ = ref.PeChain(x, nil, nil, f16Ptr(inv))
			yF64, _ = ref.PeChainF64(x, nil, nil, f16Ptr(inv))
		default:
			fmt.Fprintf(stderr, "error: unknown PE op %q\n", *opStr)
			return 1
		}
	}

	in0Buf, err := packTensor64B(x)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack input0: %v\n", err)
		return 1
	}
	expRNABuf, err := packTensor64B(yRNA)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack expected_rna16: %v\n", err)
		return 1
	}
	expF64Buf, err := packTensor64B(yF64)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack expected_f64: %v\n", err)
		return 1
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "error: create out-dir: %v\n", err)
		return 1
	}
	if err := os.WriteFile(filepath.Join(*outDir, "input0.bin"), in0Buf, 0o644); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	var in1Bytes int
	if isTwoSource {
		in1Buf, err := packTensor64B(z)
		if err != nil {
			fmt.Fprintf(stderr, "error: pack input1: %v\n", err)
			return 1
		}
		if err := os.WriteFile(filepath.Join(*outDir, "input1.bin"), in1Buf, 0o644); err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
		in1Bytes = len(in1Buf)
	}
	if err := os.WriteFile(filepath.Join(*outDir, "expected_rna16.bin"), expRNABuf, 0o644); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if err := os.WriteFile(filepath.Join(*outDir, "expected_f64.bin"), expF64Buf, 0o644); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	summary := map[string]any{
		"op":          op,
		"two_source":  isTwoSource,
		"out_shape":   []int{yRNA.N, yRNA.C, yRNA.H, yRNA.W},
		"in0_bytes":   len(in0Buf),
		"in1_bytes":   in1Bytes,
		"out_bytes":   len(expRNABuf),
		"edge_values": *edgeValues,
		"seed":        *seed,
		"out_dir":     *outDir,
	}
	if *asJSON {
		raw, _ := json.MarshalIndent(summary, "", "  ")
		fmt.Fprintln(stdout, string(raw))
	} else {
		fmt.Fprintf(stdout, "generated pe %s (out=%dx%dx%d, in0=%dB, out=%dB) in %s\n",
			op, yRNA.C, yRNA.H, yRNA.W, len(in0Buf), len(expRNABuf), *outDir)
	}
	return 0
}

func cmdGenLut(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("gen-lut", flag.ContinueOnError)
	fs.SetOutput(stderr)
	actName := fs.String("act", "", "Activation name (relu, copy, or any 128B LUT op)")
	h := fs.Int("h", 0, "Spatial height H")
	w := fs.Int("w", 0, "Spatial width W")
	c := fs.Int("c", 0, "Channels C")
	var alpha, beta, lo, hi, eps, scalar optFloatFlag
	fs.Var(&alpha, "alpha", "Activation alpha parameter")
	fs.Var(&beta, "beta", "Activation beta parameter")
	fs.Var(&lo, "lo", "Activation lo parameter")
	fs.Var(&hi, "hi", "Activation hi parameter")
	fs.Var(&eps, "eps", "Activation eps parameter")
	fs.Var(&scalar, "scalar", "Activation scalar parameter")
	allFP16 := fs.Bool("all-fp16", false, "Fill input with all 65,536 fp16 bit patterns 0x0000..0xffff")
	lutBin := fs.String("lut-bin", "", "Optional 128-byte LUT binary emitted by vane")
	seed := fs.Uint64("seed", 1, "Deterministic PRNG seed")
	outDir := fs.String("out-dir", "", "Output directory")
	asJSON := fs.Bool("json", false, "Emit JSON summary")

	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *actName == "" || *h <= 0 || *w <= 0 || *c <= 0 || *outDir == "" {
		fmt.Fprintln(stderr, "error: --act, --h, --w, --c, and --out-dir are required")
		return 1
	}
	if *allFP16 && (*c)*(*h)*(*w) != 65536 {
		fmt.Fprintf(stderr, "error: --all-fp16 requires C*H*W == 65536 (got %d)\n", (*c)*(*h)*(*w))
		return 1
	}

	spec, err := buildActSpec(*actName, alpha, beta, lo, hi, eps, scalar)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	rng := rand.New(rand.NewPCG(*seed, *seed^0xa0761d6478bd642f))
	x, err := ref.NewTensor(1, *c, *h, *w)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	for i := range x.Data {
		if *allFP16 {
			x.Data[i] = uint16(i & 0xffff)
		} else {
			var v float64
			switch spec.Name {
			case "sqrt", "rsqrt", "log2":
				v = 0.05 + rng.Float64()*8.0
			case "inverse":
				v = 0.25 + rng.Float64()*4.0
				if rng.IntN(2) == 0 {
					v = -v
				}
			case "exp", "exp2":
				v = rng.Float64()*6.0 - 3.0
			default:
				v = rng.Float64()*8.0 - 4.0
			}
			x.Data[i] = fp16.FromFloat64(v)
		}
	}

	yF64, err := ref.ApplyActF64(x, spec)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	var yLut *ref.Tensor
	if spec.Name != "relu" && spec.Name != "copy" {
		tableBytes, preScale, shift, err := loadOrBuildLut(spec, *lutBin)
		if err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
		yl, err := ref.ApplyLutTensor(x, tableBytes, preScale, shift)
		if err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
		yLut = &yl
	}

	inBuf, err := packTensor64B(x)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack input: %v\n", err)
		return 1
	}
	expF64Buf, err := packTensor64B(yF64)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack expected_f64: %v\n", err)
		return 1
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "error: create out-dir: %v\n", err)
		return 1
	}
	if err := os.WriteFile(filepath.Join(*outDir, "input.bin"), inBuf, 0o644); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if err := os.WriteFile(filepath.Join(*outDir, "expected_f64.bin"), expF64Buf, 0o644); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if yLut != nil {
		expLutBuf, err := packTensor64B(*yLut)
		if err != nil {
			fmt.Fprintf(stderr, "error: pack expected_lut: %v\n", err)
			return 1
		}
		if err := os.WriteFile(filepath.Join(*outDir, "expected_lut.bin"), expLutBuf, 0o644); err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
	}

	summary := map[string]any{
		"act":       spec.Name,
		"shape":     []int{1, *c, *h, *w},
		"all_fp16":  *allFP16,
		"has_lut":   yLut != nil,
		"in_bytes":  len(inBuf),
		"out_bytes": len(expF64Buf),
		"seed":      *seed,
		"out_dir":   *outDir,
	}
	if *asJSON {
		raw, _ := json.MarshalIndent(summary, "", "  ")
		fmt.Fprintln(stdout, string(raw))
	} else {
		fmt.Fprintf(stdout, "generated lut %s (%dx%dx%d, all_fp16=%t, in=%dB) in %s\n",
			spec.Name, *c, *h, *w, *allFP16, len(inBuf), *outDir)
	}
	return 0
}

func cmdGenPool(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("gen-pool", flag.ContinueOnError)
	fs.SetOutput(stderr)
	opName := fs.String("op", "", "Pool op: max or avg")
	h := fs.Int("h", 0, "Input height H")
	w := fs.Int("w", 0, "Input width W")
	c := fs.Int("c", 0, "Channels C")
	kh := fs.Int("kh", 0, "Window height KH")
	kw := fs.Int("kw", 0, "Window width KW")
	sh := fs.Int("sh", 1, "Vertical stride SH (1 or 2)")
	sw := fs.Int("sw", 1, "Horizontal stride SW (1 or 2)")
	pad := fs.String("pad", "valid", "Padding mode: valid, same, or custom")
	pt := fs.Int("pt", 0, "Custom top padding")
	pb := fs.Int("pb", 0, "Custom bottom padding")
	pl := fs.Int("pl", 0, "Custom left padding")
	pr := fs.Int("pr", 0, "Custom right padding")
	excludePad := fs.Bool("exclude-pad", true, "Exclude padded elements from avg_pool divisor")
	seed := fs.Uint64("seed", 1, "Deterministic PRNG seed")
	outDir := fs.String("out-dir", "", "Output directory")
	asJSON := fs.Bool("json", false, "Emit JSON summary")

	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *opName == "" || *h <= 0 || *w <= 0 || *c <= 0 || *kh <= 0 || *kw <= 0 || *outDir == "" {
		fmt.Fprintln(stderr, "error: --op, --h, --w, --c, --kh, --kw, and --out-dir are required")
		return 1
	}

	spec := ref.PoolSpec{
		Op:         *opName,
		KH:         *kh,
		KW:         *kw,
		SH:         *sh,
		SW:         *sw,
		Pad:        *pad,
		PT:         *pt,
		PB:         *pb,
		PL:         *pl,
		PR:         *pr,
		ExcludePad: *excludePad,
	}

	rng := rand.New(rand.NewPCG(*seed, *seed^0x517cc1b727220a95))
	x, err := ref.NewTensor(1, *c, *h, *w)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	for i := range x.Data {
		x.Data[i] = fp16.FromFloat64(rng.Float64()*4.0 - 2.0)
	}

	y, err := ref.EvalPool(x, spec)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	inBuf, err := packTensor64B(x)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack input: %v\n", err)
		return 1
	}
	expBuf, err := packTensor64B(y)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack expected: %v\n", err)
		return 1
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "error: create out-dir: %v\n", err)
		return 1
	}
	if err := os.WriteFile(filepath.Join(*outDir, "input.bin"), inBuf, 0o644); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if err := os.WriteFile(filepath.Join(*outDir, "expected_f64.bin"), expBuf, 0o644); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	summary := map[string]any{
		"op":          strings.ToLower(*opName),
		"in_shape":    []int{1, *c, *h, *w},
		"out_shape":   []int{y.N, y.C, y.H, y.W},
		"kh":          *kh,
		"kw":          *kw,
		"sh":          *sh,
		"sw":          *sw,
		"pad":         *pad,
		"exclude_pad": *excludePad,
		"in_bytes":    len(inBuf),
		"out_bytes":   len(expBuf),
		"seed":        *seed,
		"out_dir":     *outDir,
	}
	if *asJSON {
		raw, _ := json.MarshalIndent(summary, "", "  ")
		fmt.Fprintln(stdout, string(raw))
	} else {
		fmt.Fprintf(stdout, "generated pool %s (%dx%dx%d -> %dx%dx%d, in=%dB, out=%dB) in %s\n",
			strings.ToLower(*opName), *c, *h, *w, y.C, y.H, y.W, len(inBuf), len(expBuf), *outDir)
	}
	return 0
}

func cmdGenReduce(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("gen-reduce", flag.ContinueOnError)
	fs.SetOutput(stderr)
	opName := fs.String("op", "", "Reduce op (sum, mean, max, min, l1, sum_square, l2, inv_sum, rsqrt_sum_square, rsqrt_mean_square)")
	h := fs.Int("h", 0, "Input height H")
	w := fs.Int("w", 0, "Input width W")
	c := fs.Int("c", 0, "Channels C")
	axesStr := fs.String("axes", "", "Reduced axes: w, h, c, hw, cw, ch, or chw")
	eps := fs.Float64("eps", 0.0, "Epsilon for rsqrt_sum_square / rsqrt_mean_square")
	seed := fs.Uint64("seed", 1, "Deterministic PRNG seed")
	outDir := fs.String("out-dir", "", "Output directory")
	asJSON := fs.Bool("json", false, "Emit JSON summary")

	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *opName == "" || *h <= 0 || *w <= 0 || *c <= 0 || *axesStr == "" || *outDir == "" {
		fmt.Fprintln(stderr, "error: --op, --h, --w, --c, --axes, and --out-dir are required")
		return 1
	}
	rc, rh, rw, err := ref.ParseReduceAxes(*axesStr)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	opLow := strings.ToLower(*opName)
	rng := rand.New(rand.NewPCG(*seed, *seed^0x94d049bb133111eb))
	x, err := ref.NewTensor(1, *c, *h, *w)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	for i := range x.Data {
		var v float64
		if opLow == "inv_sum" {
			v = 0.25 + rng.Float64()*1.5
		} else {
			v = rng.Float64()*2.0 - 1.0
		}
		x.Data[i] = fp16.FromFloat64(v)
	}

	y, err := ref.EvalReduce(x, ref.ReduceSpec{
		Op:      opLow,
		ReduceC: rc,
		ReduceH: rh,
		ReduceW: rw,
		Eps:     *eps,
	})
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	inBuf, err := packTensor64B(x)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack input: %v\n", err)
		return 1
	}
	expBuf, err := packTensor64B(y)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack expected: %v\n", err)
		return 1
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "error: create out-dir: %v\n", err)
		return 1
	}
	if err := os.WriteFile(filepath.Join(*outDir, "input.bin"), inBuf, 0o644); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	if err := os.WriteFile(filepath.Join(*outDir, "expected_f64.bin"), expBuf, 0o644); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	summary := map[string]any{
		"op":        opLow,
		"axes":      strings.ToLower(*axesStr),
		"eps":       *eps,
		"in_shape":  []int{1, *c, *h, *w},
		"out_shape": []int{y.N, y.C, y.H, y.W},
		"in_bytes":  len(inBuf),
		"out_bytes": len(expBuf),
		"seed":      *seed,
		"out_dir":   *outDir,
	}
	if *asJSON {
		raw, _ := json.MarshalIndent(summary, "", "  ")
		fmt.Fprintln(stdout, string(raw))
	} else {
		fmt.Fprintf(stdout, "generated reduce %s axes=%s (%dx%dx%d -> %dx%dx%d, in=%dB, out=%dB) in %s\n",
			opLow, strings.ToLower(*axesStr), *c, *h, *w, y.C, y.H, y.W, len(inBuf), len(expBuf), *outDir)
	}
	return 0
}

func packTight64Surface(t ref.Tensor) ([]byte, int, int, int, error) {
	rowStride := ref.AlignUp(2*t.W, 64)
	planeStride := t.H * rowStride
	batchStride := t.C * planeStride
	buf, err := ref.PackSurface(t, batchStride, planeStride, rowStride, batchStride)
	return buf, batchStride, planeStride, rowStride, err
}

func cmdGenGraph(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("gen-graph", flag.ContinueOnError)
	fs.SetOutput(stderr)
	specPath := fs.String("spec", "", "Path to multi-task graph JSON template/spec")
	outDir := fs.String("out-dir", "", "Output directory for generated inputs, weights, graph_resolved.json, and expected outputs")
	seed := fs.Uint64("seed", 1, "Deterministic PRNG seed")
	asJSON := fs.Bool("json", false, "Emit JSON summary")

	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *specPath == "" {
		fmt.Fprintln(stderr, "error: --spec PATH is required")
		return 1
	}
	if *outDir == "" {
		fmt.Fprintln(stderr, "error: --out-dir DIR is required")
		return 1
	}

	rawSpec, err := os.ReadFile(*specPath)
	if err != nil {
		fmt.Fprintf(stderr, "error: read graph spec %q: %v\n", *specPath, err)
		return 1
	}
	var spec ref.GraphSpec
	if err := json.Unmarshal(rawSpec, &spec); err != nil {
		fmt.Fprintf(stderr, "error: parse graph spec %q: %v\n", *specPath, err)
		return 1
	}
	if len(spec.Inputs) < 1 || len(spec.Inputs) > 3 {
		fmt.Fprintf(stderr, "error: graph must have 1..3 inputs, got %d\n", len(spec.Inputs))
		return 1
	}
	if len(spec.Nodes) == 0 {
		fmt.Fprintln(stderr, "error: graph must have at least 1 node")
		return 1
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "error: create out-dir: %v\n", err)
		return 1
	}
	writeFile := func(name string, data []byte) error {
		return os.WriteFile(filepath.Join(*outDir, name), data, 0o644)
	}

	rng := rand.New(rand.NewPCG(*seed, *seed^0x9e3779b97f4a7c15))

	type dims3 struct{ c, h, w int }
	shapes := make(map[string]dims3, len(spec.Inputs)+len(spec.Nodes))
	inTensors := make(map[string]ref.Tensor, len(spec.Inputs))
	inFiles := make([]string, 0, len(spec.Inputs))

	for idx, inp := range spec.Inputs {
		if inp.Name == "" || inp.C <= 0 || inp.H <= 0 || inp.W <= 0 {
			fmt.Fprintf(stderr, "error: invalid input[%d] (%q: %dx%dx%d)\n", idx, inp.Name, inp.C, inp.H, inp.W)
			return 1
		}
		t, err := ref.NewTensor(1, inp.C, inp.H, inp.W)
		if err != nil {
			fmt.Fprintf(stderr, "error: input %q: %v\n", inp.Name, err)
			return 1
		}
		for i := range t.Data {
			t.Data[i] = fp16.FromFloat64(rng.Float64()*2.0 - 1.0)
		}
		buf, _, _, _, err := packTight64Surface(t)
		if err != nil {
			fmt.Fprintf(stderr, "error: pack input %q: %v\n", inp.Name, err)
			return 1
		}
		fname := fmt.Sprintf("input%d.bin", idx)
		if err := writeFile(fname, buf); err != nil {
			fmt.Fprintf(stderr, "error: write %s: %v\n", fname, err)
			return 1
		}
		if len(spec.Inputs) == 1 {
			if err := writeFile("input.bin", buf); err != nil {
				fmt.Fprintf(stderr, "error: write input.bin: %v\n", err)
				return 1
			}
		}
		inFiles = append(inFiles, fname)
		inTensors[inp.Name] = t
		shapes[inp.Name] = dims3{inp.C, inp.H, inp.W}
	}

	resolved := spec
	resolved.Nodes = make([]ref.GraphNodeSpec, len(spec.Nodes))
	copy(resolved.Nodes, spec.Nodes)

	nodeWeights := make(map[string]ref.GraphNodeWeights)
	for i := range resolved.Nodes {
		nd := &resolved.Nodes[i]
		op := strings.ToLower(nd.Op)
		switch op {
		case "conv1x1":
			if len(nd.Inputs) != 1 {
				fmt.Fprintf(stderr, "error: node %q (conv1x1): expected 1 input\n", nd.ID)
				return 1
			}
			inD, ok := shapes[nd.Inputs[0]]
			if !ok {
				fmt.Fprintf(stderr, "error: node %q: unknown input %q\n", nd.ID, nd.Inputs[0])
				return 1
			}
			if nd.Cin != 0 && nd.Cin != inD.c {
				fmt.Fprintf(stderr, "error: node %q (conv1x1): cin %d != input channels %d\n", nd.ID, nd.Cin, inD.c)
				return 1
			}
			cin := inD.c
			if nd.Cout <= 0 {
				fmt.Fprintf(stderr, "error: node %q (conv1x1): cout must be > 0\n", nd.ID)
				return 1
			}
			wScale := math.Min(0.4, 1.5/math.Sqrt(float64(cin)))
			w := make([]uint16, nd.Cout*cin)
			for j := range w {
				w[j] = fp16.FromFloat64((rng.Float64()*2.0 - 1.0) * wScale)
			}
			wName := fmt.Sprintf("%s_weights.bin", nd.ID)
			if err := writeFile(wName, fp16.EncodeLE(w)); err != nil {
				fmt.Fprintf(stderr, "error: write %s: %v\n", wName, err)
				return 1
			}
			nd.Weights = wName
			nd.Cin = cin

			wantsBias, _, err := nd.ConvBiasSpec()
			if err != nil {
				fmt.Fprintf(stderr, "error: %v\n", err)
				return 1
			}
			var b []uint16
			if wantsBias {
				b = make([]uint16, nd.Cout)
				for j := range b {
					b[j] = fp16.FromFloat64((rng.Float64() - 0.5) * 0.5)
				}
				bName := fmt.Sprintf("%s_bias.bin", nd.ID)
				if err := writeFile(bName, fp16.EncodeLE(b)); err != nil {
					fmt.Fprintf(stderr, "error: write %s: %v\n", bName, err)
					return 1
				}
				rawB, _ := json.Marshal(bName)
				nd.Bias = rawB
			} else {
				nd.Bias = nil
			}
			nd.HasBias = false
			nodeWeights[nd.ID] = ref.GraphNodeWeights{Weights: w, Bias: b}
			shapes[nd.ID] = dims3{nd.Cout, inD.h, inD.w}

		case "conv3x3":
			if len(nd.Inputs) != 1 {
				fmt.Fprintf(stderr, "error: node %q (conv3x3): expected 1 input\n", nd.ID)
				return 1
			}
			inD, ok := shapes[nd.Inputs[0]]
			if !ok {
				fmt.Fprintf(stderr, "error: node %q: unknown input %q\n", nd.ID, nd.Inputs[0])
				return 1
			}
			mode := strings.ToLower(nd.Mode)
			if mode == "" {
				mode = "dense"
			}
			if nd.Cin != 0 && nd.Cin != inD.c {
				fmt.Fprintf(stderr, "error: node %q (conv3x3): cin %d != input channels %d\n", nd.ID, nd.Cin, inD.c)
				return 1
			}
			cin := inD.c
			cout := nd.Cout
			groups := 1
			if mode == "depthwise" {
				groups = cin
				if cout == 0 {
					cout = cin
				}
			}
			if cout <= 0 {
				fmt.Fprintf(stderr, "error: node %q (conv3x3): cout must be > 0\n", nd.ID)
				return 1
			}
			stride := nd.Stride
			if stride == 0 {
				stride = 1
			}
			pad := strings.ToLower(nd.Pad)
			if pad == "" {
				pad = "same"
			}
			ho, wo, _, _, err := ref.Conv3x3Geometry(inD.h, inD.w, stride, pad)
			if err != nil {
				fmt.Fprintf(stderr, "error: node %q: %v\n", nd.ID, err)
				return 1
			}
			cinPerG := cin / groups
			wScale := math.Min(0.35, 1.5/math.Sqrt(float64(cinPerG*9)))
			w := make([]uint16, cout*cinPerG*9)
			for j := range w {
				w[j] = fp16.FromFloat64((rng.Float64()*2.0 - 1.0) * wScale)
			}
			wName := fmt.Sprintf("%s_weights.bin", nd.ID)
			if err := writeFile(wName, fp16.EncodeLE(w)); err != nil {
				fmt.Fprintf(stderr, "error: write %s: %v\n", wName, err)
				return 1
			}
			nd.Weights = wName
			nd.Cin = cin
			nd.Cout = cout

			wantsBias, _, err := nd.ConvBiasSpec()
			if err != nil {
				fmt.Fprintf(stderr, "error: %v\n", err)
				return 1
			}
			var b []uint16
			if wantsBias {
				b = make([]uint16, cout)
				for j := range b {
					b[j] = fp16.FromFloat64((rng.Float64() - 0.5) * 0.5)
				}
				bName := fmt.Sprintf("%s_bias.bin", nd.ID)
				if err := writeFile(bName, fp16.EncodeLE(b)); err != nil {
					fmt.Fprintf(stderr, "error: write %s: %v\n", bName, err)
					return 1
				}
				rawB, _ := json.Marshal(bName)
				nd.Bias = rawB
			} else {
				nd.Bias = nil
			}
			nd.HasBias = false
			nodeWeights[nd.ID] = ref.GraphNodeWeights{Weights: w, Bias: b}
			shapes[nd.ID] = dims3{cout, ho, wo}

		case "pe":
			if len(nd.Inputs) == 1 {
				inD, ok := shapes[nd.Inputs[0]]
				if !ok {
					fmt.Fprintf(stderr, "error: node %q: unknown input %q\n", nd.ID, nd.Inputs[0])
					return 1
				}
				shapes[nd.ID] = inD
			} else if len(nd.Inputs) == 2 {
				d1, ok1 := shapes[nd.Inputs[0]]
				d2, ok2 := shapes[nd.Inputs[1]]
				if !ok1 || !ok2 {
					fmt.Fprintf(stderr, "error: node %q: unknown input in %v\n", nd.ID, nd.Inputs)
					return 1
				}
				bcast := func(a, b int, axis string) (int, error) {
					if a == b {
						return a, nil
					}
					if a == 1 {
						return b, nil
					}
					if b == 1 {
						return a, nil
					}
					return 0, fmt.Errorf("node %q (pe): incompatible broadcast on %s: %d vs %d", nd.ID, axis, a, b)
				}
				bc, err := bcast(d1.c, d2.c, "C")
				if err != nil {
					fmt.Fprintf(stderr, "error: %v\n", err)
					return 1
				}
				bh, err := bcast(d1.h, d2.h, "H")
				if err != nil {
					fmt.Fprintf(stderr, "error: %v\n", err)
					return 1
				}
				bw, err := bcast(d1.w, d2.w, "W")
				if err != nil {
					fmt.Fprintf(stderr, "error: %v\n", err)
					return 1
				}
				shapes[nd.ID] = dims3{bc, bh, bw}
			} else {
				fmt.Fprintf(stderr, "error: node %q (pe): expected 1 or 2 inputs\n", nd.ID)
				return 1
			}

		case "lut":
			if len(nd.Inputs) != 1 {
				fmt.Fprintf(stderr, "error: node %q (lut): expected 1 input\n", nd.ID)
				return 1
			}
			inD, ok := shapes[nd.Inputs[0]]
			if !ok {
				fmt.Fprintf(stderr, "error: node %q: unknown input %q\n", nd.ID, nd.Inputs[0])
				return 1
			}
			shapes[nd.ID] = inD

		case "max_pool", "avg_pool":
			if len(nd.Inputs) != 1 {
				fmt.Fprintf(stderr, "error: node %q (%s): expected 1 input\n", nd.ID, op)
				return 1
			}
			inD, ok := shapes[nd.Inputs[0]]
			if !ok {
				fmt.Fprintf(stderr, "error: node %q: unknown input %q\n", nd.ID, nd.Inputs[0])
				return 1
			}
			oh, ow, _, _, _, _, err := ref.PoolGeometry(inD.h, inD.w, ref.PoolSpec{
				Op: op, KH: nd.KH, KW: nd.KW, SH: nd.SH, SW: nd.SW,
				Pad: nd.Pad, PT: nd.PT, PB: nd.PB, PL: nd.PL, PR: nd.PR,
			})
			if err != nil {
				fmt.Fprintf(stderr, "error: node %q: %v\n", nd.ID, err)
				return 1
			}
			shapes[nd.ID] = dims3{inD.c, oh, ow}

		case "reduce":
			if len(nd.Inputs) != 1 {
				fmt.Fprintf(stderr, "error: node %q (reduce): expected 1 input\n", nd.ID)
				return 1
			}
			inD, ok := shapes[nd.Inputs[0]]
			if !ok {
				fmt.Fprintf(stderr, "error: node %q: unknown input %q\n", nd.ID, nd.Inputs[0])
				return 1
			}
			rc, rh, rw, err := ref.ParseReduceAxes(nd.Axes)
			if err != nil {
				fmt.Fprintf(stderr, "error: node %q: %v\n", nd.ID, err)
				return 1
			}
			oc, oh, ow := inD.c, inD.h, inD.w
			if rc {
				oc = 1
			}
			if rh {
				oh = 1
			}
			if rw {
				ow = 1
			}
			shapes[nd.ID] = dims3{oc, oh, ow}

		default:
			fmt.Fprintf(stderr, "error: node %q: unsupported op %q\n", nd.ID, nd.Op)
			return 1
		}
	}

	yQ, _, err := ref.EvalGraphQ1516(resolved, inTensors, nodeWeights)
	if err != nil {
		fmt.Fprintf(stderr, "error: Q15.16 graph reference failed: %v\n", err)
		return 1
	}
	yF, _, err := ref.EvalGraphF64(resolved, inTensors, nodeWeights)
	if err != nil {
		fmt.Fprintf(stderr, "error: float64 graph reference failed: %v\n", err)
		return 1
	}

	expQBuf, _, _, _, err := packTight64Surface(yQ)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack expected_q1516: %v\n", err)
		return 1
	}
	expFBuf, _, _, _, err := packTight64Surface(yF)
	if err != nil {
		fmt.Fprintf(stderr, "error: pack expected_f64: %v\n", err)
		return 1
	}

	resolvedJSON, err := json.MarshalIndent(resolved, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "error: marshal graph_resolved.json: %v\n", err)
		return 1
	}
	if err := writeFile("graph_resolved.json", append(resolvedJSON, '\n')); err != nil {
		fmt.Fprintf(stderr, "error: write graph_resolved.json: %v\n", err)
		return 1
	}
	if err := writeFile("expected_q1516.bin", expQBuf); err != nil {
		fmt.Fprintf(stderr, "error: write expected_q1516.bin: %v\n", err)
		return 1
	}
	if err := writeFile("expected_f64.bin", expFBuf); err != nil {
		fmt.Fprintf(stderr, "error: write expected_f64.bin: %v\n", err)
		return 1
	}

	summary := map[string]any{
		"target":      resolved.Target,
		"placement":   resolved.Placement,
		"node_count":  len(resolved.Nodes),
		"input_files": inFiles,
		"out_shape":   []int{yQ.N, yQ.C, yQ.H, yQ.W},
		"out_bytes":   len(expQBuf),
		"seed":        *seed,
		"resolved":    filepath.Join(*outDir, "graph_resolved.json"),
		"out_dir":     *outDir,
	}
	if *asJSON {
		raw, _ := json.MarshalIndent(summary, "", "  ")
		fmt.Fprintln(stdout, string(raw))
	} else {
		fmt.Fprintf(stdout, "generated graph (%d nodes, out=%dx%dx%d, %dB) in %s\n",
			len(resolved.Nodes), yQ.C, yQ.H, yQ.W, len(expQBuf), *outDir)
	}
	return 0
}

type manifestSurface struct {
	ShortName string `json:"short_name"`
	Direction string `json:"direction"`
	Shape     []int  `json:"shape"`
	Strides   []int  `json:"strides"`
	Span      int    `json:"span"`
}

type manifestDoc struct {
	Surfaces []manifestSurface `json:"surfaces"`
}

func cmdCompareConv(cmdName string, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cmdName, flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath := fs.String("manifest", "", "Path to manifest.json emitted by vane")
	actualPath := fs.String("actual", "", "Path to actual output surface binary")
	expectedPath := fs.String("expected", "", "Path to expected output surface binary")
	atol := fs.Float64("atol", 1e-3, "Absolute tolerance")
	rtol := fs.Float64("rtol", 1e-3, "Relative tolerance")
	maxULPVal := fs.Int("max-ulp", -1, "Maximum allowed fp16 ULP error (negative disables)")
	asJSON := fs.Bool("json", false, "Emit JSON comparison metrics")

	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *manifestPath == "" || *actualPath == "" || *expectedPath == "" {
		fmt.Fprintln(stderr, "error: --manifest, --actual, and --expected are all required")
		return 1
	}

	mBytes, err := os.ReadFile(*manifestPath)
	if err != nil {
		fmt.Fprintf(stderr, "error: read manifest %q: %v\n", *manifestPath, err)
		return 1
	}
	var doc manifestDoc
	if err := json.Unmarshal(mBytes, &doc); err != nil {
		fmt.Fprintf(stderr, "error: parse manifest %q: %v\n", *manifestPath, err)
		return 1
	}

	var outSurf *manifestSurface
	for i := range doc.Surfaces {
		if doc.Surfaces[i].Direction == "output" {
			outSurf = &doc.Surfaces[i]
			break
		}
	}
	if outSurf == nil {
		fmt.Fprintln(stderr, "error: manifest contains no output surface")
		return 1
	}
	if len(outSurf.Shape) != 4 || len(outSurf.Strides) != 3 {
		fmt.Fprintf(stderr, "error: invalid output surface shape %v or strides %v\n", outSurf.Shape, outSurf.Strides)
		return 1
	}

	actBytes, err := os.ReadFile(*actualPath)
	if err != nil {
		fmt.Fprintf(stderr, "error: read actual %q: %v\n", *actualPath, err)
		return 1
	}
	expBytes, err := os.ReadFile(*expectedPath)
	if err != nil {
		fmt.Fprintf(stderr, "error: read expected %q: %v\n", *expectedPath, err)
		return 1
	}

	n, c, h, w := outSurf.Shape[0], outSurf.Shape[1], outSurf.Shape[2], outSurf.Shape[3]
	bs, ps, rs := outSurf.Strides[0], outSurf.Strides[1], outSurf.Strides[2]

	actTensor, err := ref.UnpackSurface(actBytes, n, c, h, w, bs, ps, rs)
	if err != nil {
		fmt.Fprintf(stderr, "error: unpack actual surface: %v\n", err)
		return 1
	}
	expTensor, err := ref.UnpackSurface(expBytes, n, c, h, w, bs, ps, rs)
	if err != nil {
		fmt.Fprintf(stderr, "error: unpack expected surface: %v\n", err)
		return 1
	}

	var maxULP *int
	if *maxULPVal >= 0 {
		v := *maxULPVal
		maxULP = &v
	}

	res, err := ref.CompareFP16(actTensor.Data, expTensor.Data, *atol, *rtol, maxULP)
	if err != nil {
		fmt.Fprintf(stderr, "error: compare failed: %v\n", err)
		return 1
	}

	if *asJSON {
		raw, err := json.MarshalIndent(res, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "error: marshal comparison: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
	} else {
		fmt.Fprintf(stdout, "passed=%t max_abs=%.6g max_rel=%.6g rmse=%.6g max_ulp=%d exact=%d/%d nonfinite_mismatches=%d\n",
			res.Passed, res.MaxAbsErr, res.MaxRelErr, res.RMSE, res.MaxULPErr, res.ExactFP16Count, res.TotalElements, res.NonfiniteMismatches)
	}

	if !res.Passed {
		return 1
	}
	return 0
}

func main() {
	os.Exit(runCLI(os.Args[1:], ane.LinuxTransport{}, os.Stdout, os.Stderr))
}
