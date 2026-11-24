// Command vane-runner provides Linux accel/ane ABI v3 device probing, HWX execution,
// deterministic conv test vector generation, and strided surface comparison.
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
	"strings"

	"github.com/niteria/vane/internal/ane"
	"github.com/niteria/vane/internal/fp16"
	"github.com/niteria/vane/internal/ref"
)

type stringSliceFlag []string

func (s *stringSliceFlag) String() string {
	return strings.Join(*s, ",")
}

func (s *stringSliceFlag) Set(v string) error {
	*s = append(*s, v)
	return nil
}

func usage(w io.Writer) {
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  vane-runner probe [--device PATH] [--json]")
	fmt.Fprintln(w, "  vane-runner run --hwx PATH [--sections-dir DIR] [--input PATH]... [--output-out PATH]")
	fmt.Fprintln(w, "                  [--expected PATH] [--runs N] [--in-place] [--atol F] [--rtol F]")
	fmt.Fprintln(w, "                  [--max-ulp N] [--device PATH] [--json] [--json-out PATH]")
	fmt.Fprintln(w, "  vane-runner gen-conv1x1 --h H --w W --cin C --cout O [--bias] [--relu] [--seed N] --out-dir DIR [--json]")
	fmt.Fprintln(w, "  vane-runner gen-conv3x3 --mode <dense|depthwise> --h H --w W --cin C --cout O --stride <1|2> --pad <same|valid> [--bias] [--relu] [--seed N] --out-dir DIR [--json]")
	fmt.Fprintln(w, "  vane-runner compare-conv --manifest PATH --actual PATH --expected PATH [--atol F] [--rtol F] [--max-ulp N] [--json]")
	fmt.Fprintln(w, "  vane-runner compare-conv1x1 --manifest PATH --actual PATH --expected PATH [--atol F] [--rtol F] [--max-ulp N] [--json]")
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
	case "gen-conv1x1":
		return cmdGenConv1x1(args[1:], stdout, stderr)
	case "gen-conv3x3":
		return cmdGenConv3x3(args[1:], stdout, stderr)
	case "compare-conv":
		return cmdCompareConv("compare-conv", args[1:], stdout, stderr)
	case "compare-conv1x1":
		return cmdCompareConv("compare-conv1x1", args[1:], stdout, stderr)
	default:
		fmt.Fprintf(stderr, "error: unknown subcommand %q\n", args[0])
		usage(stderr)
		return 1
	}
}

func cmdProbe(args []string, tr ane.Transport, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("probe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	device := fs.String("device", "", "Explicit /dev/accel/accel* device path")
	asJSON := fs.Bool("json", false, "Emit JSON output")
	if err := fs.Parse(args); err != nil {
		return 1
	}

	devs, err := ane.ProbeDevices(tr, *device)
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
		fmt.Fprintln(stderr, "skip: no matching accel/ane ABI v3 device found")
		return ane.ExitSkip
	}
	if *asJSON {
		enc, _ := json.MarshalIndent(devs, "", "  ")
		fmt.Fprintln(stdout, string(enc))
		return 0
	}
	for _, d := range devs {
		fmt.Fprintf(stdout, "device=%s abi=v%d chip=H%d caps_size=%d\n", d.Path, d.ABIVersion, d.ChipFamily, d.Size)
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
	outputOut := fs.String("output-out", "", "Write output surface bytes to this path")
	expectedPath := fs.String("expected", "", "Expected output fp16 binary for direct byte comparison")
	runs := fs.Int("runs", 1, "Number of consecutive runs for repeatability verification")
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
		Runs:     *runs,
		InPlace:  *inPlace,
		Expected: expectedBytes,
		Atol:     *atol,
		Rtol:     *rtol,
		MaxULP:   maxULP,
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
		sumMS := 0.0
		for _, v := range report.LatenciesMS {
			sumMS += v
		}
		meanMS := sumMS / float64(len(report.LatenciesMS))
		fmt.Fprintf(stdout, "device=%s abi=v%d chip=H%d mode=%s runs=%d mean_ms=%.3f sha256=%s\n",
			report.Device, report.ABIVersion, report.ChipFamily, report.Mode, report.Runs, meanMS, report.OutputSHA256)
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

	rng := rand.New(rand.NewPCG(*seed, *seed^0x9e3779b97f4a7c15))

	x, err := ref.NewTensor(1, *cin, *h, *w)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	for i := range x.Data {
		// Uniform in [-1.0, 1.0] quantized to fp16
		x.Data[i] = fp16.FromFloat64(rng.Float64()*2.0 - 1.0)
	}

	// Scale weights inversely with sqrt(Cin) so accumulations stay well within Q15.16 & fp16 range
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

	yF64, err := ref.Conv2d1x1(x, weights, *cout, bias, *relu)
	if err != nil {
		fmt.Fprintf(stderr, "error: float64 reference failed: %v\n", err)
		return 1
	}
	yQ1516, err := ref.Conv2d1x1Q1516(x, weights, *cout, bias, *relu)
	if err != nil {
		fmt.Fprintf(stderr, "error: Q15.16 reference failed: %v\n", err)
		return 1
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

	summary := map[string]any{
		"h":               *h,
		"w":               *w,
		"cin":             *cin,
		"cout":            *cout,
		"bias":            *withBias,
		"relu":            *relu,
		"seed":            *seed,
		"row_stride":      rowStride,
		"plane_stride":    planeStride,
		"in_bytes":        len(inSurface),
		"out_bytes":       len(expF64Surface),
		"weights_bytes":   len(weights) * 2,
		"bias_bytes":      len(bias) * 2,
		"out_dir":         *outDir,
	}
	if *asJSON {
		raw, _ := json.MarshalIndent(summary, "", "  ")
		fmt.Fprintln(stdout, string(raw))
	} else {
		fmt.Fprintf(stdout, "generated conv1x1 (%dx%d, %d->%d, bias=%t, relu=%t, seed=%d) in %s (in=%dB out=%dB weights=%dB)\n",
			*h, *w, *cin, *cout, *withBias, *relu, *seed, *outDir, len(inSurface), len(expF64Surface), len(weights)*2)
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

	yF64, err := ref.Conv2d3x3Pad(x, weights, *cout, bias, *stride, *pad, groups, *relu)
	if err != nil {
		fmt.Fprintf(stderr, "error: float64 reference failed: %v\n", err)
		return 1
	}
	yQ1516, err := ref.Conv2d3x3Q1516(x, weights, *cout, bias, *stride, *pad, groups, *relu)
	if err != nil {
		fmt.Fprintf(stderr, "error: Q15.16 reference failed: %v\n", err)
		return 1
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
		"relu":             *relu,
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
		fmt.Fprintf(stdout, "generated conv3x3 %s (%dx%d->%dx%d, %d->%d, s=%d, pad=%s, bias=%t, relu=%t, seed=%d) in %s (in=%dB out=%dB weights=%dB)\n",
			*mode, *h, *w, ho, wo, *cin, *cout, *stride, *pad, *withBias, *relu, *seed, *outDir, len(inSurface), len(expF64Surface), len(weights)*2)
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
	manifestPath := fs.String("manifest", "", "Path to manifest.json emitted by vane compile-conv1x1 or compile-conv3x3")
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
