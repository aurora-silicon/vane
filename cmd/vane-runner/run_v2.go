package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/aurora-silicon/vane/internal/ane"
)

func cmdRunV2(args []string, tr ane.Transport, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("run-v2", flag.ContinueOnError)
	fs.SetOutput(stderr)
	bundlePath := fs.String("bundle", "", "Path to ABI v2 JSON bundle manifest")
	var inputPaths stringSliceFlag
	fs.Var(&inputPaths, "input", "Input binary file (repeatable in bundle input IO order)")
	var altInputPaths stringSliceFlag
	fs.Var(&altInputPaths, "alt-input", "Alternate input binary file used on odd runs when --runs > 1 (repeatable)")
	outputOut := fs.String("output-out", "", "Write primary output buffer bytes to this path")
	expectedPath := fs.String("expected", "", "Expected output fp16 binary for direct byte comparison")
	priority := fs.Int("priority", 0, "Override bundle execution priority (2..7)")
	timeoutMS := fs.Int("timeout-ms", 0, "Override bundle completion timeout in ms (0..5000; 0 uses kernel default 5000)")
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
	if *bundlePath == "" {
		fmt.Fprintln(stderr, "error: --bundle PATH is required")
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

	visited := map[string]bool{}
	fs.Visit(func(f *flag.Flag) {
		visited[f.Name] = true
	})

	bundle, err := ane.LoadV2Bundle(*bundlePath)
	if err != nil {
		fmt.Fprintf(stderr, "error: load bundle %q: %v\n", *bundlePath, err)
		return 1
	}
	if visited["priority"] {
		if *priority < 0 || *priority > math.MaxUint32 {
			fmt.Fprintf(stderr, "error: --priority %d out of range [%d, %d]\n", *priority, ane.MinPriorityV2, ane.MaxPriorityV2)
			return 1
		}
		bundle.Priority = uint32(*priority)
	}
	if visited["timeout-ms"] {
		if *timeoutMS < 0 || *timeoutMS > math.MaxUint32 {
			fmt.Fprintf(stderr, "error: --timeout-ms %d out of range [0, %d]\n", *timeoutMS, ane.MaxTimeoutMSV2)
			return 1
		}
		bundle.TimeoutMS = uint32(*timeoutMS)
	}
	if err := ane.ValidateV2Bundle(bundle); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
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

	fd, caps, err := ane.DiscoverDeviceABI(tr, *device, ane.ABIV2, ane.ChipH14)
	if err != nil {
		if errors.Is(err, ane.ErrNoMatchingDevice) {
			fmt.Fprintf(stderr, "skip: %v\n", err)
			return ane.ExitSkip
		}
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	defer tr.Close(fd)

	outBytes, report, err := ane.ExecuteV2(tr, fd, caps, bundle, inputPayloads, ane.ExecOptions{
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
