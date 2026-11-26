package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/aurora-silicon/vane/internal/frontend"
)

func cmdImportONNX(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("import-onnx", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "Path to binary .onnx model file")
	outDir := fs.String("out", "", "Output directory for graph.json and fp16 weight .bin files")
	target := fs.String("target", "h17", "Target chip profile (h16 or h17)")
	placement := fs.String("placement", "auto", "Default graph edge placement (auto, l2, or dram)")
	var inputShapeFlags stringSliceFlag
	fs.Var(&inputShapeFlags, "input-shape", "Static input shape override NAME=C,H,W (repeatable)")
	sfPath := fs.String("safetensors", "", "Optional .safetensors archive supplying/overriding initializer weights")
	noFuse := fs.Bool("no-fuse", false, "Disable Conv+BN folding and Conv+Activation/Silu fusion")
	allowViewOps := fs.Bool("allow-view-ops", false, "Allow emitting non-no-op view ops (transpose/concat/reshape/slice)")
	asJSON := fs.Bool("json", false, "Emit JSON summary")

	if err := fs.Parse(args); err != nil {
		return 1
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "error: unexpected positional arguments: %s\n", strings.Join(fs.Args(), " "))
		return 1
	}
	if *modelPath == "" {
		fmt.Fprintln(stderr, "error: --model PATH is required")
		return 1
	}
	if *outDir == "" {
		fmt.Fprintln(stderr, "error: --out DIR is required")
		return 1
	}

	inputShapes := make(map[string][3]int, len(inputShapeFlags))
	for _, raw := range inputShapeFlags {
		name, dims, err := frontend.ParseInputShapeOverride(raw)
		if err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
		inputShapes[name] = dims
	}

	var sf *frontend.SafeTensorsFile
	if *sfPath != "" {
		var err error
		sf, err = frontend.LoadSafeTensors(*sfPath)
		if err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
	}

	model, err := frontend.LoadONNX(*modelPath)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	res, err := frontend.LowerONNX(model, frontend.LowerOptions{
		Target:       *target,
		Placement:    *placement,
		InputShapes:  inputShapes,
		SafeTensors:  sf,
		NoFuse:       *noFuse,
		AllowViewOps: *allowViewOps,
	})
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	if err := frontend.WriteLoweredGraph(*outDir, res); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	type importReport struct {
		frontend.LowerSummary
		OutDir    string `json:"out_dir"`
		GraphFile string `json:"graph_file"`
	}
	rep := importReport{
		LowerSummary: res.Summary,
		OutDir:       *outDir,
		GraphFile:    filepath.Join(*outDir, "graph.json"),
	}

	if *asJSON {
		raw, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "error: marshal summary: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
		return 0
	}

	inDesc := make([]string, len(res.Summary.Inputs))
	for i, inp := range res.Summary.Inputs {
		inDesc[i] = fmt.Sprintf("%s=[1,%d,%d,%d]", inp.Name, inp.C, inp.H, inp.W)
	}
	outShape := res.Summary.OutputShape
	fmt.Fprintf(stdout, "imported onnx -> %s (target=%s inputs=%s output=%s[%d,%d,%d,%d] nodes=%d fused_bn=%d fused_act=%d elided=%d weights=%d/%dB)\n",
		rep.GraphFile, res.Summary.Target, strings.Join(inDesc, ","),
		res.Summary.Output, outShape[0], outShape[1], outShape[2], outShape[3],
		res.Summary.NodeCount, res.Summary.FusedBNCount, res.Summary.FusedActCount,
		res.Summary.ElidedCount, res.Summary.WeightFiles, res.Summary.WeightBytes)
	return 0
}

func cmdInspectSafeTensors(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("inspect-safetensors", flag.ContinueOnError)
	fs.SetOutput(stderr)
	filePath := fs.String("file", "", "Path to .safetensors file")
	exportDir := fs.String("export-dir", "", "Optional output directory to export float tensors as fp16 .bin files")
	strict := fs.Bool("strict", false, "Reject overlapping tensor byte spans")
	asJSON := fs.Bool("json", false, "Emit JSON report")

	if err := fs.Parse(args); err != nil {
		return 1
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "error: unexpected positional arguments: %s\n", strings.Join(fs.Args(), " "))
		return 1
	}
	if *filePath == "" {
		fmt.Fprintln(stderr, "error: --file PATH is required")
		return 1
	}

	sf, err := frontend.LoadSafeTensorsWithOptions(*filePath, frontend.SafeTensorsOptions{Strict: *strict})
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}

	var exported []frontend.ExportedTensor
	if *exportDir != "" {
		exported, err = sf.ExportFP16Dir(*exportDir)
		if err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
	}

	tensors := make([]frontend.SafeTensorInfo, 0, len(sf.Names))
	var totalBytes int64
	for _, name := range sf.Names {
		info := sf.Tensors[name]
		tensors = append(tensors, info)
		totalBytes += info.ByteLen
	}

	type inspectReport struct {
		File       string                    `json:"file"`
		TensorCnt  int                       `json:"tensor_count"`
		TotalBytes int64                     `json:"total_bytes"`
		Metadata   map[string]string         `json:"metadata,omitempty"`
		Tensors    []frontend.SafeTensorInfo `json:"tensors"`
		Exported   []frontend.ExportedTensor `json:"exported,omitempty"`
	}
	rep := inspectReport{
		File:       *filePath,
		TensorCnt:  len(tensors),
		TotalBytes: totalBytes,
		Metadata:   sf.Metadata,
		Tensors:    tensors,
		Exported:   exported,
	}

	if *asJSON {
		raw, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "error: marshal inspect report: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
		return 0
	}

	fmt.Fprintf(stdout, "safetensors: %s (%d tensors, %d bytes)\n", *filePath, len(tensors), totalBytes)
	for _, info := range tensors {
		fmt.Fprintf(stdout, "  %-32s dtype=%-4s shape=%v bytes=%d\n", info.Name, info.DType, info.Shape, info.ByteLen)
	}
	if len(exported) > 0 {
		fmt.Fprintf(stdout, "exported %d fp16 tensor files to %s\n", len(exported), *exportDir)
	}
	return 0
}
