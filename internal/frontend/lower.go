package frontend

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/aurora-silicon/vane/internal/fp16"
	"github.com/aurora-silicon/vane/internal/ref"
)

// VaneGraphInput matches Vane's compile-graph JSON input descriptor.
type VaneGraphInput struct {
	Name      string `json:"name"`
	C         int    `json:"c"`
	H         int    `json:"h"`
	W         int    `json:"w"`
	ShortName string `json:"short_name,omitempty"`
}

// VaneGraphNode matches Vane's compile-graph JSON node descriptor.
type VaneGraphNode struct {
	ID           string            `json:"id"`
	Op           string            `json:"op"`
	Inputs       []string          `json:"inputs"`
	OutPlacement string            `json:"out_placement,omitempty"`
	Mode         string            `json:"mode,omitempty"`
	Cin          int               `json:"cin,omitempty"`
	Cout         int               `json:"cout,omitempty"`
	Stride       int               `json:"stride,omitempty"`
	Pad          string            `json:"pad,omitempty"`
	Weights      string            `json:"weights,omitempty"`
	Bias         string            `json:"bias,omitempty"`
	Relu         bool              `json:"relu,omitempty"`
	Act          *ref.GraphActSpec `json:"act,omitempty"`
	PeOp         string            `json:"pe_op,omitempty"`
	Alpha        *float64          `json:"alpha,omitempty"`
	Beta         *float64          `json:"beta,omitempty"`
	Scalar       *float64          `json:"scalar,omitempty"`
	KH           int               `json:"kh,omitempty"`
	KW           int               `json:"kw,omitempty"`
	SH           int               `json:"sh,omitempty"`
	SW           int               `json:"sw,omitempty"`
	PT           int               `json:"pt,omitempty"`
	PB           int               `json:"pb,omitempty"`
	PL           int               `json:"pl,omitempty"`
	PR           int               `json:"pr,omitempty"`
	ExcludePad   *bool             `json:"exclude_pad,omitempty"`
	ReduceOp     string            `json:"reduce_op,omitempty"`
	Axes         string            `json:"axes,omitempty"`
	Eps          *float64          `json:"eps,omitempty"`
	Perm         []int             `json:"perm,omitempty"`
	Begin        []int             `json:"begin,omitempty"`
	End          []int             `json:"end,omitempty"`
	Step         []int             `json:"step,omitempty"`
	Axis         *int              `json:"axis,omitempty"`
	OutShape     []int             `json:"out_shape,omitempty"`
}

// VaneGraphSpec is the top-level compile-graph JSON specification emitted by LowerONNX.
type VaneGraphSpec struct {
	Target    string           `json:"target"`
	Placement string           `json:"placement,omitempty"`
	Inputs    []VaneGraphInput `json:"inputs"`
	Nodes     []VaneGraphNode  `json:"nodes"`
	Output    string           `json:"output,omitempty"`
}

// LowerOptions configures ONNX -> Vane graph lowering.
type LowerOptions struct {
	// Target specifies the Vane compilation target ("h16" or "h17"; default "h17").
	Target string
	// Placement specifies the default edge placement ("auto", "l2", "dram"; default "auto").
	Placement string
	// InputShapes maps ONNX input tensor names to explicit [C, H, W] overrides.
	InputShapes map[string][3]int
	// SafeTensors optionally overrides ONNX initializers by matching tensor name.
	SafeTensors *SafeTensorsFile
	// NoFuse disables Conv+BN folding, Conv+Activation fusion, and Silu/MatMul+Add pattern fusion.
	NoFuse bool
	// AllowViewOps opts into emitting non-no-op view nodes (transpose, concat, reshape, slice).
	// True no-op views (identity transpose, same-shape reshape/flatten) are always elided.
	AllowViewOps bool
}

// LowerSummary reports statistics about a lowered ONNX graph.
type LowerSummary struct {
	Target          string           `json:"target"`
	Placement       string           `json:"placement"`
	Inputs          []VaneGraphInput `json:"inputs"`
	Output          string           `json:"output"`
	OutputShape     [4]int           `json:"output_shape"`
	ONNXNodeCount   int              `json:"onnx_node_count"`
	NodeCount       int              `json:"node_count"`
	FusedBNCount    int              `json:"fused_bn_count"`
	FusedActCount   int              `json:"fused_act_count"`
	ElidedCount     int              `json:"elided_count"`
	WeightFiles     int              `json:"weight_files"`
	WeightBytes     int64            `json:"weight_bytes"`
	SafeTensorsUsed int              `json:"safetensors_used"`
}

// LowerResult holds the lowered Vane graph spec, fp16 weight binary blobs, and summary.
type LowerResult struct {
	Spec    VaneGraphSpec
	Weights map[string][]byte
	Summary LowerSummary
}

// ParseInputShapeOverride parses a CLI --input-shape specification of the form
// "name=C,H,W" or "name=1,C,H,W".
func ParseInputShapeOverride(raw string) (string, [3]int, error) {
	eq := strings.IndexByte(raw, '=')
	if eq <= 0 || eq == len(raw)-1 {
		return "", [3]int{}, fmt.Errorf("invalid --input-shape %q (expected name=C,H,W)", raw)
	}
	name := strings.TrimSpace(raw[:eq])
	parts := strings.Split(strings.TrimSpace(raw[eq+1:]), ",")
	var dims []int
	for _, p := range parts {
		v, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil || v <= 0 {
			return "", [3]int{}, fmt.Errorf("invalid --input-shape %q: dimension %q must be a positive integer", raw, p)
		}
		dims = append(dims, v)
	}
	switch len(dims) {
	case 3:
		return name, [3]int{dims[0], dims[1], dims[2]}, nil
	case 4:
		if dims[0] != 1 {
			return "", [3]int{}, fmt.Errorf("invalid --input-shape %q: batch dimension N must be 1, got %d", raw, dims[0])
		}
		return name, [3]int{dims[1], dims[2], dims[3]}, nil
	default:
		return "", [3]int{}, fmt.Errorf("invalid --input-shape %q (expected name=C,H,W or name=1,C,H,W)", raw)
	}
}

// WriteLoweredGraph writes deterministic graph.json and fp16 weight .bin files into outDir.
func WriteLoweredGraph(outDir string, res *LowerResult) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("create output dir %q: %w", outDir, err)
	}
	wNames := make([]string, 0, len(res.Weights))
	for k := range res.Weights {
		wNames = append(wNames, k)
	}
	sort.Strings(wNames)
	for _, fname := range wNames {
		if err := os.WriteFile(filepath.Join(outDir, fname), res.Weights[fname], 0o644); err != nil {
			return fmt.Errorf("write weight file %q: %w", fname, err)
		}
	}
	rawJSON, err := json.MarshalIndent(res.Spec, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal graph.json: %w", err)
	}
	if err := os.WriteFile(filepath.Join(outDir, "graph.json"), append(rawJSON, '\n'), 0o644); err != nil {
		return fmt.Errorf("write graph.json: %w", err)
	}
	return nil
}

// constSource abstracts a constant tensor coming from an ONNX initializer,
// Constant node attribute, or SafeTensors override.
type constSource struct {
	name      string
	dims      []int64
	onnx      *ONNXTensor
	sf        *SafeTensorsFile
	inlineF64 []float64
	inlineI64 []int64
}

func (c *constSource) float64s() ([]float64, error) {
	if c.inlineF64 != nil {
		out := make([]float64, len(c.inlineF64))
		copy(out, c.inlineF64)
		return out, nil
	}
	if c.sf != nil {
		return c.sf.ToFloat64(c.name)
	}
	if c.onnx != nil {
		return c.onnx.ToFloat64()
	}
	return nil, fmt.Errorf("constant %q has no float data", c.name)
}

func (c *constSource) int64s() ([]int64, error) {
	if c.inlineI64 != nil {
		out := make([]int64, len(c.inlineI64))
		copy(out, c.inlineI64)
		return out, nil
	}
	if c.sf != nil {
		return c.sf.ToInt64(c.name)
	}
	if c.onnx != nil {
		return c.onnx.ToInt64()
	}
	return nil, fmt.Errorf("constant %q has no integer data", c.name)
}

func (c *constSource) isScalar() (bool, error) {
	n, err := shapeProductChecked(c.dims)
	if err != nil {
		return false, err
	}
	return n == 1, nil
}

// irNode is a mutable intermediate node used during fusion before emitting Vane JSON.
type irNode struct {
	orig      ONNXNode
	opType    string
	name      string
	inputs    []string
	output    string
	foldedW   []float64
	foldedWSh []int64
	foldedB   []float64
	hasFoldB  bool
	fusedRelu bool
	fusedAct  *ref.GraphActSpec
}

func sanitizeNodeID(raw string, used map[string]bool) string {
	var b strings.Builder
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			b.WriteByte(c)
		} else {
			b.WriteByte('_')
		}
	}
	base := strings.Trim(b.String(), "_")
	if base == "" {
		base = "node"
	}
	if !used[base] {
		used[base] = true
		return base
	}
	for i := 2; ; i++ {
		cand := fmt.Sprintf("%s_%d", base, i)
		if !used[cand] {
			used[cand] = true
			return cand
		}
	}
}

func resolveAlias(alias map[string]string, name string) string {
	cur := name
	for steps := 0; steps < 256; steps++ {
		nxt, ok := alias[cur]
		if !ok || nxt == cur {
			return cur
		}
		cur = nxt
	}
	return cur
}

func sameShape64(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func extractConstantNode(nd ONNXNode) (*constSource, error) {
	if len(nd.Outputs) != 1 || nd.Outputs[0] == "" {
		return nil, fmt.Errorf("Constant node %q: expected 1 output", nd.Name)
	}
	outName := nd.Outputs[0]
	if t, ok := nd.AttrTensor("value"); ok {
		cp := *t
		if cp.Name == "" {
			cp.Name = outName
		}
		return &constSource{name: outName, dims: cp.Dims, onnx: &cp}, nil
	}
	if a, ok := nd.Attr("value_float"); ok {
		return &constSource{name: outName, dims: []int64{}, inlineF64: []float64{float64(a.F)}}, nil
	}
	if a, ok := nd.Attr("value_int"); ok {
		return &constSource{name: outName, dims: []int64{}, inlineI64: []int64{a.I}}, nil
	}
	if fs, ok := nd.AttrFloats("value_floats"); ok {
		out := make([]float64, len(fs))
		for i, f := range fs {
			out[i] = float64(f)
		}
		return &constSource{name: outName, dims: []int64{int64(len(fs))}, inlineF64: out}, nil
	}
	if is, ok := nd.AttrInts("value_ints"); ok {
		cp := make([]int64, len(is))
		copy(cp, is)
		return &constSource{name: outName, dims: []int64{int64(len(is))}, inlineI64: cp}, nil
	}
	return nil, fmt.Errorf("Constant node %q: missing supported value attribute", nd.Name)
}

func nodeLabel(nd *irNode, idx int) string {
	if nd.name != "" {
		return nd.name
	}
	if nd.output != "" {
		return nd.output
	}
	return fmt.Sprintf("%s_%d", strings.ToLower(nd.opType), idx)
}

// LowerONNX lowers a decoded ONNXModel into a Vane compile-graph specification
// and little-endian fp16 weight/bias binary blobs.
func LowerONNX(model *ONNXModel, opts LowerOptions) (*LowerResult, error) {
	if model == nil || !model.HasGraph {
		return nil, fmt.Errorf("onnx model has no graph")
	}
	target := strings.ToLower(strings.TrimSpace(opts.Target))
	if target == "" {
		target = "h17"
	}
	if target != "h16" && target != "h17" {
		return nil, fmt.Errorf("unsupported target %q (expected h16 or h17)", target)
	}
	placement := strings.ToLower(strings.TrimSpace(opts.Placement))
	if placement == "" {
		placement = "auto"
	}
	if placement != "auto" && placement != "l2" && placement != "dram" {
		return nil, fmt.Errorf("unsupported placement %q (expected auto, l2, or dram)", placement)
	}

	g := model.Graph
	if len(g.Outputs) != 1 {
		return nil, fmt.Errorf("graph %q: expected exactly 1 output, got %d", g.Name, len(g.Outputs))
	}

	// 1. Collect initializers (with optional SafeTensors overrides).
	consts := make(map[string]*constSource, len(g.Initializers))
	initNames := make(map[string]bool, len(g.Initializers))
	sfUsed := 0
	for i := range g.Initializers {
		t := &g.Initializers[i]
		initNames[t.Name] = true
		if opts.SafeTensors != nil {
			if sfInfo, ok := opts.SafeTensors.Tensors[t.Name]; ok {
				if !sameShape64(t.Dims, sfInfo.Shape) {
					return nil, fmt.Errorf("safetensors override for initializer %q: shape %v != ONNX shape %v",
						t.Name, sfInfo.Shape, t.Dims)
				}
				consts[t.Name] = &constSource{
					name: t.Name,
					dims: sfInfo.Shape,
					sf:   opts.SafeTensors,
				}
				sfUsed++
				continue
			}
		}
		consts[t.Name] = &constSource{
			name: t.Name,
			dims: t.Dims,
			onnx: t,
		}
	}

	// Also expose SafeTensors entries that aren't listed as initializers (for external weight archives).
	if opts.SafeTensors != nil {
		for _, name := range opts.SafeTensors.Names {
			if _, exists := consts[name]; !exists {
				info := opts.SafeTensors.Tensors[name]
				consts[name] = &constSource{
					name: name,
					dims: info.Shape,
					sf:   opts.SafeTensors,
				}
			}
		}
	}

	// 2. Resolve runtime graph inputs (excluding inputs supplied as initializers or SafeTensors constants).
	var graphInputs []VaneGraphInput
	shapes := make(map[string][4]int)
	vaneIDOf := make(map[string]string)
	usedIDs := make(map[string]bool)

	for _, inp := range g.Inputs {
		if initNames[inp.Name] {
			continue
		}
		if _, isConst := consts[inp.Name]; isConst {
			continue
		}
		var c, h, w int
		if ov, ok := opts.InputShapes[inp.Name]; ok {
			c, h, w = ov[0], ov[1], ov[2]
		} else {
			if !inp.HasShape || len(inp.Dims) == 0 {
				return nil, fmt.Errorf("input %q has unknown shape (provide --input-shape %s=C,H,W)", inp.Name, inp.Name)
			}
			switch len(inp.Dims) {
			case 4:
				for axis, d := range inp.Dims {
					if !d.HasValue || d.Value <= 0 {
						return nil, fmt.Errorf("input %q has dynamic/invalid dim[%d] (%q) (provide --input-shape %s=C,H,W)",
							inp.Name, axis, d.Param, inp.Name)
					}
				}
				if inp.Dims[0].Value != 1 {
					return nil, fmt.Errorf("input %q batch dimension N=%d != 1 (only N=1 is supported; override with --input-shape %s=C,H,W)",
						inp.Name, inp.Dims[0].Value, inp.Name)
				}
				c, h, w = int(inp.Dims[1].Value), int(inp.Dims[2].Value), int(inp.Dims[3].Value)
			case 2:
				for axis, d := range inp.Dims {
					if !d.HasValue || d.Value <= 0 {
						return nil, fmt.Errorf("input %q has dynamic/invalid dim[%d] (%q) (provide --input-shape %s=C,H,W)",
							inp.Name, axis, d.Param, inp.Name)
					}
				}
				if inp.Dims[0].Value != 1 {
					return nil, fmt.Errorf("input %q batch dimension N=%d != 1", inp.Name, inp.Dims[0].Value)
				}
				c, h, w = int(inp.Dims[1].Value), 1, 1
			default:
				return nil, fmt.Errorf("input %q has unsupported rank %d (expected 4D [1,C,H,W] or override with --input-shape %s=C,H,W)",
					inp.Name, len(inp.Dims), inp.Name)
			}
		}

		id := sanitizeNodeID(inp.Name, usedIDs)
		graphInputs = append(graphInputs, VaneGraphInput{Name: id, C: c, H: h, W: w})
		shapes[inp.Name] = [4]int{1, c, h, w}
		vaneIDOf[inp.Name] = id
	}
	if len(graphInputs) == 0 {
		return nil, fmt.Errorf("graph %q has no runtime inputs", g.Name)
	}
	if len(graphInputs) > 3 {
		return nil, fmt.Errorf("graph %q has %d runtime inputs (Vane supports 1..3 inputs)", g.Name, len(graphInputs))
	}

	// 3. Stage 1: Elide Constant, Identity, Dropout and build mutable IR node list.
	alias := make(map[string]string)
	var irNodes []*irNode
	elidedCount := 0

	for _, nd := range g.Nodes {
		switch nd.OpType {
		case "Constant":
			cs, err := extractConstantNode(nd)
			if err != nil {
				return nil, err
			}
			if existing, ok := consts[cs.name]; ok && existing.sf != nil {
				if !sameShape64(cs.dims, existing.dims) {
					return nil, fmt.Errorf("safetensors override for Constant %q: shape %v != ONNX Constant shape %v",
						cs.name, existing.dims, cs.dims)
				}
			} else {
				consts[cs.name] = cs
			}
			elidedCount++
			continue
		case "Identity":
			if len(nd.Inputs) < 1 || len(nd.Outputs) < 1 {
				return nil, fmt.Errorf("Identity node %q: expected 1 input and 1 output", nd.Name)
			}
			alias[nd.Outputs[0]] = resolveAlias(alias, nd.Inputs[0])
			elidedCount++
			continue
		case "Dropout":
			if len(nd.Inputs) < 1 || len(nd.Outputs) < 1 {
				return nil, fmt.Errorf("Dropout node %q: expected at least 1 input and 1 output", nd.Name)
			}
			alias[nd.Outputs[0]] = resolveAlias(alias, nd.Inputs[0])
			elidedCount++
			continue
		}

		if len(nd.Outputs) == 0 || nd.Outputs[0] == "" {
			return nil, fmt.Errorf("node %q (%s): missing output tensor name", nd.Name, nd.OpType)
		}
		inps := make([]string, len(nd.Inputs))
		for i, in := range nd.Inputs {
			if in != "" {
				inps[i] = resolveAlias(alias, in)
			}
		}
		irNodes = append(irNodes, &irNode{
			orig:   nd,
			opType: nd.OpType,
			name:   nd.Name,
			inputs: inps,
			output: nd.Outputs[0],
		})
	}

	graphOutput := resolveAlias(alias, g.Outputs[0].Name)

	// 4. Prune nodes unreachable from graphOutput while preserving topological order.
	irNodes = pruneUnreachable(irNodes, graphOutput)

	// 5. Pattern fusion passes (unless NoFuse).
	fusedBNCount := 0
	fusedActCount := 0
	if !opts.NoFuse {
		// 5a. Collapse Mul(x, Sigmoid(x)) / Mul(Sigmoid(x), x) -> synthetic Silu(x).
		irNodes = collapseSiluPatterns(irNodes, graphOutput)

		// 5b. Fold Conv + BatchNormalization algebraically in float64.
		var err error
		irNodes, fusedBNCount, err = foldConvBatchNorm(irNodes, consts, graphOutput)
		if err != nil {
			return nil, err
		}

		// 5c. Fold MatMul(A, const B) + Add(_, const bias) -> Gemm-equivalent linear with bias.
		irNodes, err = foldMatMulAddBias(irNodes, consts, graphOutput)
		if err != nil {
			return nil, err
		}

		// 5d. Fuse Conv / Gemm / MatMul + single-consumer Activation (Relu/Gelu/Sigmoid/Tanh/Silu).
		irNodes, fusedActCount = fuseConvActivations(irNodes, graphOutput)
	}

	// Track actual SafeTensors usage for tensors outside initializers if referenced by live nodes.
	sfCounted := make(map[string]bool)
	for _, init := range g.Initializers {
		if opts.SafeTensors != nil {
			if _, ok := opts.SafeTensors.Tensors[init.Name]; ok {
				sfCounted[init.Name] = true
			}
		}
	}
	if opts.SafeTensors != nil {
		for _, nd := range irNodes {
			for _, in := range nd.inputs {
				if in == "" || sfCounted[in] {
					continue
				}
				if _, ok := opts.SafeTensors.Tensors[in]; ok {
					sfCounted[in] = true
					sfUsed++
				}
			}
		}
	}

	// 6. Emit Vane graph nodes + static shape inference + fp16 weights.
	opset := model.DefaultOpsetVersion()
	weights := make(map[string][]byte)
	var vaneNodes []VaneGraphNode

	addWeightFile := func(nodeID, suffix string, vals []uint16) string {
		fname := fmt.Sprintf("%s_%s.bin", nodeID, suffix)
		weights[fname] = fp16.EncodeLE(vals)
		return fname
	}

	lookupShape := func(tensorName, ctx string) ([4]int, string, error) {
		sh, ok := shapes[tensorName]
		if !ok {
			return [4]int{}, "", fmt.Errorf("%s: unknown runtime tensor %q (missing producer or constant)", ctx, tensorName)
		}
		vid, ok := vaneIDOf[tensorName]
		if !ok {
			return [4]int{}, "", fmt.Errorf("%s: internal error: no Vane ID for tensor %q", ctx, tensorName)
		}
		return sh, vid, nil
	}

	emitNode := func(nd VaneGraphNode, onnxOut string, outShape [4]int) {
		vaneNodes = append(vaneNodes, nd)
		if onnxOut != "" {
			shapes[onnxOut] = outShape
			vaneIDOf[onnxOut] = nd.ID
		}
	}

	for idx, nd := range irNodes {
		ctx := fmt.Sprintf("node %q (%s)", nodeLabel(nd, idx), nd.opType)
		baseID := sanitizeNodeID(nodeLabel(nd, idx), usedIDs)

		switch nd.opType {
		case "Conv":
			if len(nd.inputs) < 2 {
				return nil, fmt.Errorf("%s: expected at least 2 inputs (X, W)", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			cin, h, w := inSh[1], inSh[2], inSh[3]

			var wF64 []float64
			var wDims []int64
			if nd.foldedW != nil {
				wF64 = nd.foldedW
				wDims = nd.foldedWSh
			} else {
				wc, ok := consts[nd.inputs[1]]
				if !ok {
					return nil, fmt.Errorf("%s: weight input %q must be a constant initializer", ctx, nd.inputs[1])
				}
				wDims = wc.dims
				wF64, err = wc.float64s()
				if err != nil {
					return nil, fmt.Errorf("%s: weight %q: %w", ctx, nd.inputs[1], err)
				}
			}
			if len(wDims) != 4 {
				return nil, fmt.Errorf("%s: weight must be 4D [Cout,Cin/G,kH,kW], got shape %v", ctx, wDims)
			}
			cout, cinPerG, kh, kw := int(wDims[0]), int(wDims[1]), int(wDims[2]), int(wDims[3])
			group := int(nd.orig.AttrInt("group", 1))
			if group <= 0 || cin%group != 0 || cout%group != 0 || cin/group != cinPerG {
				return nil, fmt.Errorf("%s: incompatible Conv channels cin=%d cout=%d group=%d weight_cin_per_g=%d",
					ctx, cin, cout, group, cinPerG)
			}

			// Validate dilations
			if dils, ok := nd.orig.AttrInts("dilations"); ok {
				if len(dils) != 2 || dils[0] != 1 || dils[1] != 1 {
					return nil, fmt.Errorf("%s: unsupported dilations %v (only [1,1] is supported)", ctx, dils)
				}
			}
			// Read strides
			sh, sw := 1, 1
			if st, ok := nd.orig.AttrInts("strides"); ok {
				if len(st) != 2 {
					return nil, fmt.Errorf("%s: strides must have length 2, got %v", ctx, st)
				}
				sh, sw = int(st[0]), int(st[1])
			}
			// Read pads & auto_pad
			autoPad := strings.ToUpper(nd.orig.AttrString("auto_pad", "NOTSET"))
			pads := []int64{0, 0, 0, 0}
			if ps, ok := nd.orig.AttrInts("pads"); ok {
				if len(ps) != 4 {
					return nil, fmt.Errorf("%s: pads must have length 4, got %v", ctx, ps)
				}
				pads = ps
			}

			// Extract optional bias (folded or raw input[2])
			var biasF64 []float64
			if nd.hasFoldB {
				biasF64 = nd.foldedB
			} else if len(nd.inputs) >= 3 && nd.inputs[2] != "" {
				bc, ok := consts[nd.inputs[2]]
				if !ok {
					return nil, fmt.Errorf("%s: bias input %q must be a constant initializer", ctx, nd.inputs[2])
				}
				if len(bc.dims) != 1 || int(bc.dims[0]) != cout {
					return nil, fmt.Errorf("%s: bias %q must have shape [%d], got %v", ctx, nd.inputs[2], cout, bc.dims)
				}
				biasF64, err = bc.float64s()
				if err != nil {
					return nil, fmt.Errorf("%s: bias %q: %w", ctx, nd.inputs[2], err)
				}
			}

			wFP16, err := float64SliceToFP16(wF64, baseID+"_weights")
			if err != nil {
				return nil, fmt.Errorf("%s: %w", ctx, err)
			}
			wFile := addWeightFile(baseID, "weights", wFP16)

			var bFile string
			if biasF64 != nil {
				bFP16, err := float64SliceToFP16(biasF64, baseID+"_bias")
				if err != nil {
					return nil, fmt.Errorf("%s: %w", ctx, err)
				}
				bFile = addWeightFile(baseID, "bias", bFP16)
			}

			if kh == 1 && kw == 1 {
				if sh != 1 || sw != 1 {
					return nil, fmt.Errorf("%s: conv1x1 requires strides [1,1], got [%d,%d]", ctx, sh, sw)
				}
				if group != 1 {
					return nil, fmt.Errorf("%s: conv1x1 requires group=1, got %d", ctx, group)
				}
				switch autoPad {
				case "", "NOTSET":
					if pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0 {
						return nil, fmt.Errorf("%s: conv1x1 requires pads [0,0,0,0], got %v", ctx, pads)
					}
				case "VALID", "SAME_UPPER", "SAME_LOWER":
				default:
					return nil, fmt.Errorf("%s: conv1x1 unsupported auto_pad %q", ctx, autoPad)
				}
				emitNode(VaneGraphNode{
					ID:      baseID,
					Op:      "conv1x1",
					Inputs:  []string{inID},
					Cin:     cin,
					Cout:    cout,
					Weights: wFile,
					Bias:    bFile,
					Relu:    nd.fusedRelu,
					Act:     nd.fusedAct,
				}, nd.output, [4]int{1, cout, h, w})

			} else if kh == 3 && kw == 3 {
				if sh != sw || (sh != 1 && sh != 2) {
					return nil, fmt.Errorf("%s: conv3x3 requires symmetric stride 1 or 2, got [%d,%d]", ctx, sh, sw)
				}
				var mode string
				if group == 1 {
					mode = "dense"
				} else if group == cin && cout == cin {
					mode = "depthwise"
				} else {
					return nil, fmt.Errorf("%s: grouped conv3x3 (group=%d, cin=%d, cout=%d) is not supported (only dense group=1 or depthwise group=cin=cout)",
						ctx, group, cin, cout)
				}

				padMode, ho, wo, err := resolveConv3x3Padding(h, w, sh, autoPad, pads, ctx)
				if err != nil {
					return nil, err
				}
				emitNode(VaneGraphNode{
					ID:      baseID,
					Op:      "conv3x3",
					Mode:    mode,
					Inputs:  []string{inID},
					Cin:     cin,
					Cout:    cout,
					Stride:  sh,
					Pad:     padMode,
					Weights: wFile,
					Bias:    bFile,
					Relu:    nd.fusedRelu,
					Act:     nd.fusedAct,
				}, nd.output, [4]int{1, cout, ho, wo})
			} else {
				return nil, fmt.Errorf("%s: unsupported Conv kernel size %dx%d (supported: 1x1, 3x3)", ctx, kh, kw)
			}

		case "BatchNormalization":
			// Standalone (unfused) BatchNormalization -> diagonal conv1x1 in float64 -> fp16.
			if len(nd.inputs) < 5 {
				return nil, fmt.Errorf("%s: expected 5 inputs (X, scale, B, mean, var)", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			c, h, w := inSh[1], inSh[2], inSh[3]
			factor, biasF64, err := computeBNAffine(nd, consts, c, ctx)
			if err != nil {
				return nil, err
			}
			diagW := make([]float64, c*c)
			for ch := 0; ch < c; ch++ {
				diagW[ch*c+ch] = factor[ch]
			}
			wFP16, err := float64SliceToFP16(diagW, baseID+"_weights")
			if err != nil {
				return nil, fmt.Errorf("%s: %w", ctx, err)
			}
			bFP16, err := float64SliceToFP16(biasF64, baseID+"_bias")
			if err != nil {
				return nil, fmt.Errorf("%s: %w", ctx, err)
			}
			wFile := addWeightFile(baseID, "weights", wFP16)
			bFile := addWeightFile(baseID, "bias", bFP16)
			emitNode(VaneGraphNode{
				ID:      baseID,
				Op:      "conv1x1",
				Inputs:  []string{inID},
				Cin:     c,
				Cout:    c,
				Weights: wFile,
				Bias:    bFile,
				Relu:    nd.fusedRelu,
				Act:     nd.fusedAct,
			}, nd.output, [4]int{1, c, h, w})

		case "Gemm", "MatMul":
			// Gemm operates on 2D [1, Cin] (normalized to [1, Cin, 1, 1]); MatMul projects
			// along the channel axis Cin -> Cout on [1, Cin, H, W]. Both lower to conv1x1
			// with [Cout, Cin] row-major fp16 weights for compatibility with vane compile-graph.
			if len(nd.inputs) < 2 {
				return nil, fmt.Errorf("%s: expected at least 2 inputs", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			cin, h, w := inSh[1], inSh[2], inSh[3]
			if nd.opType == "Gemm" && (h != 1 || w != 1) {
				return nil, fmt.Errorf("%s: Gemm requires 2D input [1, Cin] (normalized [1, Cin, 1, 1]), got [1, %d, %d, %d]", ctx, cin, h, w)
			}
			bc, ok := consts[nd.inputs[1]]
			if !ok {
				return nil, fmt.Errorf("%s: weight input %q must be a constant initializer", ctx, nd.inputs[1])
			}
			if len(bc.dims) != 2 {
				return nil, fmt.Errorf("%s: weight %q must be 2D, got shape %v", ctx, nd.inputs[1], bc.dims)
			}
			bRaw, err := bc.float64s()
			if err != nil {
				return nil, fmt.Errorf("%s: weight %q: %w", ctx, nd.inputs[1], err)
			}

			transA := int64(0)
			transB := int64(0)
			alpha := 1.0
			beta := 1.0
			if nd.opType == "Gemm" {
				transA = nd.orig.AttrInt("transA", 0)
				transB = nd.orig.AttrInt("transB", 0)
				alpha = nd.orig.AttrFloat("alpha", 1.0)
				beta = nd.orig.AttrFloat("beta", 1.0)
			}
			if transA != 0 {
				return nil, fmt.Errorf("%s: transA=%d is not supported (expected 0)", ctx, transA)
			}

			var cout int
			var wF64 []float64
			if transB != 0 {
				// B shape [Cout, Cin]
				cout = int(bc.dims[0])
				if int(bc.dims[1]) != cin {
					return nil, fmt.Errorf("%s: weight shape [%d,%d] (transB=1) incompatible with Cin=%d", ctx, bc.dims[0], bc.dims[1], cin)
				}
				wF64 = make([]float64, cout*cin)
				for i, v := range bRaw {
					wF64[i] = alpha * v
				}
			} else {
				// B shape [Cin, Cout] -> transpose to [Cout, Cin]
				if int(bc.dims[0]) != cin {
					return nil, fmt.Errorf("%s: weight shape [%d,%d] (transB=0) incompatible with Cin=%d", ctx, bc.dims[0], bc.dims[1], cin)
				}
				cout = int(bc.dims[1])
				wF64 = make([]float64, cout*cin)
				for c := 0; c < cin; c++ {
					for o := 0; o < cout; o++ {
						wF64[o*cin+c] = alpha * bRaw[c*cout+o]
					}
				}
			}

			var biasF64 []float64
			if nd.hasFoldB {
				if len(nd.foldedB) != cout {
					return nil, fmt.Errorf("%s: folded bias length %d != Cout=%d", ctx, len(nd.foldedB), cout)
				}
				biasF64 = nd.foldedB
			} else if nd.opType == "Gemm" && len(nd.inputs) >= 3 && nd.inputs[2] != "" {
				cc, ok := consts[nd.inputs[2]]
				if !ok {
					return nil, fmt.Errorf("%s: bias input %q must be a constant initializer", ctx, nd.inputs[2])
				}
				cRaw, err := cc.float64s()
				if err != nil {
					return nil, fmt.Errorf("%s: bias %q: %w", ctx, nd.inputs[2], err)
				}
				if len(cRaw) == 1 {
					biasF64 = make([]float64, cout)
					for o := range biasF64 {
						biasF64[o] = beta * cRaw[0]
					}
				} else if len(cRaw) == cout {
					biasF64 = make([]float64, cout)
					for o := range biasF64 {
						biasF64[o] = beta * cRaw[o]
					}
				} else {
					return nil, fmt.Errorf("%s: bias length %d incompatible with Cout=%d", ctx, len(cRaw), cout)
				}
			}

			wFP16, err := float64SliceToFP16(wF64, baseID+"_weights")
			if err != nil {
				return nil, fmt.Errorf("%s: %w", ctx, err)
			}
			wFile := addWeightFile(baseID, "weights", wFP16)
			var bFile string
			if biasF64 != nil {
				bFP16, err := float64SliceToFP16(biasF64, baseID+"_bias")
				if err != nil {
					return nil, fmt.Errorf("%s: %w", ctx, err)
				}
				bFile = addWeightFile(baseID, "bias", bFP16)
			}
			emitNode(VaneGraphNode{
				ID:      baseID,
				Op:      "conv1x1",
				Inputs:  []string{inID},
				Cin:     cin,
				Cout:    cout,
				Weights: wFile,
				Bias:    bFile,
				Relu:    nd.fusedRelu,
				Act:     nd.fusedAct,
			}, nd.output, [4]int{1, cout, h, w})

		case "Add", "Sub", "Mul", "Div", "Max", "Min":
			if len(nd.inputs) != 2 {
				return nil, fmt.Errorf("%s: expected 2 inputs, got %d", ctx, len(nd.inputs))
			}
			lhsName, rhsName := nd.inputs[0], nd.inputs[1]
			lhsC, lhsIsConst := consts[lhsName]
			rhsC, rhsIsConst := consts[rhsName]

			if lhsIsConst && rhsIsConst {
				return nil, fmt.Errorf("%s: both inputs (%q, %q) are constants; constant folding for binary ops is not supported", ctx, lhsName, rhsName)
			}

			// Scalar constant RHS or LHS on Add/Sub/Mul/Div
			if (rhsIsConst || lhsIsConst) && (nd.opType == "Add" || nd.opType == "Sub" || nd.opType == "Mul" || nd.opType == "Div") {
				var cs *constSource
				var tensorName string
				constOnLeft := lhsIsConst
				if lhsIsConst {
					cs = lhsC
					tensorName = rhsName
				} else {
					cs = rhsC
					tensorName = lhsName
				}
				isSc, err := cs.isScalar()
				if err != nil {
					return nil, fmt.Errorf("%s: %w", ctx, err)
				}
				if !isSc {
					return nil, fmt.Errorf("%s: non-scalar constant operand %q (shape %v) is not supported for %s", ctx, cs.name, cs.dims, nd.opType)
				}
				vals, err := cs.float64s()
				if err != nil {
					return nil, fmt.Errorf("%s: %w", ctx, err)
				}
				sc := vals[0]
				if err := validateFiniteFP16Scalar(sc, ctx); err != nil {
					return nil, err
				}
				inSh, inID, err := lookupShape(tensorName, ctx)
				if err != nil {
					return nil, err
				}

				switch nd.opType {
				case "Add":
					emitNode(VaneGraphNode{ID: baseID, Op: "pe", PeOp: "sadd", Inputs: []string{inID}, Scalar: &sc}, nd.output, inSh)
				case "Mul":
					emitNode(VaneGraphNode{ID: baseID, Op: "pe", PeOp: "smul", Inputs: []string{inID}, Scalar: &sc}, nd.output, inSh)
				case "Sub":
					peOp := "ssub"
					if constOnLeft {
						peOp = "srsub"
					}
					emitNode(VaneGraphNode{ID: baseID, Op: "pe", PeOp: peOp, Inputs: []string{inID}, Scalar: &sc}, nd.output, inSh)
				case "Div":
					if !constOnLeft {
						if fp16.ToFloat64(fp16.FromFloat64(sc)) == 0 {
							return nil, fmt.Errorf("%s: scalar division by zero (%g)", ctx, sc)
						}
						if !fp16.IsFinite(fp16.FromFloat64(1.0 / fp16.ToFloat64(fp16.FromFloat64(sc)))) {
							return nil, fmt.Errorf("%s: scalar division reciprocal overflows fp16 (%g)", ctx, sc)
						}
						emitNode(VaneGraphNode{ID: baseID, Op: "pe", PeOp: "sdiv", Inputs: []string{inID}, Scalar: &sc}, nd.output, inSh)
					} else {
						// c / x = smul(inverse(x), c)
						invID := sanitizeNodeID(baseID+"_inv", usedIDs)
						emitNode(VaneGraphNode{ID: invID, Op: "lut", Inputs: []string{inID}, Act: &ref.GraphActSpec{Name: "inverse"}}, "", inSh)
						emitNode(VaneGraphNode{ID: baseID, Op: "pe", PeOp: "smul", Inputs: []string{invID}, Scalar: &sc}, nd.output, inSh)
					}
				}
				continue
			}

			// Binary runtime tensor op
			shA, idA, err := lookupShape(lhsName, ctx)
			if err != nil {
				return nil, err
			}
			shB, idB, err := lookupShape(rhsName, ctx)
			if err != nil {
				return nil, err
			}
			outSh, err := broadcastShape4(shA, shB, ctx)
			if err != nil {
				return nil, err
			}
			if nd.opType == "Div" {
				// a / b = mul(a, inverse(b))
				invID := sanitizeNodeID(baseID+"_inv", usedIDs)
				emitNode(VaneGraphNode{ID: invID, Op: "lut", Inputs: []string{idB}, Act: &ref.GraphActSpec{Name: "inverse"}}, "", shB)
				emitNode(VaneGraphNode{ID: baseID, Op: "pe", PeOp: "mul", Inputs: []string{idA, invID}}, nd.output, outSh)
			} else {
				emitNode(VaneGraphNode{
					ID:     baseID,
					Op:     "pe",
					PeOp:   strings.ToLower(nd.opType),
					Inputs: []string{idA, idB},
				}, nd.output, outSh)
			}

		case "Abs":
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: expected 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			emitNode(VaneGraphNode{ID: baseID, Op: "pe", PeOp: "abs", Inputs: []string{inID}}, nd.output, inSh)

		case "Neg":
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: expected 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			negOne := -1.0
			emitNode(VaneGraphNode{ID: baseID, Op: "pe", PeOp: "smul", Inputs: []string{inID}, Scalar: &negOne}, nd.output, inSh)

		case "Relu", "Sigmoid", "Tanh", "Exp", "Gelu", "Erf", "Sqrt", "Sign", "Silu", "Swish":
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: expected 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			actName := strings.ToLower(nd.opType)
			if actName == "swish" {
				actName = "silu"
			}
			emitNode(VaneGraphNode{
				ID:     baseID,
				Op:     "lut",
				Inputs: []string{inID},
				Act:    &ref.GraphActSpec{Name: actName},
			}, nd.output, inSh)

		case "Reciprocal":
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: expected 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			emitNode(VaneGraphNode{
				ID:     baseID,
				Op:     "lut",
				Inputs: []string{inID},
				Act:    &ref.GraphActSpec{Name: "inverse"},
			}, nd.output, inSh)

		case "Elu":
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: expected 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			alpha := nd.orig.AttrFloat("alpha", 1.0)
			if err := validateFiniteFP16Scalar(alpha, ctx); err != nil {
				return nil, err
			}
			act := &ref.GraphActSpec{Name: "elu", Alpha: &alpha}
			emitNode(VaneGraphNode{ID: baseID, Op: "lut", Inputs: []string{inID}, Act: act}, nd.output, inSh)

		case "LeakyRelu":
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: expected 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			alpha := nd.orig.AttrFloat("alpha", 0.01)
			if err := validateFiniteFP16Scalar(alpha, ctx); err != nil {
				return nil, err
			}
			act := &ref.GraphActSpec{Name: "leaky_relu", Alpha: &alpha}
			emitNode(VaneGraphNode{ID: baseID, Op: "lut", Inputs: []string{inID}, Act: act}, nd.output, inSh)

		case "HardSigmoid":
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: expected 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			alpha := nd.orig.AttrFloat("alpha", 0.2)
			beta := nd.orig.AttrFloat("beta", 0.5)
			if math.IsNaN(alpha) || math.IsInf(alpha, 0) || alpha <= 0 {
				return nil, fmt.Errorf("%s: HardSigmoid alpha must be finite and > 0 (got %g)", ctx, alpha)
			}
			if err := validateFiniteFP16Scalar(alpha, ctx); err != nil {
				return nil, err
			}
			if err := validateFiniteFP16Scalar(beta, ctx); err != nil {
				return nil, err
			}
			if err := validateFiniteFP16Scalar(alpha/0.2, ctx); err != nil {
				return nil, err
			}
			if fp16.FromFloat64(alpha/0.2) == fp16.PosZero {
				return nil, fmt.Errorf("%s: HardSigmoid alpha (%g) underflows pre-LUT scale to 0", ctx, alpha)
			}
			act := &ref.GraphActSpec{Name: "hard_sigmoid", Alpha: &alpha, Beta: &beta}
			emitNode(VaneGraphNode{ID: baseID, Op: "lut", Inputs: []string{inID}, Act: act}, nd.output, inSh)

		case "Clip":
			if len(nd.inputs) < 1 {
				return nil, fmt.Errorf("%s: expected at least 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			lo, hasLo, err := readOptionalScalarParam(nd, consts, 1, "min", ctx)
			if err != nil {
				return nil, err
			}
			hi, hasHi, err := readOptionalScalarParam(nd, consts, 2, "max", ctx)
			if err != nil {
				return nil, err
			}
			if !hasLo || !hasHi {
				return nil, fmt.Errorf("%s: Clip requires both constant min and max bounds", ctx)
			}
			if err := validateFiniteFP16Scalar(lo, ctx); err != nil {
				return nil, err
			}
			if err := validateFiniteFP16Scalar(hi, ctx); err != nil {
				return nil, err
			}
			if lo > hi {
				return nil, fmt.Errorf("%s: Clip requires min <= max (got min=%g, max=%g)", ctx, lo, hi)
			}
			var act *ref.GraphActSpec
			if lo == 0 && hi == 6 {
				act = &ref.GraphActSpec{Name: "relu6"}
			} else {
				act = &ref.GraphActSpec{Name: "clip", Lo: &lo, Hi: &hi}
			}
			emitNode(VaneGraphNode{ID: baseID, Op: "lut", Inputs: []string{inID}, Act: act}, nd.output, inSh)

		case "Softplus":
			// Softplus(x) = ln(1 + exp(x)) = ln(2) * log2(1 + exp(x))
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: expected 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			expID := sanitizeNodeID(baseID+"_exp", usedIDs)
			addID := sanitizeNodeID(baseID+"_add1", usedIDs)
			log2ID := sanitizeNodeID(baseID+"_log2", usedIDs)
			one := 1.0
			ln2 := math.Ln2
			emitNode(VaneGraphNode{ID: expID, Op: "lut", Inputs: []string{inID}, Act: &ref.GraphActSpec{Name: "exp"}}, "", inSh)
			emitNode(VaneGraphNode{ID: addID, Op: "pe", PeOp: "sadd", Inputs: []string{expID}, Scalar: &one}, "", inSh)
			emitNode(VaneGraphNode{ID: log2ID, Op: "lut", Inputs: []string{addID}, Act: &ref.GraphActSpec{Name: "log2"}}, "", inSh)
			emitNode(VaneGraphNode{ID: baseID, Op: "pe", PeOp: "smul", Inputs: []string{log2ID}, Scalar: &ln2}, nd.output, inSh)

		case "MaxPool", "AveragePool":
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: expected 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			kShape, ok := nd.orig.AttrInts("kernel_shape")
			if !ok || len(kShape) != 2 || kShape[0] <= 0 || kShape[1] <= 0 {
				return nil, fmt.Errorf("%s: kernel_shape attribute must be 2 positive ints, got %v", ctx, kShape)
			}
			if nd.orig.AttrInt("ceil_mode", 0) != 0 {
				return nil, fmt.Errorf("%s: ceil_mode != 0 is not supported", ctx)
			}
			if dils, ok := nd.orig.AttrInts("dilations"); ok {
				if len(dils) != 2 || dils[0] != 1 || dils[1] != 1 {
					return nil, fmt.Errorf("%s: unsupported dilations %v (only [1,1] is supported)", ctx, dils)
				}
			}
			sh, sw := 1, 1
			if st, ok := nd.orig.AttrInts("strides"); ok {
				if len(st) != 2 {
					return nil, fmt.Errorf("%s: strides must have length 2, got %v", ctx, st)
				}
				sh, sw = int(st[0]), int(st[1])
			}
			if sh <= 0 || sw <= 0 {
				return nil, fmt.Errorf("%s: pool strides must be > 0, got [%d,%d]", ctx, sh, sw)
			}
			autoPad := strings.ToUpper(nd.orig.AttrString("auto_pad", "NOTSET"))
			pads := []int64{0, 0, 0, 0}
			if ps, ok := nd.orig.AttrInts("pads"); ok {
				if len(ps) != 4 {
					return nil, fmt.Errorf("%s: pads must have length 4, got %v", ctx, ps)
				}
				pads = ps
			}
			// ONNX pads order: [pt, pl, pb, pr]
			pt, pl, pb, pr := int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])
			var padMode string
			switch autoPad {
			case "VALID":
				padMode = "valid"
				pt, pl, pb, pr = 0, 0, 0, 0
			case "SAME_UPPER":
				padMode = "same"
				pt, pl, pb, pr = 0, 0, 0, 0
			case "SAME_LOWER":
				// ONNX SAME_LOWER assigns extra padding pixel to top/left, whereas Vane's
				// pad="same" uses SAME_UPPER. Lower SAME_LOWER with explicit custom pads.
				ohTmp := (inSh[2] + sh - 1) / sh
				padH := (ohTmp-1)*sh + int(kShape[0]) - inSh[2]
				if padH < 0 {
					padH = 0
				}
				owTmp := (inSh[3] + sw - 1) / sw
				padW := (owTmp-1)*sw + int(kShape[1]) - inSh[3]
				if padW < 0 {
					padW = 0
				}
				pt = (padH + 1) / 2
				pb = padH - pt
				pl = (padW + 1) / 2
				pr = padW - pl
				if pt == 0 && pb == 0 && pl == 0 && pr == 0 {
					padMode = "valid"
				} else {
					padMode = "custom"
				}
			case "", "NOTSET":
				if pt == 0 && pl == 0 && pb == 0 && pr == 0 {
					padMode = "valid"
				} else {
					padMode = "custom"
				}
			default:
				return nil, fmt.Errorf("%s: unsupported auto_pad %q", ctx, autoPad)
			}

			vaneOp := "max_pool"
			var exclPtr *bool
			if nd.opType == "AveragePool" {
				vaneOp = "avg_pool"
				excl := nd.orig.AttrInt("count_include_pad", 0) == 0
				exclPtr = &excl
			}
			pspec := ref.PoolSpec{
				Op: vaneOp, KH: int(kShape[0]), KW: int(kShape[1]),
				SH: sh, SW: sw, Pad: padMode,
				PT: pt, PB: pb, PL: pl, PR: pr,
			}
			oh, ow, _, _, _, _, err := ref.PoolGeometry(inSh[2], inSh[3], pspec)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", ctx, err)
			}
			vn := VaneGraphNode{
				ID:         baseID,
				Op:         vaneOp,
				Inputs:     []string{inID},
				KH:         int(kShape[0]),
				KW:         int(kShape[1]),
				SH:         sh,
				SW:         sw,
				Pad:        padMode,
				ExcludePad: exclPtr,
			}
			if padMode == "custom" {
				vn.PT, vn.PB, vn.PL, vn.PR = pt, pb, pl, pr
			}
			emitNode(vn, nd.output, [4]int{1, inSh[1], oh, ow})

		case "GlobalAveragePool":
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: expected 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			emitNode(VaneGraphNode{
				ID:       baseID,
				Op:       "reduce",
				Inputs:   []string{inID},
				ReduceOp: "mean",
				Axes:     "hw",
			}, nd.output, [4]int{1, inSh[1], 1, 1})

		case "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin":
			if len(nd.inputs) < 1 {
				return nil, fmt.Errorf("%s: expected at least 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			if nd.orig.AttrInt("keepdims", 1) == 0 {
				return nil, fmt.Errorf("%s: keepdims=0 is not supported (Vane reduce preserves 4D rank with keepdims=1)", ctx)
			}
			var rawAxes []int64
			if axAttr, ok := nd.orig.AttrInts("axes"); ok {
				rawAxes = axAttr
			} else if len(nd.inputs) >= 2 && nd.inputs[1] != "" {
				ac, ok := consts[nd.inputs[1]]
				if !ok {
					return nil, fmt.Errorf("%s: axes input %q must be a constant int64 tensor", ctx, nd.inputs[1])
				}
				rawAxes, err = ac.int64s()
				if err != nil {
					return nil, fmt.Errorf("%s: axes %q: %w", ctx, nd.inputs[1], err)
				}
			}
			if len(rawAxes) == 0 {
				if nd.orig.AttrInt("noop_with_empty_axes", 0) != 0 {
					shapes[nd.output] = inSh
					vaneIDOf[nd.output] = inID
					elidedCount++
					continue
				}
				rawAxes = []int64{1, 2, 3}
			}

			axesStr, outSh, err := resolveReduceAxes4D(rawAxes, inSh, ctx)
			if err != nil {
				return nil, err
			}
			redOp := strings.ToLower(strings.TrimPrefix(nd.opType, "Reduce"))
			if err := validateReduceBackendConstraints(redOp, axesStr, inSh, ctx); err != nil {
				return nil, err
			}
			emitNode(VaneGraphNode{
				ID:       baseID,
				Op:       "reduce",
				Inputs:   []string{inID},
				ReduceOp: redOp,
				Axes:     axesStr,
			}, nd.output, outSh)

		case "Softmax":
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: expected 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			var axStr string
			var redSh [4]int
			if opset > 0 && opset < 13 {
				// ONNX opset < 13 flattens trailing axes [axis..3] (default axis=1).
				axis := nd.orig.AttrInt("axis", 1)
				if axis < 0 {
					axis += 4
				}
				switch axis {
				case 0, 1:
					axStr = "chw"
					redSh = [4]int{1, 1, 1, 1}
				case 2:
					axStr = "hw"
					redSh = [4]int{1, inSh[1], 1, 1}
				case 3:
					axStr = "w"
					redSh = [4]int{1, inSh[1], inSh[2], 1}
				default:
					return nil, fmt.Errorf("%s: Softmax (opset %d) axis %d out of range [0,3]", ctx, opset, axis)
				}
			} else {
				axis := nd.orig.AttrInt("axis", -1)
				if axis < 0 {
					axis += 4
				}
				switch axis {
				case 1:
					axStr = "c"
					redSh = [4]int{1, 1, inSh[2], inSh[3]}
				case 2:
					axStr = "h"
					redSh = [4]int{1, inSh[1], 1, inSh[3]}
				case 3:
					axStr = "w"
					redSh = [4]int{1, inSh[1], inSh[2], 1}
				default:
					return nil, fmt.Errorf("%s: Softmax axis %d not in {1,2,3}", ctx, axis)
				}
			}
			if err := validateReduceBackendConstraints("max", axStr, inSh, ctx); err != nil {
				return nil, err
			}
			maxID := sanitizeNodeID(baseID+"_max", usedIDs)
			subID := sanitizeNodeID(baseID+"_sub", usedIDs)
			expID := sanitizeNodeID(baseID+"_exp", usedIDs)
			emitNode(VaneGraphNode{ID: maxID, Op: "reduce", Inputs: []string{inID}, ReduceOp: "max", Axes: axStr}, "", redSh)
			emitNode(VaneGraphNode{ID: subID, Op: "pe", PeOp: "sub", Inputs: []string{inID, maxID}}, "", inSh)
			emitNode(VaneGraphNode{ID: expID, Op: "lut", Inputs: []string{subID}, Act: &ref.GraphActSpec{Name: "exp"}}, "", inSh)

			var invSumID string
			if axStr == "c" && inSh[3] > 1 {
				// Vane's NE channel reducer supports max/min/sum/mean (not inv_sum) when W > 1.
				sumID := sanitizeNodeID(baseID+"_sum", usedIDs)
				invSumID = sanitizeNodeID(baseID+"_invsum", usedIDs)
				emitNode(VaneGraphNode{ID: sumID, Op: "reduce", Inputs: []string{expID}, ReduceOp: "sum", Axes: axStr}, "", redSh)
				emitNode(VaneGraphNode{ID: invSumID, Op: "lut", Inputs: []string{sumID}, Act: &ref.GraphActSpec{Name: "inverse"}}, "", redSh)
			} else {
				invSumID = sanitizeNodeID(baseID+"_invsum", usedIDs)
				emitNode(VaneGraphNode{ID: invSumID, Op: "reduce", Inputs: []string{expID}, ReduceOp: "inv_sum", Axes: axStr}, "", redSh)
			}
			emitNode(VaneGraphNode{ID: baseID, Op: "pe", PeOp: "mul", Inputs: []string{expID, invSumID}}, nd.output, inSh)

		case "Transpose":
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: expected 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			perm64, ok := nd.orig.AttrInts("perm")
			if !ok {
				perm64 = []int64{3, 2, 1, 0}
			}
			if len(perm64) != 4 {
				return nil, fmt.Errorf("%s: perm must have length 4, got %v", ctx, perm64)
			}
			var seen [4]bool
			perm := make([]int, 4)
			var outSh [4]int
			for i, p := range perm64 {
				if p < 0 || p >= 4 || seen[p] {
					return nil, fmt.Errorf("%s: invalid permutation %v", ctx, perm64)
				}
				seen[p] = true
				perm[i] = int(p)
				outSh[i] = inSh[p]
			}
			if outSh[0] != 1 {
				return nil, fmt.Errorf("%s: Transpose output batch dimension must be 1, got %d", ctx, outSh[0])
			}
			if perm[0] == 0 && perm[1] == 1 && perm[2] == 2 && perm[3] == 3 {
				shapes[nd.output] = inSh
				vaneIDOf[nd.output] = inID
				elidedCount++
				continue
			}
			if !opts.AllowViewOps {
				return nil, fmt.Errorf("%s: non-no-op view op Transpose (perm=%v) requires --allow-view-ops", ctx, perm)
			}
			emitNode(VaneGraphNode{
				ID:     baseID,
				Op:     "transpose",
				Inputs: []string{inID},
				Perm:   perm,
			}, nd.output, outSh)

		case "Concat":
			if len(nd.inputs) < 2 {
				return nil, fmt.Errorf("%s: Concat requires at least 2 inputs, got %d", ctx, len(nd.inputs))
			}
			axAttr, ok := nd.orig.Attr("axis")
			if !ok {
				return nil, fmt.Errorf("%s: Concat missing required axis attribute", ctx)
			}
			axis := int(axAttr.I)
			if axis < 0 {
				axis += 4
			}
			if axis < 1 || axis > 3 {
				return nil, fmt.Errorf("%s: Concat axis %d must be in {1,2,3}", ctx, axis)
			}
			inIDs := make([]string, len(nd.inputs))
			var outSh [4]int
			for i, inpName := range nd.inputs {
				sh, id, err := lookupShape(inpName, ctx)
				if err != nil {
					return nil, err
				}
				inIDs[i] = id
				if i == 0 {
					outSh = sh
				} else {
					for d := 0; d < 4; d++ {
						if d == axis {
							outSh[d] += sh[d]
						} else if sh[d] != outSh[d] {
							return nil, fmt.Errorf("%s: Concat input %d shape %v mismatches axis %d (%d vs %d)",
								ctx, i, sh, d, sh[d], outSh[d])
						}
					}
				}
			}
			if !opts.AllowViewOps {
				return nil, fmt.Errorf("%s: non-no-op view op Concat (axis=%d) requires --allow-view-ops", ctx, axis)
			}
			axVal := axis
			emitNode(VaneGraphNode{
				ID:     baseID,
				Op:     "concat",
				Inputs: inIDs,
				Axis:   &axVal,
			}, nd.output, outSh)

		case "Reshape":
			if len(nd.inputs) != 2 {
				return nil, fmt.Errorf("%s: Reshape requires 2 inputs (data, shape)", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			sc, ok := consts[nd.inputs[1]]
			if !ok {
				return nil, fmt.Errorf("%s: Reshape target shape %q must be a constant int64 tensor", ctx, nd.inputs[1])
			}
			rawTarget, err := sc.int64s()
			if err != nil {
				return nil, fmt.Errorf("%s: target shape %q: %w", ctx, nd.inputs[1], err)
			}
			allowZero := nd.orig.AttrInt("allowzero", 0) != 0
			outSh, err := resolveReshapeTarget(inSh, rawTarget, allowZero, ctx)
			if err != nil {
				return nil, err
			}
			if outSh == inSh {
				shapes[nd.output] = inSh
				vaneIDOf[nd.output] = inID
				elidedCount++
				continue
			}
			if !opts.AllowViewOps {
				return nil, fmt.Errorf("%s: non-no-op view op Reshape (%v -> %v) requires --allow-view-ops", ctx, inSh, outSh)
			}
			emitNode(VaneGraphNode{
				ID:       baseID,
				Op:       "reshape",
				Inputs:   []string{inID},
				OutShape: []int{outSh[0], outSh[1], outSh[2], outSh[3]},
			}, nd.output, outSh)

		case "Flatten":
			if len(nd.inputs) != 1 {
				return nil, fmt.Errorf("%s: Flatten requires 1 input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			axis := nd.orig.AttrInt("axis", 1)
			if axis < 0 {
				axis += 4
			}
			if axis != 1 {
				return nil, fmt.Errorf("%s: Flatten axis %d not supported (expected axis=1)", ctx, axis)
			}
			outSh := [4]int{1, inSh[1] * inSh[2] * inSh[3], 1, 1}
			if outSh == inSh {
				shapes[nd.output] = inSh
				vaneIDOf[nd.output] = inID
				elidedCount++
				continue
			}
			if !opts.AllowViewOps {
				return nil, fmt.Errorf("%s: non-no-op view op Flatten (%v -> %v) requires --allow-view-ops", ctx, inSh, outSh)
			}
			emitNode(VaneGraphNode{
				ID:       baseID,
				Op:       "reshape",
				Inputs:   []string{inID},
				OutShape: []int{outSh[0], outSh[1], outSh[2], outSh[3]},
			}, nd.output, outSh)

		case "Slice":
			if len(nd.inputs) < 1 {
				return nil, fmt.Errorf("%s: Slice missing data input", ctx)
			}
			inSh, inID, err := lookupShape(nd.inputs[0], ctx)
			if err != nil {
				return nil, err
			}
			beg, end, step, outSh, err := resolveSliceParams(nd, consts, inSh, ctx)
			if err != nil {
				return nil, err
			}
			if !opts.AllowViewOps {
				return nil, fmt.Errorf("%s: non-no-op view op Slice (%v -> %v) requires --allow-view-ops", ctx, inSh, outSh)
			}
			emitNode(VaneGraphNode{
				ID:     baseID,
				Op:     "slice",
				Inputs: []string{inID},
				Begin:  beg,
				End:    end,
				Step:   step,
			}, nd.output, outSh)

		default:
			return nil, fmt.Errorf("%s: unsupported ONNX operator %q", ctx, nd.opType)
		}
	}

	// Ensure graph has at least one node and ends at graphOutput.
	outSh, outID, err := lookupShape(graphOutput, "graph output")
	if err != nil {
		return nil, err
	}
	if len(vaneNodes) == 0 || vaneNodes[len(vaneNodes)-1].ID != outID {
		copyID := sanitizeNodeID(outID+"_out", usedIDs)
		vaneNodes = append(vaneNodes, VaneGraphNode{
			ID:     copyID,
			Op:     "lut",
			Inputs: []string{outID},
			Act:    &ref.GraphActSpec{Name: "copy"},
		})
		outID = copyID
	}

	var totalWeightBytes int64
	for _, wb := range weights {
		totalWeightBytes += int64(len(wb))
	}

	spec := VaneGraphSpec{
		Target:    target,
		Placement: placement,
		Inputs:    graphInputs,
		Nodes:     vaneNodes,
		Output:    outID,
	}
	return &LowerResult{
		Spec:    spec,
		Weights: weights,
		Summary: LowerSummary{
			Target:          target,
			Placement:       placement,
			Inputs:          graphInputs,
			Output:          outID,
			OutputShape:     outSh,
			ONNXNodeCount:   len(g.Nodes),
			NodeCount:       len(vaneNodes),
			FusedBNCount:    fusedBNCount,
			FusedActCount:   fusedActCount,
			ElidedCount:     elidedCount,
			WeightFiles:     len(weights),
			WeightBytes:     totalWeightBytes,
			SafeTensorsUsed: sfUsed,
		},
	}, nil
}

func pruneUnreachable(nodes []*irNode, graphOutput string) []*irNode {
	producer := make(map[string]*irNode, len(nodes))
	for _, nd := range nodes {
		producer[nd.output] = nd
	}
	live := make(map[*irNode]bool, len(nodes))
	stack := []string{graphOutput}
	for len(stack) > 0 {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		nd, ok := producer[cur]
		if !ok || live[nd] {
			continue
		}
		live[nd] = true
		for _, in := range nd.inputs {
			if in != "" {
				stack = append(stack, in)
			}
		}
	}
	out := make([]*irNode, 0, len(live))
	for _, nd := range nodes {
		if live[nd] {
			out = append(out, nd)
		}
	}
	return out
}

func countConsumers(nodes []*irNode, graphOutput string) map[string]int {
	counts := make(map[string]int, len(nodes)+1)
	for _, nd := range nodes {
		for _, in := range nd.inputs {
			if in != "" {
				counts[in]++
			}
		}
	}
	counts[graphOutput]++
	return counts
}

// collapseSiluPatterns detects Mul(x, Sigmoid(x)) or Mul(Sigmoid(x), x) where Sigmoid(x)
// has a single consumer, replacing the pair with a synthetic Silu(x) node.
func collapseSiluPatterns(nodes []*irNode, graphOutput string) []*irNode {
	counts := countConsumers(nodes, graphOutput)
	producer := make(map[string]*irNode, len(nodes))
	for _, nd := range nodes {
		producer[nd.output] = nd
	}
	removed := make(map[*irNode]bool)
	for _, nd := range nodes {
		if nd.opType != "Mul" || len(nd.inputs) != 2 {
			continue
		}
		a, b := nd.inputs[0], nd.inputs[1]
		var sigNode *irNode
		var xName string
		if p, ok := producer[b]; ok && p.opType == "Sigmoid" && len(p.inputs) == 1 && p.inputs[0] == a && counts[b] == 1 {
			sigNode = p
			xName = a
		} else if p, ok := producer[a]; ok && p.opType == "Sigmoid" && len(p.inputs) == 1 && p.inputs[0] == b && counts[a] == 1 {
			sigNode = p
			xName = b
		}
		if sigNode != nil {
			removed[sigNode] = true
			nd.opType = "Silu"
			nd.inputs = []string{xName}
		}
	}
	if len(removed) == 0 {
		return nodes
	}
	out := make([]*irNode, 0, len(nodes)-len(removed))
	for _, nd := range nodes {
		if !removed[nd] {
			out = append(out, nd)
		}
	}
	return out
}

func computeBNAffine(bn *irNode, consts map[string]*constSource, cout int, ctx string) (factor, bias []float64, err error) {
	if len(bn.inputs) < 5 {
		return nil, nil, fmt.Errorf("%s: BatchNormalization requires 5 inputs", ctx)
	}
	if bn.orig.AttrInt("training_mode", 0) != 0 {
		return nil, nil, fmt.Errorf("%s: BatchNormalization training_mode != 0 is not supported (inference mode only)", ctx)
	}
	getVec := func(idx int, label string) ([]float64, error) {
		cs, ok := consts[bn.inputs[idx]]
		if !ok {
			return nil, fmt.Errorf("%s: BatchNormalization %s %q must be a constant initializer", ctx, label, bn.inputs[idx])
		}
		v, err := cs.float64s()
		if err != nil {
			return nil, fmt.Errorf("%s: BatchNormalization %s: %w", ctx, label, err)
		}
		if len(v) != cout {
			return nil, fmt.Errorf("%s: BatchNormalization %s length %d != channels %d", ctx, label, len(v), cout)
		}
		return v, nil
	}
	scale, err := getVec(1, "scale")
	if err != nil {
		return nil, nil, err
	}
	bVec, err := getVec(2, "B")
	if err != nil {
		return nil, nil, err
	}
	mean, err := getVec(3, "mean")
	if err != nil {
		return nil, nil, err
	}
	variance, err := getVec(4, "var")
	if err != nil {
		return nil, nil, err
	}
	eps := bn.orig.AttrFloat("epsilon", 1e-5)
	if math.IsNaN(eps) || math.IsInf(eps, 0) || eps < 0 {
		return nil, nil, fmt.Errorf("%s: BatchNormalization invalid epsilon %g", ctx, eps)
	}

	factor = make([]float64, cout)
	bias = make([]float64, cout)
	for o := 0; o < cout; o++ {
		if variance[o] < 0 {
			return nil, nil, fmt.Errorf("%s: BatchNormalization negative variance[%d]=%g", ctx, o, variance[o])
		}
		denom := variance[o] + eps
		if denom <= 0 || math.IsNaN(denom) || math.IsInf(denom, 0) {
			return nil, nil, fmt.Errorf("%s: BatchNormalization invalid variance[%d]+eps=%g", ctx, o, denom)
		}
		f := scale[o] / math.Sqrt(denom)
		factor[o] = f
		bias[o] = bVec[o] - mean[o]*f
	}
	return factor, bias, nil
}

// foldConvBatchNorm algebraically folds single-consumer Conv -> BatchNormalization
// in float64 precision before fp16 quantization.
func foldConvBatchNorm(nodes []*irNode, consts map[string]*constSource, graphOutput string) ([]*irNode, int, error) {
	counts := countConsumers(nodes, graphOutput)
	producer := make(map[string]*irNode, len(nodes))
	for _, nd := range nodes {
		producer[nd.output] = nd
	}

	removed := make(map[*irNode]bool)
	fused := 0

	for i, bn := range nodes {
		if bn.opType != "BatchNormalization" || len(bn.inputs) < 5 {
			continue
		}
		conv, ok := producer[bn.inputs[0]]
		if !ok || conv.opType != "Conv" || counts[conv.output] != 1 {
			continue
		}
		if len(conv.inputs) < 2 {
			continue
		}
		ctx := fmt.Sprintf("node %q (Conv+BatchNormalization)", nodeLabel(bn, i))
		wc, ok := consts[conv.inputs[1]]
		if !ok {
			continue
		}
		if len(wc.dims) != 4 {
			return nil, 0, fmt.Errorf("%s: Conv weight %q must be 4D, got %v", ctx, conv.inputs[1], wc.dims)
		}
		cout := int(wc.dims[0])
		var wF64 []float64
		var err error
		if conv.foldedW != nil {
			wF64 = append([]float64(nil), conv.foldedW...)
		} else {
			wF64, err = wc.float64s()
			if err != nil {
				return nil, 0, fmt.Errorf("%s: Conv weight: %w", ctx, err)
			}
		}
		if cout <= 0 || len(wF64) == 0 || len(wF64)%cout != 0 {
			return nil, 0, fmt.Errorf("%s: invalid Conv weight length %d for Cout=%d", ctx, len(wF64), cout)
		}

		var convBias []float64
		if conv.hasFoldB {
			convBias = append([]float64(nil), conv.foldedB...)
		} else if len(conv.inputs) >= 3 && conv.inputs[2] != "" {
			bc, ok := consts[conv.inputs[2]]
			if !ok {
				return nil, 0, fmt.Errorf("%s: Conv bias %q must be constant", ctx, conv.inputs[2])
			}
			convBias, err = bc.float64s()
			if err != nil {
				return nil, 0, fmt.Errorf("%s: Conv bias: %w", ctx, err)
			}
			if len(convBias) != cout {
				return nil, 0, fmt.Errorf("%s: Conv bias length %d != Cout=%d", ctx, len(convBias), cout)
			}
		} else {
			convBias = make([]float64, cout)
		}

		factor, bnBiasShift, err := computeBNAffine(bn, consts, cout, ctx)
		if err != nil {
			return nil, 0, err
		}
		perOut := len(wF64) / cout
		for o := 0; o < cout; o++ {
			f := factor[o]
			base := o * perOut
			for k := 0; k < perOut; k++ {
				wF64[base+k] *= f
			}
			// (convBias[o] - mean[o]) * f + bnB[o] = convBias[o]*f + bnBiasShift[o]
			convBias[o] = convBias[o]*f + bnBiasShift[o]
		}

		conv.foldedW = wF64
		conv.foldedWSh = wc.dims
		conv.foldedB = convBias
		conv.hasFoldB = true

		delete(producer, conv.output)
		conv.output = bn.output
		producer[conv.output] = conv
		removed[bn] = true
		fused++
	}

	if len(removed) == 0 {
		return nodes, 0, nil
	}
	out := make([]*irNode, 0, len(nodes)-len(removed))
	for _, nd := range nodes {
		if !removed[nd] {
			out = append(out, nd)
		}
	}
	return out, fused, nil
}

// foldMatMulAddBias folds single-consumer MatMul(A, const B) -> Add(_, const bias)
// into a biased linear node.
func foldMatMulAddBias(nodes []*irNode, consts map[string]*constSource, graphOutput string) ([]*irNode, error) {
	counts := countConsumers(nodes, graphOutput)
	producer := make(map[string]*irNode, len(nodes))
	for _, nd := range nodes {
		producer[nd.output] = nd
	}
	removed := make(map[*irNode]bool)

	for _, add := range nodes {
		if add.opType != "Add" || len(add.inputs) != 2 {
			continue
		}
		a, b := add.inputs[0], add.inputs[1]
		var mm *irNode
		var biasName string
		if p, ok := producer[a]; ok && p.opType == "MatMul" && !p.hasFoldB && counts[a] == 1 {
			if _, okC := consts[b]; okC {
				mm = p
				biasName = b
			}
		} else if p, ok := producer[b]; ok && p.opType == "MatMul" && !p.hasFoldB && counts[b] == 1 {
			if _, okC := consts[a]; okC {
				mm = p
				biasName = a
			}
		}
		if mm == nil || len(mm.inputs) < 2 {
			continue
		}
		wc, ok := consts[mm.inputs[1]]
		if !ok || len(wc.dims) != 2 {
			continue
		}
		cout := int(wc.dims[1])
		bc := consts[biasName]
		if !isChannelBiasShape(bc.dims, int64(cout)) {
			continue
		}
		bVals, err := bc.float64s()
		if err != nil {
			return nil, err
		}
		if len(bVals) != cout {
			continue
		}
		mm.foldedB = bVals
		mm.hasFoldB = true
		delete(producer, mm.output)
		mm.output = add.output
		producer[mm.output] = mm
		removed[add] = true
	}

	if len(removed) == 0 {
		return nodes, nil
	}
	out := make([]*irNode, 0, len(nodes)-len(removed))
	for _, nd := range nodes {
		if !removed[nd] {
			out = append(out, nd)
		}
	}
	return out, nil
}

// fuseConvActivations fuses single-consumer Relu, Gelu, Sigmoid, Tanh, or Silu/Swish
// directly onto a preceding Conv, Gemm, or MatMul node.
func fuseConvActivations(nodes []*irNode, graphOutput string) ([]*irNode, int) {
	counts := countConsumers(nodes, graphOutput)
	producer := make(map[string]*irNode, len(nodes))
	for _, nd := range nodes {
		producer[nd.output] = nd
	}
	removed := make(map[*irNode]bool)
	fused := 0

	for _, act := range nodes {
		if len(act.inputs) != 1 {
			continue
		}
		var isRelu bool
		var actSpec *ref.GraphActSpec
		switch act.opType {
		case "Relu":
			isRelu = true
		case "Gelu":
			actSpec = &ref.GraphActSpec{Name: "gelu"}
		case "Sigmoid":
			actSpec = &ref.GraphActSpec{Name: "sigmoid"}
		case "Tanh":
			actSpec = &ref.GraphActSpec{Name: "tanh"}
		case "Silu", "Swish":
			actSpec = &ref.GraphActSpec{Name: "silu"}
		default:
			continue
		}
		prod, ok := producer[act.inputs[0]]
		if !ok || counts[prod.output] != 1 {
			continue
		}
		if prod.opType != "Conv" && prod.opType != "Gemm" && prod.opType != "MatMul" {
			continue
		}
		if prod.fusedRelu || prod.fusedAct != nil {
			continue
		}
		if isRelu {
			prod.fusedRelu = true
		} else {
			prod.fusedAct = actSpec
		}
		delete(producer, prod.output)
		prod.output = act.output
		producer[prod.output] = prod
		removed[act] = true
		fused++
	}

	if len(removed) == 0 {
		return nodes, 0
	}
	out := make([]*irNode, 0, len(nodes)-len(removed))
	for _, nd := range nodes {
		if !removed[nd] {
			out = append(out, nd)
		}
	}
	return out, fused
}

func validateFiniteFP16Scalar(v float64, ctx string) error {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return fmt.Errorf("%s: scalar %v must be finite", ctx, v)
	}
	if !fp16.IsFinite(fp16.FromFloat64(v)) {
		return fmt.Errorf("%s: scalar %g overflows finite fp16 range", ctx, v)
	}
	return nil
}

func readOptionalScalarParam(nd *irNode, consts map[string]*constSource, inputIdx int, attrName, ctx string) (float64, bool, error) {
	if inputIdx < len(nd.inputs) && nd.inputs[inputIdx] != "" {
		cs, ok := consts[nd.inputs[inputIdx]]
		if !ok {
			return 0, false, fmt.Errorf("%s: input %q (%s) must be a constant scalar", ctx, nd.inputs[inputIdx], attrName)
		}
		vals, err := cs.float64s()
		if err != nil {
			return 0, false, fmt.Errorf("%s: %s: %w", ctx, attrName, err)
		}
		if len(vals) != 1 {
			return 0, false, fmt.Errorf("%s: %s must be scalar, got %d elements", ctx, attrName, len(vals))
		}
		return vals[0], true, nil
	}
	if a, ok := nd.orig.Attr(attrName); ok {
		return float64(a.F), true, nil
	}
	return 0, false, nil
}

func broadcastShape4(a, b [4]int, ctx string) ([4]int, error) {
	var out [4]int
	axisNames := [4]string{"N", "C", "H", "W"}
	for i := 0; i < 4; i++ {
		if a[i] == b[i] {
			out[i] = a[i]
		} else if a[i] == 1 {
			out[i] = b[i]
		} else if b[i] == 1 {
			out[i] = a[i]
		} else {
			return out, fmt.Errorf("%s: incompatible broadcast on %s: %d vs %d", ctx, axisNames[i], a[i], b[i])
		}
	}
	return out, nil
}

func resolveReduceAxes4D(rawAxes []int64, inSh [4]int, ctx string) (string, [4]int, error) {
	if len(rawAxes) == 0 {
		return "", [4]int{}, fmt.Errorf("%s: reduce requires at least one axis", ctx)
	}
	var rc, rh, rw bool
	for _, ax := range rawAxes {
		if ax < 0 {
			ax += 4
		}
		switch ax {
		case 1:
			rc = true
		case 2:
			rh = true
		case 3:
			rw = true
		default:
			return "", [4]int{}, fmt.Errorf("%s: reduce axis %d not supported (expected subset of {1,2,3})", ctx, ax)
		}
	}
	var b strings.Builder
	outSh := inSh
	if rc {
		b.WriteByte('c')
		outSh[1] = 1
	}
	if rh {
		b.WriteByte('h')
		outSh[2] = 1
	}
	if rw {
		b.WriteByte('w')
		outSh[3] = 1
	}
	axesStr := b.String()
	if _, _, _, err := ref.ParseReduceAxes(axesStr); err != nil {
		return "", [4]int{}, fmt.Errorf("%s: %w", ctx, err)
	}
	return axesStr, outSh, nil
}

func resolveReshapeTarget(inSh [4]int, raw []int64, allowZero bool, ctx string) ([4]int, error) {
	if len(raw) == 0 {
		return [4]int{}, fmt.Errorf("%s: empty Reshape target shape", ctx)
	}
	total := int64(inSh[0]) * int64(inSh[1]) * int64(inSh[2]) * int64(inSh[3])
	resolved := make([]int64, len(raw))
	negIdx := -1
	knownProd := int64(1)

	for i, d := range raw {
		if d == -1 {
			if negIdx != -1 {
				return [4]int{}, fmt.Errorf("%s: Reshape target %v has multiple -1 dimensions", ctx, raw)
			}
			negIdx = i
			continue
		}
		if d == 0 && !allowZero {
			if i >= 4 {
				return [4]int{}, fmt.Errorf("%s: Reshape dim 0 out of range for 4D input", ctx)
			}
			d = int64(inSh[i])
		}
		if d <= 0 {
			return [4]int{}, fmt.Errorf("%s: Reshape target %v has non-positive dim[%d]=%d", ctx, raw, i, d)
		}
		resolved[i] = d
		var err error
		knownProd, err = mulInt64Checked(knownProd, d)
		if err != nil {
			return [4]int{}, fmt.Errorf("%s: %w", ctx, err)
		}
	}
	if negIdx != -1 {
		if knownProd == 0 || total%knownProd != 0 {
			return [4]int{}, fmt.Errorf("%s: cannot infer -1 in Reshape target %v for input size %d", ctx, raw, total)
		}
		resolved[negIdx] = total / knownProd
	} else if knownProd != total {
		return [4]int{}, fmt.Errorf("%s: Reshape target %v element count %d != input element count %d", ctx, resolved, knownProd, total)
	}

	switch len(resolved) {
	case 4:
		if resolved[0] != 1 {
			return [4]int{}, fmt.Errorf("%s: Reshape output batch dimension N must be 1, got %d", ctx, resolved[0])
		}
		return [4]int{1, int(resolved[1]), int(resolved[2]), int(resolved[3])}, nil
	case 2:
		if resolved[0] != 1 {
			return [4]int{}, fmt.Errorf("%s: 2D Reshape output batch dimension N must be 1, got %d", ctx, resolved[0])
		}
		return [4]int{1, int(resolved[1]), 1, 1}, nil
	default:
		return [4]int{}, fmt.Errorf("%s: Reshape target rank %d not supported (expected 4D [1,C,H,W] or 2D [1,C])", ctx, len(resolved))
	}
}

func resolveSliceParams(nd *irNode, consts map[string]*constSource, inSh [4]int, ctx string) (beg, end, step []int, outSh [4]int, err error) {
	var starts, ends, axes, steps []int64
	if len(nd.inputs) >= 3 {
		// Opset >= 10: inputs [data, starts, ends, axes?, steps?]
		readI64 := func(idx int, label string) ([]int64, error) {
			cs, ok := consts[nd.inputs[idx]]
			if !ok {
				return nil, fmt.Errorf("%s: Slice %s %q must be a constant int64 tensor", ctx, label, nd.inputs[idx])
			}
			return cs.int64s()
		}
		starts, err = readI64(1, "starts")
		if err != nil {
			return nil, nil, nil, outSh, err
		}
		ends, err = readI64(2, "ends")
		if err != nil {
			return nil, nil, nil, outSh, err
		}
		if len(nd.inputs) >= 4 && nd.inputs[3] != "" {
			axes, err = readI64(3, "axes")
			if err != nil {
				return nil, nil, nil, outSh, err
			}
		}
		if len(nd.inputs) >= 5 && nd.inputs[4] != "" {
			steps, err = readI64(4, "steps")
			if err != nil {
				return nil, nil, nil, outSh, err
			}
		}
	} else {
		// Opset < 10: attributes starts, ends, axes
		var ok1, ok2 bool
		starts, ok1 = nd.orig.AttrInts("starts")
		ends, ok2 = nd.orig.AttrInts("ends")
		if !ok1 || !ok2 {
			return nil, nil, nil, outSh, fmt.Errorf("%s: Slice missing starts/ends", ctx)
		}
		axes, _ = nd.orig.AttrInts("axes")
	}

	if len(starts) != len(ends) || len(starts) == 0 {
		return nil, nil, nil, outSh, fmt.Errorf("%s: Slice starts (%d) and ends (%d) length mismatch", ctx, len(starts), len(ends))
	}
	if len(axes) == 0 {
		axes = make([]int64, len(starts))
		for i := range axes {
			axes[i] = int64(i)
		}
	} else if len(axes) != len(starts) {
		return nil, nil, nil, outSh, fmt.Errorf("%s: Slice axes length %d != starts length %d", ctx, len(axes), len(starts))
	}
	if len(steps) == 0 {
		steps = make([]int64, len(starts))
		for i := range steps {
			steps[i] = 1
		}
	} else if len(steps) != len(starts) {
		return nil, nil, nil, outSh, fmt.Errorf("%s: Slice steps length %d != starts length %d", ctx, len(steps), len(starts))
	}

	beg = []int{0, 0, 0, 0}
	end = []int{inSh[0], inSh[1], inSh[2], inSh[3]}
	step = []int{1, 1, 1, 1}
	var seenAxis [4]bool

	for i := range starts {
		ax := axes[i]
		if ax < 0 {
			ax += 4
		}
		if ax < 0 || ax >= 4 {
			return nil, nil, nil, outSh, fmt.Errorf("%s: Slice axis %d out of range [0,3]", ctx, axes[i])
		}
		if seenAxis[ax] {
			return nil, nil, nil, outSh, fmt.Errorf("%s: Slice duplicate axis %d", ctx, ax)
		}
		seenAxis[ax] = true
		st := steps[i]
		if st <= 0 {
			return nil, nil, nil, outSh, fmt.Errorf("%s: Slice step %d must be > 0", ctx, st)
		}
		dim := int64(inSh[ax])
		s := starts[i]
		e := ends[i]
		if s < 0 {
			s += dim
		}
		if e < 0 {
			e += dim
		}
		if s < 0 {
			s = 0
		} else if s > dim {
			s = dim
		}
		if e < 0 {
			e = 0
		} else if e > dim {
			e = dim
		}
		if e <= s {
			return nil, nil, nil, outSh, fmt.Errorf("%s: Slice on axis %d produces empty range [%d,%d)", ctx, ax, s, e)
		}
		beg[ax] = int(s)
		end[ax] = int(e)
		step[ax] = int(st)
	}

	for d := 0; d < 4; d++ {
		outSh[d] = 1 + (end[d]-beg[d]-1)/step[d]
	}
	if outSh[0] != 1 {
		return nil, nil, nil, outSh, fmt.Errorf("%s: Slice output batch dimension must be 1, got %d", ctx, outSh[0])
	}
	return beg, end, step, outSh, nil
}

func isChannelBiasShape(dims []int64, cout int64) bool {
	if cout <= 0 {
		return false
	}
	switch len(dims) {
	case 1:
		return dims[0] == cout
	case 2:
		return dims[0] == 1 && dims[1] == cout
	case 4:
		return dims[0] == 1 && dims[1] == cout && dims[2] == 1 && dims[3] == 1
	default:
		return false
	}
}

func resolveConv3x3Padding(h, w, stride int, autoPad string, pads []int64, ctx string) (padMode string, ho, wo int, err error) {
	var onnxHo, onnxWo, pt, pl, pb, pr int
	switch autoPad {
	case "VALID":
		if h < 3 || w < 3 {
			return "", 0, 0, fmt.Errorf("%s: conv3x3 VALID requires H>=3 and W>=3, got (%d,%d)", ctx, h, w)
		}
		onnxHo = (h-3)/stride + 1
		onnxWo = (w-3)/stride + 1
	case "SAME_UPPER":
		onnxHo = (h + stride - 1) / stride
		onnxWo = (w + stride - 1) / stride
		padH := (onnxHo-1)*stride + 3 - h
		if padH < 0 {
			padH = 0
		}
		padW := (onnxWo-1)*stride + 3 - w
		if padW < 0 {
			padW = 0
		}
		pt = padH / 2
		pb = padH - pt
		pl = padW / 2
		pr = padW - pl
	case "SAME_LOWER":
		onnxHo = (h + stride - 1) / stride
		onnxWo = (w + stride - 1) / stride
		padH := (onnxHo-1)*stride + 3 - h
		if padH < 0 {
			padH = 0
		}
		padW := (onnxWo-1)*stride + 3 - w
		if padW < 0 {
			padW = 0
		}
		pt = (padH + 1) / 2
		pb = padH - pt
		pl = (padW + 1) / 2
		pr = padW - pl
	case "", "NOTSET":
		pt, pl, pb, pr = int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])
		if pt < 0 || pl < 0 || pb < 0 || pr < 0 {
			return "", 0, 0, fmt.Errorf("%s: conv3x3 negative pads %v", ctx, pads)
		}
		if h+pt+pb < 3 || w+pl+pr < 3 {
			return "", 0, 0, fmt.Errorf("%s: conv3x3 padded spatial dims (%d,%d) < 3", ctx, h+pt+pb, w+pl+pr)
		}
		onnxHo = (h+pt+pb-3)/stride + 1
		onnxWo = (w+pl+pr-3)/stride + 1
	default:
		return "", 0, 0, fmt.Errorf("%s: unsupported auto_pad %q", ctx, autoPad)
	}

	if hoV, woV, ptV, plV, errV := ref.Conv3x3Geometry(h, w, stride, "valid"); errV == nil {
		if onnxHo == hoV && onnxWo == woV && pt == ptV && pl == plV && pb == 0 && pr == 0 {
			return "valid", hoV, woV, nil
		}
	}
	if hoS, woS, ptS, plS, errS := ref.Conv3x3Geometry(h, w, stride, "same"); errS == nil {
		padTotalH := (hoS-1)*stride + 3 - h
		if padTotalH < 0 {
			padTotalH = 0
		}
		padTotalW := (woS-1)*stride + 3 - w
		if padTotalW < 0 {
			padTotalW = 0
		}
		pbS := padTotalH - ptS
		prS := padTotalW - plS
		if onnxHo == hoS && onnxWo == woS && pt == ptS && pl == plS && pb == pbS && pr == prS {
			return "same", hoS, woS, nil
		}
	}

	return "", 0, 0, fmt.Errorf(
		"%s: conv3x3 padding (auto_pad=%q, pads=[%d,%d,%d,%d], out=%dx%d) for %dx%d stride=%d does not match Vane conv3x3 'valid' or 'same' (Vane 'same' uses SAME_UPPER: [0,0,1,1] for even H/W and [1,1,1,1] for odd H/W at stride 2)",
		ctx, autoPad, pt, pl, pb, pr, onnxHo, onnxWo, h, w, stride)
}

func validateReduceBackendConstraints(redOp, axesStr string, inSh [4]int, ctx string) error {
	rc, rh, rw, err := ref.ParseReduceAxes(axesStr)
	if err != nil {
		return fmt.Errorf("%s: %w", ctx, err)
	}
	if rw || inSh[3] == 1 {
		return nil
	}
	if rc && !rh && !rw {
		switch strings.ToLower(redOp) {
		case "max", "min", "sum", "mean":
			return nil
		default:
			return fmt.Errorf("%s: reduce %q over axis 'c' with W=%d > 1 is unsupported by Vane reduce backend (supported: max, min, sum, mean)", ctx, redOp, inSh[3])
		}
	}
	return fmt.Errorf("%s: reduce %q over axes %q with W=%d > 1 is unsupported by Vane reduce backend (requires reducing W, axes='c', or W=1)", ctx, redOp, axesStr, inSh[3])
}
