package ref

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/niteria/vane/internal/fp16"
)

// GraphInputSpec describes a network input surface in a multi-task graph JSON spec.
type GraphInputSpec struct {
	Name      string `json:"name"`
	C         int    `json:"c"`
	H         int    `json:"h"`
	W         int    `json:"w"`
	ShortName string `json:"short_name,omitempty"`
}

// GraphActSpec represents an activation field in graph JSON, accepting either a plain
// string ("gelu") or a parameterized object ({"name": "hard_sigmoid", "alpha": 0.2, ...}).
type GraphActSpec struct {
	Name   string   `json:"name"`
	Alpha  *float64 `json:"alpha,omitempty"`
	Beta   *float64 `json:"beta,omitempty"`
	Lo     *float64 `json:"lo,omitempty"`
	Hi     *float64 `json:"hi,omitempty"`
	Eps    *float64 `json:"eps,omitempty"`
	Scalar *float64 `json:"scalar,omitempty"`
}

// UnmarshalJSON accepts either a JSON string or a JSON object for GraphActSpec.
func (a *GraphActSpec) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		*a = GraphActSpec{Name: s}
		return nil
	}
	type alias GraphActSpec
	var obj alias
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	*a = GraphActSpec(obj)
	return nil
}

// MarshalJSON emits a plain string when no parameters are set, or an object otherwise.
func (a GraphActSpec) MarshalJSON() ([]byte, error) {
	if a.Alpha == nil && a.Beta == nil && a.Lo == nil && a.Hi == nil && a.Eps == nil && a.Scalar == nil {
		return json.Marshal(a.Name)
	}
	type alias GraphActSpec
	return json.Marshal(alias(a))
}

// ToActSpec converts GraphActSpec to a normalized ActSpec with defaults matching the CLI.
func (a GraphActSpec) ToActSpec() (ActSpec, error) {
	low := strings.ToLower(a.Name)
	spec := ActSpec{Name: low}
	switch low {
	case "", "none", "relu", "copy", "relu6", "abs", "sign", "tanh", "erf", "sigmoid", "gelu", "silu", "exp2", "exp", "sqrt", "inverse", "log2":
		return spec, nil
	case "clip":
		if a.Lo == nil || a.Hi == nil {
			return spec, fmt.Errorf("clip requires both lo and hi")
		}
		spec.Lo, spec.Hi = *a.Lo, *a.Hi
		return spec, nil
	case "leaky_relu":
		spec.Alpha = 0.01
		if a.Alpha != nil {
			spec.Alpha = *a.Alpha
		}
		return spec, nil
	case "threshold":
		if a.Scalar != nil {
			spec.Scalar = *a.Scalar
		} else if a.Alpha != nil {
			spec.Scalar = *a.Alpha
		} else if a.Lo != nil {
			spec.Scalar = *a.Lo
		}
		return spec, nil
	case "thresholded_relu":
		spec.Alpha = 1.0
		if a.Alpha != nil {
			spec.Alpha = *a.Alpha
		} else if a.Scalar != nil {
			spec.Alpha = *a.Scalar
		} else if a.Lo != nil {
			spec.Alpha = *a.Lo
		}
		return spec, nil
	case "hard_sigmoid":
		spec.Alpha, spec.Beta = 0.2, 0.5
		if a.Alpha != nil {
			spec.Alpha = *a.Alpha
		}
		if a.Beta != nil {
			spec.Beta = *a.Beta
		}
		return spec, nil
	case "clamp_max":
		if a.Scalar != nil {
			spec.Scalar = *a.Scalar
		} else if a.Lo != nil {
			spec.Scalar = *a.Lo
		} else if a.Alpha != nil {
			spec.Scalar = *a.Alpha
		}
		return spec, nil
	case "clamp_min":
		if a.Scalar != nil {
			spec.Scalar = *a.Scalar
		} else if a.Hi != nil {
			spec.Scalar = *a.Hi
		} else if a.Alpha != nil {
			spec.Scalar = *a.Alpha
		}
		return spec, nil
	case "elu":
		spec.Alpha = 1.0
		if a.Alpha != nil {
			spec.Alpha = *a.Alpha
		}
		return spec, nil
	case "scaled_tanh":
		spec.Alpha, spec.Beta = 1.0, 1.0
		if a.Alpha != nil {
			spec.Alpha = *a.Alpha
		}
		if a.Beta != nil {
			spec.Beta = *a.Beta
		}
		return spec, nil
	case "rsqrt":
		if a.Eps != nil {
			spec.Eps = *a.Eps
		}
		return spec, nil
	default:
		return spec, fmt.Errorf("unknown activation %q", a.Name)
	}
}

// GraphNodeSpec describes one node in topological order inside a graph.json spec.
// Note: Bias is stored as json.RawMessage so the same JSON field name "bias" can hold
// a filename string or boolean on conv nodes, or a numeric scalar on PE chain nodes.
type GraphNodeSpec struct {
	ID           string          `json:"id"`
	Op           string          `json:"op"`
	Inputs       []string        `json:"inputs"`
	OutPlacement string          `json:"out_placement,omitempty"`
	Mode         string          `json:"mode,omitempty"`
	Cin          int             `json:"cin,omitempty"`
	Cout         int             `json:"cout,omitempty"`
	Stride       int             `json:"stride,omitempty"`
	Pad          string          `json:"pad,omitempty"`
	Weights      string          `json:"weights,omitempty"`
	Bias         json.RawMessage `json:"bias,omitempty"`
	HasBias      bool            `json:"has_bias,omitempty"`
	Relu         bool            `json:"relu,omitempty"`
	Act          *GraphActSpec   `json:"act,omitempty"`
	PeOp         string          `json:"pe_op,omitempty"`
	Alpha        *float64        `json:"alpha,omitempty"`
	Beta         *float64        `json:"beta,omitempty"`
	InScale      *float64        `json:"in_scale,omitempty"`
	Scale        *float64        `json:"scale,omitempty"`
	Scalar       *float64        `json:"scalar,omitempty"`
}

// ConvBiasSpec parses a conv node's Bias / HasBias fields, returning (wantsBias, path, err).
func (n GraphNodeSpec) ConvBiasSpec() (bool, string, error) {
	if len(n.Bias) == 0 || string(n.Bias) == "null" {
		return n.HasBias, "", nil
	}
	var b bool
	if err := json.Unmarshal(n.Bias, &b); err == nil {
		return b || n.HasBias, "", nil
	}
	var s string
	if err := json.Unmarshal(n.Bias, &s); err == nil {
		if s == "" {
			return n.HasBias, "", nil
		}
		return true, s, nil
	}
	return false, "", fmt.Errorf("node %q: conv 'bias' must be a path string or boolean", n.ID)
}

// PeBiasFloat parses a PE chain node's numeric Bias field (if present).
func (n GraphNodeSpec) PeBiasFloat() (*float64, error) {
	if len(n.Bias) == 0 || string(n.Bias) == "null" {
		return nil, nil
	}
	var v float64
	if err := json.Unmarshal(n.Bias, &v); err != nil {
		return nil, fmt.Errorf("node %q: PE 'bias' must be a number: %w", n.ID, err)
	}
	return &v, nil
}

// ResolveConvAct resolves a conv node's activation fields into (ActSpec, isFusedLut, error).
func (n GraphNodeSpec) ResolveConvAct() (ActSpec, bool, error) {
	if n.Act == nil || strings.ToLower(n.Act.Name) == "" {
		if n.Relu {
			return ActSpec{Name: "relu"}, false, nil
		}
		return ActSpec{Name: "none"}, false, nil
	}
	low := strings.ToLower(n.Act.Name)
	if low == "none" {
		if n.Relu {
			return ActSpec{}, false, fmt.Errorf("node %q: cannot combine relu=true with act=none", n.ID)
		}
		return ActSpec{Name: "none"}, false, nil
	}
	if low == "relu" {
		return ActSpec{Name: "relu"}, false, nil
	}
	if n.Relu {
		return ActSpec{}, false, fmt.Errorf("node %q: cannot combine relu=true with LUT act=%q", n.ID, n.Act.Name)
	}
	spec, err := n.Act.ToActSpec()
	if err != nil {
		return ActSpec{}, false, fmt.Errorf("node %q: %w", n.ID, err)
	}
	return spec, true, nil
}

// GraphSpec is the top-level multi-task DAG JSON specification shared with `vane compile-graph`.
type GraphSpec struct {
	Target    string           `json:"target"`
	Placement string           `json:"placement,omitempty"`
	Inputs    []GraphInputSpec `json:"inputs"`
	Nodes     []GraphNodeSpec  `json:"nodes"`
}

// GraphNodeWeights supplies fp16 weights and optional fp16 bias for a conv node.
type GraphNodeWeights struct {
	Weights []uint16
	Bias    []uint16
}

func graphF16CanonZero(v float64) uint16 {
	u := fp16.FromFloat64(v)
	if u == fp16.NegZero {
		return fp16.PosZero
	}
	return u
}

func resolvePe1ChainParams(nd GraphNodeSpec) (inScale, bias, scale *uint16, err error) {
	optU16 := func(p *float64) *uint16 {
		if p == nil {
			return nil
		}
		u := graphF16CanonZero(*p)
		return &u
	}
	switch strings.ToLower(nd.PeOp) {
	case "linear":
		a := uint16(fp16.FromFloat64(1.0))
		if nd.Alpha != nil {
			a = graphF16CanonZero(*nd.Alpha)
		}
		b := uint16(fp16.PosZero)
		if nd.Beta != nil {
			b = graphF16CanonZero(*nd.Beta)
		}
		return &a, &b, nil, nil
	case "chain":
		pb, err := nd.PeBiasFloat()
		if err != nil {
			return nil, nil, nil, err
		}
		return optU16(nd.InScale), optU16(pb), optU16(nd.Scale), nil
	case "sadd":
		if nd.Scalar == nil {
			return nil, nil, nil, fmt.Errorf("node %q: sadd requires scalar", nd.ID)
		}
		b := graphF16CanonZero(*nd.Scalar)
		return nil, &b, nil, nil
	case "ssub":
		if nd.Scalar == nil {
			return nil, nil, nil, fmt.Errorf("node %q: ssub requires scalar", nd.ID)
		}
		sBits := graphF16CanonZero(*nd.Scalar)
		var b uint16
		if (sBits & 0x7fff) == 0 {
			b = fp16.PosZero
		} else {
			b = sBits ^ 0x8000
		}
		return nil, &b, nil, nil
	case "srsub":
		if nd.Scalar == nil {
			return nil, nil, nil, fmt.Errorf("node %q: srsub requires scalar", nd.ID)
		}
		negOne := fp16.FromFloat64(-1.0)
		b := graphF16CanonZero(*nd.Scalar)
		return &negOne, &b, nil, nil
	case "smul":
		if nd.Scalar == nil {
			return nil, nil, nil, fmt.Errorf("node %q: smul requires scalar", nd.ID)
		}
		a := graphF16CanonZero(*nd.Scalar)
		return &a, nil, nil, nil
	case "sdiv":
		if nd.Scalar == nil {
			return nil, nil, nil, fmt.Errorf("node %q: sdiv requires scalar", nd.ID)
		}
		sv := fp16.ToFloat64(graphF16CanonZero(*nd.Scalar))
		if sv == 0 {
			return nil, nil, nil, fmt.Errorf("node %q: sdiv scalar cannot be zero", nd.ID)
		}
		inv := 1.0 / sv
		u := graphF16CanonZero(inv)
		if !fp16.IsFinite(u) {
			return nil, nil, nil, fmt.Errorf("node %q: sdiv reciprocal overflows fp16", nd.ID)
		}
		return &u, nil, nil, nil
	default:
		return nil, nil, nil, fmt.Errorf("node %q: unknown PE chain op %q", nd.ID, nd.PeOp)
	}
}

func peUnaryF64(op string, x Tensor) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	out, _ := NewTensor(x.N, x.C, x.H, x.W)
	switch op {
	case "abs":
		for i, h := range x.Data {
			out.Data[i] = fp16.FromFloat64(math.Abs(fp16.ToFloat64(h)))
		}
	case "square":
		for i, h := range x.Data {
			v := fp16.ToFloat64(h)
			out.Data[i] = fp16.FromFloat64(v * v)
		}
	default:
		return Tensor{}, fmt.Errorf("unsupported PE unary op %q", op)
	}
	return out, nil
}

func evalGraph(spec GraphSpec, inputs map[string]Tensor, nodeWeights map[string]GraphNodeWeights, exactHW bool) (Tensor, map[string]Tensor, error) {
	if len(spec.Inputs) < 1 || len(spec.Inputs) > 3 {
		return Tensor{}, nil, fmt.Errorf("graph must have 1..3 inputs, got %d", len(spec.Inputs))
	}
	if len(spec.Nodes) == 0 {
		return Tensor{}, nil, fmt.Errorf("graph must have at least 1 node")
	}

	env := make(map[string]Tensor, len(spec.Inputs)+len(spec.Nodes))
	for _, inp := range spec.Inputs {
		t, ok := inputs[inp.Name]
		if !ok {
			return Tensor{}, nil, fmt.Errorf("missing graph input tensor %q", inp.Name)
		}
		if err := t.Validate(); err != nil {
			return Tensor{}, nil, fmt.Errorf("invalid input tensor %q: %w", inp.Name, err)
		}
		if t.N != 1 || t.C != inp.C || t.H != inp.H || t.W != inp.W {
			return Tensor{}, nil, fmt.Errorf("input tensor %q shape (%d,%d,%d,%d) != spec (1,%d,%d,%d)",
				inp.Name, t.N, t.C, t.H, t.W, inp.C, inp.H, inp.W)
		}
		env[inp.Name] = t
	}

	var lastOut Tensor
	for _, nd := range spec.Nodes {
		if nd.ID == "" {
			return Tensor{}, nil, fmt.Errorf("graph node has empty id")
		}
		if _, exists := env[nd.ID]; exists {
			return Tensor{}, nil, fmt.Errorf("duplicate tensor/node id %q", nd.ID)
		}

		var out Tensor
		var err error
		switch strings.ToLower(nd.Op) {
		case "conv1x1":
			if len(nd.Inputs) != 1 {
				return Tensor{}, nil, fmt.Errorf("node %q (conv1x1): expected 1 input, got %d", nd.ID, len(nd.Inputs))
			}
			x, ok := env[nd.Inputs[0]]
			if !ok {
				return Tensor{}, nil, fmt.Errorf("node %q: unknown input %q", nd.ID, nd.Inputs[0])
			}
			nw, ok := nodeWeights[nd.ID]
			if !ok {
				return Tensor{}, nil, fmt.Errorf("node %q: missing weights", nd.ID)
			}
			actSpec, isLut, errAct := nd.ResolveConvAct()
			if errAct != nil {
				return Tensor{}, nil, errAct
			}
			if exactHW {
				if isLut {
					tbl, preScale, shift, errL := BuildLutTable(actSpec)
					if errL != nil {
						return Tensor{}, nil, fmt.Errorf("node %q: %w", nd.ID, errL)
					}
					out, err = Conv2d1x1Q1516Lut(x, nw.Weights, nd.Cout, nw.Bias, tbl, preScale, shift)
				} else {
					out, err = Conv2d1x1Q1516(x, nw.Weights, nd.Cout, nw.Bias, actSpec.Name == "relu")
				}
			} else {
				out, err = Conv2d1x1Act(x, nw.Weights, nd.Cout, nw.Bias, actSpec)
			}

		case "conv3x3":
			if len(nd.Inputs) != 1 {
				return Tensor{}, nil, fmt.Errorf("node %q (conv3x3): expected 1 input, got %d", nd.ID, len(nd.Inputs))
			}
			x, ok := env[nd.Inputs[0]]
			if !ok {
				return Tensor{}, nil, fmt.Errorf("node %q: unknown input %q", nd.ID, nd.Inputs[0])
			}
			nw, ok := nodeWeights[nd.ID]
			if !ok {
				return Tensor{}, nil, fmt.Errorf("node %q: missing weights", nd.ID)
			}
			mode := strings.ToLower(nd.Mode)
			if mode == "" {
				mode = "dense"
			}
			groups := 1
			cout := nd.Cout
			switch mode {
			case "dense":
				groups = 1
			case "depthwise":
				groups = x.C
				if cout == 0 {
					cout = x.C
				}
			default:
				return Tensor{}, nil, fmt.Errorf("node %q: unsupported conv3x3 mode %q", nd.ID, nd.Mode)
			}
			stride := nd.Stride
			if stride == 0 {
				stride = 1
			}
			pad := strings.ToLower(nd.Pad)
			if pad == "" {
				pad = "same"
			}
			actSpec, isLut, errAct := nd.ResolveConvAct()
			if errAct != nil {
				return Tensor{}, nil, errAct
			}
			if exactHW {
				if isLut {
					tbl, preScale, shift, errL := BuildLutTable(actSpec)
					if errL != nil {
						return Tensor{}, nil, fmt.Errorf("node %q: %w", nd.ID, errL)
					}
					out, err = Conv2d3x3Q1516Lut(x, nw.Weights, cout, nw.Bias, stride, pad, groups, tbl, preScale, shift)
				} else {
					out, err = Conv2d3x3Q1516(x, nw.Weights, cout, nw.Bias, stride, pad, groups, actSpec.Name == "relu")
				}
			} else {
				out, err = Conv2d3x3PadAct(x, nw.Weights, cout, nw.Bias, stride, pad, groups, actSpec)
			}

		case "pe":
			peOp := strings.ToLower(nd.PeOp)
			switch peOp {
			case "abs", "square":
				if len(nd.Inputs) != 1 {
					return Tensor{}, nil, fmt.Errorf("node %q (pe %s): expected 1 input", nd.ID, peOp)
				}
				x, ok := env[nd.Inputs[0]]
				if !ok {
					return Tensor{}, nil, fmt.Errorf("node %q: unknown input %q", nd.ID, nd.Inputs[0])
				}
				if exactHW {
					out, err = PeUnary(peOp, x)
				} else {
					out, err = peUnaryF64(peOp, x)
				}
			case "linear", "chain", "sadd", "ssub", "srsub", "smul", "sdiv":
				if len(nd.Inputs) != 1 {
					return Tensor{}, nil, fmt.Errorf("node %q (pe %s): expected 1 input", nd.ID, peOp)
				}
				x, ok := env[nd.Inputs[0]]
				if !ok {
					return Tensor{}, nil, fmt.Errorf("node %q: unknown input %q", nd.ID, nd.Inputs[0])
				}
				inSc, b, sc, errP := resolvePe1ChainParams(nd)
				if errP != nil {
					return Tensor{}, nil, errP
				}
				if exactHW {
					out, err = PeChain(x, inSc, b, sc)
				} else {
					out, err = PeChainF64(x, inSc, b, sc)
				}
			case "add", "sub", "rsub", "mul", "max", "min":
				if len(nd.Inputs) != 2 {
					return Tensor{}, nil, fmt.Errorf("node %q (pe %s): expected 2 inputs", nd.ID, peOp)
				}
				x, ok1 := env[nd.Inputs[0]]
				if !ok1 {
					return Tensor{}, nil, fmt.Errorf("node %q: unknown input %q", nd.ID, nd.Inputs[0])
				}
				z, ok2 := env[nd.Inputs[1]]
				if !ok2 {
					return Tensor{}, nil, fmt.Errorf("node %q: unknown input %q", nd.ID, nd.Inputs[1])
				}
				if exactHW {
					out, err = PeBinary(peOp, x, z, false)
				} else {
					out, err = PeBinaryF64(peOp, x, z, false)
				}
			default:
				return Tensor{}, nil, fmt.Errorf("node %q: unknown pe_op %q", nd.ID, nd.PeOp)
			}

		case "lut":
			if len(nd.Inputs) != 1 {
				return Tensor{}, nil, fmt.Errorf("node %q (lut): expected 1 input", nd.ID)
			}
			x, ok := env[nd.Inputs[0]]
			if !ok {
				return Tensor{}, nil, fmt.Errorf("node %q: unknown input %q", nd.ID, nd.Inputs[0])
			}
			if nd.Act == nil || nd.Act.Name == "" {
				return Tensor{}, nil, fmt.Errorf("node %q (lut): missing act field", nd.ID)
			}
			actSpec, errA := nd.Act.ToActSpec()
			if errA != nil {
				return Tensor{}, nil, fmt.Errorf("node %q: %w", nd.ID, errA)
			}
			if exactHW {
				switch actSpec.Name {
				case "relu":
					out, err = Relu(x)
				case "copy":
					cp := make([]uint16, len(x.Data))
					copy(cp, x.Data)
					out = Tensor{N: x.N, C: x.C, H: x.H, W: x.W, Data: cp}
				default:
					tbl, preScale, shift, errL := BuildLutTable(actSpec)
					if errL != nil {
						return Tensor{}, nil, fmt.Errorf("node %q: %w", nd.ID, errL)
					}
					out, err = ApplyLutTensor(x, tbl, preScale, shift)
				}
			} else {
				out, err = ApplyActF64(x, actSpec)
			}

		default:
			return Tensor{}, nil, fmt.Errorf("node %q: unsupported op %q", nd.ID, nd.Op)
		}

		if err != nil {
			return Tensor{}, nil, fmt.Errorf("node %q: %w", nd.ID, err)
		}
		env[nd.ID] = out
		lastOut = out
	}

	return lastOut, env, nil
}

// EvalGraphQ1516 evaluates a multi-task DAG using the exact hardware arithmetic models:
// Q15.16 NE conv accumulation, 128-byte NE LUT interpolation, and H17 PE RNA16 rounding.
func EvalGraphQ1516(spec GraphSpec, inputs map[string]Tensor, nodeWeights map[string]GraphNodeWeights) (Tensor, map[string]Tensor, error) {
	return evalGraph(spec, inputs, nodeWeights, true)
}

// EvalGraphF64 evaluates a multi-task DAG using analytical float64 operators rounded to fp16
// at each node boundary.
func EvalGraphF64(spec GraphSpec, inputs map[string]Tensor, nodeWeights map[string]GraphNodeWeights) (Tensor, map[string]Tensor, error) {
	return evalGraph(spec, inputs, nodeWeights, false)
}
