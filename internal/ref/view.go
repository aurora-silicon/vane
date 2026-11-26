package ref

import "fmt"

// Transpose permutes the 4D NCHW axes of x according to perm (a permutation of {0,1,2,3}),
// preserving exact IEEE-754 binary16 bit patterns.
func Transpose(x Tensor, perm [4]int) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	var seen [4]bool
	for i, p := range perm {
		if p < 0 || p > 3 {
			return Tensor{}, fmt.Errorf("transpose: perm[%d]=%d out of range 0..3", i, p)
		}
		if seen[p] {
			return Tensor{}, fmt.Errorf("transpose: duplicate axis %d in perm %v", p, perm)
		}
		seen[p] = true
	}

	inShape := [4]int{x.N, x.C, x.H, x.W}
	outShape := [4]int{inShape[perm[0]], inShape[perm[1]], inShape[perm[2]], inShape[perm[3]]}
	out, err := NewTensor(outShape[0], outShape[1], outShape[2], outShape[3])
	if err != nil {
		return Tensor{}, err
	}

	var outIdx, inIdx [4]int
	for on := 0; on < out.N; on++ {
		outIdx[0] = on
		for oc := 0; oc < out.C; oc++ {
			outIdx[1] = oc
			for oh := 0; oh < out.H; oh++ {
				outIdx[2] = oh
				for ow := 0; ow < out.W; ow++ {
					outIdx[3] = ow
					inIdx[perm[0]] = outIdx[0]
					inIdx[perm[1]] = outIdx[1]
					inIdx[perm[2]] = outIdx[2]
					inIdx[perm[3]] = outIdx[3]
					out.Data[out.Index(on, oc, oh, ow)] = x.Data[x.Index(inIdx[0], inIdx[1], inIdx[2], inIdx[3])]
				}
			}
		}
	}
	return out, nil
}

// Slice extracts x[begin[0]:end[0]:step[0], ..., begin[3]:end[3]:step[3]],
// preserving exact IEEE-754 binary16 bit patterns.
func Slice(x Tensor, begin, end, step [4]int) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	inShape := [4]int{x.N, x.C, x.H, x.W}
	var outShape [4]int
	for ax := 0; ax < 4; ax++ {
		if step[ax] <= 0 {
			return Tensor{}, fmt.Errorf("slice: step[%d]=%d must be >= 1", ax, step[ax])
		}
		if begin[ax] < 0 || begin[ax] >= end[ax] || end[ax] > inShape[ax] {
			return Tensor{}, fmt.Errorf("slice: axis %d range [%d, %d) invalid for dim %d", ax, begin[ax], end[ax], inShape[ax])
		}
		outShape[ax] = (end[ax] - begin[ax] + step[ax] - 1) / step[ax]
	}

	out, err := NewTensor(outShape[0], outShape[1], outShape[2], outShape[3])
	if err != nil {
		return Tensor{}, err
	}
	for on := 0; on < out.N; on++ {
		inN := begin[0] + on*step[0]
		for oc := 0; oc < out.C; oc++ {
			inC := begin[1] + oc*step[1]
			for oh := 0; oh < out.H; oh++ {
				inH := begin[2] + oh*step[2]
				for ow := 0; ow < out.W; ow++ {
					inW := begin[3] + ow*step[3]
					out.Data[out.Index(on, oc, oh, ow)] = x.Data[x.Index(inN, inC, inH, inW)]
				}
			}
		}
	}
	return out, nil
}

// Concat concatenates two or more 4D NCHW tensors along axis (0..3),
// preserving exact IEEE-754 binary16 bit patterns.
func Concat(xs []Tensor, axis int) (Tensor, error) {
	if len(xs) < 2 {
		return Tensor{}, fmt.Errorf("concat: expected at least 2 inputs, got %d", len(xs))
	}
	if axis < 0 || axis > 3 {
		return Tensor{}, fmt.Errorf("concat: axis %d out of range 0..3", axis)
	}
	if err := xs[0].Validate(); err != nil {
		return Tensor{}, fmt.Errorf("concat input[0]: %w", err)
	}
	base := [4]int{xs[0].N, xs[0].C, xs[0].H, xs[0].W}
	totalAxis := 0
	for i, t := range xs {
		if err := t.Validate(); err != nil {
			return Tensor{}, fmt.Errorf("concat input[%d]: %w", i, err)
		}
		sh := [4]int{t.N, t.C, t.H, t.W}
		for ax := 0; ax < 4; ax++ {
			if ax != axis && sh[ax] != base[ax] {
				return Tensor{}, fmt.Errorf("concat: input[%d] axis %d dim %d != input[0] dim %d", i, ax, sh[ax], base[ax])
			}
		}
		totalAxis += sh[axis]
	}

	outShape := base
	outShape[axis] = totalAxis
	out, err := NewTensor(outShape[0], outShape[1], outShape[2], outShape[3])
	if err != nil {
		return Tensor{}, err
	}

	offset := 0
	for _, t := range xs {
		for n := 0; n < t.N; n++ {
			for c := 0; c < t.C; c++ {
				for h := 0; h < t.H; h++ {
					for w := 0; w < t.W; w++ {
						idx := [4]int{n, c, h, w}
						idx[axis] += offset
						out.Data[out.Index(idx[0], idx[1], idx[2], idx[3])] = t.Data[t.Index(n, c, h, w)]
					}
				}
			}
		}
		sh := [4]int{t.N, t.C, t.H, t.W}
		offset += sh[axis]
	}
	return out, nil
}

// SplitPart extracts partition part of splitting x along axis into slices of lengths sizes,
// preserving exact IEEE-754 binary16 bit patterns.
func SplitPart(x Tensor, axis int, sizes []int, part int) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if axis < 0 || axis > 3 {
		return Tensor{}, fmt.Errorf("split: axis %d out of range 0..3", axis)
	}
	if len(sizes) == 0 {
		return Tensor{}, fmt.Errorf("split: sizes must be non-empty")
	}
	if part < 0 || part >= len(sizes) {
		return Tensor{}, fmt.Errorf("split: part %d out of range 0..%d", part, len(sizes)-1)
	}
	inShape := [4]int{x.N, x.C, x.H, x.W}
	total := 0
	start := 0
	for i, sz := range sizes {
		if sz <= 0 {
			return Tensor{}, fmt.Errorf("split: sizes[%d]=%d must be >= 1", i, sz)
		}
		if i < part {
			start += sz
		}
		total += sz
	}
	if total != inShape[axis] {
		return Tensor{}, fmt.Errorf("split: sum(sizes)=%d != input axis %d dim %d", total, axis, inShape[axis])
	}

	begin := [4]int{0, 0, 0, 0}
	end := inShape
	begin[axis] = start
	end[axis] = start + sizes[part]
	return Slice(x, begin, end, [4]int{1, 1, 1, 1})
}

// Reshape reinterprets the contiguous NCHW fp16 elements of x with outShape,
// preserving exact IEEE-754 binary16 bit patterns.
func Reshape(x Tensor, outShape [4]int) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	for ax, d := range outShape {
		if d <= 0 {
			return Tensor{}, fmt.Errorf("reshape: outShape[%d]=%d must be >= 1", ax, d)
		}
	}
	want := outShape[0] * outShape[1] * outShape[2] * outShape[3]
	if want != len(x.Data) {
		return Tensor{}, fmt.Errorf("reshape: element count mismatch (%d != %d)", want, len(x.Data))
	}
	out, err := NewTensor(outShape[0], outShape[1], outShape[2], outShape[3])
	if err != nil {
		return Tensor{}, err
	}
	copy(out.Data, x.Data)
	return out, nil
}
