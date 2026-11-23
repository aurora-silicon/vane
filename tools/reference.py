"""Independent float64/NumPy reference operators and fp16 comparison harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ComparisonResult:
    """Summary metrics for comparing two fp16 tensors."""

    max_abs_err: float
    max_rel_err: float
    rmse: float
    exact_fp16_count: int
    total_elements: int
    max_ulp_err: int
    nonfinite_mismatches: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_fp16_array(x: Any) -> np.ndarray:
    """Convert array-like input to contiguous IEEE-754 float16."""
    return np.ascontiguousarray(np.asarray(x, dtype=np.float16))


def _fp16_to_f64(x: Any) -> np.ndarray:
    """Quantize input to IEEE-754 float16 and widen to float64 for accumulation."""
    return _to_fp16_array(x).astype(np.float64)


def _fp16_ordered_int(arr_f16: np.ndarray) -> np.ndarray:
    """Map finite IEEE-754 float16 values to a monotonic int32 ULP lattice (-0.0 == +0.0 -> 0)."""
    bits = np.ascontiguousarray(arr_f16, dtype=np.float16).view(np.uint16)
    mag = (bits & np.uint16(0x7FFF)).astype(np.int32)
    neg = (bits & np.uint16(0x8000)) != 0
    return np.where(neg, -mag, mag)


def compare_fp16(
    actual: Any,
    expected: Any,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    max_ulp: int | None = None,
) -> ComparisonResult:
    """Compare two fp16 tensors with absolute, relative, RMSE, ULP, and non-finite metrics."""
    act = _to_fp16_array(actual)
    exp = _to_fp16_array(expected)
    if act.shape != exp.shape:
        raise ValueError(f"shape mismatch: actual {act.shape} != expected {exp.shape}")

    total = int(act.size)
    if total == 0:
        return ComparisonResult(
            max_abs_err=0.0,
            max_rel_err=0.0,
            rmse=0.0,
            exact_fp16_count=0,
            total_elements=0,
            max_ulp_err=0,
            nonfinite_mismatches=0,
            passed=True,
        )

    act_f64 = act.astype(np.float64)
    exp_f64 = exp.astype(np.float64)

    act_nan = np.isnan(act_f64)
    exp_nan = np.isnan(exp_f64)
    both_nan = act_nan & exp_nan
    nan_mismatch = act_nan ^ exp_nan

    act_inf = np.isinf(act_f64)
    exp_inf = np.isinf(exp_f64)
    both_same_inf = act_inf & exp_inf & (np.sign(act_f64) == np.sign(exp_f64))
    inf_mismatch = (act_inf | exp_inf) & ~both_same_inf & ~act_nan & ~exp_nan

    nonfinite_mismatch_mask = nan_mismatch | inf_mismatch
    nonfinite_mismatches = int(np.count_nonzero(nonfinite_mismatch_mask))

    both_finite = np.isfinite(act_f64) & np.isfinite(exp_f64)
    exact_mask = both_nan | both_same_inf | (both_finite & (act == exp))
    exact_count = int(np.count_nonzero(exact_mask))

    if np.any(both_finite):
        a_fin = act_f64[both_finite]
        e_fin = exp_f64[both_finite]
        abs_diff = np.abs(a_fin - e_fin)
        max_abs_err = float(np.max(abs_diff))
        rmse = float(np.sqrt(np.mean(abs_diff * abs_diff)))

        denom = np.abs(e_fin)
        rel_diff = np.where(abs_diff == 0.0, 0.0, np.inf)
        np.divide(abs_diff, denom, out=rel_diff, where=denom > 0.0)
        max_rel_err = float(np.max(rel_diff))

        a_ord = _fp16_ordered_int(act[both_finite])
        e_ord = _fp16_ordered_int(exp[both_finite])
        ulp_diff = np.abs(a_ord - e_ord)
        max_ulp_err = int(np.max(ulp_diff))

        tol_ok = bool(np.all(abs_diff <= (atol + rtol * np.abs(e_fin))))
        ulp_ok = True if max_ulp is None else (max_ulp_err <= max_ulp)
    else:
        max_abs_err = 0.0
        max_rel_err = 0.0
        rmse = 0.0
        max_ulp_err = 0
        tol_ok = True
        ulp_ok = True

    passed = (nonfinite_mismatches == 0) and tol_ok and ulp_ok
    return ComparisonResult(
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        rmse=rmse,
        exact_fp16_count=exact_count,
        total_elements=total,
        max_ulp_err=max_ulp_err,
        nonfinite_mismatches=nonfinite_mismatches,
        passed=passed,
    )


def compare_arrays(
    actual: Any,
    expected: Any,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    max_ulp: int | None = None,
) -> ComparisonResult:
    """Alias for `compare_fp16` accepting array-like inputs."""
    return compare_fp16(actual, expected, atol=atol, rtol=rtol, max_ulp=max_ulp)


def compare_fp16_bytes(
    actual_bytes: bytes | bytearray | memoryview,
    expected_bytes: bytes | bytearray | memoryview,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    max_ulp: int | None = None,
) -> ComparisonResult:
    """Compare two raw little-endian fp16 byte buffers."""
    if len(actual_bytes) != len(expected_bytes):
        raise ValueError(
            f"byte length mismatch: actual {len(actual_bytes)} != expected {len(expected_bytes)}"
        )
    if len(actual_bytes) % 2 != 0:
        raise ValueError(f"fp16 byte buffer length {len(actual_bytes)} must be even")
    act = np.frombuffer(actual_bytes, dtype="<f2")
    exp = np.frombuffer(expected_bytes, dtype="<f2")
    return compare_fp16(act, exp, atol=atol, rtol=rtol, max_ulp=max_ulp)


def pack_surface_fp16(
    arr: Any,
    *,
    batch_stride: int | None = None,
    plane_stride: int | None = None,
    row_stride: int | None = None,
    total_size: int | None = None,
) -> bytes:
    """Pack an (N, C, H, W) fp16 array into a strided little-endian byte surface."""
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f2"))
    if a.ndim != 4:
        raise ValueError(f"expected 4D (N, C, H, W) tensor, got shape {a.shape}")
    n, c, h, w = a.shape
    min_row = w * 2
    rs = min_row if row_stride is None else int(row_stride)
    if rs < min_row:
        raise ValueError(f"row_stride {rs} < width * 2 ({min_row})")
    min_plane = h * rs
    ps = min_plane if plane_stride is None else int(plane_stride)
    if ps < min_plane:
        raise ValueError(f"plane_stride {ps} < height * row_stride ({min_plane})")
    min_batch = c * ps
    bs = min_batch if batch_stride is None else int(batch_stride)
    if bs < min_batch:
        raise ValueError(f"batch_stride {bs} < channels * plane_stride ({min_batch})")
    span = n * bs
    size = span if total_size is None else int(total_size)
    if size < span:
        raise ValueError(f"total_size {size} < span ({span})")

    buf = bytearray(size)
    row_bytes = w * 2
    for ni in range(n):
        b_off = ni * bs
        for ci in range(c):
            p_off = b_off + ci * ps
            for hi in range(h):
                r_off = p_off + hi * rs
                buf[r_off : r_off + row_bytes] = a[ni, ci, hi].tobytes()
    return bytes(buf)


def unpack_surface_fp16(
    data: bytes | bytearray | memoryview,
    shape: tuple[int, int, int, int],
    *,
    batch_stride: int | None = None,
    plane_stride: int | None = None,
    row_stride: int | None = None,
) -> np.ndarray:
    """Unpack a strided little-endian fp16 byte surface into an (N, C, H, W) float16 array."""
    if len(shape) != 4:
        raise ValueError(f"expected 4D shape (N, C, H, W), got {shape}")
    n, c, h, w = (int(d) for d in shape)
    if n <= 0 or c <= 0 or h <= 0 or w <= 0:
        raise ValueError(f"all dimensions must be > 0, got {shape}")

    min_row = w * 2
    rs = min_row if row_stride is None else int(row_stride)
    if rs < min_row:
        raise ValueError(f"row_stride {rs} < width * 2 ({min_row})")
    min_plane = h * rs
    ps = min_plane if plane_stride is None else int(plane_stride)
    if ps < min_plane:
        raise ValueError(f"plane_stride {ps} < height * row_stride ({min_plane})")
    min_batch = c * ps
    bs = min_batch if batch_stride is None else int(batch_stride)
    if bs < min_batch:
        raise ValueError(f"batch_stride {bs} < channels * plane_stride ({min_batch})")
    span = n * bs
    mv = memoryview(data)
    if len(mv) < span:
        raise ValueError(f"buffer size {len(mv)} < required span {span}")

    out = np.empty((n, c, h, w), dtype=np.float16)
    row_bytes = w * 2
    for ni in range(n):
        b_off = ni * bs
        for ci in range(c):
            p_off = b_off + ci * ps
            for hi in range(h):
                r_off = p_off + hi * rs
                out[ni, ci, hi] = np.frombuffer(mv[r_off : r_off + row_bytes], dtype="<f2")
    return out


def _parse_pair(v: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(v, int):
        return (v, v)
    if len(v) == 2:
        return (int(v[0]), int(v[1]))
    raise ValueError(f"{name} must be an int or 2-tuple, got {v!r}")


def relu(x: Any) -> np.ndarray:
    """Reference fp16 ReLU evaluated in float64."""
    xf = _fp16_to_f64(x)
    return np.maximum(xf, 0.0).astype(np.float16)


def conv2d_1x1(
    x: Any,
    weight: Any,
    bias: Any | None = None,
    relu: bool = False,
) -> np.ndarray:
    """Reference 1x1 convolution with float64 accumulation and fp16 output rounding.

    Args:
        x: Input tensor of shape (N, Cin, H, W).
        weight: Filter tensor of shape (Cout, Cin) or (Cout, Cin, 1, 1).
        bias: Optional bias vector of shape (Cout,).
        relu: If True, apply ReLU before rounding to fp16.
    """
    xf = _fp16_to_f64(x)
    if xf.ndim != 4:
        raise ValueError(f"conv2d_1x1 expects 4D input (N, Cin, H, W), got {xf.shape}")
    wf = _fp16_to_f64(weight)
    if wf.ndim == 4:
        if wf.shape[2:] != (1, 1):
            raise ValueError(f"conv2d_1x1 expects 1x1 kernel, got {wf.shape}")
        wf = wf[:, :, 0, 0]
    elif wf.ndim != 2:
        raise ValueError(f"conv2d_1x1 expects 2D or 4D weight, got {wf.shape}")

    _, cin, _, _ = xf.shape
    cout, w_cin = wf.shape
    if w_cin != cin:
        raise ValueError(f"conv2d_1x1 channel mismatch: input Cin={cin}, weight Cin={w_cin}")

    y = np.einsum("nchw,oc->nohw", xf, wf, optimize=True)
    if bias is not None:
        bf = _fp16_to_f64(bias)
        if bf.shape != (cout,):
            raise ValueError(f"conv2d_1x1 bias shape {bf.shape} != ({cout},)")
        y = y + bf.reshape(1, cout, 1, 1)
    if relu:
        y = np.maximum(y, 0.0)
    return y.astype(np.float16)


def conv2d_3x3(
    x: Any,
    weight: Any,
    bias: Any | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 1,
    groups: int = 1,
    relu: bool = False,
) -> np.ndarray:
    """Reference dense, grouped, or depthwise 3x3 convolution in float64 -> fp16.

    Args:
        x: Input tensor of shape (N, Cin, H, W).
        weight: Filter tensor of shape (Cout, Cin // groups, 3, 3).
        bias: Optional bias vector of shape (Cout,).
        stride: Spatial stride (int or (sh, sw)).
        padding: Zero-padding (int or (ph, pw)).
        groups: Group count (1 for dense, Cin == Cout for depthwise).
        relu: If True, apply ReLU before rounding to fp16.
    """
    xf = _fp16_to_f64(x)
    wf = _fp16_to_f64(weight)
    if xf.ndim != 4:
        raise ValueError(f"conv2d_3x3 expects 4D input (N, Cin, H, W), got {xf.shape}")
    if wf.ndim != 4 or wf.shape[2:] != (3, 3):
        raise ValueError(f"conv2d_3x3 expects weight shape (Cout, Cin//g, 3, 3), got {wf.shape}")

    sh, sw = _parse_pair(stride, "stride")
    ph, pw = _parse_pair(padding, "padding")
    if sh <= 0 or sw <= 0:
        raise ValueError(f"stride must be > 0, got {(sh, sw)}")
    if ph < 0 or pw < 0:
        raise ValueError(f"padding must be >= 0, got {(ph, pw)}")
    if groups <= 0:
        raise ValueError(f"groups must be > 0, got {groups}")

    n, cin, h, w = xf.shape
    cout, cin_per_g, _, _ = wf.shape
    if cin % groups != 0 or cout % groups != 0:
        raise ValueError(f"Cin={cin} and Cout={cout} must be divisible by groups={groups}")
    if cin_per_g != cin // groups:
        raise ValueError(f"weight Cin//groups={cin_per_g} != {cin // groups}")

    padded = np.pad(xf, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant", constant_values=0.0)
    hp, wp = padded.shape[2], padded.shape[3]
    if hp < 3 or wp < 3:
        raise ValueError(f"padded spatial size {(hp, wp)} smaller than 3x3 kernel")
    hout = (hp - 3) // sh + 1
    wout = (wp - 3) // sw + 1

    cout_per_g = cout // groups
    y = np.zeros((n, cout, hout, wout), dtype=np.float64)

    for g in range(groups):
        c_in_start = g * cin_per_g
        c_out_start = g * cout_per_g
        x_g = padded[:, c_in_start : c_in_start + cin_per_g, :, :]
        w_g = wf[c_out_start : c_out_start + cout_per_g]
        acc = np.zeros((n, cout_per_g, hout, wout), dtype=np.float64)
        for kh in range(3):
            for kw in range(3):
                patch = x_g[:, :, kh : kh + hout * sh : sh, kw : kw + wout * sw : sw]
                acc += np.einsum("nchw,oc->nohw", patch, w_g[:, :, kh, kw], optimize=True)
        y[:, c_out_start : c_out_start + cout_per_g, :, :] = acc

    if bias is not None:
        bf = _fp16_to_f64(bias)
        if bf.shape != (cout,):
            raise ValueError(f"conv2d_3x3 bias shape {bf.shape} != ({cout},)")
        y = y + bf.reshape(1, cout, 1, 1)
    if relu:
        y = np.maximum(y, 0.0)
    return y.astype(np.float16)


def elementwise(op: str, a: Any, b: Any) -> np.ndarray:
    """Reference two-source elementwise op (add, sub, mul, max, min) in float64 -> fp16."""
    af = _fp16_to_f64(a)
    bf = _fp16_to_f64(b)
    if op == "add":
        y = af + bf
    elif op == "sub":
        y = af - bf
    elif op == "mul":
        y = af * bf
    elif op == "max":
        y = np.maximum(af, bf)
    elif op == "min":
        y = np.minimum(af, bf)
    else:
        raise ValueError(f"unsupported elementwise op '{op}' (expected add, sub, mul, max, min)")
    return y.astype(np.float16)


def affine(x: Any, scale: Any, bias: Any, relu: bool = False) -> np.ndarray:
    """Reference affine transform `x * scale + bias` in float64 -> fp16.

    When `x` is 4D (N, C, H, W) and `scale`/`bias` are 1D of length C, they are
    broadcast along channel axis 1; otherwise standard NumPy broadcasting applies.
    """
    xf = _fp16_to_f64(x)
    sf = _fp16_to_f64(scale)
    bf = _fp16_to_f64(bias)
    if xf.ndim == 4:
        c = xf.shape[1]
        if sf.ndim == 1 and sf.shape[0] == c:
            sf = sf.reshape(1, c, 1, 1)
        if bf.ndim == 1 and bf.shape[0] == c:
            bf = bf.reshape(1, c, 1, 1)
    y = xf * sf + bf
    if relu:
        y = np.maximum(y, 0.0)
    return y.astype(np.float16)
