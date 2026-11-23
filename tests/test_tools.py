"""Unit and CLI integration tests for vane Python tools."""

from __future__ import annotations

import ctypes
import errno
import json
import mmap
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tools import ane_runner, reference


REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_vane_bin() -> Path:
    env_bin = os.environ.get("VANE_BIN")
    if env_bin:
        return Path(env_bin)
    return REPO_ROOT / ".lake" / "build" / "bin" / "vane"


class FakeMmap:
    """In-memory mmap buffer backed by a shared bytearray slice."""

    def __init__(self, backing: bytearray, writable: bool) -> None:
        self._backing = backing
        self._writable = writable
        self.closed = False

    def __getitem__(self, key: slice) -> bytes:
        if self.closed:
            raise ValueError("mmap closed")
        return bytes(self._backing[key])

    def __setitem__(self, key: slice, value: bytes) -> None:
        if self.closed:
            raise ValueError("mmap closed")
        if not self._writable:
            raise TypeError("mmap is read-only")
        self._backing[key] = value

    def close(self) -> None:
        self.closed = True


class MockAneTransport(ane_runner.DeviceTransport):
    """In-memory mock of Linux accel/ane ABI v3 ioctl + mmap contract."""

    def __init__(
        self,
        *,
        devices: dict[str, dict[str, Any]],
        compute_fn: Callable[[list[bytes], int], bytes] | None = None,
        host_page_size: int = ane_runner.PAGE_SIZE_16K,
    ) -> None:
        self.devices = devices
        self.compute_fn = compute_fn
        self._host_page_size = host_page_size
        self._next_fd = 100
        self._open_fds: dict[int, dict[str, Any]] = {}
        self.observed_output_poison_ok: list[bool] = []
        self.loaded_sections: dict[int, bytes] = {}
        self.exec_calls = 0

    def list_devices(self) -> list[str]:
        return sorted(self.devices.keys())

    def open(self, path: str) -> int:
        if path not in self.devices:
            raise OSError(errno.ENOENT, "No such file or directory", path)
        spec = self.devices[path]
        if "open_errno" in spec:
            raise OSError(spec["open_errno"], "open failed", path)
        fd = self._next_fd
        self._next_fd += 1
        self._open_fds[fd] = {
            "path": path,
            "spec": spec,
            "loaded": None,
            "surfaces": [],
        }
        return fd

    def close(self, fd: int) -> None:
        self._open_fds.pop(fd, None)

    def page_size(self) -> int:
        return self._host_page_size

    def mmap(self, fd: int, length: int, prot: int, flags: int, offset: int) -> FakeMmap:
        state = self._open_fds[fd]
        loaded = state["loaded"]
        if loaded is None:
            raise OSError(errno.EINVAL, "no loaded program")
        idx = (offset >> 32) - 1
        if idx < 0 or idx >= len(loaded["surface_sizes"]):
            raise OSError(errno.EINVAL, "invalid mmap offset")
        direction = loaded["surface_directions"][idx]
        writable = bool(prot & mmap.PROT_WRITE)
        if direction == 2 and writable:
            raise OSError(errno.EACCES, "output surface cannot be mapped writable")
        return FakeMmap(state["surfaces"][idx], writable=writable)

    def ioctl(self, fd: int, request: int, buf: bytearray) -> None:
        state = self._open_fds[fd]
        spec = state["spec"]

        if request == ane_runner.DRM_IOCTL_ANE_GET_CAPS:
            if spec.get("get_caps_errno"):
                raise OSError(spec["get_caps_errno"], "GET_CAPS failed")
            flags, _sz, _abi, _chip, _ss, _bs, _es, pad = ane_runner.STRUCT_GET_CAPS.unpack(buf)
            if flags != 0 or pad != 0:
                raise OSError(errno.EINVAL, "nonzero flags/pad")
            buf[:] = ane_runner.STRUCT_GET_CAPS.pack(
                0,
                ane_runner.STRUCT_GET_CAPS.size,
                spec["abi_version"],
                spec["chip_family"],
                0,
                0,
                0,
                0,
            )
            return

        if request == ane_runner.DRM_IOCTL_ANE_H16_LOAD:
            if state["loaded"] is not None:
                raise OSError(errno.EBUSY, "already loaded")
            fields = ane_runner.STRUCT_H16_LOAD.unpack(buf)
            hwx_ptr, hwx_size, sec_ptr, sec_count, flags, _sc, pad = fields[:7]
            if sec_count != 4 or flags != 0 or pad != 0:
                raise OSError(errno.EINVAL, "bad load header")
            hwx_bytes = ctypes.string_at(hwx_ptr, hwx_size)
            sec_raw = ctypes.string_at(sec_ptr, sec_count * ane_runner.STRUCT_H16_SECTION.size)
            seen_kinds: set[int] = set()
            for i in range(sec_count):
                kind, sz, dptr = ane_runner.STRUCT_H16_SECTION.unpack_from(
                    sec_raw, i * ane_runner.STRUCT_H16_SECTION.size
                )
                seen_kinds.add(kind)
                self.loaded_sections[kind] = ctypes.string_at(dptr, sz)
            if seen_kinds != {1, 2, 3, 4}:
                raise OSError(errno.EINVAL, "missing section kinds")

            # Derive surface spans & directions directly from the HWX tensor descriptors
            surf_sizes, surf_dirs = self._parse_hwx_surfaces(hwx_bytes)
            state["loaded"] = {
                "surface_sizes": surf_sizes,
                "surface_directions": surf_dirs,
            }
            state["surfaces"] = [
                bytearray(ane_runner.align_up(sz, ane_runner.PAGE_SIZE_16K)) for sz in surf_sizes
            ]
            padded_sizes = surf_sizes + [0] * (4 - len(surf_sizes))
            padded_dirs = surf_dirs + [0] * (4 - len(surf_dirs))
            buf[:] = ane_runner.STRUCT_H16_LOAD.pack(
                hwx_ptr,
                hwx_size,
                sec_ptr,
                sec_count,
                0,
                len(surf_sizes),
                0,
                *padded_sizes,
                *padded_dirs,
            )
            return

        if request == ane_runner.DRM_IOCTL_ANE_H16_EXEC:
            loaded = state["loaded"]
            if loaded is None:
                raise OSError(errno.EINVAL, "no program loaded")
            bufs_ptr, count, flags = ane_runner.STRUCT_H16_EXEC.unpack(buf)
            surf_sizes = loaded["surface_sizes"]
            surf_dirs = loaded["surface_directions"]
            if count != len(surf_sizes):
                raise OSError(errno.EINVAL, "surface count mismatch")

            self.exec_calls += 1
            if flags == ane_runner.DRM_ANE_H16_EXEC_IN_PLACE:
                if bufs_ptr != 0:
                    raise OSError(errno.EINVAL, "in-place requires buffers_ptr == 0")
                inputs = [
                    bytes(state["surfaces"][i][: surf_sizes[i]])
                    for i, d in enumerate(surf_dirs)
                    if d == 1
                ]
                out_idx = surf_dirs.index(2)
                out_sz = surf_sizes[out_idx]
                out_data = self._run_compute(inputs, out_sz)
                state["surfaces"][out_idx][:out_sz] = out_data
                return

            if flags != 0:
                raise OSError(errno.EINVAL, "unknown exec flags")

            raw_bufs = ctypes.string_at(bufs_ptr, count * ane_runner.STRUCT_H16_BUFFER.size)
            inputs = []
            out_ptr = 0
            out_sz = 0
            for i in range(count):
                ptr, sz = ane_runner.STRUCT_H16_BUFFER.unpack_from(
                    raw_bufs, i * ane_runner.STRUCT_H16_BUFFER.size
                )
                if sz != surf_sizes[i]:
                    raise OSError(errno.EINVAL, f"surface {i} size mismatch")
                if surf_dirs[i] == 1:
                    inputs.append(ctypes.string_at(ptr, sz))
                else:
                    out_ptr = ptr
                    out_sz = sz
                    pre = ctypes.string_at(ptr, sz)
                    self.observed_output_poison_ok.append(all(b == 0xFF for b in pre))

            out_data = self._run_compute(inputs, out_sz)
            ctypes.memmove(out_ptr, out_data, out_sz)
            return

        raise OSError(errno.ENOTTY, f"unknown ioctl {request:#x}")

    def _run_compute(self, inputs: list[bytes], out_size: int) -> bytes:
        if self.compute_fn is not None:
            return self.compute_fn(inputs, self.exec_calls)
        # Default deterministic compute: echo first input (truncated/zero-padded to out_size)
        base = inputs[0] if inputs else b""
        return (base + bytes(out_size))[:out_size]

    @staticmethod
    def _parse_hwx_surfaces(hwx: bytes) -> tuple[list[int], list[int]]:
        ncmds = struct.unpack_from("<I", hwx, 0x10)[0]
        cursor = 32
        buf_refs: list[str] = []
        tensors: dict[str, tuple[int, int]] = {}  # short_name -> (direction, span)
        for _ in range(ncmds):
            cmd, cmdsize = struct.unpack_from("<2I", hwx, cursor)
            if cmd == 0x40:
                raw = hwx[cursor + 24 : cursor + 32]
                buf_refs.append(raw[: raw.find(0)].decode("ascii"))
            elif cmd == 4:
                subtype = struct.unpack_from("<I", hwx, cursor + 8)[0]
                if subtype == 3:
                    direction = struct.unpack_from("<I", hwx, cursor + 0x14)[0]
                    n = struct.unpack_from("<I", hwx, cursor + 0x28)[0]
                    batch_stride = struct.unpack_from("<Q", hwx, cursor + 0x50)[0]
                    shortoff = struct.unpack_from("<I", hwx, cursor + 0x7C)[0]
                    end = hwx.find(0, cursor + shortoff, cursor + cmdsize)
                    sname = hwx[cursor + shortoff : end].decode("ascii")
                    tensors[sname] = (direction, n * batch_stride)
            cursor += cmdsize
        sizes = [tensors[nm][1] for nm in buf_refs]
        dirs = [tensors[nm][0] for nm in buf_refs]
        return sizes, dirs


class TestUapiLayout(unittest.TestCase):
    """Verify ane_runner struct sizes and ioctl encodings match Linux accel/ane ABI v3."""

    def test_struct_sizes(self) -> None:
        self.assertEqual(ane_runner.STRUCT_GET_CAPS.size, 32)
        self.assertEqual(ane_runner.STRUCT_H16_SECTION.size, 16)
        self.assertEqual(ane_runner.STRUCT_H16_LOAD.size, 88)
        self.assertEqual(ane_runner.STRUCT_H16_BUFFER.size, 16)
        self.assertEqual(ane_runner.STRUCT_H16_EXEC.size, 16)

    def test_ioctl_constants(self) -> None:
        self.assertEqual(
            ane_runner.DRM_IOCTL_ANE_GET_CAPS,
            (3 << 30) | (32 << 16) | (0x64 << 8) | (0x40 + 0x00),
        )
        self.assertEqual(
            ane_runner.DRM_IOCTL_ANE_H16_LOAD,
            (3 << 30) | (88 << 16) | (0x64 << 8) | (0x40 + 0x07),
        )
        self.assertEqual(
            ane_runner.DRM_IOCTL_ANE_H16_EXEC,
            (1 << 30) | (16 << 16) | (0x64 << 8) | (0x40 + 0x08),
        )
        self.assertEqual(ane_runner.drm_ane_h16_mmap_offset(0), 1 << 32)
        self.assertEqual(ane_runner.drm_ane_h16_mmap_offset(3), 4 << 32)


class TestReferenceHarness(unittest.TestCase):
    """Verify float64/NumPy reference operators, packing, and fp16 error metrics."""

    def test_compare_fp16_metrics_and_ulp(self) -> None:
        a = np.array([1.0, -0.0, 2.0, np.inf, np.nan], dtype=np.float16)
        b = np.array([1.0, +0.0, 2.0, np.inf, np.nan], dtype=np.float16)
        res = reference.compare_fp16(a, b, atol=0.0, rtol=0.0, max_ulp=0)
        self.assertTrue(res.passed)
        self.assertEqual(res.max_ulp_err, 0)
        self.assertEqual(res.nonfinite_mismatches, 0)
        self.assertEqual(res.exact_fp16_count, 5)

        # Bump 1.0 by exactly 1 ULP in float16 (0x3c00 -> 0x3c01)
        one_plus_ulp = np.array([0x3C01], dtype=np.uint16).view(np.float16)
        one = np.array([1.0], dtype=np.float16)
        res_ulp = reference.compare_fp16(one_plus_ulp, one, atol=1e-2, rtol=1e-2, max_ulp=1)
        self.assertTrue(res_ulp.passed)
        self.assertEqual(res_ulp.max_ulp_err, 1)
        self.assertEqual(res_ulp.exact_fp16_count, 0)

        res_fail = reference.compare_fp16(one_plus_ulp, one, atol=1e-2, rtol=1e-2, max_ulp=0)
        self.assertFalse(res_fail.passed)

        # Non-finite mismatch fails
        res_nf = reference.compare_fp16(
            np.array([np.nan], dtype=np.float16), np.array([1.0], dtype=np.float16)
        )
        self.assertFalse(res_nf.passed)
        self.assertEqual(res_nf.nonfinite_mismatches, 1)

    def test_strided_surface_pack_unpack_roundtrip(self) -> None:
        rng = np.random.default_rng(123)
        arr = rng.normal(size=(2, 3, 5, 7)).astype(np.float16)
        packed = reference.pack_surface_fp16(
            arr, row_stride=32, plane_stride=256, batch_stride=1024, total_size=4096
        )
        self.assertEqual(len(packed), 4096)
        unpacked = reference.unpack_surface_fp16(
            packed, (2, 3, 5, 7), row_stride=32, plane_stride=256, batch_stride=1024
        )
        np.testing.assert_array_equal(unpacked, arr)

    def test_conv2d_1x1(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(1, 4, 3, 5)).astype(np.float16)
        w = rng.normal(size=(6, 4, 1, 1)).astype(np.float16)
        b = rng.normal(size=(6,)).astype(np.float16)

        y = reference.conv2d_1x1(x, w, bias=b, relu=True)
        # Direct scalar float64 check
        xf = x.astype(np.float64)
        wf = w[:, :, 0, 0].astype(np.float64)
        bf = b.astype(np.float64)
        expected = np.zeros((1, 6, 3, 5), dtype=np.float64)
        for o in range(6):
            for c in range(4):
                expected[0, o] += xf[0, c] * wf[o, c]
            expected[0, o] = np.maximum(expected[0, o] + bf[o], 0.0)
        np.testing.assert_array_equal(y, expected.astype(np.float16))

    def test_conv2d_3x3_dense_and_depthwise(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(size=(1, 3, 5, 5)).astype(np.float16)
        w_dense = rng.normal(size=(4, 3, 3, 3)).astype(np.float16)
        b_dense = rng.normal(size=(4,)).astype(np.float16)

        y_dense = reference.conv2d_3x3(x, w_dense, bias=b_dense, stride=2, padding=1, groups=1, relu=False)
        self.assertEqual(y_dense.shape, (1, 4, 3, 3))

        # Manual pixel verification at (0, 2, 1, 1) -> center (2, 2) in input
        xf = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
        val = float(np.sum(xf[0, :, 2:5, 2:5] * w_dense[2].astype(np.float64)) + b_dense[2].astype(np.float64))
        self.assertEqual(y_dense[0, 2, 1, 1], np.float16(val))

        # Depthwise 3x3 (groups == Cin == Cout)
        w_dw = rng.normal(size=(3, 1, 3, 3)).astype(np.float16)
        y_dw = reference.conv2d_3x3(x, w_dw, bias=None, stride=1, padding=1, groups=3, relu=True)
        self.assertEqual(y_dw.shape, (1, 3, 5, 5))
        for ch in range(3):
            v = float(np.sum(xf[0, ch, 1:4, 1:4] * w_dw[ch, 0].astype(np.float64)))
            self.assertEqual(y_dw[0, ch, 1, 1], np.float16(max(v, 0.0)))

    def test_elementwise_and_affine(self) -> None:
        a = np.array([[[[1.0, -2.0], [3.5, 0.25]]]], dtype=np.float16)
        b = np.array([[[[0.5, 4.0], [-1.5, 0.25]]]], dtype=np.float16)
        np.testing.assert_array_equal(reference.elementwise("add", a, b), (a.astype(np.float64) + b.astype(np.float64)).astype(np.float16))
        np.testing.assert_array_equal(reference.elementwise("sub", a, b), (a.astype(np.float64) - b.astype(np.float64)).astype(np.float16))
        np.testing.assert_array_equal(reference.elementwise("mul", a, b), (a.astype(np.float64) * b.astype(np.float64)).astype(np.float16))
        np.testing.assert_array_equal(reference.elementwise("max", a, b), np.maximum(a.astype(np.float64), b.astype(np.float64)).astype(np.float16))
        np.testing.assert_array_equal(reference.elementwise("min", a, b), np.minimum(a.astype(np.float64), b.astype(np.float64)).astype(np.float16))

        # Channel-wise affine on 4D tensor with ReLU
        x = np.array([[[[1.0, -3.0]], [[2.0, 0.5]]]], dtype=np.float16)  # (1, 2, 1, 2)
        scale = np.array([2.0, -1.0], dtype=np.float16)
        bias = np.array([0.5, 1.0], dtype=np.float16)
        y = reference.affine(x, scale, bias, relu=True)
        expected = np.array([[[[2.5, 0.0]], [[0.0, 0.5]]]], dtype=np.float16)
        np.testing.assert_array_equal(y, expected)


class TestRunnerAndCliIntegration(unittest.TestCase):
    """End-to-end tests of vane CLI + ane_runner against an in-memory mock device."""

    def test_skip_when_no_device(self) -> None:
        vane_bin = _find_vane_bin()
        self.assertTrue(vane_bin.exists(), f"vane binary not found at {vane_bin}")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "demo"
            subprocess.run(
                [str(vane_bin), "emit-demo", "--target", "h16", "--out-dir", str(out_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            transport = MockAneTransport(devices={})
            rc = ane_runner.run_cli(
                ["--hwx", str(out_dir / "program.hwx"), "--sections", str(out_dir)],
                transport=transport,
            )
            self.assertEqual(rc, ane_runner.EXIT_SKIP)

    def test_emit_demo_validate_and_mock_exec_copy_and_inplace(self) -> None:
        vane_bin = _find_vane_bin()
        self.assertTrue(vane_bin.exists(), f"vane binary not found at {vane_bin}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            demo_dir = tmp_path / "demo_h17"
            subprocess.run(
                [
                    str(vane_bin),
                    "emit-demo",
                    "--target",
                    "h17",
                    "--out-dir",
                    str(demo_dir),
                    "--with-kernel",
                    "--with-scratch",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            # Validate emitted HWX and 4 sections with vane validate
            val = subprocess.run(
                [
                    str(vane_bin),
                    "validate",
                    "--hwx",
                    str(demo_dir / "program.hwx"),
                    "--sections",
                    str(demo_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("status: ok", val.stdout)
            self.assertIn("target: h17", val.stdout)

            # Verify pure-Python section builder matches Lean `vane sections` byte-for-byte
            # and emits OPERATION BAR slot bindings in canonical ascending slot order
            hwx_bytes = (demo_dir / "program.hwx").read_bytes()
            py_secs = ane_runner.build_sections_from_hwx(hwx_bytes)
            for kind, fname in ane_runner.SECTION_FILENAMES.items():
                self.assertEqual(py_secs[kind], (demo_dir / fname).read_bytes(), f"mismatch on {fname}")

            op_bytes = py_secs[ane_runner.DRM_ANE_H16_SECTION_OPERATION]
            ref_count = struct.unpack_from("<I", op_bytes, 0x0C)[0]
            # demo_h17 references slots 6, 7, 8, 3 in encounter order -> emitted canonically as 3, 6, 7, 8
            emitted_pairs = [
                struct.unpack_from("<II", op_bytes, 0x10 + 8 * j) for j in range(ref_count)
            ]
            self.assertEqual(
                emitted_pairs,
                [(3, 6), (6, 5), (7, 4), (8, ane_runner.DRM_ANE_H16_RESOURCE_KERNEL)],
            )

            manifest = json.loads((demo_dir / "manifest.json").read_text(encoding="utf-8"))
            in_span = manifest["surfaces"][0]["span"]
            rng = np.random.default_rng(42)
            in_arr = rng.normal(size=(in_span // 2,)).astype(np.float16)
            in_bytes = in_arr.astype("<f2").tobytes()
            expected_arr = reference.relu(in_arr)
            expected_bytes = expected_arr.astype("<f2").tobytes()

            in_path = tmp_path / "in0.bin"
            exp_path = tmp_path / "expected.bin"
            json_report = tmp_path / "report.json"
            out_bin = tmp_path / "out.bin"
            in_path.write_bytes(in_bytes)
            exp_path.write_bytes(expected_bytes)

            def relu_compute(inputs: list[bytes], _call: int) -> bytes:
                arr = np.frombuffer(inputs[0], dtype="<f2")
                return reference.relu(arr).astype("<f2").tobytes()

            transport = MockAneTransport(
                devices={
                    "/dev/accel/accel0": {"get_caps_errno": errno.ENOTTY},  # non-ANE skipped
                    "/dev/accel/accel1": {
                        "abi_version": ane_runner.DRM_ANE_ABI_V3,
                        "chip_family": ane_runner.DRM_ANE_CHIP_H17,
                    },
                },
                compute_fn=relu_compute,
            )

            # Copy mode with 3 repeatability runs + poison check + reference comparison
            rc = ane_runner.run_cli(
                [
                    "--hwx",
                    str(demo_dir / "program.hwx"),
                    "--sections",
                    str(demo_dir),
                    "--input",
                    str(in_path),
                    "--expected",
                    str(exp_path),
                    "--runs",
                    "3",
                    "--out",
                    str(out_bin),
                    "--json-out",
                    str(json_report),
                ],
                transport=transport,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(transport.observed_output_poison_ok, [True, True, True])
            self.assertEqual(out_bin.read_bytes(), expected_bytes)

            report = json.loads(json_report.read_text(encoding="utf-8"))
            self.assertEqual(report["device"], "/dev/accel/accel1")
            self.assertEqual(report["runs"], 3)
            self.assertTrue(report["repeatable"])
            self.assertTrue(report["comparison"]["passed"])
            self.assertEqual(report["comparison"]["max_ulp_err"], 0)

            # In-place mmap mode
            rc_ip = ane_runner.run_cli(
                [
                    "--hwx",
                    str(demo_dir / "program.hwx"),
                    "--sections",
                    str(demo_dir),
                    "--input",
                    str(in_path),
                    "--expected",
                    str(exp_path),
                    "--runs",
                    "2",
                    "--in-place",
                ],
                transport=transport,
            )
            self.assertEqual(rc_ip, 0)

            # Repeatability failure detection (output changes on run 2)
            def flaky_compute(inputs: list[bytes], call: int) -> bytes:
                data = bytearray(relu_compute(inputs, call))
                if call % 2 == 0:
                    data[0] ^= 0x01
                return bytes(data)

            flaky_transport = MockAneTransport(
                devices={
                    "/dev/accel/accel0": {
                        "abi_version": ane_runner.DRM_ANE_ABI_V3,
                        "chip_family": ane_runner.DRM_ANE_CHIP_H17,
                    }
                },
                compute_fn=flaky_compute,
            )
            rc_flaky = ane_runner.run_cli(
                [
                    "--hwx",
                    str(demo_dir / "program.hwx"),
                    "--sections",
                    str(demo_dir),
                    "--input",
                    str(in_path),
                    "--runs",
                    "2",
                ],
                transport=flaky_transport,
            )
            self.assertEqual(rc_flaky, 1)

            # In-place mode rejects non-16 KiB host page size
            transport_4k = MockAneTransport(
                devices={
                    "/dev/accel/accel0": {
                        "abi_version": ane_runner.DRM_ANE_ABI_V3,
                        "chip_family": ane_runner.DRM_ANE_CHIP_H17,
                    }
                },
                compute_fn=relu_compute,
                host_page_size=4096,
            )
            rc_4k = ane_runner.run_cli(
                [
                    "--hwx",
                    str(demo_dir / "program.hwx"),
                    "--sections",
                    str(demo_dir),
                    "--in-place",
                ],
                transport=transport_4k,
            )
            self.assertEqual(rc_4k, 1)

            # Permission error (EACCES) fails with exit code 1 (not skip code 4)
            transport_perm = MockAneTransport(
                devices={"/dev/accel/accel0": {"open_errno": errno.EACCES}}
            )
            rc_perm = ane_runner.run_cli(
                ["--hwx", str(demo_dir / "program.hwx"), "--sections", str(demo_dir)],
                transport=transport_perm,
            )
            self.assertEqual(rc_perm, 1)

            # Reference mismatch returns exit code 1
            wrong_exp = tmp_path / "wrong_expected.bin"
            wrong_exp.write_bytes(bytes(len(expected_bytes)))
            rc_ref_fail = ane_runner.run_cli(
                [
                    "--hwx",
                    str(demo_dir / "program.hwx"),
                    "--vane-bin",
                    str(vane_bin),
                    "--input",
                    str(in_path),
                    "--expected",
                    str(wrong_exp),
                    "--atol",
                    "1e-4",
                    "--rtol",
                    "1e-4",
                ],
                transport=transport,
            )
            self.assertEqual(rc_ref_fail, 1)

            # Pure-Python section builder mode (neither --sections nor --vane-bin)
            rc_pysec = ane_runner.run_cli(
                [
                    "--hwx",
                    str(demo_dir / "program.hwx"),
                    "--input",
                    str(in_path),
                    "--expected",
                    str(exp_path),
                ],
                transport=transport,
            )
            self.assertEqual(rc_pysec, 0)

            # vane validate rejects corrupted HWX with non-zero exit code
            corrupt_hwx = tmp_path / "corrupt.hwx"
            corrupt_data = bytearray(hwx_bytes)
            corrupt_data[0] ^= 0xFF
            corrupt_hwx.write_bytes(bytes(corrupt_data))
            bad_val = subprocess.run(
                [str(vane_bin), "validate", "--hwx", str(corrupt_hwx)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(bad_val.returncode, 0)
            self.assertIn("magic", bad_val.stderr.lower())


if __name__ == "__main__":
    unittest.main()
