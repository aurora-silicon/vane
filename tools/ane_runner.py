#!/usr/bin/env python3
"""Linux accel/ane ABI v3 device test runner and host-built LOAD section builder."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import glob
import hashlib
import json
import mmap
import os
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from tools import reference
except ImportError:  # pragma: no cover - direct script invocation fallback
    import reference  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Linux accel/ane ABI v3 constants & ioctl definitions
# ---------------------------------------------------------------------------

DRM_ANE_ABI_V1 = 1
DRM_ANE_ABI_V2 = 2
DRM_ANE_ABI_V3 = 3

DRM_ANE_CHIP_H13 = 13
DRM_ANE_CHIP_H14 = 14
DRM_ANE_CHIP_H16 = 16
DRM_ANE_CHIP_H17 = 17

DRM_ANE_GET_CAPS = 0x00
DRM_ANE_H16_LOAD = 0x07
DRM_ANE_H16_EXEC = 0x08

DRM_ANE_H16_SECTION_GENERIC = 1
DRM_ANE_H16_SECTION_OPERATION = 2
DRM_ANE_H16_SECTION_PROCEDURE = 3
DRM_ANE_H16_SECTION_TEXT_PROPERTIES = 4
DRM_ANE_H16_SECTIONS = 4

DRM_ANE_H16_SECTION_MIN = 0x28
DRM_ANE_H16_SECTION_MAX = 0x10000

DRM_ANE_H16_RESOURCE_GENERIC = 1002
DRM_ANE_H16_RESOURCE_OPERATION = 1003
DRM_ANE_H16_RESOURCE_PROCEDURE = 1004
DRM_ANE_H16_RESOURCE_TEXT_PROPERTIES = 1005
DRM_ANE_H16_RESOURCE_TEXT = 1006
DRM_ANE_H16_RESOURCE_KERNEL = 1007

DRM_ANE_H16_SURFACES = 4
DRM_ANE_H16_EXEC_IN_PLACE = 1 << 0

PAGE_SIZE_16K = 0x4000
EXIT_SKIP = 4

_IOC_WRITE = 1
_IOC_READ = 2
_IOC_RW = _IOC_READ | _IOC_WRITE
_DRM_IOCTL_BASE = ord("d")
_DRM_COMMAND_BASE = 0x40


def drm_ioc(direction: int, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (_DRM_IOCTL_BASE << 8) | (_DRM_COMMAND_BASE + nr)


def drm_ane_h16_mmap_offset(surface_index: int) -> int:
    return (surface_index + 1) << 32


STRUCT_GET_CAPS = struct.Struct("<8I")  # 32 bytes
STRUCT_H16_SECTION = struct.Struct("<IIQ")  # 16 bytes
STRUCT_H16_LOAD = struct.Struct("<QQQIIII4Q4I")  # 88 bytes
STRUCT_H16_BUFFER = struct.Struct("<QQ")  # 16 bytes
STRUCT_H16_EXEC = struct.Struct("<QII")  # 16 bytes

DRM_IOCTL_ANE_GET_CAPS = drm_ioc(_IOC_RW, DRM_ANE_GET_CAPS, STRUCT_GET_CAPS.size)
DRM_IOCTL_ANE_H16_LOAD = drm_ioc(_IOC_RW, DRM_ANE_H16_LOAD, STRUCT_H16_LOAD.size)
DRM_IOCTL_ANE_H16_EXEC = drm_ioc(_IOC_WRITE, DRM_ANE_H16_EXEC, STRUCT_H16_EXEC.size)

SECTION_FILENAMES: dict[int, str] = {
    DRM_ANE_H16_SECTION_GENERIC: "generic.bin",
    DRM_ANE_H16_SECTION_OPERATION: "operation.bin",
    DRM_ANE_H16_SECTION_PROCEDURE: "procedure.bin",
    DRM_ANE_H16_SECTION_TEXT_PROPERTIES: "text-properties.bin",
}


def align_up(value: int, alignment: int) -> int:
    rem = value % alignment
    return value if rem == 0 else value + (alignment - rem)


# ---------------------------------------------------------------------------
# Pure-Python host-built LOAD section builder (from validated HWX bytes)
# ---------------------------------------------------------------------------

def _read_fixed_ascii(buf: bytes, off: int, width: int) -> str:
    raw = buf[off : off + width]
    nul = raw.find(0)
    if nul < 0:
        raise ValueError(f"missing NUL in fixed string at {off:#x}")
    return raw[:nul].decode("ascii")


def build_sections_from_hwx(hwx: bytes) -> dict[int, bytes]:
    """Derive the 4 host-built LOAD sections deterministically from HWX bytes."""
    if len(hwx) < 32:
        raise ValueError("HWX too small (< 32 bytes)")
    magic, cpu_type, cpu_subtype, file_type, ncmds, sizeofcmds, flags, reserved = struct.unpack_from(
        "<8I", hwx, 0
    )
    if magic != 0xBEEFFACE or cpu_type != 0x80 or file_type != 2 or flags != 0x200000 or reserved != 0:
        raise ValueError("invalid HWX file header")
    if cpu_subtype not in (7, 9):
        raise ValueError(f"unsupported HWX CPU subtype {cpu_subtype}")

    cursor = 32
    cmd_end = 32 + sizeofcmds
    if cmd_end > len(hwx):
        raise ValueError("HWX load commands exceed file length")

    text_sec: tuple[int, int, int] | None = None  # (addr, size, offset)
    kernel_addr: int | None = None
    scratch_info: tuple[int, int] | None = None  # (addr, size)
    surfaces: list[tuple[int, int, int]] = []  # (direction, addr, vmsize)
    prog_desc: bytes | None = None

    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<2I", hwx, cursor)
        if cmd == 0x19:  # LC_SEGMENT_64
            segname = _read_fixed_ascii(hwx, cursor + 8, 16)
            vmaddr, vmsize, fileoff, filesize, _maxprot, initprot, nsects, _sflags = struct.unpack_from(
                "<4Q4I", hwx, cursor + 0x18
            )
            for si in range(nsects):
                soff = cursor + 72 + si * 80
                sectname = _read_fixed_ascii(hwx, soff, 16)
                saddr, ssize = struct.unpack_from("<2Q", hwx, soff + 0x20)
                sfileoff = struct.unpack_from("<I", hwx, soff + 0x30)[0]
                if segname == "__TEXT" and sectname == "__text":
                    text_sec = (saddr, ssize, sfileoff)
                elif segname == "__KERN_0" and sectname == "__kern_0":
                    kernel_addr = saddr
                elif segname == "__DATA" and sectname == "__bss":
                    scratch_info = (saddr, ssize)
                elif segname == "__FVMLIB":
                    surfaces.append((initprot, saddr, vmsize))
        elif cmd == 4:
            subtype = struct.unpack_from("<I", hwx, cursor + 8)[0]
            if subtype == 4:
                prog_desc = hwx[cursor : cursor + cmdsize]
        cursor += cmdsize

    if text_sec is None or prog_desc is None:
        raise ValueError("HWX missing __TEXT,__text or program descriptor")

    _, text_size, text_off = text_sec
    text_bytes = hwx[text_off : text_off + text_size]

    # Walk task stream to count tasks and collect BAR slots referenced with bit 29 set
    pos = 16
    task_count = 0
    bar_slots: list[int] = []
    while pos < len(text_bytes):
        w0 = struct.unpack_from("<I", text_bytes, pos)[0]
        words = (w0 >> 16) & 0x7FF
        if words == 0:
            pos += 16
            continue
        task_end = pos + 4 * words
        w8 = struct.unpack_from("<I", text_bytes, pos + 32)[0]
        hdr_words = 10 if (w8 & 3) == 3 else 9
        rec = pos + 4 * hdr_words
        while rec < task_end:
            h = struct.unpack_from("<I", text_bytes, rec)[0]
            rec += 4
            if (h >> 31) & 1:
                mask = (h >> 15) & 0xFFFF
                count = 1 + bin(mask).count("1")
            else:
                count = ((h >> 15) & 0x3F) + 1
                if (h >> 29) & 1:
                    slot = (h >> 23) & 0x3F
                    if slot not in bar_slots:
                        bar_slots.append(slot)
            rec += 4 * count
        task_count += 1
        pos = align_up(task_end, 16) if task_end < len(text_bytes) else task_end

    # Assign buffer IDs: output = 4, inputs = 5, 6, ..., scratch = next
    next_in = 5
    surface_ids: list[int] = []
    for direction, _, _ in surfaces:
        if direction == 2:
            surface_ids.append(4)
        else:
            surface_ids.append(next_in)
            next_in += 1
    scratch_id: int | None = next_in if scratch_info is not None else None

    # 1. GENERIC
    entries = len(surfaces) + (1 if scratch_info is not None else 0)
    gen = bytearray(0x208 + entries * 0x30)
    struct.pack_into("<II", gen, 0x000, 1, 0x10)
    struct.pack_into("<I", gen, 0x204, entries)
    for i, (direction, _addr, vmsize) in enumerate(surfaces):
        base = 0x208 + i * 0x30
        struct.pack_into("<III", gen, base + 0x00, 1, surface_ids[i], direction - 1)
        struct.pack_into("<I", gen, base + 0x10, direction)
        struct.pack_into("<QI", gen, base + 0x20, vmsize, 0xFFFF)
    if scratch_info is not None and scratch_id is not None:
        base = 0x208 + len(surfaces) * 0x30
        struct.pack_into("<III", gen, base + 0x00, 1, scratch_id, 0)
        struct.pack_into("<I", gen, base + 0x10, 1)
        struct.pack_into("<QI", gen, base + 0x20, scratch_info[1], 0xFFFF)

    # 2. OPERATION
    bindings: list[tuple[int, int]] = []
    has_kernel_binding = False
    for slot in bar_slots:
        vm_base = struct.unpack_from("<Q", prog_desc, 0x10 * (slot + 1))[0]
        if vm_base == 0:
            continue
        tag: int | None = None
        for i, (_dir, saddr, _vmsize) in enumerate(surfaces):
            if saddr == vm_base:
                tag = surface_ids[i]
                break
        if tag is None and scratch_info is not None and scratch_info[0] == vm_base:
            tag = scratch_id
        if tag is None and kernel_addr is not None and kernel_addr == vm_base:
            tag = DRM_ANE_H16_RESOURCE_KERNEL
            has_kernel_binding = True
        if tag is None:
            raise ValueError(f"BAR slot {slot} VM base {vm_base:#x} does not match any resource")
        bindings.append((slot, tag))

    if not bindings:
        raise ValueError("task stream references no non-zero BAR slots")
    if kernel_addr is not None and not has_kernel_binding:
        raise ValueError("HWX has __KERN_0 but no kernel BAR binding")

    op = bytearray(0x410)
    struct.pack_into("<I", op, 0x00, 1)
    struct.pack_into("<I", op, 0x0C, len(bindings))
    for j, (slot, tag) in enumerate(bindings):
        struct.pack_into("<II", op, 0x10 + 8 * j, slot, tag)

    # 3. PROCEDURE
    proc = bytearray(0x38)
    struct.pack_into("<IIQQ8I", proc, 0x00, 1, 0, 0x18, 0x20, 1, 3, 0, 0, 1, 0, 0xFFFFFFFF, 4)

    # 4. TEXT_PROPERTIES
    tprop = bytearray(0x28)
    struct.pack_into("<IIIIQQQ", tprop, 0x00, 1, 0, 0, task_count, 0, 0, text_size)

    return {
        DRM_ANE_H16_SECTION_GENERIC: bytes(gen),
        DRM_ANE_H16_SECTION_OPERATION: bytes(op),
        DRM_ANE_H16_SECTION_PROCEDURE: bytes(proc),
        DRM_ANE_H16_SECTION_TEXT_PROPERTIES: bytes(tprop),
    }


def load_sections(
    hwx_path: Path,
    hwx_bytes: bytes,
    *,
    sections_dir: Path | None = None,
    vane_bin: Path | None = None,
) -> dict[int, bytes]:
    """Load the 4 LOAD sections from disk, via `vane sections`, or via built-in builder."""
    if sections_dir is not None:
        out: dict[int, bytes] = {}
        for kind, fname in SECTION_FILENAMES.items():
            out[kind] = (sections_dir / fname).read_bytes()
        return out

    if vane_bin is not None:
        with tempfile.TemporaryDirectory(prefix="vane-sec-") as tmpdir:
            subprocess.run(
                [str(vane_bin), "sections", "--hwx", str(hwx_path), "--out-dir", tmpdir],
                check=True,
                capture_output=True,
                text=True,
            )
            tmp = Path(tmpdir)
            return {kind: (tmp / fname).read_bytes() for kind, fname in SECTION_FILENAMES.items()}

    return build_sections_from_hwx(hwx_bytes)


# ---------------------------------------------------------------------------
# Device transport abstraction (allows offline mock testing)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceCaps:
    path: str
    size: int
    abi_version: int
    chip_family: int


@dataclass(frozen=True)
class LoadedProgram:
    surface_count: int
    surface_sizes: tuple[int, ...]
    surface_directions: tuple[int, ...]


class DeviceTransport:
    """Injectable low-level OS interface for Linux /dev/accel/accel* interaction."""

    def list_devices(self) -> list[str]:
        return sorted(glob.glob("/dev/accel/accel*"))

    def open(self, path: str) -> int:
        return os.open(path, os.O_RDWR | os.O_CLOEXEC)

    def close(self, fd: int) -> None:
        os.close(fd)

    def ioctl(self, fd: int, request: int, buf: bytearray) -> None:
        fcntl.ioctl(fd, request, buf, True)

    def mmap(self, fd: int, length: int, prot: int, flags: int, offset: int) -> Any:
        return mmap.mmap(fd, length, flags=flags, prot=prot, offset=offset)

    def page_size(self) -> int:
        return int(os.sysconf("SC_PAGE_SIZE"))


class NoMatchingDeviceError(RuntimeError):
    """Raised when no matching ABI v3 accel/ane node is available (skip condition)."""


def query_caps(transport: DeviceTransport, fd: int, path: str) -> DeviceCaps:
    buf = bytearray(STRUCT_GET_CAPS.size)
    transport.ioctl(fd, DRM_IOCTL_ANE_GET_CAPS, buf)
    _flags, size, abi_version, chip_family, _sec_sz, _bind_sz, _exec_sz, _pad = STRUCT_GET_CAPS.unpack(buf)
    return DeviceCaps(path=path, size=size, abi_version=abi_version, chip_family=chip_family)


def discover_device(
    transport: DeviceTransport,
    *,
    device_path: str | None = None,
    expected_chip_family: int | None = None,
) -> tuple[int, DeviceCaps]:
    """Open and return `(fd, DeviceCaps)` for a matching ABI v3 accel/ane device.

    Raises:
        NoMatchingDeviceError: If no matching device is present and no open error occurred.
        OSError: If opening a candidate device failed with a non-ENOENT error (e.g. EACCES).
    """
    candidates = [device_path] if device_path is not None else transport.list_devices()
    if not candidates:
        raise NoMatchingDeviceError("no /dev/accel/accel* devices found")

    for path in candidates:
        try:
            fd = transport.open(path)
        except OSError as exc:
            if exc.errno == errno.ENOENT and device_path is None:
                continue
            if exc.errno == errno.ENOENT and device_path is not None:
                raise NoMatchingDeviceError(f"device '{path}' does not exist") from exc
            raise

        try:
            caps = query_caps(transport, fd, path)
        except OSError:
            transport.close(fd)
            continue

        if caps.abi_version != DRM_ANE_ABI_V3:
            transport.close(fd)
            continue
        if expected_chip_family is not None and caps.chip_family != expected_chip_family:
            transport.close(fd)
            continue
        if caps.chip_family not in (DRM_ANE_CHIP_H16, DRM_ANE_CHIP_H17):
            transport.close(fd)
            continue
        return fd, caps

    raise NoMatchingDeviceError("no matching accel/ane ABI v3 device found")


def load_program(
    transport: DeviceTransport,
    fd: int,
    hwx_bytes: bytes,
    sections: dict[int, bytes],
) -> LoadedProgram:
    """Issue DRM_IOCTL_ANE_H16_LOAD and return kernel-reported surface descriptors."""
    hwx_buf = ctypes.create_string_buffer(hwx_bytes, len(hwx_bytes))
    sec_bufs: list[Any] = []
    sec_array = bytearray()

    for kind in (
        DRM_ANE_H16_SECTION_GENERIC,
        DRM_ANE_H16_SECTION_OPERATION,
        DRM_ANE_H16_SECTION_PROCEDURE,
        DRM_ANE_H16_SECTION_TEXT_PROPERTIES,
    ):
        if kind not in sections:
            raise ValueError(f"missing required LOAD section kind {kind}")
        sbytes = sections[kind]
        if not (DRM_ANE_H16_SECTION_MIN <= len(sbytes) <= DRM_ANE_H16_SECTION_MAX) or len(sbytes) % 4 != 0:
            raise ValueError(f"section kind {kind} has invalid size {len(sbytes)}")
        cbuf = ctypes.create_string_buffer(sbytes, len(sbytes))
        sec_bufs.append(cbuf)
        sec_array += STRUCT_H16_SECTION.pack(kind, len(sbytes), ctypes.addressof(cbuf))

    sec_array_buf = ctypes.create_string_buffer(bytes(sec_array), len(sec_array))
    load_req = bytearray(
        STRUCT_H16_LOAD.pack(
            ctypes.addressof(hwx_buf),
            len(hwx_bytes),
            ctypes.addressof(sec_array_buf),
            DRM_ANE_H16_SECTIONS,
            0,  # flags
            0,  # surface_count (out)
            0,  # pad
            0, 0, 0, 0,  # surface_size[4]
            0, 0, 0, 0,  # surface_direction[4]
        )
    )
    transport.ioctl(fd, DRM_IOCTL_ANE_H16_LOAD, load_req)
    # Keep buffers alive through the ioctl call
    _ = (hwx_buf, sec_bufs, sec_array_buf)

    fields = STRUCT_H16_LOAD.unpack(load_req)
    surface_count = int(fields[5])
    if not (2 <= surface_count <= DRM_ANE_H16_SURFACES):
        raise RuntimeError(f"kernel returned invalid surface_count {surface_count}")
    surface_sizes = tuple(int(x) for x in fields[7 : 7 + surface_count])
    surface_dirs = tuple(int(x) for x in fields[11 : 11 + surface_count])
    return LoadedProgram(
        surface_count=surface_count,
        surface_sizes=surface_sizes,
        surface_directions=surface_dirs,
    )


@dataclass
class RunReport:
    device: str
    abi_version: int
    chip_family: int
    mode: str
    runs: int
    surface_count: int
    surface_sizes: list[int]
    surface_directions: list[int]
    output_size: int
    output_sha256: str
    latencies_ms: list[float]
    repeatable: bool
    comparison: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "abi_version": self.abi_version,
            "chip_family": self.chip_family,
            "mode": self.mode,
            "runs": self.runs,
            "surface_count": self.surface_count,
            "surface_sizes": self.surface_sizes,
            "surface_directions": self.surface_directions,
            "output_size": self.output_size,
            "output_sha256": self.output_sha256,
            "latencies_ms": self.latencies_ms,
            "repeatable": self.repeatable,
            "comparison": self.comparison,
        }


def execute_program(
    transport: DeviceTransport,
    fd: int,
    caps: DeviceCaps,
    loaded: LoadedProgram,
    input_payloads: list[bytes],
    *,
    runs: int = 1,
    in_place: bool = False,
    expected_bytes: bytes | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    max_ulp: int | None = None,
) -> tuple[bytes, RunReport]:
    """Execute a loaded program `runs` times in copy or in-place mmap mode."""
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")

    input_indices = [i for i, d in enumerate(loaded.surface_directions) if d == 1]
    output_indices = [i for i, d in enumerate(loaded.surface_directions) if d == 2]
    if len(output_indices) != 1:
        raise RuntimeError(f"expected exactly 1 output surface, got {len(output_indices)}")
    out_idx = output_indices[0]
    out_size = loaded.surface_sizes[out_idx]

    if not input_payloads:
        resolved_inputs = [bytes(loaded.surface_sizes[i]) for i in input_indices]
    else:
        if len(input_payloads) != len(input_indices):
            raise ValueError(
                f"provided {len(input_payloads)} input buffers, but program expects {len(input_indices)}"
            )
        resolved_inputs = []
        for idx, payload in zip(input_indices, input_payloads):
            exp_sz = loaded.surface_sizes[idx]
            if len(payload) != exp_sz:
                raise ValueError(
                    f"input surface {idx} size mismatch: got {len(payload)} bytes, expected {exp_sz}"
                )
            resolved_inputs.append(payload)

    latencies_ms: list[float] = []
    first_output: bytes | None = None

    if in_place:
        psize = transport.page_size()
        if psize != PAGE_SIZE_16K:
            raise RuntimeError(
                f"--in-place requires 16 KiB host page size ({PAGE_SIZE_16K}), got {psize}"
            )
        mappings: list[Any] = []
        try:
            for i in range(loaded.surface_count):
                map_len = align_up(loaded.surface_sizes[i], PAGE_SIZE_16K)
                prot = (mmap.PROT_READ | mmap.PROT_WRITE) if loaded.surface_directions[i] == 1 else mmap.PROT_READ
                mm = transport.mmap(fd, map_len, prot, mmap.MAP_SHARED, drm_ane_h16_mmap_offset(i))
                mappings.append(mm)

            for run_idx in range(runs):
                for s_idx, payload in zip(input_indices, resolved_inputs):
                    mappings[s_idx][0 : len(payload)] = payload
                exec_req = bytearray(
                    STRUCT_H16_EXEC.pack(0, loaded.surface_count, DRM_ANE_H16_EXEC_IN_PLACE)
                )
                t0 = time.perf_counter()
                transport.ioctl(fd, DRM_IOCTL_ANE_H16_EXEC, exec_req)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

                out_bytes = bytes(mappings[out_idx][0:out_size])
                if first_output is None:
                    first_output = out_bytes
                elif out_bytes != first_output:
                    raise RuntimeError(
                        f"non-deterministic output on run {run_idx + 1}: "
                        f"sha256 {hashlib.sha256(out_bytes).hexdigest()} != "
                        f"run 1 sha256 {hashlib.sha256(first_output).hexdigest()}"
                    )
        finally:
            for mm in mappings:
                mm.close()
    else:
        # Allocate per-surface C buffers; poison output buffer with 0xff before every run.
        surf_bufs: list[Any] = []
        in_map = dict(zip(input_indices, resolved_inputs))
        for i in range(loaded.surface_count):
            sz = loaded.surface_sizes[i]
            if i in in_map:
                cbuf = ctypes.create_string_buffer(in_map[i], sz)
            else:
                cbuf = ctypes.create_string_buffer(sz)
            surf_bufs.append(cbuf)

        buf_array = bytearray()
        for i, cbuf in enumerate(surf_bufs):
            buf_array += STRUCT_H16_BUFFER.pack(ctypes.addressof(cbuf), loaded.surface_sizes[i])
        buf_array_c = ctypes.create_string_buffer(bytes(buf_array), len(buf_array))

        for run_idx in range(runs):
            ctypes.memset(surf_bufs[out_idx], 0xFF, out_size)
            exec_req = bytearray(
                STRUCT_H16_EXEC.pack(ctypes.addressof(buf_array_c), loaded.surface_count, 0)
            )
            t0 = time.perf_counter()
            transport.ioctl(fd, DRM_IOCTL_ANE_H16_EXEC, exec_req)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            out_bytes = surf_bufs[out_idx].raw[:out_size]
            if first_output is None:
                first_output = out_bytes
            elif out_bytes != first_output:
                raise RuntimeError(
                    f"non-deterministic output on run {run_idx + 1}: "
                    f"sha256 {hashlib.sha256(out_bytes).hexdigest()} != "
                    f"run 1 sha256 {hashlib.sha256(first_output).hexdigest()}"
                )
        _ = (surf_bufs, buf_array_c)

    assert first_output is not None
    digest = hashlib.sha256(first_output).hexdigest()
    comp_dict: dict[str, Any] | None = None
    if expected_bytes is not None:
        comp = reference.compare_fp16_bytes(
            first_output, expected_bytes, atol=atol, rtol=rtol, max_ulp=max_ulp
        )
        comp_dict = comp.to_dict()

    report = RunReport(
        device=caps.path,
        abi_version=caps.abi_version,
        chip_family=caps.chip_family,
        mode="in-place" if in_place else "copy",
        runs=runs,
        surface_count=loaded.surface_count,
        surface_sizes=list(loaded.surface_sizes),
        surface_directions=list(loaded.surface_directions),
        output_size=out_size,
        output_sha256=digest,
        latencies_ms=latencies_ms,
        repeatable=True,
        comparison=comp_dict,
    )
    return first_output, report


def run_cli(argv: list[str] | None = None, *, transport: DeviceTransport | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an HWX program on Linux accel/ane (ABI v3)."
    )
    parser.add_argument("--hwx", type=Path, required=True, help="Path to HWX program file.")
    parser.add_argument(
        "--sections",
        type=Path,
        default=None,
        help="Directory containing generic.bin, operation.bin, procedure.bin, text-properties.bin.",
    )
    parser.add_argument(
        "--vane-bin",
        type=Path,
        default=None,
        help="Path to vane CLI binary used to build LOAD sections when --sections is omitted.",
    )
    parser.add_argument("--device", type=str, default=None, help="Explicit /dev/accel/accel* device path.")
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        default=[],
        help="Input binary file (repeat in HWX input surface order; defaults to zeros).",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of consecutive runs for repeatability.")
    parser.add_argument("--in-place", action="store_true", help="Use mmap in-place execution mode.")
    parser.add_argument("--expected", type=Path, default=None, help="Expected output fp16 binary for reference comparison.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for reference comparison.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for reference comparison.")
    parser.add_argument("--max-ulp", type=int, default=None, help="Maximum allowed fp16 ULP error.")
    parser.add_argument("--out", type=Path, default=None, help="Write output surface bytes to this path.")
    parser.add_argument("--json-out", type=Path, default=None, help="Write structured JSON run report to this path.")

    args = parser.parse_args(argv)
    dev_transport = transport if transport is not None else DeviceTransport()

    hwx_bytes = args.hwx.read_bytes()
    if len(hwx_bytes) < 32:
        print(f"error: HWX file '{args.hwx}' is truncated", file=sys.stderr)
        return 1
    cpu_subtype = struct.unpack_from("<I", hwx_bytes, 8)[0]
    expected_chip = DRM_ANE_CHIP_H16 if cpu_subtype == 7 else (DRM_ANE_CHIP_H17 if cpu_subtype == 9 else None)

    sections = load_sections(args.hwx, hwx_bytes, sections_dir=args.sections, vane_bin=args.vane_bin)
    input_payloads = [p.read_bytes() for p in args.inputs]
    expected_bytes = args.expected.read_bytes() if args.expected is not None else None

    try:
        fd, caps = discover_device(
            dev_transport,
            device_path=args.device,
            expected_chip_family=expected_chip,
        )
    except NoMatchingDeviceError as exc:
        print(f"skip: {exc}", file=sys.stderr)
        return EXIT_SKIP
    except OSError as exc:
        print(f"error: failed to open device: {exc}", file=sys.stderr)
        return 1

    try:
        loaded = load_program(dev_transport, fd, hwx_bytes, sections)
        output_bytes, report = execute_program(
            dev_transport,
            fd,
            caps,
            loaded,
            input_payloads,
            runs=args.runs,
            in_place=args.in_place,
            expected_bytes=expected_bytes,
            atol=args.atol,
            rtol=args.rtol,
            max_ulp=args.max_ulp,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        dev_transport.close(fd)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(output_bytes)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    mean_ms = sum(report.latencies_ms) / len(report.latencies_ms)
    print(
        f"device={report.device} abi=v{report.abi_version} chip=H{report.chip_family} "
        f"mode={report.mode} runs={report.runs} mean_ms={mean_ms:.3f} sha256={report.output_sha256}"
    )

    if report.comparison is not None:
        comp = report.comparison
        print(
            f"reference: passed={comp['passed']} max_abs={comp['max_abs_err']:.6g} "
            f"max_rel={comp['max_rel_err']:.6g} rmse={comp['rmse']:.6g} "
            f"max_ulp={comp['max_ulp_err']} exact={comp['exact_fp16_count']}/{comp['total_elements']}"
        )
        if not comp["passed"]:
            return 1

    return 0


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
