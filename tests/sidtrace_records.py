"""Binary record layout for sidtrace's output artifacts.

sidtrace emits two files per tune:

* ``<prefix>.sidwr.bin`` -- the timestamped SID register-write stream (the
  render/dump gate). Packed (no struct padding), little-endian, in field order:
  ``int64 cycle, uint16 addr, uint8 reg, uint8 val`` -> 12 bytes/record.

* ``<prefix>.distill.bin`` -- the compact SDST artifact: the in-emulator
  DISTILLATION of the run (access-type map, post-init song-data snapshot,
  PC-tagged SID-write summary, indexed-read VSA summary). A few KB per tune. This
  REPLACES the retired multi-GB cycle-by-cycle bus trace; the analysis is done in
  C++ during emulation, not in Python over a raw stream. The exact layout is in
  ``src/sidtrace.cpp`` (and ``parse_sdst`` below).
"""

import struct

import numpy as np

# <prefix>.sidwr.bin : every SID register write ($D400-$D7FF, rw=1)
#   int64 cycle, uint16 addr, uint8 reg, uint8 val   -> 12 bytes/record
SIDWR_DT = np.dtype([("cycle", "<i8"), ("addr", "<u2"), ("reg", "u1"), ("val", "u1")])
assert SIDWR_DT.itemsize == 12

# Per-address access-type bits (mirror src/c64/membus_trace.h AccBits).
ACC_EXEC_INIT = 1 << 0
ACC_READ_INIT = 1 << 1
ACC_WRITE_INIT = 1 << 2
ACC_EXEC_PLAY = 1 << 3
ACC_READ_PLAY = 1 << 4
ACC_WRITE_PLAY = 1 << 5

_SIDW_ENTRY = struct.Struct("<HBBIB3x")  # pc, reg, pad, count, lastVal, pad[3]
_IDXR_ENTRY = struct.Struct("<HHiBBHI")  # pc, base, stride, idxMin, idxMax, pad, count


def load_sidwr(path):
    return np.fromfile(path, dtype=SIDWR_DT)


def parse_sdst(path):
    """Parse a ``<prefix>.distill.bin`` SDST artifact into a dict.

    Returns the header fields plus ``acc`` (uint8[65536] access-type map), ``ram``
    (uint8[65536] post-init song-data snapshot, 0 elsewhere), ``sid_writes`` (list
    of ``(pc, reg, count, lastVal)``), and ``idx_reads`` (list of
    ``(pc, base, stride, idxMin, idxMax, count)``)."""
    with open(path, "rb") as handle:
        buf = handle.read()
    if buf[:4] != b"SDST":
        raise ValueError(f"not an SDST artifact: {buf[:4]!r}")
    off = 4
    version, _r = struct.unpack_from("<HH", buf, off)
    off += 4
    init, play, load, subtune, nframes, _r2 = struct.unpack_from("<6H", buf, off)
    off += 12
    (cpf,) = struct.unpack_from("<I", buf, off)
    off += 4
    (t0,) = struct.unpack_from("<q", buf, off)
    off += 8
    (load_len,) = struct.unpack_from("<I", buf, off)
    off += 4

    acc = np.zeros(65536, dtype=np.uint8)
    ram = np.zeros(65536, dtype=np.uint8)
    sid_writes = []
    idx_reads = []
    while off < len(buf):
        tag = buf[off : off + 4]
        off += 4
        if tag == b"END\x00":
            break
        if tag in (b"ACMP", b"SNAP"):
            (nbytes,) = struct.unpack_from("<I", buf, off)
            off += 4
            body = buf[off : off + nbytes]
            off += nbytes
            if tag == b"ACMP":
                pos = 0
                addr = 0
                while pos + 3 <= len(body) and addr < 65536:
                    run = body[pos] | (body[pos + 1] << 8)
                    acc[addr : addr + run] = body[pos + 2]
                    addr += run
                    pos += 3
            else:
                pos = 0
                while pos + 4 <= len(body):
                    a = body[pos] | (body[pos + 1] << 8)
                    ln = body[pos + 2] | (body[pos + 3] << 8)
                    pos += 4
                    ram[a : a + ln] = np.frombuffer(
                        body[pos : pos + ln], dtype=np.uint8
                    )
                    pos += ln
        elif tag == b"SIDW":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                pc, reg, _p, count, last_val = _SIDW_ENTRY.unpack_from(buf, off)
                off += _SIDW_ENTRY.size
                sid_writes.append((pc, reg, count, last_val))
        elif tag == b"IDXR":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                pc, base, stride, imin, imax, _p, count = _IDXR_ENTRY.unpack_from(
                    buf, off
                )
                off += _IDXR_ENTRY.size
                idx_reads.append((pc, base, stride, imin, imax, count))
        else:
            raise ValueError(f"unknown SDST section {tag!r}")

    return {
        "version": version,
        "init": init,
        "play": play,
        "load": load,
        "subtune": subtune,
        "nframes": nframes,
        "cycles_per_frame": cpf,
        "t0_cycle": t0,
        "load_len": load_len,
        "acc": acc,
        "ram": ram,
        "sid_writes": sid_writes,
        "idx_reads": idx_reads,
    }


def parse_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k] = v
    return meta
