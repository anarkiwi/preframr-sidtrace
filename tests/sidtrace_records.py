"""Binary record layout for sidtrace's output artifacts.

sidtrace emits two files per tune:

* ``<prefix>.sidwr.bin`` -- the timestamped SID register-write stream (the
  render/dump gate). Packed (no struct padding), little-endian, in field order:
  ``int64 cycle, uint16 addr, uint8 reg, uint8 val`` -> 12 bytes/record.

* ``<prefix>.distill.bin`` -- the compact SDST artifact: the in-emulator
  DISTILLATION of the run (access-type map, post-init song-data snapshot,
  PC-tagged SID-write summary, indexed-read VSA summary, and the per-SID-write
  DATA-FLOW summary SIDDF). A few KB per tune. This REPLACES the retired multi-GB
  cycle-by-cycle bus trace; the analysis is done in C++ during emulation, not in
  Python over a raw stream. The exact layout is in ``src/sidtrace.cpp`` (and
  ``parse_sdst`` below).

  SIDDF is the Stage-1 addition (design 3.1): for each STA $D4xx code site
  (keyed by (PC,reg), the same key space as SIDW, so O(code sites) not O(frames))
  it carries a bounded in-emulator backward dynamic slice of the written value --
  the slice PCs, classified leaves (immediate / ram_read / state_cell[=SMC] /
  exogenous / out_of_window), the value-chain ALU op sequence, a strided-interval
  VSA for any indexed read feeding the write, and the min/max/first value. It
  turns "PC wrote register r" into "here is HOW r was computed" and stays flat as
  the capture window grows.

  STSQ (design 3.2) is the inter-frame sample sequence of each SIDDF-flagged state
  cell (Stage-C recurrence/Berlekamp-Massey input). SDCU (design 2.2/2.3, Stage 3)
  is the per-state-cell UPDATE DAG: for a fast mid-call SMC accumulator the
  SID-write slice bottoms out at the shadow cell as a leaf, so SDCU carries the
  backward slice of the STORE or in-place RMW that recomputed the cell -- the
  update cell'=f(cell,...) the host generalizes into U. Keyed by cell address;
  O(state cells), flat vs frames. (Both parse the same SiddfSite shape; for SDCU
  the `pc` field is the state-cell address.)
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

# SIDDF (per-write data-flow summary, design 3.1). Variable-length entries.
_SIDDF_HEAD = struct.Struct("<HBBIBBBxHiBB")
#  pc u16, reg u8, flags u8, count u32, valLo u8, valHi u8, valFirst u8, pad u8,
#  strideBase u16, strideStep i32, strideIdxMin u8, strideIdxMax u8
_SIDDF_LEAF = struct.Struct("<BxHBx")  # kind u8, pad, addr u16, value u8, pad

# STSQ (inter-frame state-cell sample sequence, design 3.2). Variable-length.
#  addr u16, flags u8, pad, totalFrames u32, firstSeenFrame u32, nSamples u16
_STSQ_HEAD = struct.Struct("<HBxIIH")

# Recovery-offload sections (the structural-capture additions; appended after
# SDCU so the sections above stay byte-identical).
_IDXS_ENTRY = struct.Struct("<HBBiHIBBHH")
_PWLK_HEAD = struct.Struct("<HBBBBIH")
_RELO_ENTRY = struct.Struct("<HHHHiiBBHI")
_SDAC_HEAD = struct.Struct("<HBBH")
_SDAC_ADDEND = struct.Struct("<BH")
_DIGI_REC = struct.Struct("<IIBBBBII")
_TMPO_ENTRY = struct.Struct("<HBB")
# IWLK (per-(pc,voice) instrument-table freq walk-index): pc u16, voice u8, pad,
# nFrames u32, then nFrames * (index u8). The per-frame index into the RAM
# instrument freq-table the freq-feeding IDXR PC walked (the freq-mod generator).
_IWLK_HEAD = struct.Struct("<HBxI")

# Leaf kinds (mirror membus_trace.h LeafKind).
LK_IMMEDIATE = 0
LK_RAM_READ = 1
LK_STATE_CELL = 2
LK_EXOGENOUS = 3
LK_OUT_OF_WINDOW = 4
LEAF_KIND_NAME = {
    LK_IMMEDIATE: "immediate",
    LK_RAM_READ: "ram_read",
    LK_STATE_CELL: "state_cell",
    LK_EXOGENOUS: "exogenous",
    LK_OUT_OF_WINDOW: "out_of_window",
}

# ALU ops (mirror membus_trace.h AluOp).
(
    ALU_NONE,
    ALU_ADC,
    ALU_SBC,
    ALU_AND,
    ALU_ORA,
    ALU_EOR,
    ALU_ASL,
    ALU_LSR,
    ALU_ROL,
    ALU_ROR,
    ALU_INC,
    ALU_DEC,
    ALU_CMP,
) = range(13)
ALU_NAME = {
    0: "NONE",
    1: "ADC",
    2: "SBC",
    3: "AND",
    4: "ORA",
    5: "EOR",
    6: "ASL",
    7: "LSR",
    8: "ROL",
    9: "ROR",
    10: "INC",
    11: "DEC",
    12: "CMP",
}


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
    siddf = []
    stateseq = []
    sdcu = []
    idx_supp = []
    ptr_walks = []
    relo_copies = []
    sid_accum = []
    digi = None
    tempo_cands = []
    iwlk_walks = []
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
        elif tag in (b"SDDF", b"SDCU"):
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            dest = siddf if tag == b"SDDF" else sdcu
            for _ in range(nent):
                (
                    pc,
                    reg,
                    flags,
                    count,
                    vlo,
                    vhi,
                    vfirst,
                    sbase,
                    sstep,
                    simin,
                    simax,
                ) = _SIDDF_HEAD.unpack_from(buf, off)
                off += _SIDDF_HEAD.size
                (npcs,) = struct.unpack_from("<H", buf, off)
                off += 2
                pcs = list(struct.unpack_from(f"<{npcs}H", buf, off))
                off += 2 * npcs
                (nleaves,) = struct.unpack_from("<H", buf, off)
                off += 2
                leaves = []
                for _ in range(nleaves):
                    kind, addr, value = _SIDDF_LEAF.unpack_from(buf, off)
                    off += _SIDDF_LEAF.size
                    leaves.append((kind, addr, value))
                (nops,) = struct.unpack_from("<H", buf, off)
                off += 2
                ops = list(struct.unpack_from(f"<{nops}B", buf, off)) if nops else []
                off += nops
                # mid-call value sequence (SDCU's genuine generator state stream;
                # empty for SDDF). The BM input for the LFSR-vs-not verdict.
                (nval,) = struct.unpack_from("<H", buf, off)
                off += 2
                valseq = (list(struct.unpack_from(f"<{nval}B", buf, off))
                          if nval else [])
                off += nval
                dest.append(
                    {
                        "pc": pc,
                        "reg": reg,
                        "flags": flags,
                        "count": count,
                        "val_lo": vlo,
                        "val_hi": vhi,
                        "val_first": vfirst,
                        "has_stride": bool(flags & 1),
                        "any_out_of_window": bool(flags & 2),
                        "stride_base": sbase,
                        "stride_step": sstep,
                        "stride_idx_min": simin,
                        "stride_idx_max": simax,
                        "slice_pcs": pcs,
                        "leaves": leaves,
                        "op_seq": ops,
                        "val_seq": valseq,
                    }
                )
        elif tag == b"STSQ":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                addr, flags, total_frames, first_seen, nsamp = _STSQ_HEAD.unpack_from(
                    buf, off
                )
                off += _STSQ_HEAD.size
                samples = list(struct.unpack_from(f"<{nsamp}B", buf, off))
                off += nsamp
                stateseq.append(
                    {
                        "addr": addr,
                        "holds_to_end": bool(flags & 1),
                        "wide": bool(flags & 2),
                        "total_frames": total_frames,
                        "first_seen_frame": first_seen,
                        "n_samples": nsamp,
                        "samples": samples,
                    }
                )
        elif tag == b"IDXS":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                pc, flags, nsamp, scale, basefit, mask, s0i, s1i, s0a, s1a = (
                    _IDXS_ENTRY.unpack_from(buf, off)
                )
                off += _IDXS_ENTRY.size
                idx_supp.append(
                    {
                        "pc": pc,
                        "scale_set": bool(flags & 1),
                        "targets_in_image": bool(flags & 2),
                        "targets_read_as_data": bool(flags & 4),
                        "n_samp": nsamp,
                        "scale": scale,
                        "base_fit": basefit,
                        "feeds_reg_mask": mask,
                        "samp_idx": (s0i, s1i),
                        "samp_addr": (s0a, s1a),
                    }
                )
        elif tag == b"PWLK":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                zp, flags, ymin, ymax, _p, count, nadv = _PWLK_HEAD.unpack_from(
                    buf, off
                )
                off += _PWLK_HEAD.size
                ptrs = list(struct.unpack_from(f"<{nadv}H", buf, off))
                off += 2 * nadv
                frames = list(struct.unpack_from(f"<{nadv}I", buf, off))
                off += 4 * nadv
                # advY (u8) + advPc (u16): the authored orderlist POSITION and the
                # consuming voice's read PC at each advance (appended after advFrame).
                adv_y = list(struct.unpack_from(f"<{nadv}B", buf, off))
                off += nadv
                adv_pc = list(struct.unpack_from(f"<{nadv}H", buf, off))
                off += 2 * nadv
                ptr_walks.append(
                    {
                        "zp": zp,
                        "is_load": bool(flags & 1),
                        "is_store": bool(flags & 2),
                        "y_min": ymin,
                        "y_max": ymax,
                        "count": count,
                        "ptr_vals": ptrs,
                        "adv_frames": frames,
                        "adv_y": adv_y,
                        "adv_pc": adv_pc,
                    }
                )
        elif tag == b"RELO":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                spc, srpc, sb, db, ss, ds, imin, imax, _p, count = (
                    _RELO_ENTRY.unpack_from(buf, off)
                )
                off += _RELO_ENTRY.size
                relo_copies.append(
                    {
                        "store_pc": spc,
                        "src_read_pc": srpc,
                        "src_base": sb,
                        "dst_base": db,
                        "src_stride": ss,
                        "dst_stride": ds,
                        "idx_min": imin,
                        "idx_max": imax,
                        "count": count,
                        "delta": (db - sb) & 0xFFFF,
                    }
                )
        elif tag == b"SDAC":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                pc, reg, _p, nadd = _SDAC_HEAD.unpack_from(buf, off)
                off += _SDAC_HEAD.size
                addends = []
                for _ in range(nadd):
                    op, cell = _SDAC_ADDEND.unpack_from(buf, off)
                    off += _SDAC_ADDEND.size
                    addends.append((op, cell))
                sid_accum.append({"pc": pc, "reg": reg, "addends": addends})
        elif tag == b"DIGI":
            mean1k, maxd418, notetbl, _p0, _p1, _p2, fc, nwr = _DIGI_REC.unpack_from(
                buf, off
            )
            off += _DIGI_REC.size
            digi = {
                "writes_per_frame_mean": mean1k / 1000.0,
                "max_subframe_d418": maxd418,
                "note_table_idxr_present": bool(notetbl),
                "n_frames": fc,
                "n_sid_writes": nwr,
            }
        elif tag == b"TMPO":
            (ncand,) = struct.unpack_from("<H", buf, off)
            off += 2
            for _ in range(ncand):
                cell, reload, _p = _TMPO_ENTRY.unpack_from(buf, off)
                off += _TMPO_ENTRY.size
                tempo_cands.append({"cell": cell, "reload": reload})
        elif tag == b"IWLK":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                pc, voice, nfr = _IWLK_HEAD.unpack_from(buf, off)
                off += _IWLK_HEAD.size
                index = np.frombuffer(buf[off : off + nfr], dtype=np.uint8).copy()
                off += nfr
                iwlk_walks.append({"pc": pc, "voice": voice, "index": index})
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
        "siddf": siddf,
        "stateseq": stateseq,
        "sdcu": sdcu,
        "idx_supp": idx_supp,
        "ptr_walks": ptr_walks,
        "relo_copies": relo_copies,
        "sid_accum": sid_accum,
        "digi": digi,
        "tempo_cands": tempo_cands,
        "iwlk_walks": iwlk_walks,
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
