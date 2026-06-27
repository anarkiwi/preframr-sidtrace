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

# SETL (settle-detection + classification): a single fixed record.
#  settle_found u8, structure_present u8, pad[2], settle_frame u32,
#  n_steady_idx u32, n_steady_ptr u32, settle_hold u32, settle_cap u32
_SETL_REC = struct.Struct("<BBxxIIIII")
# SSTR (steady-state structure): post-settle table-access provenance.
#  per indexed-read entry: pc u16, base u16, stride i32, idxMin u8, idxMax u8,
#                          flags u8, pad u8, count u32, feedsRegMask u32
_SSTR_IDX = struct.Struct("<HHiBBBxII")
#  per pointer-walk entry: zp u16, pad u16, advCount u32
_SSTR_PTR = struct.Struct("<HxxI")
# VEVT (per-voice post-settle note timeline): per voice header then events.
#  voice u8, pad u8, pad u16, nEvents u32; then nEvents * (frame u32, freq u16,
#  ctrl u8, gate u8).
_VEVT_HEAD = struct.Struct("<BBHI")
_VEVT_REC = struct.Struct("<IHBB")

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

# SDIR ordered-SSA op names (AluOp + IrExtraOp; mirror membus_trace.h).
IR_OP_NAME = dict(ALU_NAME)
IR_OP_NAME.update({13: "LOAD", 14: "NEG_IF_BIT7", 15: "CARRY"})

# SDIR operand tags / addressing modes / klasses (mirror membus_trace.h).
IO_IMM, IO_MEM, IO_PRIOR, IO_NODE = range(4)
IR_MODE_NAME = {
    0: "ABS",
    1: "ABS_X",
    2: "ABS_Y",
    3: "ZP",
    4: "ZP_X",
    5: "IND_Y",
    0xFF: "NONE",
}
IR_KLASS_NAME = {0: "SONG_DATA", 1: "TABLE", 2: "SCRATCH", 3: "IO"}

# Guard comparison kinds (mirror membus_trace.h CmpOp). The SEMANTIC predicate
# that holds when a guarded SDCU variant's store fires.
(
    CMP_NONE,
    CMP_EQ,
    CMP_NE,
    CMP_LT,
    CMP_GE,
    CMP_MI,
    CMP_PL,
    CMP_BIT_SET,
    CMP_BIT_CLR,
    CMP_VS,
    CMP_VC,
    CMP_UNRESOLVED,
) = range(12)
CMP_NAME = {
    0: "NONE",
    1: "EQ",
    2: "NE",
    3: "LT",
    4: "GE",
    5: "MI",
    6: "PL",
    7: "BIT_SET",
    8: "BIT_CLR",
    9: "VS",
    10: "VC",
    11: "UNRESOLVED",
}

# SDCU guard header (design 3): cmpOp u8, sense u8, immediate u8, flags u8
# (bit0 = unresolved), then nLeaves u16 + nLeaves leaf records.
_SDCU_GUARD = struct.Struct("<BBBB")


def load_sidwr(path):
    return np.fromfile(path, dtype=SIDWR_DT)


def _parse_sdir_operand(buf, off):
    """Parse one SDIR tagged operand. Returns ``(operand_dict, new_offset)``."""
    (tag,) = struct.unpack_from("<B", buf, off)
    off += 1
    if tag == IO_IMM:
        (imm,) = struct.unpack_from("<i", buf, off)
        off += 4
        return {"tag": "Imm", "imm": imm}, off
    if tag == IO_MEM:
        base, mode, index_cell, width, klass, value = struct.unpack_from(
            "<HBHBBB", buf, off
        )
        off += 8
        return (
            {
                "tag": "Mem",
                "base": base,
                "mode": IR_MODE_NAME.get(mode, "?"),
                "mode_raw": mode,
                "index_cell": index_cell,
                "width": width,
                "klass": IR_KLASS_NAME.get(klass, "?"),
                "klass_raw": klass,
                "value": value,
            },
            off,
        )
    if tag == IO_PRIOR:
        cell, value = struct.unpack_from("<HB", buf, off)
        off += 3
        return {"tag": "Prior", "cell": cell, "value": value}, off
    if tag == IO_NODE:
        (node,) = struct.unpack_from("<H", buf, off)
        off += 2
        return {"tag": "Node", "node": node}, off
    raise ValueError(f"bad SDIR operand tag {tag}")


def _parse_sdir_gen(buf, off):
    """Parse one SDIR generator (a state cell's faithful ordered-SSA program).

    Returns ``(gen_dict, new_offset)``. The generator carries its post-settle
    ``init`` seed, the ``tables`` any ``Mem(TABLE)`` operand reads (base/stride/
    bytes), and the observed ``variants``; each variant is an ordered SSA
    ``body`` (one node per executed ALU/load, execution order) plus the evaluable
    ``guards`` (reused from SDCU) and the ``store_node`` whose result is written.
    """
    cell, dest_width, init, nvar = struct.unpack_from("<HBBB", buf, off)
    off += 5
    (ntab,) = struct.unpack_from("<H", buf, off)
    off += 2
    tables = []
    for _ in range(ntab):
        base, stride, length = struct.unpack_from("<HBH", buf, off)
        off += 5
        data = list(buf[off : off + length])
        off += length
        tables.append({"base": base, "stride": stride, "bytes": data})
    variants = []
    for _ in range(nvar):
        store_pc, store_node = struct.unpack_from("<HH", buf, off)
        off += 4
        (nguards,) = struct.unpack_from("<B", buf, off)
        off += 1
        guards = []
        for _ in range(nguards):
            cmp_op, sense, imm, gflags = _SDCU_GUARD.unpack_from(buf, off)
            off += _SDCU_GUARD.size
            (gnl,) = struct.unpack_from("<H", buf, off)
            off += 2
            gleaves = []
            for _ in range(gnl):
                kind, addr, value = _SIDDF_LEAF.unpack_from(buf, off)
                off += _SIDDF_LEAF.size
                gleaves.append((kind, addr, value))
            guards.append(
                {
                    "cmp": cmp_op,
                    "cmp_name": CMP_NAME.get(cmp_op, "?"),
                    "sense": sense,
                    "imm": imm,
                    "unresolved": bool(gflags & 1),
                    "leaves": gleaves,
                }
            )
        (nnodes,) = struct.unpack_from("<H", buf, off)
        off += 2
        body = []
        for _ in range(nnodes):
            op, carry_in, nargs = struct.unpack_from("<BBB", buf, off)
            off += 3
            args = []
            for _ in range(nargs):
                operand, off = _parse_sdir_operand(buf, off)
                args.append(operand)
            body.append(
                {
                    "op": IR_OP_NAME.get(op, "?"),
                    "op_raw": op,
                    "carry_in": carry_in,
                    "args": args,
                }
            )
        variants.append(
            {
                "store_pc": store_pc,
                "store_node": store_node,
                "guards": guards,
                "body": body,
            }
        )
    return (
        {
            "cell": cell,
            "dest_width": dest_width,
            "init": init,
            "tables": tables,
            "variants": variants,
        },
        off,
    )


def _parse_siddf_entry(buf, off):
    """Parse one SDDF-shaped data-flow entry (also the SDCU variant value body).

    Returns ``(entry_dict, new_offset)``. The leading head's ``pc`` slot is the
    code site for SDDF and the variant's store PC for SDCU."""
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
    # mid-call value sequence (SDCU's genuine generator state stream; empty for
    # SDDF). The Berlekamp-Massey input for the LFSR-vs-not verdict.
    (nval,) = struct.unpack_from("<H", buf, off)
    off += 2
    valseq = list(struct.unpack_from(f"<{nval}B", buf, off)) if nval else []
    off += nval
    entry = {
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
    return entry, off


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
    sdir = []
    idx_supp = []
    ptr_walks = []
    relo_copies = []
    sid_accum = []
    digi = None
    tempo_cands = []
    iwlk_walks = []
    settle = None
    steady_idx = []
    steady_ptr = []
    voice_events = []
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
        elif tag == b"SDDF":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                entry, off = _parse_siddf_entry(buf, off)
                siddf.append(entry)
        elif tag == b"SDCU":
            # version 2: per cell -> cell u16, nVariants u16; each variant is an
            # SDDF-shaped value entry (pc slot = store_pc) followed by the guard
            # chain. (A version-1 reader saw a single SDDF-shaped summary per cell.)
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                cell, nvar = struct.unpack_from("<HH", buf, off)
                off += 4
                variants = []
                for _ in range(nvar):
                    entry, off = _parse_siddf_entry(buf, off)
                    entry["store_pc"] = entry.pop("pc")
                    (nguards,) = struct.unpack_from("<B", buf, off)
                    off += 1
                    guards = []
                    for _ in range(nguards):
                        cmp_op, sense, imm, gflags = _SDCU_GUARD.unpack_from(buf, off)
                        off += _SDCU_GUARD.size
                        (gnl,) = struct.unpack_from("<H", buf, off)
                        off += 2
                        gleaves = []
                        for _ in range(gnl):
                            kind, addr, value = _SIDDF_LEAF.unpack_from(buf, off)
                            off += _SIDDF_LEAF.size
                            gleaves.append((kind, addr, value))
                        guards.append(
                            {
                                "cmp": cmp_op,
                                "cmp_name": CMP_NAME.get(cmp_op, "?"),
                                "sense": sense,
                                "imm": imm,
                                "unresolved": bool(gflags & 1),
                                "leaves": gleaves,
                            }
                        )
                    entry["guards"] = guards
                    variants.append(entry)
                sdcu.append({"cell": cell, "variants": variants})
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
        elif tag == b"SETL":
            found, present, frame, nidx, nptr, hold, cap = _SETL_REC.unpack_from(
                buf, off
            )
            off += _SETL_REC.size
            settle = {
                "settle_found": bool(found),
                "structure_present": bool(present),
                "settle_frame": frame,
                "n_steady_idx": nidx,
                "n_steady_ptr": nptr,
                "settle_hold": hold,
                "settle_cap": cap,
            }
        elif tag == b"SSTR":
            (nidx,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nidx):
                pc, base, stride, imin, imax, flags, count, mask = (
                    _SSTR_IDX.unpack_from(buf, off)
                )
                off += _SSTR_IDX.size
                steady_idx.append(
                    {
                        "pc": pc,
                        "base": base,
                        "stride": stride,
                        "idx_min": imin,
                        "idx_max": imax,
                        "targets_data": bool(flags & 1),
                        "in_image": bool(flags & 2),
                        "count": count,
                        "feeds_reg_mask": mask,
                    }
                )
            (nptr,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nptr):
                zp, adv = _SSTR_PTR.unpack_from(buf, off)
                off += _SSTR_PTR.size
                steady_ptr.append({"zp": zp, "adv_count": adv})
        elif tag == b"VEVT":
            (nvoice,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nvoice):
                voice, _p8, _p16, nev = _VEVT_HEAD.unpack_from(buf, off)
                off += _VEVT_HEAD.size
                events = []
                for _ in range(nev):
                    frame, freq, ctrl, gate = _VEVT_REC.unpack_from(buf, off)
                    off += _VEVT_REC.size
                    events.append(
                        {"frame": frame, "freq": freq, "ctrl": ctrl, "gate": gate}
                    )
                voice_events.append({"voice": voice, "events": events})
        elif tag == b"SDIR":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                gen, off = _parse_sdir_gen(buf, off)
                sdir.append(gen)
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
        "sdir": sdir,
        "idx_supp": idx_supp,
        "ptr_walks": ptr_walks,
        "relo_copies": relo_copies,
        "sid_accum": sid_accum,
        "digi": digi,
        "tempo_cands": tempo_cands,
        "iwlk_walks": iwlk_walks,
        "settle": settle,
        "steady_idx": steady_idx,
        "steady_ptr": steady_ptr,
        "voice_events": voice_events,
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
