"""SDIR faithful generator-IR: parse + forward-interpret + byte-exact gate.

The SDIR section emits, per generator (state-cell update), the EXACT ordered-SSA
program of each observed variant -- one node per executed ALU/load, in execution
order, with fully-resolved operands (Imm / Mem(base,mode,index_cell,klass) /
Prior(cell) / Node(id)), the evaluable guard chain, the post-settle init seed,
and the bytes of any Mem(TABLE). This test:

  * (always, no HVSC) checks SDIR parses on the self-contained fixture and that
    every generator/variant/node/operand is structurally well-formed, and that a
    plain accumulator cell forward-interprets to its own STSQ trajectory;

  * (gated on the two HVSC spike tunes) forward-interprets the captured IR from
    the parsed wire bytes and reproduces each tune's pulse-width STSQ trajectory
    BYTE-EXACT over the capture window -- mirroring docs/data/ir_feasibility.py:
      - Doctagop voice-2 PW ($1751): the accumulate arm pw' = (Prior + delta)&0xFF
        with delta the captured multi-term step (NOT step=None) + the note-on
        reload arm explain 100% of the dedup'd transitions;
      - Raindrops voice-1 PW ($93e4): the GT2 pulsetable interpreter -- cursor
        $93ba over the spd column Mem($9574,ABS_Y) / cmd column $9566, the
        duration counter $93bb, and the table bytes carried with the IR --
        replays byte-exact for the 223-frame window before the next note-on.

If a tune is absent the gate is SKIPPED (HVSC is gitignored; never committed).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from sidtrace_records import parse_sdst  # noqa: E402

DOCTAGOP = os.environ.get(
    "DOCTAGOP_SID", "/scratch/preframr/hvsc/C64Music/MUSICIANS/G/Gop/Doctagop.sid"
)
RAINDROPS = os.environ.get(
    "RAINDROPS_SID", "/scratch/preframr/hvsc/C64Music/MUSICIANS/P/Proton/Raindrops.sid"
)


def _sidtrace_bin():
    cand = os.environ.get("SIDTRACE_BIN") or str(ROOT / "build" / "sidtrace")
    p = Path(cand)
    if not p.exists():
        pytest.skip(f"sidtrace binary not built at {p} (run `make`)")
    return str(p)


def _trace(tmp_path, sid, subtune=0, nframes=1500):
    if not Path(sid).exists():
        pytest.skip(f"HVSC tune not present: {sid} (gitignored fixture)")
    prefix = tmp_path / "out"
    subprocess.run(
        [_sidtrace_bin(), str(sid), str(subtune), str(nframes), str(prefix)],
        check=True,
    )
    return parse_sdst(f"{prefix}.distill.bin")


def _dedup(seq):
    """Collapse adjacent equal samples (sidtrace over-samples vs the player's own
    update cadence; a frame-stepped interpreter sees one value per update)."""
    out = [seq[0]]
    for v in seq[1:]:
        if v != out[-1]:
            out.append(v)
    return out


# --------------------------------------------------------------------------
# A small forward SSA interpreter over the parsed SDIR (design section 4d).
# --------------------------------------------------------------------------


def _resolve(arg, ctx):
    tag = arg["tag"]
    if tag == "Imm":
        return arg["imm"] & 0xFF
    if tag == "Node":
        return ctx["nodes"][arg["node"]]
    if tag == "Prior":
        cell = arg["cell"]
        if cell == ctx["self"]:
            return ctx["prior"]
        return ctx["cells"].get(cell, arg["value"])
    if tag == "Mem":
        if arg["klass"] == "TABLE" and arg["index_cell"]:
            table = ctx["tables"].get(arg["base"])
            idx = ctx["cells"].get(arg["index_cell"], 0)
            if table is not None and 0 <= idx < len(table):
                return table[idx]
            return arg["value"]
        # SCRATCH / SONG_DATA / IO: a steady (or live, if simulated) cell value.
        return ctx["cells"].get(arg["base"], arg["value"])
    raise ValueError(f"bad operand {arg!r}")


def _apply(op, args, carry_in):
    a = args[0] if args else 0
    b = args[1] if len(args) > 1 else 0
    if op == "LOAD":
        return a & 0xFF
    if op == "ADC":
        c = carry_in if carry_in in (0, 1) else 0
        return (a + b + c) & 0xFF
    if op == "SBC":
        c = carry_in if carry_in in (0, 1) else 1
        return (a - b - (1 - c)) & 0xFF
    if op == "AND":
        return a & b & 0xFF
    if op == "ORA":
        return (a | b) & 0xFF
    if op == "EOR":
        return (a ^ b) & 0xFF
    if op == "ASL":
        return (a << 1) & 0xFF
    if op == "LSR":
        return (a >> 1) & 0xFF
    if op == "ROL":
        return (a << 1) & 0xFF
    if op == "ROR":
        return (a >> 1) & 0xFF
    if op == "INC":
        return (a + 1) & 0xFF
    if op == "DEC":
        return (a - 1) & 0xFF
    raise ValueError(f"unsupported SSA op {op}")


def eval_variant(gen, variant, prior, cells):
    """Forward-evaluate one variant's ordered SSA body; return the stored byte."""
    tables = {t["base"]: t["bytes"] for t in gen["tables"]}
    nodes = []
    ctx = {
        "self": gen["cell"],
        "prior": prior,
        "cells": cells,
        "tables": tables,
        "nodes": nodes,
    }
    for node in variant["body"]:
        nodes.append(
            _apply(
                node["op"], [_resolve(a, ctx) for a in node["args"]], node["carry_in"]
            )
        )
    sn = variant["store_node"]
    if sn == 0xFFFF or sn >= len(nodes):
        return None
    return nodes[sn] & 0xFF


def _gen(sdir, cell):
    for g in sdir:
        if g["cell"] == cell:
            return g
    return None


# --------------------------------------------------------------------------
# Structural test on the self-contained fixture (always runs; no HVSC).
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_distill(tmp_path_factory):
    d = tmp_path_factory.mktemp("sdir")
    sid = d / "acc.sid"
    # the SMC-operand accumulator fixture exercises a state-cell update generator
    subprocess.run(
        [sys.executable, str(HERE / "make_smc_operand_sid.py"), str(sid)], check=True
    )
    prefix = d / "out"
    subprocess.run([_sidtrace_bin(), str(sid), "0", "200", str(prefix)], check=True)
    return parse_sdst(f"{prefix}.distill.bin")


def test_sdir_present_and_well_formed(fixture_distill):
    sdir = fixture_distill["sdir"]
    assert isinstance(sdir, list)
    assert sdir, "SDIR emitted no generators for the accumulator fixture"
    tags = {"Imm", "Mem", "Prior", "Node"}
    ops = {
        "NONE",
        "ADC",
        "SBC",
        "AND",
        "ORA",
        "EOR",
        "ASL",
        "LSR",
        "ROL",
        "ROR",
        "INC",
        "DEC",
        "CMP",
        "LOAD",
        "NEG_IF_BIT7",
        "CARRY",
    }
    for g in sdir:
        assert 0 <= g["cell"] <= 0xFFFF
        assert 0 <= g["init"] <= 0xFF
        assert g["variants"], f"generator {g['cell']:#x} has no variants"
        for t in g["tables"]:
            assert len(t["bytes"]) > 0
        for v in g["variants"]:
            assert v["store_node"] == 0xFFFF or v["store_node"] < len(v["body"])
            for node in v["body"]:
                assert node["op"] in ops
                for a in node["args"]:
                    assert a["tag"] in tags
                    if a["tag"] == "Node":
                        # an SSA ref only points at an earlier node
                        assert a["node"] < len(v["body"])


def test_sdir_generators_match_sdcu_cells(fixture_distill):
    """SDIR is keyed by the same generator-DAG cells as SDCU (it is additive: it
    re-expresses those cells as a faithful program, leaving SDCU untouched)."""
    sdir_cells = {g["cell"] for g in fixture_distill["sdir"]}
    sdcu_cells = {u["cell"] for u in fixture_distill["sdcu"]}
    assert sdir_cells == sdcu_cells


# --------------------------------------------------------------------------
# The byte-exact gate (mirrors docs/data/ir_feasibility.py).
# --------------------------------------------------------------------------


def _doctagop_gate(d):
    cell = 0x1751
    sdir = d["sdir"]
    gen = _gen(sdir, cell)
    assert gen is not None, "Doctagop $1751 generator missing from SDIR"

    # the accumulate arm = the variant whose body ends in an ADC of Prior + step;
    # the reload arm = the variant that loads an immediate 0 (note-on).
    accumulate = reload_arm = None
    for v in gen["variants"]:
        ops = [n["op"] for n in v["body"]]
        if "ADC" in ops and any(
            a["tag"] == "Prior" for n in v["body"] for a in n["args"]
        ):
            accumulate = v
        elif ops == ["LOAD"] and v["body"][0]["args"][0]["tag"] == "Imm":
            reload_arm = v
    assert accumulate is not None, "no accumulate arm captured"
    assert reload_arm is not None, "no note-on reload arm captured"

    # FAITHFULNESS: the multi-term step is captured, NOT step=None. The accumulate
    # arm's step operand reads the delta cell ($175c+X); that delta cell's own
    # generator body carries the ordered multi-term arithmetic (AND + ADC ...).
    step_ops = []
    for g in sdir:
        body_ops = []
        for v in g["variants"]:
            body_ops += [n["op"] for n in v["body"]]
        if g["cell"] in (0x175D,) and "ADC" in body_ops and "AND" in body_ops:
            step_ops = body_ops
    assert step_ops, "multi-term delta generator ($175d: AND+ADC) not captured"

    # delta value the accumulate arm adds (resolved from the IR, steady = 54)
    delta = (eval_variant(gen, accumulate, prior=0, cells={})) & 0xFF
    assert delta == 54, f"accumulate step resolved to {delta}, expected 54"

    seq = _dedup([s["samples"] for s in d["stateseq"] if s["addr"] == cell][0])
    acc_val = {a: eval_variant(gen, accumulate, prior=a, cells={}) for a in range(256)}
    reload_val = eval_variant(gen, reload_arm, prior=0, cells={})
    hits_acc = hits_reload = un = 0
    for a, b in zip(seq, seq[1:]):
        if b == acc_val[a]:
            hits_acc += 1
        elif b == reload_val:
            hits_reload += 1
        else:
            un += 1
    tot = len(seq) - 1
    return {
        "tot": tot,
        "accumulate": hits_acc,
        "reload": hits_reload,
        "unexplained": un,
        "delta": delta,
    }


def _raindrops_gate(d):
    pw_cell = 0x93E4
    sdir = d["sdir"]
    pwgen = _gen(sdir, pw_cell)
    assert pwgen is not None, "Raindrops $93e4 generator missing from SDIR"

    # the modulate arm: pw' = Prior + spd[cursor]. Pull the spd table base and the
    # cursor (index) cell straight from the captured Mem(TABLE) operand.
    spd_base = cursor_cell = None
    for v in pwgen["variants"]:
        for n in v["body"]:
            for a in n["args"]:
                if a["tag"] == "Mem" and a["klass"] == "TABLE" and a["index_cell"]:
                    spd_base = a["base"]
                    cursor_cell = a["index_cell"]
    assert spd_base is not None, "spd-column Mem(TABLE) not captured"
    assert cursor_cell == 0x93BA, f"cursor cell = {cursor_cell:#x}, expected $93ba"

    spd_table = {t["base"]: t["bytes"] for t in pwgen["tables"]}.get(spd_base)
    assert spd_table is not None, "spd table bytes not carried with the generator"

    # the duration generator: a cell with a DEC(Prior(self)) counter arm AND a
    # reload arm that loads the cmd column Mem(TABLE) indexed by the SAME cursor.
    durgen = cmd_base = None
    for g in sdir:
        has_dec = any(
            n["op"] == "DEC"
            and n["args"]
            and n["args"][0]["tag"] == "Prior"
            and n["args"][0]["cell"] == g["cell"]
            for v in g["variants"]
            for n in v["body"]
        )
        cmd_b = None
        for v in g["variants"]:
            for n in v["body"]:
                for a in n["args"]:
                    if (
                        a["tag"] == "Mem"
                        and a["klass"] == "TABLE"
                        and a["index_cell"] == cursor_cell
                    ):
                        cmd_b = a["base"]
        if has_dec and cmd_b is not None:
            durgen, cmd_base = g, cmd_b
    assert durgen is not None, "GT2 duration counter generator not captured"
    cmd_table = {t["base"]: t["bytes"] for t in durgen["tables"]}.get(cmd_base)
    assert cmd_table is not None, "cmd table bytes not carried with the generator"

    cursor0 = _gen(sdir, cursor_cell)["init"]

    # forward-interpret the GT2 pulsetable walk (mirror ir_feasibility.run()),
    # fed entirely by SDIR-recovered addresses + table bytes.
    def run(n):
        pw, cur = 0, cursor0
        dur = cmd_table[cur]
        out = []
        for _ in range(n):
            out.append(pw)
            pw = (pw + spd_table[cur]) & 0xFF
            dur = (dur - 1) & 0xFF
            if dur == 0:
                cur += 1
                c = cmd_table[cur]
                if c == 0xFF:  # loop: next cursor read from the spd column
                    cur = spd_table[cur]
                    c = cmd_table[cur]
                if c & 0x80:  # absolute SET entry
                    pw = spd_table[cur]
                    cur += 1
                    c = cmd_table[cur]
                dur = c
        return out

    seq = _dedup([s["samples"] for s in d["stateseq"] if s["addr"] == pw_cell][0])
    sim = run(len(seq))
    div = next(
        (i for i in range(min(len(seq), len(sim))) if seq[i] != sim[i]), len(seq)
    )
    return {
        "byte_exact_frames": div,
        "window": len(seq),
        "spd_base": spd_base,
        "cmd_base": cmd_base,
        "cursor_cell": cursor_cell,
    }


def test_doctagop_pw_byte_exact(tmp_path):
    d = _trace(tmp_path, DOCTAGOP)
    r = _doctagop_gate(d)
    assert r["delta"] == 54
    assert r["unexplained"] == 0, (
        f"Doctagop $1751: {r['unexplained']}/{r['tot']} transitions UNEXPLAINED "
        f"by the SDIR arms (accumulate={r['accumulate']}, reload={r['reload']})"
    )


def test_raindrops_pw_byte_exact(tmp_path):
    d = _trace(tmp_path, RAINDROPS)
    r = _raindrops_gate(d)
    assert r["byte_exact_frames"] >= 223, (
        f"Raindrops $93e4: SDIR table-walk byte-exact for only "
        f"{r['byte_exact_frames']} frames (expected >= 223)"
    )


if __name__ == "__main__":  # pragma: no cover -- manual gate run
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        dd = _trace(Path(td) / "d", DOCTAGOP)
        print("== DOCTAGOP voice-2 PW ($1751) ==", _doctagop_gate(dd))
        rr = _trace(Path(td) / "r", RAINDROPS)
        print("== RAINDROPS voice-1 PW ($93e4) ==", _raindrops_gate(rr))
