#!/usr/bin/env python3
"""Generate a tiny self-contained PSID exercising a BRANCH-CONDITIONED state-cell
update -- the triangle-sweep idiom the SDCU branch-structure extension recovers
(design 3).

A pulse-width shadow byte is swept up and down: a free-running frame counter
``cnt`` (INC'd every play-call, a clean +1 mod-256 accumulator) selects, via its
high bit, whether the shadow is incremented (``ADC``) or decremented (``SBC``)
this frame. The shadow drives voice-0 pulse-width ($D402).

To make the shadow a STATE cell the SID-write slice surfaces (so it seeds the
generator DAG), the shadow IS the self-modified immediate operand of the
``LDA #imm ; STA $D402`` SID write -- the DefMon SMC-operand class, reused here.
The update reads/writes that operand byte by absolute address.

  play:
    INC cnt                 ; free-running +1 counter (the toggle generator)
    LDA cnt
    BPL up                  ; cnt < 128 (N=0) -> up; cnt >= 128 -> down
    ; down (cnt >= 128):
    LDA PWOP : SEC : SBC #3 : STA PWOP      ; variant B: shadow -= 3   (guard: cnt MI)
    JMP emit
  up:
    LDA PWOP : CLC : ADC #3 : STA PWOP      ; variant A: shadow += 3   (guard: cnt PL)
  emit:
    LDA #<shadow>  (operand @ PWOP)         ; SMC operand = the shadow byte
    STA $D402                               ; voice-0 pulse-width lo
    LDA #$08 : STA $D403
    LDA #$41 : STA $D404                    ; pulse + gate
    LDA #$0F : STA $D418
    RTS

So PWOP has TWO store sites (the ADC path and the SBC path) -> two SDCU
variants with distinct ops, each guarded on the common toggle cell ``cnt``,
and ``cnt`` is itself an emitted straight-line +1 counter.
"""

import struct
import sys

LOAD_ADDR = 0x1000
INIT_ADDR = 0x1000
CNT_ADDR = 0x1080  # free-running frame counter (past the code)


def build_code():
    clo, chi = CNT_ADDR & 0xFF, (CNT_ADDR >> 8) & 0xFF

    init = [
        0xA9,
        0x00,  # LDA #$00
        0x8D,
        0x18,
        0xD4,  # STA $D418
        0x8D,
        clo,
        chi,  # STA cnt = 0
        0x60,  # RTS
    ]
    play_addr = LOAD_ADDR + len(init)

    # Resolve the labels/operands from the fixed play layout (offsets below).
    up_off = 20
    emit_off = 29
    pwop_off = emit_off + 1  # operand byte of `LDA #imm` at emit
    pwop = play_addr + pwop_off
    pl, ph = pwop & 0xFF, (pwop >> 8) & 0xFF
    bpl_off = 6
    bpl_rel = (play_addr + up_off) - (play_addr + bpl_off + 2)
    assert 0 <= bpl_rel < 128, bpl_rel
    emit_addr = play_addr + emit_off
    el, eh = emit_addr & 0xFF, (emit_addr >> 8) & 0xFF

    play = [
        0xEE,
        clo,
        chi,  # 0  INC cnt
        0xAD,
        clo,
        chi,  # 3  LDA cnt
        0x10,
        bpl_rel,  # 6  BPL up
        0xAD,
        pl,
        ph,  # 8  LDA PWOP        (down block)
        0x38,  # 11 SEC
        0xE9,
        0x03,  # 12 SBC #$03
        0x8D,
        pl,
        ph,  # 14 STA PWOP        (variant B)
        0x4C,
        el,
        eh,  # 17 JMP emit
        0xAD,
        pl,
        ph,  # 20 LDA PWOP        (up: block)
        0x18,  # 23 CLC
        0x69,
        0x03,  # 24 ADC #$03
        0x8D,
        pl,
        ph,  # 26 STA PWOP        (variant A)
        0xA9,
        0x00,  # 29 LDA #imm        (emit:; operand @ pwop)
        0x8D,
        0x02,
        0xD4,  # 31 STA $D402
        0xA9,
        0x08,  # 34 LDA #$08
        0x8D,
        0x03,
        0xD4,  # 36 STA $D403
        0xA9,
        0x41,  # 39 LDA #$41
        0x8D,
        0x04,
        0xD4,  # 41 STA $D404
        0xA9,
        0x0F,  # 44 LDA #$0F
        0x8D,
        0x18,
        0xD4,  # 46 STA $D418
        0x60,  # 49 RTS
    ]
    assert len(play) == 50, len(play)

    code = init + play
    pad = (CNT_ADDR - LOAD_ADDR) - len(code)
    assert pad >= 0, f"code overruns counter cell by {-pad} bytes"
    code += [0x00] * pad + [0x00]
    return play_addr, bytes(code)


def build_psid() -> bytes:
    play_addr, code = build_code()
    c64_data = struct.pack("<H", LOAD_ADDR) + code
    header = b"PSID" + struct.pack(
        ">HHHHHHHL", 2, 0x7C, 0x0000, INIT_ADDR, play_addr, 1, 1, 0
    )
    header += b"sidtrace branch-update fixture".ljust(32, b"\x00")[:32]
    header += b"preframr-sidtrace".ljust(32, b"\x00")[:32]
    header += b"2026 public domain".ljust(32, b"\x00")[:32]
    header += struct.pack(">HBBBB", 0x0000, 0x00, 0x00, 0x00, 0x00)
    assert len(header) == 0x7C
    return header + c64_data


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "branch_update_fixture.sid"
    with open(out, "wb") as f:
        f.write(build_psid())
    sys.stderr.write(f"wrote {out}\n")


if __name__ == "__main__":
    main()
