#!/usr/bin/env python3
"""Generate a tiny self-contained PSID exercising the NESTED SEQUENCER CURSOR as a
looping table-walk (design 2.6 nested cursor / 5.4).

This is the closable typed level of the orderlist->pattern->note cursor: an axis
whose output is a piecewise-constant walk ``O = V[cursor]`` over a small value
table V, where the cursor advances +1 (wrapping mod len(V)) every D play-calls
(a uniform dwell clock) and the table loops many times. The host recovers it as a
TYPED recurrence ``cursor(f) = (f // D) mod P`` with V lifted by identity, and
admits it only when V is traversed >= 2 full cycles AND it renders the FULL stream
residual-zero (no branch re-execution; design 2.8 admission + 5).

Player shape (self-modifying-operand axis, the DefMon class, driving voice-0
freq-lo from a looping arpeggio table):

  play:
    INC subtick                     ; sub-dwell counter 0..D-1
    LDA subtick
    CMP #D
    BNE skip
      LDA #0  : STA subtick         ; reset sub-dwell
      INC cursor                    ; advance the cursor (mod P done by table wrap)
      LDA cursor : AND #(P-1) : STA cursor
    skip:
    LDX cursor
    LDA TABLE,X                     ; V[cursor] (the value table, lifted by identity)
    STA opnd                        ; poke the LDX operand (self-modify)
    LDX #v  (operand @ opnd)
    STX $D400                       ; voice-0 freq-lo = V[cursor]
    LDA #$0F : STA $D418
    RTS

The dwell D and table length P are chosen so the walk loops several times inside
a short capture, so the cursor closes as a fixed-size recurrence from a short
window and reproduces the full stream byte-exact.
"""

import struct
import sys

LOAD_ADDR = 0x1000
INIT_ADDR = 0x1000
PLAY_ADDR = 0x1010

# the value table V (a looping arpeggio of freq-lo bytes); P = len(TABLE) (power of 2
# so the AND #(P-1) wrap is exact).
TABLE = [0x10, 0x20, 0x30, 0x40]
P = len(TABLE)
DWELL = 3   # cursor advances every DWELL play-calls

SUBTICK = 0x1060
CURSOR = 0x1061
TABLE_ADDR = 0x1064


def build_code():
    sub_lo, sub_hi = SUBTICK & 0xFF, (SUBTICK >> 8) & 0xFF
    cur_lo, cur_hi = CURSOR & 0xFF, (CURSOR >> 8) & 0xFF
    tab_lo, tab_hi = TABLE_ADDR & 0xFF, (TABLE_ADDR >> 8) & 0xFF

    init = [
        0xA9, 0x00,            # LDA #$00
        0x8D, 0x18, 0xD4,      # STA $D418
        0x8D, sub_lo, sub_hi,  # STA subtick = 0
        0x8D, cur_lo, cur_hi,  # STA cursor = 0
        0x60,                  # RTS
    ]
    init += [0x00] * (PLAY_ADDR - LOAD_ADDR - len(init))
    assert len(init) == PLAY_ADDR - LOAD_ADDR, len(init)

    # Assemble play in two passes to resolve the BNE target and the LDX operand addr.
    def assemble(opnd, skip_target):
        opl, oph = opnd & 0xFF, (opnd >> 8) & 0xFF
        code = []
        code += [0xEE, sub_lo, sub_hi]           # INC subtick
        code += [0xAD, sub_lo, sub_hi]           # LDA subtick
        code += [0xC9, DWELL]                     # CMP #DWELL
        bne_at = len(code)
        code += [0xD0, 0x00]                      # BNE skip (target patched)
        code += [0xA9, 0x00, 0x8D, sub_lo, sub_hi]   # LDA #0 ; STA subtick
        code += [0xEE, cur_lo, cur_hi]            # INC cursor
        code += [0xAD, cur_lo, cur_hi, 0x29, P - 1, 0x8D, cur_lo, cur_hi]  # LDA cursor;AND #P-1;STA cursor
        skip = len(code)
        code += [0xAE, cur_lo, cur_hi]            # LDX cursor
        code += [0xBD, tab_lo, tab_hi]            # LDA TABLE,X
        code += [0x8D, opl, oph]                  # STA opnd
        code += [0xA2, 0x00]                      # LDX #$00 (operand @ opnd)
        ldx_operand_local = len(code) - 1
        code += [0x8E, 0x00, 0xD4]               # STX $D400
        code += [0xA9, 0x0F, 0x8D, 0x18, 0xD4]   # LDA #$0F ; STA $D418
        code += [0x60]                            # RTS
        # patch BNE rel: target = skip; rel = skip - (bne_at+2)
        rel = skip - (bne_at + 2)
        assert 0 <= rel < 128, rel
        code[bne_at + 1] = rel
        return code, PLAY_ADDR + ldx_operand_local

    # first pass with placeholder operand to find its address
    _, opnd = assemble(0, 0)
    play, opnd2 = assemble(opnd, 0)
    assert opnd == opnd2

    code = init + play
    # place the value table at TABLE_ADDR
    pad = (TABLE_ADDR - LOAD_ADDR) - len(code)
    assert pad >= 0, f"code overruns table by {-pad} bytes (code ends at {LOAD_ADDR+len(code):#06x})"
    code += [0x00] * pad
    code += list(TABLE)
    return bytes(code)


def build_psid() -> bytes:
    c64_data = struct.pack("<H", LOAD_ADDR) + build_code()
    header = b"PSID" + struct.pack(
        ">HHHHHHHL", 2, 0x7C, 0x0000, INIT_ADDR, PLAY_ADDR, 1, 1, 0
    )
    header += b"sidtrace cursor-walk fixture".ljust(32, b"\x00")[:32]
    header += b"preframr-sidtrace".ljust(32, b"\x00")[:32]
    header += b"2026 public domain".ljust(32, b"\x00")[:32]
    header += struct.pack(">HBBBB", 0x0000, 0x00, 0x00, 0x00, 0x00)
    assert len(header) == 0x7C
    return header + c64_data


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "cursor_walk_fixture.sid"
    with open(out, "wb") as f:
        f.write(build_psid())
    sys.stderr.write(f"wrote {out}\n")


if __name__ == "__main__":
    main()
