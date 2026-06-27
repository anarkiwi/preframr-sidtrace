#!/usr/bin/env python3
"""Generate a tiny self-contained PSID that plays a sequence of NOTES on voice 0
with GATE retriggers -- the fixture for the VEVT per-voice note-timeline section.

Every play-call increments a frame counter. Every 16th call it advances a note
cursor, writes the note's freq from a RAM table to $D400/$D401, and RETRIGGERS the
gate ($D404: gate-off then gate-on); the other 15 calls hold the gate on (no
edge). So sidtrace's VEVT capture sees one note ONSET (gate 0->1, carrying the
note freq) every 16 frames post-settle -- a clean, deterministic per-voice note
schedule.

Player shape (voice 0):

  play:
    INC counter
    LDA counter : AND #$0F : BNE hold      ; every 16th call only
    INC noteidx : LDA noteidx : AND #(P-1) : STA noteidx
    LDX noteidx
    LDA FREQLO,X : STA $D400
    LDA FREQHI,X : STA $D401
    LDA #$10 : STA $D404                    ; gate OFF (release edge)
    LDA #$11 : STA $D404                    ; gate ON  (onset edge, note freq)
    JMP done
  hold:
    LDA #$11 : STA $D404                    ; hold gate on (no edge)
  done:
    LDA #$0F : STA $D418
    RTS
"""

import struct
import sys

LOAD_ADDR = 0x1000
INIT_ADDR = 0x1000
PLAY_ADDR = 0x1010

FREQLO = [0x00, 0x40, 0x80, 0xC0, 0x10, 0x50, 0x90, 0xD0]
FREQHI = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
P = len(FREQLO)
assert len(FREQHI) == P
assert (P & (P - 1)) == 0, "P must be a power of two"

COUNTER = 0x1060
NOTEIDX = 0x1061
FREQLO_ADDR = 0x1070
FREQHI_ADDR = FREQLO_ADDR + P


def _lohi(addr):
    return addr & 0xFF, (addr >> 8) & 0xFF


def build_play():
    """Assemble the play routine, resolving the BNE/JMP targets by label."""
    cl, ch = _lohi(COUNTER)
    nl, nh = _lohi(NOTEIDX)
    flo_l, flo_h = _lohi(FREQLO_ADDR)
    fhi_l, fhi_h = _lohi(FREQHI_ADDR)

    code = bytearray()
    fixups = []  # (offset_of_operand, target_label, kind)
    labels = {}

    def emit(*bs):
        code.extend(bs)

    emit(0xEE, cl, ch)  # INC counter
    emit(0xAD, cl, ch)  # LDA counter
    emit(0x29, 0x0F)  # AND #$0F
    emit(0xD0, 0x00)  # BNE hold  (operand patched)
    fixups.append((len(code) - 1, "hold", "rel"))

    emit(0xEE, nl, nh)  # INC noteidx
    emit(0xAD, nl, nh)  # LDA noteidx
    emit(0x29, P - 1)  # AND #(P-1)
    emit(0x8D, nl, nh)  # STA noteidx
    emit(0xAE, nl, nh)  # LDX noteidx
    emit(0xBD, flo_l, flo_h)  # LDA FREQLO,X
    emit(0x8D, 0x00, 0xD4)  # STA $D400
    emit(0xBD, fhi_l, fhi_h)  # LDA FREQHI,X
    emit(0x8D, 0x01, 0xD4)  # STA $D401
    emit(0xA9, 0x10)  # LDA #$10 (gate off)
    emit(0x8D, 0x04, 0xD4)  # STA $D404
    emit(0xA9, 0x11)  # LDA #$11 (gate on)
    emit(0x8D, 0x04, 0xD4)  # STA $D404
    emit(0x4C, 0x00, 0x00)  # JMP done (operand patched)
    fixups.append((len(code) - 2, "done", "abs"))

    labels["hold"] = len(code)
    emit(0xA9, 0x11)  # LDA #$11 (hold gate on)
    emit(0x8D, 0x04, 0xD4)  # STA $D404

    labels["done"] = len(code)
    emit(0xA9, 0x0F)  # LDA #$0F
    emit(0x8D, 0x18, 0xD4)  # STA $D418
    emit(0x60)  # RTS

    for off, label, kind in fixups:
        target = labels[label]
        if kind == "rel":
            code[off] = (target - (off + 1)) & 0xFF
        else:  # abs: 16-bit target address
            addr = PLAY_ADDR + target
            code[off], code[off + 1] = _lohi(addr)
    return bytes(code)


def build_code():
    init = [
        0xA9, 0x00,            # LDA #$00
        0x8D, 0x18, 0xD4,      # STA $D418
        0x8D, COUNTER & 0xFF, (COUNTER >> 8) & 0xFF,  # STA counter = 0
        0x8D, NOTEIDX & 0xFF, (NOTEIDX >> 8) & 0xFF,  # STA noteidx = 0
        0x60,                  # RTS
    ]
    init += [0x00] * (PLAY_ADDR - LOAD_ADDR - len(init))
    assert len(init) == PLAY_ADDR - LOAD_ADDR, len(init)

    code = list(init) + list(build_play())
    pad = (FREQLO_ADDR - LOAD_ADDR) - len(code)
    assert pad >= 0, f"code overruns table by {-pad} bytes"
    code += [0x00] * pad
    code += list(FREQLO) + list(FREQHI)
    return bytes(code)


def build_psid() -> bytes:
    c64_data = struct.pack("<H", LOAD_ADDR) + build_code()
    header = b"PSID" + struct.pack(
        ">HHHHHHHL", 2, 0x7C, 0x0000, INIT_ADDR, PLAY_ADDR, 1, 1, 0
    )
    header += b"sidtrace note-gate fixture".ljust(32, b"\x00")[:32]
    header += b"preframr-sidtrace".ljust(32, b"\x00")[:32]
    header += b"2026 public domain".ljust(32, b"\x00")[:32]
    header += struct.pack(">HBBBB", 0x0000, 0x00, 0x00, 0x00, 0x00)
    assert len(header) == 0x7C
    return header + c64_data


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "note_gate_fixture.sid"
    with open(out, "wb") as f:
        f.write(build_psid())
    sys.stderr.write(f"wrote {out}\n")


if __name__ == "__main__":
    main()
