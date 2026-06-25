#!/usr/bin/env python3
"""Generate a tiny self-contained PSID whose voice-0 FREQUENCY is read DIRECTLY from
a RAM instrument freq-table indexed by a per-play-call walking cursor -- the
instrument-table freq-modulation idiom the IWLK capture targets.

This is the freq-mod generator the PR-C fitter recovers as
``freq = note_base + freqtbl[index]`` with a per-note-onset reset. Unlike the
self-modifying-operand cursor fixture (which pokes the LDX operand and so does NOT
slice freqtbl -> $D400), here the loaded table byte feeds $D400/$D401 DIRECTLY, so
the SIDDF SID-write slicer attributes the indexed read's PC to the freq register
(feedsRegMask bit 0/1) and sidtrace emits the per-frame walk-index in the IWLK
section.

Player shape (direct freq-table walk, voice 0):

  play:
    INC cursor                  ; advance the walk index every play-call
    LDA cursor : AND #(P-1) : STA cursor   ; wrap mod table length
    LDX cursor
    LDA FREQLO,X : STA $D400    ; voice-0 freq lo = freqlo[cursor]  (DIRECT slice)
    LDA FREQHI,X : STA $D401    ; voice-0 freq hi = freqhi[cursor]  (DIRECT slice)
    LDA #$11     : STA $D404    ; gate + triangle
    LDA #$0F     : STA $D418
    RTS

The walk advances +1 (mod P) per play-call, so the IWLK per-frame index is the
ramp 0,1,2,..,P-1,0,1,.. -- a clean fit target for the freq-mod recovery.
"""

import struct
import sys

LOAD_ADDR = 0x1000
INIT_ADDR = 0x1000
PLAY_ADDR = 0x1010

# A short looping freq table (P a power of two so AND #(P-1) wraps exactly). The
# values are arbitrary distinct freq-lo / freq-hi bytes (a one-octave-ish sweep).
FREQLO = [0x00, 0x40, 0x80, 0xC0, 0x10, 0x50, 0x90, 0xD0]
FREQHI = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
P = len(FREQLO)
assert len(FREQHI) == P
assert (P & (P - 1)) == 0, "P must be a power of two"

CURSOR = 0x1060
FREQLO_ADDR = 0x1070
FREQHI_ADDR = FREQLO_ADDR + P


def build_code():
    cur_lo, cur_hi = CURSOR & 0xFF, (CURSOR >> 8) & 0xFF
    flo_lo, flo_hi = FREQLO_ADDR & 0xFF, (FREQLO_ADDR >> 8) & 0xFF
    fhi_lo, fhi_hi = FREQHI_ADDR & 0xFF, (FREQHI_ADDR >> 8) & 0xFF

    init = [
        0xA9, 0x00,            # LDA #$00
        0x8D, 0x18, 0xD4,      # STA $D418
        0x8D, cur_lo, cur_hi,  # STA cursor = 0
        0x60,                  # RTS
    ]
    init += [0x00] * (PLAY_ADDR - LOAD_ADDR - len(init))
    assert len(init) == PLAY_ADDR - LOAD_ADDR, len(init)

    play = []
    play += [0xEE, cur_lo, cur_hi]                               # INC cursor
    play += [0xAD, cur_lo, cur_hi, 0x29, P - 1, 0x8D, cur_lo, cur_hi]  # LDA cur;AND;STA
    play += [0xAE, cur_lo, cur_hi]                               # LDX cursor
    play += [0xBD, flo_lo, flo_hi, 0x8D, 0x00, 0xD4]            # LDA FREQLO,X ; STA $D400
    play += [0xBD, fhi_lo, fhi_hi, 0x8D, 0x01, 0xD4]            # LDA FREQHI,X ; STA $D401
    play += [0xA9, 0x11, 0x8D, 0x04, 0xD4]                      # LDA #$11 ; STA $D404
    play += [0xA9, 0x0F, 0x8D, 0x18, 0xD4]                      # LDA #$0F ; STA $D418
    play += [0x60]                                              # RTS

    code = init + play
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
    header += b"sidtrace freq-table fixture".ljust(32, b"\x00")[:32]
    header += b"preframr-sidtrace".ljust(32, b"\x00")[:32]
    header += b"2026 public domain".ljust(32, b"\x00")[:32]
    header += struct.pack(">HBBBB", 0x0000, 0x00, 0x00, 0x00, 0x00)
    assert len(header) == 0x7C
    return header + c64_data


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "freq_table_fixture.sid"
    with open(out, "wb") as f:
        f.write(build_psid())
    sys.stderr.write(f"wrote {out}\n")


if __name__ == "__main__":
    main()
