#!/usr/bin/env python3
"""Generate a tiny self-contained PSID exercising the (zp),Y POINTER WALK (the
orderlist->pattern pointer the PWLK capture resolves, design 2.6 / recovery item
#2).

The player keeps a 16-bit pointer in a zero-page pair and advances it by a fixed
stride every play-call, dereferencing it with ``LDA (ptr),Y`` to fetch a freq-lo
byte. Each new pointer value is one PWLK ADVANCE event; advancing every frame
makes the advance count grow 1:1 with the captured frame count -- so a capture of
N >> 256 frames produces N >> 256 advances. This is the fixture that proves the
PWLK 256-advance cap was lifted to full length: the resolved orderlist->pattern
schedule must cover the WHOLE run, not its first 256 advances (real tunes:
Music_Assembler 10759 advances, GoatTracker 2726).

Player shape::

  play:
    LDY #0
    LDA (PTR),Y          ; (zp),Y pointer-walk read -> PWLK capture on zp pair PTR
    STA $D400            ; voice-0 freq-lo = *ptr
    LDA #$0F : STA $D418
    CLC                  ; advance the pointer +1 (a new pointer = a PWLK advance)
    LDA PTR  : ADC #1 : STA PTR
    LDA PTR+1: ADC #0 : STA PTR+1
    RTS

The pointer sweeps forward through RAM each frame; the dereferenced byte just
needs to exist (the access map / PWLK only care about the pointer-pair walk, not
the value), so the table region is the loaded image itself.
"""

import struct
import sys

LOAD_ADDR = 0x1000
INIT_ADDR = 0x1000
PLAY_ADDR = 0x1010

# zero-page pointer pair the (zp),Y walk dereferences (the orderlist cursor).
PTR = 0xFB  # PTR / PTR+1 = $FB/$FC
PTR_INIT = 0x1000  # start the pointer at the loaded image base


def build_code():
    pl, ph = PTR_INIT & 0xFF, (PTR_INIT >> 8) & 0xFF

    init = [
        0xA9, 0x00,            # LDA #$00
        0x8D, 0x18, 0xD4,      # STA $D418
        0xA9, pl, 0x85, PTR,   # LDA #<PTR_INIT ; STA PTR
        0xA9, ph, 0x85, PTR + 1,  # LDA #>PTR_INIT ; STA PTR+1
        0x60,                  # RTS
    ]
    init += [0x00] * (PLAY_ADDR - LOAD_ADDR - len(init))
    assert len(init) == PLAY_ADDR - LOAD_ADDR, len(init)

    play = []
    play += [0xA0, 0x00]               # LDY #$00
    play += [0xB1, PTR]                # LDA (PTR),Y   <- (zp),Y pointer walk
    play += [0x8D, 0x00, 0xD4]         # STA $D400
    play += [0xA9, 0x0F, 0x8D, 0x18, 0xD4]  # LDA #$0F ; STA $D418
    play += [0x18]                     # CLC
    play += [0xA5, PTR, 0x69, 0x01, 0x85, PTR]        # LDA PTR ; ADC #1 ; STA PTR
    play += [0xA5, PTR + 1, 0x69, 0x00, 0x85, PTR + 1]  # LDA PTR+1 ; ADC #0 ; STA PTR+1
    play += [0x60]                     # RTS

    return bytes(init + play)


def build_psid() -> bytes:
    c64_data = struct.pack("<H", LOAD_ADDR) + build_code()
    header = b"PSID" + struct.pack(
        ">HHHHHHHL", 2, 0x7C, 0x0000, INIT_ADDR, PLAY_ADDR, 1, 1, 0
    )
    header += b"sidtrace ptr-walk fixture".ljust(32, b"\x00")[:32]
    header += b"preframr-sidtrace".ljust(32, b"\x00")[:32]
    header += b"2026 public domain".ljust(32, b"\x00")[:32]
    header += struct.pack(">HBBBB", 0x0000, 0x00, 0x00, 0x00, 0x00)
    assert len(header) == 0x7C
    return header + c64_data


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "ptr_walk_fixture.sid"
    with open(out, "wb") as f:
        f.write(build_psid())
    sys.stderr.write(f"wrote {out}\n")


if __name__ == "__main__":
    main()
