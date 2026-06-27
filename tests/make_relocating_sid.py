#!/usr/bin/env python3
"""Generate a tiny self-contained PSID that DEPACKS/RELOCATES over several play
calls before settling into steady note-table playback -- the fixture for
sidtrace's SETTLE DETECTION + steady-state STRUCTURE derivation.

The first frames are INITIALISATION spread across play-calls (a multi-frame
relocation copy), exactly the case fixed settle heuristics get wrong:

  play:
    LDA done                       ; have we finished relocating?
    BNE steady
    ; --- relocation phase: copy ONE note-table byte per play-call from the
    ;     SOURCE (in the loaded image) to the RUNTIME base, advancing a cursor.
    ;     This is fresh code-writing / table movement across many frames.
    LDX relidx
    LDA SRCTBL,X
    STA DSTTBL,X                    ; runtime note table (DIFFERENT base)
    INX
    STX relidx
    CPX #TBLLEN
    BNE return                     ; still relocating
    LDA #$01 : STA done            ; relocation complete -> steady next frame
    return: RTS

  steady:
    ; --- steady playback: walk the RUNTIME note table (post-relocation base),
    ;     read freq from it, write the SID. After settle the only indexed table
    ;     walk over data-in-image is DSTTBL (the runtime base), never SRCTBL.
    INC cursor
    LDA cursor : AND #(P-1) : STA cursor
    LDX cursor
    LDA DSTTBL,X : STA $D400        ; voice-0 freq lo from the RUNTIME table
    LDA #$11 : STA $D404
    LDA #$0F : STA $D418
    RTS

The relocation runs for TBLLEN play-calls (one byte/call), so steady state is
reached at ~frame TBLLEN -- NOT frame 0. The detector must (a) see the per-frame
code/table movement stop, (b) the exec-PC footprint saturate (the steady branch's
PCs appear once, the relocation branch stops running), and (c) the SID burst stay
periodic -- and report settle_frame ~ TBLLEN + the hold window. The recovered
steady structure base MUST be DSTTBL (runtime), not SRCTBL (static image operand).
"""

import struct
import sys

LOAD_ADDR = 0x1000
INIT_ADDR = 0x1000
PLAY_ADDR = 0x1020

# Steady note table length (power of two so AND #(P-1) wraps), and the relocated
# table length (copied one byte per frame during the init phase).
P = 16
TBLLEN = 32

# Zero-page-ish runtime scratch (state cells) and the two table bases. SRCTBL is
# in the loaded image (the static source); DSTTBL is the RUNTIME base the player
# relocates the note table to -- a DIFFERENT address. Both are inside the image
# span here (so the song-data classifier sees them), but the steady walk only ever
# touches DSTTBL.
DONE = 0x10F0
RELIDX = 0x10F1
CURSOR = 0x10F2
SRCTBL = 0x1100
DSTTBL = 0x1140  # != SRCTBL: the post-relocation runtime base

SRC_NOTES = [(i * 7 + 3) & 0xFF for i in range(TBLLEN)]  # arbitrary distinct bytes


def _abs(addr):
    return [addr & 0xFF, (addr >> 8) & 0xFF]


def build_code():
    init = [
        0xA9, 0x00,                 # LDA #$00
        0x8D, *_abs(DONE),          # STA done = 0
        0x8D, *_abs(RELIDX),        # STA relidx = 0
        0x8D, *_abs(CURSOR),        # STA cursor = 0
        0x8D, 0x18, 0xD4,           # STA $D418 = 0
        0x60,                       # RTS
    ]
    init += [0x00] * (PLAY_ADDR - LOAD_ADDR - len(init))
    assert len(init) == PLAY_ADDR - LOAD_ADDR, len(init)

    # Build the play routine. We assemble in two passes to resolve the branch
    # targets (BNE steady / BNE return) by byte offset.
    def assemble():
        play = []
        play += [0xAD, *_abs(DONE)]              # LDA done
        # BNE steady  (target filled below)
        bne_steady_at = len(play)
        play += [0xD0, 0x00]
        # relocation phase
        play += [0xAE, *_abs(RELIDX)]            # LDX relidx
        play += [0xBD, *_abs(SRCTBL)]            # LDA SRCTBL,X
        play += [0x9D, *_abs(DSTTBL)]            # STA DSTTBL,X
        play += [0xE8]                           # INX
        play += [0x8E, *_abs(RELIDX)]            # STX relidx
        play += [0xE0, TBLLEN]                   # CPX #TBLLEN
        bne_return_at = len(play)
        play += [0xD0, 0x00]                     # BNE return
        play += [0xA9, 0x01, 0x8D, *_abs(DONE)]  # LDA #$01 ; STA done
        return_at = len(play)
        play += [0x60]                           # RTS (return)
        # steady phase
        steady_at = len(play)
        play += [0xEE, *_abs(CURSOR)]            # INC cursor
        play += [0xAD, *_abs(CURSOR), 0x29, P - 1, 0x8D, *_abs(CURSOR)]  # wrap
        play += [0xAE, *_abs(CURSOR)]            # LDX cursor
        play += [0xBD, *_abs(DSTTBL), 0x8D, 0x00, 0xD4]  # LDA DSTTBL,X ; STA $D400
        play += [0xA9, 0x11, 0x8D, 0x04, 0xD4]   # LDA #$11 ; STA $D404
        play += [0xA9, 0x0F, 0x8D, 0x18, 0xD4]   # LDA #$0F ; STA $D418
        play += [0x60]                           # RTS
        # patch branch displacements (relative to the byte AFTER the branch)
        play[bne_steady_at + 1] = (steady_at - (bne_steady_at + 2)) & 0xFF
        play[bne_return_at + 1] = (return_at - (bne_return_at + 2)) & 0xFF
        return play

    play = assemble()
    code = init + play
    pad = (SRCTBL - LOAD_ADDR) - len(code)
    assert pad >= 0, f"code overruns table by {-pad} bytes"
    code += [0x00] * pad
    code += list(SRC_NOTES)                       # SRCTBL contents
    # DSTTBL starts zeroed (relocation fills it); pad up to DSTTBL+TBLLEN.
    pad2 = (DSTTBL - SRCTBL) - len(SRC_NOTES)
    assert pad2 >= 0
    code += [0x00] * pad2
    code += [0x00] * TBLLEN
    return bytes(code)


def build_psid() -> bytes:
    c64_data = struct.pack("<H", LOAD_ADDR) + build_code()
    header = b"PSID" + struct.pack(
        ">HHHHHHHL", 2, 0x7C, 0x0000, INIT_ADDR, PLAY_ADDR, 1, 1, 0
    )
    header += b"sidtrace relocating fixture".ljust(32, b"\x00")[:32]
    header += b"preframr-sidtrace".ljust(32, b"\x00")[:32]
    header += b"2026 public domain".ljust(32, b"\x00")[:32]
    header += struct.pack(">HBBBB", 0x0000, 0x00, 0x00, 0x00, 0x00)
    assert len(header) == 0x7C
    return header + c64_data


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "relocating_fixture.sid"
    with open(out, "wb") as f:
        f.write(build_psid())
    sys.stderr.write(f"wrote {out} ({len(build_psid())} bytes)\n")


if __name__ == "__main__":
    main()
