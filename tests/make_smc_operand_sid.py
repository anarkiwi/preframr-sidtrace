#!/usr/bin/env python3
"""Generate a tiny self-contained PSID exercising the SELF-MODIFYING-OPERAND axis
class -- the DefMon "compiled player" shape (design 1.3 SMC / 2.3).

Goto80's DefMon tunes generate every SID axis as a flat unrolled blit
``LDX #lo / STX $D4xx`` whose ``#lo`` immediate operand byte is REWRITTEN every
play-call by the master update routine. The value the axis writes is therefore
that poked operand byte -- a self-modified operand (EXEC & WRITE during play =
SMC), which is MUTABLE STATE, not a constant. Before the Stage-5 tracer fix the
SID-write slice bottomed out at the immediate as a constant leaf, so DefMon
emitted ZERO SDCU and ZERO STATESEQ (the host got none of the structure it gets
for JCH). The fix classifies a self-modified immediate operand as a state cell at
the operand address, which re-enables SDCU (the operand's update slice) and
STATESEQ (its inter-frame value sequence).

This fixture reproduces that shape minimally and deterministically (no HVSC):

  play:
    INC counter                 ; a RAM frame counter (the generator's state)
    LDA counter
    STA opnd                    ; poke the LDX operand byte (self-modification)
    LDX #<v>  (operand @ opnd)  ; load the poked value
    STX $D400                   ; voice-0 freq-lo = the SMC operand byte
    LDA #$0F
    STA $D418
    RTS

So $D400's value each frame is the poked operand byte at ``opnd``; ``opnd`` is
EXEC&WRITE during play (the LDX fetches it as code, the STA writes it), i.e. an
SMC state cell -- exactly the class the tracer must surface.

PSID v2 layout (big-endian header) per the PSID/SID file format spec.
"""

import struct
import sys

LOAD_ADDR = 0x1000
INIT_ADDR = 0x1000
PLAY_ADDR = 0x1009
COUNTER_ADDR = 0x1050   # RAM frame counter (past the code)


def build_code():
    # init @ $1000: zero the counter + volume, RTS; pad to PLAY_ADDR ($1009).
    clo, chi = COUNTER_ADDR & 0xFF, (COUNTER_ADDR >> 8) & 0xFF
    init = [
        0xA9, 0x00,            # LDA #$00
        0x8D, 0x18, 0xD4,      # STA $D418
        0x60,                  # RTS
    ]
    init += [0x00] * (PLAY_ADDR - LOAD_ADDR - len(init))
    assert len(init) == PLAY_ADDR - LOAD_ADDR

    # play @ $1009. We assemble in two passes to resolve the operand address.
    # Layout (addresses are LOAD_ADDR + offset):
    #   EE clo chi      INC counter
    #   AD clo chi      LDA counter
    #   8D opl oph      STA opnd          (self-modify the LDX operand below)
    #   A2 vv           LDX #vv           <-- operand byte 'vv' is at 'opnd'
    #   8E 00 D4        STX $D400
    #   A9 0F           LDA #$0F
    #   8D 18 D4        STA $D418
    #   60              RTS
    play_start = PLAY_ADDR
    # offset of the LDX operand within play: INC(3)+LDA(3)+STA(3)+LDX opcode(1)
    ldx_operand_off = 3 + 3 + 3 + 1
    opnd = play_start + ldx_operand_off
    opl, oph = opnd & 0xFF, (opnd >> 8) & 0xFF
    play = [
        0xEE, clo, chi,        # INC counter
        0xAD, clo, chi,        # LDA counter
        0x8D, opl, oph,        # STA opnd  (self-modify)
        0xA2, 0x00,            # LDX #$00  (operand @ opnd, poked each frame)
        0x8E, 0x00, 0xD4,      # STX $D400
        0xA9, 0x0F,            # LDA #$0F
        0x8D, 0x18, 0xD4,      # STA $D418
        0x60,                  # RTS
    ]
    code = init + play
    pad = (COUNTER_ADDR - LOAD_ADDR) - len(code)
    assert pad >= 0, f"code overruns counter cell by {-pad} bytes"
    code += [0x00] * pad + [0x00]
    return bytes(code)


def build_psid() -> bytes:
    c64_data = struct.pack("<H", LOAD_ADDR) + build_code()
    magic = b"PSID"
    version = 2
    data_offset = 0x7C
    header = magic + struct.pack(
        ">HHHHHHHL", version, data_offset, 0x0000, INIT_ADDR, PLAY_ADDR, 1, 1, 0
    )
    header += b"sidtrace SMC-operand fixture".ljust(32, b"\x00")[:32]
    header += b"preframr-sidtrace".ljust(32, b"\x00")[:32]
    header += b"2026 public domain".ljust(32, b"\x00")[:32]
    header += struct.pack(">HBBBB", 0x0000, 0x00, 0x00, 0x00, 0x00)
    assert len(header) == data_offset
    return header + c64_data


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "smc_operand_fixture.sid"
    with open(out, "wb") as f:
        f.write(build_psid())
    sys.stderr.write(f"wrote {out}\n")


if __name__ == "__main__":
    main()
