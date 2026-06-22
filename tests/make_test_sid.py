#!/usr/bin/env python3
"""Generate a tiny, self-contained PSID for use as a test fixture.

This produces a permissively-usable .sid file from scratch -- no HVSC, no
network, no licensing concern. The embedded 6502 player is trivial: each
call of the play routine writes a handful of SID registers (frequency,
control/gate, and the master volume), so sidtrace observes structurally
valid SID register writes in $D400-$D418.

PSID v2 layout (big-endian header) per the PSID/SID file format spec.
"""

import struct
import sys

LOAD_ADDR = 0x1000  # where the C64 data is loaded
INIT_ADDR = 0x1000
PLAY_ADDR = 0x1009
COUNTER_ADDR = 0x1040  # RAM frame-counter cell, past the code

# 6502 machine code, assembled by hand.
#
# init ($1000):
#   A9 00      LDA #$00
#   8D 18 D4   STA $D418     ; volume = 0 at init (clear)
#   60         RTS
# play ($1009):
#   AD 20 10   LDA $1020     ; load a rotating frame counter
#   8D 00 D4   STA $D400     ; voice1 freq lo
#   A9 11      LDA #$11      ; freq hi
#   8D 01 D4   STA $D401     ; voice1 freq hi
#   A9 21      LDA #$21      ; control: gate on + sawtooth
#   8D 04 D4   STA $D404     ; voice1 control
#   A9 0F      LDA #$0F
#   8D 18 D4   STA $D418     ; master volume = 15
#   EE 20 10   INC $1020     ; advance the frame counter (RAM state)
#   60         RTS
# data:
#   $1040: 00   frame counter byte (COUNTER_ADDR)
_CLO = COUNTER_ADDR & 0xFF
_CHI = (COUNTER_ADDR >> 8) & 0xFF
CODE = bytes(
    [
        # init @ $1000
        0xA9,
        0x00,  # LDA #$00
        0x8D,
        0x18,
        0xD4,  # STA $D418
        0x60,  # RTS
        0x00,
        0x00,
        0x00,  # padding to reach $1009 (PLAY_ADDR)
        # play @ $1009
        0xAD,
        _CLO,
        _CHI,  # LDA COUNTER_ADDR
        0x8D,
        0x00,
        0xD4,  # STA $D400   (voice1 freq lo)
        0xA9,
        0x11,  # LDA #$11
        0x8D,
        0x01,
        0xD4,  # STA $D401   (voice1 freq hi)
        0xA9,
        0x21,  # LDA #$21
        0x8D,
        0x04,
        0xD4,  # STA $D404   (voice1 control: gate + sawtooth)
        0xA9,
        0x0F,  # LDA #$0F
        0x8D,
        0x18,
        0xD4,  # STA $D418   (master volume = 15)
        0xEE,
        _CLO,
        _CHI,  # INC COUNTER_ADDR (RAM state advances each frame)
        0x60,  # RTS
    ]
)
assert PLAY_ADDR - LOAD_ADDR == 9, "play routine must start at $1009"
# pad out to the frame-counter data cell at COUNTER_ADDR
pad = (COUNTER_ADDR - LOAD_ADDR) - len(CODE)
assert pad >= 0, f"code overruns counter cell by {-pad} bytes"
CODE = CODE + bytes(pad) + bytes([0x00])


def build_psid() -> bytes:
    # C64 data: 2-byte little-endian load address prefix + code.
    c64_data = struct.pack("<H", LOAD_ADDR) + CODE

    magic = b"PSID"
    version = 2
    data_offset = 0x7C  # v2 header is 0x7C bytes
    load_addr = 0x0000  # 0 => take load address from the c64 data prefix
    init_addr = INIT_ADDR
    play_addr = PLAY_ADDR
    songs = 1
    start_song = 1
    speed = 0x00000000  # 50Hz VBI for all songs
    name = b"sidtrace test fixture".ljust(32, b"\x00")[:32]
    author = b"preframr-sidtrace".ljust(32, b"\x00")[:32]
    released = b"2026 public domain".ljust(32, b"\x00")[:32]
    flags = 0x0000
    start_page = 0x00
    page_length = 0x00
    second_sid = 0x00
    third_sid = 0x00

    # >HHHHHHH = version,dataOffset,loadAddr,initAddr,playAddr,songs,startSong
    # >L       = speed
    header = magic + struct.pack(
        ">HHHHHHHL",
        version,
        data_offset,
        load_addr,
        init_addr,
        play_addr,
        songs,
        start_song,
        speed,
    )
    header += name + author + released
    header += struct.pack(
        ">HBBBB", flags, start_page, page_length, second_sid, third_sid
    )
    assert len(header) == data_offset, f"header len {len(header)} != {data_offset}"
    return header + c64_data


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "test_fixture.sid"
    data = build_psid()
    with open(out, "wb") as f:
        f.write(data)
    sys.stderr.write(f"wrote {out} ({len(data)} bytes)\n")


if __name__ == "__main__":
    main()
