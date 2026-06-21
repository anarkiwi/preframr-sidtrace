"""Binary record layout for sidtrace's output artifacts.

These dtypes mirror the field-by-field fwrite() in src/sidtrace.cpp exactly.
Both records are packed (no struct padding) because sidtrace writes each
field individually, little-endian, in field order.
"""
import numpy as np

# <prefix>.sidwr.bin : every SID register write ($D400-$D7FF, rw=1)
#   int64 cycle, uint16 addr, uint8 reg, uint8 val   -> 12 bytes/record
SIDWR_DT = np.dtype(
    [("cycle", "<i8"), ("addr", "<u2"), ("reg", "u1"), ("val", "u1")]
)
assert SIDWR_DT.itemsize == 12

# <prefix>.bus.bin : every CPU bus access
#   int64 cycle, uint16 addr, uint8 val, uint8 rw    -> 12 bytes/record
#   rw: 0 = read, 1 = write
BUS_DT = np.dtype(
    [("cycle", "<i8"), ("addr", "<u2"), ("val", "u1"), ("rw", "u1")]
)
assert BUS_DT.itemsize == 12


def load_sidwr(path):
    return np.fromfile(path, dtype=SIDWR_DT)


def load_bus(path):
    return np.fromfile(path, dtype=BUS_DT)


def parse_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k] = v
    return meta
