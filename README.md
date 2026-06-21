# preframr-sidtrace

`sidtrace` is a white-box SID recovery recorder built on a lightly-patched
[libsidplayfp](https://github.com/libsidplayfp/libsidplayfp). Given a `.sid`
file it runs the tune through the cycle-accurate libsidplayfp 6510/SID
emulation and emits two cycle-stamped binary artifacts:

1. **`<prefix>.sidwr.bin`** — every SID register write. This is the
   headlessvice-compatible **dump source** (it replaces VICE `vsiddump` for
   per-frame register dumps).
2. **`<prefix>.bus.bin`** — the full CPU **bus trace**: every 6510 read and
   write, cycle-stamped. This is the **provenance substrate** for the generic
   BACC program recovery.

This repo is **Phase 1** of the headlessvice-replacement plan: build the tool
and emit both artifacts, with CI + tests. The offline provenance-graph analysis
and generic BACC recovery are Phase 2 (out of scope here). The authoritative
design is `design/infra/libsidplay_callgraph_recovery_design.md` in the
`preframr-xpt` repo.

## Output record layout

Both files are **packed little-endian** records (no struct padding — each field
is written individually by the recorder). 12 bytes per record in both files.

### `<prefix>.sidwr.bin` — SID register writes

Every write to `$D400-$D7FF` (any SID chip), `rw == write`.

| field   | type   | bytes | meaning                              |
|---------|--------|-------|--------------------------------------|
| `cycle` | int64  | 8     | absolute PHI1 CPU cycle of the write |
| `addr`  | uint16 | 2     | absolute write address (`$D4xx`)     |
| `reg`   | uint8  | 1     | SID register = `addr & 0x1F`         |
| `val`   | uint8  | 1     | byte written                         |

### `<prefix>.bus.bin` — full CPU bus trace

Every CPU data-bus access (read and write).

| field   | type   | bytes | meaning                              |
|---------|--------|-------|--------------------------------------|
| `cycle` | int64  | 8     | absolute PHI1 CPU cycle of the access|
| `addr`  | uint16 | 2     | bus address                          |
| `val`   | uint8  | 1     | byte on the bus                      |
| `rw`    | uint8  | 1     | `0` = read, `1` = write              |

The SID-write log is exactly the subset of the bus trace with `rw == 1` and
`$D400 <= addr <= $D7FF`. Frames are delineated by the host from the SID-write
cycle clusters (the same way the corpus register dump uses the IRQ cycle as the
frame id).

`SIDTRACE_NOBUS=1` skips the (large) `.bus.bin` for a cheap sidwr-only pass.

### `<prefix>.meta.txt`

Plain-text tune metadata: format, subtune, song count, init/play/load
addresses, speed, `cycles_per_frame`, `total_cycles`, `n_sid_writes`,
`n_bus_accesses`, and whether a real KERNAL was supplied.

Numpy dtypes for reading the artifacts live in
[`tests/sidtrace_records.py`](tests/sidtrace_records.py).

## Build

A fresh clone needs the libsidplayfp submodule:

```sh
git clone https://github.com/anarkiwi/preframr-sidtrace
cd preframr-sidtrace
git submodule update --init --recursive
make            # produces build/sidtrace
```

Build prerequisites: a C++17 compiler, `autoconf`/`automake`/`libtool`,
`pkg-config`, `make`, and the `xa` cross-assembler (libsidplayfp assembles its
PSID driver from `.a65` source at build time). Debian/Ubuntu:
`build-essential autoconf automake libtool pkg-config xa65`.

`make` (1) applies the bus-trace patch into the pinned submodule, (2) runs
`autoreconf` + `configure` for static libs, (3) builds `libsidplayfp.a` +
`libsidplayfp-sidlite.a`, then (4) links `src/sidtrace.cpp` against them into
`build/sidtrace`. `make distclean` reverts the submodule to pristine.

## Run

```sh
build/sidtrace <file.sid> <subtune(1-based)> <nframes> <out_prefix> \
    [kernal] [basic] [chargen]
```

Optional real C64 ROM images are needed for RSID / KERNAL-calling tunes; PSID
tunes (like the bundled test fixture) need none. Example:

```sh
build/sidtrace tune.sid 1 3000 out
# -> out.sidwr.bin  out.bus.bin  out.meta.txt
```

## Determinism

`sidtrace` is **byte-reproducible**: tracing the same tune twice produces
byte-identical `.sidwr.bin` and `.bus.bin` (verified over 20 runs on Monty and
Grid_Runner, and gated in CI with `cmp`).

This is not automatic. libsidplayfp's default `SidConfig.powerOnDelay` is the
sentinel `DEFAULT_POWER_ON_DELAY` (`MAX_POWER_ON_DELAY + 1`), which makes
`Player::initialise()` pick a **random** power-on delay seeded from
`std::time(nullptr)`. A different power-on delay means a different number of
warm-up clocks before the play routine runs, so the boot lag — and the entire
trace length — drift run-to-run (the `.sidwr.bin` was observed to vary by ~kB
and occasionally collapse to almost nothing). `sidtrace` pins `powerOnDelay` to
a fixed value (default `0`), which takes the deterministic branch so every run
starts from byte-identical C64 state. The C64 RAM power-up pattern
(`SystemRAMBank::reset`, a fixed VICE-like checkerboard) and the dangling-bus
LCG (`MMU`, fixed seed) are already deterministic, so this was the sole
remaining source of variation. Override for experiments with
`SIDTRACE_POWER_ON_DELAY=<n>` (clamped to `MAX_POWER_ON_DELAY`).

### Known differences vs VICE (`vsiddump`)

These are genuine emulator-convention differences, not config bugs; they are
documented here so the consuming/recovery layer can account for them:

- **Boot prolog.** VICE emits a leading all-zero init frame (frame 0 is all
  zeros, music starts at frame 1). libsidplayfp/`sidtrace` emits **no** leading
  zero frame — its frame 0 is already the first music frame (== VICE's frame 1).
  `sidtrace` writes a raw cycle-stamped *write log*, not a frame-segmented dump,
  so it does not (and must not) fabricate a zero frame; the host segmenter /
  recovery aligns the boot offset. The remaining per-tune boot lag is a small
  fixed offset (a function of `powerOnDelay`).
- **~8% shorter tail.** On GoatTracker tunes, libsidplayfp's play routine stops
  driving the SID at ~92% of VICE's songlength (Grid_Runner: ~14680 frames vs
  VICE's ~15681) even when `sidtrace` is given a *larger* cycle budget than
  needed — the writes stop abruptly, they do not taper. This is intrinsic to
  libsidplayfp's emulation of the tune's song-end and is **independent of
  `powerOnDelay`** (verified across delays 0–8191: tail stays at ~14680). It is
  a fundamental libsidplayfp-vs-VICE difference, quantified here, not a knob.

## How the patched libsidplayfp dependency is carried

libsidplayfp is a **git submodule** (`third_party/libsidplayfp`) pinned to a
specific upstream commit (`6018c45` of `libsidplayfp/libsidplayfp`). The only
change vs upstream is a small **bus-trace instrumentation hook**, captured as a
committed patch in
[`patches/0001-membus-trace-instrumentation.patch`](patches/0001-membus-trace-instrumentation.patch)
and applied idempotently at build time. The patch is three files:

- `src/c64/membus_trace.h` — a process-global `MemBusTrace` recorder (new file);
- `src/c64/c64cpu.h` — hooks `cpuRead`/`cpuWrite` to record every bus access;
- `src/c64/c64.cpp` — installs a PHI1 clock getter so accesses are
  cycle-stamped.

The `sidlite` SID builder used for the trace is part of upstream libsidplayfp
(no patch needed). Dependabot watches both the GitHub Actions and the submodule;
a submodule bump that breaks the patch surfaces as a red CI (the patch apply
fails the build).

## Tests

```sh
make test    # builds, then runs pytest
```

The suite (`tests/test_sidtrace.py`) generates a tiny, fully self-contained PSID
fixture from scratch (`tests/make_test_sid.py` — no HVSC, no network, no
licensing concern), runs `sidtrace` on it, and asserts:

- both `.sidwr.bin` / `.bus.bin` exist with whole (12-byte) records;
- cycles are monotonically non-decreasing in both files;
- every SID write lands in `$D400-$D418` with `reg == addr & 0x1F`, `reg <= 24`;
- the SID-write log is exactly the `$D4xx`, `rw=1` subset of the bus trace
  (cycle/addr/val agree);
- `meta.txt` counts match the file sizes;
- the per-frame write cadence tracks the requested frame count and the player's
  RAM frame-counter advances (state threaded through the bus).

CI (GitHub Actions) builds the patched libsidplayfp + sidtrace and runs the
tests on every push to `main` and every PR, caching the libsidplayfp build.

## License

This repository's own sources (`src/sidtrace.cpp`, the patch, the tests) are
licensed under Apache-2.0 — see [LICENSE](LICENSE). Note that `sidtrace` links
against [libsidplayfp](https://github.com/libsidplayfp/libsidplayfp), which is
GPL-2.0-or-later, so the **compiled `sidtrace` binary** is a combined work
distributable under GPL-2.0-or-later.
