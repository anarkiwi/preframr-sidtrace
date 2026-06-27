# preframr-sidtrace

`sidtrace` is a white-box SID recovery recorder built on a lightly-patched
[libsidplayfp](https://github.com/libsidplayfp/libsidplayfp). Given a `.sid`
file it runs the tune through the cycle-accurate libsidplayfp 6510/SID emulation,
**distills the execution in the emulator**, and emits two small binary artifacts:

1. **`<prefix>.sidwr.bin`** — the timestamped SID register-write stream (one
   ~25-write burst per frame). This is the per-frame register **dump source** (it
   replaces VICE `vsiddump`) and the render/residual-zero gate.
2. **`<prefix>.distill.bin`** — the compact **SDST** artifact: the in-emulator
   distillation of the whole run (a few KB). This is the provenance substrate for
   the generic / identity BACC program recovery.

**Why distill, not dump.** The cycle-by-cycle CPU bus trace was GBs/tune —
unusable at the 60,000-tune scale (petabytes of I/O). sidtrace now does the
analysis in C++ during emulation (`src/c64/membus_trace.h`) into bounded
fixed-size accumulators (a handful of 64 KiB arrays + small maps; peak memory does
not grow with run length) and emits one few-KB SDST artifact. The retired raw bus
trace is gone.

**SMC-correct classification.** Memory is classified by ACCESS TYPE accumulated
per address: instruction-fetch (EXEC), data-read (READ), data-write (WRITE), per
init/play phase. Self-modifying code is `EXEC_PLAY & WRITE_PLAY` (executed AND
written during play); song data is `READ_PLAY & ¬WRITE_PLAY & ¬EXEC` inside the
loaded image. SMC operands are code modification, excluded from song data by the
`EXEC` term — not by a fragile write-set subtraction.

## Output layout

### `<prefix>.sidwr.bin` — SID register writes

Packed little-endian, 12 bytes/record. Every write to `$D400-$D7FF`.

| field   | type   | bytes | meaning                              |
|---------|--------|-------|--------------------------------------|
| `cycle` | int64  | 8     | absolute PHI1 CPU cycle of the write |
| `addr`  | uint16 | 2     | absolute write address (`$D4xx`)     |
| `reg`   | uint8  | 1     | SID register = `addr & 0x1F`         |
| `val`   | uint8  | 1     | byte written                         |

### `<prefix>.distill.bin` — the SDST distilled artifact

Versioned little-endian. Header (`magic "SDST"`, version, init/play/load addrs,
subtune, nframes, cycles_per_frame, t0_cycle, loaded-image length) then tagged
sections terminated by `"END\0"`:

| section | contents |
|---------|----------|
| `ACMP`  | RLE per-address ACCESS-TYPE map: `(run u16, bits u8)` over `0..65535`; `bits` = OR of EXEC/READ/WRITE × INIT/PLAY |
| `SNAP`  | post-init RAM snapshot for the SONG-DATA region only: `(addr u16, len u16, bytes[len])` runs — the verbatim song bytes, captured once at the init→play boundary |
| `SIDW`  | PC-tagged SID-write summary: `(pc u16, reg u8, _pad u8, count u32, lastVal u8, _pad[3])` — voice-lane attribution by code site |
| `IDXR`  | indexed-read VSA summary: `(pc u16, base u16, stride i32, idxMin u8, idxMax u8, _pad u16, count u32)` — table base / element size / length / traversal |
| `SDDF`  | per-SID-write **DATA-FLOW** summary, keyed by `(PC,reg)` (same key space as `SIDW`, so O(code sites) not O(frames)): a bounded in-emulator backward dynamic slice of each `STA $D4xx`'s value — the slice PCs, classified leaves (`immediate` / `ram_read` / `state_cell`[=SMC] / `exogenous` / `out_of_window`), the value-chain ALU op sequence, a strided-interval VSA for any indexed read feeding the write, and `val_lo/val_hi/val_first`. Turns "PC wrote r" into "how r was computed". Stays a flat ~1 KB as the capture window grows. |
| `STSQ`  | inter-frame **STATE-CELL sample sequence**, keyed by RAM address, for every cell `SDDF` flagged as a state leaf: `(addr u16, flags u8 [bit0=holdsToEnd, bit1=wide], _pad u8, totalFrames u32, firstSeenFrame u32, nSamples u16, then nSamples × sample u8)`. The bounded (≤`M=512`/cell) per-play-call value sequence Stage C fits the recurrence over (Daikon-style relation / Berlekamp–Massey). `holdsToEnd`: the constant-stride recurrence implied by the prefix kept holding to the end of the capture (a cheap in-emulator running check). O(state cells × M), flat once the window exceeds M and all state cells are discovered. |
| `SETL`  | **SETTLE DETECTION + CLASSIFICATION** (the player-agnostic "observe and derive" verdict). A single fixed record: `settle_found u8`, `structure_present u8`, `_pad[2]`, `settle_frame u32` (the play-call after which steady state began; the `SNAP` image is anchored here), `n_steady_idx u32`, `n_steady_ptr u32` (the `SSTR` row counts), `settle_hold u32`, `settle_cap u32` (the detector thresholds, for provenance). `structure_present`=1 ⇒ the player indexes data-region tables in the loaded image during steady playback (a data-driven tracker → route tracker-recovery); =0 ⇒ algorithmic / computed-note (e.g. *A Mind Is Born*) or digi → route generator-fit / cover-fallback. `settle_found`=0 (never settled within the cap) is itself a classification signal. See the settle-detection heuristic below. |
| `SSTR`  | **STEADY-STATE STRUCTURE**: read provenance restricted to frames AFTER settle — the tables the PLAYROUTINE indexes during steady playback, at their RUNTIME / post-relocation bases (NOT static-image operands). `n_idx u32`; then per indexed-read PC `(pc u16, base u16, stride i32, idxMin u8, idxMax u8, flags u8 [bit0=targetsData, bit1=inImage], _pad u8, count u32, feedsRegMask u32)`; then `n_ptr u32`; then per `(zp),Y` pointer pair `(zp u16, _pad u16, advCount u32)` — the post-settle orderlist→pattern advance count. The base is fitted from the OPCODE'S actual index register (not the legacy `max(x,y)` the `IDXR` fields keep), so it is the true runtime table base. |

The full byte layout is documented in `src/sidtrace.cpp` (`emit_distill`). `SDDF`
is built by a bounded backward slicer over a ring buffer of retired play-call
instructions (`src/c64/membus_trace.h`, `sliceSidWrite`); it follows data edges
(store ← value reg's last def; load ← the cell's last writer, possibly a prior
frame for persistent state; ALU ← operands; indexed ← the index def) and links a
persistent state cell to the table read that fills it across frames via a bounded
per-address filler cache — all in-emulator, nothing streamed.

### Settle detection (the `SETL` / `SSTR` "observe and derive" pass)

A tune's first frames are INITIALISATION (depack / relocate / runtime table
resolve), which runs one-shot code and writes large amounts of code and data.
Steady-state playback is reached once that work SATURATES. Rather than fragile
per-tune static-address discovery, `sidtrace` OBSERVES the run and DERIVES the
settle point, then derives the playroutine's table-access structure from steady
state and classifies whether authored data-structure is present at all.

**The heuristic.** At each play-call (frame) boundary the recorder folds three
bounded per-frame signals and declares the SETTLE FRAME as the first frame that
OPENS a window of `SETTLE_HOLD = 8` consecutive frames in which ALL hold:

- **(a) no write lands on a NEW executed-code byte** — the bulk code rewriting of
  init / depack / relocation has ended. Tracked by NOVELTY (a code byte written
  for the first time this run), not raw count: a steady self-modifying player
  (DefMon `STA opnd` each call) re-pokes the SAME operands every frame (no
  novelty), whereas a depacker writes fresh code addresses each frame. Counting
  novelty is robust to the per-frame value variation a literal-count test chokes
  on.
- **(b) no NEW PC is executed** — the player's code footprint has stopped growing,
  i.e. it has entered its fixed per-frame loop. Tracked by novelty over a
  play-phase exec-PC bitmap, so a note-trigger path that adds its PCs the first
  time it runs (then never again) does not block settle, while one-shot init /
  relocation code keeps adding PCs.
- **(c) the $D4xx SID-write cadence is PERIODIC** — a non-zero SID-write burst
  every frame (the player is producing output). We require PRESENCE, not a
  constant byte count (the count varies with note / gate events).

Thresholds: `SETTLE_HOLD = 8` frames, give up after `SETTLE_CAP = 6000`
play-calls. State is two flat 8 KiB bitmaps + a tiny `SETTLE_HOLD`-entry ring, so
the detector is flat vs run length. The chosen thresholds are emitted in `SETL`
for provenance.

**Anchoring + structure.** When settle is first declared the recorder
re-snapshots the live RAM in-emulator — so `SNAP` is anchored at the canonical
unpacked / relocated / runtime-resolved image (NOT the static load image). The
`SSTR` indexed-read / pointer-walk provenance accumulates ONLY after settle, so
its table bases are the RUNTIME (post-relocation) addresses the playroutine reads.

**Classification.** `structure_present` = the player performs an indexed
table-walk that (i) sweeps ≥ `STRUCTURE_MIN_SPAN = 4` entries, (ii) over a region
read as DATA in play, AND (iii) whose base lies in the LOADED IMAGE. Clause (iii)
is the decisive tracker-vs-computed test: authored orderlist / pattern /
instrument tables live in the image, whereas an algorithmic tune (*A Mind Is
Born*) relocates its tiny generator state into ZERO PAGE and computes notes — its
steady indexed reads are over zp/scratch, never the image, so it correctly
classifies as NO authored structure (→ generator-fit / cover-fallback). A
`(zp),Y` pointer walk ALONE is not sufficient (a generator can churn a zp pointer
pair every iteration — *A Mind* advances one ~274×/frame — which is generator
state, not an authored orderlist).

**Honest caveats.** Settle is the moment the CODE footprint saturates and output
is periodic; a player whose init shuffles only DATA on a fixed code loop with a
periodic SID burst is, by these criteria, already steady (it settles early). The
post-settle `SSTR` keeps accumulating, so a later branch-switch is still captured.
A tune that never reaches a fixed loop within the cap (continuous per-frame
depacking, some digi streamers) reports `settle_found = 0` — itself the routing
signal. `STRUCTURE_MIN_SPAN`/the in-image gate are conservative; a tracker with a
single tiny in-image table is the boundary case.

**Backward compatibility.** `SETL` and `SSTR` are appended additively before the
`END\0` terminator; every prior section (`ACMP`/`SNAP`/`SIDW`/`IDXR`/`SDDF`/
`STSQ`/`SDCU`/`IDXS`/`PWLK`/`RELO`/`SDAC`/`DIGI`/`TMPO`/`IWLK`) is byte-unchanged,
and the `.sidwr.bin` render stream is untouched, so existing `.distill.bin`
consumers keep parsing. The only contents change is that `SNAP` is now anchored at
the settle frame (a strictly better resident image); its byte LAYOUT is unchanged.

### `<prefix>.meta.txt`

Plain-text tune metadata: format, subtune, song count, init/play/load addresses,
speed, `artifact=SDST`, `cycles_per_frame`, `total_cycles`, `n_sid_writes`,
`distill_bytes`, `settle_found`, `settle_frame`, `structure_present` (the human-
readable mirror of the `SETL` verdict), and whether a real KERNAL was supplied.

Readers (numpy dtype for `.sidwr.bin`, `parse_sdst` for the artifact) live in
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
# -> out.sidwr.bin  out.distill.bin  out.meta.txt
```

## Determinism

`sidtrace` is **byte-reproducible**: tracing the same tune twice produces
byte-identical `.sidwr.bin` and `.distill.bin` (verified over many runs on Monty and
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

- both `.sidwr.bin` (whole 12-byte records) and the SDST `.distill.bin` exist and parse;
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
