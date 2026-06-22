/*
 * sidtrace - white-box SID recovery wrapper around instrumented libsidplayfp.
 *
 * Usage:
 *   sidtrace <file.sid> <subtune(1-based)> <nframes> <out_prefix> [kernal] [basic] [chargen]
 *
 * Emits:
 *   <out_prefix>.sidwr.bin   - timestamped SID register-write stream : records of
 *                              int64 cycle, uint16 addr, uint8 reg, uint8 val
 *                              (the per-frame dump / render gate; small).
 *   <out_prefix>.distill.bin - the compact SDST artifact: the in-emulator
 *                              DISTILLATION of the run (access-type map, post-init
 *                              song-data snapshot, PC-tagged SID-write summary,
 *                              indexed-read VSA summary). A few KB; see emit_distill.
 *   <out_prefix>.meta.txt    - tune metadata (init/play/load, speed, frame cycles).
 *
 * The cycle-by-cycle bus trace is RETIRED: the emulator distills the execution
 * in place (src/c64/membus_trace.h) instead of streaming a multi-GB raw trace,
 * so the pipeline scales to tens of thousands of tunes. Frames are delineated by
 * the host from the SID-write cycle clusters (the IRQ cadence).
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

#include "sidplayfp/sidplayfp.h"
#include "sidplayfp/SidTune.h"
#include "sidplayfp/SidTuneInfo.h"
#include "sidplayfp/SidConfig.h"
#include "sidplayfp/SidInfo.h"
#include "builders/sidlite-builder/sidlite.h"
#include "c64/membus_trace.h"

using libsidplayfp::MemBusTrace;

static std::vector<uint8_t> readFile(const char *path)
{
    std::vector<uint8_t> v;
    FILE *f = fopen(path, "rb");
    if (!f) return v;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) v.clear();
    fclose(f);
    return v;
}

/*
 * SDST -- the compact distilled artifact (a few KB; replaces the multi-GB raw
 * bus trace). One file per tune. Little-endian. Layout:
 *
 *   magic     char[4]   "SDST"
 *   version   u16       = 1
 *   reserved  u16       = 0
 *   init      u16       PSID init address
 *   play      u16       PSID play address
 *   load      u16       PSID load address
 *   subtune   u16       1-based subtune
 *   nframes   u16       frames requested
 *   reserved2 u16
 *   cyc/frame u32       PAL cycles per frame
 *   t0_cycle  i64       first play-phase SID write cycle (init->play anchor)
 *
 *   SECTION "ACMP" (access-type map, RLE):  the per-address AccBits, run-length
 *     encoded as (count u16, bits u8) pairs over the full 0..65535 address
 *     space.  bits = OR of EXEC/READ/WRITE x INIT/PLAY (see AccBits).  SMC and
 *     song-data are derived offline from these bits -- code/data/SMC by access
 *     TYPE, never by write-set subtraction.
 *       tag char[4] "ACMP"; nbytes u32; then nbytes of (u16 count, u8 bits).
 *
 *   SECTION "SNAP" (post-init RAM snapshot):  the verbatim bytes the recovery
 *     lifts as constants K by identity, as (addr u16, len u16, bytes[len]) runs.
 *     Two eligibility classes: (1) the classic resident song-data region (RAM,
 *     never written/executed during play, inside the loaded image span); (2) ANY
 *     cell READ AS DATA during play within zero page or the loaded image span --
 *     so a self-relocating generative player's zero-page generator tables (A Mind
 *     keeps its tables in EXEC'd zp $f3/$f7) are captured too. The host classifies
 *     code/data/SMC via ACMP; SNAP is just the verbatim byte value.
 *       tag char[4] "SNAP"; nbytes u32; then the (addr,len,bytes) runs.
 *
 *   SECTION "SIDW" (PC-tagged SID-write summary):  voice-lane attribution by
 *     code site.  count u32 entries of (pc u16, reg u8, _pad u8, n u32,
 *     lastVal u8, _pad3 u8[3]).
 *       tag char[4] "SIDW"; nentries u32; then the entries.
 *
 *   SECTION "IDXR" (indexed-read / table-access VSA summary):  count u32
 *     entries of (pc u16, base u16, stride i32, idxMin u8, idxMax u8,
 *     _pad u16, n u32) -- table base / element size / length / traversal.
 *       tag char[4] "IDXR"; nentries u32; then the entries.
 *
 *   SECTION "SIDDF" (per-SID-write DATA-FLOW summary; design 3.1).  Keyed by
 *     (issuing PC, SID register) -- the SAME key space as SIDW, so O(code sites)
 *     NOT O(frames).  For each STA $D4xx code site, a BOUNDED in-emulator backward
 *     dynamic slice (over a ring buffer of retired play-call instructions): the
 *     slice PC set (the generator's code), the classified leaves (immediate /
 *     ram_read / state_cell[=SMC] / exogenous / out-of-window via the access-type
 *     map), the ALU op sequence on the value chain, a strided-interval VSA for any
 *     indexed read feeding THIS write, and min/max/first written value.  This is
 *     the per-write computation DAG the recurrence-recovery host needs.  Tag is 4
 *     bytes "SDDF".  Layout: tag; nentries u32; then per entry:
 *        pc u16, reg u8, flags u8 (bit0=hasStride, bit1=anyOutOfWindow),
 *        count u32,
 *        valLo u8, valHi u8, valFirst u8, _pad u8,
 *        strideBase u16, strideStep i32, strideIdxMin u8, strideIdxMax u8,
 *        nPcs u16,   then nPcs * (pc u16),
 *        nLeaves u16, then nLeaves * (kind u8, _pad u8, addr u16, value u8, _pad u8),
 *        nOps u16,   then nOps * (op u8).
 *
 *   SECTION "STSQ" (bounded inter-frame STATE-CELL sample sequence; design 3.2).
 *     For each cell SIDDF flagged as a state leaf, a bounded sequence of its
 *     value across play-calls (frames): the inter-frame state samples Stage C
 *     (Daikon-style recurrence inference / Berlekamp-Massey) fits.  Keyed by RAM
 *     address; O(state cells * M), FLAT vs frame count (M is capped).  Tag is 4
 *     bytes "STSQ".  Layout: tag; nentries u32; then per entry:
 *        addr u16, flags u8 (bit0=holdsToEnd, bit1=wide), _pad u8,
 *        totalFrames u32, firstSeenFrame u32, nSamples u16,
 *        then nSamples * (sample u8).  (Cells are byte-wide in practice; the
 *        wide flag is reserved for a future 16-bit pair and is never set here.)
 *     firstSeenFrame: global play-call index of sample[0] (a cell that becomes a
 *     SIDDF state leaf mid-run starts sampling later; the host aligns to the
 *     master frame grid with this).
 *     holdsToEnd: the constant-stride (first-difference, mod byte width)
 *     recurrence implied by the sample prefix kept holding to the end of the
 *     capture (a cheap in-emulator running check); the host can trust a short
 *     window without re-deriving over the whole run.  wide: a sample exceeded
 *     0xff (a 16-bit cell pair); samples are u8 in practice.
 *
 *   SECTION "SDCU" (per-state-cell UPDATE data-flow summary; design 2.2/2.3).
 *     Same variable-length entry layout as SDDF, but keyed by the state-cell
 *     ADDRESS (in the `pc` field; `reg` unused). For a fast mid-call SMC
 *     accumulator the SID-write slice bottoms out at the shadow cell as a LEAF
 *     (the blit just copies it to the register); SDCU is the backward slice of the
 *     STORE (or in-place INC/DEC/shift RMW) that DEFINED the cell earlier in the
 *     SAME play-call -- the update DAG cell' = f(cell, C, K, immediates) the host
 *     generalizes into U. Only cells SIDDF flagged as state leaves are emitted.
 *     O(state cells), flat vs frame count.  Tag char[4] "SDCU".
 *
 *   SECTION "END\0" terminates.
 *
 * Returns the artifact size in bytes.
 */
static void wr_u16(FILE *f, uint16_t v) { fwrite(&v, 2, 1, f); }
static void wr_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void wr_i32(FILE *f, int32_t v)  { fwrite(&v, 4, 1, f); }
static void wr_i64(FILE *f, int64_t v)  { fwrite(&v, 8, 1, f); }
static void wr_u8(FILE *f, uint8_t v)   { fwrite(&v, 1, 1, f); }

static long emit_distill(const char *path, MemBusTrace &tr,
                         const SidTuneInfo *ti, int subtune, int nframes,
                         long cyclesPerFrame, long /*totalCycles*/)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    fwrite("SDST", 1, 4, f);
    wr_u16(f, 1);                              // version
    wr_u16(f, 0);                              // reserved
    wr_u16(f, ti ? (uint16_t)ti->initAddr() : 0);
    wr_u16(f, ti ? (uint16_t)ti->playAddr() : 0);
    wr_u16(f, ti ? (uint16_t)ti->loadAddr() : 0);
    wr_u16(f, (uint16_t)subtune);
    wr_u16(f, (uint16_t)nframes);
    wr_u16(f, 0);                              // reserved2
    wr_u32(f, (uint32_t)cyclesPerFrame);
    wr_i64(f, tr.firstSidWriteCycle);
    wr_u32(f, ti ? (uint32_t)ti->c64dataLen() : 0);   // loaded image length

    // ACMP: RLE the 64 KiB access-type map.
    {
        fwrite("ACMP", 1, 4, f);
        long lenPos = ftell(f);
        wr_u32(f, 0);                          // backpatched
        long start = ftell(f);
        uint32_t a = 0;
        while (a < 65536)
        {
            uint8_t bits = tr.acc[a];
            uint32_t run = 1;
            while (a + run < 65536 && tr.acc[a + run] == bits && run < 0xffff)
                run++;
            wr_u16(f, (uint16_t)run);
            wr_u8(f, bits);
            a += run;
        }
        long end = ftell(f);
        fseek(f, lenPos, SEEK_SET);
        wr_u32(f, (uint32_t)(end - start));
        fseek(f, end, SEEK_SET);
    }

    // SNAP: the post-init RAM snapshot, SPARSE -- the bytes the recovery lifts as
    // constants K by IDENTITY (design 2.4), stored as contiguous (addr u16, len
    // u16, bytes[len]) runs. This keeps the artifact tiny: we do NOT snapshot the
    // whole 64 KiB (most of which is code, scratch, or unused), only the bytes O
    // (or a state-cell filler) dereferenced.
    //
    // TWO eligibility classes (design 2.4 + 3.0 "widen SNAP"):
    //  (1) the song-data region: RAM, never written during play, never executed --
    //      the classic init/load-resident table the player only reads, bounded to
    //      the loaded image span [loadAddr, loadAddr+c64dataLen). SMC operands
    //      (EXEC & WRITE) are excluded here by the !EXEC term.
    //  (2) ANY cell READ AS DATA during play (ACC_READ_PLAY), regardless of EXEC.
    //      The song-data-region heuristic is a FORMAT ASSUMPTION; a generative /
    //      self-relocating player (A Mind Is Born relocates its whole player +
    //      its small generator tables into zero page, so the table O dereferences
    //      at runtime $f7 is ALSO EXEC there) keeps its constant tables in cells
    //      that are EXEC_PLAY. The generic recovery must lift K from wherever O
    //      dereferenced, INCLUDING executed zero page. We therefore also snapshot
    //      every RAM cell read as data during play. The host classifies code vs
    //      data vs SMC via ACMP -- the snapshot is just the verbatim byte value at
    //      the dereferenced address, never a structure claim. Still bounded: a
    //      play-call reads only a handful of distinct data cells per axis.
    {
        fwrite("SNAP", 1, 4, f);
        long lenPos = ftell(f);
        wr_u32(f, 0);
        long start = ftell(f);
        const uint32_t loadLo = ti ? ti->loadAddr() : 0;
        const uint32_t loadHi = ti ? (loadLo + ti->c64dataLen()) : 0x10000;
        auto eligible = [&](uint32_t a) -> bool {
            if (a < 0x0002 || a >= 0xd000) return false;   // RAM only, skip IO
            const uint8_t b = tr.acc[a];
            const bool writePlay = b & libsidplayfp::ACC_WRITE_PLAY;
            const bool readPlay  = b & libsidplayfp::ACC_READ_PLAY;
            const bool execAny   = b & (libsidplayfp::ACC_EXEC_INIT |
                                        libsidplayfp::ACC_EXEC_PLAY);
            // (2) read as data during play within zero page OR the loaded image
            // span: the dereferenced constant/table, including a relocated
            // player's executed zero page (A Mind keeps its generator tables in
            // zp $f3/$f7, EXEC there). Bounded to zp+image so a busy player's
            // reads of large scratch/high-RAM regions don't bloat the snapshot.
            if (readPlay && (a < 0x0100 || (a >= loadLo && a < loadHi)))
                return true;
            // (1) classic resident song-data table inside the loaded image span.
            if (a < loadLo || a >= loadHi) return false;
            return !writePlay && !execAny;
        };
        uint32_t a = 0;
        while (a < 65536)
        {
            if (!eligible(a)) { a++; continue; }
            uint32_t runStart = a;
            bool anyRead = false;
            while (a < 65536 && eligible(a))
            {
                if (tr.acc[a] & libsidplayfp::ACC_READ_PLAY) anyRead = true;
                a++;
            }
            if (!anyRead) continue;            // never read as data -> not song
            uint32_t runEnd = a;               // [runStart, runEnd)
            // chunk into <=0xffff segments to fit the 16-bit length field
            for (uint32_t s = runStart; s < runEnd; )
            {
                uint32_t len = runEnd - s;
                if (len > 0xffff) len = 0xffff;
                wr_u16(f, (uint16_t)s);
                wr_u16(f, (uint16_t)len);
                fwrite(&tr.ramSnapshot[s], 1, len, f);
                s += len;
            }
        }
        long end = ftell(f);
        fseek(f, lenPos, SEEK_SET);
        wr_u32(f, (uint32_t)(end - start));
        fseek(f, end, SEEK_SET);
    }

    // SIDW: PC-tagged SID-write summary.
    {
        fwrite("SIDW", 1, 4, f);
        wr_u32(f, (uint32_t)tr.sidWrites.size());
        for (const auto &kv : tr.sidWrites)
        {
            const uint16_t pc = (uint16_t)(kv.first >> 5);
            const uint8_t reg = (uint8_t)(kv.first & 0x1f);
            wr_u16(f, pc);
            wr_u8(f, reg);
            wr_u8(f, 0);
            wr_u32(f, (uint32_t)kv.second.count);
            wr_u8(f, kv.second.lastVal);
            wr_u8(f, 0); wr_u8(f, 0); wr_u8(f, 0);
        }
    }

    // IDXR: indexed-read VSA summary.
    {
        fwrite("IDXR", 1, 4, f);
        wr_u32(f, (uint32_t)tr.idxReads.size());
        for (const auto &kv : tr.idxReads)
        {
            wr_u16(f, kv.first);                       // pc
            wr_u16(f, kv.second.base);
            wr_i32(f, kv.second.strideGuess);
            wr_u8(f, kv.second.idxMin == 0xff ? 0 : kv.second.idxMin);
            wr_u8(f, kv.second.idxMax);
            wr_u16(f, 0);
            wr_u32(f, (uint32_t)kv.second.count);
        }
    }

    // SIDDF: per-(PC,reg) data-flow summary (the per-write computation DAG).
    {
        fwrite("SDDF", 1, 4, f);
        wr_u32(f, (uint32_t)tr.siddf.size());
        for (const auto &kv : tr.siddf)
        {
            const uint16_t pc = (uint16_t)(kv.first >> 5);
            const uint8_t  reg = (uint8_t)(kv.first & 0x1f);
            const libsidplayfp::SiddfSummary &s = kv.second;
            uint8_t flags = 0;
            if (s.hasStride)      flags |= 0x01;
            if (s.anyOutOfWindow) flags |= 0x02;
            wr_u16(f, pc);
            wr_u8(f, reg);
            wr_u8(f, flags);
            wr_u32(f, (uint32_t)s.count);
            wr_u8(f, s.valSeen ? s.valLo : 0);
            wr_u8(f, s.valHi);
            wr_u8(f, s.valFirst);
            wr_u8(f, 0);
            wr_u16(f, s.strideBase);
            wr_i32(f, s.strideStep);
            wr_u8(f, s.strideIdxMin == 0xff ? 0 : s.strideIdxMin);
            wr_u8(f, s.strideIdxMax);
            wr_u16(f, (uint16_t)s.slicePcs.size());
            for (uint16_t p : s.slicePcs) wr_u16(f, p);
            wr_u16(f, (uint16_t)s.leaves.size());
            for (const auto &l : s.leaves)
            {
                wr_u8(f, l.kind);
                wr_u8(f, 0);
                wr_u16(f, l.addr);
                wr_u8(f, l.value);
                wr_u8(f, 0);
            }
            wr_u16(f, (uint16_t)s.opSeq.size());
            for (uint8_t op : s.opSeq) wr_u8(f, op);
        }
    }

    // STSQ: bounded inter-frame state-cell sample sequences, filtered to the
    // cells SIDDF flagged as state leaves (design 3.2). O(state cells * M), flat.
    {
        fwrite("STSQ", 1, 4, f);
        long nentPos = ftell(f);
        wr_u32(f, 0);                          // nentries, backpatched
        uint32_t nent = 0;
        for (const auto &kv : tr.stateSeq)
        {
            const uint16_t addr = kv.first;
            if (!tr.isSiddfStateCell(addr)) continue;   // only real state cells
            const libsidplayfp::StateSeqCell &c = kv.second;
            uint8_t flags = 0;
            if (c.constDiff)  flags |= 0x01;   // holds_to_end
            if (c.wide)       flags |= 0x02;
            wr_u16(f, addr);
            wr_u8(f, flags);
            wr_u8(f, 0);
            wr_u32(f, (uint32_t)c.totalFrames);
            wr_u32(f, (uint32_t)c.firstSeenFrame);
            wr_u16(f, (uint16_t)c.nSamples);
            for (uint32_t i = 0; i < c.nSamples; ++i)
                wr_u8(f, (uint8_t)c.samples[i]);
            nent++;
        }
        long end = ftell(f);
        fseek(f, nentPos, SEEK_SET);
        wr_u32(f, nent);
        fseek(f, end, SEEK_SET);
    }

    // SDCU: per-state-cell UPDATE data-flow summary (design 2.2/2.3). Same
    // variable-length entry layout as SDDF, but keyed by the state-cell ADDRESS
    // (written into the `pc` field; `reg` is unused/0). For a fast mid-call SMC
    // accumulator the SID-write slice bottoms out at the cell as a leaf (the blit
    // just copies the shadow cell to the register); SDCU is the backward slice of
    // the STORE that DEFINED the cell earlier in the SAME play-call -- the update
    // DAG cell' = f(cell, C, K, immediates) the host generalizes into U. Only the
    // cells SIDDF flagged as state leaves are emitted (the handful that drive an
    // axis), keeping it O(state cells), flat vs frame count. Tag "SDCU".
    {
        fwrite("SDCU", 1, 4, f);
        long nentPos = ftell(f);
        wr_u32(f, 0);                          // nentries, backpatched
        uint32_t nent = 0;
        for (const auto &kv : tr.sdcu)
        {
            const uint16_t cellAddr = kv.first;
            if (!tr.isSiddfStateCell(cellAddr)) continue;  // only real state cells
            const libsidplayfp::SiddfSummary &s = kv.second;
            uint8_t flags = 0;
            if (s.hasStride)      flags |= 0x01;
            if (s.anyOutOfWindow) flags |= 0x02;
            wr_u16(f, cellAddr);               // key: the state-cell address
            wr_u8(f, 0);                       // reg (unused for SDCU)
            wr_u8(f, flags);
            wr_u32(f, (uint32_t)s.count);
            wr_u8(f, s.valSeen ? s.valLo : 0);
            wr_u8(f, s.valHi);
            wr_u8(f, s.valFirst);
            wr_u8(f, 0);
            wr_u16(f, s.strideBase);
            wr_i32(f, s.strideStep);
            wr_u8(f, s.strideIdxMin == 0xff ? 0 : s.strideIdxMin);
            wr_u8(f, s.strideIdxMax);
            wr_u16(f, (uint16_t)s.slicePcs.size());
            for (uint16_t p : s.slicePcs) wr_u16(f, p);
            wr_u16(f, (uint16_t)s.leaves.size());
            for (const auto &l : s.leaves)
            {
                wr_u8(f, l.kind);
                wr_u8(f, 0);
                wr_u16(f, l.addr);
                wr_u8(f, l.value);
                wr_u8(f, 0);
            }
            wr_u16(f, (uint16_t)s.opSeq.size());
            for (uint8_t op : s.opSeq) wr_u8(f, op);
            nent++;
        }
        long end = ftell(f);
        fseek(f, nentPos, SEEK_SET);
        wr_u32(f, nent);
        fseek(f, end, SEEK_SET);
    }

    fwrite("END\0", 1, 4, f);
    long sz = ftell(f);
    fclose(f);
    return sz;
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        fprintf(stderr,
            "usage: %s <file.sid> <subtune> <nframes> <out_prefix> "
            "[kernal] [basic] [chargen]\n", argv[0]);
        return 2;
    }
    const char *sidPath = argv[1];
    const int   subtune = atoi(argv[2]);
    const int   nframes = atoi(argv[3]);
    const std::string outPrefix = argv[4];

    SidTune tune(sidPath);
    if (!tune.getStatus())
    {
        fprintf(stderr, "ERR tune load: %s\n", tune.statusString());
        return 1;
    }
    tune.selectSong(subtune);

    sidplayfp engine;

    // Optional real ROMs (needed for RSID / KERNAL-calling tunes).
    std::vector<uint8_t> kernal, basic, chargen;
    if (argc > 5) { kernal  = readFile(argv[5]); }
    if (argc > 6) { basic   = readFile(argv[6]); }
    if (argc > 7) { chargen = readFile(argv[7]); }
    engine.setRoms(kernal.empty()  ? nullptr : kernal.data(),
                   basic.empty()   ? nullptr : basic.data(),
                   chargen.empty() ? nullptr : chargen.data());

    SIDLiteBuilder builder("sidlite");

    SidConfig cfg;
    cfg.frequency    = 44100;
    cfg.samplingMethod = SidConfig::INTERPOLATE;
    cfg.sidEmulation = &builder;
    // Force PAL + 6581 + 6526 to match the corpus dump host as closely as
    // possible; tune flags still override model when forced=false below.
    cfg.defaultC64Model  = SidConfig::PAL;
    cfg.defaultSidModel  = SidConfig::MOS6581;
    cfg.forceC64Model    = false;
    cfg.forceSidModel    = false;

    // DETERMINISM: libsidplayfp's default powerOnDelay is
    // SidConfig::DEFAULT_POWER_ON_DELAY == MAX_POWER_ON_DELAY+1, which is the
    // sentinel that makes Player::initialise() pick a *random* delay seeded from
    // std::time(nullptr) (see player.cpp). A different power-on delay means a
    // different number of warm-up clocks before the play routine runs, so the
    // boot lag and the whole trace length drift run-to-run (the .sidwr.bin size
    // was observed to vary by ~kB and occasionally collapse to almost nothing).
    // Pinning powerOnDelay to a fixed value <= MAX_POWER_ON_DELAY takes the
    // deterministic branch, so every run starts from byte-identical C64 state.
    // The C64 RAM power-up pattern (SystemRAMBank::reset) and the dangling-bus
    // LCG (MMU, fixed seed 3686734) are already deterministic, so this is the
    // sole remaining source of run-to-run variation. Overridable for
    // experiments via SIDTRACE_POWER_ON_DELAY.
    {
        unsigned long pod = 0;
        if (const char *e = getenv("SIDTRACE_POWER_ON_DELAY"))
            pod = strtoul(e, nullptr, 0);
        if (pod > SidConfig::MAX_POWER_ON_DELAY)
            pod = SidConfig::MAX_POWER_ON_DELAY;
        cfg.powerOnDelay = (uint_least16_t)pod;
    }

    if (!engine.config(cfg))
    {
        fprintf(stderr, "ERR config: %s\n", engine.error());
        return 1;
    }
    if (!engine.load(&tune))
    {
        fprintf(stderr, "ERR load: %s\n", engine.error());
        return 1;
    }

    const SidTuneInfo *ti = tune.getInfo();
    const SidInfo &si = engine.info();

    // PAL frame in CPU cycles. The CIA-timer IRQ cadence is what actually
    // delineates frames; we capture the full trace and the host segments it.
    const double cpuFreqHz = 985248.0;        // PAL
    const double frameHz    = 50.0;
    const long   cyclesPerFrame = (long)(cpuFreqHz / frameHz + 0.5);
    const long   totalCycles = (long)cyclesPerFrame * (long)nframes + cyclesPerFrame;

    MemBusTrace &tr = MemBusTrace::instance();
    tr.clear();
    tr.enabled = true;

    // The bus trace is no longer streamed cycle-by-cycle to disk (the old
    // <prefix>.bus.bin was GBs/tune and unusable at 60,000 tunes). Instead the
    // emulator DISTILLS the execution in-place (membus_trace.h: the access-type
    // map, the post-init RAM snapshot, the PC-tagged SID-write summary, and the
    // indexed-read VSA summary) and we emit ONE compact <prefix>.distill.bin at
    // the end. We still emit <prefix>.sidwr.bin (SID writes only -- small, one
    // burst/frame) because it is the render/residual-zero gate.
    const std::string sidPathOut     = outPrefix + ".sidwr.bin";
    const std::string distillPathOut = outPrefix + ".distill.bin";

    // Phase tag: the PSID driver runs the tune's init (subtune select + table
    // depack) during the FIRST play() chunk, then settles into the steady
    // per-frame loop. Bracket the first chunk as INIT, snapshot RAM at the
    // boundary (so the song bytes are captured verbatim, once), then PLAY.
    tr.phase = libsidplayfp::PHASE_INIT;

    long done = 0;
    const unsigned int chunk = 20000;   // libsidplayfp clamps play() to MAX_CYCLES
    bool snapped = false;
    while (done < totalCycles)
    {
        unsigned int want = (unsigned int)std::min<long>(chunk, totalCycles - done);
        int r = engine.play(want);
        if (r < 0)
        {
            fprintf(stderr, "WARN play halted at cycle %ld: %s\n", done, engine.error());
            break;
        }
        if (!snapped)
        {
            // init -> play boundary: snapshot the live RAM image (the depacked
            // song tables are now resident) and flip to the play phase.
            tr.snapshotRam();
            tr.phase = libsidplayfp::PHASE_PLAY;
            snapped = true;
        }
        done += want;
    }
    tr.enabled = false;
    if (!snapped) tr.snapshotRam();

    // <prefix>.sidwr.bin -- the timestamped SID-write stream, the render gate.
    // This is SMALL (one ~25-write burst per frame), unlike the retired
    // multi-GB full bus trace; it is held in the distiller's sidStream vector.
    {
        FILE *fs = fopen(sidPathOut.c_str(), "wb");
        for (const auto &e : tr.sidStream)
        {
            fwrite(&e.cycle, 8, 1, fs);
            fwrite(&e.addr, 2, 1, fs);
            fwrite(&e.reg, 1, 1, fs);
            fwrite(&e.val, 1, 1, fs);
        }
        fclose(fs);
    }
    const uint64_t nSid = (uint64_t)tr.sidStream.size();

    // Debug aid (off by default): dump the full post-init RAM image so the
    // zero-page-relocated players can be disassembled offline. Not part of the
    // SDST artifact.
    if (getenv("SIDTRACE_DUMP_RAM"))
    {
        const std::string rp = outPrefix + ".ram.bin";
        FILE *fr = fopen(rp.c_str(), "wb");
        if (fr) { fwrite(tr.ramSnapshot, 1, 65536, fr); fclose(fr); }
    }

    // Reclassify SIDDF memory leaves against the final access-type map (a state
    // cell first read on frame 1 before its play-write would otherwise look
    // read-only). Deterministic: depends only on the accumulated acc map.
    tr.finalizeLeaves();

    // --- emit the compact distilled artifact (SDST) ---
    const long distillBytes =
        emit_distill(distillPathOut.c_str(), tr, ti, subtune, nframes,
                     cyclesPerFrame, done);

    std::string metaPath = outPrefix + ".meta.txt";
    FILE *fm = fopen(metaPath.c_str(), "w");
    fprintf(fm, "sid=%s\n", sidPath);
    fprintf(fm, "format=%s\n", ti ? ti->formatString() : "?");
    fprintf(fm, "subtune=%d\n", subtune);
    fprintf(fm, "songs=%d\n", ti ? ti->songs() : 0);
    fprintf(fm, "init=0x%04x\n", ti ? ti->initAddr() : 0);
    fprintf(fm, "play=0x%04x\n", ti ? ti->playAddr() : 0);
    fprintf(fm, "load=0x%04x\n", ti ? ti->loadAddr() : 0);
    fprintf(fm, "speed=%s\n", si.speedString() ? si.speedString() : "?");
    fprintf(fm, "artifact=SDST\n");
    fprintf(fm, "nframes_requested=%d\n", nframes);
    fprintf(fm, "cycles_per_frame=%ld\n", cyclesPerFrame);
    fprintf(fm, "total_cycles=%ld\n", done);
    fprintf(fm, "n_sid_writes=%llu\n", (unsigned long long)nSid);
    fprintf(fm, "distill_bytes=%ld\n", distillBytes);
    fprintf(fm, "kernal=%s\n", kernal.empty() ? "none" : argv[5]);
    fclose(fm);

    fprintf(stderr, "OK %s: %llu sid writes, %ld-byte distill over %ld cycles\n",
            sidPath, (unsigned long long)nSid, distillBytes, done);
    return 0;
}
