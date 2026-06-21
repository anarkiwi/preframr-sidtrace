/*
 * sidtrace - white-box SID recovery wrapper around instrumented libsidplayfp.
 *
 * Usage:
 *   sidtrace <file.sid> <subtune(1-based)> <nframes> <out_prefix> [kernal] [basic] [chargen]
 *
 * Emits:
 *   <out_prefix>.sidwr.bin   - SID register writes : records of
 *                              int64 cycle, uint16 addr, uint8 reg, uint8 val
 *   <out_prefix>.bus.bin     - full CPU bus trace   : records of
 *                              int64 cycle, uint16 addr, uint8 val, uint8 rw
 *   <out_prefix>.meta.txt    - tune metadata (init/play/load, speed, model, frame cycles)
 *
 * Frames are delineated by the host from the SID-write cycle clusters, the
 * same way the corpus register dump uses the IRQ cycle as the frame id.
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
using libsidplayfp::MemAccess;

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
    tr.reserve(1u << 16);   // ~one play-chunk; we stream+clear, never hold the whole trace
    tr.enabled = true;

    // Stream outputs to disk as we go: peak RAM = one play-chunk of accesses,
    // not the whole capture (the old buffer-everything path hit ~928 MB/60 s).
    // SID writes ($D400-$D7FF, any chip) split from the full bus trace.
    // SIDTRACE_NOBUS=1 skips the (large) bus trace for the cheap sidwr-only
    // fidelity / program-ground-truth pass.
    std::string sidPathOut = outPrefix + ".sidwr.bin";
    std::string busPathOut = outPrefix + ".bus.bin";
    const bool wantBus = (getenv("SIDTRACE_NOBUS") == nullptr);
    FILE *fs = fopen(sidPathOut.c_str(), "wb");
    FILE *fb = wantBus ? fopen(busPathOut.c_str(), "wb") : nullptr;

    uint64_t nSid = 0, nBus = 0;
    auto flush = [&]()
    {
        for (const MemAccess &a : tr.accesses)
        {
            if (fb)
            {
                fwrite(&a.cycle, 8, 1, fb);
                fwrite(&a.addr, 2, 1, fb);
                fwrite(&a.val, 1, 1, fb);
                fwrite(&a.rw, 1, 1, fb);
            }
            nBus++;
            if (a.rw == 1 && a.addr >= 0xD400 && a.addr <= 0xD7FF)
            {
                uint16_t addr = a.addr;
                uint8_t reg = (uint8_t)(a.addr & 0x1F);
                fwrite(&a.cycle, 8, 1, fs);
                fwrite(&addr, 2, 1, fs);
                fwrite(&reg, 1, 1, fs);
                fwrite(&a.val, 1, 1, fs);
                nSid++;
            }
        }
        tr.accesses.clear();
    };

    long done = 0;
    const unsigned int chunk = 20000;   // libsidplayfp clamps play() to MAX_CYCLES
    long stalls = 0;
    while (done < totalCycles)
    {
        unsigned int want = (unsigned int)std::min<long>(chunk, totalCycles - done);
        int r = engine.play(want);
        if (r < 0)
        {
            fprintf(stderr, "WARN play halted at cycle %ld: %s\n", done, engine.error());
            flush();
            break;
        }
        uint64_t before = nBus;
        flush();
        if (nBus == before) { if (++stalls == 3)
            fprintf(stderr, "WARN bus went idle at cycle %ld (call %ld)\n", done, done/chunk);
        } else stalls = 0;
        done += want;
    }
    tr.enabled = false;
    flush();

    fclose(fs);
    if (fb) fclose(fb);

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
    fprintf(fm, "nframes_requested=%d\n", nframes);
    fprintf(fm, "cycles_per_frame=%ld\n", cyclesPerFrame);
    fprintf(fm, "total_cycles=%ld\n", done);
    fprintf(fm, "n_sid_writes=%llu\n", (unsigned long long)nSid);
    fprintf(fm, "n_bus_accesses=%llu\n", (unsigned long long)nBus);
    fprintf(fm, "kernal=%s\n", kernal.empty() ? "none" : argv[5]);
    fclose(fm);

    fprintf(stderr, "OK %s: %llu sid writes, %llu bus accesses over %ld cycles\n",
            sidPath, (unsigned long long)nSid, (unsigned long long)nBus, done);
    return 0;
}
