# sidtrace - build the instrumented libsidplayfp + the sidtrace recorder.
#
# A fresh clone needs the submodule:
#     git submodule update --init --recursive
# then:
#     make            # builds build/sidtrace
#     make test       # builds + runs the pytest smoke/structural tests
#     make clean      # removes build/ and the in-tree libsidplayfp objects
#     make distclean  # also reverts the applied patch + configure artifacts
#
# The dependency on a *patched* libsidplayfp is carried as:
#   - third_party/libsidplayfp : git submodule pinned to a known upstream SHA
#   - patches/0001-membus-trace-instrumentation.patch : the bus-trace hook,
#     applied idempotently into the submodule at build time.

LIBDIR        := third_party/libsidplayfp
PATCH         := patches/0001-membus-trace-instrumentation.patch
PATCH_STAMP   := $(LIBDIR)/.membus_patch_applied
CONFIG_STAMP  := $(LIBDIR)/.configured
LIB_A         := $(LIBDIR)/src/.libs/libsidplayfp.a
SIDLITE_A     := $(LIBDIR)/src/builders/sidlite-builder/.libs/libsidplayfp-sidlite.a

BUILD         := build
BIN           := $(BUILD)/sidtrace

CXX           ?= g++
CXXFLAGS      ?= -O2 -std=c++17
SIDTRACE_CXXFLAGS := $(CXXFLAGS) -DHAVE_CONFIG_H -I$(LIBDIR) -I$(LIBDIR)/src
LIBS          := -lpthread -lm

NPROC         := $(shell nproc 2>/dev/null || echo 2)

.PHONY: all test clean distclean lib check-submodule

all: $(BIN)

check-submodule:
	@if [ ! -f "$(LIBDIR)/configure.ac" ]; then \
	  echo "ERROR: submodule $(LIBDIR) is empty."; \
	  echo "Run: git submodule update --init --recursive"; \
	  exit 1; \
	fi

# 1. Apply the bus-trace patch into the pinned submodule (idempotent).
$(PATCH_STAMP): $(PATCH) | check-submodule
	@if git -C $(LIBDIR) apply --reverse --check $(abspath $(PATCH)) >/dev/null 2>&1; then \
	  echo "patch already applied"; \
	else \
	  git -C $(LIBDIR) apply $(abspath $(PATCH)) && echo "applied $(PATCH)"; \
	fi
	@touch $@

# 2. Regenerate the autotools build system + configure (static libs only).
$(CONFIG_STAMP): $(PATCH_STAMP)
	cd $(LIBDIR) && autoreconf -i
	cd $(LIBDIR) && ./configure --without-exsid --without-usbsid \
	  --disable-shared --enable-static
	@touch $@

# 3. Build the static libsidplayfp + sidlite builder libs.
$(LIB_A) $(SIDLITE_A): $(CONFIG_STAMP)
	$(MAKE) -C $(LIBDIR) -j$(NPROC)

lib: $(LIB_A) $(SIDLITE_A)

# 4. Compile the sidtrace recorder against the static libs.
$(BIN): src/sidtrace.cpp $(LIB_A) $(SIDLITE_A)
	@mkdir -p $(BUILD)
	$(CXX) $(SIDTRACE_CXXFLAGS) src/sidtrace.cpp \
	  $(LIB_A) $(SIDLITE_A) $(LIBS) -o $@
	@echo "built $(BIN)"

test: $(BIN)
	SIDTRACE_BIN=$(abspath $(BIN)) python3 -m pytest tests/ -v

clean:
	rm -rf $(BUILD)
	-$(MAKE) -C $(LIBDIR) clean 2>/dev/null || true

# Revert to a pristine submodule (drop patch + configure artifacts).
distclean: clean
	-cd $(LIBDIR) && git clean -fdx >/dev/null 2>&1 || true
	-cd $(LIBDIR) && git checkout -- . >/dev/null 2>&1 || true
	rm -f $(PATCH_STAMP) $(CONFIG_STAMP)
