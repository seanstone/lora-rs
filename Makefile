.DEFAULT_GOAL := all

CXX      = c++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -MMD -MP

BINDIR   = bin
BUILDDIR = build

# Shared sources — compiled into objects linked by every test
LIB_SRCS = tx/01-whitening.cpp tx/02-header.cpp tx/03-crc.cpp tx/04-hamming_enc.cpp \
           rx/01-dewhitening.cpp rx/02-header_decoder.cpp rx/03-crc_verif.cpp rx/04-hamming_dec.cpp
LIB_OBJS = $(patsubst %.cpp,$(BUILDDIR)/%.o,$(LIB_SRCS))

# Each tests/*.cpp becomes its own binary in bin/
TEST_SRCS = $(wildcard tests/*.cpp)
TEST_BINS = $(patsubst tests/%.cpp,$(BINDIR)/%,$(TEST_SRCS))
TEST_OBJS = $(patsubst tests/%.cpp,$(BUILDDIR)/tests/%.o,$(TEST_SRCS))

DEPS = $(LIB_OBJS:.o=.d) $(TEST_OBJS:.o=.d)

-include $(DEPS)

all: $(TEST_BINS)

$(BINDIR)/%: $(BUILDDIR)/tests/%.o $(LIB_OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILDDIR)/%.o: %.cpp | $(BUILDDIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BINDIR):
	mkdir -p $(BINDIR)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

run: all
	@for t in $(TEST_BINS); do echo "--- $$t ---"; ./$$t; done

clean:
	rm -rf $(BINDIR) $(BUILDDIR)

.PRECIOUS: $(BUILDDIR)/%.o
.PHONY: all run clean
