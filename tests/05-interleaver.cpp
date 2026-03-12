#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../tx/05-interleaver.h"
#include "../rx/05-deinterleaver.h"

static bool eq(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] != b[i]) return false;
    return true;
}

static int passed = 0, failed = 0;

static void check(const char* name, bool ok) {
    printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
    ok ? passed++ : failed++;
}

// Build a codeword sequence: header block (sf-2 codewords) + N payload blocks.
static std::vector<uint8_t> make_codewords(uint8_t cr, uint8_t sf, int n_payload_blocks,
                                            bool ldro = false) {
    int pay_sf   = ldro ? sf - 2 : sf;
    int pay_len  = cr + 4;
    int total    = (sf - 2) + n_payload_blocks * pay_sf;
    std::vector<uint8_t> cw(total);
    for (int i = 0; i < total; i++)
        cw[i] = (uint8_t)(i * 17 + 3) & 0xFF;
    // Mask to valid codeword width
    int hdr_cw_len = 8;
    for (int i = 0; i < sf - 2; i++)
        cw[i] &= (1 << hdr_cw_len) - 1;
    for (int i = sf - 2; i < total; i++)
        cw[i] &= (1 << pay_len) - 1;
    return cw;
}

static void test_roundtrip(const char* name, uint8_t cr, uint8_t sf,
                            int n_payload_blocks, bool ldro = false) {
    auto cw  = make_codewords(cr, sf, n_payload_blocks, ldro);
    auto sym = interleave(cw, cr, sf, ldro);
    auto rec = deinterleave(sym, cr, sf, ldro);
    check(name, eq(cw, rec));
}

// Verify interleaver produces expected symbol count.
static void test_symbol_count(const char* name, uint8_t cr, uint8_t sf,
                               int n_payload_blocks, bool ldro = false) {
    auto cw       = make_codewords(cr, sf, n_payload_blocks, ldro);
    auto sym      = interleave(cw, cr, sf, ldro);
    size_t expect = 8 + (size_t)n_payload_blocks * (cr + 4);
    check(name, sym.size() == expect);
}

// Test that a single bit flip in a symbol corrupts exactly the corresponding codeword.
static void test_single_bit_error(const char* name, uint8_t cr, uint8_t sf) {
    auto cw      = make_codewords(cr, sf, 1);
    auto sym     = interleave(cw, cr, sf);
    // Flip bit 0 of symbol 0 (header block)
    sym[0] ^= 1;
    auto rec     = deinterleave(sym, cr, sf);
    // The corruption in one symbol bit touches one bit in one codeword; rest should be clean
    // We only verify it didn't silently pass as identical
    bool differs = !eq(cw, rec);
    check(name, differs);
}

int main() {
    // Basic roundtrips at various SF/CR
    test_roundtrip("roundtrip sf7 cr1 1-payload-block",  1, 7, 1);
    test_roundtrip("roundtrip sf7 cr2 1-payload-block",  2, 7, 1);
    test_roundtrip("roundtrip sf7 cr4 1-payload-block",  4, 7, 1);
    test_roundtrip("roundtrip sf8 cr1 2-payload-blocks", 1, 8, 2);
    test_roundtrip("roundtrip sf8 cr4 2-payload-blocks", 4, 8, 2);
    test_roundtrip("roundtrip sf9 cr2 3-payload-blocks", 2, 9, 3);
    test_roundtrip("roundtrip sf12 cr4 1-payload-block", 4, 12, 1);

    // LDRO mode
    test_roundtrip("roundtrip sf10 cr2 ldro 2-blocks",   2, 10, 2, true);
    test_roundtrip("roundtrip sf12 cr4 ldro 1-block",    4, 12, 1, true);

    // Symbol count checks
    test_symbol_count("symbol_count sf7 cr1 1-block",    1, 7, 1);
    test_symbol_count("symbol_count sf8 cr4 3-blocks",   4, 8, 3);
    test_symbol_count("symbol_count sf12 cr2 2-blocks",  2, 12, 2);

    // Error detection
    test_single_bit_error("single_bit_error sf8 cr4",    4, 8);

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
