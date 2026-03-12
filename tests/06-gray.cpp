#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../tx/06-gray_demap.h"
#include "../rx/06-gray_mapping.h"

static int passed = 0, failed = 0;

static void check(const char* name, bool ok) {
    printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
    ok ? passed++ : failed++;
}

// gray_demap(x): fold-XOR, known values
static void test_gray_demap_values() {
    uint8_t sf = 8;
    // fold_xor is the standard Gray decode: gray_decode(x^(x>>1)) = x
    // so gray_demap(gray_encode(x)) should equal x
    struct { uint32_t in, out; } cases[] = {
        {0, 0}, {1, 1}, {3, 2}, {2, 3}, {6, 4}, {4, 7}, {128, 255}
    };
    bool ok = true;
    for (auto& c : cases) {
        auto r = gray_demap({c.in}, sf);
        if (r[0] != c.out) { ok = false; break; }
    }
    check("gray_demap: known values (Gray decode)", ok);
}

// gray_map(x): x ^ (x>>1), known values
static void test_gray_map_values() {
    uint8_t sf = 8;
    struct { uint32_t in, out; } cases[] = {
        {0, 0}, {1, 1}, {2, 3}, {3, 2}, {4, 6}, {7, 4}, {255, 128}
    };
    bool ok = true;
    for (auto& c : cases) {
        auto r = gray_map({c.in}, sf);
        if (r[0] != c.out) { ok = false; break; }
    }
    check("gray_map: known values (Gray encode)", ok);
}

// Perfect roundtrip: gray_map(gray_demap(x)) == x
static void test_roundtrip_sf(uint8_t sf) {
    uint32_t n = 1u << sf;
    bool ok = true;
    for (uint32_t x = 0; x < n; x++) {
        auto d = gray_demap({x}, sf);
        auto r = gray_map(d, sf);
        if (r[0] != x) { ok = false; break; }
    }
    char name[64];
    snprintf(name, sizeof(name), "gray_map(gray_demap(x)) == x  (sf=%d)", sf);
    check(name, ok);
}

// Perfect roundtrip in reverse: gray_demap(gray_map(x)) == x
static void test_roundtrip_rev_sf(uint8_t sf) {
    uint32_t n = 1u << sf;
    bool ok = true;
    for (uint32_t x = 0; x < n; x++) {
        auto g = gray_map({x}, sf);
        auto r = gray_demap(g, sf);
        if (r[0] != x) { ok = false; break; }
    }
    char name[64];
    snprintf(name, sizeof(name), "gray_demap(gray_map(x)) == x  (sf=%d)", sf);
    check(name, ok);
}

// All outputs in [0, 2^sf)
static void test_range(uint8_t sf) {
    uint32_t n = 1u << sf;
    std::vector<uint32_t> syms(n);
    for (uint32_t i = 0; i < n; i++) syms[i] = i;
    auto gm = gray_map(syms, sf);
    auto gd = gray_demap(syms, sf);
    bool ok = true;
    for (uint32_t i = 0; i < n; i++)
        if (gm[i] >= n || gd[i] >= n) { ok = false; break; }
    char name[64];
    snprintf(name, sizeof(name), "outputs in [0, 2^sf)  (sf=%d)", sf);
    check(name, ok);
}

// gray_map is a bijection
static void test_bijection(uint8_t sf) {
    uint32_t n = 1u << sf;
    std::vector<uint32_t> syms(n);
    for (uint32_t i = 0; i < n; i++) syms[i] = i;
    auto gm = gray_map(syms, sf);
    std::vector<bool> seen(n, false);
    bool ok = true;
    for (uint32_t v : gm) {
        if (seen[v]) { ok = false; break; }
        seen[v] = true;
    }
    char name[64];
    snprintf(name, sizeof(name), "gray_map is bijection  (sf=%d)", sf);
    check(name, ok);
}

int main() {
    test_gray_demap_values();
    test_gray_map_values();

    for (uint8_t sf : {5, 6, 7, 8, 9, 10, 11, 12}) {
        test_roundtrip_sf(sf);
        test_roundtrip_rev_sf(sf);
        test_range(sf);
        test_bijection(sf);
    }

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
