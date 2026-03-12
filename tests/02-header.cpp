#include <string>
#include <cassert>
#include <cstdio>
#include "../tx/01-whitening.h"
#include "../tx/02-header.h"
#include "../rx/01-dewhitening.h"
#include "../rx/02-header_decoder.h"

// --- add_header unit tests ---

static void test_add_header() {
    std::vector<uint8_t> payload = {'H', 'e', 'l', 'l', 'o'};
    auto nibbles = whiten(payload);

    {
        auto out = add_header(nibbles, false, true, 1);
        assert(out.size() == 5 + nibbles.size());
        assert(((out[0] << 4) | out[1]) == (uint8_t)payload.size());
        assert(out[2] == ((1 << 1) | 1));
        for (size_t i = 0; i < nibbles.size(); i++)
            assert(out[5 + i] == nibbles[i]);
        printf("  PASS  add_header: length, cr/crc field, payload nibbles\n");
    }
    {
        auto out = add_header(nibbles, true, true, 1);
        assert(out == nibbles);
        printf("  PASS  add_header: implicit passthrough\n");
    }
    {
        auto out = add_header(nibbles, false, false, 4);
        for (auto b : out) assert(b <= 0x0F);
        printf("  PASS  add_header: all nibbles in [0,15]\n");
    }
}

// --- decode_header unit tests ---

static void test_decode_header() {
    std::vector<uint8_t> payload = {'H', 'e', 'l', 'l', 'o'};
    auto nibbles = whiten(payload);
    auto frame   = add_header(nibbles, false, true, 1);

    {
        auto info = decode_header(frame, false);
        assert(info.valid);
        assert(info.payload_len == payload.size());
        assert(info.cr      == 1);
        assert(info.has_crc == true);
        assert(info.payload_nibbles == nibbles);
        printf("  PASS  decode_header: fields and nibbles correct\n");
    }
    {
        auto corrupted = frame;
        corrupted[4] ^= 0x01;
        assert(!decode_header(corrupted, false).valid);
        printf("  PASS  decode_header: checksum corruption detected\n");
    }
    {
        auto info = decode_header(nibbles, true, 1, payload.size(), true);
        assert(info.valid && info.payload_nibbles == nibbles);
        printf("  PASS  decode_header: implicit header passthrough\n");
    }
}

// --- full tx-rx roundtrips ---

static void test_roundtrip(const std::string& label,
                           const std::vector<uint8_t>& payload,
                           uint8_t cr, bool has_crc)
{
    auto info = decode_header(add_header(whiten(payload), false, has_crc, cr), false);
    assert(info.valid);
    assert(info.payload_len == payload.size());
    assert(info.cr == cr && info.has_crc == has_crc);
    assert(dewhiten(info.payload_nibbles) == payload);
    printf("  PASS  roundtrip: %s\n", label.c_str());
}

int main() {
    test_add_header();
    test_decode_header();
    test_roundtrip("Hello, cr=1, crc=true",  {'H','e','l','l','o'}, 1, true);
    test_roundtrip("single byte, cr=4, crc=false", {0x42}, 4, false);
    test_roundtrip("longer payload, cr=2, crc=true",
        {'L','o','R','a',' ','T','X','-','R','X'}, 2, true);

    printf("All tests passed.\n");
    return 0;
}
