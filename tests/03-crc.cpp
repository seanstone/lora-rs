#include <string>
#include <cassert>
#include <cstdio>
#include "../tx/01-whitening.h"
#include "../tx/02-header.h"
#include "../tx/03-crc.h"
#include "../rx/01-dewhitening.h"
#include "../rx/02-header_decoder.h"
#include "../rx/03-crc_verif.h"

// Full TX-RX pipeline:
//   whiten → add_header → add_crc  /  decode_header → dewhiten → verify_crc
//
// CRC nibbles are raw (unwhitened) and appended after the framed nibbles.
// On RX, payload_nibbles and crc_nibbles are separated using payload_len from
// the decoded header before dewhitening.

static void test_roundtrip(const std::string& label,
                           const std::vector<uint8_t>& payload,
                           uint8_t cr, bool has_crc)
{
    // TX
    auto frame = add_crc(add_header(whiten(payload), false, has_crc, cr),
                         payload, has_crc);

    // RX: decode header
    auto info = decode_header(frame, false);
    assert(info.valid);

    // Split: whitened payload nibbles | raw CRC nibbles
    size_t n_pay = info.payload_len * 2;
    std::vector<uint8_t> payload_nibbles(info.payload_nibbles.begin(),
                                         info.payload_nibbles.begin() + n_pay);
    std::vector<uint8_t> crc_nibbles   (info.payload_nibbles.begin() + n_pay,
                                         info.payload_nibbles.end());

    auto recovered = dewhiten(payload_nibbles);
    assert(recovered == payload);

    if (info.has_crc) {
        assert(crc_nibbles.size() == 4);
        assert(verify_crc(recovered, crc_nibbles));
    }

    printf("  PASS  roundtrip: %s\n", label.c_str());
}

int main() {
    // --- add_crc unit tests ---
    {
        std::vector<uint8_t> payload = {'H', 'e', 'l', 'l', 'o'};
        auto frame   = add_header(whiten(payload), false, true, 1);
        auto framed  = add_crc(frame, payload, true);
        assert(framed.size() == frame.size() + 4);
        // first bytes unchanged
        for (size_t i = 0; i < frame.size(); i++) assert(framed[i] == frame[i]);
        // all nibbles in [0,15]
        for (auto b : framed) assert(b <= 0x0F);
        printf("  PASS  add_crc: appends 4 CRC nibbles, all in [0,15]\n");
    }
    {
        std::vector<uint8_t> payload = {'A', 'B', 'C', 'D'};
        auto frame  = add_header(whiten(payload), false, false, 1);
        assert(add_crc(frame, payload, false) == frame);
        printf("  PASS  add_crc: no-op when has_crc=false\n");
    }

    // --- verify_crc unit tests ---
    {
        std::vector<uint8_t> payload = {'H', 'e', 'l', 'l', 'o'};
        auto frame      = add_crc(add_header(whiten(payload), false, true, 1), payload, true);
        auto info       = decode_header(frame, false);
        size_t n        = info.payload_len * 2;
        auto pay_nib    = std::vector<uint8_t>(info.payload_nibbles.begin(), info.payload_nibbles.begin() + n);
        auto crc_nib    = std::vector<uint8_t>(info.payload_nibbles.begin() + n, info.payload_nibbles.end());
        auto recovered  = dewhiten(pay_nib);
        assert(verify_crc(recovered, crc_nib));
        printf("  PASS  verify_crc: valid CRC accepted\n");
    }
    {
        std::vector<uint8_t> payload = {'H', 'e', 'l', 'l', 'o'};
        auto frame   = add_crc(add_header(whiten(payload), false, true, 1), payload, true);
        frame.back() ^= 0x0F;  // corrupt last CRC nibble
        auto info    = decode_header(frame, false);
        size_t n     = info.payload_len * 2;
        auto pay_nib = std::vector<uint8_t>(info.payload_nibbles.begin(), info.payload_nibbles.begin() + n);
        auto crc_nib = std::vector<uint8_t>(info.payload_nibbles.begin() + n, info.payload_nibbles.end());
        assert(!verify_crc(dewhiten(pay_nib), crc_nib));
        printf("  PASS  verify_crc: corrupted CRC rejected\n");
    }

    // --- full tx-rx roundtrips ---
    test_roundtrip("Hello, cr=1, crc=true",          {'H','e','l','l','o'}, 1, true);
    test_roundtrip("longer payload, cr=2, crc=true",  {'L','o','R','a',' ','T','X','-','R','X'}, 2, true);
    test_roundtrip("no CRC, cr=1",                    {'A','B','C','D'}, 1, false);

    printf("All tests passed.\n");
    return 0;
}
