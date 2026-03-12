#include <string>
#include <cassert>
#include <cstdio>
#include "../tx/01-whitening.h"
#include "../tx/02-header.h"
#include "../tx/03-crc.h"
#include "../tx/04-hamming_enc.h"
#include "../rx/01-dewhitening.h"
#include "../rx/02-header_decoder.h"
#include "../rx/03-crc_verif.h"
#include "../rx/04-hamming_dec.h"

// Full TX-RX pipeline:
//   whiten → add_header → add_crc → hamming_enc
//   hamming_dec → decode_header → dewhiten → verify_crc

static void test_roundtrip(const std::string& label,
                           const std::vector<uint8_t>& payload,
                           uint8_t cr, uint8_t sf, bool has_crc)
{
    // TX
    auto frame      = add_crc(add_header(whiten(payload), false, has_crc, cr),
                               payload, has_crc);
    auto codewords  = hamming_enc(frame, cr, sf);

    // RX
    auto decoded    = hamming_dec(codewords, cr, sf);
    auto info       = decode_header(decoded, false);
    assert(info.valid);

    size_t n_pay    = info.payload_len * 2;
    auto pay_nib    = std::vector<uint8_t>(info.payload_nibbles.begin(),
                                           info.payload_nibbles.begin() + n_pay);
    auto crc_nib    = std::vector<uint8_t>(info.payload_nibbles.begin() + n_pay,
                                           info.payload_nibbles.end());
    auto recovered  = dewhiten(pay_nib);
    assert(recovered == payload);
    if (info.has_crc) assert(verify_crc(recovered, crc_nib));

    printf("  PASS  roundtrip: %s\n", label.c_str());
}

// --- hamming_enc / hamming_dec unit tests ---

static void test_enc_dec() {
    const uint8_t sf = 7;

    for (uint8_t cr = 1; cr <= 4; cr++) {
        // Encode all 16 nibble values and verify perfect roundtrip
        std::vector<uint8_t> nibbles;
        for (int n = 0; n < 16; n++) nibbles.push_back((uint8_t)n);

        auto cw  = hamming_enc(nibbles, cr, sf);
        auto dec = hamming_dec(cw, cr, sf);
        assert(dec == nibbles);
        printf("  PASS  enc/dec all nibbles: cr=%d\n", cr);
    }
}

static void test_error_correction() {
    const uint8_t sf = 7;

    for (uint8_t cr = 3; cr <= 4; cr++) {
        std::vector<uint8_t> nibbles = {0x5, 0xA, 0xF, 0x3};
        auto cw = hamming_enc(nibbles, cr, sf);

        // Flip one bit in each codeword and verify correction
        for (size_t i = 0; i < cw.size(); i++) {
            auto corrupted = cw;
            corrupted[i] ^= 1;  // flip LSB
            auto dec = hamming_dec(corrupted, cr, sf);
            assert(dec[i] == nibbles[i]);
        }
        printf("  PASS  single-bit error correction: cr=%d\n", cr);
    }
}

int main() {
    test_enc_dec();
    test_error_correction();

    test_roundtrip("Hello, cr=1, sf=7, crc=true",
        {'H','e','l','l','o'}, 1, 7, true);
    test_roundtrip("longer payload, cr=4, sf=9, crc=true",
        {'L','o','R','a',' ','T','X','-','R','X'}, 4, 9, true);
    test_roundtrip("no CRC, cr=2, sf=8",
        {'A','B','C','D'}, 2, 8, false);

    printf("All tests passed.\n");
    return 0;
}
