#include "04-hamming_enc.h"

std::vector<uint8_t> hamming_enc(const std::vector<uint8_t>& nibbles,
                                  uint8_t cr, uint8_t sf) {
    std::vector<uint8_t> out;
    out.reserve(nibbles.size());

    for (size_t cnt = 0; cnt < nibbles.size(); cnt++) {
        uint8_t cr_app = (cnt < (size_t)(sf - 2)) ? 4 : cr;
        uint8_t d = nibbles[cnt];

        // Extract bits; nibble is MSB-first so b3=MSB, b0=LSB
        bool b3 = (d >> 3) & 1;
        bool b2 = (d >> 2) & 1;
        bool b1 = (d >> 1) & 1;
        bool b0 =  d       & 1;

        uint8_t codeword;
        if (cr_app == 1) {
            // CR 4/5: append single overall parity bit
            bool p = b3 ^ b2 ^ b1 ^ b0;
            codeword = (b0 << 4) | (b1 << 3) | (b2 << 2) | (b3 << 1) | p;
        } else {
            // CR 4/6 .. 4/8: Hamming parity bits p0..p3
            bool p0 = b0 ^ b1 ^ b2;
            bool p1 = b1 ^ b2 ^ b3;
            bool p2 = b0 ^ b1 ^ b3;
            bool p3 = b0 ^ b2 ^ b3;
            uint8_t full = (b0 << 7) | (b1 << 6) | (b2 << 5) | (b3 << 4)
                         | (p0 << 3) | (p1 << 2) | (p2 << 1) | p3;
            // Crop to (cr_app+4) bits by right-shifting off unused parity bits
            codeword = full >> (4 - cr_app);
        }
        out.push_back(codeword);
    }
    return out;
}
