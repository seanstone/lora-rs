#include "04-hamming_dec.h"

std::vector<uint8_t> hamming_dec(const std::vector<uint8_t>& codewords,
                                  uint8_t cr, uint8_t sf) {
    std::vector<uint8_t> out;
    out.reserve(codewords.size());

    for (size_t cnt = 0; cnt < codewords.size(); cnt++) {
        uint8_t cr_app = (cnt < (size_t)(sf - 2)) ? 4 : cr;
        uint8_t cw_len = cr_app + 4;
        uint8_t cw = codewords[cnt];

        // bit(i): position i (0=MSB) within the cw_len-bit codeword
        auto bit = [&](int i) -> bool {
            return (cw >> (cw_len - 1 - i)) & 1;
        };

        // Data bits: [b0, b1, b2, b3, p...]  where b3=MSB of nibble, b0=LSB
        bool b0 = bit(0), b1 = bit(1), b2 = bit(2), b3 = bit(3);

        if (cr_app == 1) {
            // CR 4/5: parity-only, no correction
            out.push_back((b3 << 3) | (b2 << 2) | (b1 << 1) | b0);
            continue;
        }

        if (cr_app == 4) {
            // Don't correct if even number of bits are 1 (uncorrectable multi-bit error)
            int ones = 0;
            for (int k = 0; k < cw_len; k++) ones += bit(k);
            if (!(ones % 2)) {
                out.push_back((b3 << 3) | (b2 << 2) | (b1 << 1) | b0);
                continue;
            }
        }

        if (cr_app >= 3) {
            bool s0 = bit(0) ^ bit(1) ^ bit(2) ^ bit(4);
            bool s1 = bit(1) ^ bit(2) ^ bit(3) ^ bit(5);
            bool s2 = bit(0) ^ bit(1) ^ bit(3) ^ bit(6);
            switch ((int)s0 | ((int)s1 << 1) | ((int)s2 << 2)) {
                case 5: b0 ^= 1; break;
                case 7: b1 ^= 1; break;
                case 3: b2 ^= 1; break;
                case 6: b3 ^= 1; break;
                default: break;
            }
        }
        // cr_app==2: syndrome detectable but uncorrectable — output data bits as-is

        // Rebuild nibble MSB-first: {b3, b2, b1, b0}
        out.push_back((b3 << 3) | (b2 << 2) | (b1 << 1) | b0);
    }
    return out;
}
