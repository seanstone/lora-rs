#include "06-gray_demap.h"

std::vector<uint32_t> gray_demap(const std::vector<uint32_t>& symbols, uint8_t sf) {
    std::vector<uint32_t> out;
    out.reserve(symbols.size());
    for (uint32_t y : symbols) {
        uint32_t r = y;
        for (int j = 1; j < sf; j++)
            r ^= (y >> j);
        out.push_back(r);
    }
    return out;
}
