#include "06-gray_mapping.h"

std::vector<uint32_t> gray_map(const std::vector<uint32_t>& symbols, uint8_t sf) {
    (void)sf;
    std::vector<uint32_t> out;
    out.reserve(symbols.size());
    for (uint32_t x : symbols)
        out.push_back(x ^ (x >> 1));
    return out;
}
