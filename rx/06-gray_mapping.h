#pragma once
#include <cstdint>
#include <vector>

// Applies Gray encoding to each demodulated symbol (RX side).
// out = x ^ (x >> 1)
// Matches the gr-lora_sdr gray_mapping block position in the RX flowgraph.
std::vector<uint32_t> gray_map(const std::vector<uint32_t>& symbols, uint8_t sf);
