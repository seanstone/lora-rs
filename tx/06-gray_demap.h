#pragma once
#include <cstdint>
#include <vector>

// Applies Gray decoding to each LoRa symbol before modulation (TX side).
// out = x ^ (x>>1) ^ (x>>2) ^ ... ^ (x>>(sf-1))  — pure fold-XOR, no shift.
// Matches the gr-lora_sdr gray_demap block position in the TX flowgraph.
std::vector<uint32_t> gray_demap(const std::vector<uint32_t>& symbols, uint8_t sf);
