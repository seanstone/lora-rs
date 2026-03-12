#pragma once
#include <cstdint>
#include <vector>

// Decodes each Hamming codeword back to a nibble (hard decoding).
// Single-bit error correction for cr>=3; detection only for cr<=2.
// The first (sf-2) codewords are decoded with cr_app=4.
std::vector<uint8_t> hamming_dec(const std::vector<uint8_t>& codewords,
                                  uint8_t cr, uint8_t sf);
