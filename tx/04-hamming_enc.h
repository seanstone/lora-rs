#pragma once
#include <cstdint>
#include <vector>

// Encodes each nibble into a Hamming codeword of (cr+4) bits.
// The first (sf-2) nibbles always use cr_app=4 regardless of cr.
std::vector<uint8_t> hamming_enc(const std::vector<uint8_t>& nibbles,
                                  uint8_t cr, uint8_t sf);
