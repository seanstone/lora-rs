/// Encode each nibble to a (cr+4)-bit Hamming codeword.
/// The first `sf-2` nibbles use cr_app=4 (header); the rest use `cr`.
pub fn hamming_enc(nibbles: &[u8], cr: u8, sf: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(nibbles.len());
    for (cnt, &d) in nibbles.iter().enumerate() {
        let cr_app = if cnt < (sf - 2) as usize { 4 } else { cr };
        let b = [
            (d >> 3) & 1 != 0, // b3 = MSB
            (d >> 2) & 1 != 0, // b2
            (d >> 1) & 1 != 0, // b1
             d       & 1 != 0, // b0 = LSB
        ];

        let codeword = if cr_app == 1 {
            // CR 4/5: data + overall parity
            let p = b[0] ^ b[1] ^ b[2] ^ b[3];
            (b[3] as u8) << 4 | (b[2] as u8) << 3 | (b[1] as u8) << 2 | (b[0] as u8) << 1 | p as u8
        } else {
            // CR 4/6..4/8: Hamming parity bits p0..p3
            let p0 = b[3] ^ b[2] ^ b[1];
            let p1 = b[2] ^ b[1] ^ b[0];
            let p2 = b[3] ^ b[2] ^ b[0];
            let p3 = b[3] ^ b[1] ^ b[0];
            let full = (b[3] as u8) << 7 | (b[2] as u8) << 6 | (b[1] as u8) << 5 | (b[0] as u8) << 4
                     | (p0 as u8) << 3 | (p1 as u8) << 2 | (p2 as u8) << 1 | p3 as u8;
            full >> (4 - cr_app)
        };
        out.push(codeword);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rx::hamming_dec::hamming_dec;

    fn roundtrip_all_nibbles(cr: u8, sf: u8) {
        let nibbles: Vec<u8> = (0..16).collect();
        let cw = hamming_enc(&nibbles, cr, sf);
        let dec = hamming_dec(&cw, cr, sf);
        assert_eq!(dec, nibbles, "cr={cr} sf={sf}");
    }

    #[test] fn roundtrip_cr1() { roundtrip_all_nibbles(1, 7); }
    #[test] fn roundtrip_cr2() { roundtrip_all_nibbles(2, 7); }
    #[test] fn roundtrip_cr3() { roundtrip_all_nibbles(3, 7); }
    #[test] fn roundtrip_cr4() { roundtrip_all_nibbles(4, 7); }

    #[test]
    fn single_bit_correction_cr3() {
        let nibbles = vec![0b1010u8];
        let sf = 7;
        let mut cw = hamming_enc(&nibbles, 3, sf);
        cw[0] ^= 0b001; // flip LSB (a parity bit position)
        // decoder should still recover the nibble
        let dec = hamming_dec(&cw, 3, sf);
        assert_eq!(dec[0], nibbles[0]);
    }

    #[test]
    fn single_bit_correction_cr4() {
        let nibbles = vec![0b1100u8];
        let sf = 7;
        let mut cw = hamming_enc(&nibbles, 4, sf);
        cw[0] ^= 1; // flip LSB
        let dec = hamming_dec(&cw, 4, sf);
        assert_eq!(dec[0], nibbles[0]);
    }
}
