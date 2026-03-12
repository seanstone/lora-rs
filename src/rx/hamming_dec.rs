/// Decode each (cr+4)-bit Hamming codeword back to a nibble.
/// The first `sf-2` codewords use cr_app=4 (header); the rest use `cr`.
pub fn hamming_dec(codewords: &[u8], cr: u8, sf: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(codewords.len());
    for (cnt, &cw) in codewords.iter().enumerate() {
        let cr_app = if cnt < (sf - 2) as usize { 4 } else { cr };
        let cw_len = cr_app + 4;

        // bit(i): i=0 is MSB of the codeword.
        // Codeword layout (encoder): [b0(nibble LSB), b1, b2, b3(nibble MSB), p0, p1, ...]
        let bit = |i: u8| -> bool { (cw >> (cw_len - 1 - i)) & 1 != 0 };

        // b[0]=b0(LSB)=bit(0), b[1]=b1, b[2]=b2, b[3]=b3(MSB)=bit(3)
        let mut b = [bit(0), bit(1), bit(2), bit(3)];

        if cr_app == 1 {
            out.push((b[3] as u8) << 3 | (b[2] as u8) << 2 | (b[1] as u8) << 1 | b[0] as u8);
            continue;
        }

        if cr_app == 4 {
            let ones: u32 = (0..cw_len).map(|k| bit(k) as u32).sum();
            if ones % 2 == 0 {
                out.push((b[3] as u8) << 3 | (b[2] as u8) << 2 | (b[1] as u8) << 1 | b[0] as u8);
                continue;
            }
        }

        if cr_app >= 3 {
            let s0 = bit(0) ^ bit(1) ^ bit(2) ^ bit(4);
            let s1 = bit(1) ^ bit(2) ^ bit(3) ^ bit(5);
            let s2 = bit(0) ^ bit(1) ^ bit(3) ^ bit(6);
            match s0 as u8 | (s1 as u8) << 1 | (s2 as u8) << 2 {
                5 => b[0] ^= true,
                7 => b[1] ^= true,
                3 => b[2] ^= true,
                6 => b[3] ^= true,
                _ => {}
            }
        }

        out.push((b[3] as u8) << 3 | (b[2] as u8) << 2 | (b[1] as u8) << 1 | b[0] as u8);
    }
    out
}
