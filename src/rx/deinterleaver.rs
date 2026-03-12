fn imod(a: i32, b: i32) -> i32 {
    a.rem_euclid(b)
}

/// Inverse permutation: recover `sf_app` codewords from `cw_len` symbols.
fn consume_block(symbols: &[u32], pos: usize, sf_app: i32, cw_len: i32, codewords: &mut Vec<u8>) {
    for row in 0..sf_app {
        let mut cw: u8 = 0;
        for i in 0..cw_len {
            let j   = imod(i - row - 1, sf_app);
            let bit = (symbols[pos + i as usize] >> (sf_app - 1 - j)) & 1;
            cw |= (bit as u8) << (cw_len - 1 - i);
        }
        codewords.push(cw);
    }
}

/// Deinterleave LoRa chirp symbols back to Hamming codewords.
pub fn deinterleave(symbols: &[u32], cr: u8, sf: u8, ldro: bool) -> Vec<u8> {
    let mut codewords = Vec::new();
    let mut pos = 0usize;
    let n = symbols.len();

    if pos + 8 <= n {
        consume_block(symbols, pos, (sf - 2) as i32, 8, &mut codewords);
        pos += 8;
    }

    let pay_sf  = if ldro { (sf - 2) as i32 } else { sf as i32 };
    let pay_len = (cr + 4) as i32;
    while pos + pay_len as usize <= n {
        consume_block(symbols, pos, pay_sf, pay_len, &mut codewords);
        pos += pay_len as usize;
    }

    codewords
}
