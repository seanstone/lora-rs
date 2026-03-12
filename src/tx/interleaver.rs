fn imod(a: i32, b: i32) -> i32 {
    a.rem_euclid(b)
}

/// Permute one block of `sf_app` codewords into `cw_len` symbols.
/// inter_bin[i][j] = cw_bin[mod(i-j-1, sf_app)][i]
fn emit_block(codewords: &[u8], pos: usize, sf_app: i32, cw_len: i32, symbols: &mut Vec<u32>) {
    for i in 0..cw_len {
        let mut sym: u32 = 0;
        for j in 0..sf_app {
            let row = imod(i - j - 1, sf_app) as usize;
            let bit = (codewords[pos + row] >> (cw_len - 1 - i)) & 1;
            sym |= (bit as u32) << (sf_app - 1 - j);
        }
        symbols.push(sym);
    }
}

/// Interleave Hamming codewords into LoRa chirp symbols.
/// Header block uses sf_app=sf-2, cw_len=8; payload uses sf_app=(ldro?sf-2:sf), cw_len=cr+4.
pub fn interleave(codewords: &[u8], cr: u8, sf: u8, ldro: bool) -> Vec<u32> {
    let mut symbols = Vec::new();
    let mut pos = 0usize;
    let n = codewords.len();

    if pos < n {
        emit_block(codewords, pos, (sf - 2) as i32, 8, &mut symbols);
        pos += (sf - 2) as usize;
    }

    let pay_sf  = if ldro { (sf - 2) as i32 } else { sf as i32 };
    let pay_len = (cr + 4) as i32;
    while pos < n {
        emit_block(codewords, pos, pay_sf, pay_len, &mut symbols);
        pos += pay_sf as usize;
    }

    symbols
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rx::deinterleaver::deinterleave;

    fn roundtrip(sf: u8, cr: u8, ldro: bool, n_payload_blocks: usize) {
        let header_cw  = (sf - 2) as usize;
        let payload_cw = (if ldro { sf - 2 } else { sf }) as usize * n_payload_blocks;
        let codewords: Vec<u8> = (0..(header_cw + payload_cw) as u8)
            .map(|i| i & 0xFF)
            .collect();
        let syms = interleave(&codewords, cr, sf, ldro);
        let recovered = deinterleave(&syms, cr, sf, ldro);
        assert_eq!(recovered, codewords, "sf={sf} cr={cr} ldro={ldro}");
    }

    #[test] fn rt_sf7_cr1()  { roundtrip(7, 1, false, 1); }
    #[test] fn rt_sf7_cr4()  { roundtrip(7, 4, false, 1); }
    #[test] fn rt_sf8_cr1()  { roundtrip(8, 1, false, 2); }
    #[test] fn rt_sf9_cr2()  { roundtrip(9, 2, false, 3); }
    #[test] fn rt_sf12_cr4() { roundtrip(12, 4, false, 1); }
    #[test] fn rt_ldro_sf10() { roundtrip(10, 2, true, 2); }
    #[test] fn rt_ldro_sf12() { roundtrip(12, 4, true, 1); }

    #[test]
    fn symbol_count_sf7_cr1() {
        let cw = vec![0u8; (7 - 2) + 7];  // 1 header block + 1 payload block
        let syms = interleave(&cw, 1, 7, false);
        assert_eq!(syms.len(), 8 + 5); // 8 header syms + cr+4=5 payload syms
    }
}
