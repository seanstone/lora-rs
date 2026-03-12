/// Fold-XOR (inverse Gray code): r = y ^ (y>>1) ^ ... ^ (y>>(sf-1)).
pub fn gray_demap(symbols: &[u32], sf: u8) -> Vec<u32> {
    symbols.iter().map(|&y| {
        let mut r = y;
        for j in 1..sf as u32 {
            r ^= y >> j;
        }
        r
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rx::gray_mapping::gray_map;

    fn roundtrip(sf: u8) {
        let n = 1u32 << sf;
        let syms: Vec<u32> = (0..n).collect();
        assert_eq!(gray_map(&gray_demap(&syms, sf), sf), syms, "sf={sf}");
        assert_eq!(gray_demap(&gray_map(&syms, sf), sf), syms, "sf={sf}");
    }

    #[test] fn rt_sf5()  { roundtrip(5); }
    #[test] fn rt_sf7()  { roundtrip(7); }
    #[test] fn rt_sf9()  { roundtrip(9); }
    #[test] fn rt_sf12() { roundtrip(12); }

    #[test]
    fn known_values() {
        // gray_demap is the inverse of gray_map: gray_demap(x^(x>>1)) == x
        for x in 0u32..128 {
            let g = x ^ (x >> 1);
            assert_eq!(gray_demap(&[g], 7)[0], x);
        }
    }
}
