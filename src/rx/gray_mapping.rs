/// Standard Gray encode: x ^ (x >> 1).
pub fn gray_map(symbols: &[u32], _sf: u8) -> Vec<u32> {
    symbols.iter().map(|&x| x ^ (x >> 1)).collect()
}
