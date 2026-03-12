use crate::tables::WHITENING_SEQ;

/// Reverse of `whiten`: recombine nibble pairs and XOR with the whitening sequence.
pub fn dewhiten(nibbles: &[u8]) -> Vec<u8> {
    let n = nibbles.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let low  = nibbles[2 * i]     ^ (WHITENING_SEQ[i] & 0x0F);
        let high = nibbles[2 * i + 1] ^ (WHITENING_SEQ[i] >> 4);
        out.push((high << 4) | low);
    }
    out
}
