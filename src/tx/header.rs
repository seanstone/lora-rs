/// Prepend a 5-nibble explicit LoRa header to whitened nibbles.
/// If `impl_head` is true, nibbles are returned unchanged.
///
/// Header layout:
///   [0] high nibble of payload byte-length
///   [1] low  nibble of payload byte-length
///   [2] (cr << 1) | has_crc
///   [3] c4 in bit 0
///   [4] c3..c0
pub fn add_header(nibbles: &[u8], impl_head: bool, has_crc: bool, cr: u8) -> Vec<u8> {
    if impl_head {
        return nibbles.to_vec();
    }

    let payload_len = (nibbles.len() / 2) as u8;
    let h0 = payload_len >> 4;
    let h1 = payload_len & 0x0F;
    let h2 = (cr << 1) | has_crc as u8;

    let bit = |v: u8, pos: u32| -> bool { (v >> pos) & 1 == 1 };

    let c4 = bit(h0, 3) ^ bit(h0, 2) ^ bit(h0, 1) ^ bit(h0, 0);
    let c3 = bit(h0, 3) ^ bit(h1, 3) ^ bit(h1, 2) ^ bit(h1, 1) ^ bit(h2, 0);
    let c2 = bit(h0, 2) ^ bit(h1, 3) ^ bit(h1, 0) ^ bit(h2, 3) ^ bit(h2, 1);
    let c1 = bit(h0, 1) ^ bit(h1, 2) ^ bit(h1, 0) ^ bit(h2, 2) ^ bit(h2, 1) ^ bit(h2, 0);
    let c0 = bit(h0, 0) ^ bit(h1, 1) ^ bit(h2, 3) ^ bit(h2, 2) ^ bit(h2, 1) ^ bit(h2, 0);

    let mut out = Vec::with_capacity(5 + nibbles.len());
    out.extend_from_slice(&[
        h0,
        h1,
        h2,
        c4 as u8,
        (c3 as u8) << 3 | (c2 as u8) << 2 | (c1 as u8) << 1 | (c0 as u8),
    ]);
    out.extend_from_slice(nibbles);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rx::header_decoder::decode_header;

    #[test]
    fn length_and_nibble_range() {
        let nibbles = vec![0u8; 10];
        let out = add_header(&nibbles, false, true, 1);
        assert_eq!(out.len(), 15);
        assert!(out.iter().all(|&n| n <= 15));
    }

    #[test]
    fn implicit_passthrough() {
        let nibbles = vec![1u8, 2, 3, 4];
        assert_eq!(add_header(&nibbles, true, true, 1), nibbles);
    }

    #[test]
    fn roundtrip_explicit() {
        let nibbles = vec![0u8; 12]; // 6 payload bytes
        let framed  = add_header(&nibbles, false, true, 1);
        let info    = decode_header(&framed, false, 0, 0, false);
        assert!(info.valid);
        assert_eq!(info.payload_len, 6);
        assert_eq!(info.cr, 1);
        assert!(info.has_crc);
        assert_eq!(info.payload_nibbles, nibbles);
    }

    #[test]
    fn roundtrip_no_crc() {
        let nibbles = vec![0u8; 2]; // 1 byte
        let framed  = add_header(&nibbles, false, false, 4);
        let info    = decode_header(&framed, false, 0, 0, false);
        assert!(info.valid);
        assert!(!info.has_crc);
        assert_eq!(info.cr, 4);
    }

    #[test]
    fn checksum_corruption_detected() {
        let nibbles  = vec![0u8; 8];
        let mut framed = add_header(&nibbles, false, true, 2);
        framed[4] ^= 1; // flip a checksum bit
        let info = decode_header(&framed, false, 0, 0, false);
        assert!(!info.valid);
    }
}
