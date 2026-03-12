pub struct FrameInfo {
    pub payload_len: u8,
    pub cr: u8,
    pub has_crc: bool,
    pub valid: bool,
    pub payload_nibbles: Vec<u8>,
}

/// Strip and validate the 5-nibble explicit header.
/// When `impl_head` is true the provided cr/pay_len/has_crc are used directly.
pub fn decode_header(
    frame: &[u8],
    impl_head: bool,
    cr: u8,
    pay_len: u32,
    has_crc: bool,
) -> FrameInfo {
    if impl_head {
        return FrameInfo {
            payload_len: pay_len as u8,
            cr,
            has_crc,
            valid: true,
            payload_nibbles: frame.to_vec(),
        };
    }

    if frame.len() < 5 {
        return FrameInfo { payload_len: 0, cr: 0, has_crc: false, valid: false, payload_nibbles: vec![] };
    }

    let (h0, h1, h2) = (frame[0], frame[1], frame[2]);
    let decoded_len = (h0 << 4) | h1;
    let decoded_crc = (h2 & 1) != 0;
    let decoded_cr  = h2 >> 1;

    let stored_chk = ((frame[3] & 1) << 4) | frame[4];

    let bit = |v: u8, pos: u32| -> bool { (v >> pos) & 1 == 1 };
    let c4 = bit(h0, 3) ^ bit(h0, 2) ^ bit(h0, 1) ^ bit(h0, 0);
    let c3 = bit(h0, 3) ^ bit(h1, 3) ^ bit(h1, 2) ^ bit(h1, 1) ^ bit(h2, 0);
    let c2 = bit(h0, 2) ^ bit(h1, 3) ^ bit(h1, 0) ^ bit(h2, 3) ^ bit(h2, 1);
    let c1 = bit(h0, 1) ^ bit(h1, 2) ^ bit(h1, 0) ^ bit(h2, 2) ^ bit(h2, 1) ^ bit(h2, 0);
    let c0 = bit(h0, 0) ^ bit(h1, 1) ^ bit(h2, 3) ^ bit(h2, 2) ^ bit(h2, 1) ^ bit(h2, 0);

    let computed_chk = (c4 as u8) << 4 | (c3 as u8) << 3 | (c2 as u8) << 2 | (c1 as u8) << 1 | c0 as u8;
    let valid = stored_chk == computed_chk && decoded_len > 0;

    FrameInfo {
        payload_len: decoded_len,
        cr: decoded_cr,
        has_crc: decoded_crc,
        valid,
        payload_nibbles: frame[5..].to_vec(),
    }
}
