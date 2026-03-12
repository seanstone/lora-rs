fn crc16_byte(mut crc: u16, mut byte: u8) -> u16 {
    for _ in 0..8 {
        if ((crc & 0x8000) >> 8) ^ (byte & 0x80) as u16 != 0 {
            crc = (crc << 1) ^ 0x1021;
        } else {
            crc <<= 1;
        }
        byte <<= 1;
    }
    crc
}

fn compute_crc(data: &[u8]) -> u16 {
    let n = data.len();
    let mut crc: u16 = 0;
    for &b in &data[..n - 2] {
        crc = crc16_byte(crc, b);
    }
    crc ^ data[n - 1] as u16 ^ ((data[n - 2] as u16) << 8)
}

/// Compare `compute_crc(payload)` against the 4 received CRC nibbles.
pub fn verify_crc(payload: &[u8], crc_nibbles: &[u8]) -> bool {
    let received = crc_nibbles[0] as u16
        | (crc_nibbles[1] as u16) <<  4
        | (crc_nibbles[2] as u16) <<  8
        | (crc_nibbles[3] as u16) << 12;
    compute_crc(payload) == received
}
