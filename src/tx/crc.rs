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

/// Append 4 CRC nibbles (CRC-CCITT, little-endian) if `has_crc` is set.
pub fn add_crc(framed_nibbles: &[u8], payload: &[u8], has_crc: bool) -> Vec<u8> {
    if !has_crc {
        return framed_nibbles.to_vec();
    }
    let crc = compute_crc(payload);
    let mut out = framed_nibbles.to_vec();
    out.push((crc       ) as u8 & 0x0F);
    out.push((crc >>  4) as u8 & 0x0F);
    out.push((crc >>  8) as u8 & 0x0F);
    out.push((crc >> 12) as u8 & 0x0F);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rx::crc_verif::verify_crc;

    #[test]
    fn appends_four_nibbles() {
        let nibbles = vec![0u8; 4];
        let payload = b"Hi".to_vec();
        let out = add_crc(&nibbles, &payload, true);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&n| n <= 15));
    }

    #[test]
    fn no_crc_passthrough() {
        let nibbles = vec![1u8, 2, 3];
        assert_eq!(add_crc(&nibbles, b"x", false), nibbles);
    }

    #[test]
    fn roundtrip() {
        let payload = b"Hello".to_vec();
        let nibbles = vec![0u8; 2];
        let with_crc = add_crc(&nibbles, &payload, true);
        let crc_nib = &with_crc[nibbles.len()..];
        assert!(verify_crc(&payload, crc_nib));
    }

    #[test]
    fn corruption_detected() {
        let payload = b"Hello".to_vec();
        let nibbles = vec![0u8; 2];
        let mut with_crc = add_crc(&nibbles, &payload, true);
        *with_crc.last_mut().unwrap() ^= 1;
        let crc_nib = &with_crc[nibbles.len()..];
        assert!(!verify_crc(&payload, crc_nib));
    }
}
