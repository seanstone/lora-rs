use crate::tables::WHITENING_SEQ;

/// XOR each byte with the whitening sequence, expand to nibble pairs.
/// out[2i] = low nibble, out[2i+1] = high nibble of (payload[i] ^ seq[i]).
pub fn whiten(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 * payload.len());
    for (i, &byte) in payload.iter().enumerate() {
        let w = byte ^ WHITENING_SEQ[i];
        out.push(w & 0x0F);
        out.push(w >> 4);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rx::dewhitening::dewhiten;

    #[test]
    fn output_length() {
        let out = whiten(b"TX-RX");
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn all_nibbles_in_range() {
        assert!(whiten(b"TX-RX").iter().all(|&n| n <= 15));
    }

    #[test]
    fn empty() {
        assert_eq!(whiten(&[]), vec![]);
    }

    #[test]
    fn roundtrip_short() {
        let p = b"TX-RX";
        assert_eq!(dewhiten(&whiten(p)), p.to_vec());
    }

    #[test]
    fn roundtrip_ascii() {
        let p = b"Hello, LoRa!";
        assert_eq!(dewhiten(&whiten(p)), p.to_vec());
    }

    #[test]
    fn roundtrip_zeros() {
        let p = vec![0u8; 8];
        assert_eq!(dewhiten(&whiten(&p)), p);
    }

    #[test]
    fn roundtrip_ff() {
        let p = vec![0xFFu8; 6];
        assert_eq!(dewhiten(&whiten(&p)), p);
    }
}
