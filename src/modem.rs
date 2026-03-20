/// High-level LoRa modem — encoder (`Tx`) and decoder (`Rx`).
///
/// These wrap the low-level DSP pipeline in `lora::tx` / `lora::rx` into
/// simple byte-oriented `modulate` / `decode` calls.

use rustfft::num_complex::Complex;

use crate::tx::{
    whitening::whiten, header::add_header, crc::add_crc,
    hamming_enc::hamming_enc, interleaver::interleave,
    gray_demap::gray_demap, modulate::modulate,
};
use crate::rx::{
    frame_sync::frame_sync, fft_demod::fft_demod,
    gray_mapping::gray_map, deinterleaver::deinterleave,
    hamming_dec::hamming_dec, header_decoder::decode_header,
    crc_verif::verify_crc, dewhitening::dewhiten,
};

// ── Tx ───────────────────────────────────────────────────────────────────────

/// LoRa modulator: encodes a raw byte payload into baseband IQ samples.
pub struct Tx {
    pub sf:           u8,
    pub cr:           u8,
    pub os_factor:    u32,
    pub sync_word:    u8,
    pub preamble_len: u16,
}

impl Tx {
    pub fn new(sf: u8, cr: u8, os_factor: u32, sync_word: u8, preamble_len: u16) -> Self {
        Self { sf, cr, os_factor, sync_word, preamble_len }
    }

    /// Produce clean (noise-free, unit-amplitude) LoRa IQ samples for `payload`.
    pub fn modulate(&self, payload: &[u8]) -> Vec<Complex<f32>> {
        let nibbles   = whiten(payload);
        let framed    = add_header(&nibbles, false, true, self.cr);
        let with_crc  = add_crc(&framed, payload, true);
        let padded    = pad_nibbles(&with_crc, self.sf, false);
        let codewords = hamming_enc(&padded, self.cr, self.sf);
        let symbols   = interleave(&codewords, self.cr, self.sf, false);
        let chirps    = gray_demap(&symbols, self.sf);
        modulate(&chirps, self.sf, self.sync_word, self.preamble_len, self.os_factor)
    }
}

pub(crate) fn pad_nibbles(nibbles: &[u8], sf: u8, ldro: bool) -> Vec<u8> {
    let pay_sf    = if ldro { (sf - 2) as usize } else { sf as usize };
    let header_cw = (sf - 2) as usize;
    let remaining = nibbles.len().saturating_sub(header_cw);
    let pad       = (pay_sf - remaining % pay_sf) % pay_sf;
    let mut v     = nibbles.to_vec();
    v.resize(v.len() + pad, 0);
    v
}

// ── DecodeResult ─────────────────────────────────────────────────────────────

/// Result of a single `Rx::decode_payload` call.
pub enum DecodeResult {
    /// CRC-verified payload + number of payload IQ samples consumed.
    Ok { payload: Vec<u8>, samples_used: usize },
    /// Header valid but not enough payload data yet — wait for more samples.
    Incomplete,
    /// Header checksum or payload CRC failed — corruption.
    Failed,
    /// Header decoded OK but payload CRC failed.  Contains the header info.
    CrcFail { payload_len: u8, cr: u8, has_crc: bool },
}

/// Result of [`Rx::decode_streaming`].
pub enum StreamDecodeResult {
    /// Payload decoded and CRC verified.
    Ok { payload: Vec<u8>, consumed: usize, freq_offset_bins: f64 },
    /// Header decoded but payload CRC failed.
    CrcFail { payload_len: u8, cr: u8, has_crc: bool, consumed: usize, freq_offset_bins: f64 },
    /// Preamble found but decode failed (header invalid or incomplete).
    /// Caller should drain `consumed` to skip past and avoid re-scanning.
    DecodeFailed { consumed: usize },
    /// No frame found in the buffer.
    None,
}

// ── Rx ───────────────────────────────────────────────────────────────────────

/// LoRa demodulator: finds and decodes packets from a stream of IQ samples.
pub struct Rx {
    pub sf:             u8,
    pub cr:             u8,
    pub os_factor:      u32,
    pub sync_word:      u8,
    pub preamble_len:   u16,
    /// Center frequency in Hz (0 = skip SFO compensation, e.g. simulation).
    pub center_freq_hz: f64,
    /// Bandwidth in Hz.
    pub bw_hz:          f64,
}

impl Rx {
    pub fn new(sf: u8, cr: u8, os_factor: u32, sync_word: u8, preamble_len: u16) -> Self {
        Self { sf, cr, os_factor, sync_word, preamble_len, center_freq_hz: 0.0, bw_hz: 0.0 }
    }

    /// Create an Rx with center frequency and bandwidth for SFO compensation.
    pub fn new_with_freq(
        sf: u8, cr: u8, os_factor: u32, sync_word: u8, preamble_len: u16,
        center_freq_hz: f64, bw_hz: f64,
    ) -> Self {
        Self { sf, cr, os_factor, sync_word, preamble_len, center_freq_hz, bw_hz }
    }

    /// Convenience: run frame sync + decode on a complete IQ buffer.
    pub fn decode(&self, iq: &[Complex<f32>]) -> Option<Vec<u8>> {
        let sync = frame_sync(
            iq, self.sf, self.sync_word, self.preamble_len, self.os_factor,
            self.center_freq_hz, self.bw_hz,
        );
        if !sync.found { return None; }
        match self.decode_payload(&sync.symbols) {
            DecodeResult::Ok { payload, .. } => Some(payload),
            _ => None,
        }
    }

    /// Streaming decode: find and decode one frame, returning the payload and
    /// the total number of input samples consumed (safe to `drain(..consumed)`
    /// from a streaming buffer).
    ///
    /// Returns `None` when no complete frame is found — `consumed` in that case
    /// is available via the lower-level `frame_sync` API if the caller needs to
    /// trim scanned-but-frameless samples.
    /// Result of [`decode_streaming`]: either a decoded payload, a CRC failure
    /// with header info, or nothing found.
    pub fn decode_streaming(&self, iq: &[Complex<f32>]) -> StreamDecodeResult {
        let sync = frame_sync(
            iq, self.sf, self.sync_word, self.preamble_len, self.os_factor,
            self.center_freq_hz, self.bw_hz,
        );
        if !sync.found { return StreamDecodeResult::None; }
        let fob = sync.freq_offset_bins;
        match self.decode_payload(&sync.symbols) {
            DecodeResult::Ok { payload, samples_used: _ } => {
                StreamDecodeResult::Ok { payload, consumed: sync.consumed, freq_offset_bins: fob }
            }
            DecodeResult::CrcFail { payload_len, cr, has_crc } => {
                StreamDecodeResult::CrcFail { payload_len, cr, has_crc, consumed: sync.consumed, freq_offset_bins: fob }
            }
            DecodeResult::Failed => StreamDecodeResult::DecodeFailed { consumed: sync.consumed },
            DecodeResult::Incomplete => StreamDecodeResult::None,
        }
    }

    /// Decode from already-extracted payload symbols (post frame-sync).
    ///
    /// Returns `Incomplete` when the header parses but there are not yet
    /// enough nibbles for the full payload + CRC.  On success `samples_used`
    /// is the exact number of payload IQ samples consumed — the caller should
    /// drain `payload_start + samples_used` from the streaming buffer.
    pub fn decode_payload(&self, symbols: &[Complex<f32>]) -> DecodeResult {
        let chirps    = fft_demod(symbols, self.sf, self.os_factor);
        let mapped    = gray_map(&chirps, self.sf);
        let codewords = deinterleave(&mapped, self.cr, self.sf, false);
        let nibbles   = hamming_dec(&codewords, self.cr, self.sf);

        if nibbles.len() < 5 { return DecodeResult::Incomplete; }

        let info = decode_header(&nibbles, false, 0, 0, false);
        if !info.valid { return DecodeResult::Failed; }

        let pay_len     = info.payload_len as usize;
        let min_nibbles = 2 * pay_len + if info.has_crc { 4 } else { 0 };
        if info.payload_nibbles.len() < min_nibbles {
            return DecodeResult::Incomplete;
        }

        let pay_nibbles = &info.payload_nibbles[..2 * pay_len];
        let payload     = dewhiten(pay_nibbles);
        if info.has_crc {
            let crc_nib = &info.payload_nibbles[2 * pay_len..2 * pay_len + 4];
            if !verify_crc(&payload, crc_nib) {
                return DecodeResult::CrcFail {
                    payload_len: info.payload_len,
                    cr: info.cr,
                    has_crc: info.has_crc,
                };
            }
        }

        let samples_used = self.payload_samples(pay_len, info.has_crc);
        DecodeResult::Ok { payload, samples_used }
    }

    /// Compute the exact number of payload IQ samples for a packet of `pay_len` bytes.
    pub fn payload_samples(&self, pay_len: usize, has_crc: bool) -> usize {
        let sf  = self.sf as usize;
        let cr  = self.cr as usize;
        let sps = (1usize << sf) * self.os_factor as usize;

        let data_nibbles   = 2 * pay_len + if has_crc { 4 } else { 0 };
        let pad            = (sf - data_nibbles % sf) % sf;
        let remaining_cw   = data_nibbles + pad;
        let payload_blocks = remaining_cw / sf;
        let total_blocks   = 1 + payload_blocks;

        total_blocks * (4 + cr) * sps
    }
}
