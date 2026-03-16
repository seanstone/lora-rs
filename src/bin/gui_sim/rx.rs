use lora::rx::{
    frame_sync::frame_sync, fft_demod::fft_demod,
    gray_mapping::gray_map, deinterleaver::deinterleave,
    hamming_dec::hamming_dec, header_decoder::decode_header,
    crc_verif::verify_crc, dewhitening::dewhiten,
};
use rustfft::num_complex::Complex;
use std::sync::Arc;

use super::shared::{SimShared, LogEntry, MAX_LOG_ENTRIES};

// ─── LoRa demodulator ─────────────────────────────────────────────────────────

pub(crate) struct Rx {
    pub sf:           u8,
    pub cr:           u8,
    pub os_factor:    u32,
    pub sync_word:    u8,
    pub preamble_len: u16,
}

pub(crate) enum DecodeResult {
    /// CRC-verified payload + number of payload IQ samples consumed.
    Ok { payload: Vec<u8>, samples_used: usize },
    /// Header valid but not enough payload data yet — wait for more samples.
    Incomplete,
    /// Header checksum or payload CRC failed — corruption.
    Failed,
}

impl Rx {
    pub fn new(sf: u8, cr: u8, os_factor: u32, sync_word: u8, preamble_len: u16) -> Self {
        Self { sf, cr, os_factor, sync_word, preamble_len }
    }

    /// Full decode from raw IQ (frame sync + demod + decode).
    #[allow(dead_code)]
    pub fn decode(&self, iq: &[Complex<f32>]) -> Option<Vec<u8>> {
        let sync = frame_sync(iq, self.sf, self.sync_word, self.preamble_len, self.os_factor);
        if !sync.found { return None; }
        match self.decode_payload(&sync.symbols) {
            DecodeResult::Ok { payload, .. } => Some(payload),
            _ => None,
        }
    }

    /// Compute the exact number of payload IQ samples for a decoded packet.
    ///
    /// This mirrors the encode chain (pad_nibbles → hamming_enc → interleave →
    /// modulate) so we know precisely where the packet ends in the stream.
    fn payload_samples(&self, pay_len: usize, has_crc: bool) -> usize {
        let sf  = self.sf as usize;
        let cr  = self.cr as usize;
        let sps = (1usize << sf) * self.os_factor as usize;

        let header_cw      = sf - 2;
        let data_nibbles   = 2 * pay_len + if has_crc { 4 } else { 0 };
        let pad            = (sf - data_nibbles % sf) % sf;
        let remaining_cw   = data_nibbles + pad;
        let payload_blocks = remaining_cw / sf;         // exact: padded to sf
        let total_blocks   = 1 + payload_blocks;        // +1 for header block
        let _              = header_cw;                  // used in encode, implicit here

        total_blocks * (4 + cr) * sps
    }

    /// Decode from already-extracted payload symbols (post frame-sync).
    ///
    /// Returns [`Incomplete`] when the header parses but there aren't enough
    /// nibbles for the full payload + CRC.  On success, `samples_used` is the
    /// exact number of payload IQ samples the packet occupies — the caller
    /// should drain `payload_start + samples_used` from the streaming buffer.
    pub fn decode_payload(&self, symbols: &[Complex<f32>]) -> DecodeResult {
        let chirps    = fft_demod(symbols, self.sf, self.os_factor);
        let mapped    = gray_map(&chirps, self.sf);
        let codewords = deinterleave(&mapped, self.cr, self.sf, false);
        let nibbles   = hamming_dec(&codewords, self.cr, self.sf);

        // Need at least 5 nibbles for the header.
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
            if !verify_crc(&payload, crc_nib) { return DecodeResult::Failed; }
        }

        let samples_used = self.payload_samples(pay_len, info.has_crc);
        DecodeResult::Ok { payload, samples_used }
    }
}

// ─── Worker thread (streaming) ───────────────────────────────────────────────

/// Chunk of mixed IQ samples from the channel.
pub(crate) struct RxJob {
    pub sf:           u8,
    pub cr:           u8,
    pub os_factor:    u32,
    pub sync_word:    u8,
    pub preamble_len: u16,
    pub samples:      Vec<Complex<f32>>,
    pub tx_gen:       u64,
}

/// Cap on the internal RX buffer to prevent unbounded growth at very low SNR.
const MAX_RX_BUFFER: usize = 1 << 22; // ~4 M samples, ~32 MB

/// Streaming RX worker.
///
/// Accumulates incoming sample chunks in an internal buffer and repeatedly
/// runs `frame_sync` to find packets.  Buffer draining is precise:
///
/// - **Success**: drain `payload_start + samples_used` (exactly the decoded
///   packet), preserving any subsequent back-to-back packets.
/// - **Incomplete**: keep the preamble — the full payload hasn't arrived yet.
/// - **Failed**: drain to `payload_start` (skip the false/corrupt preamble),
///   let the next scan skip through the garbled payload.
pub(crate) async fn rx_worker(mut jobs: tokio::sync::mpsc::UnboundedReceiver<RxJob>, shared: Arc<SimShared>) {
    let mut rx       = Rx::new(7, 4, 4, 0x12, 8);
    let mut buffer:  Vec<Complex<f32>> = Vec::new();
    let mut next_seq: u16 = 0;
    let mut rx_gen:  u64 = 0;
    // After a reset the first decoded packet re-synchronises next_seq without
    // counting any gap (stale hardware-buffered packets would otherwise look
    // like spurious loss).
    let mut resync = true;

    while let Some(job) = jobs.recv().await {
        // Re-create decoder on settings change.
        if job.sf != rx.sf || job.os_factor != rx.os_factor
            || job.sync_word != rx.sync_word || job.preamble_len != rx.preamble_len
        {
            rx = Rx::new(job.sf, job.cr, job.os_factor, job.sync_word, job.preamble_len);
            buffer.clear();
            next_seq = 0;
        }

        // Discard stale samples from before a reset — the generation
        // counter is bumped by the sim loop on every clear_buf, so any
        // RxJobs queued before the reset carry the old generation.
        if job.tx_gen != rx_gen {
            rx_gen   = job.tx_gen;
            buffer.clear();
            next_seq = 0;
            resync   = true;
        }

        buffer.extend_from_slice(&job.samples);

        // Prevent unbounded growth.
        if buffer.len() > MAX_RX_BUFFER {
            let excess = buffer.len() - MAX_RX_BUFFER / 2;
            buffer.drain(..excess);
        }

        // Minimum samples worth checking: one full symbol.
        let sps = (1usize << rx.sf) * rx.os_factor as usize;

        // Try to find and decode packets in the buffer.
        loop {
            if buffer.len() < sps { break; }

            let sync = frame_sync(&buffer, rx.sf, rx.sync_word, rx.preamble_len, rx.os_factor);

            if !sync.found {
                // No sync found — drain the safely-scanned portion.
                if sync.consumed > 0 { buffer.drain(..sync.consumed); }
                break;
            }

            // payload_start = where the payload symbols begin in the buffer.
            let payload_start = sync.consumed - sync.symbols.len();

            match rx.decode_payload(&sync.symbols) {
                DecodeResult::Ok { payload, samples_used } if payload.len() >= 2 => {
                    let seq  = u16::from_le_bytes([payload[0], payload[1]]);
                    let text = String::from_utf8_lossy(&payload[2..]).to_string();

                    // Detect lost packets via sequence gap.
                    // Skip on first packet after a reset: stale hardware-
                    // buffered samples would produce a spurious gap.
                    let gap = seq.wrapping_sub(next_seq);
                    if !resync && gap > 0 && gap <= 1000 {
                        let mut s = shared.stats.lock().unwrap();
                        s.rx_lost += gap as usize;
                        drop(s);
                        let mut log = shared.log.lock().unwrap();
                        log.push_back(LogEntry {
                            ok: false,
                            payload: format!("LOST ×{gap}"),
                        });
                        if log.len() > MAX_LOG_ENTRIES { log.pop_front(); }
                    }
                    resync   = false;
                    next_seq = seq.wrapping_add(1);

                    {
                        let mut s = shared.stats.lock().unwrap();
                        s.rx_count += 1;
                        s.last_rx  = format!("#{seq} {text}");
                    }
                    {
                        let mut log = shared.log.lock().unwrap();
                        log.push_back(LogEntry {
                            ok: true,
                            payload: format!("#{seq} {text}"),
                        });
                        if log.len() > MAX_LOG_ENTRIES { log.pop_front(); }
                    }

                    // Drain exactly this packet — preserve subsequent data.
                    let drain = (payload_start + samples_used).min(buffer.len());
                    buffer.drain(..drain);
                    continue;
                }

                DecodeResult::Ok { .. } => {
                    // Decoded but too short for a seq number.
                    let mut log = shared.log.lock().unwrap();
                    log.push_back(LogEntry { ok: false, payload: "SHORT".into() });
                    if log.len() > MAX_LOG_ENTRIES { log.pop_front(); }
                    let drain = payload_start.min(buffer.len());
                    buffer.drain(..drain);
                    continue;
                }

                DecodeResult::Incomplete => {
                    // Header valid but full payload hasn't arrived yet.
                    break;
                }

                DecodeResult::Failed => {
                    // Header checksum or CRC failed — skip past the preamble.
                    let mut log = shared.log.lock().unwrap();
                    log.push_back(LogEntry { ok: false, payload: "CRC FAIL".into() });
                    if log.len() > MAX_LOG_ENTRIES { log.pop_front(); }
                    let drain = payload_start.min(buffer.len());
                    buffer.drain(..drain);
                    continue;
                }
            }
        }
    }
}
