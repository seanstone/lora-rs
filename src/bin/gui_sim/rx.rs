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
    pub sf:        u8,
    pub cr:        u8,
    pub os_factor: u32,
}

impl Rx {
    pub fn new(sf: u8, cr: u8, os_factor: u32) -> Self { Self { sf, cr, os_factor } }

    pub fn decode(&self, iq: &[Complex<f32>]) -> Option<Vec<u8>> {
        let sync = frame_sync(iq, self.sf, 0x12, 8, self.os_factor);
        if !sync.found { return None; }
        let chirps    = fft_demod(&sync.symbols, self.sf, self.os_factor);
        let symbols   = gray_map(&chirps, self.sf);
        let codewords = deinterleave(&symbols, self.cr, self.sf, false);
        let nibbles   = hamming_dec(&codewords, self.cr, self.sf);
        let info      = decode_header(&nibbles, false, 0, 0, false);
        if !info.valid { return None; }
        let pay_len     = info.payload_len as usize;
        let min_nibbles = 2 * pay_len + if info.has_crc { 4 } else { 0 };
        if info.payload_nibbles.len() < min_nibbles { return None; }
        let pay_nibbles = &info.payload_nibbles[..2 * pay_len];
        let payload     = dewhiten(pay_nibbles);
        if info.has_crc {
            let crc_nib = &info.payload_nibbles[2 * pay_len..2 * pay_len + 4];
            if !verify_crc(&payload, crc_nib) { return None; }
        }
        Some(payload)
    }
}

// ─── Worker thread ────────────────────────────────────────────────────────────

/// Packet handed from the channel to the RX decode thread.
pub(crate) struct RxJob {
    pub sf:        u8,
    pub cr:        u8,
    pub os_factor: u32,
    pub payload:   Vec<u8>,
    pub mixed:     Vec<Complex<f32>>,
}

/// Decodes incoming packets off the critical path and updates stats / log.
pub(crate) fn rx_worker(jobs: std::sync::mpsc::Receiver<RxJob>, shared: Arc<SimShared>) {
    let mut rx = Rx::new(7, 4, 4);
    while let Ok(job) = jobs.recv() {
        if job.sf != rx.sf || job.os_factor != rx.os_factor {
            rx = Rx::new(job.sf, job.cr, job.os_factor);
        }
        let result = rx.decode(&job.mixed);
        {
            let mut s = shared.stats.lock().unwrap();
            s.total += 1;
            s.last_tx = String::from_utf8_lossy(&job.payload).to_string();
            if let Some(ref rx_payload) = result {
                if *rx_payload == job.payload { s.ok += 1; }
                s.last_rx = String::from_utf8_lossy(rx_payload).to_string();
            } else {
                s.last_rx = "—".to_string();
            }
        }
        {
            let ok = result.as_deref() == Some(job.payload.as_slice());
            let entry = LogEntry {
                ok,
                payload: if ok {
                    String::from_utf8_lossy(&job.payload).to_string()
                } else {
                    result.as_ref()
                        .map(|p| format!("ERR: {}", String::from_utf8_lossy(p)))
                        .unwrap_or_else(|| "LOST".to_string())
                },
            };
            let mut log = shared.log.lock().unwrap();
            log.push_back(entry);
            if log.len() > MAX_LOG_ENTRIES { log.pop_front(); }
        }
    }
}
