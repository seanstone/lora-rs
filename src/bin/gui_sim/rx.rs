pub(crate) use lora::modem::{Rx, DecodeResult};

use rustfft::num_complex::Complex;
use std::sync::Arc;

use super::shared::{SimShared, LogEntry, MAX_LOG_ENTRIES};
use lora::rx::frame_sync::frame_sync;

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
/// Accumulates incoming sample chunks and repeatedly runs `frame_sync`.
pub(crate) async fn rx_worker(mut jobs: tokio::sync::mpsc::UnboundedReceiver<RxJob>, shared: Arc<SimShared>) {
    let mut rx       = Rx::new(7, 4, 4, 0x12, 8);
    let mut buffer:  Vec<Complex<f32>> = Vec::new();
    let mut next_seq: u16 = 0;
    let mut rx_gen:  u64 = 0;
    let mut resync = true;

    while let Some(job) = jobs.recv().await {
        if job.sf != rx.sf || job.os_factor != rx.os_factor
            || job.sync_word != rx.sync_word || job.preamble_len != rx.preamble_len
        {
            rx = Rx::new(job.sf, job.cr, job.os_factor, job.sync_word, job.preamble_len);
            buffer.clear();
            next_seq = 0;
        }

        if job.tx_gen != rx_gen {
            rx_gen   = job.tx_gen;
            buffer.clear();
            next_seq = 0;
            resync   = true;
        }

        buffer.extend_from_slice(&job.samples);

        if buffer.len() > MAX_RX_BUFFER {
            let excess = buffer.len() - MAX_RX_BUFFER / 2;
            buffer.drain(..excess);
        }

        let sps = (1usize << rx.sf) * rx.os_factor as usize;

        loop {
            if buffer.len() < sps { break; }

            let sync = frame_sync(&buffer, rx.sf, rx.sync_word, rx.preamble_len, rx.os_factor);

            if !sync.found {
                if sync.consumed > 0 { buffer.drain(..sync.consumed); }
                break;
            }

            let payload_start = sync.consumed - sync.symbols.len();

            match rx.decode_payload(&sync.symbols) {
                DecodeResult::Ok { payload, samples_used } if payload.len() >= 2 => {
                    let seq  = u16::from_le_bytes([payload[0], payload[1]]);
                    let text = String::from_utf8_lossy(&payload[2..]).to_string();

                    let gap = seq.wrapping_sub(next_seq);
                    if !resync && gap > 0 && gap <= 1000 {
                        let mut s = shared.stats.lock().unwrap();
                        s.rx_lost += gap as usize;
                        drop(s);
                        let mut log = shared.log.lock().unwrap();
                        log.push_back(LogEntry { ok: false, payload: format!("LOST ×{gap}") });
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
                        log.push_back(LogEntry { ok: true, payload: format!("#{seq} {text}") });
                        if log.len() > MAX_LOG_ENTRIES { log.pop_front(); }
                    }

                    let drain = (payload_start + samples_used).min(buffer.len());
                    buffer.drain(..drain);
                    continue;
                }

                DecodeResult::Ok { .. } => {
                    let mut log = shared.log.lock().unwrap();
                    log.push_back(LogEntry { ok: false, payload: "SHORT".into() });
                    if log.len() > MAX_LOG_ENTRIES { log.pop_front(); }
                    let drain = payload_start.min(buffer.len());
                    buffer.drain(..drain);
                    continue;
                }

                DecodeResult::Incomplete => { break; }

                DecodeResult::Failed | DecodeResult::CrcFail { .. } => {
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
