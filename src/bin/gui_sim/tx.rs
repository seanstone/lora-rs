use lora::tx::{
    whitening::whiten, header::add_header, crc::add_crc,
    hamming_enc::hamming_enc, interleaver::interleave,
    gray_demap::gray_demap, modulate::modulate,
};
use rustfft::num_complex::Complex;

// ─── LoRa modulator ───────────────────────────────────────────────────────────

pub(crate) struct Tx {
    pub sf:        u8,
    pub cr:        u8,
    pub os_factor: u32,
}

impl Tx {
    pub fn new(sf: u8, cr: u8, os_factor: u32) -> Self { Self { sf, cr, os_factor } }

    /// Produce clean (noise-free, unit-amplitude) LoRa IQ samples.
    pub fn modulate(&self, payload: &[u8]) -> Vec<Complex<f32>> {
        let nibbles   = whiten(payload);
        let framed    = add_header(&nibbles, false, true, self.cr);
        let with_crc  = add_crc(&framed, payload, true);
        let padded    = pad_nibbles(&with_crc, self.sf, false);
        let codewords = hamming_enc(&padded, self.cr, self.sf);
        let symbols   = interleave(&codewords, self.cr, self.sf, false);
        let chirps    = gray_demap(&symbols, self.sf);
        modulate(&chirps, self.sf, 0x12, 8, self.os_factor)
    }
}

fn pad_nibbles(nibbles: &[u8], sf: u8, ldro: bool) -> Vec<u8> {
    let pay_sf    = if ldro { (sf - 2) as usize } else { sf as usize };
    let header_cw = (sf - 2) as usize;
    let remaining = nibbles.len().saturating_sub(header_cw);
    let pad       = (pay_sf - remaining % pay_sf) % pay_sf;
    let mut v     = nibbles.to_vec();
    v.resize(v.len() + pad, 0);
    v
}

// ─── Worker thread ────────────────────────────────────────────────────────────

/// TX modulation request. AWGN and gain are applied by the Channel, not here.
pub(crate) struct TxJob {
    pub sf:        u8,
    pub cr:        u8,
    pub os_factor: u32,
    pub payload:   Vec<u8>,
    pub tx_gen:       u64,
}

/// Clean (noise-free, unit-amplitude) modulated packet.
pub(crate) struct TxResult {
    pub payload: Vec<u8>,
    pub clean:   Vec<Complex<f32>>,
    pub tx_gen:     u64,
}

/// Runs LoRa modulation off the sim_loop critical path.
/// Produces clean IQ only — AWGN is applied per-sample by the Channel.
pub(crate) async fn tx_worker(
    mut jobs: tokio::sync::mpsc::UnboundedReceiver<TxJob>,
    results:  tokio::sync::mpsc::UnboundedSender<TxResult>,
) {
    let mut tx = Tx::new(7, 4, 4);
    while let Some(job) = jobs.recv().await {
        if job.sf != tx.sf || job.os_factor != tx.os_factor {
            tx = Tx::new(job.sf, job.cr, job.os_factor);
        }
        let clean = tx.modulate(&job.payload);
        let _ = results.send(TxResult { payload: job.payload, clean, tx_gen: job.tx_gen });
    }
}
