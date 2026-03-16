pub(crate) use lora::modem::Tx;

// ─── Worker thread ────────────────────────────────────────────────────────────

/// TX modulation request. AWGN and gain are applied by the Channel, not here.
pub(crate) struct TxJob {
    pub sf:           u8,
    pub cr:           u8,
    pub os_factor:    u32,
    pub sync_word:    u8,
    pub preamble_len: u16,
    pub payload:      Vec<u8>,
    pub tx_gen:       u64,
}

/// Clean (noise-free, unit-amplitude) modulated packet.
pub(crate) struct TxResult {
    pub payload: Vec<u8>,
    pub clean:   Vec<rustfft::num_complex::Complex<f32>>,
    pub tx_gen:  u64,
}

/// Runs LoRa modulation off the sim_loop critical path.
/// Produces clean IQ only — AWGN is applied per-sample by the Channel.
pub(crate) async fn tx_worker(
    mut jobs: tokio::sync::mpsc::UnboundedReceiver<TxJob>,
    results:  tokio::sync::mpsc::UnboundedSender<TxResult>,
) {
    let mut tx = Tx::new(7, 4, 4, 0x12, 8);
    while let Some(job) = jobs.recv().await {
        if job.sf != tx.sf || job.os_factor != tx.os_factor
            || job.sync_word != tx.sync_word || job.preamble_len != tx.preamble_len
        {
            tx = Tx::new(job.sf, job.cr, job.os_factor, job.sync_word, job.preamble_len);
        }
        let clean = tx.modulate(&job.payload);
        let _ = results.send(TxResult { payload: job.payload, clean, tx_gen: job.tx_gen });
    }
}
