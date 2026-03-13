use eframe::egui;
use std::{
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};

use super::channel::Channel;
use super::display::{DisplayJob, display_worker};
use super::rx::{RxJob, rx_worker};
use super::shared::SimShared;
use super::tx::{TxJob, TxResult, tx_worker};
use super::db_to_amp;

/// Tick interval for the stream thread (~60 fps waterfall scroll).
pub(crate) const TICK: Duration = Duration::from_millis(16);

/// Hard cap on FFT rows rendered per tick to prevent runaway scroll at very
/// high sample rates.
const MAX_ROWS_PER_TICK: usize = 128;

pub(crate) fn sim_loop(shared: Arc<SimShared>, ctx: Option<egui::Context>) {
    let payloads: &[&[u8]] = &[
        b"Hello, LoRa!", b"Rust 2024", b"AWGN channel",
        b"burst test",   b"lora-rs",   b"packet!",
    ];
    let mut idx = 0usize;

    // Virtual sample clock.
    let mut produced_samples: u64 = 0;
    let mut next_packet_at:   u64 = 0;
    let mut last_interval_ms: u64 = u64::MAX;
    // true while a TxJob is being processed by the worker thread.
    let mut tx_inflight = false;
    // Completed TxResult waiting to be pushed to the channel at the right time.
    let mut tx_buffered: Option<TxResult> = None;
    // true while the channel has a packet queued or in progress.
    let mut tx_in_channel = false;

    // Spawn worker threads.
    let (rx_send,     rx_recv)     = std::sync::mpsc::channel::<RxJob>();
    let (tx_job_send, tx_job_recv) = std::sync::mpsc::channel::<TxJob>();
    let (tx_res_send, tx_res_recv) = std::sync::mpsc::sync_channel::<TxResult>(1);
    { let s = shared.clone(); std::thread::spawn(move || rx_worker(rx_recv, s)); }
    std::thread::spawn(move || tx_worker(tx_job_recv, tx_res_send));

    // Display thread is only needed in GUI mode.
    let disp_send: Option<std::sync::mpsc::Sender<DisplayJob>> = if ctx.is_some() {
        let (tx, rx) = std::sync::mpsc::channel::<DisplayJob>();
        { let s = shared.clone(); std::thread::spawn(move || display_worker(rx, s)); }
        Some(tx)
    } else {
        None
    };

    let init_signal_db = *shared.signal_db.lock().unwrap();
    let init_noise_db  = *shared.noise_db.lock().unwrap();
    let mut channel = Channel::new(
        db_to_amp(init_noise_db)  / std::f32::consts::SQRT_2,
        db_to_amp(init_signal_db),
    );

    // Dispatch the next TX modulation job to the worker thread.
    macro_rules! dispatch_tx {
        ($sf:expr, $os:expr) => {{
            let payload = payloads[idx % payloads.len()].to_vec();
            idx += 1;
            let _ = tx_job_send.send(TxJob { sf: $sf, cr: 4, os_factor: $os, payload });
            tx_inflight = true;
        }};
    }

    loop {
        let tick_start = Instant::now();

        if !shared.running.load(Ordering::Relaxed) {
            std::thread::sleep(TICK);
            continue;
        }

        // Flush channel on settings change; reset virtual clock.
        if shared.clear_buf.swap(false, Ordering::Relaxed) {
            channel.clear();
            tx_buffered      = None;
            tx_in_channel    = false;
            produced_samples = 0;
            next_packet_at   = 0;
            last_interval_ms = u64::MAX;
            while tx_res_recv.try_recv().is_ok() {}
            tx_inflight = false;
        }

        let sf            = *shared.sf.lock().unwrap();
        let os_factor     = *shared.os_factor.lock().unwrap();
        let samp_rate_khz = *shared.samp_rate_khz.lock().unwrap();
        let fft_size      = *shared.fft_size.lock().unwrap();
        let signal_db     = *shared.signal_db.lock().unwrap();
        let noise_db      = *shared.noise_db.lock().unwrap();
        let interval_ms   = *shared.interval_ms.lock().unwrap();

        let signal_amp  = db_to_amp(signal_db);
        let noise_sigma = db_to_amp(noise_db) / std::f32::consts::SQRT_2;

        // Push current levels into the channel — applied per-sample in tick().
        channel.set_noise_sigma(noise_sigma);
        channel.set_signal_amp(signal_amp);

        let samp_rate_hz     = samp_rate_khz as u64 * 1000;
        let samples_per_tick = (samp_rate_hz as f64 * TICK.as_secs_f64()).round() as u64;
        let interval_samples = interval_ms * samp_rate_hz / 1000;

        if interval_ms != last_interval_ms {
            last_interval_ms = interval_ms;
            next_packet_at   = produced_samples;
        }

        // ── TX scheduler ─────────────────────────────────────────────────
        // Keep the modulation worker one packet ahead.
        if !tx_inflight { dispatch_tx!(sf, os_factor); }

        // Collect completed modulation result into the buffer.
        if tx_buffered.is_none() {
            if let Ok(res) = tx_res_recv.try_recv() {
                tx_buffered = Some(res);
                dispatch_tx!(sf, os_factor);   // pre-build the next one
            }
        }

        // Push the buffered packet into the channel when the interval has elapsed.
        if !tx_in_channel && next_packet_at <= produced_samples {
            if let Some(res) = tx_buffered.take() {
                channel.push_packet(res.payload, res.clean);
                tx_in_channel = true;
            }
            // If tx_buffered was empty the worker is still modulating; the
            // channel streams silence until the packet arrives next tick.
        }

        // ── Channel tick: produce one tick's worth of mixed samples ───────
        let n = samples_per_tick as usize;
        let (mixed, completed) = channel.tick(n);
        produced_samples += n as u64;

        // Completed packets → RX decoder; schedule next packet.
        for (payload, mixed_pkt) in completed {
            tx_in_channel  = false;
            next_packet_at = produced_samples + interval_samples;
            let _ = rx_send.send(RxJob { sf, cr: 4, os_factor, payload, mixed: mixed_pkt });
        }

        // ── Channel → display thread ──────────────────────────────────────
        let target_rows = (n / fft_size).max(1).min(MAX_ROWS_PER_TICK);
        let avail_rows  = mixed.len() / fft_size;
        let rows        = target_rows.min(avail_rows);

        if let Some(ref ds) = disp_send {
            for i in 0..rows {
                let window = mixed[i * fft_size..(i + 1) * fft_size].to_vec();
                let _ = ds.send((window, i + 1 == rows));
            }
        }
        // In headless mode mixed samples are simply discarded; Channel::tick()
        // already produced them internally.

        // ── Buffer status ─────────────────────────────────────────────────
        let pending   = channel.pending_samples();
        let lag_ms    = pending as f32 * 1000.0 / samp_rate_hz as f32;
        let overflow  = avail_rows > target_rows * 8;
        let underflow = avail_rows == 0;
        shared.buf_lag_ms   .store(lag_ms.to_bits(), Ordering::Relaxed);
        shared.buf_overflow .store(overflow,          Ordering::Relaxed);
        shared.buf_underflow.store(underflow,         Ordering::Relaxed);

        shared.waterfall_plot.set_freq(fft_size as f64 / 2.0);
        shared.waterfall_plot.set_bw(fft_size as f64);

        if let Some(ref c) = ctx { c.request_repaint(); }

        let elapsed = tick_start.elapsed();
        if elapsed < TICK { std::thread::sleep(TICK - elapsed); }
    }
}
