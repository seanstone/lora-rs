use eframe::egui;
use std::{
    collections::VecDeque,
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};

use super::channel::Channel;
use super::display::{DisplayJob, display_worker};
use super::driver::Driver;
use super::rx::{RxJob, rx_worker};
use super::shared::SimShared;
use super::tx::{TxJob, TxResult, tx_worker};
use super::db_to_amp;

#[cfg(feature = "uhd")]
use super::uhd_device::UhdDevice;

fn make_driver(shared: &SimShared) -> Box<dyn Driver> {
    let noise_sigma = db_to_amp(*shared.noise_db.lock().unwrap()) / std::f32::consts::SQRT_2;
    let signal_amp  = db_to_amp(*shared.signal_db.lock().unwrap());

    #[cfg(feature = "uhd")]
    if shared.use_uhd.load(Ordering::Relaxed) {
        let args       = shared.uhd_args.lock().unwrap().clone();
        let freq       = *shared.uhd_freq_hz.lock().unwrap();
        let rx_gain    = *shared.uhd_rx_gain_db.lock().unwrap();
        let tx_gain    = *shared.uhd_tx_gain_db.lock().unwrap();
        let sr_hz      = *shared.samp_rate_khz.lock().unwrap() as f64 * 1000.0;
        let os         = *shared.os_factor.lock().unwrap() as f64;
        let bw_hz      = sr_hz / os;
        match UhdDevice::new(&args, freq, sr_hz, bw_hz, rx_gain, tx_gain) {
            Ok(dev) => return Box::new(dev),
            Err(e)  => {
                eprintln!("[uhd] open failed: {e} — falling back to sim");
                shared.use_uhd.store(false, Ordering::Relaxed);
            }
        }
    }

    Box::new(Channel::new(noise_sigma, signal_amp))
}

/// Tick interval for the stream thread (~60 fps waterfall scroll).
pub(crate) const TICK: Duration = Duration::from_millis(16);

/// Hard cap on FFT rows rendered per tick to prevent runaway scroll at very
/// high sample rates.
const MAX_ROWS_PER_TICK: usize = 128;

/// Number of packets to keep pre-modulated in the TX pipeline.
/// Needs to cover the worst case of ceil(tick / packet_duration) + 1.
/// SF7 at 4 MSPS: packet ≈ 7.7 ms, tick = 16 ms → ~3 packets/tick → depth 4.
const TX_PIPELINE_DEPTH: usize = 4;

pub(crate) fn sim_loop(shared: Arc<SimShared>, ctx: Option<egui::Context>) {
    let payloads: &[&[u8]] = &[
        b"Hello, LoRa!", b"Rust 2024", b"AWGN channel",
        b"burst test",   b"lora-rs",   b"packet!",
    ];
    let mut payload_idx = 0usize;
    let mut seq         = 0u16;
    let mut tx_gen         = 0u64;   // tx_generation counter — incremented on reset

    // Virtual sample clock.
    let mut produced_samples: u64 = 0;
    let mut next_packet_at:   u64 = 0;
    let mut last_interval_ms: u64 = u64::MAX;
    // Track UHD gain values so we only call into hardware on change.
    let mut last_uhd_rx_gain = f64::NAN;
    let mut last_uhd_tx_gain = f64::NAN;
    // Pre-modulated packets ready to push into the channel.
    let mut tx_queue: VecDeque<TxResult> = VecDeque::new();
    // Jobs sent to the TX worker that haven't produced a result yet.
    let mut jobs_in_flight: usize = 0;

    // Spawn worker threads.
    let (rx_send,     rx_recv)     = std::sync::mpsc::channel::<RxJob>();
    let (tx_job_send, tx_job_recv) = std::sync::mpsc::channel::<TxJob>();
    let (tx_res_send, tx_res_recv) = std::sync::mpsc::channel::<TxResult>();
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

    let mut driver: Box<dyn Driver> = make_driver(&shared);

    // Build a numbered payload: [seq_le16][text].
    macro_rules! dispatch_tx {
        ($sf:expr, $os:expr) => {{
            let text = payloads[payload_idx % payloads.len()];
            payload_idx += 1;
            let mut payload = Vec::with_capacity(2 + text.len());
            payload.extend_from_slice(&seq.to_le_bytes());
            payload.extend_from_slice(text);
            seq = seq.wrapping_add(1);
            let _ = tx_job_send.send(TxJob { sf: $sf, cr: 4, os_factor: $os, payload, tx_gen });
            jobs_in_flight += 1;
        }};
    }

    loop {
        let tick_start = Instant::now();

        if !shared.running.load(Ordering::Relaxed) {
            std::thread::sleep(TICK);
            continue;
        }

        // Rebuild driver on UHD settings change; flush on any settings change.
        let should_rebuild = shared.rebuild_driver.swap(false, Ordering::Relaxed);
        let should_clear   = shared.clear_buf.swap(false, Ordering::Relaxed);
        if should_rebuild || should_clear {
            if should_rebuild {
                // Drop the old driver first so its UHD handles are closed
                // before make_driver opens new ones (C glue uses global state).
                driver = Box::new(Channel::new(0.0, 1.0));
                driver = make_driver(&shared);
            } else {
                driver.clear();
            }
            tx_queue.clear();
            while tx_res_recv.try_recv().is_ok() {}
            jobs_in_flight   = 0;
            produced_samples = 0;
            next_packet_at   = 0;
            last_interval_ms = u64::MAX;
            seq              = 0;
            payload_idx      = 0;
            tx_gen           += 1;
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

        // Push current levels into the driver — applied per-sample in tick().
        driver.set_noise_sigma(noise_sigma);
        driver.set_signal_amp(signal_amp);

        // Apply hardware gains on change (no-op for sim channel).
        let uhd_rx_gain = *shared.uhd_rx_gain_db.lock().unwrap();
        let uhd_tx_gain = *shared.uhd_tx_gain_db.lock().unwrap();
        if uhd_rx_gain != last_uhd_rx_gain { last_uhd_rx_gain = uhd_rx_gain; driver.set_hw_rx_gain(uhd_rx_gain); }
        if uhd_tx_gain != last_uhd_tx_gain { last_uhd_tx_gain = uhd_tx_gain; driver.set_hw_tx_gain(uhd_tx_gain); }

        let samp_rate_hz     = samp_rate_khz as u64 * 1000;
        let samples_per_tick = (samp_rate_hz as f64 * TICK.as_secs_f64()).round() as u64;
        let interval_samples = interval_ms * samp_rate_hz / 1000;

        if interval_ms != last_interval_ms {
            last_interval_ms = interval_ms;
            next_packet_at   = produced_samples;
        }

        // ── TX scheduler ─────────────────────────────────────────────────
        // Collect all completed modulation results into the local queue.
        while let Ok(res) = tx_res_recv.try_recv() {
            jobs_in_flight = jobs_in_flight.saturating_sub(1);
            if res.tx_gen == tx_gen {
                tx_queue.push_back(res);
            }
            // Stale result from before a reset — silently discard.
        }

        // ── Push packets into the driver ──────────────────────────────────
        // Hard cap: never let the driver buffer grow beyond 4 ticks of samples
        // (prevents tx_count racing ahead of rx_count at interval=0).
        let max_pending      = samples_per_tick as usize * 4;
        // Fill target for interval=0: keep at least one full tick pre-buffered
        // so the driver never runs dry mid-tick (which would create silence gaps).
        let fill_target      = samples_per_tick as usize;

        let mut starved = false;
        if next_packet_at <= produced_samples {
            loop {
                if driver.pending_samples() >= max_pending { break; }
                match tx_queue.pop_front() {
                    Some(res) => {
                        let pkt_seq  = u16::from_le_bytes([res.payload[0], res.payload[1]]);
                        let pkt_text = String::from_utf8_lossy(&res.payload[2..]).to_string();
                        {
                            let mut s = shared.stats.lock().unwrap();
                            s.tx_count += 1;
                            s.last_tx   = format!("#{pkt_seq} {pkt_text}");
                        }
                        driver.push_samples(res.clean);
                        next_packet_at = produced_samples + interval_samples;

                        // For non-zero intervals: one push per interval tick.
                        // For interval=0: keep looping until the fill target is met.
                        if interval_samples > 0 { break; }
                        if driver.pending_samples() >= fill_target { break; }
                    }
                    None => { starved = true; break; }
                }
            }
        }
        shared.tx_starved.store(starved, Ordering::Relaxed);

        // Replenish the pipeline based on what the push loop just consumed.
        // Dispatch must happen *after* the push so we see the post-push queue
        // depth and send exactly as many new jobs as were consumed this tick.
        while tx_queue.len() + jobs_in_flight < TX_PIPELINE_DEPTH {
            dispatch_tx!(sf, os_factor);
        }

        // ── Driver tick: produce one tick's worth of mixed samples ────────
        let n     = samples_per_tick as usize;
        let mixed = driver.tick(n);
        // Advance by actual length: hardware drivers may return more than n
        // (they drain all buffered samples to prevent backlog accumulation).
        produced_samples += mixed.len() as u64;

        // ── Mixed samples → RX worker (continuous stream) ────────────────
        let _ = rx_send.send(RxJob {
            sf, cr: 4, os_factor,
            samples: mixed.clone(),
            tx_gen,
        });

        // ── Mixed samples → display thread ───────────────────────────────
        let target_rows = (n / fft_size).max(1).min(MAX_ROWS_PER_TICK);
        let avail_rows  = mixed.len() / fft_size;
        let rows        = target_rows.min(avail_rows);

        if let Some(ref ds) = disp_send {
            for i in 0..rows {
                let window = mixed[i * fft_size..(i + 1) * fft_size].to_vec();
                let _ = ds.send((window, i + 1 == rows));
            }
        }

        // ── Buffer status ─────────────────────────────────────────────────
        let pending   = driver.pending_samples();
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
