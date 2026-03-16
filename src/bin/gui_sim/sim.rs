use eframe::egui;
use std::{
    collections::VecDeque,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};

use super::channel::Channel;
use super::display::{DisplayJob, display_worker};
use super::driver::Driver;
use super::rx::{RxJob, rx_worker};
use super::shared::SimShared;
use super::tx::{TxJob, TxResult, tx_worker};
use super::db_to_amp;

#[cfg(feature = "uhd")]
use lora::uhd::UhdDevice;

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
const TX_PIPELINE_DEPTH: usize = 4;

/// Sleep for the remainder of a tick, or a full tick on WASM.
async fn tick_sleep(remaining: Duration) {
    #[cfg(not(target_arch = "wasm32"))]
    tokio::time::sleep(remaining).await;
    #[cfg(target_arch = "wasm32")]
    gloo_timers::future::TimeoutFuture::new(remaining.as_millis() as u32).await;
}

/// Spawn a task on the appropriate executor.
/// Native: tokio thread-pool (requires Send). WASM: JS event loop.
fn spawn_task<F>(f: F)
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    #[cfg(not(target_arch = "wasm32"))]
    { let _ = tokio::task::spawn(f); }
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(f);
}

pub(crate) async fn sim_loop(shared: Arc<SimShared>, ctx: Option<egui::Context>) {
    let payloads: &[&[u8]] = &[
        b"Hello, LoRa!", b"Rust 2024", b"AWGN channel",
        b"burst test",   b"lora-rs",   b"packet!",
    ];
    let mut payload_idx = 0usize;
    let mut seq         = 0u16;
    let mut tx_gen      = 0u64;

    let mut produced_samples: u64 = 0;
    let mut next_packet_at:   u64 = 0;
    let mut last_interval_ms: u64 = u64::MAX;
    let mut last_uhd_rx_gain = f64::NAN;
    let mut last_uhd_tx_gain = f64::NAN;
    let mut tx_queue: VecDeque<TxResult> = VecDeque::new();
    let mut jobs_in_flight: usize = 0;

    // Spawn worker tasks.
    let (rx_send,     rx_recv)     = tokio::sync::mpsc::unbounded_channel::<RxJob>();
    let (tx_job_send, tx_job_recv) = tokio::sync::mpsc::unbounded_channel::<TxJob>();
    let (tx_res_send, tx_res_recv) = tokio::sync::mpsc::unbounded_channel::<TxResult>();
    { let s = shared.clone(); spawn_task(rx_worker(rx_recv, s)); }
    spawn_task(tx_worker(tx_job_recv, tx_res_send));

    // Display task is only needed in GUI mode.
    let disp_send: Option<tokio::sync::mpsc::Sender<DisplayJob>> = if ctx.is_some() {
        let (tx, rx) = tokio::sync::mpsc::channel::<DisplayJob>(256);
        { let s = shared.clone(); spawn_task(display_worker(rx, s)); }
        Some(tx)
    } else {
        None
    };

    // Mutable tx_res_recv — must be declared after spawn_task calls above.
    let mut tx_res_recv = tx_res_recv;

    let mut driver: Box<dyn Driver> = make_driver(&shared);
    let mut display_carry: Vec<rustfft::num_complex::Complex<f32>> = Vec::new();
    #[cfg(feature = "uhd")]
    let mut parked_uhd: Option<(Box<dyn Driver>, String)> = None;
    #[cfg(feature = "uhd")]
    let mut active_uhd_args: String = String::new();

    macro_rules! dispatch_tx {
        ($sf:expr, $os:expr, $sw:expr, $pl:expr) => {{
            let text = payloads[payload_idx % payloads.len()];
            payload_idx += 1;
            let mut payload = Vec::with_capacity(2 + text.len());
            payload.extend_from_slice(&seq.to_le_bytes());
            payload.extend_from_slice(text);
            seq = seq.wrapping_add(1);
            let _ = tx_job_send.send(TxJob {
                sf: $sf, cr: 4, os_factor: $os,
                sync_word: $sw, preamble_len: $pl,
                payload, tx_gen,
            });
            jobs_in_flight += 1;
        }};
    }

    loop {
        #[cfg(not(target_arch = "wasm32"))]
        let tick_start = std::time::Instant::now();

        if shared.quit.load(Ordering::Relaxed) { break; }
        if !shared.running.load(Ordering::Relaxed) {
            tick_sleep(TICK).await;
            continue;
        }

        let should_rebuild = shared.rebuild_driver.swap(false, Ordering::Relaxed);
        let should_clear   = shared.clear_buf.swap(false, Ordering::Relaxed);
        if should_rebuild || should_clear {
            if should_rebuild {
                #[cfg(feature = "uhd")]
                {
                    let use_uhd = shared.use_uhd.load(Ordering::Relaxed);
                    let cur_args = shared.uhd_args.lock().unwrap().clone();
                    if use_uhd {
                        let freq  = *shared.uhd_freq_hz.lock().unwrap();
                        let rxg   = *shared.uhd_rx_gain_db.lock().unwrap();
                        let txg   = *shared.uhd_tx_gain_db.lock().unwrap();
                        let sr_hz = *shared.samp_rate_khz.lock().unwrap() as f64 * 1000.0;
                        let bw_hz = sr_hz / *shared.os_factor.lock().unwrap() as f64;

                        if driver.is_parkable() && active_uhd_args == cur_args {
                            driver.park();
                            driver.unpark(freq, sr_hz, bw_hz, rxg, txg);
                        } else {
                            let reuse = parked_uhd.as_ref()
                                .map(|(_, args)| args == &cur_args)
                                .unwrap_or(false);

                            if reuse {
                                let (mut dev, _) = parked_uhd.take().unwrap();
                                dev.unpark(freq, sr_hz, bw_hz, rxg, txg);
                                driver = dev;
                            } else {
                                parked_uhd = None;
                                driver = Box::new(Channel::new(0.0, 1.0));
                                shared.uhd_loading.store(true, Ordering::Relaxed);
                                if let Some(ref c) = ctx { c.request_repaint(); }
                                let result = UhdDevice::new(&cur_args, freq, sr_hz, bw_hz, rxg, txg);
                                shared.uhd_loading.store(false, Ordering::Relaxed);
                                match result {
                                    Ok(dev) => driver = Box::new(dev),
                                    Err(e)  => {
                                        eprintln!("[uhd] open failed: {e} — falling back to sim");
                                        shared.use_uhd.store(false, Ordering::Relaxed);
                                    }
                                }
                            }
                            active_uhd_args = cur_args;
                        }
                        last_uhd_rx_gain = f64::NAN;
                        last_uhd_tx_gain = f64::NAN;
                    } else {
                        if driver.is_parkable() {
                            let noise = db_to_amp(*shared.noise_db.lock().unwrap()) / std::f32::consts::SQRT_2;
                            let sig   = db_to_amp(*shared.signal_db.lock().unwrap());
                            let mut old = std::mem::replace(&mut driver, Box::new(Channel::new(noise, sig)));
                            old.park();
                            parked_uhd = Some((old, cur_args));
                        } else {
                            driver = make_driver(&shared);
                        }
                    }
                }
                #[cfg(not(feature = "uhd"))]
                {
                    driver = Box::new(Channel::new(0.0, 1.0));
                    driver = make_driver(&shared);
                }
            } else {
                driver.clear();
            }
            tx_queue.clear();
            display_carry.clear();
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
        let sync_word     = *shared.sync_word.lock().unwrap();
        let preamble_len  = *shared.preamble_len.lock().unwrap();

        let signal_amp  = db_to_amp(signal_db);
        let noise_sigma = db_to_amp(noise_db) / std::f32::consts::SQRT_2;

        driver.set_noise_sigma(noise_sigma);
        driver.set_signal_amp(signal_amp);

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

        while let Ok(res) = tx_res_recv.try_recv() {
            jobs_in_flight = jobs_in_flight.saturating_sub(1);
            if res.tx_gen == tx_gen {
                tx_queue.push_back(res);
            }
        }

        let max_pending = samples_per_tick as usize * 4;
        let fill_target = samples_per_tick as usize;

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

                        if interval_samples > 0 { break; }
                        if driver.pending_samples() >= fill_target { break; }
                    }
                    None => { starved = true; break; }
                }
            }
        }
        shared.tx_starved.store(starved, Ordering::Relaxed);

        while tx_queue.len() + jobs_in_flight < TX_PIPELINE_DEPTH {
            dispatch_tx!(sf, os_factor, sync_word, preamble_len);
        }

        let n     = samples_per_tick as usize;
        let mixed = driver.tick(n);
        produced_samples += mixed.len() as u64;

        let _ = rx_send.send(RxJob {
            sf, cr: 4, os_factor, sync_word, preamble_len,
            samples: mixed.clone(),
            tx_gen,
        });

        display_carry.extend_from_slice(&mixed);
        let avail_rows = display_carry.len() / fft_size;
        let rows       = avail_rows.min(MAX_ROWS_PER_TICK);

        if let Some(ref ds) = disp_send {
            if rows > 0 {
                for i in 0..rows {
                    let is_last = i + 1 == rows;
                    let window = display_carry[i * fft_size..(i + 1) * fft_size].to_vec();
                    if is_last {
                        let _ = ds.send((window, true)).await;
                    } else if ds.try_send((window, false)).is_err() {
                        let last = display_carry[(rows-1)*fft_size..rows*fft_size].to_vec();
                        let _ = ds.send((last, true)).await;
                        break;
                    }
                }
            } else {
                let len = display_carry.len().min(fft_size);
                let mut w = display_carry[..len].to_vec();
                w.resize(fft_size, rustfft::num_complex::Complex::new(0.0, 0.0));
                let _ = ds.send((w, true)).await;
            }
        }
        display_carry.drain(..rows * fft_size);

        let pending   = driver.pending_samples();
        let lag_ms    = pending as f32 * 1000.0 / samp_rate_hz as f32;
        let overflow  = mixed.len() / fft_size > (n / fft_size).max(1) * 8;
        let underflow = avail_rows == 0;
        shared.buf_lag_ms   .store(lag_ms.to_bits(), Ordering::Relaxed);
        shared.buf_overflow .store(overflow,          Ordering::Relaxed);
        shared.buf_underflow.store(underflow,         Ordering::Relaxed);

        shared.waterfall_plot.set_freq(fft_size as f64 / 2.0);
        shared.waterfall_plot.set_bw(fft_size as f64);

        if let Some(ref c) = ctx { c.request_repaint(); }

        // Sleep for the remainder of the tick.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let elapsed = tick_start.elapsed();
            if elapsed < TICK { tick_sleep(TICK - elapsed).await; }
        }
        #[cfg(target_arch = "wasm32")]
        tick_sleep(TICK).await;
    }
}
