/// LoRa GUI simulator — raw spectrum + waterfall of a continuous IQ stream.
///
/// Architecture
/// ───────────
/// The sim thread maintains a `VecDeque<Complex<f32>>` IQ buffer fed by a
/// packet scheduler.  Every ~16 ms tick it drains one FFT window from the
/// buffer (or synthesises a pure-noise window when empty) and pushes it to the
/// waterfall/spectrum plots.
///
/// * interval → ∞  →  steady noise roll, no signal.
/// * interval → 0  →  continuous back-to-back LoRa packets with no gaps.
///
/// Signal and noise amplitudes are set independently; SNR is inferred.
///
/// Usage: gui_sim [sf] [snr_db]   (defaults: sf=7  snr=10)

use lora::tx::{
    whitening::whiten, header::add_header, crc::add_crc,
    hamming_enc::hamming_enc, interleaver::interleave,
    gray_demap::gray_demap, modulate::modulate,
};
use lora::rx::{
    frame_sync::frame_sync, fft_demod::fft_demod,
    gray_mapping::gray_map, deinterleaver::deinterleave,
    hamming_dec::hamming_dec, header_decoder::decode_header,
    crc_verif::verify_crc, dewhitening::dewhiten,
};
use lora::ui::{Chart, SpectrumPlot, WaterfallPlot};

use eframe::egui;
use egui::{Slider, Vec2};
use rand::{Rng, RngExt, SeedableRng};
use rustfft::{FftPlanner, num_complex::Complex};
use std::{
    collections::VecDeque,
    f64::consts::TAU,
    sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU32, Ordering}},
    time::{Duration, Instant},
};

// ─── Options ──────────────────────────────────────────────────────────────────

const SR_OPTIONS_KHZ: &[u32] = &[125, 250, 500, 1000, 2000, 4000];
const BW_OPTIONS_KHZ: &[u32] = &[125, 250, 500, 1000];

const DEFAULT_SF:           u8    = 7;
const DEFAULT_SAMP_RATE_KHZ: u32  = 1000;
const DEFAULT_BW_KHZ:        u32  = 250;
const DEFAULT_FFT_SIZE:      usize = 1024;
const DEFAULT_SIGNAL_DB:     f32   = -20.0;
const DEFAULT_NOISE_DB:      f32   = -35.0;
const DEFAULT_INTERVAL_MS:   u64   = 500;

/// Tick interval for the stream thread (~60 fps waterfall scroll).
const TICK: Duration = Duration::from_millis(16);

/// Hard cap on FFT rows rendered per tick to prevent runaway scroll at very
/// high sample rates.  At 4 MHz SR + 1024-pt FFT the natural rate is ~62 rows
/// per 16 ms tick, which is already visible; this only kicks in above that.
const MAX_ROWS_PER_TICK: usize = 128;

fn khz_label(v: u32) -> String {
    if v >= 1000 { format!("{}M", v / 1000) } else { format!("{}k", v) }
}

fn effective_sr_and_os(samp_rate_khz: u32, bw_khz: u32) -> (u32, u32) {
    let sr = samp_rate_khz.max(bw_khz);
    (sr, sr / bw_khz)
}

/// Convert amplitude dBFS to linear amplitude.
fn db_to_amp(db: f32) -> f32 { 10f32.powf(db / 20.0) }

/// SNR in dB from signal and noise amplitude dBFS values.
fn snr_db(signal_db: f32, noise_db: f32) -> f32 { signal_db - noise_db }

// ─── TX ───────────────────────────────────────────────────────────────────────

struct Tx {
    sf:        u8,
    cr:        u8,
    os_factor: u32,
}

impl Tx {
    fn new(sf: u8, cr: u8, os_factor: u32) -> Self { Self { sf, cr, os_factor } }

    fn modulate(&self, payload: &[u8]) -> Vec<Complex<f32>> {
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

// ─── Channel ──────────────────────────────────────────────────────────────────

/// Free-running AWGN channel.  Silence and TX bursts are both pushed as
/// explicit IQ samples; `read()` just drains the pre-filled queue.
///
/// Future device mode: replace `push_silence` / `inject` / `read` with SDR
/// driver calls.
struct Channel {
    noise_sigma: f32,
    pending:     VecDeque<Complex<f32>>,
}

impl Channel {
    fn new(noise_sigma: f32) -> Self {
        Self { noise_sigma, pending: VecDeque::new() }
    }

    fn set_noise_sigma(&mut self, s: f32) { self.noise_sigma = s; }
    fn pending_len(&self) -> usize        { self.pending.len() }
    fn clear(&mut self)                   { self.pending.clear(); }

    /// Push pre-built mixed IQ directly into the stream (no per-sample work).
    fn push_prebuilt(&mut self, samples: &[Complex<f32>]) {
        self.pending.extend(samples.iter().copied());
    }

    /// Push `n` samples of pure AWGN (silence / inter-packet gap) into the stream.
    fn push_silence(&mut self, n: usize, rng: &mut impl Rng) {
        for _ in 0..n { self.pending.push_back(self.noise_sample(rng)); }
    }

    /// Drain `n` samples from the stream, synthesising pure AWGN for any
    /// sample that has no pending signal.
    fn read(&mut self, n: usize, rng: &mut impl Rng) -> Vec<Complex<f32>> {
        (0..n)
            .map(|_| self.pending.pop_front().unwrap_or_else(|| self.noise_sample(rng)))
            .collect()
    }

    fn noise_sample(&self, rng: &mut impl Rng) -> Complex<f32> {
        Complex::new(
            box_muller(rng) * self.noise_sigma,
            box_muller(rng) * self.noise_sigma,
        )
    }
}

/// Box-Muller: one unit-normal sample from two uniform draws.
fn box_muller(rng: &mut impl Rng) -> f32 {
    let u1 = (rng.random::<f32>() + 1e-7_f32).min(1.0);
    let u2 = rng.random::<f32>();
    (-2.0_f32 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

// ─── RX ───────────────────────────────────────────────────────────────────────

struct Rx {
    sf:        u8,
    cr:        u8,
    os_factor: u32,
}

impl Rx {
    fn new(sf: u8, cr: u8, os_factor: u32) -> Self { Self { sf, cr, os_factor } }

    fn decode(&self, iq: &[Complex<f32>]) -> Option<Vec<u8>> {
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

// ─── Raw spectrum ─────────────────────────────────────────────────────────────

/// Hann-windowed FFT of exactly one `fft_size`-sample window, fftshifted.
/// Returns `Vec<[f64;2]>`: x = bin (0..fft_size), y = power dB.
fn spectrum_window(
    samples:  &[Complex<f32>],
    hann:     &[f32],
    buf:      &mut Vec<Complex<f32>>,
    fft:      &dyn rustfft::Fft<f32>,
) -> Vec<[f64; 2]> {
    let n = buf.len();
    for (b, (s, h)) in buf.iter_mut().zip(samples.iter().zip(hann.iter())) {
        *b = s * h;
    }
    fft.process(buf);
    let half = n / 2;
    (0..n).map(|i| {
        let src = (i + half) % n;
        let pdb = 10.0 * (buf[src].norm_sqr() as f64 + 1e-20_f64).log10();
        [i as f64, pdb.max(-120.0)]
    }).collect()
}

// ─── Worker threads ───────────────────────────────────────────────────────────

/// TX generation request: the worker does modulate + AWGN mixing off-thread.
struct TxJob {
    sf:          u8,
    cr:          u8,
    os_factor:   u32,
    payload:     Vec<u8>,
    signal_amp:  f32,   // snapshot at dispatch time
    noise_sigma: f32,
    rng_seed:    u64,   // reproducible per-packet noise
}

/// Result returned by tx_worker: pre-built mixed IQ ready to inject.
struct TxResult {
    payload: Vec<u8>,
    mixed:   Vec<Complex<f32>>,
}

/// Runs TX modulation and AWGN mixing off the sim_loop critical path.
fn tx_worker(
    jobs:    std::sync::mpsc::Receiver<TxJob>,
    results: std::sync::mpsc::SyncSender<TxResult>,
) {
    let mut tx = Tx::new(7, 4, 4);
    for job in jobs {
        if job.sf != tx.sf || job.os_factor != tx.os_factor {
            tx = Tx::new(job.sf, job.cr, job.os_factor);
        }
        let iq = tx.modulate(&job.payload);
        let mut rng = rand::rngs::StdRng::seed_from_u64(job.rng_seed);
        let mixed: Vec<_> = iq.iter()
            .map(|&s| {
                let n = Complex::new(
                    box_muller(&mut rng) * job.noise_sigma,
                    box_muller(&mut rng) * job.noise_sigma,
                );
                s * job.signal_amp + n
            })
            .collect();
        // If the channel is full the sim is shutting down; drop gracefully.
        let _ = results.send(TxResult { payload: job.payload, mixed });
    }
}

/// Packet handed from the fill thread to the RX decode thread.
struct RxJob {
    sf:        u8,
    cr:        u8,
    os_factor: u32,
    payload:   Vec<u8>,
    mixed:     Vec<Complex<f32>>,
}

/// Decodes incoming packets off the critical path and updates stats / log.
fn rx_worker(jobs: std::sync::mpsc::Receiver<RxJob>, shared: Arc<SimShared>) {
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

/// (IQ window, is_last_in_batch): last window also updates the spectrum plot.
type DisplayJob = (Vec<Complex<f32>>, bool);

/// Runs Hann-windowed FFT on each incoming window and pushes to the plots.
fn display_worker(jobs: std::sync::mpsc::Receiver<DisplayJob>, shared: Arc<SimShared>) {
    let mut cur_fft_size = 0usize;
    let mut hann: Vec<f32>             = Vec::new();
    let mut fft_buf: Vec<Complex<f32>> = Vec::new();
    let mut planner = FftPlanner::<f32>::new();
    let mut fft: Arc<dyn rustfft::Fft<f32>> = planner.plan_fft_forward(1);

    while let Ok((window, is_last)) = jobs.recv() {
        let fft_size = window.len();
        if fft_size != cur_fft_size {
            cur_fft_size = fft_size;
            hann = (0..fft_size)
                .map(|i| (0.5 * (1.0 - (TAU * i as f64 / fft_size as f64).cos())) as f32)
                .collect();
            fft_buf = vec![Complex::new(0.0_f32, 0.0_f32); fft_size];
            fft = planner.plan_fft_forward(fft_size);
        }
        let spec = spectrum_window(&window, &hann, &mut fft_buf, fft.as_ref());
        shared.waterfall_plot.update(spec.clone());
        if is_last { shared.spectrum_plot.update(spec); }
    }
}

// ─── Shared simulation state ──────────────────────────────────────────────────

struct SimShared {
    running:        AtomicBool,
    clear_buf:      AtomicBool,   // flush IQ buffer on settings change
    sf:             Mutex<u8>,
    os_factor:      Mutex<u32>,
    samp_rate_khz:  Mutex<u32>,
    fft_size:       Mutex<usize>,
    signal_db:      Mutex<f32>,   // signal amplitude in dBFS
    noise_db:       Mutex<f32>,   // noise amplitude in dBFS
    interval_ms:    Mutex<u64>,
    spectrum_plot:  Arc<SpectrumPlot>,
    waterfall_plot: Arc<WaterfallPlot>,
    stats:          Mutex<Stats>,
    log:            Mutex<VecDeque<LogEntry>>,
    /// Display-buffer lag in ms (f32 bits stored in AtomicU32 for lock-free reads).
    buf_lag_ms:     AtomicU32,
    buf_overflow:   AtomicBool,
    buf_underflow:  AtomicBool,
}

#[derive(Default, Clone)]
struct Stats {
    total:   usize,
    ok:      usize,
    last_tx: String,
    last_rx: String,
}

#[derive(Clone)]
struct LogEntry {
    ok:      bool,
    payload: String,
}

const MAX_LOG_ENTRIES: usize = 200;

// ─── Sim thread ───────────────────────────────────────────────────────────────

fn sim_loop(shared: Arc<SimShared>, ctx: Option<egui::Context>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x10_4a);
    let payloads: &[&[u8]] = &[
        b"Hello, LoRa!", b"Rust 2024", b"AWGN channel",
        b"burst test",   b"lora-rs",   b"packet!",
    ];
    let mut idx = 0usize;

    // Virtual sample clock.
    let mut filled_samples:   u64 = 0;
    let mut next_packet_at:   u64 = 0;
    let mut last_interval_ms: u64 = u64::MAX;
    let mut tx_inflight  = false;
    // Packet samples not yet pushed to channel.pending; drained one tick at a time
    // to keep the display buffer shallow and settings changes near-instant.
    let mut tx_remainder: std::collections::VecDeque<Complex<f32>> = std::collections::VecDeque::new();

    // Spawn worker threads.
    let (rx_send,   rx_recv)   = std::sync::mpsc::channel::<RxJob>();
    // sync_channel(1): tx_worker stays at most one packet ahead.
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

    let init_noise_db = *shared.noise_db.lock().unwrap();
    let mut channel = Channel::new(
        db_to_amp(init_noise_db) / std::f32::consts::SQRT_2,
    );

    // Helper: dispatch the next TX job to the worker thread.
    macro_rules! dispatch_tx {
        ($sf:expr, $os:expr, $sig:expr, $ns:expr) => {{
            let payload = payloads[idx % payloads.len()].to_vec();
            idx += 1;
            let _ = tx_job_send.send(TxJob {
                sf: $sf, cr: 4, os_factor: $os,
                payload,
                signal_amp:  $sig,
                noise_sigma: $ns,
                rng_seed:    rng.random(),
            });
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
            tx_remainder.clear();
            filled_samples   = 0;
            next_packet_at   = 0;
            last_interval_ms = u64::MAX;
            // Drain any stale result and dispatch a fresh job with new settings.
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

        // push_silence still uses the channel's noise settings.
        channel.set_noise_sigma(noise_sigma);

        // ── TX: sample-accurate fill ──────────────────────────────────────
        let samp_rate_hz     = samp_rate_khz as u64 * 1000;
        let samples_per_tick = (samp_rate_hz as f64 * TICK.as_secs_f64()).round() as u64;
        let interval_samples = interval_ms * samp_rate_hz / 1000;

        if interval_ms != last_interval_ms {
            last_interval_ms = interval_ms;
            next_packet_at   = filled_samples;
        }

        // Ensure a job is always in flight so the worker stays busy.
        if !tx_inflight {
            dispatch_tx!(sf, os_factor, signal_amp, noise_sigma);
        }

        let target = filled_samples + samples_per_tick;

        while filled_samples < target {
            // Phase 1: trickle leftover packet samples (one tick's worth at a time).
            if !tx_remainder.is_empty() {
                let n = (target - filled_samples).min(tx_remainder.len() as u64) as usize;
                // drain n items from the front into a temporary slice
                let batch: Vec<_> = (0..n).map(|_| tx_remainder.pop_front().unwrap()).collect();
                channel.push_prebuilt(&batch);
                filled_samples += n as u64;
                continue;
            }

            if next_packet_at <= filled_samples {
                // Try to collect the pre-built packet; fill silence if not ready yet.
                match tx_res_recv.try_recv() {
                    Ok(res) => {
                        let pkt_len = res.mixed.len() as u64;
                        // Fix next_packet_at BEFORE modifying filled_samples.
                        next_packet_at = filled_samples + pkt_len + interval_samples;

                        // Push the first chunk this tick; rest goes to trickle buffer.
                        let first_n = (target - filled_samples).min(pkt_len) as usize;
                        channel.push_prebuilt(&res.mixed[..first_n]);
                        filled_samples += first_n as u64;
                        if first_n < res.mixed.len() {
                            tx_remainder.extend(res.mixed[first_n..].iter().copied());
                        }

                        // Hand full mixed IQ to RX thread and pre-build the next packet.
                        let _ = rx_send.send(RxJob {
                            sf, cr: 4, os_factor,
                            payload: res.payload,
                            mixed:   res.mixed,
                        });
                        dispatch_tx!(sf, os_factor, signal_amp, noise_sigma);
                    }
                    Err(_) => {
                        // Worker still generating; fill this tick with silence.
                        let n = (target - filled_samples) as usize;
                        channel.push_silence(n, &mut rng);
                        filled_samples = target;
                    }
                }
            } else {
                let silence_end = next_packet_at.min(target);
                let n = (silence_end - filled_samples) as usize;
                channel.push_silence(n, &mut rng);
                filled_samples += n as u64;
            }
        }

        // ── Channel → display thread: drain one tick's worth of FFT rows ─
        let target_rows = (samples_per_tick as usize / fft_size).max(1).min(MAX_ROWS_PER_TICK);
        let avail_rows  = channel.pending_len() / fft_size;
        let rows        = target_rows.min(avail_rows.max(1)); // always at least 1 (noise fallback)

        if let Some(ref ds) = disp_send {
            for i in 0..rows {
                let window = channel.read(fft_size, &mut rng);
                let _ = ds.send((window, i + 1 == rows));
            }
        } else {
            // Headless: drain the channel to prevent unbounded growth.
            for _ in 0..rows { channel.read(fft_size, &mut rng); }
        }

        // ── Buffer status ─────────────────────────────────────────────────
        // Total unplayed samples = pending + trickle remainder.
        let unplayed   = channel.pending_len() + tx_remainder.len();
        let lag_ms     = unplayed as f32 * 1000.0 / samp_rate_hz as f32;
        let overflow   = avail_rows > target_rows * 8; // >8× a tick's worth backed up
        let underflow  = avail_rows == 0;
        shared.buf_lag_ms   .store(lag_ms.to_bits(),         Ordering::Relaxed);
        shared.buf_overflow .store(overflow,                  Ordering::Relaxed);
        shared.buf_underflow.store(underflow,                 Ordering::Relaxed);

        shared.waterfall_plot.set_freq(fft_size as f64 / 2.0);
        shared.waterfall_plot.set_bw(fft_size as f64);

        if let Some(ref c) = ctx { c.request_repaint(); }

        let elapsed = tick_start.elapsed();
        if elapsed < TICK { std::thread::sleep(TICK - elapsed); }
    }
}

// ─── GUI app ──────────────────────────────────────────────────────────────────

struct GuiApp {
    shared:          Arc<SimShared>,
    spectrum_chart:  Chart,
    waterfall_chart: Chart,
    thread_started:  bool,
    signal_db:     f32,
    noise_db:      f32,
    interval_ms:   u64,
    sf:            u8,
    samp_rate_khz: u32,
    bw_khz:        u32,
    fft_size:      usize,
}

impl GuiApp {
    fn new(sf: u8) -> Self {
        let samp_rate_khz  = DEFAULT_SAMP_RATE_KHZ;
        let bw_khz         = DEFAULT_BW_KHZ;
        let (_, os_factor) = effective_sr_and_os(samp_rate_khz, bw_khz);
        let fft_size       = DEFAULT_FFT_SIZE;
        let signal_db      = DEFAULT_SIGNAL_DB;
        let noise_db       = DEFAULT_NOISE_DB;

        let init_spec: Vec<[f64; 2]> = (0..fft_size).map(|i| [i as f64, -80.0]).collect();

        let spectrum_plot  = SpectrumPlot::new("Spectrum",   init_spec.clone(), -80.0, 80.0);
        let waterfall_plot = WaterfallPlot::new("Waterfall", init_spec,         -80.0);
        waterfall_plot.set_freq(fft_size as f64 / 2.0);
        waterfall_plot.set_bw(fft_size as f64);

        let mut spectrum_chart = Chart::new("spectrum");
        spectrum_chart.set_x_limits([0.0, fft_size as f64]);
        spectrum_chart.set_y_limits([-90.0, 50.0]);
        spectrum_chart.set_link_axis("lora_link", true, false);
        spectrum_chart.set_link_cursor("lora_link", true, false);
        spectrum_chart.add(spectrum_plot.clone());

        let mut waterfall_chart = Chart::new("waterfall");
        waterfall_chart.set_x_limits([0.0, fft_size as f64]);
        waterfall_chart.set_y_limits([0.0, 1.0]);
        waterfall_chart.set_link_axis("lora_link", true, false);
        waterfall_chart.set_link_cursor("lora_link", true, false);
        waterfall_chart.add(waterfall_plot.clone());

        let (eff_sr, _) = effective_sr_and_os(samp_rate_khz, bw_khz);

        let shared = Arc::new(SimShared {
            running:        AtomicBool::new(true),
            clear_buf:      AtomicBool::new(false),
            sf:             Mutex::new(sf),
            os_factor:      Mutex::new(os_factor),
            samp_rate_khz:  Mutex::new(eff_sr),
            fft_size:       Mutex::new(fft_size),
            signal_db:      Mutex::new(signal_db),
            noise_db:       Mutex::new(noise_db),
            interval_ms:    Mutex::new(DEFAULT_INTERVAL_MS),
            spectrum_plot,
            waterfall_plot,
            stats:          Mutex::new(Stats::default()),
            log:            Mutex::new(VecDeque::new()),
            buf_lag_ms:     AtomicU32::new(0),
            buf_overflow:   AtomicBool::new(false),
            buf_underflow:  AtomicBool::new(false),
        });

        Self {
            shared,
            spectrum_chart,
            waterfall_chart,
            thread_started: false,
            signal_db,
            noise_db,
            interval_ms: DEFAULT_INTERVAL_MS,
            sf,
            samp_rate_khz,
            bw_khz,
            fft_size,
        }
    }

    fn restore_defaults(&mut self) {
        self.sf            = DEFAULT_SF;
        self.samp_rate_khz = DEFAULT_SAMP_RATE_KHZ;
        self.bw_khz        = DEFAULT_BW_KHZ;
        self.fft_size      = DEFAULT_FFT_SIZE;
        self.signal_db     = DEFAULT_SIGNAL_DB;
        self.noise_db      = DEFAULT_NOISE_DB;
        self.interval_ms   = DEFAULT_INTERVAL_MS;
        *self.shared.signal_db.lock().unwrap()   = self.signal_db;
        *self.shared.noise_db.lock().unwrap()    = self.noise_db;
        *self.shared.interval_ms.lock().unwrap() = self.interval_ms;
        self.rebuild_plots();
    }

    fn rebuild_plots(&mut self) {
        let (eff_sr, os_factor) = effective_sr_and_os(self.samp_rate_khz, self.bw_khz);
        self.samp_rate_khz = eff_sr;
        self.spectrum_chart.set_x_limits([0.0, self.fft_size as f64]);
        self.waterfall_chart.set_x_limits([0.0, self.fft_size as f64]);
        *self.shared.sf.lock().unwrap()            = self.sf;
        *self.shared.os_factor.lock().unwrap()     = os_factor;
        *self.shared.samp_rate_khz.lock().unwrap() = eff_sr;
        *self.shared.fft_size.lock().unwrap()      = self.fft_size;
        self.shared.waterfall_plot.set_freq(self.fft_size as f64 / 2.0);
        self.shared.waterfall_plot.set_bw(self.fft_size as f64);
        *self.shared.stats.lock().unwrap() = Stats::default();
        self.shared.log.lock().unwrap().clear();
        self.shared.clear_buf.store(true, Ordering::Relaxed);
    }
}

/// Consume the remaining row space with a zero-height spacer when the next
/// group won't fully fit, so `horizontal_wrapped` starts a fresh row.
///
/// Uses cursor + max_rect to measure actual remaining width, which is more
/// reliable than `available_width()` inside a wrapped layout.
fn newline_if_needed(ui: &mut egui::Ui, needed: f32) {
    let cursor_x = ui.cursor().min.x;
    let left_x   = ui.max_rect().min.x;
    let right_x  = ui.max_rect().max.x;
    // Don't trigger when we're already at (or very close to) the start of a row.
    // Use 2 × item_spacing as the threshold to handle any post-wrap cursor offset.
    if cursor_x <= left_x + ui.spacing().item_spacing.x * 2.0 { return; }
    let remaining = right_x - cursor_x;
    // Nothing to do if already past the right edge (wrap will happen naturally).
    if remaining <= 0.0 { return; }
    if remaining < needed {
        // Overshoot by 1 px so the cursor lands strictly past right_x, which
        // guarantees horizontal_wrapped starts a new row for the next widget.
        ui.allocate_exact_size(egui::vec2(remaining + 1.0, 0.0), egui::Sense::hover());
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.thread_started {
            self.thread_started = true;
            let shared = self.shared.clone();
            let ctx2   = ctx.clone();
            std::thread::spawn(move || sim_loop(shared, Some(ctx2)));
        }

        let running = self.shared.running.load(Ordering::Relaxed);
        let stats   = self.shared.stats.lock().unwrap().clone();
        let log     = self.shared.log.lock().unwrap().clone();

        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            // ── Row 1: signal parameters ──────────────────────────────────────
            ui.horizontal_wrapped(|ui| {
                if ui.button(if running { "⏸ Pause" } else { "▶ Run" }).clicked() {
                    self.shared.running.store(!running, Ordering::Relaxed);
                }

                ui.separator();

                let mut changed = false;

                newline_if_needed(ui, 230.0);
                ui.horizontal(|ui| {
                    ui.label("SF:");
                    for s in [7u8, 8, 9, 10, 11, 12] {
                        if ui.selectable_label(self.sf == s, format!("{s}")).clicked()
                            && self.sf != s
                        { self.sf = s; changed = true; }
                    }
                });

                ui.separator();

                newline_if_needed(ui, 270.0);
                ui.horizontal(|ui| {
                    ui.label("SR:");
                    for &sr in SR_OPTIONS_KHZ {
                        if ui.selectable_label(self.samp_rate_khz == sr, khz_label(sr)).clicked()
                            && self.samp_rate_khz != sr
                        { self.samp_rate_khz = sr; changed = true; }
                    }
                });

                ui.separator();

                newline_if_needed(ui, 250.0);
                ui.horizontal(|ui| {
                    ui.label("BW:");
                    for &bw in BW_OPTIONS_KHZ {
                        if ui.selectable_label(self.bw_khz == bw, khz_label(bw)).clicked()
                            && self.bw_khz != bw
                        { self.bw_khz = bw; changed = true; }
                    }
                    let (eff_sr, os) = effective_sr_and_os(self.samp_rate_khz, self.bw_khz);
                    ui.label(if eff_sr != self.samp_rate_khz {
                        format!("×{os}↑")
                    } else {
                        format!("×{os}")
                    });
                });

                ui.separator();

                newline_if_needed(ui, 250.0);
                ui.horizontal(|ui| {
                    ui.label("FFT:");
                    for &sz in &[1024usize, 2048, 4096, 8192] {
                        if ui.selectable_label(self.fft_size == sz, format!("{sz}")).clicked()
                            && self.fft_size != sz
                        { self.fft_size = sz; changed = true; }
                    }
                });

                if changed { self.rebuild_plots(); }

                ui.separator();

                if ui.button("⟳ Defaults").clicked() {
                    self.restore_defaults();
                }
            });

            // ── Row 2: levels + interval ──────────────────────────────────────
            ui.horizontal_wrapped(|ui| {
                ui.horizontal(|ui| {
                    ui.label("Sig:");
                    let h = ui.spacing().interact_size.y;
                    if ui.add_sized([120.0, h], Slider::new(&mut self.signal_db, -40.0..=20.0)
                        .suffix(" dBFS").step_by(1.0)).changed()
                    {
                        *self.shared.signal_db.lock().unwrap() = self.signal_db;
                    }
                });

                ui.separator();

                newline_if_needed(ui, 190.0);
                ui.horizontal(|ui| {
                    ui.label("Noise:");
                    let h = ui.spacing().interact_size.y;
                    if ui.add_sized([120.0, h], Slider::new(&mut self.noise_db, -80.0..=0.0)
                        .suffix(" dBFS").step_by(1.0)).changed()
                    {
                        *self.shared.noise_db.lock().unwrap() = self.noise_db;
                    }
                });

                newline_if_needed(ui, 110.0);
                ui.label(format!("SNR: {:.0} dB", snr_db(self.signal_db, self.noise_db)));

                ui.separator();

                newline_if_needed(ui, 220.0);
                ui.horizontal(|ui| {
                    ui.label("Interval:");
                    let h = ui.spacing().interact_size.y;
                    let mut interval_f = self.interval_ms as f32;
                    if ui.add_sized([120.0, h], Slider::new(&mut interval_f, 0.0..=5000.0)
                        .suffix(" ms").step_by(50.0)).changed()
                    {
                        self.interval_ms = interval_f as u64;
                        *self.shared.interval_ms.lock().unwrap() = self.interval_ms;
                    }
                });
            });

            // ── Row 3: buffer status ──────────────────────────────────────────
            ui.horizontal(|ui| {
                let lag_ms   = f32::from_bits(self.shared.buf_lag_ms  .load(Ordering::Relaxed));
                let overflow  = self.shared.buf_overflow .load(Ordering::Relaxed);
                let underflow = self.shared.buf_underflow.load(Ordering::Relaxed);

                let (color, label) = if underflow {
                    (egui::Color32::from_rgb(220, 160, 0), "⚠ UNDERFLOW")
                } else if overflow {
                    (egui::Color32::from_rgb(220, 60, 60),  "⚠ OVERFLOW")
                } else {
                    (egui::Color32::from_rgb(100, 180, 100), "●")
                };
                ui.colored_label(color, label);
                ui.label(format!("buf {lag_ms:.0} ms"));
            });
        });

        egui::SidePanel::right("msg_log")
            .min_width(200.0)
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Messages");
                ui.separator();

                // ── Stats + reset ─────────────────────────────────────────────
                let per = if stats.total > 0 {
                    100.0 * (stats.total - stats.ok) as f32 / stats.total as f32
                } else { 0.0 };
                ui.horizontal(|ui| {
                    ui.label(format!("{}/{} ok", stats.ok, stats.total));
                    ui.label(format!("PER {:.0}%", per));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("↺ Reset").clicked() {
                            *self.shared.stats.lock().unwrap() = Stats::default();
                            self.shared.log.lock().unwrap().clear();
                        }
                    });
                });
                ui.separator();

                // ── Scrolling message log ─────────────────────────────────────
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for entry in &log {
                            ui.horizontal(|ui| {
                                let (dot, color) = if entry.ok {
                                    ("●", egui::Color32::from_rgb(80, 200, 80))
                                } else {
                                    ("●", egui::Color32::from_rgb(220, 60, 60))
                                };
                                ui.colored_label(color, dot);
                                ui.label(&entry.payload);
                            });
                        }
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let h = ui.available_height();
            let w = ui.available_width();
            ui.allocate_ui(Vec2::new(w, h * 0.40), |ui| self.spectrum_chart.ui(ui));
            ui.allocate_ui(Vec2::new(w, h * 0.60), |ui| self.waterfall_chart.ui(ui));
        });

        // Repaint is driven by the sim thread via ctx.request_repaint().
    }
}

// ─── CLI / headless mode ──────────────────────────────────────────────────────

/// Run the simulator headlessly and print per-packet results to stdout.
///
/// Useful for automated tests and CI: exits once `packet_count` packets have
/// been decoded (success or failure).
pub fn run_headless(sf: u8, snr_db_val: f32, packet_count: usize) {
    let signal_db = DEFAULT_SIGNAL_DB;
    let noise_db  = signal_db - snr_db_val;
    let samp_rate_khz = DEFAULT_SAMP_RATE_KHZ;
    let bw_khz        = DEFAULT_BW_KHZ;
    let (eff_sr, os_factor) = effective_sr_and_os(samp_rate_khz, bw_khz);
    let fft_size = DEFAULT_FFT_SIZE;

    let init_spec: Vec<[f64; 2]> = (0..fft_size).map(|i| [i as f64, -80.0]).collect();
    let spectrum_plot  = SpectrumPlot::new("Spectrum",   init_spec.clone(), -80.0, 80.0);
    let waterfall_plot = WaterfallPlot::new("Waterfall", init_spec,         -80.0);

    let shared = Arc::new(SimShared {
        running:        AtomicBool::new(true),
        clear_buf:      AtomicBool::new(false),
        sf:             Mutex::new(sf),
        os_factor:      Mutex::new(os_factor),
        samp_rate_khz:  Mutex::new(eff_sr),
        fft_size:       Mutex::new(fft_size),
        signal_db:      Mutex::new(signal_db),
        noise_db:       Mutex::new(noise_db),
        interval_ms:    Mutex::new(0),   // back-to-back packets
        spectrum_plot,
        waterfall_plot,
        stats:          Mutex::new(Stats::default()),
        log:            Mutex::new(VecDeque::new()),
        buf_lag_ms:     AtomicU32::new(0),
        buf_overflow:   AtomicBool::new(false),
        buf_underflow:  AtomicBool::new(false),
    });

    { let s = shared.clone(); std::thread::spawn(move || sim_loop(s, None)); }

    let mut printed = 0usize;
    loop {
        std::thread::sleep(Duration::from_millis(50));
        let log = shared.log.lock().unwrap().clone();
        // Print any entries we haven't shown yet.
        for entry in log.iter().skip(printed) {
            let mark = if entry.ok { "OK  " } else { "FAIL" };
            println!("[{mark}] {}", entry.payload);
        }
        printed = log.len();

        let stats = shared.stats.lock().unwrap().clone();
        if stats.total >= packet_count {
            shared.running.store(false, Ordering::Relaxed);
            let per = if stats.total > 0 {
                100.0 * (stats.total - stats.ok) as f32 / stats.total as f32
            } else { 0.0 };
            println!("─── SF={sf}  SNR={snr_db_val:.1} dB  {}/{} ok  PER {per:.1}% ───",
                     stats.ok, stats.total);
            break;
        }
    }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Check for --cli flag anywhere in the args list.
    let cli_mode = args.iter().any(|a| a == "--cli" || a == "-c");

    if cli_mode {
        // Usage: gui_sim --cli [sf=7] [snr_db=10] [packets=20]
        let positional: Vec<&str> = args.iter()
            .filter(|a| !a.starts_with('-') && *a != &args[0])
            .map(|s| s.as_str())
            .collect();
        let sf         = positional.first().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_SF);
        let snr_db     = positional.get(1)  .and_then(|s| s.parse().ok()).unwrap_or(10.0_f32);
        let packets    = positional.get(2)  .and_then(|s| s.parse().ok()).unwrap_or(20_usize);
        run_headless(sf, snr_db, packets);
        return;
    }

    let sf = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_SF);

    eframe::run_native(
        "LoRa Link Simulator",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_title("LoRa Link Simulator")
                .with_inner_size([1400.0, 750.0]),
            ..Default::default()
        },
        Box::new(move |_cc| Ok(Box::new(GuiApp::new(sf)))),
    ).unwrap();
}
