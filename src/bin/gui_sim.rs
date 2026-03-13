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
    sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}},
    time::{Duration, Instant},
};

// ─── Options ──────────────────────────────────────────────────────────────────

const SR_OPTIONS_KHZ: &[u32] = &[125, 250, 500, 1000, 2000, 4000];
const BW_OPTIONS_KHZ: &[u32] = &[125, 250, 500, 1000];

/// Tick interval for the stream thread (~60 fps waterfall scroll).
const TICK: Duration = Duration::from_millis(16);

/// Maximum FFT rows drained from the IQ buffer per tick.
/// When the buffer is full (interval → 0) this caps the scroll rate.
const MAX_ROWS_PER_TICK: usize = 4;

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

/// Free-running AWGN channel.  TX bursts are injected into the stream;
/// `read()` drains buffered signal-plus-noise or synthesises pure noise when idle.
///
/// Future device mode: replace `inject` / `read` with SDR driver calls.
struct Channel {
    signal_amp:  f32,
    noise_sigma: f32,
    pending:     VecDeque<Complex<f32>>,
}

impl Channel {
    fn new(signal_amp: f32, noise_sigma: f32) -> Self {
        Self { signal_amp, noise_sigma, pending: VecDeque::new() }
    }

    fn set_signal_amp(&mut self, a: f32)  { self.signal_amp  = a; }
    fn set_noise_sigma(&mut self, s: f32) { self.noise_sigma = s; }
    fn pending_len(&self) -> usize        { self.pending.len() }
    fn clear(&mut self)                   { self.pending.clear(); }

    /// Scale the TX burst, mix with AWGN, push into the stream, and return
    /// the mixed IQ so the caller can immediately hand it to the RX for stats.
    fn inject(&mut self, iq: &[Complex<f32>], rng: &mut impl Rng) -> Vec<Complex<f32>> {
        let mixed: Vec<_> = iq.iter()
            .map(|&s| s * self.signal_amp + self.noise_sample(rng))
            .collect();
        self.pending.extend(mixed.iter().copied());
        mixed
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

// ─── Shared simulation state ──────────────────────────────────────────────────

struct SimShared {
    running:        AtomicBool,
    clear_buf:      AtomicBool,   // flush IQ buffer on settings change
    sf:             Mutex<u8>,
    os_factor:      Mutex<u32>,
    fft_size:       Mutex<usize>,
    signal_db:      Mutex<f32>,   // signal amplitude in dBFS
    noise_db:       Mutex<f32>,   // noise amplitude in dBFS
    interval_ms:    Mutex<u64>,
    spectrum_plot:  Arc<SpectrumPlot>,
    waterfall_plot: Arc<WaterfallPlot>,
    stats:          Mutex<Stats>,
}

#[derive(Default, Clone)]
struct Stats {
    total:   usize,
    ok:      usize,
    last_tx: String,
    last_rx: String,
}

// ─── Sim thread ───────────────────────────────────────────────────────────────

fn sim_loop(shared: Arc<SimShared>, ctx: egui::Context) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x10_4a);
    let payloads: &[&[u8]] = &[
        b"Hello, LoRa!", b"Rust 2024", b"AWGN channel",
        b"burst test",   b"lora-rs",   b"packet!",
    ];
    let mut idx         = 0usize;
    let mut last_packet = Instant::now() - Duration::from_secs(60); // fire immediately

    // Read initial settings to build TX / Channel / RX.
    let init_sf        = *shared.sf.lock().unwrap();
    let init_os        = *shared.os_factor.lock().unwrap();
    let init_signal_db = *shared.signal_db.lock().unwrap();
    let init_noise_db  = *shared.noise_db.lock().unwrap();

    let mut tx      = Tx::new(init_sf, 4, init_os);
    let mut channel = Channel::new(
        db_to_amp(init_signal_db),
        db_to_amp(init_noise_db) / std::f32::consts::SQRT_2,
    );
    let mut rx      = Rx::new(init_sf, 4, init_os);

    // FFT state — rebuilt when fft_size changes.
    let mut cur_fft_size = 0usize;
    let mut hann: Vec<f32>             = Vec::new();
    let mut fft_buf: Vec<Complex<f32>> = Vec::new();
    let mut planner = FftPlanner::<f32>::new();
    let mut fft: Arc<dyn rustfft::Fft<f32>> = planner.plan_fft_forward(1);

    loop {
        let tick_start = Instant::now();

        if !shared.running.load(Ordering::Relaxed) {
            std::thread::sleep(TICK);
            continue;
        }

        // Flush channel on settings change.
        if shared.clear_buf.swap(false, Ordering::Relaxed) {
            channel.clear();
            last_packet = Instant::now() - Duration::from_secs(60);
        }

        let sf          = *shared.sf.lock().unwrap();
        let os_factor   = *shared.os_factor.lock().unwrap();
        let fft_size    = *shared.fft_size.lock().unwrap();
        let signal_db   = *shared.signal_db.lock().unwrap();
        let noise_db    = *shared.noise_db.lock().unwrap();
        let interval_ms = *shared.interval_ms.lock().unwrap();

        // Propagate updated amplitude settings to the channel.
        channel.set_signal_amp(db_to_amp(signal_db));
        channel.set_noise_sigma(db_to_amp(noise_db) / std::f32::consts::SQRT_2);

        // Rebuild TX / RX when SF or os_factor changes.
        if sf != tx.sf || os_factor != tx.os_factor {
            tx = Tx::new(sf, 4, os_factor);
            rx = Rx::new(sf, 4, os_factor);
        }

        // Rebuild FFT plan if size changed.
        if fft_size != cur_fft_size {
            cur_fft_size = fft_size;
            hann = (0..fft_size)
                .map(|i| (0.5 * (1.0 - (TAU * i as f64 / fft_size as f64).cos())) as f32)
                .collect();
            fft_buf = vec![Complex::new(0.0_f32, 0.0_f32); fft_size];
            fft = planner.plan_fft_forward(fft_size);
        }

        // ── TX: packet scheduler ──────────────────────────────────────────
        let should_send = if interval_ms == 0 {
            channel.pending_len() < fft_size * 2
        } else {
            last_packet.elapsed().as_millis() as u64 >= interval_ms
                && channel.pending_len() < fft_size * 16
        };

        if should_send {
            last_packet = Instant::now();
            let payload = payloads[idx % payloads.len()];
            idx += 1;

            // TX → Channel: inject returns the mixed IQ for immediate RX decode.
            let iq    = tx.modulate(payload);
            let mixed = channel.inject(&iq, &mut rng);

            // RX: decode the mixed burst for link stats.
            let result = rx.decode(&mixed);
            {
                let mut s = shared.stats.lock().unwrap();
                s.total += 1;
                s.last_tx = String::from_utf8_lossy(payload).to_string();
                if let Some(rx_payload) = &result {
                    if rx_payload == payload { s.ok += 1; }
                    s.last_rx = String::from_utf8_lossy(rx_payload).to_string();
                } else {
                    s.last_rx = "—".to_string();
                }
            }
        }

        // ── Channel → display: drain up to MAX_ROWS_PER_TICK FFT windows ─
        let rows = if channel.pending_len() >= fft_size {
            MAX_ROWS_PER_TICK.min(channel.pending_len() / fft_size)
        } else {
            1  // always emit at least one row; channel fills shortfall with noise
        };

        let mut last_spec: Option<Vec<[f64; 2]>> = None;
        for _ in 0..rows {
            let window = channel.read(fft_size, &mut rng);
            let spec   = spectrum_window(&window, &hann, &mut fft_buf, fft.as_ref());
            shared.waterfall_plot.update(spec.clone());
            last_spec = Some(spec);
        }

        if let Some(s) = last_spec {
            shared.spectrum_plot.update(s);
        }
        shared.waterfall_plot.set_freq(fft_size as f64 / 2.0);
        shared.waterfall_plot.set_bw(fft_size as f64);

        ctx.request_repaint();

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
        let samp_rate_khz = 1000u32;
        let bw_khz        = 250u32;
        let (_, os_factor) = effective_sr_and_os(samp_rate_khz, bw_khz);
        let fft_size      = 1024usize;
        let signal_db     = -20.0f32;  // typical received signal well below full scale
        let noise_db      = -35.0f32;  // SNR = 15 dB — good outdoor link, noise visible

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

        let shared = Arc::new(SimShared {
            running:        AtomicBool::new(true),
            clear_buf:      AtomicBool::new(false),
            sf:             Mutex::new(sf),
            os_factor:      Mutex::new(os_factor),
            fft_size:       Mutex::new(fft_size),
            signal_db:      Mutex::new(signal_db),
            noise_db:       Mutex::new(noise_db),
            interval_ms:    Mutex::new(500),
            spectrum_plot,
            waterfall_plot,
            stats:          Mutex::new(Stats::default()),
        });

        Self {
            shared,
            spectrum_chart,
            waterfall_chart,
            thread_started: false,
            signal_db,
            noise_db,
            interval_ms: 500,
            sf,
            samp_rate_khz,
            bw_khz,
            fft_size,
        }
    }

    fn rebuild_plots(&mut self) {
        let (eff_sr, os_factor) = effective_sr_and_os(self.samp_rate_khz, self.bw_khz);
        self.samp_rate_khz = eff_sr;
        self.spectrum_chart.set_x_limits([0.0, self.fft_size as f64]);
        self.waterfall_chart.set_x_limits([0.0, self.fft_size as f64]);
        *self.shared.sf.lock().unwrap()        = self.sf;
        *self.shared.os_factor.lock().unwrap() = os_factor;
        *self.shared.fft_size.lock().unwrap()  = self.fft_size;
        self.shared.waterfall_plot.set_freq(self.fft_size as f64 / 2.0);
        self.shared.waterfall_plot.set_bw(self.fft_size as f64);
        *self.shared.stats.lock().unwrap() = Stats::default();
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
            std::thread::spawn(move || sim_loop(shared, ctx2));
        }

        let running = self.shared.running.load(Ordering::Relaxed);
        let stats   = self.shared.stats.lock().unwrap().clone();

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

            // ── Row 3: stats ──────────────────────────────────────────────────
            ui.horizontal_wrapped(|ui| {
                let per = if stats.total > 0 {
                    100.0 * (stats.total - stats.ok) as f32 / stats.total as f32
                } else { 0.0 };
                ui.label(format!(
                    "TX: {:12}  RX: {:12}  {}/{} ok  PER {:.0}%",
                    format!("{:?}", stats.last_tx),
                    format!("{:?}", stats.last_rx),
                    stats.ok, stats.total, per,
                ));
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

// ─── Entry point ─────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sf = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7_u8);

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
