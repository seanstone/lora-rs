/// LoRa GUI simulator — raw spectrum + waterfall of the received IQ.
///
/// The spectrum/waterfall is a plain Hann-windowed FFT of the raw IQ, fully
/// decoupled from the receiver.  AWGN frames show a flat noise floor; LoRa
/// packets show the chirp sweeping through the centre of the band.
///
/// Sample rate = 4 × signal bandwidth by default (os_factor = 4).
/// With bandwidth narrower than the sample rate the LoRa signal occupies
/// only the central fraction of the display.
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
    f64::consts::TAU,
    sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}},
    time::Duration,
};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Fixed simulated sample rate in kHz.  os_factor = SAMP_RATE_KHZ / bw_khz.
const SAMP_RATE_KHZ: u32 = 1000;

/// LoRa bandwidth options (kHz).  Default 250 kHz → os_factor = 4.
const BW_OPTIONS_KHZ: &[u32] = &[125, 250, 500, 1000];

// ─── TX / RX helpers ─────────────────────────────────────────────────────────

fn pad_nibbles(nibbles: &[u8], sf: u8, ldro: bool) -> Vec<u8> {
    let pay_sf    = if ldro { (sf - 2) as usize } else { sf as usize };
    let header_cw = (sf - 2) as usize;
    let remaining = nibbles.len().saturating_sub(header_cw);
    let pad       = (pay_sf - remaining % pay_sf) % pay_sf;
    let mut v     = nibbles.to_vec();
    v.resize(v.len() + pad, 0);
    v
}

fn tx_packet(payload: &[u8], sf: u8, cr: u8, os_factor: u32) -> Vec<Complex<f32>> {
    let nibbles   = whiten(payload);
    let framed    = add_header(&nibbles, false, true, cr);
    let with_crc  = add_crc(&framed, payload, true);
    let padded    = pad_nibbles(&with_crc, sf, false);
    let codewords = hamming_enc(&padded, cr, sf);
    let symbols   = interleave(&codewords, cr, sf, false);
    let chirps    = gray_demap(&symbols, sf);
    modulate(&chirps, sf, 0x12, 8, os_factor)
}

fn rx_packet(iq: &[Complex<f32>], sf: u8, cr: u8, os_factor: u32) -> Option<Vec<u8>> {
    let sync      = frame_sync(iq, sf, 0x12, 8, os_factor);
    if !sync.found { return None; }
    let chirps    = fft_demod(&sync.symbols, sf, os_factor);
    let symbols   = gray_map(&chirps, sf);
    let codewords = deinterleave(&symbols, cr, sf, false);
    let nibbles   = hamming_dec(&codewords, cr, sf);
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

fn add_awgn(signal: &[Complex<f32>], snr_db: f32, rng: &mut impl Rng) -> Vec<Complex<f32>> {
    let snr_lin = 10_f32.powf(snr_db / 10.0);
    let sigma   = (0.5 / snr_lin).sqrt();
    signal.iter().map(|&s| {
        let u1 = (rng.random::<f32>() + 1e-7_f32).min(1.0);
        let u2 = rng.random::<f32>();
        let re  = (-2.0_f32 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        let u3 = (rng.random::<f32>() + 1e-7_f32).min(1.0);
        let u4 = rng.random::<f32>();
        let im  = (-2.0_f32 * u3.ln()).sqrt() * (std::f32::consts::TAU * u4).cos();
        s + Complex::new(re * sigma, im * sigma)
    }).collect()
}

// ─── Raw spectrum ─────────────────────────────────────────────────────────────

/// Sliding Hann-windowed FFT of `iq` with step = `fft_size`.
///
/// Returns one `Vec<[f64;2]>` per window: x = bin (0..fft_size), y = power dB.
/// The output is fftshifted so DC is at bin fft_size/2.  With os_factor > 1
/// the LoRa signal (bandwidth 1/os_factor of the sample rate) appears as a
/// chirp sweep occupying the central fft_size/os_factor bins.
fn raw_spectra(iq: &[Complex<f32>], fft_size: usize) -> Vec<Vec<[f64; 2]>> {
    let mut planner = FftPlanner::<f32>::new();
    let fft         = planner.plan_fft_forward(fft_size);
    let mut buf     = vec![Complex::new(0.0_f32, 0.0_f32); fft_size];

    let hann: Vec<f32> = (0..fft_size)
        .map(|i| (0.5 * (1.0 - (TAU * i as f64 / fft_size as f64).cos())) as f32)
        .collect();

    let mut out = Vec::new();
    let mut w   = 0usize;
    while w + fft_size <= iq.len() {
        for (b, (s, h)) in buf.iter_mut().zip(iq[w..w + fft_size].iter().zip(hann.iter())) {
            *b = s * h;
        }
        fft.process(&mut buf);

        let half = fft_size / 2;
        let spec: Vec<[f64; 2]> = (0..fft_size).map(|i| {
            let src = (i + half) % fft_size;
            let pdb = 10.0 * (buf[src].norm_sqr() as f64 + 1e-20_f64).log10();
            [i as f64, pdb.max(-100.0)]
        }).collect();
        out.push(spec);
        w += fft_size;
    }
    out
}

// ─── Shared simulation state ──────────────────────────────────────────────────

struct SimShared {
    running:        AtomicBool,
    sf:             Mutex<u8>,
    os_factor:      Mutex<u32>,
    fft_size:       Mutex<usize>,
    snr_db:         Mutex<f32>,
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

fn sim_loop(shared: Arc<SimShared>, ctx: egui::Context) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x10_4a);
    let payloads: &[&[u8]] = &[
        b"Hello, LoRa!", b"Rust 2024", b"AWGN channel",
        b"burst test",   b"lora-rs",   b"packet!",
    ];
    let mut idx = 0usize;

    loop {
        if !shared.running.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }

        let sf          = *shared.sf.lock().unwrap();
        let os_factor   = *shared.os_factor.lock().unwrap();
        let fft_size    = *shared.fft_size.lock().unwrap();
        let snr_db      = *shared.snr_db.lock().unwrap();
        let interval_ms = *shared.interval_ms.lock().unwrap();
        let payload     = payloads[idx % payloads.len()];
        idx += 1;

        // TX → channel.
        let iq    = tx_packet(payload, sf, 4, os_factor);
        let noisy = add_awgn(&iq, snr_db, &mut rng);

        // Raw spectrum — always produces rows regardless of decode outcome.
        let spectra = raw_spectra(&noisy, fft_size);
        for spec in &spectra {
            shared.waterfall_plot.update(spec.clone());
        }
        if let Some(last) = spectra.last() {
            shared.spectrum_plot.update(last.clone());
        }
        shared.waterfall_plot.set_freq(fft_size as f64 / 2.0);
        shared.waterfall_plot.set_bw(fft_size as f64);

        // RX — for stats only, fully decoupled from the display.
        let result = rx_packet(&noisy, sf, 4, os_factor);
        {
            let mut s = shared.stats.lock().unwrap();
            s.total += 1;
            s.last_tx = String::from_utf8_lossy(payload).to_string();
            if let Some(rx) = &result {
                if rx == payload { s.ok += 1; }
                s.last_rx = String::from_utf8_lossy(rx).to_string();
            } else {
                s.last_rx = "—".to_string();
            }
        }

        ctx.request_repaint();
        std::thread::sleep(Duration::from_millis(interval_ms));
    }
}

// ─── GUI app ──────────────────────────────────────────────────────────────────

struct GuiApp {
    shared:          Arc<SimShared>,
    spectrum_chart:  Chart,
    waterfall_chart: Chart,
    thread_started:  bool,
    snr_db:      f32,
    interval_ms: u64,
    sf:          u8,
    bw_khz:      u32,   // signal bandwidth; os_factor = SAMP_RATE_KHZ / bw_khz
    fft_size:    usize,
}

impl GuiApp {
    fn new(sf: u8, snr_db: f32) -> Self {
        // Default: 250 kHz BW → os_factor = 4 (sample rate = 4 × BW).
        let bw_khz    = 250u32;
        let os_factor = SAMP_RATE_KHZ / bw_khz;  // = 4
        let fft_size  = 1024usize;

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
            sf:             Mutex::new(sf),
            os_factor:      Mutex::new(os_factor),
            fft_size:       Mutex::new(fft_size),
            snr_db:         Mutex::new(snr_db),
            interval_ms:    Mutex::new(250),
            spectrum_plot,
            waterfall_plot,
            stats:          Mutex::new(Stats::default()),
        });

        Self {
            shared,
            spectrum_chart,
            waterfall_chart,
            thread_started: false,
            snr_db,
            interval_ms: 250,
            sf,
            bw_khz,
            fft_size,
        }
    }

    fn rebuild_plots(&mut self) {
        let os_factor = SAMP_RATE_KHZ / self.bw_khz;
        self.spectrum_chart.set_x_limits([0.0, self.fft_size as f64]);
        self.waterfall_chart.set_x_limits([0.0, self.fft_size as f64]);
        *self.shared.sf.lock().unwrap()        = self.sf;
        *self.shared.os_factor.lock().unwrap() = os_factor;
        *self.shared.fft_size.lock().unwrap()  = self.fft_size;
        self.shared.waterfall_plot.set_freq(self.fft_size as f64 / 2.0);
        self.shared.waterfall_plot.set_bw(self.fft_size as f64);
        *self.shared.stats.lock().unwrap() = Stats::default();
    }
}

fn bw_label(bw_khz: u32) -> &'static str {
    match bw_khz {
        125  => "125k",
        250  => "250k",
        500  => "500k",
        1000 => "1M",
        _    => "?",
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

        // ── Top control bar ────────────────────────────────────────────────
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Start / Stop
                let btn_label = if running { "⏸ Pause" } else { "▶ Run" };
                if ui.button(btn_label).clicked() {
                    self.shared.running.store(!running, Ordering::Relaxed);
                }

                ui.separator();

                // SF selector
                ui.label("SF:");
                let mut changed = false;
                for s in [7u8, 8, 9, 10, 11, 12] {
                    if ui.selectable_label(self.sf == s, format!("{s}")).clicked() && self.sf != s {
                        self.sf = s;
                        changed = true;
                    }
                }

                ui.separator();

                // BW selector — sample rate is shown as a fixed label.
                // os_factor = SAMP_RATE_KHZ / bw_khz; default 250 kHz → ×4.
                ui.label(format!("SR: {} MHz  BW:", SAMP_RATE_KHZ as f32 / 1000.0));
                for &bw in BW_OPTIONS_KHZ {
                    if ui.selectable_label(self.bw_khz == bw, bw_label(bw)).clicked()
                        && self.bw_khz != bw
                    {
                        self.bw_khz = bw;
                        changed = true;
                    }
                }

                ui.separator();

                // FFT size selector
                ui.label("FFT:");
                for &sz in &[1024usize, 2048, 4096, 8192] {
                    if ui.selectable_label(self.fft_size == sz, format!("{sz}")).clicked()
                        && self.fft_size != sz
                    {
                        self.fft_size = sz;
                        changed = true;
                    }
                }
                if changed { self.rebuild_plots(); }

                ui.separator();

                // SNR slider
                ui.label("SNR:");
                if ui.add(Slider::new(&mut self.snr_db, -15.0..=20.0).suffix(" dB").step_by(0.5)).changed() {
                    *self.shared.snr_db.lock().unwrap() = self.snr_db;
                }

                ui.separator();

                // Interval slider
                ui.label("Interval:");
                let mut interval_f = self.interval_ms as f32;
                if ui.add(Slider::new(&mut interval_f, 0.0..=2000.0).suffix(" ms").step_by(50.0)).changed() {
                    self.interval_ms = interval_f as u64;
                    *self.shared.interval_ms.lock().unwrap() = self.interval_ms;
                }

                ui.separator();

                // Stats
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

        // ── Plots ─────────────────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            let h = ui.available_height();
            let w = ui.available_width();
            ui.allocate_ui(Vec2::new(w, h * 0.40), |ui| {
                self.spectrum_chart.ui(ui);
            });
            ui.allocate_ui(Vec2::new(w, h * 0.60), |ui| {
                self.waterfall_chart.ui(ui);
            });
        });

        if running {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
    }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sf     = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7_u8);
    let snr_db = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10.0_f32);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("LoRa Link Simulator")
            .with_inner_size([1280.0, 750.0]),
        ..Default::default()
    };

    eframe::run_native(
        "LoRa Link Simulator",
        options,
        Box::new(move |_cc| Ok(Box::new(GuiApp::new(sf, snr_db)))),
    ).unwrap();
}
