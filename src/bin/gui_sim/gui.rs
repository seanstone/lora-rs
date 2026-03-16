use eframe::egui;
use egui::{Slider, Vec2};
use lora::ui::{Chart, SpectrumPlot, WaterfallPlot};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU32, Ordering}},
};

use super::shared::{LogEntry, SimShared, Stats};
use super::sim::sim_loop;
use super::{
    DEFAULT_SF, DEFAULT_SAMP_RATE_KHZ, DEFAULT_BW_KHZ, DEFAULT_FFT_SIZE,
    DEFAULT_SIGNAL_DB, DEFAULT_NOISE_DB, DEFAULT_INTERVAL_MS,
    SR_OPTIONS_KHZ, BW_OPTIONS_KHZ,
    khz_label, effective_sr_and_os, snr_db, waterfall_total_secs,
};

/// Screen width below which the layout switches to a mobile-friendly mode:
/// the messages side panel collapses into a toggleable bottom drawer.
const MOBILE_BREAKPOINT: f32 = 600.0;

pub(crate) struct GuiApp {
    shared:          Arc<SimShared>,
    spectrum_chart:  Chart,
    waterfall_chart: Chart,
    thread_started:  bool,
    /// Shared with `main()` so the join can happen after the window closes.
    /// Not present on WASM — the sim task runs on the JS event loop.
    #[cfg(not(target_arch = "wasm32"))]
    sim_thread:      Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>>,
    /// Last known X bounds — used to detect which chart panned/zoomed so we
    /// can copy the new range to the other chart next frame.
    last_synced_x:   [f64; 2],
    signal_db:       f32,
    noise_db:        f32,
    interval_ms:     u64,
    sf:              u8,
    samp_rate_khz:   u32,
    bw_khz:          u32,
    fft_size:        usize,
    // ── UHD hardware device UI state ──────────────────────────────────────────
    uhd_args:        String,
    uhd_freq_mhz:    f64,
    uhd_rx_gain_db:  f64,
    uhd_tx_gain_db:  f64,
    /// Whether the messages bottom drawer is open (mobile layout only).
    msg_drawer_open: bool,
    /// Whether the settings drawer is open (mobile layout only).
    menu_open: bool,
}

impl GuiApp {
    pub fn new(
        sf: u8,
        #[cfg(not(target_arch = "wasm32"))]
        sim_thread: Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>>,
    ) -> Self {
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
        let wf_total = waterfall_total_secs(eff_sr, fft_size);
        waterfall_chart.set_y_time_display(wf_total);

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
            tx_starved:     AtomicBool::new(false),
            use_uhd:        AtomicBool::new(false),
            uhd_args:       Mutex::new(String::new()),
            uhd_freq_hz:    Mutex::new(915e6),
            uhd_rx_gain_db: Mutex::new(40.0),
            uhd_tx_gain_db: Mutex::new(40.0),
            rebuild_driver: AtomicBool::new(false),
            uhd_loading:    AtomicBool::new(false),
            quit:           AtomicBool::new(false),
        });

        Self {
            shared,
            spectrum_chart,
            waterfall_chart,
            thread_started: false,
            #[cfg(not(target_arch = "wasm32"))]
            sim_thread,
            last_synced_x:  [0.0, fft_size as f64],
            signal_db,
            noise_db,
            interval_ms: DEFAULT_INTERVAL_MS,
            sf,
            samp_rate_khz,
            bw_khz,
            fft_size,
            uhd_args:       String::new(),
            uhd_freq_mhz:   915.0,
            uhd_rx_gain_db: 40.0,
            uhd_tx_gain_db: 40.0,
            msg_drawer_open: false,
            menu_open: false,
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

        // Capture old hardware-relevant values *before* updating shared state
        // so we can detect whether a hardware reconfigure is actually needed.
        #[cfg(feature = "uhd")]
        let uhd_hw_changed = self.shared.use_uhd.load(Ordering::Relaxed) && {
            let old_sr = *self.shared.samp_rate_khz.lock().unwrap();
            let old_os = *self.shared.os_factor.lock().unwrap();
            eff_sr != old_sr || os_factor != old_os
        };

        self.spectrum_chart.set_x_limits([0.0, self.fft_size as f64]);
        self.waterfall_chart.set_x_limits([0.0, self.fft_size as f64]);
        let wf_total = waterfall_total_secs(eff_sr, self.fft_size);
        self.waterfall_chart.set_y_time_display(wf_total);
        *self.shared.sf.lock().unwrap()            = self.sf;
        *self.shared.os_factor.lock().unwrap()     = os_factor;
        *self.shared.samp_rate_khz.lock().unwrap() = eff_sr;
        *self.shared.fft_size.lock().unwrap()      = self.fft_size;
        self.shared.waterfall_plot.set_freq(self.fft_size as f64 / 2.0);
        self.shared.waterfall_plot.set_bw(self.fft_size as f64);
        *self.shared.stats.lock().unwrap() = Stats::default();
        self.shared.log.lock().unwrap().clear();
        self.shared.clear_buf.store(true, Ordering::Relaxed);
        // Only trigger a hardware rebuild when the UHD sample rate or bandwidth
        // actually changes.  SF and FFT-size changes don't affect hardware.
        #[cfg(feature = "uhd")]
        if uhd_hw_changed {
            self.shared.rebuild_driver.store(true, Ordering::Relaxed);
        }
    }
}

/// Consume the remaining row space with a zero-height spacer when the next
/// group won't fully fit, so `horizontal_wrapped` starts a fresh row.
fn newline_if_needed(ui: &mut egui::Ui, needed: f32) {
    let cursor_x = ui.cursor().min.x;
    let left_x   = ui.max_rect().min.x;
    let right_x  = ui.max_rect().max.x;
    if cursor_x <= left_x + ui.spacing().item_spacing.x * 2.0 { return; }
    let remaining = right_x - cursor_x;
    if remaining <= 0.0 { return; }
    if remaining < needed {
        ui.allocate_exact_size(egui::vec2(remaining + 1.0, 0.0), egui::Sense::hover());
    }
}

fn show_messages_panel(
    ui: &mut egui::Ui,
    stats: &Stats,
    log: &std::collections::VecDeque<LogEntry>,
    shared: &SimShared,
) {
    let accounted = stats.rx_count + stats.rx_lost;
    let per = if accounted > 0 {
        100.0 * stats.rx_lost as f32 / accounted as f32
    } else { 0.0 };
    ui.horizontal(|ui| {
        ui.label(format!("{}/{} rx", stats.rx_count, stats.tx_count));
        if stats.rx_lost > 0 {
            ui.colored_label(
                egui::Color32::from_rgb(220, 60, 60),
                format!("{} lost", stats.rx_lost),
            );
        }
        ui.label(format!("PER {:.0}%", per));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("↺ Reset").clicked() {
                *shared.stats.lock().unwrap() = Stats::default();
                shared.log.lock().unwrap().clear();
                shared.clear_buf.store(true, Ordering::Relaxed);
            }
        });
    });
    ui.separator();

    egui::ScrollArea::vertical()
        .stick_to_bottom(true)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for entry in log {
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
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.thread_started {
            self.thread_started = true;
            let shared = self.shared.clone();
            let ctx2   = ctx.clone();
            #[cfg(not(target_arch = "wasm32"))]
            {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(3)
                    .enable_all()
                    .build()
                    .unwrap();
                *self.sim_thread.lock().unwrap() =
                    Some(std::thread::spawn(move || rt.block_on(sim_loop(shared, Some(ctx2)))));
            }
            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_futures::spawn_local(sim_loop(shared, Some(ctx2)));
        }

        let screen_w = ctx.input(|i| i.screen_rect().width());
        let is_mobile = screen_w < MOBILE_BREAKPOINT;

        let running = self.shared.running.load(Ordering::Relaxed);
        let stats   = self.shared.stats.lock().unwrap().clone();
        let log     = self.shared.log.lock().unwrap().clone();

        // ── Compute shared state used in both mobile and desktop layouts ──────
        #[cfg(feature = "uhd")]
        let sim_mode = !self.shared.use_uhd.load(Ordering::Relaxed);
        #[cfg(not(feature = "uhd"))]
        let sim_mode = true;

        let lag_ms     = f32::from_bits(self.shared.buf_lag_ms  .load(Ordering::Relaxed));
        let overflow   = self.shared.buf_overflow .load(Ordering::Relaxed);
        let underflow  = self.shared.buf_underflow.load(Ordering::Relaxed);
        let tx_starved = self.shared.tx_starved   .load(Ordering::Relaxed);

        if is_mobile {
            // ── Mobile: compact toolbar ───────────────────────────────────────
            egui::TopBottomPanel::top("controls").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button(if running { "⏸" } else { "▶" }).clicked() {
                        self.shared.running.store(!running, Ordering::Relaxed);
                    }
                    let menu_label = if self.menu_open { "✕" } else { "☰" };
                    if ui.button(menu_label).clicked() {
                        self.menu_open = !self.menu_open;
                    }
                    ui.separator();
                    let (buf_color, buf_label) = if underflow {
                        (egui::Color32::from_rgb(220, 160, 0), "⚠ UF")
                    } else if overflow {
                        (egui::Color32::from_rgb(220, 60, 60), "⚠ OF")
                    } else {
                        (egui::Color32::from_rgb(100, 180, 100), "●")
                    };
                    ui.colored_label(buf_color, buf_label);
                    ui.label(format!("{lag_ms:.0} ms"));
                    if tx_starved {
                        ui.colored_label(egui::Color32::from_rgb(220, 160, 0), "TX!")
                            .on_hover_text("TX modulator can't keep up — gaps in the IQ stream");
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let msg_label = if self.msg_drawer_open {
                            "✉▼".to_string()
                        } else if !log.is_empty() {
                            format!("✉({})", log.len())
                        } else {
                            "✉".to_string()
                        };
                        if ui.button(msg_label).clicked() {
                            self.msg_drawer_open = !self.msg_drawer_open;
                        }
                    });
                });
            });

            // ── Mobile: settings drawer ───────────────────────────────────────
            if self.menu_open {
                egui::TopBottomPanel::top("settings_drawer").show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let h = ui.spacing().interact_size.y;
                        let mut changed = false;

                        egui::Grid::new("mobile_settings")
                            .num_columns(2)
                            .spacing([8.0, 6.0])
                            .show(ui, |ui| {
                                // SF
                                ui.label("SF:");
                                ui.horizontal_wrapped(|ui| {
                                    for s in [7u8, 8, 9, 10, 11, 12] {
                                        if ui.selectable_label(self.sf == s, format!("{s}")).clicked()
                                            && self.sf != s
                                        { self.sf = s; changed = true; }
                                    }
                                });
                                ui.end_row();

                                // SR
                                ui.label("SR:");
                                ui.horizontal_wrapped(|ui| {
                                    for &sr in SR_OPTIONS_KHZ {
                                        if ui.selectable_label(self.samp_rate_khz == sr, khz_label(sr)).clicked()
                                            && self.samp_rate_khz != sr
                                        { self.samp_rate_khz = sr; changed = true; }
                                    }
                                });
                                ui.end_row();

                                // BW
                                ui.label("BW:");
                                ui.horizontal_wrapped(|ui| {
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
                                ui.end_row();

                                // FFT
                                ui.label("FFT:");
                                ui.horizontal_wrapped(|ui| {
                                    for &sz in &[1024usize, 2048, 4096, 8192] {
                                        if ui.selectable_label(self.fft_size == sz, format!("{sz}")).clicked()
                                            && self.fft_size != sz
                                        { self.fft_size = sz; changed = true; }
                                    }
                                });
                                ui.end_row();

                                // Interval
                                ui.label("Interval:");
                                let mut interval_f = self.interval_ms as f32;
                                if ui.add_sized(
                                    [ui.available_width(), h],
                                    Slider::new(&mut interval_f, 0.0..=5000.0).suffix(" ms").step_by(50.0),
                                ).changed() {
                                    self.interval_ms = interval_f as u64;
                                    *self.shared.interval_ms.lock().unwrap() = self.interval_ms;
                                }
                                ui.end_row();

                                // TX gain
                                ui.label("TX gain:");
                                if sim_mode {
                                    if ui.add_sized(
                                        [ui.available_width(), h],
                                        Slider::new(&mut self.signal_db, -40.0..=20.0).suffix(" dBFS").step_by(1.0),
                                    ).changed() {
                                        *self.shared.signal_db.lock().unwrap() = self.signal_db;
                                    }
                                } else {
                                    #[cfg(feature = "uhd")]
                                    if ui.add_sized(
                                        [ui.available_width(), h],
                                        Slider::new(&mut self.uhd_tx_gain_db, 0.0..=89.0).suffix(" dB").step_by(0.5),
                                    ).changed() {
                                        *self.shared.uhd_tx_gain_db.lock().unwrap() = self.uhd_tx_gain_db;
                                    }
                                }
                                ui.end_row();

                                // Noise / RX gain
                                if sim_mode {
                                    ui.label("Noise:");
                                    if ui.add_sized(
                                        [ui.available_width(), h],
                                        Slider::new(&mut self.noise_db, -80.0..=0.0).suffix(" dBFS").step_by(1.0),
                                    ).changed() {
                                        *self.shared.noise_db.lock().unwrap() = self.noise_db;
                                    }
                                } else {
                                    #[cfg(feature = "uhd")]
                                    {
                                        ui.label("RX gain:");
                                        if ui.add_sized(
                                            [ui.available_width(), h],
                                            Slider::new(&mut self.uhd_rx_gain_db, 0.0..=76.0).suffix(" dB").step_by(0.5),
                                        ).changed() {
                                            *self.shared.uhd_rx_gain_db.lock().unwrap() = self.uhd_rx_gain_db;
                                        }
                                    }
                                }
                                ui.end_row();

                                // UHD-specific rows
                                #[cfg(feature = "uhd")]
                                if !sim_mode {
                                    ui.label("Args:");
                                    if ui.add_sized(
                                        [ui.available_width(), h],
                                        egui::TextEdit::singleline(&mut self.uhd_args).hint_text("addr=… or empty"),
                                    ).lost_focus() {
                                        *self.shared.uhd_args.lock().unwrap() = self.uhd_args.clone();
                                        self.shared.rebuild_driver.store(true, Ordering::Relaxed);
                                    }
                                    ui.end_row();

                                    ui.label("Freq:");
                                    if ui.add_sized(
                                        [ui.available_width(), h],
                                        egui::DragValue::new(&mut self.uhd_freq_mhz).speed(0.1).suffix(" MHz").range(1.0..=6000.0),
                                    ).changed() {
                                        *self.shared.uhd_freq_hz.lock().unwrap() = self.uhd_freq_mhz * 1e6;
                                        self.shared.rebuild_driver.store(true, Ordering::Relaxed);
                                    }
                                    ui.end_row();
                                }
                            });

                        if changed { self.rebuild_plots(); }

                        ui.add_space(2.0);
                        ui.horizontal(|ui| {
                            if sim_mode {
                                ui.label(format!("SNR: {:.0} dB", snr_db(self.signal_db, self.noise_db)));
                            }
                            #[cfg(feature = "uhd")]
                            {
                                let use_uhd = !sim_mode;
                                if ui.selectable_label(sim_mode, "Sim").clicked() && use_uhd {
                                    self.shared.use_uhd.store(false, Ordering::Relaxed);
                                    self.shared.rebuild_driver.store(true, Ordering::Relaxed);
                                    *self.shared.stats.lock().unwrap() = Stats::default();
                                    self.shared.log.lock().unwrap().clear();
                                }
                                if ui.selectable_label(use_uhd, "UHD").clicked() && sim_mode {
                                    self.shared.use_uhd.store(true, Ordering::Relaxed);
                                    self.shared.rebuild_driver.store(true, Ordering::Relaxed);
                                    *self.shared.stats.lock().unwrap() = Stats::default();
                                    self.shared.log.lock().unwrap().clear();
                                }
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button("⟳ Defaults").clicked() {
                                    self.restore_defaults();
                                }
                            });
                        });
                        ui.add_space(4.0);
                    });
                });
            }
        } else {
            // ── Desktop: existing multi-row controls panel ────────────────────
            egui::TopBottomPanel::top("controls").show(ctx, |ui| {
                // ── Row 1: signal parameters ──────────────────────────────────
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

                    ui.separator();

                    if ui.button("⟳ Defaults").clicked() {
                        self.restore_defaults();
                    }
                });

                // ── Row 2: driver + gains ─────────────────────────────────────
                ui.horizontal_wrapped(|ui| {
                    let h = ui.spacing().interact_size.y;

                    // Driver selector (UHD feature only)
                    #[cfg(feature = "uhd")]
                    {
                        let use_uhd = !sim_mode;
                        if ui.selectable_label(sim_mode, "Sim").clicked() && use_uhd {
                            self.shared.use_uhd.store(false, Ordering::Relaxed);
                            self.shared.rebuild_driver.store(true, Ordering::Relaxed);
                            *self.shared.stats.lock().unwrap() = Stats::default();
                            self.shared.log.lock().unwrap().clear();
                        }
                        if ui.selectable_label(use_uhd, "UHD").clicked() && sim_mode {
                            self.shared.use_uhd.store(true, Ordering::Relaxed);
                            self.shared.rebuild_driver.store(true, Ordering::Relaxed);
                            *self.shared.stats.lock().unwrap() = Stats::default();
                            self.shared.log.lock().unwrap().clear();
                        }

                        if !sim_mode {
                            ui.separator();
                            let mut uhd_changed = false;
                            ui.label("Args:");
                            if ui.add_sized([120.0, h], egui::TextEdit::singleline(&mut self.uhd_args)
                                .hint_text("addr=… or empty")).lost_focus()
                            {
                                uhd_changed = true;
                            }
                            ui.separator();
                            ui.label("Freq:");
                            if ui.add_sized([100.0, h], egui::DragValue::new(&mut self.uhd_freq_mhz)
                                .speed(0.1).suffix(" MHz").range(1.0..=6000.0)).changed()
                            {
                                uhd_changed = true;
                            }
                            if uhd_changed {
                                *self.shared.uhd_args.lock().unwrap()    = self.uhd_args.clone();
                                *self.shared.uhd_freq_hz.lock().unwrap() = self.uhd_freq_mhz * 1e6;
                                self.shared.rebuild_driver.store(true, Ordering::Relaxed);
                            }
                        }

                        ui.separator();
                    }

                    // TX gain — sim: signal_db (dBFS);  UHD: uhd_tx_gain_db (dB)
                    newline_if_needed(ui, 190.0);
                    ui.horizontal(|ui| {
                        ui.label("TX gain:");
                        if sim_mode {
                            if ui.add_sized([120.0, h], Slider::new(&mut self.signal_db, -40.0..=20.0)
                                .suffix(" dBFS").step_by(1.0)).changed()
                            {
                                *self.shared.signal_db.lock().unwrap() = self.signal_db;
                            }
                        } else {
                            #[cfg(feature = "uhd")]
                            if ui.add_sized([120.0, h], Slider::new(&mut self.uhd_tx_gain_db, 0.0..=89.0)
                                .suffix(" dB").step_by(0.5)).changed()
                            {
                                *self.shared.uhd_tx_gain_db.lock().unwrap() = self.uhd_tx_gain_db;
                            }
                        }
                    });

                    ui.separator();

                    // Noise (sim) / RX gain (UHD)
                    newline_if_needed(ui, 190.0);
                    ui.horizontal(|ui| {
                        if sim_mode {
                            ui.label("Noise:");
                            if ui.add_sized([120.0, h], Slider::new(&mut self.noise_db, -80.0..=0.0)
                                .suffix(" dBFS").step_by(1.0)).changed()
                            {
                                *self.shared.noise_db.lock().unwrap() = self.noise_db;
                            }
                        } else {
                            #[cfg(feature = "uhd")]
                            {
                                ui.label("RX gain:");
                                if ui.add_sized([120.0, h], Slider::new(&mut self.uhd_rx_gain_db, 0.0..=76.0)
                                    .suffix(" dB").step_by(0.5)).changed()
                                {
                                    *self.shared.uhd_rx_gain_db.lock().unwrap() = self.uhd_rx_gain_db;
                                }
                            }
                        }
                    });

                    // SNR readout — sim only
                    if sim_mode {
                        newline_if_needed(ui, 110.0);
                        ui.label(format!("SNR: {:.0} dB", snr_db(self.signal_db, self.noise_db)));
                    }
                });

                // ── Row 3: buffer status ──────────────────────────────────────
                ui.horizontal(|ui| {
                    let (color, label) = if underflow {
                        (egui::Color32::from_rgb(220, 160, 0), "⚠ UNDERFLOW")
                    } else if overflow {
                        (egui::Color32::from_rgb(220, 60, 60),  "⚠ OVERFLOW")
                    } else {
                        (egui::Color32::from_rgb(100, 180, 100), "●")
                    };
                    ui.colored_label(color, label);
                    ui.label(format!("buf {lag_ms:.0} ms"));

                    if tx_starved {
                        ui.separator();
                        ui.colored_label(
                            egui::Color32::from_rgb(220, 160, 0),
                            "⚠ TX slow",
                        ).on_hover_text("TX modulator can't keep up with the requested interval — gaps in the IQ stream");
                    }
                });
            });
        }

        // ── Messages panel — side panel on desktop, bottom drawer on mobile ──
        if !is_mobile {
            egui::SidePanel::right("msg_log")
                .exact_width(260.0)
                .show(ctx, |ui| {
                    show_messages_panel(ui, &stats, &log, &self.shared);
                });
        } else if self.msg_drawer_open {
            egui::TopBottomPanel::bottom("msg_drawer")
                .resizable(true)
                .default_height(240.0)
                .min_height(120.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Messages");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("✕").clicked() {
                                self.msg_drawer_open = false;
                            }
                        });
                    });
                    ui.separator();
                    show_messages_panel(ui, &stats, &log, &self.shared);
                });
        }

        // Update x-axis label mode: MHz when UHD is active, bins otherwise.
        #[cfg(feature = "uhd")]
        {
            if self.shared.use_uhd.load(Ordering::Relaxed) {
                let center_hz = *self.shared.uhd_freq_hz.lock().unwrap();
                let bw_hz     = self.bw_khz as f64 * 1000.0;
                let fft       = self.fft_size;
                self.spectrum_chart .set_x_freq_display(center_hz, bw_hz, fft);
                self.waterfall_chart.set_x_freq_display(center_hz, bw_hz, fft);
            } else {
                self.spectrum_chart .clear_x_freq_display();
                self.waterfall_chart.clear_x_freq_display();
            }
        }

        // ── "Opening USRP…" modal overlay ─────────────────────────────────────
        #[cfg(feature = "uhd")]
        if self.shared.uhd_loading.load(Ordering::Relaxed) {
            egui::Modal::new(egui::Id::new("uhd_loading_modal")).show(ctx, |ui| {
                ui.set_min_width(220.0);
                ui.vertical_centered(|ui| {
                    ui.add_space(8.0);
                    ui.add(egui::Spinner::new().size(32.0));
                    ui.add_space(6.0);
                    ui.heading("Opening USRP…");
                    ui.add_space(8.0);
                });
            });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let h = ui.available_height();
            let w = ui.available_width();
            ui.allocate_ui(Vec2::new(w, h * 0.40), |ui| self.spectrum_chart.ui(ui));
            ui.allocate_ui(Vec2::new(w, h * 0.60), |ui| self.waterfall_chart.ui(ui));
        });

        // Cross-sync X bounds: whichever chart changed since last frame
        // propagates its new range to the other one next frame.
        let sx = self.spectrum_chart.last_x_bounds();
        let wx = self.waterfall_chart.last_x_bounds();
        if sx != self.last_synced_x {
            self.waterfall_chart.sync_x_bounds(sx);
            self.last_synced_x = sx;
        } else if wx != self.last_synced_x {
            self.spectrum_chart.sync_x_bounds(wx);
            self.last_synced_x = wx;
        }

        // Repaint is driven by the sim thread via ctx.request_repaint().
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Signal the sim thread to exit.  `running` alone won't do it — false
        // there means "paused, keep looping".  `quit` is the actual exit flag.
        // The join happens in main() after run_native returns so the window
        // closes immediately without freezing.
        self.shared.quit.store(true, Ordering::Relaxed);
    }
}
