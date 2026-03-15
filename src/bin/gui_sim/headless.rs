use lora::ui::{SpectrumPlot, WaterfallPlot};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU32, Ordering}},
    time::Duration,
};

use super::shared::{SimShared, Stats};
use super::sim::sim_loop;
use super::{
    DEFAULT_SAMP_RATE_KHZ, DEFAULT_BW_KHZ, DEFAULT_FFT_SIZE,
    DEFAULT_SIGNAL_DB, DEFAULT_INTERVAL_MS,
    effective_sr_and_os,
};

/// Run the simulator headlessly with the **same** shared-state defaults as
/// GUI mode (sample rate, BW, FFT size, signal/noise levels, interval).
///
/// The sim_loop is spawned identically to how `GuiApp` spawns it, except
/// no egui context is passed (so no display worker). This makes headless
/// output directly comparable to what the GUI would show.
pub(crate) fn run_headless(sf: u8, snr_db_val: f32, packet_count: usize) {
    // ── Same defaults as GuiApp::new ─────────────────────────────────────────
    let samp_rate_khz = DEFAULT_SAMP_RATE_KHZ;
    let bw_khz        = DEFAULT_BW_KHZ;
    let (eff_sr, os_factor) = effective_sr_and_os(samp_rate_khz, bw_khz);
    let fft_size  = DEFAULT_FFT_SIZE;
    let signal_db = DEFAULT_SIGNAL_DB;
    let noise_db  = signal_db - snr_db_val;

    let init_spec: Vec<[f64; 2]> = (0..fft_size).map(|i| [i as f64, -80.0]).collect();
    let spectrum_plot  = SpectrumPlot::new("Spectrum",   init_spec.clone(), -80.0, 80.0);
    let waterfall_plot = WaterfallPlot::new("Waterfall", init_spec,         -80.0);
    waterfall_plot.set_freq(fft_size as f64 / 2.0);
    waterfall_plot.set_bw(fft_size as f64);

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

    { let s = shared.clone(); std::thread::spawn(move || sim_loop(s, None)); }

    let mut printed = 0usize;
    loop {
        std::thread::sleep(Duration::from_millis(50));
        let log = shared.log.lock().unwrap().clone();
        for entry in log.iter().skip(printed) {
            let mark = if entry.ok { "OK  " } else { "FAIL" };
            println!("[{mark}] {}", entry.payload);
        }
        printed = log.len();

        let stats = shared.stats.lock().unwrap().clone();
        if stats.tx_count >= packet_count {
            // Give the RX pipeline time to drain remaining packets.
            std::thread::sleep(Duration::from_secs(2));
            let log = shared.log.lock().unwrap().clone();
            for entry in log.iter().skip(printed) {
                let mark = if entry.ok { "OK  " } else { "FAIL" };
                println!("[{mark}] {}", entry.payload);
            }
            let stats = shared.stats.lock().unwrap().clone();
            shared.running.store(false, Ordering::Relaxed);
            let accounted = stats.rx_count + stats.rx_lost;
            let per = if accounted > 0 {
                100.0 * stats.rx_lost as f32 / accounted as f32
            } else { 0.0 };
            println!("─── SF={sf}  SNR={snr_db_val:.1} dB  {}/{} rx  {} lost  PER {per:.1}% ───",
                     stats.rx_count, stats.tx_count, stats.rx_lost);
            break;
        }
    }
}
