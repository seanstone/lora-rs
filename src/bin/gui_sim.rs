/// LoRa GUI/CLI simulator — raw spectrum + waterfall of a continuous IQ stream.
///
/// Module layout (src/bin/gui_sim/)
/// ─────────────────────────────────
///   shared   — SimShared, Stats, LogEntry
///   tx       — Tx modulator, TxJob/TxResult, tx_worker
///   channel  — Channel: pure streaming per-sample AWGN mixer (no packet tracking)
///   rx       — Rx demodulator, streaming RxJob worker with frame-sync buffer drain
///   display  — spectrum_window, display_worker
///   sim      — sim_loop (drives TX→Channel→RX streaming pipeline)
///   gui      — GuiApp (eframe/egui)
///   headless — run_headless (CLI / CI mode)
///
/// Data flow: TX numbers packets in the payload ([seq_u16_le][text]), modulates
/// clean IQ, and pushes raw samples into the channel.  The channel adds AWGN
/// and streams mixed samples out — it has no concept of packet boundaries.
/// The RX worker accumulates mixed samples, runs frame_sync to discover packets,
/// decodes them, and detects lost packets via sequence gaps.
///
/// Usage:
///   gui_sim [sf]                  — GUI mode
///   gui_sim --cli [sf] [snr] [n]  — headless mode, print n packet results

#[path = "gui_sim/shared.rs"]   mod shared;
#[path = "gui_sim/tx.rs"]       mod tx;
#[path = "gui_sim/channel.rs"]  mod channel;
#[path = "gui_sim/rx.rs"]       mod rx;
#[path = "gui_sim/display.rs"]  mod display;
#[path = "gui_sim/sim.rs"]      mod sim;
#[path = "gui_sim/gui.rs"]      mod gui;
#[path = "gui_sim/headless.rs"] mod headless;

// ─── Shared constants ─────────────────────────────────────────────────────────

pub(crate) const SR_OPTIONS_KHZ:        &[u32]  = &[125, 250, 500, 1000, 2000, 4000];
pub(crate) const BW_OPTIONS_KHZ:        &[u32]  = &[125, 250, 500, 1000];

pub(crate) const DEFAULT_SF:            u8      = 7;
pub(crate) const DEFAULT_SAMP_RATE_KHZ: u32     = 1000;
pub(crate) const DEFAULT_BW_KHZ:        u32     = 250;
pub(crate) const DEFAULT_FFT_SIZE:      usize   = 1024;
pub(crate) const DEFAULT_SIGNAL_DB:     f32     = -20.0;
pub(crate) const DEFAULT_NOISE_DB:      f32     = -35.0;
pub(crate) const DEFAULT_INTERVAL_MS:   u64     = 500;

// ─── Shared helpers ───────────────────────────────────────────────────────────

pub(crate) fn khz_label(v: u32) -> String {
    if v >= 1000 { format!("{}M", v / 1000) } else { format!("{}k", v) }
}

pub(crate) fn effective_sr_and_os(samp_rate_khz: u32, bw_khz: u32) -> (u32, u32) {
    let sr = samp_rate_khz.max(bw_khz);
    (sr, sr / bw_khz)
}

/// Convert amplitude dBFS to linear amplitude.
pub(crate) fn db_to_amp(db: f32) -> f32 { 10f32.powf(db / 20.0) }

/// SNR in dB from signal and noise amplitude dBFS values.
pub(crate) fn snr_db(signal_db: f32, noise_db: f32) -> f32 { signal_db - noise_db }

// ─── Entry point ──────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cli_mode = args.iter().any(|a| a == "--cli" || a == "-c");

    if cli_mode {
        // Usage: gui_sim --cli [sf=7] [snr_db=10] [packets=20]
        let positional: Vec<&str> = args.iter()
            .filter(|a| !a.starts_with('-') && *a != &args[0])
            .map(|s| s.as_str())
            .collect();
        let sf      = positional.first().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_SF);
        let snr     = positional.get(1).and_then(|s| s.parse().ok()).unwrap_or(10.0_f32);
        let packets = positional.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_usize);
        headless::run_headless(sf, snr, packets);
        return;
    }

    let sf = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_SF);
    use eframe::egui;
    eframe::run_native(
        "LoRa Link Simulator",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_title("LoRa Link Simulator")
                .with_inner_size([1400.0, 750.0]),
            ..Default::default()
        },
        Box::new(move |_cc| Ok(Box::new(gui::GuiApp::new(sf)))),
    ).unwrap();
}
