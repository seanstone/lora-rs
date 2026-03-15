use lora::ui::{SpectrumPlot, WaterfallPlot};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU32}},
};

pub(crate) struct SimShared {
    pub running:        AtomicBool,
    pub clear_buf:      AtomicBool,
    pub sf:             Mutex<u8>,
    pub os_factor:      Mutex<u32>,
    pub samp_rate_khz:  Mutex<u32>,
    pub fft_size:       Mutex<usize>,
    pub signal_db:      Mutex<f32>,
    pub noise_db:       Mutex<f32>,
    pub interval_ms:    Mutex<u64>,
    pub spectrum_plot:  Arc<SpectrumPlot>,
    pub waterfall_plot: Arc<WaterfallPlot>,
    pub stats:          Mutex<Stats>,
    pub log:            Mutex<VecDeque<LogEntry>>,
    /// Display-buffer lag in ms (f32 bits stored in AtomicU32 for lock-free reads).
    pub buf_lag_ms:     AtomicU32,
    pub buf_overflow:   AtomicBool,
    pub buf_underflow:  AtomicBool,
}

#[derive(Default, Clone)]
pub(crate) struct Stats {
    pub tx_count: usize,
    pub rx_count: usize,
    pub rx_lost:  usize,
    pub last_tx:  String,
    pub last_rx:  String,
}

#[derive(Clone)]
pub(crate) struct LogEntry {
    pub ok:      bool,
    pub payload: String,
}

pub(crate) const MAX_LOG_ENTRIES: usize = 200;
