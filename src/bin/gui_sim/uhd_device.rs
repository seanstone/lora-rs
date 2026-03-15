use rustfft::num_complex::Complex;
use std::{
    collections::VecDeque,
    ffi::CString,
    sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}},
    thread::JoinHandle,
};

use super::driver::Driver;

// ── FFI declarations ──────────────────────────────────────────────────────────

mod ffi {
    unsafe extern "C" {
        pub fn uhd_glue_open(
            args:        *const std::os::raw::c_char,
            rx_buf_out:  *mut usize,
            tx_buf_out:  *mut usize,
        ) -> std::os::raw::c_int;
        pub fn uhd_glue_close();

        pub fn uhd_glue_set_rx_rate(rate: f64);
        pub fn uhd_glue_set_rx_bw  (bw:   f64);
        pub fn uhd_glue_set_rx_freq(freq: f64);
        pub fn uhd_glue_set_rx_gain(gain: f64);

        pub fn uhd_glue_set_tx_rate(rate: f64);
        pub fn uhd_glue_set_tx_bw  (bw:   f64);
        pub fn uhd_glue_set_tx_freq(freq: f64);
        pub fn uhd_glue_set_tx_gain(gain: f64);

        pub fn uhd_glue_start_rx();
        pub fn uhd_glue_stop_rx();

        pub fn uhd_glue_recv(buf: *mut   i16, n: usize) -> usize;
        pub fn uhd_glue_send(buf: *const i16, n: usize) -> usize;

        pub fn uhd_glue_rx_buf_size() -> usize;
        pub fn uhd_glue_tx_buf_size() -> usize;
    }
}

// ── UhdDevice ─────────────────────────────────────────────────────────────────

pub(crate) struct UhdDevice {
    /// Interleaved i16 I/Q pairs queued for the TX thread.
    tx_queue:   Arc<Mutex<VecDeque<i16>>>,
    /// Received f32 I/Q samples accumulated by the RX thread.
    rx_buffer:  Arc<Mutex<VecDeque<Complex<f32>>>>,
    /// Digital TX amplitude scaling (applied when f32 → i16).
    signal_amp: Arc<Mutex<f32>>,
    /// Signals worker threads to exit.
    running:    Arc<AtomicBool>,
    /// Worker thread handles — joined in Drop before closing UHD handles.
    tx_thread:  Option<JoinHandle<()>>,
    rx_thread:  Option<JoinHandle<()>>,
}

impl UhdDevice {
    /// Open the USRP described by `args` and configure RX/TX.
    ///
    /// `samp_rate_hz` and `bw_hz` are the effective sample rate / bandwidth
    /// already computed by `effective_sr_and_os`.
    pub fn new(
        args:        &str,
        freq_hz:     f64,
        samp_rate_hz: f64,
        bw_hz:       f64,
        rx_gain_db:  f64,
        tx_gain_db:  f64,
    ) -> Result<Self, String> {
        let c_args = CString::new(args).map_err(|e| e.to_string())?;

        let mut rx_buf_size = 0usize;
        let mut tx_buf_size = 0usize;
        let ret = unsafe {
            ffi::uhd_glue_open(c_args.as_ptr(), &mut rx_buf_size, &mut tx_buf_size)
        };
        if ret != 0 { return Err("uhd_glue_open failed".into()); }

        unsafe {
            ffi::uhd_glue_set_rx_rate(samp_rate_hz);
            ffi::uhd_glue_set_rx_bw(bw_hz);
            ffi::uhd_glue_set_rx_freq(freq_hz);
            ffi::uhd_glue_set_rx_gain(rx_gain_db);

            ffi::uhd_glue_set_tx_rate(samp_rate_hz);
            ffi::uhd_glue_set_tx_bw(bw_hz);
            ffi::uhd_glue_set_tx_freq(freq_hz);
            ffi::uhd_glue_set_tx_gain(tx_gain_db);

            ffi::uhd_glue_start_rx();
        }

        let tx_queue   = Arc::new(Mutex::new(VecDeque::<i16>::new()));
        let rx_buffer  = Arc::new(Mutex::new(VecDeque::<Complex<f32>>::new()));
        let signal_amp = Arc::new(Mutex::new(1.0f32));
        let running    = Arc::new(AtomicBool::new(true));

        // ── TX worker ─────────────────────────────────────────────────────────
        // Drains the tx_queue in chunks of tx_buf_size; pads with zeros when
        // the queue is empty so the hardware TX stream never stalls.
        let tx_thread = {
            let tx_q = tx_queue.clone();
            let run  = running.clone();
            std::thread::spawn(move || {
                let buf_size    = tx_buf_size.max(1024);
                let needed_i16  = buf_size * 2;
                let mut send_buf = vec![0i16; needed_i16];
                while run.load(Ordering::Relaxed) {
                    {
                        let mut q   = tx_q.lock().unwrap();
                        let avail   = q.len().min(needed_i16);
                        for b in send_buf[..avail].iter_mut() { *b = q.pop_front().unwrap(); }
                        for b in send_buf[avail..].iter_mut() { *b = 0; }
                    }
                    unsafe { ffi::uhd_glue_send(send_buf.as_ptr(), buf_size); }
                }
            })
        };

        // ── RX worker ─────────────────────────────────────────────────────────
        // Calls uhd_glue_recv in a loop (0.1 s timeout); converts sc16 →
        // Complex<f32> and appends to rx_buffer.
        let rx_thread = {
            let rx_buf = rx_buffer.clone();
            let run    = running.clone();
            std::thread::spawn(move || {
                let buf_size = rx_buf_size.max(1024);
                let mut buf  = vec![0i16; buf_size * 2];
                while run.load(Ordering::Relaxed) {
                    let n = unsafe { ffi::uhd_glue_recv(buf.as_mut_ptr(), buf_size) };
                    if n > 0 {
                        let samples: Vec<Complex<f32>> = buf[..n * 2]
                            .chunks_exact(2)
                            .map(|c| Complex::new(c[0] as f32 / 32768.0, c[1] as f32 / 32768.0))
                            .collect();
                        rx_buf.lock().unwrap().extend(samples);
                    }
                }
            })
        };

        Ok(Self { tx_queue, rx_buffer, signal_amp, running,
                  tx_thread: Some(tx_thread), rx_thread: Some(rx_thread) })
    }
}

impl Drop for UhdDevice {
    fn drop(&mut self) {
        // Signal threads to stop and issue a stream-stop so uhd_glue_recv
        // returns promptly (0.1 s timeout) rather than blocking up to 3 s.
        self.running.store(false, Ordering::Relaxed);
        unsafe { ffi::uhd_glue_stop_rx(); }
        // Wait for both threads to finish before freeing UHD handles —
        // otherwise they may call uhd_glue_send/recv on freed pointers.
        if let Some(h) = self.tx_thread.take() { let _ = h.join(); }
        if let Some(h) = self.rx_thread.take() { let _ = h.join(); }
        unsafe { ffi::uhd_glue_close(); }
    }
}

impl Driver for UhdDevice {
    /// Convert f32 unit-amplitude samples → sc16 and queue for the TX thread.
    fn push_samples(&mut self, samples: Vec<Complex<f32>>) {
        let amp = *self.signal_amp.lock().unwrap();
        let mut q = self.tx_queue.lock().unwrap();
        for s in samples {
            q.push_back((s.re * amp * 32767.0).clamp(-32768.0, 32767.0) as i16);
            q.push_back((s.im * amp * 32767.0).clamp(-32768.0, 32767.0) as i16);
        }
    }

    /// Block until at least `n` received samples are available, then drain them.
    fn tick(&mut self, n: usize) -> Vec<Complex<f32>> {
        loop {
            if self.rx_buffer.lock().unwrap().len() >= n { break; }
            std::thread::sleep(std::time::Duration::from_micros(500));
        }
        let mut buf = self.rx_buffer.lock().unwrap();
        buf.drain(..n).collect()
    }

    /// Number of i16 values in the TX queue, expressed as complex samples.
    fn pending_samples(&self) -> usize {
        self.tx_queue.lock().unwrap().len() / 2
    }

    fn clear(&mut self) {
        self.tx_queue.lock().unwrap().clear();
        self.rx_buffer.lock().unwrap().clear();
    }

    /// Applied as a digital gain when converting f32 → i16 in push_samples.
    fn set_signal_amp(&mut self, amp: f32) {
        *self.signal_amp.lock().unwrap() = amp;
    }

    /// No-op for hardware — noise comes from the RF environment.
    fn set_noise_sigma(&mut self, _sigma: f32) {}

    fn set_hw_rx_gain(&mut self, db: f64) {
        unsafe { ffi::uhd_glue_set_rx_gain(db); }
    }

    fn set_hw_tx_gain(&mut self, db: f64) {
        unsafe { ffi::uhd_glue_set_tx_gain(db); }
    }
}
