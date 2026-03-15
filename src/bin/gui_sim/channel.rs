use rand::{Rng, RngExt, SeedableRng};
use rustfft::num_complex::Complex;
use std::collections::VecDeque;

// ─── Channel ──────────────────────────────────────────────────────────────────
//
// Pure streaming per-sample AWGN mixer. The channel has no concept of packet
// boundaries — it is a FIFO of clean IQ samples with per-sample noise added on
// output.
//
// TX side: push clean samples via `push_samples()`.
// RX side: call `tick(n)` to pull `n` mixed (signal + noise) samples.
//
// This design maps directly to a hardware driver: replace `tick()` with real
// ADC reads and `push_samples()` with DAC writes.

pub(crate) struct Channel {
    /// Instantaneous noise sigma — updated each tick, applied per sample.
    pub noise_sigma: f32,
    /// Instantaneous TX signal gain — updated each tick, applied per sample.
    pub signal_amp:  f32,
    rng:             rand::rngs::StdRng,
    /// Clean samples waiting to be streamed out with AWGN.
    buffer:          VecDeque<Complex<f32>>,
}

impl Channel {
    pub fn new(noise_sigma: f32, signal_amp: f32) -> Self {
        Self {
            noise_sigma,
            signal_amp,
            rng:    rand::rngs::StdRng::seed_from_u64(0xC0FFee),
            buffer: VecDeque::new(),
        }
    }

    pub fn set_noise_sigma(&mut self, s: f32) { self.noise_sigma = s; }
    pub fn set_signal_amp(&mut self, a: f32)  { self.signal_amp  = a; }

    /// Enqueue clean (noise-free, unit-amplitude) IQ samples for streaming.
    pub fn push_samples(&mut self, samples: Vec<Complex<f32>>) {
        self.buffer.extend(samples);
    }

    /// Flush all queued samples.
    pub fn clear(&mut self) { self.buffer.clear(); }

    /// Samples still queued (for buffer-lag reporting).
    pub fn pending_samples(&self) -> usize { self.buffer.len() }

    /// Produce exactly `n` mixed samples.
    ///
    /// Each sample = clean × signal_amp + AWGN(noise_sigma). During silence
    /// (buffer empty) the clean component is zero — only noise is emitted.
    pub fn tick(&mut self, n: usize) -> Vec<Complex<f32>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let clean = self.buffer.pop_front()
                .map(|s| s * self.signal_amp)
                .unwrap_or_default();
            let noise = Complex::new(
                box_muller(&mut self.rng) * self.noise_sigma,
                box_muller(&mut self.rng) * self.noise_sigma,
            );
            out.push(clean + noise);
        }
        out
    }
}

/// Box-Muller: one unit-normal sample from two uniform draws.
fn box_muller(rng: &mut impl Rng) -> f32 {
    let u1 = (rng.random::<f32>() + 1e-7_f32).min(1.0);
    let u2 = rng.random::<f32>();
    (-2.0_f32 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}
