/// IQ transport layer abstraction and AWGN simulation channel.

use rand::{Rng, RngExt, SeedableRng};
use rustfft::num_complex::Complex;
use std::collections::VecDeque;

// ── Driver trait ──────────────────────────────────────────────────────────────

/// Abstraction over the IQ transport layer (simulation channel or SDR hardware).
///
/// The simulation loop calls:
/// - [`push_samples`](Driver::push_samples) to hand off clean TX samples.
/// - [`tick`](Driver::tick) to consume `n` samples from the receive path
///   (blocks on hardware, instant on [`Channel`]).
/// - [`pending_samples`](Driver::pending_samples) for throttle / lag display.
/// - [`clear`](Driver::clear) on reset or settings change.
/// - [`set_signal_amp`](Driver::set_signal_amp) /
///   [`set_noise_sigma`](Driver::set_noise_sigma) for level control.
pub trait Driver: Send {
    fn push_samples(&mut self, samples: Vec<Complex<f32>>);

    /// Produce exactly `n` received samples.  Blocks on hardware; instant in sim.
    fn tick(&mut self, n: usize) -> Vec<Complex<f32>>;

    /// Approximate number of TX samples still queued / in-flight.
    fn pending_samples(&self) -> usize;

    /// Flush all in-flight state (called on reset / settings change).
    fn clear(&mut self);

    fn set_signal_amp(&mut self, amp: f32);

    /// No-op for hardware drivers (noise comes from the RF environment).
    fn set_noise_sigma(&mut self, sigma: f32);

    /// Apply hardware RX gain in dB.  No-op for the simulation channel.
    fn set_hw_rx_gain(&mut self, _db: f64) {}

    /// Apply hardware TX gain in dB.  No-op for the simulation channel.
    fn set_hw_tx_gain(&mut self, _db: f64) {}

    /// Whether this driver supports park/unpark without full teardown.
    fn is_parkable(&self) -> bool { false }

    /// Stop hardware streaming and idle worker threads.  No-op for sim.
    fn park(&mut self) {}

    /// Restart hardware streaming with the given RF settings.  No-op for sim.
    fn unpark(&mut self, _freq_hz: f64, _sr_hz: f64, _bw_hz: f64,
               _rx_gain_db: f64, _tx_gain_db: f64) {}
}

// ── Channel ───────────────────────────────────────────────────────────────────

/// Pure streaming per-sample AWGN mixer.
///
/// Has no concept of packet boundaries — it is a FIFO of clean IQ samples
/// with per-sample noise added on output.
///
/// - TX side: push clean samples via [`push_samples`](Channel::push_samples).
/// - RX side: call [`tick`](Channel::tick) to pull `n` mixed samples.
///
/// This interface mirrors the [`Driver`] trait so a hardware driver can be
/// swapped in by simply replacing `Channel` with an SDR backend.
pub struct Channel {
    /// Instantaneous noise sigma — updated each tick, applied per sample.
    pub noise_sigma: f32,
    /// Instantaneous TX signal gain — updated each tick, applied per sample.
    pub signal_amp:  f32,
    rng:    rand::rngs::StdRng,
    buffer: VecDeque<Complex<f32>>,
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

    /// Enqueue clean (noise-free, unit-amplitude) IQ samples.
    pub fn push_samples(&mut self, samples: Vec<Complex<f32>>) {
        self.buffer.extend(samples);
    }

    /// Flush all queued samples.
    pub fn clear(&mut self) { self.buffer.clear(); }

    /// Samples still queued (for buffer-lag reporting).
    pub fn pending_samples(&self) -> usize { self.buffer.len() }

    /// Produce exactly `n` mixed samples (clean × signal_amp + AWGN).
    ///
    /// During silence (buffer empty) only noise is emitted.
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

    pub fn set_noise_sigma(&mut self, s: f32) { self.noise_sigma = s; }
    pub fn set_signal_amp(&mut self, a: f32)  { self.signal_amp  = a; }
}

impl Driver for Channel {
    fn push_samples(&mut self, samples: Vec<Complex<f32>>) { self.push_samples(samples); }
    fn tick(&mut self, n: usize) -> Vec<Complex<f32>>      { self.tick(n) }
    fn pending_samples(&self) -> usize                     { self.pending_samples() }
    fn clear(&mut self)                                    { self.clear(); }
    fn set_signal_amp(&mut self, amp: f32)                 { self.set_signal_amp(amp); }
    fn set_noise_sigma(&mut self, sigma: f32)              { self.set_noise_sigma(sigma); }
}

/// Box-Muller transform: one unit-normal variate from two uniform draws.
fn box_muller(rng: &mut impl Rng) -> f32 {
    let u1 = (rng.random::<f32>() + 1e-7_f32).min(1.0);
    let u2 = rng.random::<f32>();
    (-2.0_f32 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}
