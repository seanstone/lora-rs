/// Self-contained FFT spectrum analyzer for IQ sample visualization.
///
/// Owns the Hann window, FFT plan, and scratch buffer so callers don't need
/// a direct `rustfft` dependency.

use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;

pub struct SpectrumAnalyzer {
    fft_size: usize,
    hann:     Vec<f32>,
    buf:      Vec<Complex<f32>>,
    fft:      Arc<dyn rustfft::Fft<f32>>,
}

impl SpectrumAnalyzer {
    pub fn new(fft_size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let hann: Vec<f32> = (0..fft_size)
            .map(|i| {
                let x = std::f64::consts::TAU * i as f64 / fft_size as f64;
                (0.5 * (1.0 - x.cos())) as f32
            })
            .collect();
        Self {
            fft_size,
            hann,
            buf: vec![Complex::new(0.0, 0.0); fft_size],
            fft,
        }
    }

    /// Hann-windowed FFT of exactly `fft_size` IQ samples, fftshifted.
    ///
    /// Returns `Vec<[f64; 2]>`: `[bin_index, power_dB]`.
    /// Input shorter than `fft_size` is zero-padded; longer is truncated.
    pub fn compute(&mut self, samples: &[Complex<f32>]) -> Vec<[f64; 2]> {
        let n = self.fft_size;
        let half = n / 2;

        // Window + zero-pad.
        for i in 0..n {
            self.buf[i] = if i < samples.len() {
                samples[i] * self.hann[i]
            } else {
                Complex::new(0.0, 0.0)
            };
        }

        self.fft.process(&mut self.buf);

        (0..n)
            .map(|i| {
                let src = (i + half) % n;
                let pdb = 10.0 * (self.buf[src].norm_sqr() as f64 + 1e-20).log10();
                [i as f64, pdb.max(-120.0)]
            })
            .collect()
    }

    pub fn fft_size(&self) -> usize {
        self.fft_size
    }
}
