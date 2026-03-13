use rustfft::{FftPlanner, num_complex::Complex};
use std::{f64::consts::TAU, sync::Arc};

use super::shared::SimShared;

/// (IQ window, is_last_in_batch): last window also updates the spectrum plot.
pub(crate) type DisplayJob = (Vec<Complex<f32>>, bool);

/// Hann-windowed FFT of exactly one `fft_size`-sample window, fftshifted.
/// Returns `Vec<[f64;2]>`: x = bin (0..fft_size), y = power dB.
pub(crate) fn spectrum_window(
    samples: &[Complex<f32>],
    hann:    &[f32],
    buf:     &mut Vec<Complex<f32>>,
    fft:     &dyn rustfft::Fft<f32>,
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

/// Runs Hann-windowed FFT on each incoming window and pushes to the plots.
pub(crate) fn display_worker(jobs: std::sync::mpsc::Receiver<DisplayJob>, shared: Arc<SimShared>) {
    let mut cur_fft_size = 0usize;
    let mut hann: Vec<f32>             = Vec::new();
    let mut fft_buf: Vec<Complex<f32>> = Vec::new();
    let mut planner = FftPlanner::<f32>::new();
    let mut fft: Arc<dyn rustfft::Fft<f32>> = planner.plan_fft_forward(1);

    while let Ok((window, is_last)) = jobs.recv() {
        let fft_size = window.len();
        if fft_size != cur_fft_size {
            cur_fft_size = fft_size;
            hann = (0..fft_size)
                .map(|i| (0.5 * (1.0 - (TAU * i as f64 / fft_size as f64).cos())) as f32)
                .collect();
            fft_buf = vec![Complex::new(0.0_f32, 0.0_f32); fft_size];
            fft = planner.plan_fft_forward(fft_size);
        }
        let spec = spectrum_window(&window, &hann, &mut fft_buf, fft.as_ref());
        shared.waterfall_plot.update(spec.clone());
        if is_last { shared.spectrum_plot.update(spec); }
    }
}
