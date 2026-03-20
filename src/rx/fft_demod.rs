use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::TAU;

fn make_downchirp(sf: u8) -> Vec<Complex<f32>> {
    let n = 1usize << sf;
    (0..n).map(|k| {
        let t     = k as f64;
        let phase = t * t / (2.0 * n as f64) - 0.5 * t;
        let (sin, cos) = (TAU * phase).sin_cos();
        Complex::new(cos as f32, -sin as f32)
    }).collect()
}

/// Demodulate each symbol window: center-sample decimate, dechirp, FFT argmax.
///
/// Input: raw oversampled samples (as returned by `frame_sync`).
/// `os_factor = samp_rate / bw`; each symbol is `2^sf * os_factor` samples.
pub fn fft_demod(samples: &[Complex<f32>], sf: u8, os_factor: u32) -> Vec<u32> {
    let n         = 1usize << sf;
    let sps       = n * os_factor as usize;
    let downchirp = make_downchirp(sf);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buf = vec![Complex::new(0.0_f32, 0.0_f32); n];
    let mut out = Vec::with_capacity(samples.len() / sps);

    let mut w = 0usize;
    while w + sps <= samples.len() {
        // Stride decimation: take sample 0 of each os_factor-wide group.
        for (i, b) in buf.iter_mut().enumerate() {
            *b = samples[w + os_factor as usize * i];
        }
        // Dechirp.
        for (b, d) in buf.iter_mut().zip(downchirp.iter()) {
            *b = *b * d;
        }
        fft.process(&mut buf);

        let idx = buf.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        out.push(idx);
        w += sps;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tx::modulate::modulate;
    use crate::rx::frame_sync::frame_sync;

    fn roundtrip(sf: u8, preamble_len: u16, n_syms: usize, os_factor: u32) {
        let n    = 1u32 << sf;
        let syms: Vec<u32> = (0..n_syms).map(|i| (i as u32 * 13 + 7) % n).collect();

        let iq   = modulate(&syms, sf, 0x12, preamble_len, os_factor);
        let sync = frame_sync(&iq, sf, 0x12, preamble_len, os_factor, 0.0, 0.0);
        assert!(sync.found, "frame_sync failed sf={sf} pl={preamble_len} ns={n_syms} os={os_factor}");

        let recovered = fft_demod(&sync.symbols, sf, os_factor);
        assert_eq!(recovered.len(), n_syms, "count mismatch sf={sf} os={os_factor}");
        assert_eq!(recovered, syms, "symbol mismatch sf={sf} os={os_factor}");
    }

    // os_factor = 1 (original behaviour)
    #[test] fn rt_sf7_ns1()  { roundtrip(7,  8, 1,  1); }
    #[test] fn rt_sf7_ns5()  { roundtrip(7,  8, 5,  1); }
    #[test] fn rt_sf7_ns10() { roundtrip(7,  8, 10, 1); }
    #[test] fn rt_sf8_ns4()  { roundtrip(8,  8, 4,  1); }
    #[test] fn rt_sf9_ns3()  { roundtrip(9,  8, 3,  1); }
    #[test] fn rt_sf7_pl6()  { roundtrip(7,  6, 2,  1); }
    #[test] fn rt_sf12_ns1() { roundtrip(12, 8, 1,  1); }

    // os_factor = 2
    #[test] fn rt_sf7_ns5_os2()  { roundtrip(7, 8, 5,  2); }
    #[test] fn rt_sf8_ns4_os2()  { roundtrip(8, 8, 4,  2); }
    #[test] fn rt_sf9_ns3_os2()  { roundtrip(9, 8, 3,  2); }

    // os_factor = 4
    #[test] fn rt_sf7_ns3_os4()  { roundtrip(7, 8, 3,  4); }
}
