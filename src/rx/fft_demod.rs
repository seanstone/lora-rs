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

/// Demodulate each N-sample window: dechirp + FFT argmax.
/// No -1 shift — matches standalone `gray_demap` which has no +1 offset.
pub fn fft_demod(samples: &[Complex<f32>], sf: u8) -> Vec<u32> {
    let n         = 1usize << sf;
    let downchirp = make_downchirp(sf);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buf = vec![Complex::new(0.0_f32, 0.0_f32); n];
    let mut out = Vec::with_capacity(samples.len() / n);

    let mut w = 0usize;
    while w + n <= samples.len() {
        for (b, (s, d)) in buf.iter_mut().zip(samples[w..w + n].iter().zip(downchirp.iter())) {
            *b = s * d;
        }
        fft.process(&mut buf);

        let idx = buf.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        out.push(idx);
        w += n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tx::modulate::modulate;
    use crate::rx::frame_sync::frame_sync;

    fn roundtrip(sf: u8, preamble_len: u16, n_syms: usize) {
        let n    = 1u32 << sf;
        let syms: Vec<u32> = (0..n_syms).map(|i| (i as u32 * 13 + 7) % n).collect();

        let iq   = modulate(&syms, sf, 0x12, preamble_len);
        let sync = frame_sync(&iq, sf, 0x12, preamble_len);
        assert!(sync.found, "frame_sync failed sf={sf} pl={preamble_len} ns={n_syms}");

        let recovered = fft_demod(&sync.symbols, sf);
        assert_eq!(recovered.len(), n_syms, "count mismatch sf={sf}");
        assert_eq!(recovered, syms, "symbol mismatch sf={sf}");
    }

    #[test] fn rt_sf7_ns1()  { roundtrip(7,  8, 1);  }
    #[test] fn rt_sf7_ns5()  { roundtrip(7,  8, 5);  }
    #[test] fn rt_sf7_ns10() { roundtrip(7,  8, 10); }
    #[test] fn rt_sf8_ns4()  { roundtrip(8,  8, 4);  }
    #[test] fn rt_sf9_ns3()  { roundtrip(9,  8, 3);  }
    #[test] fn rt_sf7_pl6()  { roundtrip(7,  6, 2);  }
    #[test] fn rt_sf12_ns1() { roundtrip(12, 8, 1);  }
}
