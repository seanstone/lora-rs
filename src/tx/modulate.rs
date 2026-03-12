use rustfft::num_complex::Complex;
use std::f64::consts::TAU;

fn append_upchirp(out: &mut Vec<Complex<f32>>, id: u32, sf: u8) {
    let n      = 1u32 << sf;
    let n_fold = n - id;
    for k in 0..n {
        let t     = k as f64;
        let phase = t * t / (2.0 * n as f64)
            + if k < n_fold {
                (id as f64 / n as f64 - 0.5) * t
            } else {
                (id as f64 / n as f64 - 1.5) * t
            };
        let (sin, cos) = (TAU * phase).sin_cos();
        out.push(Complex::new(cos as f32, sin as f32));
    }
}

fn append_downchirp(out: &mut Vec<Complex<f32>>, sf: u8, samples: usize) {
    let n = 1u32 << sf;
    for k in 0..samples {
        let t     = k as f64;
        let phase = t * t / (2.0 * n as f64) - 0.5 * t;
        let (sin, cos) = (TAU * phase).sin_cos();
        out.push(Complex::new(cos as f32, -sin as f32)); // conjugate
    }
}

/// Generate LoRa IQ samples: preamble + sync word + 2.25 downchirps + payload.
pub fn modulate(
    symbols:      &[u32],
    sf:           u8,
    sync_word:    u8,
    preamble_len: u16,
) -> Vec<Complex<f32>> {
    let n   = 1usize << sf;
    let sw0 = ((sync_word as u32 & 0xF0) >> 4) << 3;
    let sw1 =  (sync_word as u32 & 0x0F)       << 3;

    let mut out = Vec::with_capacity((preamble_len as usize + 4 + symbols.len()) * n + n / 4);

    for _ in 0..preamble_len {
        append_upchirp(&mut out, 0, sf);
    }
    append_upchirp(&mut out, sw0, sf);
    append_upchirp(&mut out, sw1, sf);
    append_downchirp(&mut out, sf, 2 * n + n / 4);
    for &sym in symbols {
        append_upchirp(&mut out, sym, sf);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_length() {
        for (sf, pl, ns) in [(7u8, 8u16, 0usize), (7, 8, 5), (8, 8, 10), (9, 6, 3), (12, 8, 1)] {
            let n      = 1usize << sf;
            let syms   = vec![0u32; ns];
            let out    = modulate(&syms, sf, 0x12, pl);
            let expect = (pl as usize + 2 + ns) * n + 2 * n + n / 4;
            assert_eq!(out.len(), expect, "sf={sf} pl={pl} ns={ns}");
        }
    }

    #[test]
    fn unit_magnitude() {
        for sf in [7u8, 8, 9, 12] {
            let syms = vec![0u32, 1, (1u32 << sf) - 1];
            for s in modulate(&syms, sf, 0x12, 8) {
                assert!((s.norm() - 1.0).abs() < 1e-5, "sf={sf}");
            }
        }
    }

    #[test]
    fn preamble_chirps_identical() {
        for sf in [7u8, 8, 9] {
            let n   = 1usize << sf;
            let out = modulate(&[], sf, 0x12, 8);
            for p in 1..8usize {
                for k in 0..n {
                    let diff = (out[k] - out[p * n + k]).norm();
                    assert!(diff < 1e-5, "sf={sf} p={p} k={k}");
                }
            }
        }
    }

    #[test]
    fn distinct_symbols_differ() {
        for sf in [7u8, 8, 9, 12] {
            let n    = 1usize << sf;
            let out0 = modulate(&[0], sf, 0x12, 1);
            let out1 = modulate(&[1], sf, 0x12, 1);
            let off  = out0.len() - n;
            let diff = (0..n).any(|k| (out0[off + k] - out1[off + k]).norm() > 1e-5);
            assert!(diff, "sf={sf}");
        }
    }

    #[test]
    fn sync_word_differs_from_preamble() {
        for sf in [7u8, 8, 9] {
            let n   = 1usize << sf;
            let out = modulate(&[], sf, 0x12, 4);
            let diff = (0..n).any(|k| (out[k] - out[4 * n + k]).norm() > 1e-5);
            assert!(diff, "sf={sf}");
        }
    }
}
