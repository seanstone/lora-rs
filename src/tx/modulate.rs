use rustfft::num_complex::Complex;
use std::f64::consts::TAU;

fn append_upchirp(out: &mut Vec<Complex<f32>>, id: u32, sf: u8, os_factor: u32) {
    let n        = 1u32 << sf;
    let sps      = n * os_factor;
    let n_fold   = (n - id) * os_factor; // sample index where frequency wraps
    for k in 0..sps {
        let t     = k as f64 / os_factor as f64;
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

fn append_downchirp(out: &mut Vec<Complex<f32>>, sf: u8, os_factor: u32, samples: usize) {
    let n = 1u32 << sf;
    for k in 0..samples {
        let t     = k as f64 / os_factor as f64;
        let phase = t * t / (2.0 * n as f64) - 0.5 * t;
        let (sin, cos) = (TAU * phase).sin_cos();
        out.push(Complex::new(cos as f32, -sin as f32)); // conjugate
    }
}

/// Generate LoRa IQ samples: preamble + sync word + 2.25 downchirps + payload.
///
/// `os_factor = samp_rate / bw`. Each symbol is `2^sf * os_factor` samples.
pub fn modulate(
    symbols:      &[u32],
    sf:           u8,
    sync_word:    u8,
    preamble_len: u16,
    os_factor:    u32,
) -> Vec<Complex<f32>> {
    let n   = 1usize << sf;
    let sps = n * os_factor as usize; // samples per symbol
    let sw0 = ((sync_word as u32 & 0xF0) >> 4) << 3;
    let sw1 =  (sync_word as u32 & 0x0F)       << 3;

    let mut out = Vec::with_capacity((preamble_len as usize + 4 + symbols.len()) * sps + sps / 4);

    for _ in 0..preamble_len {
        append_upchirp(&mut out, 0, sf, os_factor);
    }
    append_upchirp(&mut out, sw0, sf, os_factor);
    append_upchirp(&mut out, sw1, sf, os_factor);
    append_downchirp(&mut out, sf, os_factor, 2 * sps + sps / 4);
    for &sym in symbols {
        // Standard LoRa symbol convention: add 1 mod N before modulation.
        // Real LoRa chips (SX1276/SX1262) use this offset; the RX compensates
        // with -1 in fft_demod.  This ensures interop with external nodes.
        let id = (sym + 1) % (1u32 << sf);
        append_upchirp(&mut out, id, sf, os_factor);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_length() {
        for (sf, pl, ns, os) in [
            (7u8, 8u16, 0usize, 1u32), (7, 8, 5, 1), (8, 8, 10, 1), (9, 6, 3, 1), (12, 8, 1, 1),
            (7, 8, 5, 2), (8, 8, 4, 4),
        ] {
            let n      = 1usize << sf;
            let sps    = n * os as usize;
            let syms   = vec![0u32; ns];
            let out    = modulate(&syms, sf, 0x12, pl, os);
            let expect = (pl as usize + 2 + ns) * sps + 2 * sps + sps / 4;
            assert_eq!(out.len(), expect, "sf={sf} pl={pl} ns={ns} os={os}");
        }
    }

    #[test]
    fn unit_magnitude() {
        for (sf, os) in [(7u8, 1u32), (8, 1), (9, 2), (12, 4)] {
            let syms = vec![0u32, 1, (1u32 << sf) - 1];
            for s in modulate(&syms, sf, 0x12, 8, os) {
                assert!((s.norm() - 1.0).abs() < 1e-5, "sf={sf} os={os}");
            }
        }
    }

    #[test]
    fn preamble_chirps_identical() {
        for (sf, os) in [(7u8, 1u32), (8, 2)] {
            let sps = (1usize << sf) * os as usize;
            let out = modulate(&[], sf, 0x12, 8, os);
            for p in 1..8usize {
                for k in 0..sps {
                    let diff = (out[k] - out[p * sps + k]).norm();
                    assert!(diff < 1e-5, "sf={sf} os={os} p={p} k={k}");
                }
            }
        }
    }

    #[test]
    fn distinct_symbols_differ() {
        for (sf, os) in [(7u8, 1u32), (9, 2)] {
            let sps  = (1usize << sf) * os as usize;
            let out0 = modulate(&[0], sf, 0x12, 1, os);
            let out1 = modulate(&[1], sf, 0x12, 1, os);
            let off  = out0.len() - sps;
            let diff = (0..sps).any(|k| (out0[off + k] - out1[off + k]).norm() > 1e-5);
            assert!(diff, "sf={sf} os={os}");
        }
    }

    #[test]
    fn sync_word_differs_from_preamble() {
        for (sf, os) in [(7u8, 1u32), (8, 2)] {
            let sps = (1usize << sf) * os as usize;
            let out = modulate(&[], sf, 0x12, 4, os);
            let diff = (0..sps).any(|k| (out[k] - out[4 * sps + k]).norm() > 1e-5);
            assert!(diff, "sf={sf} os={os}");
        }
    }
}
