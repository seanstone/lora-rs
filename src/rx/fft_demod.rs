use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::TAU;

/// Build a CFO-corrected downchirp matching gr-lora_sdr's fft_demod_impl.cc:
///   1. build_upchirp(mod(cfo_int, N))
///   2. conjugate to get downchirp
///   3. multiply by exp(-j*2*pi*cfo_frac/N*n) for fractional correction
fn make_downchirp_cfo(sf: u8, cfo_int: f64, cfo_frac: f64) -> Vec<Complex<f32>> {
    let n = 1usize << sf;
    let id = ((cfo_int as isize).rem_euclid(n as isize)) as usize;
    let n_fold = n - id;

    (0..n).map(|k| {
        let t = k as f64;
        // upchirp with circular shift by id
        let phase_up = t * t / (2.0 * n as f64)
            + if k < n_fold {
                (id as f64 / n as f64 - 0.5) * t
            } else {
                (id as f64 / n as f64 - 1.5) * t
            };
        // conjugate to get downchirp
        let (sin_up, cos_up) = (TAU * phase_up).sin_cos();
        let dc = Complex::new(cos_up as f32, -sin_up as f32);

        // fractional CFO correction: exp(-j*2*pi*cfo_frac/N*n)
        let cfo_phase = -TAU * cfo_frac / n as f64 * k as f64;
        let (sin_c, cos_c) = cfo_phase.sin_cos();
        let cfo_rot = Complex::new(cos_c as f32, sin_c as f32);

        dc * cfo_rot
    }).collect()
}

/// Plain downchirp (zero CFO) — backward-compatible for simulation / self-to-self.
fn make_downchirp(sf: u8) -> Vec<Complex<f32>> {
    make_downchirp_cfo(sf, 0.0, 0.0)
}

/// Demodulate payload symbols with CFO-corrected dechirping and per-symbol SFO
/// compensation, matching the gr-lora_sdr reference design.
///
/// * `cfo_int`, `cfo_frac`: integer and fractional CFO from `frame_sync`.
/// * `sfo_hat`: sampling frequency offset estimate from `frame_sync`.
///
/// When all three are 0.0 (simulation / self-to-self), this reduces to the
/// original plain-downchirp argmax demod.
pub fn fft_demod(
    samples: &[Complex<f32>],
    sf: u8,
    os_factor: u32,
    cfo_int: f64,
    cfo_frac: f64,
    sfo_hat: f64,
) -> Vec<u32> {
    let n   = 1usize << sf;
    let sps = n * os_factor as usize;
    let os  = os_factor as usize;

    // Build CFO-corrected downchirp (matches gr-lora_sdr fft_demod_impl.cc:284-290)
    let downchirp = make_downchirp_cfo(sf, cfo_int, cfo_frac);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buf = vec![Complex::new(0.0_f32, 0.0_f32); n];
    let mut out = Vec::with_capacity(samples.len() / sps);

    // Per-symbol SFO tracking (matches gr-lora_sdr SFO_COMPENSATION state)
    let mut sfo_cum: f64 = 0.0;
    let mut w = 0usize;

    while w + sps <= samples.len() {
        // Downsample: take every os_factor-th sample.
        // In gr-lora_sdr, the center-of-group offset (os/2) and STO correction
        // are applied during the frame_sync downsampling step. Here, frame_sync
        // has already applied STO as a sample-level shift, so we sample at the
        // start of each group to maintain phase consistency.
        for (i, b) in buf.iter_mut().enumerate() {
            let idx = w + os * i;
            if idx < samples.len() {
                *b = samples[idx];
            } else {
                *b = Complex::new(0.0, 0.0);
            }
        }

        // Dechirp with CFO-corrected downchirp
        for (b, d) in buf.iter_mut().zip(downchirp.iter()) {
            *b = *b * d;
        }
        fft.process(&mut buf);

        let idx = buf.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        // Standard LoRa symbol convention: subtract 1 mod N
        // (matches gr-lora_sdr fft_demod_impl.cc:313)
        // Real LoRa chips (SX1276/SX1262) modulate with an implicit +1 offset,
        // so the RX must compensate with -1.
        let sym = if idx == 0 { n as u32 - 1 } else { idx - 1 };
        out.push(sym);

        // SFO compensation: adjust consumed samples when drift exceeds half a
        // sample period (matches gr-lora_sdr frame_sync_impl.cc:856-860)
        let mut consume = sps as isize;
        if sfo_cum.abs() > 1.0 / (2.0 * os as f64) {
            let sign = if sfo_cum > 0.0 { 1isize } else { -1isize };
            consume -= sign;
            sfo_cum -= sign as f64 / os as f64;
        }
        sfo_cum += sfo_hat;

        w = (w as isize + consume).max(0) as usize;
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

        let recovered = fft_demod(
            &sync.symbols, sf, os_factor,
            sync.cfo_int, sync.cfo_frac, sync.sfo_hat,
        );
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
