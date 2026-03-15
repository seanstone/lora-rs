use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::TAU;

pub struct FrameSyncResult {
    pub found:    bool,
    pub symbols:  Vec<Complex<f32>>,
    /// Number of input samples consumed. Safe to drain from a streaming buffer.
    ///
    /// When `found == true`: covers everything through the end of the extracted
    /// payload symbols.  When `found == false`: covers all samples that were
    /// scanned without finding a complete preamble (any partial preamble at the
    /// tail is *kept*).
    pub consumed: usize,
}

fn make_downchirp(sf: u8) -> Vec<Complex<f32>> {
    let n = 1usize << sf;
    (0..n).map(|k| {
        let t     = k as f64;
        let phase = t * t / (2.0 * n as f64) - 0.5 * t;
        let (sin, cos) = (TAU * phase).sin_cos();
        Complex::new(cos as f32, -sin as f32)
    }).collect()
}

/// Dechirp + FFT one symbol window of `sps = n * os_factor` raw samples.
/// Center-sample decimates to N before the N-point FFT.
fn symbol_bin(
    buf:       &mut Vec<Complex<f32>>,
    samples:   &[Complex<f32>],  // sps samples
    os_factor: usize,
    downchirp: &[Complex<f32>],
    fft:       &dyn rustfft::Fft<f32>,
) -> usize {
    // Stride decimation: take sample 0 of each os_factor-wide group.
    for (i, b) in buf.iter_mut().enumerate() {
        *b = samples[os_factor * i];
    }
    // Dechirp then FFT.
    for (b, d) in buf.iter_mut().zip(downchirp.iter()) {
        *b = *b * d;
    }
    fft.process(buf);
    buf.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Detect a LoRa preamble in `samples` and return the raw payload IQ windows.
///
/// `os_factor = samp_rate / bw`. Each symbol occupies `2^sf * os_factor` samples.
pub fn frame_sync(
    samples:      &[Complex<f32>],
    sf:           u8,
    _sync_word:   u8,
    preamble_len: u16,
    os_factor:    u32,
) -> FrameSyncResult {
    let n         = 1usize << sf;
    let sps       = n * os_factor as usize;
    let n_up_req  = preamble_len as usize - 3;
    let downchirp = make_downchirp(sf);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buf            = vec![Complex::new(0.0_f32, 0.0_f32); n];
    let mut consec         = 0usize;
    let mut preamble_start = 0usize;

    let mut w = 0usize;
    while w + sps <= samples.len() {
        let bin   = symbol_bin(&mut buf, &samples[w..w + sps], os_factor as usize, &downchirp, fft.as_ref());
        let is_up = bin <= 1 || bin >= n - 1;

        if is_up {
            if consec == 0 { preamble_start = w; }
            consec += 1;
            if consec >= n_up_req {
                let payload_start = preamble_start + (preamble_len as usize + 4) * sps + sps / 4;
                if payload_start + sps <= samples.len() {
                    let len = ((samples.len() - payload_start) / sps) * sps;
                    return FrameSyncResult {
                        found:    true,
                        symbols:  samples[payload_start..payload_start + len].to_vec(),
                        consumed: payload_start + len,
                    };
                }
                // Preamble found but not enough payload yet — keep it.
                return FrameSyncResult { found: false, symbols: vec![], consumed: preamble_start };
            }
        } else {
            consec = 0;
        }
        w += sps;
    }

    // Partial preamble at the tail → keep from preamble_start onward.
    let consumed = if consec > 0 { preamble_start } else { w };
    FrameSyncResult { found: false, symbols: vec![], consumed }
}
