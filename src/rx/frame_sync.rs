use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::TAU;

pub struct FrameSyncResult {
    pub found:   bool,
    pub symbols: Vec<Complex<f32>>,
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

fn symbol_bin(buf: &mut Vec<Complex<f32>>, samples: &[Complex<f32>], downchirp: &[Complex<f32>], fft: &dyn rustfft::Fft<f32>) -> usize {
    for (b, (s, d)) in buf.iter_mut().zip(samples.iter().zip(downchirp.iter())) {
        *b = s * d;
    }
    fft.process(buf);
    buf.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Detect a LoRa preamble in `samples` and return the payload IQ windows.
/// Scanning is N-sample aligned (os_factor=1, ideal signal assumed).
pub fn frame_sync(
    samples:      &[Complex<f32>],
    sf:           u8,
    _sync_word:   u8,
    preamble_len: u16,
) -> FrameSyncResult {
    let n         = 1usize << sf;
    let n_up_req  = preamble_len as usize - 3;
    let downchirp = make_downchirp(sf);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buf     = vec![Complex::new(0.0_f32, 0.0_f32); n];
    let mut consec  = 0usize;
    let mut preamble_start = 0usize;

    let mut w = 0usize;
    while w + n <= samples.len() {
        let bin    = symbol_bin(&mut buf, &samples[w..w + n], &downchirp, fft.as_ref());
        let is_up  = bin <= 1 || bin >= n - 1;

        if is_up {
            if consec == 0 { preamble_start = w; }
            consec += 1;
            if consec >= n_up_req {
                let payload_start = preamble_start + (preamble_len as usize + 4) * n + n / 4;
                if payload_start + n <= samples.len() {
                    let len = ((samples.len() - payload_start) / n) * n;
                    return FrameSyncResult {
                        found:   true,
                        symbols: samples[payload_start..payload_start + len].to_vec(),
                    };
                }
                break;
            }
        } else {
            consec = 0;
        }
        w += n;
    }

    FrameSyncResult { found: false, symbols: vec![] }
}
