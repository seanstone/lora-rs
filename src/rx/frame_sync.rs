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

/// Are two FFT bins within ±1 (mod n)?
fn bins_close(a: usize, b: usize, n: usize) -> bool {
    a == b || (a + 1) % n == b || (b + 1) % n == a
}

/// Detect a LoRa preamble in `samples` and return the raw payload IQ windows.
///
/// Works on **arbitrarily aligned** streams: the preamble's upchirps are
/// periodic, so every sps-strided window within the preamble produces the
/// *same* FFT bin — the bin value encodes the fractional chirp offset.
/// We detect `n_up_req` consecutive windows with a consistent bin and use
/// the circular mean of those bins to compute the precise alignment.
pub fn frame_sync(
    samples:      &[Complex<f32>],
    sf:           u8,
    sync_word:    u8,
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
    let mut preamble_bin   = 0usize;

    // Circular accumulator for sub-bin precision via circular mean.
    let mut sin_sum = 0.0_f64;
    let mut cos_sum = 0.0_f64;

    let mut w = 0usize;
    'search: while w + sps <= samples.len() {
        let bin = symbol_bin(&mut buf, &samples[w..w + sps], os_factor as usize, &downchirp, fft.as_ref());

        if consec == 0 {
            // Start a new candidate preamble run.
            preamble_start = w;
            preamble_bin   = bin;
            consec         = 1;
            let angle = TAU * bin as f64 / n as f64;
            sin_sum = angle.sin();
            cos_sum = angle.cos();
        } else if (consec == 1 && bin == preamble_bin)
               || (consec >= 2 && bins_close(bin, preamble_bin, n)) {
            // First two windows must match exactly to avoid contaminated
            // partial-chirp windows anchoring preamble_start too early.
            // After two exact matches, allow ±1 for noise tolerance.
            consec += 1;
            let angle = TAU * bin as f64 / n as f64;
            sin_sum += angle.sin();
            cos_sum += angle.cos();

            if consec >= n_up_req {
                // ── Preamble found — compute aligned payload start ────────
                // Circular mean of the detected bins for sub-bin precision.
                let mean_angle = sin_sum.atan2(cos_sum);
                let mean_bin   = ((mean_angle / TAU * n as f64).round() as isize)
                    .rem_euclid(n as isize) as usize;

                // A dechirped upchirp at fractional offset d produces FFT
                // bin = (N − d/os_factor) mod N.  Invert to recover d.
                let d = ((n - mean_bin) % n) * os_factor as usize;

                // Scan forward past remaining upchirps to find the end
                // of the preamble (first non-upchirp window = sync word).
                // This is robust regardless of which chirp started the run.
                let mut end_w = w + sps;
                while end_w + sps <= samples.len() {
                    let next_bin = symbol_bin(
                        &mut buf,
                        &samples[end_w..end_w + sps],
                        os_factor as usize, &downchirp, fft.as_ref(),
                    );
                    if bins_close(next_bin, preamble_bin, n) {
                        end_w += sps;
                    } else {
                        break;
                    }
                }
                // eprintln!("[sync] pre_start={preamble_start} w={w} end_w={end_w} upchirps={} d={d} buf={}",
                //           (end_w - preamble_start) / sps, samples.len());
                // end_w = first non-upchirp window in the sps-stride scan.
                //
                // When d < sps/2: the sync word begins inside window [end_w..end_w+sps]
                //   at offset d  →  sync_start = end_w + d.
                // When d ≥ sps/2: the previous window [end_w-sps..end_w] still looked
                //   like an upchirp (upchirp portion d > sps/2 dominated), so the sync
                //   word actually started one window earlier at offset d within it
                //   →  sync_start = end_w + d - sps.
                //
                // Payload follows: 2 sync symbols + 2.25 SFD symbols = 4.25 sps.
                let sync_start    = if 2 * d < sps { end_w + d } else { end_w + d - sps };
                let payload_start = sync_start + 4 * sps + sps / 4;

                // Validate the two sync-word chirps when we have enough samples.
                // sw0 / sw1 are the upchirp IDs encoded from each nibble of the
                // sync word byte: sw0 = high_nibble << 3, sw1 = low_nibble << 3.
                if sync_start + 2 * sps <= samples.len() {
                    let exp_sw0 = ((sync_word as usize & 0xF0) >> 4) << 3;
                    let exp_sw1 =  (sync_word as usize & 0x0F)       << 3;
                    let det_sw0 = symbol_bin(
                        &mut buf, &samples[sync_start..sync_start + sps],
                        os_factor as usize, &downchirp, fft.as_ref(),
                    );
                    let det_sw1 = symbol_bin(
                        &mut buf, &samples[sync_start + sps..sync_start + 2 * sps],
                        os_factor as usize, &downchirp, fft.as_ref(),
                    );
                    if !bins_close(det_sw0, exp_sw0, n) || !bins_close(det_sw1, exp_sw1, n) {
                        // Sync word mismatch — different network.  Skip past
                        // the preamble end and restart the search.
                        consec  = 0;
                        sin_sum = 0.0;
                        cos_sum = 0.0;
                        w       = end_w;
                        continue 'search;
                    }
                }

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
            // Bin changed — restart the run from this window.
            preamble_start = w;
            preamble_bin   = bin;
            consec         = 1;
            let angle = TAU * bin as f64 / n as f64;
            sin_sum = angle.sin();
            cos_sum = angle.cos();
        }
        w += sps;
    }

    // Partial preamble at the tail → keep from preamble_start onward.
    let consumed = if consec > 0 { preamble_start } else { w };
    FrameSyncResult { found: false, symbols: vec![], consumed }
}
