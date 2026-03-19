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
    /// Estimated frequency offset in bins (fractional).  Positive means the
    /// transmitter is higher than our centre frequency.  Only valid when
    /// `found == true`.
    pub freq_offset_bins: f64,
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
/// Returns (peak_bin, peak_magnitude_squared).
fn symbol_bin_mag(
    buf:       &mut Vec<Complex<f32>>,
    samples:   &[Complex<f32>],  // sps samples
    os_factor: usize,
    downchirp: &[Complex<f32>],
    fft:       &dyn rustfft::Fft<f32>,
) -> (usize, f32) {
    for (i, b) in buf.iter_mut().enumerate() {
        *b = samples[os_factor * i];
    }
    for (b, d) in buf.iter_mut().zip(downchirp.iter()) {
        *b = *b * d;
    }
    fft.process(buf);
    buf.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .map(|(i, c)| (i, c.norm_sqr()))
        .unwrap_or((0, 0.0))
}

fn symbol_bin(
    buf: &mut Vec<Complex<f32>>,
    samples: &[Complex<f32>],
    os_factor: usize,
    downchirp: &[Complex<f32>],
    fft: &dyn rustfft::Fft<f32>,
) -> usize {
    symbol_bin_mag(buf, samples, os_factor, downchirp, fft).0
}

/// Are two FFT bins within ±`tol` (mod n)?
fn bins_close_tol(a: usize, b: usize, n: usize, tol: usize) -> bool {
    let diff = if a >= b { a - b } else { b - a };
    let diff = diff.min(n - diff); // wrap-around distance
    diff <= tol
}

/// Are two FFT bins within ±1 (mod n)?
fn bins_close(a: usize, b: usize, n: usize) -> bool {
    bins_close_tol(a, b, n, 1)
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
    // Require fewer consecutive preamble matches than the full preamble length
    // to tolerate weak signals.  Minimum 4 to avoid false positives.
    let n_up_req  = (preamble_len as usize).saturating_sub(5).max(4);
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

    // Preamble tolerance: first 2 windows must match exactly, then ±2 bins
    // to handle noise and minor frequency drift within a single packet.
    let pre_tol: usize = 2;
    // Sync word tolerance: ±3 bins to handle frequency offset between nodes.
    let sw_tol: usize  = 3;

    let not_found = |consumed| FrameSyncResult {
        found: false, symbols: vec![], consumed, freq_offset_bins: 0.0,
    };

    let mut w = 0usize;
    'search: while w + sps <= samples.len() {
        let bin = symbol_bin(&mut buf, &samples[w..w + sps], os_factor as usize, &downchirp, fft.as_ref());

        if consec == 0 {
            preamble_start = w;
            preamble_bin   = bin;
            consec         = 1;
            let angle = TAU * bin as f64 / n as f64;
            sin_sum = angle.sin();
            cos_sum = angle.cos();
        } else if (consec == 1 && bin == preamble_bin)
               || (consec >= 2 && bins_close_tol(bin, preamble_bin, n, pre_tol)) {
            consec += 1;
            let angle = TAU * bin as f64 / n as f64;
            sin_sum += angle.sin();
            cos_sum += angle.cos();

            if consec >= n_up_req {
                // ── Preamble found — compute aligned payload start ────────
                let mean_angle = sin_sum.atan2(cos_sum);
                let mean_bin   = ((mean_angle / TAU * n as f64).round() as isize)
                    .rem_euclid(n as isize) as usize;

                let d = ((n - mean_bin) % n) * os_factor as usize;

                // Scan forward past remaining upchirps.
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

                let sync_start    = if 2 * d < sps { end_w + d } else { end_w + d - sps };
                let payload_start = sync_start + 4 * sps + sps / 4;

                // Validate the two sync-word chirps with offset compensation.
                // The preamble bin tells us the frequency offset: an unshifted
                // preamble produces bin 0 (or N-d/os for alignment d).  The
                // preamble_bin we detected includes both alignment AND frequency
                // offset.  The sync word expected bins also include alignment,
                // so we add the preamble_bin as an offset.
                if sync_start + 2 * sps <= samples.len() {
                    // sync_start is aligned to the chirp boundary, so the
                    // dechirped bins give the raw sync word values directly
                    // (alignment offset already removed).  Use wider tolerance
                    // to handle residual frequency offset.
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
                    if !bins_close_tol(det_sw0, exp_sw0, n, sw_tol)
                    || !bins_close_tol(det_sw1, exp_sw1, n, sw_tol) {
                        consec  = 0;
                        sin_sum = 0.0;
                        cos_sum = 0.0;
                        w       = end_w;
                        continue 'search;
                    }
                }

                // ── Estimate frequency offset ────────────────────────────
                // Preamble upchirps at alignment d produce bin = (N - d/os) % N.
                // The *actual* detected mean_bin includes both alignment and
                // frequency offset.  Since d was derived from mean_bin via
                // d = (N - mean_bin) % N * os, the alignment portion cancels
                // and any residual is noise.
                //
                // For a more precise estimate, use the sub-bin circular mean
                // directly: the fractional part tells us the frequency offset.
                let mean_bin_frac = mean_angle / TAU * n as f64;
                let expected_bin  = ((n - d / os_factor as usize) % n) as f64;
                let offset_bins   = mean_bin_frac - expected_bin;
                // Wrap to [-N/2, N/2).
                let offset_bins = if offset_bins > n as f64 / 2.0 {
                    offset_bins - n as f64
                } else if offset_bins < -(n as f64 / 2.0) {
                    offset_bins + n as f64
                } else {
                    offset_bins
                };

                if payload_start + sps <= samples.len() {
                    let len = ((samples.len() - payload_start) / sps) * sps;

                    // Apply frequency correction: rotate each sample to
                    // compensate for the estimated offset.
                    let f_corr = -(offset_bins as f64) / (n as f64 * os_factor as f64);
                    let mut corrected = samples[payload_start..payload_start + len].to_vec();
                    for (i, s) in corrected.iter_mut().enumerate() {
                        let phase = TAU * f_corr * i as f64;
                        let (sin, cos) = phase.sin_cos();
                        *s = *s * Complex::new(cos as f32, sin as f32);
                    }

                    return FrameSyncResult {
                        found:    true,
                        symbols:  corrected,
                        consumed: payload_start + len,
                        freq_offset_bins: offset_bins,
                    };
                }
                return FrameSyncResult {
                    found: false, symbols: vec![], consumed: preamble_start,
                    freq_offset_bins: offset_bins,
                };
            }
        } else {
            preamble_start = w;
            preamble_bin   = bin;
            consec         = 1;
            let angle = TAU * bin as f64 / n as f64;
            sin_sum = angle.sin();
            cos_sum = angle.cos();
        }
        w += sps;
    }

    let consumed = if consec > 0 { preamble_start } else { w };
    not_found(consumed)
}
