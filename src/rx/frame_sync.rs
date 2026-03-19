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

/// Dechirp + FFT one symbol window.  Returns full magnitude-squared spectrum.
fn symbol_spectrum(
    buf:       &mut Vec<Complex<f32>>,
    samples:   &[Complex<f32>],
    os_factor: usize,
    downchirp: &[Complex<f32>],
    fft:       &dyn rustfft::Fft<f32>,
) {
    for (i, b) in buf.iter_mut().enumerate() {
        *b = samples[os_factor * i];
    }
    for (b, d) in buf.iter_mut().zip(downchirp.iter()) {
        *b = *b * d;
    }
    fft.process(buf);
}

/// Peak bin from a dechirped FFT buffer.
fn peak_bin(buf: &[Complex<f32>]) -> usize {
    buf.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Dechirp + FFT → peak bin (convenience wrapper).
fn symbol_bin(
    buf:       &mut Vec<Complex<f32>>,
    samples:   &[Complex<f32>],
    os_factor: usize,
    downchirp: &[Complex<f32>],
    fft:       &dyn rustfft::Fft<f32>,
) -> usize {
    symbol_spectrum(buf, samples, os_factor, downchirp, fft);
    peak_bin(buf)
}

/// Are two FFT bins within ±`tol` (mod n)?
fn bins_close_tol(a: usize, b: usize, n: usize, tol: usize) -> bool {
    let diff = if a >= b { a - b } else { b - a };
    let diff = diff.min(n - diff);
    diff <= tol
}

/// Detect a LoRa preamble in `samples` and return the raw payload IQ windows.
///
/// Uses **non-coherent accumulation** for weak-signal detection:
/// instead of requiring consecutive per-window bin matches, we accumulate
/// the dechirped FFT magnitude-squared spectra over a sliding window of
/// `preamble_len` symbols.  The preamble signal adds coherently at one bin
/// while noise averages out, giving ~10·log10(preamble_len) dB processing gain.
///
/// When the accumulated peak exceeds a threshold relative to the mean,
/// we declare a preamble detection and proceed to sync word validation.
pub fn frame_sync(
    samples:      &[Complex<f32>],
    sf:           u8,
    sync_word:    u8,
    preamble_len: u16,
    os_factor:    u32,
) -> FrameSyncResult {
    let n         = 1usize << sf;
    let sps       = n * os_factor as usize;
    let pre_len   = preamble_len as usize;
    let downchirp = make_downchirp(sf);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buf = vec![Complex::new(0.0_f32, 0.0_f32); n];

    // Sync word tolerance: ±3 bins.
    let sw_tol: usize = 3;

    // Detection threshold: accumulated peak must be this many times above mean.
    // Detection thresholds (peak/mean ratio).
    // Strong: per-window alignment works → full decode pipeline.
    // Weak: trust accumulation, skip sync word, rely on CRC.
    let strong_threshold: f64 = 4.0;  // trigger accumulation check
    let weak_decode_threshold: f64 = 10.0;  // attempt payload decode without sync word

    let not_found = |consumed| FrameSyncResult {
        found: false, symbols: vec![], consumed, freq_offset_bins: 0.0,
    };

    let n_windows = samples.len().saturating_sub(sps - 1) / sps;
    if n_windows < pre_len + 5 {
        return not_found(0);
    }

    // ── Phase 1: Non-coherent preamble search ────────────────────────────
    // Accumulate magnitude-squared spectra over a sliding window.
    // acc[bin] = sum of |FFT[bin]|² over `pre_len` consecutive windows.
    let mut acc = vec![0.0_f64; n];

    // Fill initial accumulator with first `pre_len` windows.
    // Also keep per-window spectra for subtraction when sliding.
    let mut ring: Vec<Vec<f64>> = Vec::with_capacity(pre_len);
    for i in 0..pre_len {
        let start = i * sps;
        symbol_spectrum(&mut buf, &samples[start..start + sps], os_factor as usize, &downchirp, fft.as_ref());
        let mag2: Vec<f64> = buf.iter().map(|c| c.norm_sqr() as f64).collect();
        for (a, m) in acc.iter_mut().zip(mag2.iter()) {
            *a += m;
        }
        ring.push(mag2);
    }

    let mut ring_idx = 0usize; // oldest entry in ring buffer
    let mut best_w = 0usize;   // window index where we found preamble

    // Slide one window at a time, checking for preamble.
    // w = index of the first window in the current accumulation.
    let mut w_start = 0usize;
    loop {
        // Check if current accumulation has a dominant peak.
        let total: f64 = acc.iter().sum();
        let mean = total / n as f64;

        let (peak_idx, &peak_val) = acc.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let ratio = if mean > 0.0 { peak_val / mean } else { 0.0 };
        if ratio > strong_threshold {
            // Preamble candidate found!
            best_w = w_start;
            let preamble_bin = peak_idx;
            let bin_tol = 3usize; // per-window tolerance for strong signal matching

            // ── Phase 2: Per-window scan to find actual preamble boundaries ──
            // The accumulation window [w_start .. w_start+pre_len) may only
            // partially overlap the preamble.  Scan a wider region to find
            // the longest consecutive run of windows matching preamble_bin.
            let scan_start = w_start.saturating_sub(pre_len);
            let scan_end   = (w_start + 2 * pre_len + 4).min(n_windows);

            let mut best_run_start = 0usize;
            let mut best_run_len   = 0usize;
            let mut cur_run_start  = scan_start;
            let mut cur_run_len    = 0usize;

            for wi in scan_start..scan_end {
                let widx = wi * sps;
                if widx + sps > samples.len() { break; }
                let bin = symbol_bin(&mut buf, &samples[widx..widx + sps],
                    os_factor as usize, &downchirp, fft.as_ref());
                if bins_close_tol(bin, preamble_bin, n, bin_tol) {
                    if cur_run_len == 0 { cur_run_start = wi; }
                    cur_run_len += 1;
                    if cur_run_len > best_run_len {
                        best_run_len   = cur_run_len;
                        best_run_start = cur_run_start;
                    }
                } else {
                    cur_run_len = 0;
                }
            }

            // Use per-window run if strong enough, otherwise trust accumulation directly.
            let (mean_bin, good_count, end_w) = if best_run_len >= 4 {
                // Strong signal: use the precise per-window consecutive run.
                let mut sin_sum = 0.0_f64;
                let mut cos_sum = 0.0_f64;
                for wi in best_run_start..best_run_start + best_run_len {
                    let widx = wi * sps;
                    let bin = symbol_bin(&mut buf, &samples[widx..widx + sps],
                        os_factor as usize, &downchirp, fft.as_ref());
                    let angle = TAU * bin as f64 / n as f64;
                    sin_sum += angle.sin();
                    cos_sum += angle.cos();
                }
                let mean_angle = sin_sum.atan2(cos_sum);
                let mb = ((mean_angle / TAU * n as f64).round() as isize)
                    .rem_euclid(n as isize) as usize;
                (mb, best_run_len, (best_run_start + best_run_len) * sps)
            } else {
                // Weak signal: trust the accumulation bin directly.
                // The accumulation window ends at w_start + pre_len.
                (preamble_bin, 0, (w_start + pre_len) * sps)
            };

            {
                let d = ((n - mean_bin) % n) * os_factor as usize;

                let sync_start    = if 2 * d < sps { end_w + d } else { end_w + d - sps };
                let payload_start = sync_start + 4 * sps + sps / 4;

                // ── Phase 3: Sync word validation ────────────────────────
                if sync_start + 2 * sps <= samples.len() {
                    let exp_sw0 = ((sync_word as usize & 0xF0) >> 4) << 3;
                    let exp_sw1 =  (sync_word as usize & 0x0F)       << 3;
                    let det_sw0 = symbol_bin(&mut buf,
                        &samples[sync_start..sync_start + sps],
                        os_factor as usize, &downchirp, fft.as_ref());
                    let det_sw1 = symbol_bin(&mut buf,
                        &samples[sync_start + sps..sync_start + 2 * sps],
                        os_factor as usize, &downchirp, fft.as_ref());

                    let sw_ok = bins_close_tol(det_sw0, exp_sw0, n, sw_tol)
                             && bins_close_tol(det_sw1, exp_sw1, n, sw_tol);

                    if good_count > 0 {
                        // Strong signal: per-window alignment is reliable,
                        // so sync word validation is meaningful.
                        eprintln!("[sync] preamble: bin={preamble_bin} run={good_count} \
                                   peak/mean={:.1}, sw0: det={det_sw0} exp={exp_sw0}, \
                                   sw1: det={det_sw1} exp={exp_sw1}",
                                   peak_val / mean);
                        if !sw_ok {
                            eprintln!("[sync] sync word REJECTED (tol=±{sw_tol})");
                            let skip_to = w_start + pre_len + 1;
                            if skip_to >= n_windows { break; }
                            while w_start < skip_to {
                                let new_w = w_start + pre_len;
                                if new_w >= n_windows { return not_found(w_start * sps); }
                                let old = &ring[ring_idx];
                                for (a, o) in acc.iter_mut().zip(old.iter()) { *a -= o; }
                                let new_start = new_w * sps;
                                symbol_spectrum(&mut buf, &samples[new_start..new_start + sps],
                                    os_factor as usize, &downchirp, fft.as_ref());
                                let mag2: Vec<f64> = buf.iter().map(|c| c.norm_sqr() as f64).collect();
                                for (a, m) in acc.iter_mut().zip(mag2.iter()) { *a += m; }
                                ring[ring_idx] = mag2;
                                ring_idx = (ring_idx + 1) % pre_len;
                                w_start += 1;
                            }
                            continue;
                        }
                        eprintln!("[sync] sync word OK → decoding payload");
                    } else if ratio > weak_decode_threshold {
                        // Weak signal with strong accumulation: skip sync word,
                        // let payload CRC be the final gatekeeper.
                        eprintln!("[sync] weak preamble: bin={preamble_bin} \
                                   peak/mean={ratio:.1} → skipping sync word, trying payload decode");
                    } else {
                        // Accumulation triggered but too weak to trust.
                        // Skip past and continue searching.
                        eprintln!("[sync] marginal detection: bin={preamble_bin} \
                                   peak/mean={ratio:.1} — skipping");
                        let skip_to = w_start + pre_len + 1;
                        if skip_to >= n_windows { break; }
                        while w_start < skip_to {
                            let new_w = w_start + pre_len;
                            if new_w >= n_windows { return not_found(w_start * sps); }
                            let old = &ring[ring_idx];
                            for (a, o) in acc.iter_mut().zip(old.iter()) { *a -= o; }
                            let new_start = new_w * sps;
                            symbol_spectrum(&mut buf, &samples[new_start..new_start + sps],
                                os_factor as usize, &downchirp, fft.as_ref());
                            let mag2: Vec<f64> = buf.iter().map(|c| c.norm_sqr() as f64).collect();
                            for (a, m) in acc.iter_mut().zip(mag2.iter()) { *a += m; }
                            ring[ring_idx] = mag2;
                            ring_idx = (ring_idx + 1) % pre_len;
                            w_start += 1;
                        }
                        continue;
                    }
                }

                let offset_bins = 0.0_f64;
                if payload_start + sps <= samples.len() {
                    let len = ((samples.len() - payload_start) / sps) * sps;

                    // For weak detections, limit consumed to just past the
                    // accumulation window.  The alignment is uncertain, so
                    // consuming the full estimated payload would eat into
                    // subsequent strong packets in the buffer.
                    let consumed = if good_count > 0 {
                        payload_start + len // strong: alignment is precise
                    } else {
                        // weak: only consume up to end of accumulation window
                        // plus a small margin for the sync+SFD region.
                        (end_w + 5 * sps).min(payload_start + len)
                    };

                    return FrameSyncResult {
                        found:    true,
                        symbols:  samples[payload_start..payload_start + len].to_vec(),
                        consumed,
                        freq_offset_bins: offset_bins,
                    };
                }
                // Preamble found but not enough payload yet.
                return FrameSyncResult {
                    found: false, symbols: vec![],
                    consumed: best_w * sps,
                    freq_offset_bins: offset_bins,
                };
            }
        }

        // ── Slide window forward by one symbol ───────────────────────────
        let new_w = w_start + pre_len;
        if new_w >= n_windows { break; }

        // Subtract oldest window, add new window.
        let old = &ring[ring_idx];
        for (a, o) in acc.iter_mut().zip(old.iter()) { *a -= o; }

        let new_start = new_w * sps;
        symbol_spectrum(&mut buf, &samples[new_start..new_start + sps],
            os_factor as usize, &downchirp, fft.as_ref());
        let mag2: Vec<f64> = buf.iter().map(|c| c.norm_sqr() as f64).collect();
        for (a, m) in acc.iter_mut().zip(mag2.iter()) { *a += m; }
        ring[ring_idx] = mag2;
        ring_idx = (ring_idx + 1) % pre_len;

        w_start += 1;
    }

    not_found(if n_windows > 0 { (n_windows - pre_len).max(1) * sps } else { 0 })
}
