use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::TAU;

pub struct FrameSyncResult {
    pub found:    bool,
    pub symbols:  Vec<Complex<f32>>,
    /// Number of input samples consumed. Safe to drain from a streaming buffer.
    pub consumed: usize,
    /// Estimated total frequency offset in bins (integer + fractional).
    pub freq_offset_bins: f64,
}

// ── Reference chirps ─────────────────────────────────────────────────────────

fn make_downchirp(sf: u8) -> Vec<Complex<f32>> {
    let n = 1usize << sf;
    (0..n).map(|k| {
        let t     = k as f64;
        let phase = t * t / (2.0 * n as f64) - 0.5 * t;
        let (sin, cos) = (TAU * phase).sin_cos();
        Complex::new(cos as f32, -sin as f32)
    }).collect()
}

fn make_upchirp(sf: u8) -> Vec<Complex<f32>> {
    let n = 1usize << sf;
    (0..n).map(|k| {
        let t     = k as f64;
        let phase = t * t / (2.0 * n as f64) - 0.5 * t;
        let (sin, cos) = (TAU * phase).sin_cos();
        Complex::new(cos as f32, sin as f32)
    }).collect()
}

// ── DSP helpers ──────────────────────────────────────────────────────────────

/// Dechirp + FFT one symbol window.  Writes result into `buf`.
fn symbol_spectrum(
    buf:       &mut [Complex<f32>],
    samples:   &[Complex<f32>],
    os_factor: usize,
    ref_chirp: &[Complex<f32>],
    fft:       &dyn rustfft::Fft<f32>,
) {
    let n = buf.len();
    for (i, b) in buf.iter_mut().enumerate() {
        *b = samples[os_factor * i];
    }
    for (b, d) in buf.iter_mut().zip(ref_chirp.iter()) {
        *b = *b * d;
    }
    fft.process(buf);
    // Normalise so magnitudes don't scale with N.
    let inv = 1.0 / n as f32;
    for b in buf.iter_mut() { *b = *b * inv; }
}

/// Peak bin from FFT magnitude-squared.
fn peak_bin(buf: &[Complex<f32>]) -> usize {
    buf.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Dechirp + FFT → peak bin.
fn symbol_bin(
    buf: &mut [Complex<f32>], samples: &[Complex<f32>],
    os_factor: usize, ref_chirp: &[Complex<f32>], fft: &dyn rustfft::Fft<f32>,
) -> usize {
    symbol_spectrum(buf, samples, os_factor, ref_chirp, fft);
    peak_bin(buf)
}

/// Are two FFT bins within ±`tol` (mod n)?
fn bins_close_tol(a: usize, b: usize, n: usize, tol: usize) -> bool {
    let diff = if a >= b { a - b } else { b - a };
    diff.min(n - diff) <= tol
}

/// RCTSL sub-bin interpolation (Cui Yang, eq. 15).
/// Given the three magnitude-squared values around the peak (Y_{-1}, Y_0, Y_1)
/// and N (FFT size), returns the fractional bin offset in [-0.5, 0.5).
fn rctsl_frac(y_m1: f64, y_0: f64, y_p1: f64, n: usize) -> f64 {
    let u  = 64.0 * n as f64 / 406.5506497;
    let v  = u * 2.4674;
    let wa = (y_p1 - y_m1) / (u * (y_p1 + y_m1) + v * y_0);
    wa * n as f64 / std::f64::consts::PI
}

// ── Main entry point ─────────────────────────────────────────────────────────

/// Detect a LoRa preamble and return payload symbols with CFO/STO correction.
///
/// Algorithm (inspired by gr-lora_sdr / Tapparel et al.):
/// 1. Non-coherent accumulation over `preamble_len` windows for detection
/// 2. Per-window scan to find precise preamble boundaries (strong signals)
/// 3. Multi-symbol CFO estimation with 2N-point FFT + RCTSL interpolation
/// 4. STO estimation with accumulated 2N-point FFT + RCTSL
/// 5. Sync word validation (strong signals) or CRC gatekeeper (weak)
/// 6. CFO-corrected + STO-aligned payload extraction
pub fn frame_sync(
    samples:        &[Complex<f32>],
    sf:             u8,
    sync_word:      u8,
    preamble_len:   u16,
    os_factor:      u32,
    center_freq_hz: f64,
    bw_hz:          f64,
) -> FrameSyncResult {
    let n         = 1usize << sf;
    let sps       = n * os_factor as usize;
    let pre_len   = preamble_len as usize;
    let downchirp = make_downchirp(sf);
    let upchirp   = make_upchirp(sf);

    let mut planner = FftPlanner::<f32>::new();
    let fft_n  = planner.plan_fft_forward(n);

    let mut buf = vec![Complex::new(0.0_f32, 0.0); n];

    let sw_tol: usize             = 3;
    let strong_threshold: f64     = 4.0;
    let weak_decode_threshold: f64 = 6.0;

    let not_found = |consumed| FrameSyncResult {
        found: false, symbols: vec![], consumed, freq_offset_bins: 0.0,
    };

    let n_windows = samples.len().saturating_sub(sps - 1) / sps;
    if n_windows < pre_len + 5 { return not_found(0); }

    // ── Phase 1: Non-coherent preamble accumulation ──────────────────────
    let mut acc = vec![0.0_f64; n];
    let mut ring: Vec<Vec<f64>> = Vec::with_capacity(pre_len);
    for i in 0..pre_len {
        let start = i * sps;
        symbol_spectrum(&mut buf, &samples[start..start + sps],
            os_factor as usize, &downchirp, fft_n.as_ref());
        let mag2: Vec<f64> = buf.iter().map(|c| c.norm_sqr() as f64).collect();
        for (a, m) in acc.iter_mut().zip(mag2.iter()) { *a += m; }
        ring.push(mag2);
    }
    let mut ring_idx = 0usize;
    let mut w_start  = 0usize;

    loop {
        let total: f64 = acc.iter().sum();
        let mean = total / n as f64;
        let (peak_idx, &peak_val) = acc.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
        let ratio = if mean > 0.0 { peak_val / mean } else { 0.0 };

        if ratio > strong_threshold {
            let preamble_bin = peak_idx;
            let bin_tol = 4usize;

            // ── Phase 2: Find preamble boundaries ────────────────────────
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
                    os_factor as usize, &downchirp, fft_n.as_ref());
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

            let strong = best_run_len >= 3;
            // Estimate CFO/STO even for weaker signals if we found ≥2 consecutive matching bins.
            let up_symb_to_use = if best_run_len >= 2 { best_run_len.min(pre_len - 1) } else { 0 };

            // Determine preamble region in sample space.
            let (pre_sample_start, end_w) = if strong {
                (best_run_start * sps, (best_run_start + best_run_len) * sps)
            } else {
                (w_start * sps, (w_start + pre_len) * sps)
            };

            // ── Phase 3: CFO estimation (multi-symbol, 2N-point FFT) ─────
            let cfo_frac = if up_symb_to_use >= 2 {
                // Concatenate `up_symb_to_use` dechirped symbols, zero-pad to 2×, FFT.
                let cat_len = up_symb_to_use * n;
                let fft_len = 2 * cat_len;
                let fft_2n  = planner.plan_fft_forward(fft_len);
                let mut cfo_buf = vec![Complex::new(0.0_f32, 0.0); fft_len];

                // Dechirp each preamble symbol (downsampled) and concatenate.
                for s in 0..up_symb_to_use {
                    let sym_start = pre_sample_start + s * sps;
                    for i in 0..n {
                        let sample = samples[sym_start + i * os_factor as usize];
                        cfo_buf[s * n + i] = sample * downchirp[i];
                    }
                }
                // Zero-padding already in place (vec initialized to 0).

                fft_2n.process(&mut cfo_buf);

                // Find peak and apply RCTSL.
                let mag2: Vec<f64> = cfo_buf.iter().map(|c| c.norm_sqr() as f64).collect();
                let k0 = mag2.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);

                let y_m1 = mag2[(k0 + fft_len - 1) % fft_len];
                let y_0  = mag2[k0];
                let y_p1 = mag2[(k0 + 1) % fft_len];

                let ka = rctsl_frac(y_m1, y_0, y_p1, n);
                let k_residual = ((k0 as f64 + ka) / 2.0 / up_symb_to_use as f64).rem_euclid(1.0);
                let cfo = if k_residual > 0.5 { k_residual - 1.0 } else { k_residual };

                eprintln!("[sync] CFO estimate: {cfo:.4} bins ({:.1} Hz @ BW/N)",
                    cfo * 250000.0 / n as f64);
                cfo
            } else {
                0.0
            };

            // ── Phase 4: STO estimation (accumulated 2N-point FFT) ───────
            let sto_frac = if up_symb_to_use >= 2 {
                // Accumulate magnitude-squared of 2N-point FFT across symbols.
                let fft_len = 2 * n;
                let fft_2n  = planner.plan_fft_forward(fft_len);
                let mut sto_acc = vec![0.0_f64; fft_len];
                let mut sto_buf = vec![Complex::new(0.0_f32, 0.0); fft_len];

                for s in 0..up_symb_to_use {
                    let sym_start = pre_sample_start + s * sps;
                    // Dechirp with CFO correction.
                    for i in 0..n {
                        let global_i = (s * n + i) as f64;
                        let cfo_phase = -TAU * cfo_frac / n as f64 * global_i;
                        let (sin, cos) = cfo_phase.sin_cos();
                        let cfo_rot = Complex::new(cos as f32, sin as f32);
                        let corrected = samples[sym_start + i * os_factor as usize] * cfo_rot;
                        sto_buf[i] = corrected * downchirp[i];
                    }
                    // Zero-pad.
                    for b in &mut sto_buf[n..] { *b = Complex::new(0.0, 0.0); }

                    fft_2n.process(&mut sto_buf);
                    for (a, b) in sto_acc.iter_mut().zip(sto_buf.iter()) {
                        *a += b.norm_sqr() as f64;
                    }
                    // Reset buf for next iteration.
                    for b in &mut sto_buf { *b = Complex::new(0.0, 0.0); }
                }

                let k0 = sto_acc.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);

                let y_m1 = sto_acc[(k0 + fft_len - 1) % fft_len];
                let y_0  = sto_acc[k0];
                let y_p1 = sto_acc[(k0 + 1) % fft_len];

                let ka = rctsl_frac(y_m1, y_0, y_p1, n);
                let k_residual = ((k0 as f64 + ka) / 2.0).rem_euclid(1.0);
                let sto = if k_residual > 0.5 { k_residual - 1.0 } else { k_residual };

                eprintln!("[sync] STO estimate: {sto:.4} (fractional sample offset)");
                sto
            } else {
                0.0
            };

            // ── Phase 5: Compute alignment and sync word position ────────
            // For strong signals, use circular mean of per-window bins.
            // For weak signals, use the accumulation bin directly.
            let mean_bin = if strong {
                let mut sin_sum = 0.0_f64;
                let mut cos_sum = 0.0_f64;
                for wi in best_run_start..best_run_start + best_run_len {
                    let widx = wi * sps;
                    let bin = symbol_bin(&mut buf, &samples[widx..widx + sps],
                        os_factor as usize, &downchirp, fft_n.as_ref());
                    let angle = TAU * bin as f64 / n as f64;
                    sin_sum += angle.sin();
                    cos_sum += angle.cos();
                }
                let mean_angle = sin_sum.atan2(cos_sum);
                ((mean_angle / TAU * n as f64).round() as isize).rem_euclid(n as isize) as usize
            } else {
                preamble_bin
            };

            let d = ((n - mean_bin) % n) * os_factor as usize;
            let sync_start    = if 2 * d < sps { end_w + d } else { end_w + d - sps };
            let payload_start = sync_start + 4 * sps + sps / 4;

            // ── Phase 6: Sync word validation / threshold gating ────────
            // Gate weak signals before expensive CFO/SFO extraction.
            if sync_start + 2 * sps <= samples.len() {
                let exp_sw0 = ((sync_word as usize & 0xF0) >> 4) << 3;
                let exp_sw1 =  (sync_word as usize & 0x0F)       << 3;
                let det_sw0 = symbol_bin(&mut buf,
                    &samples[sync_start..sync_start + sps],
                    os_factor as usize, &downchirp, fft_n.as_ref());
                let det_sw1 = symbol_bin(&mut buf,
                    &samples[sync_start + sps..sync_start + 2 * sps],
                    os_factor as usize, &downchirp, fft_n.as_ref());

                let sw_ok = bins_close_tol(det_sw0, exp_sw0, n, sw_tol)
                         && bins_close_tol(det_sw1, exp_sw1, n, sw_tol);

                if strong {
                    eprintln!("[sync] preamble: bin={preamble_bin} run={best_run_len} \
                               peak/mean={ratio:.1}, sw: [{det_sw0},{det_sw1}] exp=[{exp_sw0},{exp_sw1}]");
                    if !sw_ok {
                        eprintln!("[sync] sync word REJECTED (tol=±{sw_tol})");
                        let skip_to = w_start + pre_len + 1;
                        if skip_to >= n_windows { break; }
                        while w_start < skip_to {
                            let new_w = w_start + pre_len;
                            if new_w >= n_windows { return not_found(w_start * sps); }
                            let old = &ring[ring_idx];
                            for (a, o) in acc.iter_mut().zip(old.iter()) { *a -= o; }
                            let s = new_w * sps;
                            symbol_spectrum(&mut buf, &samples[s..s + sps],
                                os_factor as usize, &downchirp, fft_n.as_ref());
                            let mag2: Vec<f64> = buf.iter().map(|c| c.norm_sqr() as f64).collect();
                            for (a, m) in acc.iter_mut().zip(mag2.iter()) { *a += m; }
                            ring[ring_idx] = mag2;
                            ring_idx = (ring_idx + 1) % pre_len;
                            w_start += 1;
                        }
                        continue;
                    }
                } else if ratio > weak_decode_threshold {
                    eprintln!("[sync] weak preamble: bin={preamble_bin} run={best_run_len} peak/mean={ratio:.1}");
                } else {
                    eprintln!("[sync] marginal: bin={preamble_bin} run={best_run_len} peak/mean={ratio:.1} — skip");
                    let skip_to = w_start + pre_len + 1;
                    if skip_to >= n_windows { break; }
                    while w_start < skip_to {
                        let new_w = w_start + pre_len;
                        if new_w >= n_windows { return not_found(w_start * sps); }
                        let old = &ring[ring_idx];
                        for (a, o) in acc.iter_mut().zip(old.iter()) { *a -= o; }
                        let s = new_w * sps;
                        symbol_spectrum(&mut buf, &samples[s..s + sps],
                            os_factor as usize, &downchirp, fft_n.as_ref());
                        let mag2: Vec<f64> = buf.iter().map(|c| c.norm_sqr() as f64).collect();
                        for (a, m) in acc.iter_mut().zip(mag2.iter()) { *a += m; }
                        ring[ring_idx] = mag2;
                        ring_idx = (ring_idx + 1) % pre_len;
                        w_start += 1;
                    }
                    continue;
                }
            }

            // ── Phase 6b: Integer CFO from downchirp symbols ─────────────
            // Only computed after passing sync word / threshold gate above.
            // The two downchirps sit at sync_start + 2*sps .. +4*sps.
            // Dechirping a downchirp with the upchirp reference yields
            // down_val ≈ 2 × cfo_int  (Tapparel et al.).
            let dc_start = sync_start + 2 * sps;
            let cfo_int = if dc_start + sps <= samples.len() {
                let down_val = symbol_bin(
                    &mut buf,
                    &samples[dc_start..dc_start + sps],
                    os_factor as usize,
                    &upchirp,
                    fft_n.as_ref(),
                );
                if (down_val) < n / 2 {
                    (down_val as f64 / 2.0).floor()
                } else {
                    ((down_val as f64 - n as f64) / 2.0).floor()
                }
            } else {
                0.0
            };

            let total_cfo = cfo_int + cfo_frac;

            // ── Phase 6c: SFO estimate from total CFO ────────────────────
            // SFO = (total_cfo × bw) / center_freq  (clock offset relationship).
            let sfo_hat = if center_freq_hz > 0.0 {
                total_cfo * bw_hz / center_freq_hz
            } else {
                0.0
            };

            eprintln!("[sync] → decoding: run={best_run_len} cfo_int={cfo_int:.0} cfo_frac={cfo_frac:.4} \
                       total={total_cfo:.4} sfo={sfo_hat:.6}");

            // ── Phase 7: Extract payload with CFO + SFO + STO correction ─
            if payload_start + sps <= samples.len() {
                let n_payload_samps = samples.len() - payload_start;
                let n_payload_syms  = n_payload_samps / sps;
                let len = n_payload_syms * sps;

                // Evolve STO from preamble start to payload start:
                //   preamble_len upchirps + 2 sync word + 2.25 downchirps
                let sto_evolved = sto_frac + sfo_hat * (pre_len as f64 + 4.25);
                let sto_offset  = (sto_evolved * os_factor as f64).round() as isize;

                let mut corrected = Vec::with_capacity(len);
                let mut sfo_cum: f64 = 0.0;
                let mut sample_adjust: isize = 0;
                let os = os_factor as f64;

                for sym in 0..n_payload_syms {
                    for i in 0..sps {
                        let global_i = sym * sps + i;
                        // Source index: payload base + position + STO + cumulative SFO drift.
                        let src_idx = payload_start as isize
                            + global_i as isize
                            + sto_offset
                            + sample_adjust;
                        let src_idx = src_idx.clamp(0, samples.len() as isize - 1) as usize;

                        // Phase rotation by total CFO (integer + fractional).
                        let phase = -TAU * total_cfo / n as f64 * global_i as f64;
                        let (sin, cos) = phase.sin_cos();
                        let rot = Complex::new(cos as f32, sin as f32);
                        corrected.push(samples[src_idx] * rot);
                    }

                    // SFO tracking: accumulate drift per symbol, adjust when
                    // it exceeds half an oversampled sample period.
                    sfo_cum += sfo_hat;
                    if sfo_cum.abs() > 0.5 / os {
                        let adj = if sfo_cum > 0.0 { 1_isize } else { -1 };
                        sample_adjust += adj;
                        sfo_cum -= adj as f64 / os;
                    }
                }

                let consumed = if strong {
                    payload_start + len
                } else {
                    (end_w + 5 * sps).min(payload_start + len)
                };

                return FrameSyncResult {
                    found: true,
                    symbols: corrected,
                    consumed,
                    freq_offset_bins: total_cfo,
                };
            }

            // Preamble found but not enough payload yet.
            return FrameSyncResult {
                found: false, symbols: vec![],
                consumed: if strong { best_run_start * sps } else { w_start * sps },
                freq_offset_bins: total_cfo,
            };
        }

        // ── Slide window ─────────────────────────────────────────────────
        let new_w = w_start + pre_len;
        if new_w >= n_windows { break; }

        let old = &ring[ring_idx];
        for (a, o) in acc.iter_mut().zip(old.iter()) { *a -= o; }
        let s = new_w * sps;
        symbol_spectrum(&mut buf, &samples[s..s + sps],
            os_factor as usize, &downchirp, fft_n.as_ref());
        let mag2: Vec<f64> = buf.iter().map(|c| c.norm_sqr() as f64).collect();
        for (a, m) in acc.iter_mut().zip(mag2.iter()) { *a += m; }
        ring[ring_idx] = mag2;
        ring_idx = (ring_idx + 1) % pre_len;

        w_start += 1;
    }

    not_found(if n_windows > 0 { (n_windows.saturating_sub(pre_len)).max(1) * sps } else { 0 })
}
