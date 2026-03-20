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

fn symbol_spectrum(
    buf: &mut [Complex<f32>], samples: &[Complex<f32>], os_factor: usize,
    ref_chirp: &[Complex<f32>], fft: &dyn rustfft::Fft<f32>,
) {
    for (i, b) in buf.iter_mut().enumerate() {
        *b = samples[os_factor * i] * ref_chirp[i];
    }
    fft.process(buf);
    let inv = 1.0 / buf.len() as f32;
    for b in buf.iter_mut() { *b = *b * inv; }
}

fn symbol_bin(
    buf: &mut [Complex<f32>], samples: &[Complex<f32>], os_factor: usize,
    ref_chirp: &[Complex<f32>], fft: &dyn rustfft::Fft<f32>,
) -> usize {
    symbol_spectrum(buf, samples, os_factor, ref_chirp, fft);
    buf.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .map(|(i, _)| i).unwrap_or(0)
}

fn bins_close(a: usize, b: usize, n: usize, tol: usize) -> bool {
    let diff = if a >= b { a - b } else { b - a };
    diff.min(n - diff) <= tol
}

fn most_frequent(vals: &[usize]) -> usize {
    let mut best = 0;
    let mut best_count = 0;
    for &v in vals {
        let count = vals.iter().filter(|&&x| x == v).count();
        if count > best_count { best_count = count; best = v; }
    }
    best
}

fn rctsl_frac(y_m1: f64, y_0: f64, y_p1: f64, n: usize) -> f64 {
    let u  = 64.0 * n as f64 / 406.5506497;
    let v  = u * 2.4674;
    let wa = (y_p1 - y_m1) / (u * (y_p1 + y_m1) + v * y_0);
    wa * n as f64 / std::f64::consts::PI
}

// ── CFO/STO estimators ──────────────────────────────────────────────────────

/// Bernier CFO_frac estimator: phase slope between consecutive symbols.
fn estimate_cfo_frac_bernier(
    samples: &[Complex<f32>], aligned_start: usize, n_symbols: usize,
    sps: usize, n: usize, os_factor: usize,
    downchirp: &[Complex<f32>], planner: &mut FftPlanner<f32>,
) -> f64 {
    if n_symbols < 2 { return 0.0; }
    let fft = planner.plan_fft_forward(n);
    let mut fft_results: Vec<Vec<Complex<f32>>> = Vec::with_capacity(n_symbols);
    let mut peak_bins = Vec::with_capacity(n_symbols);
    let mut peak_mags = Vec::with_capacity(n_symbols);

    for s in 0..n_symbols {
        let sym_start = aligned_start + s * sps;
        if sym_start + (n - 1) * os_factor >= samples.len() { break; }
        let mut buf = vec![Complex::new(0.0_f32, 0.0); n];
        for i in 0..n { buf[i] = samples[sym_start + i * os_factor] * downchirp[i]; }
        fft.process(&mut buf);
        let (pk_bin, pk_mag) = buf.iter().enumerate()
            .map(|(i, c)| (i, c.norm_sqr() as f64))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap_or((0, 0.0));
        peak_bins.push(pk_bin);
        peak_mags.push(pk_mag);
        fft_results.push(buf);
    }
    if fft_results.len() < 2 { return 0.0; }

    let strongest = peak_mags.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
    let idx_max = peak_bins[strongest];

    let mut re_sum = 0.0_f64;
    let mut im_sum = 0.0_f64;
    for i in 0..fft_results.len() - 1 {
        let a = fft_results[i][idx_max];
        let b = fft_results[i + 1][idx_max];
        re_sum += (a.re * b.re + a.im * b.im) as f64;
        im_sum += (a.im * b.re - a.re * b.im) as f64;
    }
    -im_sum.atan2(re_sum) / TAU
}

/// STO_frac estimator: RCTSL on accumulated 2N-point FFT.
fn estimate_sto_frac(
    samples: &[Complex<f32>], aligned_start: usize, n_symbols: usize,
    sps: usize, n: usize, os_factor: usize, cfo_frac: f64,
    downchirp: &[Complex<f32>], planner: &mut FftPlanner<f32>,
) -> f64 {
    if n_symbols < 2 { return 0.0; }
    let fft_len = 2 * n;
    let fft_2n = planner.plan_fft_forward(fft_len);
    let mut sto_acc = vec![0.0_f64; fft_len];
    let mut sto_buf = vec![Complex::new(0.0_f32, 0.0); fft_len];

    for s in 0..n_symbols {
        let sym_start = aligned_start + s * sps;
        if sym_start + (n - 1) * os_factor >= samples.len() { break; }
        for i in 0..n {
            let g = (s * n + i) as f64;
            let ph = -TAU * cfo_frac / n as f64 * g;
            let (sin, cos) = ph.sin_cos();
            let rot = Complex::new(cos as f32, sin as f32);
            sto_buf[i] = samples[sym_start + i * os_factor] * rot * downchirp[i];
        }
        for b in &mut sto_buf[n..] { *b = Complex::new(0.0, 0.0); }
        fft_2n.process(&mut sto_buf);
        for (a, b) in sto_acc.iter_mut().zip(sto_buf.iter()) { *a += b.norm_sqr() as f64; }
        for b in &mut sto_buf { *b = Complex::new(0.0, 0.0); }
    }

    let k0 = sto_acc.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0);
    let y_m1 = sto_acc[(k0 + fft_len - 1) % fft_len];
    let y_0  = sto_acc[k0];
    let y_p1 = sto_acc[(k0 + 1) % fft_len];
    let ka = rctsl_frac(y_m1, y_0, y_p1, n);
    let k_residual = ((k0 as f64 + ka) / 2.0).rem_euclid(1.0);
    if k_residual > 0.5 { k_residual - 1.0 } else { k_residual }
}

// ── Demod helper for SYNC phase ─────────────────────────────────────────────

fn demod_sym(
    buf: &mut [Complex<f32>], samples: &[Complex<f32>], pos: usize,
    _sps: usize, os_factor: usize,
    cfo_corr: &[Complex<f32>], ref_chirp: &[Complex<f32>],
    fft: &dyn rustfft::Fft<f32>,
) -> Option<usize> {
    let n = buf.len();
    if pos + (n - 1) * os_factor >= samples.len() { return None; }
    for (i, b) in buf.iter_mut().enumerate() {
        *b = samples[pos + os_factor * i] * cfo_corr[i] * ref_chirp[i];
    }
    fft.process(buf);
    Some(buf.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .map(|(i, _)| i).unwrap_or(0))
}

// ── Main entry point ─────────────────────────────────────────────────────────

/// Hybrid LoRa preamble detector:
///   Phase 1: Non-coherent accumulation (robust at low SNR)
///   Phase 2: gr-lora_sdr-style SYNC walk (correct position tracking)
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

    let detect_threshold: f64 = 6.0;

    let not_found = |consumed: usize| FrameSyncResult {
        found: false, symbols: vec![], consumed, freq_offset_bins: 0.0,
    };

    let n_windows = samples.len().saturating_sub(sps - 1) / sps;
    if n_windows < pre_len + 5 { return not_found(0); }

    // ════════════════════════════════════════════════════════════════════
    // Phase 1: Non-coherent accumulation for preamble detection.
    // Sums magnitude-squared spectra across pre_len windows.
    // Robust at low per-symbol SNR because it averages over many symbols.
    // ════════════════════════════════════════════════════════════════════
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

        if ratio > detect_threshold {
            // ════════════════════════════════════════════════════════════
            // Phase 2: gr-lora_sdr-style SYNC walk.
            //
            // The accumulation window [w_start, w_start+pre_len) detected
            // a preamble.  k_hat = peak_idx = (τ + ε) mod N.
            //
            // Walk forward from end_w through the remaining frame:
            //   additional upchirps → sync words → downchirps → payload
            // ════════════════════════════════════════════════════════════
            let k_hat = peak_idx;
            let end_w_samples = (w_start + pre_len) * sps;

            // Coarse alignment: adjust read position by (N - k_hat) * os.
            let align_offset = ((n - k_hat) % n) * os_factor as usize;
            let mut pos = end_w_samples - sps + align_offset;

            // CFO_frac estimation (Bernier) on aligned preamble.
            let aligned_pre_start = w_start * sps + ((n - k_hat) % n) * os_factor as usize;
            let up_symb_to_use = pre_len.saturating_sub(2);

            let cfo_frac = estimate_cfo_frac_bernier(
                samples, aligned_pre_start, up_symb_to_use,
                sps, n, os_factor as usize, &downchirp, &mut planner,
            );

            let sto_frac = estimate_sto_frac(
                samples, aligned_pre_start, up_symb_to_use,
                sps, n, os_factor as usize, cfo_frac, &downchirp, &mut planner,
            );

            // CFO_frac correction vector for sync symbol demod.
            let cfo_corr: Vec<Complex<f32>> = (0..n).map(|i| {
                let phase = -TAU * cfo_frac / n as f64 * i as f64;
                let (sin, cos) = phase.sin_cos();
                Complex::new(cos as f32, sin as f32)
            }).collect();

            // ── NET_ID1: skip additional upchirps (bins near 0) ──────
            let net_id1;
            loop {
                let b = match demod_sym(
                    &mut buf, samples, pos, sps, os_factor as usize,
                    &cfo_corr, &downchirp, fft_n.as_ref(),
                ) {
                    Some(b) => b,
                    None    => return not_found(w_start * sps),
                };
                if b <= 1 || b >= n - 1 {
                    pos += sps;
                    continue;
                }
                net_id1 = b;
                break;
            }
            pos += sps;

            // ── NET_ID2 ──────────────────────────────────────────────
            let net_id2 = match demod_sym(
                &mut buf, samples, pos, sps, os_factor as usize,
                &cfo_corr, &downchirp, fft_n.as_ref(),
            ) {
                Some(b) => b,
                None    => return not_found(w_start * sps),
            };
            pos += sps;

            // ── DOWNCHIRP1 — consume ─────────────────────────────────
            if pos + sps > samples.len() { return not_found(w_start * sps); }
            pos += sps;

            // ── DOWNCHIRP2 — extract down_val ────────────────────────
            let down_val = match demod_sym(
                &mut buf, samples, pos, sps, os_factor as usize,
                &cfo_corr, &upchirp, fft_n.as_ref(),
            ) {
                Some(b) => b,
                None    => return not_found(w_start * sps),
            };
            pos += sps;

            // Integer CFO from downchirp.
            let cfo_int = if down_val < n / 2 {
                (down_val as f64 / 2.0).floor()
            } else {
                ((down_val as f64 - n as f64) / 2.0).floor()
            };
            let total_cfo = cfo_int + cfo_frac;

            // SFO estimate.
            let sfo_hat = if center_freq_hz > 0.0 {
                total_cfo * bw_hz / center_freq_hz
            } else {
                0.0
            };

            // Final alignment: quarter downchirp + cfo_int correction.
            let cfo_int_samples = (os_factor as isize) * (cfo_int as isize);
            let payload_start = (pos as isize + sps as isize / 4 + cfo_int_samples)
                .max(0) as usize;

            // ── Sync word validation ─────────────────────────────────
            let exp_sw0 = ((sync_word as usize & 0xF0) >> 4) << 3;
            let exp_sw1 =  (sync_word as usize & 0x0F)       << 3;
            let sw_ok = bins_close(net_id1, exp_sw0, n, 4)
                     && bins_close(net_id2, exp_sw1, n, 4);

            eprintln!("[sync] ratio={ratio:.1} k_hat={k_hat} cfo_int={cfo_int:.0} \
                       cfo_frac={cfo_frac:.4} total={total_cfo:.4} sfo={sfo_hat:.6} \
                       sw=[{net_id1},{net_id2}] exp=[{exp_sw0},{exp_sw1}] ok={sw_ok}");

            if !sw_ok {
                // Skip past this detection and retry.
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

            // ════════════════════════════════════════════════════════════
            // Payload extraction with CFO + STO + SFO correction.
            // ════════════════════════════════════════════════════════════
            if payload_start + sps > samples.len() {
                eprintln!("[sync] not enough payload: pos={payload_start} len={}",
                    samples.len());
                return not_found(w_start * sps);
            }

            let n_payload_samps = samples.len() - payload_start;
            let n_payload_syms  = n_payload_samps / sps;
            let len = n_payload_syms * sps;
            eprintln!("[sync] payload: {n_payload_syms} symbols from pos {payload_start}");

            let sto_evolved = sto_frac + sfo_hat * (pre_len as f64 + 4.25);
            let sto_offset  = (sto_evolved * os_factor as f64).round() as isize;

            let mut corrected = Vec::with_capacity(len);
            let mut sfo_cum: f64 = 0.0;
            let mut sample_adjust: isize = 0;
            let os = os_factor as f64;

            for sym in 0..n_payload_syms {
                for i in 0..sps {
                    let global_i = sym * sps + i;
                    let src_idx = payload_start as isize
                        + global_i as isize + sto_offset + sample_adjust;
                    let src_idx = src_idx.clamp(0, samples.len() as isize - 1) as usize;
                    let phase = -TAU * total_cfo / n as f64 * global_i as f64;
                    let (sin, cos) = phase.sin_cos();
                    let rot = Complex::new(cos as f32, sin as f32);
                    corrected.push(samples[src_idx] * rot);
                }
                sfo_cum += sfo_hat;
                if sfo_cum.abs() > 0.5 / os {
                    let adj = if sfo_cum > 0.0 { 1_isize } else { -1 };
                    sample_adjust += adj;
                    sfo_cum -= adj as f64 / os;
                }
            }

            return FrameSyncResult {
                found: true,
                symbols: corrected,
                consumed: payload_start + len,
                freq_offset_bins: total_cfo,
            };
        }

        // ── Slide window by 1 ────────────────────────────────────────────
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
