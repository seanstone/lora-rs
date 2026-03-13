use rand::{Rng, RngExt, SeedableRng};
use rustfft::num_complex::Complex;
use std::collections::VecDeque;

// ─── Channel ──────────────────────────────────────────────────────────────────
//
// The channel is a *streaming* per-sample AWGN mixer.
//
// TX side (upstream / future SDR-TX)
//   Push clean, noise-free LoRa packets via `push_packet()`. The channel
//   queues them and streams them sample-by-sample.
//
// RX side (downstream / future SDR-RX)
//   Call `tick(n)` each simulation step. It produces exactly `n` mixed
//   samples and reports any packets that completed during the step so the
//   caller can forward them to the RX decoder.
//
// Because AWGN is added *per sample* at the instantaneous noise level, a
// settings change mid-packet affects only the remaining samples — matching
// real RF channel behaviour. Future SDR integration: replace `tick()` with
// a driver read that returns real ADC samples.

/// A clean (noise-free, unit-amplitude) TX packet waiting in the channel.
pub(crate) struct TxPacketIq {
    pub payload: Vec<u8>,
    pub clean:   Vec<Complex<f32>>,
}

/// State for the packet currently being streamed through the channel.
struct ActivePacket {
    payload:   Vec<u8>,
    clean:     Vec<Complex<f32>>,
    offset:    usize,
    mixed_acc: Vec<Complex<f32>>,  // mixed samples accumulated for the RX decoder
}

pub(crate) struct Channel {
    /// Instantaneous noise sigma — updated each tick, applied per sample.
    pub noise_sigma: f32,
    /// Instantaneous TX signal gain — updated each tick, applied per sample.
    pub signal_amp:  f32,
    rng:             rand::rngs::StdRng,
    /// Clean packets waiting to enter the active slot.
    tx_queue:        VecDeque<TxPacketIq>,
    /// Packet currently being mixed sample-by-sample.
    active:          Option<ActivePacket>,
}

impl Channel {
    pub fn new(noise_sigma: f32, signal_amp: f32) -> Self {
        Self {
            noise_sigma,
            signal_amp,
            rng:      rand::rngs::StdRng::seed_from_u64(0xC0FFee),
            tx_queue: VecDeque::new(),
            active:   None,
        }
    }

    pub fn set_noise_sigma(&mut self, s: f32) { self.noise_sigma = s; }
    pub fn set_signal_amp(&mut self, a: f32)  { self.signal_amp  = a; }

    /// Enqueue a clean (noise-free) packet for streaming.
    pub fn push_packet(&mut self, payload: Vec<u8>, clean: Vec<Complex<f32>>) {
        self.tx_queue.push_back(TxPacketIq { payload, clean });
    }

    /// Flush all queued and in-progress packets.
    pub fn clear(&mut self) {
        self.tx_queue.clear();
        self.active = None;
    }

    /// Samples still queued or in progress (for buffer-lag reporting).
    pub fn pending_samples(&self) -> usize {
        let queued: usize = self.tx_queue.iter().map(|p| p.clean.len()).sum();
        let active = self.active.as_ref().map_or(0, |ap| ap.clean.len() - ap.offset);
        queued + active
    }

    /// Produce exactly `n` mixed samples.
    ///
    /// Returns `(mixed_samples, completed_packets)`. Each completed packet is
    /// `(original_payload, full_mixed_iq)` ready for the RX decoder.
    ///
    /// AWGN is applied **per sample** at the instantaneous `noise_sigma` /
    /// `signal_amp` — a settings change mid-call affects all remaining samples.
    pub fn tick(&mut self, n: usize) -> (Vec<Complex<f32>>, Vec<(Vec<u8>, Vec<Complex<f32>>)>) {
        let mut out  = Vec::with_capacity(n);
        let mut done = Vec::new();

        for _ in 0..n {
            // Promote the next queued packet to the active slot.
            if self.active.is_none() {
                if let Some(pkt) = self.tx_queue.pop_front() {
                    let cap = pkt.clean.len();
                    self.active = Some(ActivePacket {
                        payload:   pkt.payload,
                        clean:     pkt.clean,
                        offset:    0,
                        mixed_acc: Vec::with_capacity(cap),
                    });
                }
            }

            // Clean signal sample (zero during silence / inter-packet gap).
            let clean = match self.active {
                Some(ref ap) => ap.clean[ap.offset] * self.signal_amp,
                None         => Complex::new(0.0, 0.0),
            };

            // Per-sample AWGN at the *current* noise level.
            let noise = Complex::new(
                box_muller(&mut self.rng) * self.noise_sigma,
                box_muller(&mut self.rng) * self.noise_sigma,
            );
            let mixed = clean + noise;
            out.push(mixed);

            // Advance the active packet; accumulate mixed sample for RX.
            if let Some(ref mut ap) = self.active {
                ap.mixed_acc.push(mixed);
                ap.offset += 1;
                if ap.offset >= ap.clean.len() {
                    let finished = self.active.take().unwrap();
                    done.push((finished.payload, finished.mixed_acc));
                }
            }
        }

        (out, done)
    }
}

/// Box-Muller: one unit-normal sample from two uniform draws.
fn box_muller(rng: &mut impl Rng) -> f32 {
    let u1 = (rng.random::<f32>() + 1e-7_f32).min(1.0);
    let u2 = rng.random::<f32>();
    (-2.0_f32 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}
