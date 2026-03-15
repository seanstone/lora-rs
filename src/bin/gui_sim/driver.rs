use rustfft::num_complex::Complex;

/// Abstraction over the IQ transport layer (sim channel or UHD hardware).
///
/// The sim loop calls:
///   - `push_samples` to hand off clean TX samples.
///   - `tick(n)` to consume n samples from the receive path (blocks on hardware).
///   - `pending_samples` for throttle / lag display.
///   - `clear` on reset / settings change.
///   - `set_signal_amp` / `set_noise_sigma` for level control.
pub(crate) trait Driver: Send {
    fn push_samples(&mut self, samples: Vec<Complex<f32>>);

    /// Produce exactly `n` received samples.  May block (hardware) or be instant (sim).
    fn tick(&mut self, n: usize) -> Vec<Complex<f32>>;

    /// Approximate number of TX samples still queued / in-flight.
    fn pending_samples(&self) -> usize;

    /// Flush all in-flight state (called on reset / settings change).
    fn clear(&mut self);

    fn set_signal_amp(&mut self, amp: f32);

    /// No-op for hardware drivers (noise comes from the RF environment).
    fn set_noise_sigma(&mut self, sigma: f32);

    /// Apply hardware RX gain in dB.  No-op for the sim channel.
    fn set_hw_rx_gain(&mut self, _db: f64) {}

    /// Apply hardware TX gain in dB.  No-op for the sim channel.
    fn set_hw_tx_gain(&mut self, _db: f64) {}

    /// Whether this driver supports park/unpark without full teardown.
    fn is_parkable(&self) -> bool { false }

    /// Stop hardware streaming and idle worker threads.  No-op for sim.
    fn park(&mut self) {}

    /// Restart hardware streaming, re-applying the given RF settings.  No-op for sim.
    fn unpark(&mut self, _freq_hz: f64, _sr_hz: f64, _bw_hz: f64,
              _rx_gain_db: f64, _tx_gain_db: f64) {}
}
