/*
 * uhd_glue.c — thin C wrapper around the UHD C API.
 *
 * Exposes a minimal stateless(-ish) interface that Rust can call via FFI
 * without needing to construct any UHD structs directly.
 *
 * One USRP device at a time; not reentrant.
 */

#include <uhd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Global device state ───────────────────────────────────────────────────── */

static uhd_usrp_handle       g_usrp        = NULL;
static uhd_rx_streamer_handle g_rx_streamer = NULL;
static uhd_rx_metadata_handle g_rx_metadata = NULL;
static uhd_tx_streamer_handle g_tx_streamer = NULL;
static uhd_tx_metadata_handle g_tx_metadata = NULL;

static size_t g_rx_buf_size = 0;
static size_t g_tx_buf_size = 0;

/* ── Open / close ──────────────────────────────────────────────────────────── */

/* Returns 0 on success, -1 on failure.
 * rx_buf_out / tx_buf_out receive the max samples-per-packet sizes. */
int uhd_glue_open(const char *args,
                  size_t     *rx_buf_out,
                  size_t     *tx_buf_out)
{
    /* Make USRP */
    if (uhd_usrp_make(&g_usrp, args) != UHD_ERROR_NONE || !g_usrp) {
        fprintf(stderr, "[uhd_glue] uhd_usrp_make failed\n");
        return -1;
    }

    /* ── RX streamer ── */
    uhd_rx_streamer_make(&g_rx_streamer);

    size_t rx_ch = 0;
    uhd_stream_args_t rx_args = {
        .cpu_format   = "sc16",
        .otw_format   = "sc16",
        .args         = "",
        .channel_list = &rx_ch,
        .n_channels   = 1
    };
    uhd_usrp_get_rx_stream(g_usrp, &rx_args, g_rx_streamer);
    uhd_rx_streamer_max_num_samps(g_rx_streamer, &g_rx_buf_size);
    uhd_rx_metadata_make(&g_rx_metadata);

    /* ── TX streamer ── */
    uhd_tx_streamer_make(&g_tx_streamer);

    size_t tx_ch = 0;
    uhd_stream_args_t tx_args = {
        .cpu_format   = "sc16",
        .otw_format   = "sc16",
        .args         = "",
        .channel_list = &tx_ch,
        .n_channels   = 1
    };
    uhd_usrp_get_tx_stream(g_usrp, &tx_args, g_tx_streamer);
    uhd_tx_streamer_max_num_samps(g_tx_streamer, &g_tx_buf_size);
    uhd_tx_metadata_make(&g_tx_metadata, 0, 0, 0.0, 0, 0);

    if (rx_buf_out) *rx_buf_out = g_rx_buf_size;
    if (tx_buf_out) *tx_buf_out = g_tx_buf_size;
    return 0;
}

void uhd_glue_close(void)
{
    if (g_rx_metadata)  uhd_rx_metadata_free(&g_rx_metadata);
    if (g_rx_streamer)  uhd_rx_streamer_free(&g_rx_streamer);
    if (g_tx_metadata)  uhd_tx_metadata_free(&g_tx_metadata);
    if (g_tx_streamer)  uhd_tx_streamer_free(&g_tx_streamer);
    if (g_usrp)         uhd_usrp_free(&g_usrp);
    g_usrp = NULL; g_rx_streamer = NULL; g_tx_streamer = NULL;
    g_rx_metadata = NULL; g_tx_metadata = NULL;
}

/* ── RX configuration ─────────────────────────────────────────────────────── */

void uhd_glue_set_rx_rate(double rate) {
    uhd_usrp_set_rx_rate(g_usrp, rate, 0);
}
void uhd_glue_set_rx_bw(double bw) {
    uhd_usrp_set_rx_bandwidth(g_usrp, bw, 0);
}
void uhd_glue_set_rx_freq(double freq) {
    uhd_tune_request_t req = {
        .target_freq      = freq,
        .rf_freq_policy   = UHD_TUNE_REQUEST_POLICY_AUTO,
        .dsp_freq_policy  = UHD_TUNE_REQUEST_POLICY_AUTO,
        .args             = ""
    };
    uhd_tune_result_t result;
    uhd_usrp_set_rx_freq(g_usrp, &req, 0, &result);
}
void uhd_glue_set_rx_gain(double gain) {
    uhd_usrp_set_rx_gain(g_usrp, gain, 0, "");
}

/* ── TX configuration ─────────────────────────────────────────────────────── */

void uhd_glue_set_tx_rate(double rate) {
    uhd_usrp_set_tx_rate(g_usrp, rate, 0);
}
void uhd_glue_set_tx_bw(double bw) {
    uhd_usrp_set_tx_bandwidth(g_usrp, bw, 0);
}
void uhd_glue_set_tx_freq(double freq) {
    uhd_tune_request_t req = {
        .target_freq      = freq,
        .rf_freq_policy   = UHD_TUNE_REQUEST_POLICY_AUTO,
        .dsp_freq_policy  = UHD_TUNE_REQUEST_POLICY_AUTO,
        .args             = ""
    };
    uhd_tune_result_t result;
    uhd_usrp_set_tx_freq(g_usrp, &req, 0, &result);
}
void uhd_glue_set_tx_gain(double gain) {
    uhd_usrp_set_tx_gain(g_usrp, gain, 0, "");
}

/* ── Streaming control ────────────────────────────────────────────────────── */

void uhd_glue_start_rx(void) {
    uhd_stream_cmd_t cmd = {
        .stream_mode       = UHD_STREAM_MODE_START_CONTINUOUS,
        .num_samps         = 0,
        .stream_now        = 1,
        .time_spec_full_secs = 0,
        .time_spec_frac_secs = 0.0
    };
    uhd_rx_streamer_issue_stream_cmd(g_rx_streamer, &cmd);
}

void uhd_glue_stop_rx(void) {
    uhd_stream_cmd_t cmd = {
        .stream_mode       = UHD_STREAM_MODE_STOP_CONTINUOUS,
        .num_samps         = 0,
        .stream_now        = 1,
        .time_spec_full_secs = 0,
        .time_spec_frac_secs = 0.0
    };
    uhd_rx_streamer_issue_stream_cmd(g_rx_streamer, &cmd);
}

/* ── Data transfer ────────────────────────────────────────────────────────── */

/* Receive up to `n` sc16 samples into `buf`.  Returns samples actually received. */
size_t uhd_glue_recv(int16_t *buf, size_t n)
{
    void   *buffers[1] = { buf };
    size_t  items_recvd = 0;
    uhd_rx_streamer_recv(g_rx_streamer, buffers, n,
                         &g_rx_metadata, 0.1, 0, &items_recvd);
    return items_recvd;
}

/* Send `n` sc16 samples from `buf`.  Returns samples actually sent. */
size_t uhd_glue_send(const int16_t *buf, size_t n)
{
    const void *buffers[1] = { buf };
    size_t      items_sent = 0;
    uhd_tx_streamer_send(g_tx_streamer, buffers, n,
                         &g_tx_metadata, 0.1, &items_sent);
    return items_sent;
}

size_t uhd_glue_rx_buf_size(void) { return g_rx_buf_size; }
size_t uhd_glue_tx_buf_size(void) { return g_tx_buf_size; }
