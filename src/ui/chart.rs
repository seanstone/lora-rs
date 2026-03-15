use egui::{Event, Vec2, Vec2b, Id};
use egui_plot::{Legend, Plot, PlotBounds, GridMark, GridInput, AxisHints};
use std::sync::{Arc, Mutex};
use std::ops::RangeInclusive;
use crate::ui::Plottable;

pub struct Chart {
    name:          String,
    plots:         Arc<Mutex<Vec<Arc<dyn Plottable>>>>,
    x_limits:      [f64; 2],
    y_limits:      [f64; 2],
    x_bounds:      Option<[f64; 2]>,
    y_bounds:      Option<[f64; 2]>,
    last_x_bounds: [f64; 2],
    last_y_bounds: [f64; 2],
    x_min_span:    f64,
    y_min_span:    f64,
    link_axis:     (Id, Vec2b),
    link_cursor:   (Id, Vec2b),
    /// When Some, x-axis labels show MHz instead of bin indices.
    /// (offset_mhz, scale_mhz_per_bin) where freq = offset + bin * scale.
    x_freq:        Option<(f64, f64)>,
    /// When Some, y-axis labels show elapsed time instead of raw [0,1] values.
    /// Value is the total time in seconds represented by Y=0..1.
    y_time_secs:   Option<f64>,
}

impl Chart {
    pub fn new(name: &str) -> Self {
        Self {
            name:          name.to_string(),
            plots:         Default::default(),
            x_limits:      [0.0, 1.0],
            y_limits:      [-100.0, 50.0],
            x_bounds:      None,
            y_bounds:      None,
            last_x_bounds: [0.0, 1.0],
            last_y_bounds: [-100.0, 50.0],
            x_min_span:    1.0,
            y_min_span:    5.0,
            link_axis:     (Id::new(""), Vec2b::new(false, false)),
            link_cursor:   (Id::new(""), Vec2b::new(false, false)),
            x_freq:        None,
            y_time_secs:   None,
        }
    }

    /// Show frequency (MHz) on the x-axis instead of FFT bin indices.
    /// Call every frame while in UHD mode; pass `None` to revert to bins.
    pub fn set_x_freq_display(&mut self, center_hz: f64, bw_hz: f64, fft_size: usize) {
        let scale  = bw_hz / fft_size as f64 / 1e6;
        let offset = (center_hz - bw_hz / 2.0) / 1e6;
        self.x_freq = Some((offset, scale));
    }

    pub fn clear_x_freq_display(&mut self) { self.x_freq = None; }

    /// Set total elapsed time (seconds) that the waterfall Y=[0,1] span represents.
    /// Tick labels will render as "Xms" or "X.Xs".
    pub fn set_y_time_display(&mut self, total_secs: f64) {
        self.y_time_secs = Some(total_secs);
    }

    pub fn add(&mut self, item: Arc<dyn Plottable>) {
        self.plots.lock().unwrap().push(item);
    }

    pub fn set_x_limits(&mut self, limits: [f64; 2]) {
        self.x_limits = limits;
        self.x_bounds = Some(limits);
        self.last_x_bounds = limits;
    }

    pub fn set_y_limits(&mut self, limits: [f64; 2]) {
        self.y_limits = limits;
        self.y_bounds = Some(limits);
        self.last_y_bounds = limits;
    }

    /// X bounds as of the last rendered frame (already clamped by limit_bounds).
    pub fn last_x_bounds(&self) -> [f64; 2] { self.last_x_bounds }

    /// Override the X bounds for the next frame (bypasses limit_bounds, but
    /// the value should already be clamped by the source chart).
    pub fn sync_x_bounds(&mut self, bounds: [f64; 2]) { self.x_bounds = Some(bounds); }

    pub fn set_link_axis(&mut self, id: &str, x: bool, y: bool) {
        self.link_axis = (id.to_string().into(), Vec2b::new(x, y));
    }

    pub fn set_link_cursor(&mut self, id: &str, x: bool, y: bool) {
        self.link_cursor = (id.to_string().into(), Vec2b::new(x, y));
    }

    /// Render the chart inside `ui`.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let input = ui.input(|i| {
            let scroll = i.events.iter().find_map(|e| match e {
                Event::MouseWheel { delta, .. } => Some(*delta),
                _ => None,
            });
            (scroll, i.pointer.primary_down(), i.modifiers)
        });

        let plots   = self.plots.clone();
        let plot_id = self.name.clone();
        let x_freq  = self.x_freq;
        let plot    = Plot::new(plot_id)
            .legend(Legend::default())
            .show_axes(true)
            .custom_x_axes(self.x_axes())
            .custom_y_axes(self.y_axes())
            .show_grid(true)
            .allow_drag(false)
            .allow_zoom(true)
            .allow_scroll(false)
            .link_axis(self.link_axis.0, self.link_axis.1)
            .link_cursor(self.link_cursor.0, self.link_cursor.1)
            .label_formatter(move |name, value| {
                plots.lock().unwrap().iter()
                    .map(|p| p.label_formatter(name, value))
                    .collect::<Vec<_>>()
                    .join("")
            });
        // Conditionally attach the freq-aligned grid spacer.
        let plot = if let Some((offset, scale)) = x_freq {
            plot.x_grid_spacer(move |input| x_grid_freq(input, offset, scale))
        } else {
            plot
        };

        plot.show(ui, |plot_ui| {
            self.handle_bounds(plot_ui);
            self.handle_input(input, plot_ui);
            for item in self.plots.lock().unwrap().iter() {
                item.plot(plot_ui);
            }
        });
    }

    fn x_axes(&self) -> Vec<AxisHints<'static>> {
        let freq = self.x_freq;
        let fmt = move |mark: GridMark, _: &RangeInclusive<f64>| match freq {
            Some((offset, scale)) => format!("{:.3}", offset + mark.value * scale),
            None                  => format!("{:.0}", mark.value),
        };
        vec![AxisHints::new_x().formatter(fmt).placement(egui_plot::VPlacement::Bottom)]
    }

    fn y_axes(&self) -> Vec<AxisHints<'static>> {
        let time_total = self.y_time_secs;
        let fmt = move |mark: GridMark, _: &RangeInclusive<f64>| match time_total {
            Some(total) => {
                let secs = mark.value * total;
                if secs < 1.0 { format!("{:.0}ms", secs * 1000.0) }
                else          { format!("{:.1}s",  secs) }
            }
            None => format!("{:.0}", mark.value),
        };
        vec![AxisHints::new_y().formatter(fmt).min_thickness(50.0)]
    }

    fn handle_input(
        &self,
        input: (Option<Vec2>, bool, egui::Modifiers),
        plot_ui: &mut egui_plot::PlotUi<'_>,
    ) {
        let (scroll, pointer_down, modifiers) = input;
        let bounds = plot_ui.plot_bounds();

        if let Some(mut scroll) = scroll {
            scroll = Vec2::splat(scroll.x + scroll.y);
            const ZOOM_SPEED: f32 = 2.0;
            let mut zoom = if modifiers.shift {
                Vec2::from([(scroll.x * ZOOM_SPEED / 10.0).exp(), (scroll.y * ZOOM_SPEED / 10.0).exp()])
            } else {
                Vec2::from([(scroll.x * ZOOM_SPEED / 10.0).exp(), 1.0])
            };
            let xspan = bounds.max()[0] - bounds.min()[0];
            let yspan = bounds.max()[1] - bounds.min()[1];
            if zoom[0] < 1. && xspan >= self.x_limits[1] - self.x_limits[0] { zoom[0] = 1.; }
            if zoom[1] < 1. && yspan >= self.y_limits[1] - self.y_limits[0] { zoom[1] = 1.; }
            if zoom[0] > 1. && xspan <= self.x_min_span * zoom[0] as f64 { zoom[0] = (xspan / self.x_min_span) as f32; }
            if zoom[1] > 1. && yspan <= self.y_min_span * zoom[1] as f64 { zoom[1] = (yspan / self.y_min_span) as f32; }
            if let Some(pos) = plot_ui.pointer_coordinate() {
                plot_ui.zoom_bounds(zoom, pos);
            }
        }

        if plot_ui.response().hovered() && pointer_down {
            let mut dt = -plot_ui.pointer_coordinate_drag_delta();
            let ominx = bounds.min()[0] + (dt[0] as f64) < self.x_limits[0];
            let omaxx = bounds.max()[0] + (dt[0] as f64) > self.x_limits[1];
            let ominy = bounds.min()[1] + (dt[1] as f64) < self.y_limits[0];
            let omaxy = bounds.max()[1] + (dt[1] as f64) > self.y_limits[1];
            if ominx      { dt[0] = (self.x_limits[0] - bounds.min()[0]) as f32; }
            else if omaxx { dt[0] = (self.x_limits[1] - bounds.max()[0]) as f32; }
            if ominy      { dt[1] = (self.y_limits[0] - bounds.min()[1]) as f32; }
            else if omaxy { dt[1] = (self.y_limits[1] - bounds.max()[1]) as f32; }
            plot_ui.translate_bounds(dt);
        }
    }

    fn limit_bounds(b: [f64; 2], lim: [f64; 2], min_span: f64, last: [f64; 2]) -> [f64; 2] {
        if b[1] - b[0] < min_span           { last }
        else if b[1] - b[0] > lim[1] - lim[0] { lim }
        else if b[1] > lim[1]               { [lim[1] - (b[1] - b[0]), lim[1]] }
        else if b[0] < lim[0]               { [lim[0], lim[0] + (b[1] - b[0])] }
        else                                { b }
    }

    fn handle_bounds(&mut self, plot_ui: &mut egui_plot::PlotUi<'_>) {
        let bounds = plot_ui.plot_bounds();
        let xb = self.x_bounds.take().unwrap_or_else(||
            Self::limit_bounds([bounds.min()[0], bounds.max()[0]], self.x_limits, self.x_min_span, self.last_x_bounds));
        let yb = self.y_bounds.take().unwrap_or_else(||
            Self::limit_bounds([bounds.min()[1], bounds.max()[1]], self.y_limits, self.y_min_span, self.last_y_bounds));
        plot_ui.set_plot_bounds(PlotBounds::from_min_max([xb[0], yb[0]], [xb[1], yb[1]]));
        self.last_x_bounds = xb;
        self.last_y_bounds = yb;
    }
}

/// X grid spacer for frequency-display mode.
/// Emits exactly two levels: major (3–8 visible lines) and minor (major/5).
/// egui_plot renders minor lines at lower alpha because their step_size is
/// smaller relative to the major step_size.
fn x_grid_freq(input: GridInput, offset_mhz: f64, scale_mhz_per_bin: f64) -> Vec<GridMark> {
    let (min_bin, max_bin) = input.bounds;
    let min_mhz = offset_mhz + min_bin * scale_mhz_per_bin;
    let max_mhz = offset_mhz + max_bin * scale_mhz_per_bin;
    let span    = (max_mhz - min_mhz).abs();

    // Coarse candidates (major lines), largest first.
    let candidates = [
        100.0_f64, 50.0, 25.0, 10.0, 5.0, 2.5, 1.0,
        0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001,
    ];
    // Major: first (largest) step that still gives ≥ 3 lines (step ≤ span/3).
    let major_mhz = candidates.iter().copied()
        .find(|&s| s <= span / 3.0)
        .unwrap_or(candidates[candidates.len() - 1]);
    let minor_mhz = major_mhz / 5.0;

    let major_bins = major_mhz / scale_mhz_per_bin;
    let minor_bins = minor_mhz / scale_mhz_per_bin;

    let mut marks = vec![];
    // Major lines — full prominence (largest step_size).
    let i_min = (min_mhz / major_mhz).floor() as i64;
    let i_max = (max_mhz / major_mhz).ceil()  as i64;
    for i in i_min..=i_max {
        let bin = (i as f64 * major_mhz - offset_mhz) / scale_mhz_per_bin;
        marks.push(GridMark { value: bin, step_size: major_bins });
    }
    // Minor lines — lower prominence because step_size is 5× smaller,
    // so egui_plot renders them at proportionally lower alpha.
    let i_min = (min_mhz / minor_mhz).floor() as i64;
    let i_max = (max_mhz / minor_mhz).ceil()  as i64;
    for i in i_min..=i_max {
        if i % 5 == 0 { continue; } // skip positions already covered by major
        let bin = (i as f64 * minor_mhz - offset_mhz) / scale_mhz_per_bin;
        marks.push(GridMark { value: bin, step_size: minor_bins });
    }
    marks
}
