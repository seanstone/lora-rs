use egui::{Event, Vec2, Vec2b, Id};
use egui_plot::{Legend, Plot, PlotBounds, GridMark, AxisHints};
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
        }
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
        let plot    = Plot::new(plot_id)
            .legend(Legend::default())
            .show_axes(true)
            .custom_x_axes(Self::x_axes())
            .custom_y_axes(Self::y_axes())
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

        plot.show(ui, |plot_ui| {
            self.handle_bounds(plot_ui);
            self.handle_input(input, plot_ui);
            for item in self.plots.lock().unwrap().iter() {
                item.plot(plot_ui);
            }
        });
    }

    fn x_axes() -> Vec<AxisHints<'static>> {
        let fmt = |mark: GridMark, _: &RangeInclusive<f64>| format!("{:.0}", mark.value);
        vec![AxisHints::new_x().formatter(fmt).placement(egui_plot::VPlacement::Bottom)]
    }

    fn y_axes() -> Vec<AxisHints<'static>> {
        let fmt = |mark: GridMark, _: &RangeInclusive<f64>| format!("{:.0}", mark.value);
        vec![AxisHints::new_y().formatter(fmt).min_thickness(40.0)]
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
            let mut zoom = if modifiers.shift {
                Vec2::from([(scroll.x * 0.1 / 10.0).exp(), (scroll.y * 0.1 / 10.0).exp()])
            } else {
                Vec2::from([(scroll.x * 0.1 / 10.0).exp(), 1.0])
            };
            let xspan = bounds.max()[0] - bounds.min()[0];
            let yspan = bounds.max()[1] - bounds.min()[1];
            if zoom[0] < 1. && xspan >= self.x_limits[1] - self.x_limits[0] { zoom[0] = 1.; }
            if zoom[1] < 1. && yspan >= self.y_limits[1] - self.y_limits[0] { zoom[1] = 1.; }
            if zoom[0] > 1. && xspan <= self.x_min_span * zoom[0] as f64 { zoom[0] = (xspan / self.x_min_span) as f32; }
            if zoom[1] > 1. && yspan <= self.y_min_span * zoom[1] as f64 { zoom[1] = (yspan / self.y_min_span) as f32; }
            plot_ui.zoom_bounds_around_hovered(zoom);
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
