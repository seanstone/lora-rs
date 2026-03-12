use egui::{Color32, Id};
use egui_plot::{Line, LineStyle, PlotPoint};
use std::sync::{Arc, Mutex};
use crate::ui::Plottable;

pub struct SpectrumPlot {
    name:      String,
    spectrum:  Mutex<Vec<[f64; 2]>>,
    reference: f64,   // noise floor in dB (e.g. -60)
    span:      f64,   // dynamic range above reference (e.g. 80 dB)
    id:        Mutex<Option<Id>>,
}

impl SpectrumPlot {
    pub fn new(name: &str, spectrum: Vec<[f64; 2]>, reference: f64, span: f64) -> Arc<Self> {
        Arc::new(Self {
            name: name.to_string(),
            spectrum: Mutex::new(spectrum),
            reference,
            span,
            id: Mutex::new(None),
        })
    }

    pub fn update(&self, spectrum: Vec<[f64; 2]>) {
        *self.spectrum.lock().unwrap() = spectrum;
    }
}

impl Plottable for SpectrumPlot {
    fn plot(&self, plot_ui: &mut egui_plot::PlotUi<'_>) {
        let reference = self.reference;
        let span      = self.span;
        let line = Line::new(
                self.name.clone(),
                self.spectrum.lock().unwrap().clone(),
            )
            .id(self.name.clone())
            .gradient_color(
                Arc::new(move |point: PlotPoint| {
                    let v = ((point.y - reference) / span).clamp(0.0, 1.0) as f32;
                    Color32::DARK_RED.lerp_to_gamma(Color32::LIGHT_RED, v)
                }),
                true,
            )
            .fill(reference as f32)
            .fill_alpha(0.3)
            .style(LineStyle::Solid);
        plot_ui.add(line);
    }

    fn label_formatter(&self, _name: &str, value: &PlotPoint) -> String {
        let bin      = value.x;
        let spectrum = self.spectrum.lock().unwrap();
        let point    = spectrum.iter().filter(|p| p[0] <= bin).last().copied();
        match point {
            Some([b, p]) => format!("bin {:.0}\n{:.1} dB", b, p),
            None         => String::new(),
        }
    }

    fn id(&self) -> Option<egui::Id> {
        *self.id.lock().unwrap()
    }
}
