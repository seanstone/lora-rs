use egui::{vec2, Color32, ColorImage, Id, TextureOptions};
use egui_plot::{PlotImage, PlotPoint, PlotUi};
use std::sync::{Arc, Mutex};
use crate::ui::Plottable;

pub struct WaterfallPlot {
    name:      String,
    spectrum:  Mutex<Vec<[f64; 2]>>,
    reference: f64,
    id:        Mutex<Option<Id>>,
    image:     Mutex<ColorImage>,
    width:     Mutex<usize>,
    height:    Mutex<usize>,
    texture:   Mutex<Option<egui::TextureHandle>>,
    freq:      Mutex<f64>,   // center x in plot coords
    bw:        Mutex<f64>,   // total width in plot coords
}

impl WaterfallPlot {
    pub fn new(name: &str, spectrum: Vec<[f64; 2]>, reference: f64) -> Arc<Self> {
        let width  = spectrum.len().max(1);
        let height = 256;
        let image  = ColorImage::new([width, height], vec![Color32::BLACK; width * height]);
        Arc::new(Self {
            name: name.to_string(),
            spectrum: Mutex::new(spectrum),
            reference,
            id: Mutex::new(None),
            image: Mutex::new(image),
            width: Mutex::new(width),
            height: Mutex::new(height),
            texture: Mutex::new(None),
            freq: Mutex::new(0.0),
            bw:   Mutex::new(1.0),
        })
    }

    /// Viridis colormap — perceptually uniform purple → blue → teal → green → yellow.
    pub fn colormap_viridis(v: f32) -> Color32 {
        let v = v.clamp(0.0, 1.0);
        const VIRIDIS: [(u8, u8, u8); 6] = [
            (68,  1,  84),
            (59,  82, 139),
            (33, 145, 140),
            (94, 201,  98),
            (253, 231, 37),
            (253, 231, 37),
        ];
        let n      = VIRIDIS.len() - 1;
        let scaled = v * n as f32;
        let i      = scaled.floor() as usize;
        let t      = scaled - i as f32;
        let (r1, g1, b1) = VIRIDIS[i.min(n)];
        let (r2, g2, b2) = VIRIDIS[(i + 1).min(n)];
        Color32::from_rgb(
            (r1 as f32 + (r2 as f32 - r1 as f32) * t) as u8,
            (g1 as f32 + (g2 as f32 - g1 as f32) * t) as u8,
            (b1 as f32 + (b2 as f32 - b1 as f32) * t) as u8,
        )
    }

    /// Push a new spectrum row into the rolling buffer.
    pub fn update(&self, spectrum: Vec<[f64; 2]>) {
        let width = spectrum.len().max(1);
        {
            let prev = self.spectrum.lock().unwrap().len();
            if width != prev {
                let height = *self.height.lock().unwrap();
                *self.width.lock().unwrap() = width;
                *self.image.lock().unwrap() =
                    ColorImage::new([width, height], vec![Color32::BLACK; width * height]);
            }
        }
        *self.spectrum.lock().unwrap() = spectrum.clone();

        let mut img    = self.image.lock().unwrap();
        let w          = *self.width.lock().unwrap();
        let h          = *self.height.lock().unwrap();
        let reference  = self.reference;

        // Scroll: copy each row down one position.
        for y in (1..h).rev() {
            let row = img.pixels[(y - 1) * w..y * w].to_vec();
            img.pixels[y * w..(y + 1) * w].copy_from_slice(&row);
        }
        // Insert new row at top.
        for (x, &[_, power]) in spectrum.iter().enumerate() {
            img.pixels[x] = Self::colormap_viridis(((power - reference) / 80.0) as f32);
        }
    }

    pub fn set_freq(&self, freq: f64) { *self.freq.lock().unwrap() = freq; }
    pub fn set_bw(&self,   bw:   f64) { *self.bw.lock().unwrap()   = bw;   }
}

impl Plottable for WaterfallPlot {
    fn plot(&self, plot_ui: &mut PlotUi<'_>) {
        let mut tex_lock = self.texture.lock().unwrap();
        let img = self.image.lock().unwrap().clone();

        if let Some(tex) = tex_lock.as_mut() {
            tex.set(img, TextureOptions::default());
        } else {
            *tex_lock = Some(plot_ui.ctx().load_texture("waterfall", img, Default::default()));
        }

        if let Some(tex) = &*tex_lock {
            plot_ui.image(PlotImage::new(
                "Waterfall",
                tex,
                PlotPoint::new(*self.freq.lock().unwrap(), 0.5),
                vec2(*self.bw.lock().unwrap() as f32, 1.0),
            ));
        }
    }

    fn label_formatter(&self, _name: &str, _value: &PlotPoint) -> String {
        String::new()
    }

    fn id(&self) -> Option<egui::Id> {
        *self.id.lock().unwrap()
    }
}
