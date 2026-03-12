pub trait Plottable: Send + Sync {
    fn plot(&self, plot_ui: &mut egui_plot::PlotUi<'_>);
    fn label_formatter(&self, name: &str, value: &egui_plot::PlotPoint) -> String;
    fn id(&self) -> Option<egui::Id>;
}
