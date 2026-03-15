use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::{window, HtmlCanvasElement};
use tracing_wasm::{WASMLayer, WASMLayerConfigBuilder};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::Registry;
use tracing::Level;

use super::gui::GuiApp;
use super::DEFAULT_SF;

#[wasm_bindgen(start)]
pub async fn start() {
    console_error_panic_hook::set_once();

    let wasm_layer = WASMLayer::new(
        WASMLayerConfigBuilder::default()
            .set_max_level(Level::DEBUG)
            .build(),
    );
    let subscriber = Registry::default().with(wasm_layer);
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");

    spawn_local(async {
        let canvas = window().unwrap()
            .document().unwrap()
            .get_element_by_id("canvas").unwrap()
            .unchecked_into::<HtmlCanvasElement>();

        let _ = eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|_cc| Ok(Box::new(GuiApp::new(DEFAULT_SF)))),
            )
            .await;
    });
}
