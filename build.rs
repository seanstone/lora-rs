fn main() {
    if std::env::var("CARGO_FEATURE_UHD").is_ok() {
        // Compile the C glue that wraps the UHD C API.
        cc::Build::new()
            .file("src/bin/gui_sim/uhd_glue.c")
            .include("/opt/homebrew/include")
            .flag("-std=c99")
            .compile("uhd_glue");

        // Link against libuhd (C++ library, but exposes C API).
        println!("cargo:rustc-link-lib=uhd");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
    }
}
