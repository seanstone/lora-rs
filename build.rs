fn main() {
    if std::env::var("CARGO_FEATURE_UHD").is_ok() {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

        let mut build = cc::Build::new();
        build
            .file("src/bin/gui_sim/uhd_glue.c")
            .flag("-std=c99");

        // UHD is a C++ library that exposes a C API; the C++ runtime differs
        // by platform. Headers/libs also live in non-standard places on macOS.
        if target_os == "macos" {
            build.include("/opt/homebrew/include");
            println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
            println!("cargo:rustc-link-lib=c++");
        } else {
            println!("cargo:rustc-link-lib=stdc++");
        }

        build.compile("uhd_glue");
        println!("cargo:rustc-link-lib=uhd");
    }
}
