fn main() {
    println!("cargo:rerun-if-changed=src/ffi_engine.cpp");
    println!("cargo:rerun-if-changed=src/MultiMergeSort.hpp");
    println!("cargo:rerun-if-changed=build.rs");

    // Toggles read from the environment, so A/B benchmarking never requires
    // editing a source file. rerun-if-env-changed makes cargo rebuild the C++
    // side when one of them flips - without it, cargo would silently reuse the
    // previous object file and you would benchmark the wrong binary.
    println!("cargo:rerun-if-env-changed=MULTIMERGE_KWAY");
    println!("cargo:rerun-if-env-changed=MULTIMERGE_BIDIR");
    println!("cargo:rerun-if-env-changed=MULTIMERGE_GNU");
    println!("cargo:rerun-if-env-changed=MULTIMERGE_NATIVE");

    let mut build = cc::Build::new();
    build.cpp(true);

    // The engine needs libstdc++ specifically: <parallel/algorithm> is a GNU
    // extension and -fopenmp/-lgomp are the GCC spellings. Checking only for
    // MSVC let clang through, which fails later with a confusing error.
    let compiler = build.get_compiler();
    if !compiler.is_like_gnu() {
        panic!(
            "\n\n\
            ERROR: cc-rs did not select GCC/MinGW (g++).\n\
            This project uses __gnu_parallel, OpenMP and GCC-style flags.\n\
            \n\
            On Windows, build on the GNU target:\n\
            \x20  cargo +stable-x86_64-pc-windows-gnu build\n\
            On macOS, Apple clang has no libgomp - install GCC via Homebrew and\n\
            set CXX=g++-14 (or similar).\n\n"
        );
    }

    build
        .flag("-std=c++20")
        .flag("-O3")
        .flag("-fopenmp");

    // Fallback engine for high-entropy data. ON by default uses
    // __gnu_parallel::stable_sort; set MULTIMERGE_GNU=0 to use the internal
    // chunk_parallel_sort instead.
    let gnu = std::env::var("MULTIMERGE_GNU")
        .map(|v| !v.is_empty() && v != "0").unwrap_or(true);
    if gnu { build.define("MULTIMERGE_USE_GNU_PARALLEL", None); }

    // Cache-blocked k-way merge. ON by default.
    // Disable for A/B benchmarking with:
    //     $env:MULTIMERGE_KWAY = "0"   (PowerShell)
    //     MULTIMERGE_KWAY=0            (bash)
    // Optional fan-out override, must be a power of two (default 8):
    //     $env:MULTIMERGE_KWAY = "16"
    // Default: ON. Set MULTIMERGE_KWAY=0 to fall back to the plain binary
    // merge tree (useful for A/B benchmarking).
    let kway = std::env::var("MULTIMERGE_KWAY").unwrap_or_else(|_| "1".to_string());
    let kway_on = !kway.is_empty() && kway != "0";
    if kway_on {
        build.define("MULTIMERGE_KWAY", None);
        if let Ok(k) = kway.parse::<u32>() {
            if k > 1 {
                build.define("KWAY_FANOUT", k.to_string().as_str());
            }
        }
    }

    // Bidirectional leaf merge (merge_front + merge_back running as two tasks).
    // ON by default - this is the core mechanic of the project. The original
    // `if(k > 32768)` was unreachable, so the two halves always ran
    // sequentially; the threshold is now tied to leaf_size so the task really
    // spawns. Disable for A/B benchmarking with MULTIMERGE_BIDIR=0.
    let bidir = std::env::var("MULTIMERGE_BIDIR")
        .map(|v| !v.is_empty() && v != "0").unwrap_or(true);
    if bidir { build.define("MULTIMERGE_BIDIR", None); }

    // -march=native is opt-in because it makes the binary non-portable.
    // Measurements showed this workload is memory-bandwidth bound, so do not
    // expect much from it - it is here for completeness, not as a fix.
    let native = std::env::var("MULTIMERGE_NATIVE").is_ok();
    if native {
        build.flag("-march=native").flag("-funroll-loops");
    }

    // Echo the active configuration. Twice during this project a benchmark was
    // run against a different build than intended; this line makes the mode
    // visible in normal cargo output.
    println!(
        "cargo:warning=multimerge C++ engine: kway={} fanout={} bidir={} gnu_parallel={} native={}",
        if kway_on { "ON" } else { "off" },
        if kway_on && kway.parse::<u32>().map(|k| k > 1).unwrap_or(false) {
            kway.clone()
        } else if kway_on {
            "8 (default)".to_string()
        } else {
            "-".to_string()
        },
        if bidir { "ON" } else { "off" },
        if gnu { "ON" } else { "off (chunk_parallel_sort)" },
        if native { "ON" } else { "off" }
    );

    build
        .file("src/ffi_engine.cpp")
        .compile("cxx_engine");

    println!("cargo:rustc-link-lib=gomp");
}
