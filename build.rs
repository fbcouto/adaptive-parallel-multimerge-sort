fn main() {
    println!("cargo:rerun-if-changed=src/ffi_engine.cpp");
    println!("cargo:rerun-if-changed=src/MultiMergeSort.hpp");

    let mut build = cc::Build::new();
    build.cpp(true);

    // Safeguard: The C++ engine strictly requires GCC to leverage GNU Parallel and OpenMP.
    let compiler = build.get_compiler();
    if compiler.is_like_msvc() {
        panic!(
            "\n\n\
            ERROR: cc-rs selected MSVC (cl.exe) instead of GCC/MinGW.\n\
            This project uses __gnu_parallel and GCC-style flags and must be compiled with g++.\n\
            \n\
            Fix this by ensuring you are on a GNU target:\n\
            \x20  cargo +stable-x86_64-pc-windows-gnu build\n\n"
        );
    }

    build
        .flag("-std=c++20")          // Modern C++ standard
        .flag("-O3")                 // Maximum optimization
        .flag("-fopenmp");           // Enable OpenMP Work-Stealing

    // Leverages native GCC parallel algorithms as a fallback for high-entropy data
    build.define("MULTIMERGE_USE_GNU_PARALLEL", None);

    build
        .file("src/ffi_engine.cpp")
        .compile("cxx_engine");

    // Link the native OpenMP library
    println!("cargo:rustc-link-lib=gomp");
}