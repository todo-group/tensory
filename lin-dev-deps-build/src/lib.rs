pub fn build() {
    // mkl
    if let Some(link_arg) = std::env::var_os("DEP_LINALG_BACKEND_LINKARG") {
        println!("cargo::rustc-link-arg={}", link_arg.display());
    }
}
