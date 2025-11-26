fn main() {
    if let Some(link_arg) = std::env::var_os("DEP_MKL_CORE_LINKARG") {
        println!("cargo::metadata=LINKARG={}", link_arg.display());
    }
}
