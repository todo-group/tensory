fn main() {
    if cfg!(feature = "__test-build") {
        lin_dev_deps_build::build();
    }
}
