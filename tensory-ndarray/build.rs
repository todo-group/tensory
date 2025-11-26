#[cfg(feature = "__test-build")]
fn main() {
    lin_dev_deps_build::build();
}

#[cfg(not(feature = "__test-build"))]
fn main() {}
