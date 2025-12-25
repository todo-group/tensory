# Tensory

[![Crates.io](https://img.shields.io/crates/v/tensory.svg)](https://crates.io/crates/tensory) [![Docs.rs](https://docs.rs/tensory/badge.svg)](https://docs.rs/tensory) <!-- [![CI](https://github.com/todo-group/tensory/actions/workflows/ci.yml/badge.svg)](https://github.com/todo-group/tensory/actions) -->

![Logo](assets/logo.svg)

**Tensory** is a fast, minimal, and extensible tensor network computation library written in Rust.

## Features

- **High performance**: Built with Rust's zero-cost abstractions and efficient memory management.
- **Modular design**: Easy to extend with custom tensor structures and backends.
- **Rust idioms**: Follows Rust best practices for safety and concurrency.
- **Cross-platform**: Supports Linux, macOS, and Windows.

## Example

Add Tensory to your `Cargo.toml`:

```toml
[dependencies]
tensory-core = { git = "https://github.com/todo-group/tensory.git" }
tensory-basic = { git = "https://github.com/todo-group/tensory.git" }
tensory-linalg = { git = "https://github.com/todo-group/tensory.git" }
tensory-ndarray = { git = "https://github.com/todo-group/tensory.git" }
anyhow = "1.0.100"
openblas-src = { version = "0.10.13", features = ["system"] }
blas-src = { version = "0.14.0", features = ["openblas"] }
```

Then, use it in your code:

```rust
use tensory_basic::mapper::VecMapper;
use tensory_core::prelude::*;
use tensory_linalg::prelude::*;
use tensory_ndarray::{NdDenseTensor, NdDenseTensorExt};

extern crate blas_src;

type Tensor<'a, E> = NdDenseTensor<E, VecMapper<&'a str>>;

fn main() -> anyhow::Result<()> {
    let t = Tensor::<f64>::random(lm!["a"=>10, "b"=>15, "c"=>20, "d"=>25])?;
    let (u, s, v) = (&t).svd(ls![&"a", &"b"], "us", "sv")?.with(((),))?;
    Ok(())
}
```

## Getting Help

Read [Examples](https://github.com/todo-group/tensory/example) for sample codes, or see API [Docs](https://docs.rs/tensory) to know details.

## Roadmap

- High-Dimentional-Diagonal Tensor
- Normalized tensor (for TRG)
- Sparse Tensor
- Lazy Contraction Tensor
- Allocation-aware interface
- GPU backend (CUDA, ROCm, ...)
- Grassmann Tensor
- Symbolic Function Tensor

## License

This project is licensed under the [**UNDEFIND!** License](https://github.com/todo-group/tensory/LICENSE).

## Acknowledgements

Inspired by:

- ITensor (C++)
- ndarray (Rust)
- grassmanntn (python)
