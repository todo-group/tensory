# Tensory

[![Crates.io](https://img.shields.io/crates/v/tensory.svg)](https://crates.io/crates/tensory) [![Docs.rs](https://docs.rs/tensory/badge.svg)](https://docs.rs/tensory) [![CI](https://github.com/todo-group/tensory/actions/workflows/ci.yml/badge.svg)](https://github.com/yourname/tensory/actions)

![Logo](assets/logo.svg)

**Tensory** is a fast, minimal, and extensible tensor network computation library written in Rust.

> [!NOTE]
> The name **tensory** is already taken by a company, so we need to select the alternative name!

## Features

- **High performance**: Built with Rust's zero-cost abstractions and efficient memory management.
- **Modular design**: Easy to extend with custom tensor structures and backends.
- **Rust idioms**: Follows Rust best practices for safety and concurrency.
- **Cross-platform**: Supports Linux, macOS, and Windows.

## Example

Add Tensory to your `Cargo.toml`:

```toml
[dependencies]
tensory = "0.1"
```

Then, use it in your code:

```rust
fn main() {
    todo!()
}
```

## Getting Help

Read [Examples](https://github.com/todo-group/tensory/example) for sample codes, or see API [Docs](https://docs.rs/tensory) to know details.

## Roadmap

- High-Dimentional-Diagonal Tensor
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
