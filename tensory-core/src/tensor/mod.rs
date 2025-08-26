// pub mod error;

/// layer 1 tensor concept: tensor with axes indexed with usize
mod repr;
pub use repr::*;

/// broker between local usize axis description (layer 1 description) and global ID axis description (layer 2 description)
mod broker;
pub use broker::*;

/// layer 2 tensor concept: tensor with legs indexed with ID
mod tensor;
pub use tensor::*;

// /// layer 3 tensor concept: layer 2 tensor with executor registered
// pub mod tensor_with_executor;
