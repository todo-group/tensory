use ndarray::{ArrayBase, CowArray, Data, Dimension, Ix2, Order, ShapeArg};

use super::error::TenalgError;

type Result<T> = core::result::Result<T, TenalgError>;

pub fn ten_to_mat<ST: Data, DT: Dimension, Sh>(
    ten: &ArrayBase<ST, DT>,
    shape: Sh,
) -> Result<CowArray<ST::Elem, Ix2>>
where
    (Sh, Order): ShapeArg<Dim = Ix2>,
    ST::Elem: Clone,
{
    let mat: CowArray<ST::Elem, Ix2> = ten.to_shape((shape, Order::ColumnMajor))?;
    Ok(mat)
}

pub fn mat_to_ten<ST: Data, Sh>(
    mat: &ArrayBase<ST, Ix2>,
    shape: Sh,
) -> Result<CowArray<ST::Elem, <(Sh, Order) as ShapeArg>::Dim>>
where
    (Sh, Order): ShapeArg,
    ST::Elem: Clone,
{
    let ten = mat.to_shape((shape, Order::ColumnMajor))?;
    Ok(ten)
}
