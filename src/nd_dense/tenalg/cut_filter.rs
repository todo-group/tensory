use ndarray::Ix;
use ndarray_linalg::Scalar;

pub trait CutFilter<E: Scalar> {
    fn min_ix(&self) -> Option<Ix>;
    fn max_ix(&self) -> Option<Ix>;
    fn cutoff(&self) -> Option<E::Real>;
}

impl<E: Scalar> CutFilter<E> for () {
    fn min_ix(&self) -> Option<Ix> {
        None
    }
    fn max_ix(&self) -> Option<Ix> {
        None
    }
    fn cutoff(&self) -> Option<E::Real> {
        None
    }
}

impl<E: Scalar, F1: CutFilter<E>, F2: CutFilter<E>> CutFilter<E> for (F1, F2) {
    fn min_ix(&self) -> Option<Ix> {
        self.0.min_ix().or(self.1.min_ix())
    }
    fn max_ix(&self) -> Option<Ix> {
        self.0.max_ix().or(self.1.max_ix())
    }
    fn cutoff(&self) -> Option<E::Real> {
        self.0.cutoff().or(self.1.cutoff())
    }
}

pub struct MaxIx {
    max_ix: Ix,
}

#[allow(non_snake_case)]
pub fn MaxIx(max_ix: Ix) -> MaxIx {
    MaxIx { max_ix }
}

impl<E: Scalar> CutFilter<E> for MaxIx {
    fn min_ix(&self) -> Option<Ix> {
        None
    }
    fn max_ix(&self) -> Option<Ix> {
        Some(self.max_ix)
    }
    fn cutoff(&self) -> Option<E::Real> {
        None
    }
}

pub struct MinIx {
    min_ix: Ix,
}

#[allow(non_snake_case)]
pub fn MinIx(min_ix: Ix) -> MinIx {
    MinIx { min_ix }
}

impl<E: Scalar> CutFilter<E> for MinIx {
    fn min_ix(&self) -> Option<Ix> {
        Some(self.min_ix)
    }
    fn max_ix(&self) -> Option<Ix> {
        None
    }
    fn cutoff(&self) -> Option<E::Real> {
        None
    }
}

pub struct ClampIx {
    min_ix: Ix,
    max_ix: Ix,
}

#[allow(non_snake_case)]
pub fn ClampIx(min_ix: Ix, max_ix: Ix) -> ClampIx {
    ClampIx { min_ix, max_ix }
}

impl<E: Scalar> CutFilter<E> for ClampIx {
    fn min_ix(&self) -> Option<Ix> {
        Some(self.min_ix)
    }
    fn max_ix(&self) -> Option<Ix> {
        Some(self.max_ix)
    }
    fn cutoff(&self) -> Option<E::Real> {
        None
    }
}

pub struct Cutoff<E> {
    cutoff: E,
}

#[allow(non_snake_case)]
pub fn Cutoff<E>(cutoff: E) -> Cutoff<E> {
    Cutoff { cutoff }
}

impl<E: Scalar> CutFilter<E> for Cutoff<E::Real> {
    fn min_ix(&self) -> Option<Ix> {
        None
    }
    fn max_ix(&self) -> Option<Ix> {
        None
    }
    fn cutoff(&self) -> Option<E::Real> {
        Some(self.cutoff)
    }
}
