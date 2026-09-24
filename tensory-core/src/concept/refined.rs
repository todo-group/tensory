pub unsafe trait RefinedFrom<Raw> {
    type Err;
    fn verify(raw: &Raw) -> Result<(), Self::Err>;
    unsafe fn from_raw_unchecked(raw: Raw) -> Self;
    fn into_raw(self) -> Raw;
}

pub struct FromRawError<Raw, Err>(pub Raw, pub Err);

pub trait RefinedExt<Raw>: RefinedFrom<Raw> {
    fn from_raw(raw: Raw) -> Result<Self, FromRawError<Raw, Self::Err>>
    where
        Self: Sized;
}

impl<T: RefinedFrom<Raw>, Raw> RefinedExt<Raw> for T {
    fn from_raw(raw: Raw) -> Result<Self, FromRawError<Raw, Self::Err>>
    where
        Self: Sized,
    {
        if let Err(e) = Self::verify(&raw) {
            Err(FromRawError(raw, e))
        } else {
            Ok(unsafe { Self::from_raw_unchecked(raw) })
        }
    }
}
