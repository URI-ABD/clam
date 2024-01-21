//! Trait for individual data points.

use core::fmt::Debug;

use distances::Number;

/// Trait for individual data points.
pub trait Instance: Debug + Send + Sync + Clone {
    /// Convert the instance to a byte vector.
    fn to_bytes(&self) -> Vec<u8>;

    /// Convert a byte vector to an instance.
    ///
    /// # Errors
    ///
    /// If the byte vector cannot be parsed into an instance.
    fn from_bytes(bytes: &[u8]) -> Result<Self, String>
    where
        Self: Sized;

    /// The name of the type of instance.
    fn type_name() -> String;

    /// Save the instance to a file.
    ///
    /// # Errors
    ///
    /// If the file cannot be written to.
    fn save<W: std::io::Write>(&self, writer: &mut W) -> Result<(), String> {
        let bytes = self.to_bytes();
        let num_bytes = bytes.len().to_be_bytes();
        writer
            .write_all(&num_bytes)
            .and_then(|()| writer.write_all(&bytes))
            .map_err(|e| e.to_string())
    }

    /// Load the instance from a file.
    ///
    /// # Errors
    ///
    /// If the file cannot be read or the instance cannot be parsed.
    fn load<R: std::io::Read>(reader: &mut R) -> Result<Self, String>
    where
        Self: Sized,
    {
        let mut num_bytes = vec![0; <usize as Number>::num_bytes()];
        reader.read_exact(&mut num_bytes).map_err(|e| e.to_string())?;
        let num_bytes = <usize as Number>::from_be_bytes(&num_bytes);

        let mut buf = vec![0; num_bytes];
        reader.read_exact(&mut buf).map_err(|e| e.to_string())?;

        Self::from_bytes(&buf)
    }
}

impl<T: Number> Instance for Vec<T> {
    fn to_bytes(&self) -> Vec<u8> {
        self.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() % T::num_bytes() == 0 {
            Ok(bytes
                .chunks_exact(T::num_bytes())
                .map(|x| T::from_le_bytes(x))
                .collect::<Self>())
        } else {
            Err(format!(
                "Expected a multiple of {} bytes, got {}",
                T::num_bytes(),
                bytes.len()
            ))
        }
    }

    fn type_name() -> String {
        format!("Vec<{}>", T::type_name())
    }
}

impl Instance for String {
    fn to_bytes(&self) -> Vec<u8> {
        Self::as_bytes(self).to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Self::from_utf8(bytes.to_vec()).map_err(|e| e.to_string())
    }

    fn type_name() -> String {
        "String".to_string()
    }
}

impl Instance for bool {
    fn to_bytes(&self) -> Vec<u8> {
        vec![<u8 as From<_>>::from(*self)]
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String>
    where
        Self: Sized,
    {
        if bytes.len() == 1 {
            Ok(bytes[0] != 0)
        } else {
            Err(format!("Expected 1 byte, got {}", bytes.len()))
        }
    }

    fn type_name() -> String {
        "bool".to_string()
    }
}

/// Macro to implement `Instance` for all `Number` types from `distances`.
///
/// This for using these types as metadata.
macro_rules! impl_instance_number {
    ($($ty:ty),*) => {
        $(
            impl Instance for $ty {
                fn to_bytes(&self) -> Vec<u8> {
                    self.to_le_bytes().to_vec()
                }

                fn from_bytes(bytes: &[u8]) -> Result<Self, String>
                where
                    Self: Sized,
                {
                    if bytes.len() == <$ty as Number>::num_bytes() {
                        Ok(<$ty as Number>::from_le_bytes(bytes))
                    } else {
                        Err(format!("Expected {} bytes, got {}", <$ty as Number>::num_bytes(), bytes.len()))
                    }
                }

                fn type_name() -> String {
                    stringify!($ty).to_string()
                }
            }
        )*
    }
}

impl_instance_number!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, isize, i128, f32, f64);
