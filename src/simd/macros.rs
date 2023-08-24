macro_rules! define_ty {
    ($id:ident, $($elem_tys:ident),+) => {
        #[derive(Copy, Clone, PartialEq, Debug)]
        #[allow(non_camel_case_types)]
        pub struct $id($($elem_tys),*);
    }
}

macro_rules! impl_minimal {
    ($id:ident, $elem_ty:ident, $elem_count:expr, $($elem_name:ident),+) => {
        impl $id {
            #[allow(clippy::complexity)]
            #[inline]
            pub const fn new($($elem_name: $elem_ty),*) -> Self {
                $id($($elem_name),*)
            }

            #[inline]
            pub const fn lanes() -> usize {
                $elem_count
            }

             pub const fn splat(value: $elem_ty) -> Self {
                $id($({
                    // this just allows us to repeat over the elements
                    #[allow(non_camel_case_types, dead_code)]
                    struct $elem_name;
                    value
                }),*)
            }
        }
    };
}

macro_rules! impl_op2 {
    ($trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            type Output = $typ;
            fn $fn(self, rhs: Self) -> Self::Output {
                Self(
                    self.0 $op rhs.0,
                    self.1 $op rhs.1,
                )
            }
        }
    };
    (assn $trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            fn $fn(&mut self, rhs: Self) {
                    self.0 $op rhs.0;
                    self.1 $op rhs.1;
            }
        }
    };
}

macro_rules! impl_op4 {
    ($trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            type Output = $typ;
            fn $fn(self, rhs: Self) -> Self::Output {
                Self(
                    self.0 $op rhs.0,
                    self.1 $op rhs.1,
                    self.2 $op rhs.2,
                    self.3 $op rhs.3,
                )
            }
        }
    };
    (assn $trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            fn $fn(&mut self, rhs: Self) {
                    self.0 $op rhs.0;
                    self.1 $op rhs.1;
                    self.2 $op rhs.2;
                    self.3 $op rhs.3;
            }
        }
    };
}

macro_rules! impl_op8 {
    ($trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            type Output = $typ;
            fn $fn(self, rhs: Self) -> Self::Output {
                Self(
                    self.0 $op rhs.0,
                    self.1 $op rhs.1,
                    self.2 $op rhs.2,
                    self.3 $op rhs.3,
                    self.4 $op rhs.4,
                    self.5 $op rhs.5,
                    self.6 $op rhs.6,
                    self.7 $op rhs.7,
                )
            }
        }
    };
    (assn $trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            fn $fn(&mut self, rhs: Self) {
                    self.0 $op rhs.0;
                    self.1 $op rhs.1;
                    self.2 $op rhs.2;
                    self.3 $op rhs.3;
                    self.4 $op rhs.4;
                    self.5 $op rhs.5;
                    self.6 $op rhs.6;
                    self.7 $op rhs.7;
            }
        }
    };
}

macro_rules! impl_euclidean {
    ($name:ident, $ty:ty) => {
        use super::Naive;
        impl $name {
            /// Calculate the squared distance between two slices
            pub fn euclidean_inner(a: &[$ty], b: &[$ty]) -> $name {
                let i = $name::from_slice(a);
                let j = $name::from_slice(b);
                let u = i - j;
                u * u
            }

            /// Calculate euclidean distance between two slices of equal length,
            /// using auto-vectorized SIMD primitives
            pub fn squared_distance(a: &[$ty], b: &[$ty]) -> $ty {
                assert_eq!(a.len(), b.len());
                if a.len() < $name::lanes() {
                    return Naive::squared_distance(a, b);
                }

                let mut i = 0;
                let mut sum = $name::splat(0 as $ty);
                while a.len() - $name::lanes() >= i {
                    sum += $name::euclidean_inner(
                        &a[i..i + $name::lanes()],
                        &b[i..i + $name::lanes()],
                    );
                    i += $name::lanes();
                }

                let mut sum = sum.horizontal_add();
                if i < a.len() {
                    sum += Naive::squared_distance(&a[i..], &b[i..]);
                }
                sum
            }

            pub fn distance(a: &[$ty], b: &[$ty]) -> $ty {
                $name::squared_distance(a, b).sqrt()
            }
        }
    };
}

macro_rules! impl_naive {
    ($ty1:ty, $ty2:ty) => {
        impl Naive for &[$ty1] {
            type Output = $ty2;
            type Ty = $ty1;
            fn squared_distance(self, other: Self) -> $ty2 {
                assert_eq!(self.len(), other.len());

                let mut sum = 0 as $ty2;
                for i in 0..self.len() {
                    let d = self[i] - other[i];
                    sum += (d * d) as $ty2;
                }
                sum
            }

            fn distance(self, other: Self) -> $ty2 {
                Naive::squared_distance(self, other).sqrt()
            }
        }
        impl Naive for &Vec<$ty1> {
            type Output = $ty2;
            type Ty = $ty1;
            fn squared_distance(self, other: Self) -> $ty2 {
                assert_eq!(self.len(), other.len());

                let mut sum = 0 as $ty2;
                for i in 0..self.len() {
                    let d = self[i] - other[i];
                    sum += (d * d) as $ty2;
                }
                sum
            }

            fn distance(self, other: Self) -> $ty2 {
                Naive::squared_distance(self, other).sqrt()
            }
        }
    };
}
