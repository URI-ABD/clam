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

macro_rules! impl_op16 {
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
                    self.8 $op rhs.8,
                    self.9 $op rhs.9,
                    self.10 $op rhs.10,
                    self.11 $op rhs.11,
                    self.12 $op rhs.12,
                    self.13 $op rhs.13,
                    self.14 $op rhs.14,
                    self.15 $op rhs.15,
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
                    self.8 $op rhs.8;
                    self.9 $op rhs.9;
                    self.10 $op rhs.10;
                    self.11 $op rhs.11;
                    self.12 $op rhs.12;
                    self.13 $op rhs.13;
                    self.14 $op rhs.14;
                    self.15 $op rhs.15;
            }
        }
    };
}

macro_rules! impl_distances {
    ($name:ident, $ty:ty) => {
        use super::Naive;
        impl $name {
            /// Calculate the squared distance between two SIMD lane-slices
            pub fn euclidean_inner(a: &[$ty], b: &[$ty]) -> $name {
                let i = $name::from_slice(a);
                let j = $name::from_slice(b);
                let u = i - j;
                u * u
            }

            /// Calculate the cosine accumulators (3) between two SIMD lane-slices
            pub fn cosine_inner(a: &[$ty], b: &[$ty]) -> [$name; 3] {
                let i = $name::from_slice(a);
                let j = $name::from_slice(b);
                [i * i, j * j, i * j]
            }

            /// Calculate euclidean distance between two slices of equal length,
            /// using auto-vectorized SIMD primitives
            pub fn squared_euclidean(a: &[$ty], b: &[$ty]) -> $ty {
                assert_eq!(a.len(), b.len());
                if a.len() < $name::lanes() {
                    return Naive::squared_euclidean(a, b);
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
                    sum += Naive::squared_euclidean(&a[i..], &b[i..]);
                }
                sum
            }

            pub fn euclidean(a: &[$ty], b: &[$ty]) -> $ty {
                $name::squared_euclidean(a, b).sqrt()
            }

            pub fn cosine_acc(a: &[$ty], b: &[$ty]) -> [$ty; 3] {
                assert_eq!(a.len(), b.len());
                if a.len() < $name::lanes() {
                    return Naive::cosine_acc(a, b);
                }
                let mut i = 0;
                let [mut xx, mut yy, mut xy] = [
                    $name::splat(0 as $ty),
                    $name::splat(0 as $ty),
                    $name::splat(0 as $ty),
                ];
                while a.len() - $name::lanes() >= i {
                    let [xxs, yys, xys] =
                        $name::cosine_inner(&a[i..i + $name::lanes()], &b[i..i + $name::lanes()]);
                    xx += xxs;
                    yy += yys;
                    xy += xys;
                    i += $name::lanes();
                }
                let mut xxsum = xx.horizontal_add();
                let mut yysum = yy.horizontal_add();
                let mut xysum = xy.horizontal_add();
                if i < a.len() {
                    let [xxs, yys, xys] = Naive::cosine_acc(&a[i..], &b[i..]);
                    xxsum += xxs;
                    yysum += yys;
                    xysum += xys;
                }
                [xxsum, yysum, xysum]
            }
            pub fn cosine(a: &[$ty], b: &[$ty]) -> $ty {
                let [xx, yy, xy] = $name::cosine_acc(a, b);
                let eps = <$ty>::EPSILON;
                if xx < eps || yy < eps || xy < eps {
                    1 as $ty
                } else {
                    let d = 1 as $ty - xy / (xx * yy).sqrt();
                    if d < eps {
                        0 as $ty
                    } else {
                        d
                    }
                }
            }
        }
    };
}

macro_rules! impl_naive {
    ($ty1:ty, $ty2:ty) => {
        impl Naive for &[$ty1] {
            type Output = $ty2;
            type Ty = $ty1;
            fn squared_euclidean(self, other: Self) -> Self::Output {
                assert_eq!(self.len(), other.len());

                let mut sum = 0 as Self::Output;
                for i in 0..self.len() {
                    let d = self[i] - other[i];
                    sum += (d * d) as Self::Output;
                }
                sum
            }

            fn euclidean(self, other: Self) -> Self::Output {
                Naive::squared_euclidean(self, other).sqrt()
            }

            fn cosine_acc(self, other: Self) -> [Self::Output; 3] {
                self.iter()
                    .zip(other.iter())
                    .fold([0 as Self::Output; 3], |[xx, yy, xy], (&a, &b)| {
                        [a.mul_add(a, xx), b.mul_add(b, yy), a.mul_add(b, xy)]
                    })
            }

            fn cosine(self, other: Self) -> Self::Output {
                let [xx, yy, xy] = Naive::cosine_acc(self, other);
                let eps = Self::Output::EPSILON;
                if xx < eps || yy < eps || xy < eps {
                    1 as Self::Output
                } else {
                    let d = 1 as Self::Output - xy / (xx * yy).sqrt();
                    if d < eps {
                        0 as Self::Output
                    } else {
                        d
                    }
                }
            }
        }
        impl Naive for &Vec<$ty1> {
            type Output = $ty2;
            type Ty = $ty1;
            fn squared_euclidean(self, other: Self) -> $ty2 {
                assert_eq!(self.len(), other.len());

                let mut sum = 0 as $ty2;
                for i in 0..self.len() {
                    let d = self[i] - other[i];
                    sum += (d * d) as $ty2;
                }
                sum
            }

            fn euclidean(self, other: Self) -> $ty2 {
                Naive::squared_euclidean(self, other).sqrt()
            }

            fn cosine_acc(self, other: Self) -> [Self::Output; 3] {
                self.iter()
                    .zip(other.iter())
                    .fold([0 as Self::Output; 3], |[xx, yy, xy], (&a, &b)| {
                        [a.mul_add(a, xx), b.mul_add(b, yy), a.mul_add(b, xy)]
                    })
            }

            fn cosine(self, other: Self) -> Self::Output {
                let [xx, yy, xy] = Naive::cosine_acc(self, other);
                let eps = Self::Output::EPSILON;
                if xx < eps || yy < eps || xy < eps {
                    1 as Self::Output
                } else {
                    let d = 1 as Self::Output - xy / (xx * yy).sqrt();
                    if d < eps {
                        0 as Self::Output
                    } else {
                        d
                    }
                }
            }
        }
    };
}
