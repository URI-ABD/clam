/// To account for floating-point imprecision
pub const EPSILON: f64 = 1e-12;

/// Sorts three values in non-ascending order
pub fn sort_three([a, b, c]: [f64; 3]) -> [f64; 3] {
    let (a, b) = if b > a { (b, a) } else { (a, b) };
    let (b, c) = if c > b { (c, b) } else { (b, c) };
    let (a, b) = if b > a { (b, a) } else { (a, b) };
    [a, b, c]
}

/// Checks that we have a valid non-zero length
pub fn check_length([a, b]: [char; 2], val: f64) -> Result<(), TriangleErr> {
    if val.abs() <= EPSILON || val.is_sign_negative() || val.is_nan() || val.is_infinite() {
        Err(TriangleErr::InvalidLength([a, b], val))
    } else {
        Ok(())
    }
}

/// The difference between largest value and the sum of the two other values
pub fn get_diff([ab, ac, bc]: [f64; 3]) -> f64 {
    let [ab, ac, bc] = sort_three([ab, ac, bc]);
    ab - ac - bc
}

pub fn are_colinear([ab, ac, bc]: [f64; 3]) -> bool {
    let diff = get_diff([ab, ac, bc]);
    diff.abs() < EPSILON
}

pub fn makes_triangle([ab, ac, bc]: [f64; 3]) -> bool {
    let [ab, ac, bc] = [ab, ac, bc];
    let diff = get_diff([ab, ac, bc]);
    diff.abs() >= EPSILON && diff < 0.
}

/// Checks that we can make a triangle with the given (valid) lengths.
pub fn check_triangle([a, b, c]: [char; 3], [ab, ac, bc]: [f64; 3]) -> Result<(), TriangleErr> {
    let diff = get_diff([ab, ac, bc]);
    if diff.abs() < EPSILON {
        Err(TriangleErr::ColinearVertices([a, b, c], [ab, ac, bc]))
    } else if diff > 0. {
        Err(TriangleErr::TriangleInequality([a, b, c], [ab, ac, bc]))
    } else {
        Ok(())
    }
}

/// Checks that we have a valid angle. `val` must be passed as either the
/// cosine or the sine of the angle being validated.
pub fn check_angle(a: char, val: f64) -> Result<(), TriangleErr> {
    if !(-1. ..=1.).contains(&val) {
        Err(TriangleErr::InvalidAngle(a, val))
    } else {
        Ok(())
    }
}

/// The geometry of a triangle in an arbitrary metric space.
#[derive(Debug, Clone)]
pub struct Triangle {
    ab: f64,
    bc: f64,
    ac: f64,
    ab_sq: f64,
    ac_sq: f64,
    bc_sq: f64,
    cos_a: f64,
    r_sq: f64,
    cm_sq: f64,
}

impl Triangle {
    /// A private constructor
    fn new_unchecked([ab, ac, bc, ab_sq, ac_sq, bc_sq, ab_ac, cos_a]: [f64; 8]) -> Self {
        Self {
            ab,
            bc,
            ac,
            ab_sq,
            ac_sq,
            bc_sq,
            cos_a,
            r_sq: bc_sq / (4. * (1. - cos_a * cos_a)),
            cm_sq: ac_sq + 0.25 * ab_sq - ab_ac * cos_a,
        }
    }

    /// Make a `Triangle` from three edge lengths that have already been
    /// validated.
    pub fn with_edges_unchecked([ab, ac, bc]: [f64; 3]) -> Self {
        let [ab_sq, ac_sq, bc_sq, ab_ac] = [ab * ab, ac * ac, bc * bc, ab * ac];
        let cos_a = (ab_sq + ac_sq - bc_sq) / (2. * ab_ac);
        Self::new_unchecked([ab, ac, bc, ab_sq, ac_sq, bc_sq, ab_ac, cos_a])
    }

    /// Make a `Triangle` from three edge lengths that need to be validated.
    pub fn with_edges([a, b, c]: [char; 3], [ab, ac, bc]: [f64; 3]) -> Result<Self, TriangleErr> {
        check_length([a, b], ab)?;
        check_length([a, c], ac)?;
        check_length([b, c], bc)?;
        check_triangle([a, b, c], [ab, ac, bc])?;
        Ok(Self::with_edges_unchecked([ab, ac, bc]))
    }

    fn with_cos_unchecked([cos_a, ab, ac]: [f64; 3]) -> Self {
        let [ab_sq, ac_sq, ab_ac] = [ab * ab, ac * ac, ab * ac];
        let bc_sq = ab_sq + ac_sq - 2. * ab_ac * cos_a;
        let bc = bc_sq.sqrt();
        Self::new_unchecked([ab, ac, bc, ab_sq, ac_sq, bc_sq, ab_ac, cos_a])
    }

    /// Make a `Triangle` from two edge lengths and cosine of their interior
    /// angle, all of which need to be validated. The cosine of the angle must
    /// be passed as the 0th float in the 3-slice.
    pub fn with_cos([a, b, c]: [char; 3], [cos_a, ab, ac]: [f64; 3]) -> Result<Self, TriangleErr> {
        check_angle('a', cos_a)?;
        check_length([a, b], ab)?;
        check_length([a, c], ac)?;
        Ok(Self::with_cos_unchecked([cos_a, ab, ac]))
    }

    pub fn edge_lengths(&self) -> [f64; 3] {
        [self.ab, self.ac, self.bc]
    }

    pub fn edge_lengths_sq(&self) -> [f64; 3] {
        [self.ab_sq, self.ac_sq, self.bc_sq]
    }

    pub fn cos_a(&self) -> f64 {
        self.cos_a
    }

    /// The squared circumradius.
    pub fn r_sq(&self) -> f64 {
        self.r_sq
    }

    /// The distance from the vertex `c` to the mid-point `m` of the edge `ab`.
    pub fn cm_sq(&self) -> f64 {
        self.cm_sq
    }
}

/// Helpful error wrapper for illegal lengths and triangles.
#[derive(Debug)]
pub enum TriangleErr {
    InvalidLength([char; 2], f64),
    InvalidAngle(char, f64),
    ColinearVertices([char; 3], [f64; 3]),
    TriangleInequality([char; 3], [f64; 3]),
}

impl std::fmt::Display for TriangleErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLength([a, b], val) => write!(f, "invalid length {a}{b}: {val:.12e}"),
            Self::InvalidAngle(a, val) => write!(f, "illegal angle {a}: {val:.12e}"),
            Self::ColinearVertices([a, b, c], [ab, ac, bc]) => write!(
                f,
                "Colinear vertices in {a}{b}{c} with edges [{ab:.12e}, {ac:.12e}, {bc:.12e}]"
            ),
            Self::TriangleInequality([a, b, c], [ab, ac, bc]) => write!(
                f,
                "Triangle Inequality violated in {a}{b}{c} with edges [{ab:.12e}, {ac:.12e}, {bc:.12e}]"
            ),
        }
    }
}

impl std::error::Error for TriangleErr {}
