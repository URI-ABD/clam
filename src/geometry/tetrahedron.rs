use super::triangle::*;

#[derive(Debug)]
pub struct Tetrahedron {
    ab: f64,
    ac: f64,
    bc: f64,
    ad: f64,
    bd: f64,
    cd: f64,
    r: f64,
    ab_sq: f64,
    ac_sq: f64,
    bc_sq: f64,
    ad_sq: f64,
    bd_sq: f64,
    cd_sq: f64,
    r_sq: f64,
    abc: Triangle,
    od_sq: Option<f64>,
    od: Option<f64>,
}

impl Tetrahedron {
    // TODO: Do this with &Triangle and no cloning
    pub fn with_triangle_unchecked(abc: Triangle, [ad, bd, cd]: [f64; 3]) -> Self {
        let [ab, ac, bc] = abc.edge_lengths();
        let [ab_sq, ac_sq, bc_sq] = abc.edge_lengths_sq();

        Self {
            ab,
            ac,
            bc,
            ad,
            bd,
            cd,
            r: abc.r_sq().sqrt(),
            ab_sq,
            ac_sq,
            bc_sq,
            ad_sq: ad * ad,
            bd_sq: bd * bd,
            cd_sq: cd * cd,
            r_sq: abc.r_sq(),
            abc,
            od_sq: None,
            od: None,
        }
    }

    pub fn with_edges_unchecked([ab, ac, bc, ad, bd, cd]: [f64; 6]) -> Self {
        // let abc = Triangle::with_edges_unchecked([ab, ac, bc]);
        let abc = Triangle::with_edges_unchecked([ab, ac, bc]);
        Self::with_triangle_unchecked(abc, [ad, bd, cd])
    }

    pub fn with_edges([a, b, c, d]: [char; 4], [ab, ac, bc, ad, bd, cd]: [f64; 6]) -> Result<Self, TriangleErr> {
        check_length([a, b], ab)?;
        check_length([a, c], ac)?;
        check_length([b, c], bc)?;
        check_length([a, d], ad)?;
        check_length([b, d], bd)?;
        check_length([c, d], cd)?;
        check_triangle([a, b, c], [ab, ac, bc])?;
        check_triangle([a, b, d], [ab, ad, bd])?;
        check_triangle([a, c, d], [ac, ad, cd])?;
        check_triangle([b, c, d], [bc, bd, cd])?;

        Ok(Self::with_edges_unchecked([ab, ac, bc, ad, bd, cd]))
    }

    pub fn edge_lengths(&self) -> [f64; 6] {
        [self.ab, self.ac, self.bc, self.ad, self.bd, self.cd]
    }

    pub fn edge_lengths_sq(&self) -> [f64; 6] {
        [self.ab_sq, self.ac_sq, self.bc_sq, self.ad_sq, self.bd_sq, self.cd_sq]
    }

    fn case_1(&self) -> Option<f64> {
        let [ab_sq, ac_sq, _, ad_sq, bd_sq, cd_sq] = self.edge_lengths_sq();

        let pd_sq = if (bd_sq - ab_sq - ad_sq).abs() < EPSILON {
            // D projects to A
            Some(ad_sq)
        } else if (ad_sq - ab_sq - bd_sq).abs() < EPSILON {
            // D projects to B
            Some(bd_sq)
        } else if (ad_sq - ac_sq - cd_sq).abs() < EPSILON {
            // D projects to C
            Some(cd_sq)
        } else {
            // Not case 1
            None
        };
        pd_sq.map(|pd_sq: f64| self.r_sq + pd_sq)
    }

    // fn case_2_ab(&self, [ab, ad, bd]: [f64; 3]) -> f64 {
    //     if bd > ab && bd > ad {
    //         self.r_sq + ab * bd
    //     } else {
    //         self.r_sq + ad * (ad - ab)
    //     }
    // }

    // #[inline(never)]
    // fn case_2(&self) -> Option<f64> {
    //     let [ab, ac, bc, ad, bd, cd] = self.edge_lengths();
    //     if are_colinear([ab, ad, bd]) {
    //         Some(self.case_2_ab([ab, ad, bd]))
    //     } else if are_colinear([ac, ad, cd]) {
    //         Some(self.case_2_ab([ac, ad, cd]))
    //     } else if are_colinear([bc, cd, bd]) {
    //         Some(self.case_2_ab([bc, cd, bd]))
    //     } else {
    //         None
    //     }
    // }

    fn case_3(&self) -> f64 {
        let [ab, ac, _, ad, _, _] = self.edge_lengths();
        let [ab_sq, ac_sq, _, ad_sq, bd_sq, cd_sq] = self.edge_lengths_sq();
        let [r, r_sq] = [self.r, self.r_sq];
        let cos_a = self.abc.cos_a();
        let sin_a_sq = 1. - cos_a * cos_a;

        // compute the dihedral angle at the edge AB:
        // https://math.stackexchange.com/questions/314970/dihedral-angles-between-tetrahedron-faces-from-triangles-angles-at-the-tip
        let cos_acd = (ad_sq + ac_sq - cd_sq) / (2. * ad * ac);
        let cos_bad = (ad_sq + ab_sq - bd_sq) / (2. * ad * ab);
        let sin_bad_sq = 1. - cos_bad * cos_bad;

        let numerator = cos_acd - cos_bad * cos_a;
        if numerator.abs() < EPSILON {
            // case 3a: D projects onto the line AB.
            let dq_sq = ad_sq * sin_bad_sq;
            let aq = ad * cos_bad;
            let bq = ab - aq;
            let aq = aq.abs();

            let oq_sq = if bq > ab && bq > aq {
                r_sq + ab * bq
            } else {
                r_sq + aq * (aq - ab)
            };

            oq_sq + dq_sq
        } else {
            let cos_ab_sq = numerator * numerator / sin_bad_sq * sin_a_sq;
            let od_sq = if cos_ab_sq < 1. {
                // case 3d
                ad_sq * sin_bad_sq * (1. - cos_ab_sq)
            } else {
                // cases 3b and 3c
                let cos_oab = ab / (2. * r);
                let sin_oab_sq = 1. - cos_oab * cos_oab;
                let cos_oad = cos_a * cos_oab + numerator.signum() * (sin_oab_sq * sin_a_sq).sqrt();
                r_sq + ad_sq - 2. * r * ad * cos_oad
            };
            assert!(!od_sq.is_sign_negative(), "\n`od_sq` was negative: {od_sq:.12}, \ncos_ab_sq={cos_ab_sq:.12}, \nsin_bad_sq={sin_bad_sq:.12}, \ncos_bad={cos_bad:.12}, \nsin_a_sq={sin_a_sq:.12}, \ncos_a={cos_a:.12} \nabc={:?}!\n", self.abc);
            od_sq
        }
    }

    pub fn od_sq(&mut self) -> f64 {
        if let Some(od_sq) = self.od_sq {
            od_sq
        } else {
            let od_sq = if let Some(od_sq) = self.case_1() {
                od_sq
            // } else if let Some(od_sq) = self.case_2() {
            //     od_sq
            } else {
                self.case_3()
            };
            assert!(!od_sq.is_sign_negative(), "`od_sq` was negative: {od_sq:.12}");
            self.od_sq = Some(od_sq);
            od_sq
        }
    }

    pub fn od(&mut self) -> f64 {
        if let Some(od) = self.od {
            od
        } else {
            let od = self.od_sq().sqrt();
            self.od = Some(od);
            od
        }
    }
}
