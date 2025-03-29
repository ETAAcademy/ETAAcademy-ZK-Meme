use std::ops::{Add, Mul};

use batch::{prover::Prover as BatchProver, verifier::Verifier as BatchVerifier};
use deepfold::{prover::Prover as DeepfoldProver, verifier::Verifier as DeepfoldVerifier};
use rand::{thread_rng, Rng};
use util::{
    algebra::{coset::Coset, field::MyField, polynomial::MultilinearPolynomial},
    random_oracle::RandomOracle,
    sumcheck::{prover::SumcheckProver, verifier::SumcheckVerifier},
};

use util::{CODE_RATE, SECURITY_BITS};

#[derive(Clone)]
pub struct Matrix<T: MyField> {
    rows: Vec<Vec<T>>,
    padded_rows: Vec<Vec<T>>,
    splitted_matrices: Vec<Vec<Vec<T>>>,
    dom_row_size: usize,
    dom_col_size: usize,
    polys: Vec<Vec<MultilinearPolynomial<T>>>,
    naive_poly: MultilinearPolynomial<T>,
}

impl<T: MyField> Matrix<T> {
    pub fn new(entries: Vec<Vec<T>>) -> Self {
        let (dom_row_size, _sub) = find_dom_power_of_2(entries.len());
        let (dom_col_size, _sub) = find_dom_power_of_2(entries[0].len());

        let padded_row_size = find_power_of_2(entries.len());
        let padded_col_size = find_power_of_2(entries[0].len());

        let padded_rows = pad(&entries, padded_row_size, padded_col_size);

        let mut mat = Matrix {
            rows: entries.clone(),
            padded_rows: padded_rows.clone(),
            splitted_matrices: vec![],
            dom_row_size,
            dom_col_size,
            polys: vec![],
            naive_poly: MultilinearPolynomial::new(padded_rows.into_iter().flatten().collect()),
        };

        mat.padding_and_split();
        mat.to_poly();

        mat
    }

    pub fn sample(row_size: usize, col_size: usize) -> Self {
        let mut mat = Vec::with_capacity(row_size);
        for _ in 0..row_size {
            let mut row = Vec::with_capacity(col_size);
            for _ in 0..col_size {
                row.push(T::random_element());
            }
            mat.push(row);
        }
        Self::new(mat)
    }

    pub fn get_row_size(&self) -> usize {
        return self.rows.len();
    }

    pub fn get_padded_row_size(&self) -> usize {
        return self.padded_rows.len();
    }

    pub fn get_col_size(&self) -> usize {
        return self.rows[0].len();
    }

    pub fn get_padded_col_size(&self) -> usize {
        return self.padded_rows[0].len();
    }

    pub fn get_entry(&self, i: usize, j: usize) -> T {
        self.rows[i][j]
    }

    fn padding_and_split(&mut self) {
        let mut padded_mat = self.rows.clone();
        let row_size = padded_mat.len();
        let col_size = padded_mat[0].len();

        let (dom_row_size, sub_row_size) = find_dom_power_of_2(row_size);
        let (dom_col_size, sub_col_size) = find_dom_power_of_2(col_size);
        self.dom_row_size = dom_row_size;
        self.dom_col_size = dom_col_size;

        for i in 0..row_size {
            while padded_mat[i].len() < dom_col_size + sub_col_size {
                padded_mat[i].push(T::from_int(0));
            }
        }

        while padded_mat.len() < dom_row_size + sub_row_size {
            padded_mat.push(vec![T::from_int(0); dom_col_size + sub_col_size]);
        }

        self.splitted_matrices = vec![
            extract_submatrix(&padded_mat, 0..dom_row_size, 0..dom_col_size),
            extract_submatrix(
                &padded_mat,
                0..dom_row_size,
                dom_col_size..dom_col_size + sub_col_size,
            ),
            extract_submatrix(
                &padded_mat,
                dom_row_size..dom_row_size + sub_row_size,
                0..dom_col_size,
            ),
            extract_submatrix(
                &padded_mat,
                dom_row_size..dom_row_size + sub_row_size,
                dom_col_size..dom_col_size + sub_col_size,
            ),
        ];
    }

    pub fn to_poly(&mut self) {
        self.polys = vec![
            vec![
                MultilinearPolynomial::new(
                    self.splitted_matrices[0]
                        .clone()
                        .into_iter()
                        .flatten()
                        .collect(),
                ),
                MultilinearPolynomial::new(
                    self.splitted_matrices[1]
                        .clone()
                        .into_iter()
                        .flatten()
                        .collect(),
                ),
            ],
            vec![
                MultilinearPolynomial::new(
                    self.splitted_matrices[2]
                        .clone()
                        .into_iter()
                        .flatten()
                        .collect(),
                ),
                MultilinearPolynomial::new(
                    self.splitted_matrices[3]
                        .clone()
                        .into_iter()
                        .flatten()
                        .collect(),
                ),
            ],
        ];
    }

    pub fn mask_poly(&self, mask_size: usize) -> MultilinearPolynomial<T> {
        MultilinearPolynomial::new(self.naive_poly.coefficients()[0..1 << mask_size].to_vec())
    }

    pub fn evalutate_as_poly(&self, point: Vec<T>) -> T {
        self.naive_poly.evaluate(&point)
    }
}

impl<T: MyField> Mul for Matrix<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(matrix_multiply(&self.rows, &rhs.rows))
    }
}

pub fn mat_mult<T: MyField>(a: &Matrix<T>, b: &Matrix<T>, c: &Matrix<T>) -> usize {
    // 1. Check random element of C: sample r1, r2 and get the value of C(r1, r2)
    let mut rng = thread_rng();
    let r_1 = rng.gen_range(0..c.get_row_size());
    let r_2 = rng.gen_range(0..c.get_col_size());

    let _claimed_value_c = c.get_entry(r_1, r_2);

    // 2. do sumcheck on polyA * polyB = polyC, polyA and polyB are combinations of submatrices' polys
    let poly_a = a.naive_poly.clone();
    let poly_b = b.naive_poly.clone();

    let variable_number = a.get_padded_col_size().ilog2() as usize;
    let oracle = RandomOracle::new(variable_number, 1);

    let f = a.mask_poly(variable_number);
    let mut sc_prover = SumcheckProver::new(variable_number, f, &oracle);
    let mut sc_verifier = SumcheckVerifier::new(variable_number, &oracle);
    sc_prover.prove();
    sc_prover.send_sumcheck_values(&mut sc_verifier);
    let (challenge, _scalar) = sc_verifier.verify();

    // 3. the final output of polyA and polyB, C(r1, r2) should be fed into a batch deepfold
    let mut point_a = int_2_field_vec(r_1 as u64, a.get_padded_row_size().ilog2() as usize).clone();
    point_a.extend(&challenge);
    let mut point_b = challenge.clone();
    point_b.extend(int_2_field_vec::<T>(
        r_2 as u64,
        b.get_padded_col_size().ilog2() as usize,
    ));

    let _claimed_value_a = poly_a.evaluate(&point_a);
    let _claimed_value_b = poly_b.evaluate(&point_b);

    // let flatten_polys = vec![
    //     a.polys
    //         .clone()
    //         .into_iter()
    //         .flatten()
    //         .collect::<Vec<MultilinearPolynomial<T>>>(),
    //     b.polys
    //         .clone()
    //         .into_iter()
    //         .flatten()
    //         .collect::<Vec<MultilinearPolynomial<T>>>(),
    // ];
    let mut full_polys = a
        .polys
        .clone()
        .into_iter()
        .flatten()
        .collect::<Vec<MultilinearPolynomial<T>>>();
    full_polys.extend(b.polys.clone().into_iter().flatten());
    // random shuffle for polys to assign a shared evaluate point

    let random_shuffle_poly = full_polys.clone().into_iter().reduce(|f, g| f + g).unwrap();
    let sc_vn = random_shuffle_poly.variable_num();
    let oracle = RandomOracle::new(sc_vn, 1);
    let mut sc_prover = SumcheckProver::new(sc_vn, random_shuffle_poly, &oracle);
    let mut sc_verifier = SumcheckVerifier::new(sc_vn, &oracle);
    sc_prover.prove();
    sc_prover.send_sumcheck_values(&mut sc_verifier);
    let (df_point, _scalar) = sc_verifier.verify();

    let full_polys = sort_and_fill_polys(&mut full_polys, T::random_element());
    let df_vn = full_polys.first().map_or(0, |p| p.variable_num());
    let mut interpolate_cosets = vec![Coset::new(1 << (df_vn + CODE_RATE), T::from_int(1))];
    for i in 1..df_vn {
        interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
    }

    let oracle = RandomOracle::new(df_vn, SECURITY_BITS / CODE_RATE);
    let df_prover = BatchProver::new(df_vn, &interpolate_cosets, full_polys, &oracle);
    let commit = df_prover.commit_polynomial();
    let mut df_verifier = BatchVerifier::new(df_vn, &interpolate_cosets, commit, &oracle);
    // let point = get_random_shuffle_point::<T>(df_vn);
    // let point = df_verifier.get_open_point();
    df_verifier.set_open_point(&df_point);
    let proof = df_prover.generate_proof(df_point);
    let size = proof.size();
    assert!(df_verifier.verify(proof));
    size
}

pub fn naive_opening<T: MyField>(a: &Matrix<T>, b: &Matrix<T>, c: &Matrix<T>) -> usize {
    let mut rng = thread_rng();
    let r_1 = rng.gen_range(0..c.get_row_size());
    let r_2 = rng.gen_range(0..c.get_col_size());

    let poly_a = a.naive_poly.clone();
    let poly_b = b.naive_poly.clone();
    let poly_c = c.naive_poly.clone();

    let variable_number = a.get_padded_col_size().ilog2() as usize;
    let oracle = RandomOracle::new(variable_number, 1);

    let f = a.mask_poly(variable_number);
    let mut sc_prover = SumcheckProver::new(variable_number, f, &oracle);
    let mut sc_verifier = SumcheckVerifier::new(variable_number, &oracle);
    sc_prover.prove();
    sc_prover.send_sumcheck_values(&mut sc_verifier);
    let (challenge, _scalar) = sc_verifier.verify();

    let mut point_a =
        int_2_field_vec::<T>(r_1 as u64, a.get_padded_row_size().ilog2() as usize).clone();
    point_a.extend(&challenge);
    let mut point_b = challenge.clone();
    point_b.extend(int_2_field_vec::<T>(
        r_2 as u64,
        b.get_padded_col_size().ilog2() as usize,
    ));
    let mut point_c =
        int_2_field_vec::<T>(r_1 as u64, a.get_padded_row_size().ilog2() as usize).clone();
    point_c.extend(int_2_field_vec::<T>(
        r_2 as u64,
        b.get_padded_col_size().ilog2() as usize,
    ));

    let mut full_polys = a
        .polys
        .clone()
        .into_iter()
        .flatten()
        .collect::<Vec<MultilinearPolynomial<T>>>();
    full_polys.extend(b.polys.clone().into_iter().flatten());
    let mut total_size = 0;
    for p in full_polys {
        let df_vn = p.variable_num();
        let mut interpolate_cosets = vec![Coset::new(1 << (df_vn + CODE_RATE), T::from_int(1))];
        for i in 1..df_vn + 1 {
            interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
        }

        let oracle = RandomOracle::new(df_vn, SECURITY_BITS / CODE_RATE);
        let df_prover = DeepfoldProver::new(df_vn, &interpolate_cosets, p, &oracle, 1);
        let commit = df_prover.commit_polynomial();
        let df_verifier = DeepfoldVerifier::new(df_vn, &interpolate_cosets, commit, &oracle, 1);
        // let point = get_random_shuffle_point::<T>(df_vn);
        let point = df_verifier.get_open_point();
        let proof = df_prover.generate_proof(point);
        let size = proof.size();
        total_size += size;
        assert!(df_verifier.verify(proof));
    }
    total_size
}

pub fn naive_mat_mult<T: MyField>(a: &Matrix<T>, b: &Matrix<T>, c: &Matrix<T>) -> usize {
    let mut rng = thread_rng();
    let r_1 = rng.gen_range(0..c.get_row_size());
    let r_2 = rng.gen_range(0..c.get_col_size());

    let poly_a = a.naive_poly.clone();
    let poly_b = b.naive_poly.clone();
    let poly_c = c.naive_poly.clone();

    let variable_number = a.get_padded_col_size().ilog2() as usize;
    let oracle = RandomOracle::new(variable_number, 1);

    let f = a.mask_poly(variable_number);
    let mut sc_prover = SumcheckProver::new(variable_number, f, &oracle);
    let mut sc_verifier = SumcheckVerifier::new(variable_number, &oracle);
    sc_prover.prove();
    sc_prover.send_sumcheck_values(&mut sc_verifier);
    let (challenge, _scalar) = sc_verifier.verify();

    let mut point_a =
        int_2_field_vec::<T>(r_1 as u64, a.get_padded_row_size().ilog2() as usize).clone();
    point_a.extend(&challenge);
    let mut point_b = challenge.clone();
    point_b.extend(int_2_field_vec::<T>(
        r_2 as u64,
        b.get_padded_col_size().ilog2() as usize,
    ));
    let mut point_c =
        int_2_field_vec::<T>(r_1 as u64, a.get_padded_row_size().ilog2() as usize).clone();
    point_c.extend(int_2_field_vec::<T>(
        r_2 as u64,
        b.get_padded_col_size().ilog2() as usize,
    ));

    // let polys = vec![poly_a, poly_b, poly_c];
    let polys = vec![poly_a, poly_b];
    // let points = vec![point_a, point_b, point_c];
    let points = vec![point_a, point_b];
    let mut total_size = 0;
    for i in 0..polys.len() {
        let poly = polys[i].clone();
        let point = points[i].clone();
        let df_vn = poly.variable_num();
        let mut interpolate_cosets = vec![Coset::new(1 << (df_vn + CODE_RATE), T::from_int(1))];
        for i in 1..df_vn + 1 {
            interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
        }

        let oracle = RandomOracle::new(df_vn, SECURITY_BITS / CODE_RATE);
        let df_prover = DeepfoldProver::new(df_vn, &interpolate_cosets, poly, &oracle, 1);
        let commit = df_prover.commit_polynomial();
        let mut df_verifier = DeepfoldVerifier::new(df_vn, &interpolate_cosets, commit, &oracle, 1);
        // let point = get_random_shuffle_point::<T>(df_vn);
        // let point = df_verifier.get_open_point();
        df_verifier.set_open_point(&point);
        let proof = df_prover.generate_proof(point);
        let size = proof.size();
        total_size += size;
        assert!(df_verifier.verify(proof));
    }
    // let random_shuffle_poly = vec![poly_a.clone(), poly_b.clone(), poly_c.clone()]
    //     .into_iter()
    //     .reduce(|f, g| f + g)
    //     .unwrap();
    // let sc_vn = random_shuffle_poly.variable_num();
    // let oracle = RandomOracle::new(sc_vn, 1);
    // let mut sc_prover = SumcheckProver::new(sc_vn, random_shuffle_poly, &oracle);
    // let mut sc_verifier = SumcheckVerifier::new(sc_vn, &oracle);
    // sc_prover.prove();
    // sc_prover.send_sumcheck_values(&mut sc_verifier);
    // let (df_point, _scalar) = sc_verifier.verify();

    // let random_comb = T::random_element();
    // let df_poly = vec![poly_a, poly_b, poly_c]
    //     .into_iter()
    //     .reduce(|f, g| f * random_comb + g)
    //     .unwrap();

    total_size
}

fn pad<T: MyField>(rows: &Vec<Vec<T>>, to_row_size: usize, to_col_size: usize) -> Vec<Vec<T>> {
    let mut mat = rows.clone();

    for i in 0..rows.len() {
        while mat[i].len() < to_col_size {
            mat[i].push(T::from_int(0));
        }
    }

    while mat.len() < to_row_size {
        mat.push(vec![T::from_int(0); to_col_size]);
    }

    mat
}

fn find_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }

    let mut current_power = 1;
    while current_power < n {
        current_power <<= 1;
    }

    current_power
}

fn find_dom_power_of_2(n: usize) -> (usize, usize) {
    let mut pow = 0;
    while 1 << pow <= n {
        pow += 1;
    }
    pow -= 1;
    let dom = 1 << pow;
    let res = n - dom;

    let mut pow = 0;
    while 1 << pow < res {
        pow += 1;
    }
    let sub = 1 << pow;
    (dom, sub)
}

fn int_2_field_vec<T: MyField>(n: u64, size: usize) -> Vec<T> {
    let mut res = Vec::new();

    let mut bits = if n == 0 {
        1
    } else {
        n.trailing_zeros() as usize + 1
    };

    while bits > 0 {
        res.push(T::from_int((n >> bits - 1) & 1));
        bits -= 1;
    }

    while res.len() < size {
        res.push(T::from_int(0));
    }

    res
}

fn extract_submatrix<T: MyField>(
    mat: &Vec<Vec<T>>,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) -> Vec<Vec<T>> {
    let mut submatrix = Vec::new();

    for row_index in rows {
        if let Some(row) = mat.get(row_index) {
            let mut subrow = Vec::new();
            for col_index in cols.clone().into_iter() {
                if let Some(&value) = row.get(col_index) {
                    subrow.push(value);
                }
            }
            submatrix.push(subrow);
        }
    }

    submatrix
}

fn sort_and_fill_polys<T: MyField>(
    polys: &mut Vec<MultilinearPolynomial<T>>,
    r: T,
) -> Vec<MultilinearPolynomial<T>> {
    polys.sort_by_key(|p| p.variable_num());
    polys.reverse();

    let mut combined_polys = vec![];
    let mut current_ind = 0;
    // combine polys with same degree
    while current_ind < polys.len() {
        let mut tmp_poly_coeffs = polys[current_ind].coefficients().clone();
        let degree = polys[current_ind].variable_num();
        let mut w = r;
        for j in current_ind + 1..polys.len() + 1 {
            if j < polys.len() && polys[j].variable_num() == degree {
                tmp_poly_coeffs = tmp_poly_coeffs
                    .iter()
                    .zip(polys[j].coefficients().iter())
                    .map(|(&f, &g)| f + w * g)
                    .collect();
                w *= r;
            } else {
                current_ind = j;
                break;
            }
        }
        combined_polys.push(MultilinearPolynomial::new(tmp_poly_coeffs));
    }

    // add zero polys
    let mut full_polys = vec![];
    let mut current_deg = combined_polys.first().map_or(0, |p| p.variable_num());
    let mut current_ind = 0;
    while current_deg > 0 {
        if current_ind < combined_polys.len()
            && combined_polys[current_ind].variable_num() == current_deg
        {
            full_polys.push(combined_polys[current_ind].clone());
            current_ind += 1;
        } else {
            full_polys.push(MultilinearPolynomial::new(vec![
                T::from_int(0);
                1 << current_deg
            ]));
        }
        current_deg -= 1;
    }

    full_polys
}

fn matrix_multiply<T: MyField>(a: &Vec<Vec<T>>, b: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let rows_a = a.len();
    let cols_a = a[0].len();
    let rows_b = b.len();
    let cols_b = b[0].len();

    if cols_a != rows_b {
        panic!(
            "Matrix multiplication failed: number of columns in A must equal number of rows in B."
        );
    }

    let mut result = vec![vec![T::from_int(0); cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            for k in 0..cols_a {
                result[i][j] = result[i][j] + a[i][k] * b[k][j];
            }
        }
    }

    result
}
