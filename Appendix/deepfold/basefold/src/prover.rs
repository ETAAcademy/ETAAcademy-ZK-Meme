use util::{
    algebra::{
        coset::Coset,
        field::MyField,
        polynomial::{EqMultilinear, MultilinearPolynomial, Polynomial},
    },
    interpolation::InterpolateValue,
    merkle_tree::MERKLE_ROOT_SIZE,
    query_result::QueryResult,
    random_oracle::RandomOracle,
};

use crate::verifier::Verifier;

#[derive(Clone)]
pub struct Prover<T: MyField> {
    total_round: usize,
    interpolate_cosets: Vec<Coset<T>>,
    polynomial: MultilinearPolynomial<T>,
    interpolations: Vec<InterpolateValue<T>>,
    hypercube_interpolation: Vec<T>,
    sumcheck_value: Vec<(T, T, T)>,
    oracle: RandomOracle<T>,
    final_poly: Option<Polynomial<T>>,
    step: usize,
}

impl<T: MyField> Prover<T> {
    pub fn new(
        total_round: usize,
        interpolate_cosets: &Vec<Coset<T>>,
        polynomial: MultilinearPolynomial<T>,
        oracle: &RandomOracle<T>,
        step: usize,
    ) -> Self {
        Prover {
            total_round,
            interpolate_cosets: interpolate_cosets.clone(),
            interpolations: vec![InterpolateValue::new(
                interpolate_cosets[0].fft(polynomial.coefficients().clone()),
                1 << step,
            )],
            hypercube_interpolation: polynomial.evaluate_hypercube(),
            polynomial,
            sumcheck_value: vec![],
            oracle: oracle.clone(),
            final_poly: None,
            step,
        }
    }

    pub fn commit_polynomial(&self) -> [u8; MERKLE_ROOT_SIZE] {
        self.interpolations[0].commit()
    }

    pub fn commit_foldings(&self, verifier: &mut Verifier<T>) {
        for i in 1..self.total_round / self.step {
            let interpolation = &self.interpolations[i];
            verifier.receive_folding_root(interpolation.leave_num(), interpolation.commit());
        }
        for i in &self.sumcheck_value {
            verifier.receive_sumcheck_value(i.clone());
        }
        verifier.set_final_poly(self.final_poly.clone().unwrap());
    }

    pub fn send_evaluation(&self, verifier: &mut Verifier<T>, point: &Vec<T>) {
        verifier.set_evalutation(self.polynomial.evaluate(point));
    }

    fn evaluation_next_domain(&self, round: usize, challenges: Vec<T>) -> Vec<T> {
        let mut get_folding_value = self.interpolations[round].value.clone();
        for j in 0..self.step {
            let len = self.interpolate_cosets[round * self.step + j].size();
            let coset = &self.interpolate_cosets[round * self.step + j];
            let mut tmp_folding_value = vec![];
            let challenge = challenges[j];
            for i in 0..(len / 2) {
                let x = get_folding_value[i];
                let nx = get_folding_value[i + len / 2];
                let new_v = (x + nx) + challenge * (x - nx) * coset.element_inv_at(i);
                tmp_folding_value.push(new_v * T::inverse_2());
            }
            get_folding_value = tmp_folding_value;
        }
        get_folding_value
    }

    fn sumcheck_next_domain(hypercube_interpolation: &mut Vec<T>, m: usize, challenge: T) {
        for i in 0..m {
            hypercube_interpolation[i] *= T::from_int(1) - challenge;
            let tmp = hypercube_interpolation[i + m] * challenge;
            hypercube_interpolation[i] += tmp;
        }
    }

    pub fn prove(&mut self, point: &Vec<T>) {
        let mut poly_hypercube = self.hypercube_interpolation.clone();
        let mut eq_hypercube = EqMultilinear::new(point.clone()).evaluate_hypercube();
        for i in 0..self.total_round / self.step {
            let mut challenges = vec![];
            // step 1 calculate the sumcheck part and sends to v
            for j in 0..self.step {
                let m = 1 << (self.total_round - (i * self.step + j) - 1);
                let challenge = self.oracle.folding_challenges[i * self.step + j];
                let (sum_0, sum_1, sum_2) = (0..m).into_iter().fold(
                    (T::from_int(0), T::from_int(0), T::from_int(0)),
                    |acc, x| {
                        let p_0 = poly_hypercube[x];
                        let p_1 = poly_hypercube[x + m];
                        let e_0 = eq_hypercube[x];
                        let e_1 = eq_hypercube[x + m];
                        (
                            acc.0 + p_0 * e_0,
                            acc.1 + p_1 * e_1,
                            acc.2 + (p_1 + p_1 - p_0) * (e_1 + e_1 - e_0),
                        )
                    },
                );
                self.sumcheck_value.push((sum_0, sum_1, sum_2));

                Self::sumcheck_next_domain(&mut poly_hypercube, m, challenge);
                Self::sumcheck_next_domain(&mut eq_hypercube, m, challenge);
                challenges.push(challenge);
            }

            // step 2 calculate the folding phase
            let next_evalutation = self.evaluation_next_domain(i, challenges);
            if i < self.total_round / self.step - 1 {
                self.interpolations
                    .push(InterpolateValue::new(next_evalutation, 1 << self.step));
            } else {
                self.interpolations.push(InterpolateValue::new(
                    next_evalutation.clone(),
                    1 << self.step,
                ));
                self.final_poly = Some(Polynomial::new(
                    self.interpolate_cosets[(i + 1) * self.step].ifft(next_evalutation),
                ));
            }
        }
    }

    pub fn query(&self) -> Vec<QueryResult<T>> {
        let mut res = vec![];
        let mut leaf_indices = self.oracle.query_list.clone();

        for i in 0..self.total_round / self.step + 1 {
            let len = self.interpolate_cosets[i * self.step].size();
            leaf_indices = leaf_indices
                .iter_mut()
                .map(|v| *v % (len >> self.step))
                .collect();
            leaf_indices.sort();
            leaf_indices.dedup();
            res.push(self.interpolations[i].query(&leaf_indices));
        }
        res
    }
}
