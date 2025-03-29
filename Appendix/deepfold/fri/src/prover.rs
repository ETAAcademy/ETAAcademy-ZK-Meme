use super::verifier::Verifier;
use util::algebra::field::batch_inverse;
use util::algebra::polynomial::Polynomial;

use util::merkle_tree::MERKLE_ROOT_SIZE;
use util::query_result::QueryResult;
use util::{
    algebra::{coset::Coset, field::MyField},
    interpolation::InterpolateValue,
    random_oracle::RandomOracle,
};

#[derive(Clone)]
pub struct Prover<T: MyField> {
    total_round: usize,
    polynomial: Polynomial<T>,
    interpolate_cosets: Vec<Coset<T>>,
    interpolations: Vec<InterpolateValue<T>>,
    oracle: RandomOracle<T>,
    final_poly: Option<Polynomial<T>>,
    step: usize,
}

impl<T: MyField> Prover<T> {
    pub fn new(
        total_round: usize,
        interpolate_coset: &Vec<Coset<T>>, // L0, L1, ...
        polynomial: Polynomial<T>,
        oracle: &RandomOracle<T>,
        step: usize,
    ) -> Prover<T> {
        let interpolate_polynomial = InterpolateValue::new(
            interpolate_coset[0].fft(polynomial.coefficients().clone()),
            1 << step,
        );

        Prover {
            total_round,
            polynomial,
            interpolate_cosets: interpolate_coset.clone(),
            interpolations: vec![interpolate_polynomial],
            oracle: oracle.clone(),
            final_poly: None,
            step: step,
        }
    }

    pub fn commit_polynomial(&self) -> [u8; MERKLE_ROOT_SIZE] {
        self.interpolations[0].commit()
    }

    pub fn commit_foldings_multi_step(&self, verifier: &mut Verifier<T>) {
        // Todo: 边缘case检测
        for i in 1..self.total_round / self.step as usize + 1 {
            let interpolation = &self.interpolations[i];
            verifier.receive_interpolation_root(interpolation.leave_num(), interpolation.commit());
        }
        verifier.set_final_poly(self.final_poly.clone().unwrap());
    }

    fn evaluation_next_domain(
        &self,
        folding_value: &Vec<T>,
        round: usize,
        challenge: Vec<T>,
    ) -> Vec<T> {
        let mut res = vec![];
        let mut tmp_folding_value = folding_value.clone();

        for j in 0..self.step as usize {
            let coset = self.interpolate_cosets[self.step * round + j].clone();
            res = vec![];
            let mut new_v;
            let len = coset.size();
            for i in 0..(len / 2) {
                let x = tmp_folding_value[i];
                let nx = tmp_folding_value[i + len / 2];
                new_v = (x + nx) + challenge[j] * (x - nx) * coset.element_inv_at(i);
                res.push(new_v);
            }
            tmp_folding_value = res.clone();
            // Todo: init coset for every round outside bench
            // coset = coset.pow(2);
        }
        res
    }

    pub fn prove(&mut self, point: T) -> T {
        let mut res = None;
        for i in 0..self.total_round / self.step as usize {
            let mut challenge = vec![];
            for j in 0..self.step {
                challenge.push(self.oracle.folding_challenges[self.step * i + j])
            }
            // let challenge = self.oracle.folding_challenges[i];
            let next_evalutation = if i == 0 {
                let inv = batch_inverse(
                    &self.interpolate_cosets[0]
                        .all_elements()
                        .into_iter()
                        .map(|x| x - point)
                        .collect(),
                );
                res = Some(self.polynomial.evaluation_at(point));
                let v = self.interpolations[0].value.clone();
                // Cauchy: h(x) = (f(x)-v) * (x-z)^(-1)
                self.evaluation_next_domain(
                    &v.into_iter()
                        .zip(inv.into_iter())
                        .map(|(x, inv)| (x - res.unwrap()) * inv)
                        .collect(),
                    i,
                    challenge,
                )
            } else {
                self.evaluation_next_domain(&self.interpolations[i].value, i, challenge)
            };
            if i < self.total_round / self.step as usize - 1 {
                self.interpolations
                    .push(InterpolateValue::new(next_evalutation, 1 << self.step));
            } else {
                // self.final_value = Some(next_evalutation[0]);
                // self.final_values = Some(next_evalutation);
                self.interpolations.push(InterpolateValue::new(
                    next_evalutation.clone(),
                    1 << self.step,
                ));
                self.final_poly = Some(Polynomial::new(
                    self.interpolate_cosets[(i + 1) * self.step].ifft(next_evalutation.clone()),
                ));
            }
        }
        res.unwrap()
    }

    pub fn query(&self) -> Vec<QueryResult<T>> {
        let mut folding_res = vec![];
        let mut leaf_indices = self.oracle.query_list.clone();

        for i in 0..self.total_round / self.step + 1 {
            let len = self.interpolate_cosets[i * self.step].size();
            leaf_indices = leaf_indices
                .iter_mut()
                .map(|v| *v % (len >> self.step))
                .collect();
            leaf_indices.sort();
            leaf_indices.dedup();

            folding_res.push(self.interpolations[i].query(&leaf_indices));
        }
        folding_res
    }
}
