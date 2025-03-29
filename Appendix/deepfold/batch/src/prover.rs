use util::{
    algebra::{coset::Coset, field::MyField, polynomial::MultilinearPolynomial},
    interpolation::InterpolateValue,
    query_result::QueryResult,
    random_oracle::RandomOracle,
};

use crate::{Commit, DeepEval, Proof};

#[derive(Clone)]
pub struct Prover<T: MyField> {
    total_round: usize,
    interpolate_cosets: Vec<Coset<T>>,
    function_interpolations: Vec<InterpolateValue<T>>,
    folding_interpolations: Vec<InterpolateValue<T>>,
    extra_v: Vec<T>,
    deep_eval: Vec<DeepEval<T>>,
    evals: Vec<Vec<T>>,
    polynomials: Vec<MultilinearPolynomial<T>>,
    shuffle_eval: Option<DeepEval<T>>,
    oracle: RandomOracle<T>,
    final_value: Option<T>,
}

impl<T: MyField> Prover<T> {
    pub fn new(
        total_round: usize,
        interpolate_cosets: &Vec<Coset<T>>,
        polynomials: Vec<MultilinearPolynomial<T>>,
        oracle: &RandomOracle<T>,
    ) -> Self {
        Prover {
            total_round,
            interpolate_cosets: interpolate_cosets.clone(),
            function_interpolations: interpolate_cosets
                .iter()
                .zip(polynomials.iter())
                .enumerate()
                .map(|(i, (set, poly))| {
                    InterpolateValue::new(
                        set.fft(poly.coefficients().clone()),
                        if i == 0 { 2 } else { 1 },
                    )
                })
                .collect(),
            folding_interpolations: vec![],
            deep_eval: vec![],
            extra_v: polynomials
                .iter()
                .enumerate()
                .map(|(i, x)| x.evaluate_as_polynomial(oracle.alpha[i]))
                .collect(),
            polynomials,
            evals: vec![],
            shuffle_eval: None,
            oracle: oracle.clone(),
            final_value: None,
        }
    }

    pub fn commit_polynomial(&self) -> Vec<Commit<T>> {
        (0..self.total_round)
            .into_iter()
            .map(|i| Commit {
                merkle_root: self.function_interpolations[i].commit(),
                deep: self.extra_v[i],
            })
            .collect()
    }

    fn evaluation_next_domain(&self, round: usize, challenge: T) -> Vec<T> {
        let mut res = vec![];
        let len = self.interpolate_cosets[round].size();
        let get_folding_value = if round > 0 {
            &self.folding_interpolations[round - 1].value
        } else {
            &self.function_interpolations[0].value
        };
        let coset = &self.interpolate_cosets[round];
        let challenge_2 = challenge * challenge;
        for i in 0..(len / 2) {
            let x = get_folding_value[i];
            let nx = get_folding_value[i + len / 2];
            let new_v = (x + nx) + challenge * (x - nx) * coset.element_inv_at(i);
            res.push(if round == self.total_round - 1 {
                new_v * T::inverse_2()
            } else {
                new_v * T::inverse_2()
                    + self.function_interpolations[round + 1].value[i] * challenge_2
            });
        }
        res
    }

    pub fn prove(&mut self, point: Vec<T>) {
        let mut polynomial = self.polynomials[0].clone();
        self.shuffle_eval = Some(DeepEval {
            point: point.clone(),
            first_eval: polynomial.evaluate(&point),
            else_evals: vec![],
        });
        self.evals.push(
            (1..self.total_round)
                .into_iter()
                .map(|i| self.polynomials[i].evaluate(&point[i..].to_vec()))
                .collect(),
        );
        for i in 0..self.total_round {
            self.shuffle_eval
                .as_mut()
                .unwrap()
                .append_else_eval(&polynomial);
            let mut alpha = self.oracle.deep[i];
            self.deep_eval.push({
                let deep_point = std::iter::successors(Some(alpha), |&x| Some(x * x))
                    .take(self.total_round - i)
                    .collect::<Vec<_>>();
                DeepEval {
                    point: deep_point,
                    first_eval: polynomial.evaluate_as_polynomial(alpha),
                    else_evals: vec![],
                }
            });
            self.evals.push({
                let mut res = vec![];
                for j in (i + 1)..self.total_round {
                    alpha *= alpha;
                    res.push(self.polynomials[j].evaluate_as_polynomial(alpha));
                }
                res
            });
            for deep in &mut self.deep_eval {
                deep.append_else_eval(&polynomial);
            }

            let challenge = self.oracle.folding_challenges[i];
            let next_evalutation = self.evaluation_next_domain(i, challenge);
            if i < self.total_round - 1 {
                polynomial.fold_self(challenge);
                polynomial.add_mult(&self.polynomials[i + 1], challenge * challenge);

                self.folding_interpolations
                    .push(InterpolateValue::new(next_evalutation, 2));
            } else {
                self.final_value = Some(next_evalutation[0]);
            }
        }
    }

    pub fn query(&self) -> (Vec<QueryResult<T>>, Vec<QueryResult<T>>) {
        let mut folding_res = vec![];
        let mut function_res = vec![];
        let mut leaf_indices = self.oracle.query_list.clone();

        for i in 0..self.total_round {
            let len = self.interpolate_cosets[i].size();
            leaf_indices = leaf_indices.iter_mut().map(|v| *v % (len >> 1)).collect();
            leaf_indices.sort();
            leaf_indices.dedup();
            if i == 0 {
                function_res.push(self.function_interpolations[i].query(&leaf_indices));
            } else {
                folding_res.push(self.folding_interpolations[i - 1].query(&leaf_indices));
            }
            if i != self.total_round - 1 {
                function_res.push(self.function_interpolations[i + 1].query(&leaf_indices));
            }
        }
        (folding_res, function_res)
    }

    pub fn generate_proof(mut self, point: Vec<T>) -> Proof<T> {
        self.prove(point);
        let query_result = self.query();
        Proof {
            merkle_root: (1..self.total_round)
                .into_iter()
                .map(|x| self.folding_interpolations[x - 1].commit())
                .collect(),
            query_result,
            deep_evals: self
                .deep_eval
                .iter()
                .map(|x| (x.first_eval, x.else_evals.clone()))
                .collect(),
            shuffle_evals: self.shuffle_eval.as_ref().unwrap().else_evals.clone(),
            final_value: self.final_value.unwrap(),
            evaluation: self.shuffle_eval.as_ref().unwrap().first_eval,
            out_evals: self.evals.clone(),
        }
    }
}
