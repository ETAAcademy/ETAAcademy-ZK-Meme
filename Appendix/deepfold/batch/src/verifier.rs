use util::random_oracle::RandomOracle;
use util::{
    algebra::{coset::Coset, field::MyField},
    merkle_tree::MerkleTreeVerifier,
};

use crate::{Commit, DeepEval, Proof};

#[derive(Clone)]
pub struct Verifier<T: MyField> {
    total_round: usize,
    interpolate_cosets: Vec<Coset<T>>,
    polynomial_roots: Vec<MerkleTreeVerifier>,
    folding_roots: Vec<MerkleTreeVerifier>,
    first_deep: Vec<T>,
    oracle: RandomOracle<T>,
    final_value: Option<T>,
    shuffle_eval: Option<DeepEval<T>>,
    deep_evals: Vec<DeepEval<T>>,
    open_point: Vec<T>,
}

impl<T: MyField> Verifier<T> {
    pub fn new(
        total_round: usize,
        coset: &Vec<Coset<T>>,
        commit: Vec<Commit<T>>,
        oracle: &RandomOracle<T>,
    ) -> Self {
        Verifier {
            total_round,
            interpolate_cosets: coset.clone(),
            oracle: oracle.clone(),
            polynomial_roots: commit
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    MerkleTreeVerifier::new(
                        if i == 0 {
                            coset[i].size() / 2
                        } else {
                            coset[i].size()
                        },
                        &c.merkle_root,
                    )
                })
                .collect(),
            folding_roots: vec![],
            first_deep: commit.iter().map(|x| x.deep).collect(),
            final_value: None,
            shuffle_eval: None,
            deep_evals: vec![],
            open_point: (0..total_round).map(|_| T::random_element()).collect(),
        }
    }

    pub fn get_open_point(&self) -> Vec<T> {
        self.open_point.clone()
    }

    pub fn set_open_point(&mut self, point: &Vec<T>) {
        self.open_point = point.clone();
    }

    pub fn verify(mut self, proof: Proof<T>) -> bool {
        self.final_value = Some(proof.final_value);
        let mut leave_number = self.interpolate_cosets[0].size() / 2;
        for merkle_root in proof.merkle_root {
            leave_number /= 2;
            self.folding_roots.push(MerkleTreeVerifier {
                merkle_root,
                leave_number,
            });
        }
        self.shuffle_eval = Some(DeepEval {
            point: self.open_point.clone(),
            first_eval: proof.evaluation,
            else_evals: proof.shuffle_evals,
        });
        proof
            .deep_evals
            .into_iter()
            .enumerate()
            .for_each(|(idx, (first_eval, else_evals))| {
                self.deep_evals.push(DeepEval {
                    point: std::iter::successors(Some(self.oracle.deep[idx]), |&x| Some(x * x))
                        .take(self.total_round - idx)
                        .collect::<Vec<_>>(),
                    first_eval,
                    else_evals,
                });
            });
        let (folding_proof, function_proof) = &proof.query_result;
        let mut leaf_indices = self.oracle.query_list.clone();
        for i in 0..self.total_round {
            let domain_size = self.interpolate_cosets[i].size();
            leaf_indices = leaf_indices
                .iter_mut()
                .map(|v| *v % (domain_size >> 1))
                .collect();
            leaf_indices.sort();
            leaf_indices.dedup();
            if i == 0 {
                function_proof[i].verify_merkle_tree(&leaf_indices, 2, &self.polynomial_roots[i]);
            } else {
                folding_proof[i - 1].verify_merkle_tree(
                    &leaf_indices,
                    2,
                    &self.folding_roots[i - 1],
                );
            }
            if i < self.total_round - 1 {
                function_proof[i + 1].verify_merkle_tree(
                    &leaf_indices,
                    1,
                    &self.polynomial_roots[i + 1],
                );
            } else {
                let challenges = self.oracle.folding_challenges[0..self.total_round].to_vec();
                assert_eq!(
                    self.shuffle_eval
                        .as_ref()
                        .unwrap()
                        .verify(&challenges, &proof.out_evals[0]),
                    self.final_value.unwrap()
                );
                for (j, de) in self.deep_evals.iter().enumerate() {
                    assert_eq!(
                        de.verify(&challenges, &proof.out_evals[j + 1]),
                        self.final_value.unwrap()
                    );
                }
            }
            let folding_value = if i == 0 {
                &function_proof[0].proof_values
            } else {
                &folding_proof[i - 1].proof_values
            };
            let challenge = self.oracle.folding_challenges[i];
            for j in &leaf_indices {
                let x = folding_value[j];
                let nx = folding_value[&(j + domain_size / 2)];
                let v =
                    x + nx + challenge * (x - nx) * self.interpolate_cosets[i].element_inv_at(*j);
                if i == self.total_round - 1 {
                    assert_eq!(v * T::inverse_2(), self.final_value.unwrap());
                } else {
                    assert_eq!(
                        v * T::inverse_2()
                            + challenge * challenge * function_proof[i + 1].proof_values[j],
                        folding_proof[i].proof_values[j]
                    );
                }
            }
        }
        true
    }
}
