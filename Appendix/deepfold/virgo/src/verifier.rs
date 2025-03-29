use std::collections::HashMap;

use util::algebra::polynomial::{Polynomial, VanishingPolynomial};
use util::merkle_tree::MERKLE_ROOT_SIZE;
use util::query_result::QueryResult;
use util::random_oracle::RandomOracle;
use util::{
    algebra::{coset::Coset, field::MyField},
    merkle_tree::MerkleTreeVerifier,
};

#[derive(Clone)]
pub struct FriVerifier<T: MyField> {
    total_round: usize,
    interpolate_cosets: Vec<Coset<T>>,
    vector_interpolation_coset: Coset<T>,
    u_root: MerkleTreeVerifier,
    h_root: Option<MerkleTreeVerifier>,
    folding_root: Vec<MerkleTreeVerifier>,
    oracle: RandomOracle<T>,
    vanishing_polynomial: VanishingPolynomial<T>,
    final_poly: Option<Polynomial<T>>,
    evaluation: Option<T>,
    open_point: Option<Vec<T>>,
    step: usize,
}

impl<T: MyField> FriVerifier<T> {
    pub fn new(
        total_round: usize,
        coset: &Vec<Coset<T>>,
        vector_interpolation_coset: &Coset<T>,
        polynomial_commitment: [u8; MERKLE_ROOT_SIZE],
        oracle: &RandomOracle<T>,
        step: usize,
    ) -> Self {
        FriVerifier {
            total_round,
            interpolate_cosets: coset.clone(),
            vector_interpolation_coset: vector_interpolation_coset.clone(),
            u_root: MerkleTreeVerifier {
                leave_number: coset[0].size() / (1 << step),
                merkle_root: polynomial_commitment,
            },
            h_root: None,
            folding_root: vec![],
            oracle: oracle.clone(),
            vanishing_polynomial: VanishingPolynomial::new(vector_interpolation_coset),
            final_poly: None,
            open_point: None,
            evaluation: None,
            step,
        }
    }

    pub fn set_evaluation(&mut self, v: T) {
        self.evaluation = Some(v);
    }

    pub fn get_open_point(&mut self) -> Vec<T> {
        let point = (0..self.total_round)
            .map(|_| T::random_element())
            .collect::<Vec<T>>();
        self.open_point = Some(point.clone());
        point
    }

    pub fn set_h_root(&mut self, h_root: [u8; MERKLE_ROOT_SIZE]) {
        self.h_root = Some(MerkleTreeVerifier {
            merkle_root: h_root,
            leave_number: self.interpolate_cosets[0].size() / (1 << self.step),
        });
    }

    pub fn receive_folding_root(
        &mut self,
        leave_number: usize,
        folding_root: [u8; MERKLE_ROOT_SIZE],
    ) {
        self.folding_root.push(MerkleTreeVerifier {
            leave_number,
            merkle_root: folding_root,
        });
    }

    pub fn set_final_poly(&mut self, poly: Polynomial<T>) {
        self.final_poly = Some(poly);
    }

    pub fn verify(
        &self,
        folding_proofs: &Vec<QueryResult<T>>,
        v_values: &HashMap<usize, T>,
        function_proofs: &Vec<QueryResult<T>>,
    ) -> bool {
        let mut leaf_indices = self.oracle.query_list.clone();
        let rlc = self.oracle.rlc;
        let h_size = T::from_int(self.vector_interpolation_coset.size() as u64);
        for i in 0..self.total_round / self.step {
            // cauchy: verify mt and define get_folding_value fn outside step loop
            let len = self.interpolate_cosets[i * self.step].size() / (1 << self.step);
            leaf_indices = leaf_indices.iter_mut().map(|v| *v % len).collect();
            leaf_indices.sort();
            leaf_indices.dedup();

            if i == 0 {
                assert!(function_proofs[0].verify_merkle_tree(
                    &leaf_indices,
                    1 << self.step,
                    &self.u_root
                ));
                assert!(function_proofs[1].verify_merkle_tree(
                    &leaf_indices,
                    1 << self.step,
                    &self.h_root.as_ref().unwrap()
                ));
            } else {
                folding_proofs[i - 1].verify_merkle_tree(
                    &leaf_indices,
                    1 << self.step,
                    &self.folding_root[i - 1],
                );
            }

            let get_folding_value = |index: &usize| {
                if i == 0 {
                    let u = function_proofs[0].proof_values[index];
                    let h = function_proofs[1].proof_values[index];
                    let v = v_values[index];
                    let x = self.interpolate_cosets[i * self.step].element_at(*index);
                    let x_inv = self.interpolate_cosets[i * self.step].element_inv_at(*index);

                    let mut res = u;
                    let mut acc = rlc;
                    res += h * acc;
                    acc *= rlc;
                    res += acc
                        * (u * v * h_size
                            - self.vanishing_polynomial.evaluation_at(x) * h * h_size
                            - self.evaluation.unwrap())
                        * x_inv;
                    res
                } else {
                    folding_proofs[i - 1].proof_values[index]
                }
            };

            // cauchy: get folding values for multi steps
            for k in &leaf_indices {
                let mut x;
                let mut nx;
                let mut verify_values = vec![];
                let mut verify_inds = vec![];
                for j in 0..(1 << self.step) {
                    // Init verify values, which is the total values in the first step
                    let ind = k + j * len;
                    verify_values.push(get_folding_value(&ind));
                    verify_inds.push(ind);
                }

                for j in 0..self.step {
                    let challenge = self.oracle.folding_challenges[i * self.step + j];
                    let size = verify_values.len();
                    let mut tmp_values = vec![];
                    let mut tmp_inds = vec![];
                    for l in 0..size / 2 {
                        x = verify_values[l];
                        nx = verify_values[l + size / 2];
                        tmp_values.push(
                            x + nx
                                + challenge
                                    * (x - nx)
                                    * self.interpolate_cosets[i * self.step + j]
                                        .element_inv_at(verify_inds[l]),
                        );
                        tmp_inds.push(verify_inds[l]);
                    }
                    verify_values = tmp_values;
                    verify_inds = tmp_inds;
                }
                assert_eq!(verify_values.len(), 1);

                let v = verify_values[0];
                if i < self.total_round / self.step - 1 {
                    if v != folding_proofs[i].proof_values[k] {
                        panic!("{}", i)
                    }
                } else {
                    let point = self.interpolate_cosets[(i + 1) * self.step].element_at(*k);
                    if v != self.final_poly.as_ref().unwrap().evaluation_at(point) {
                        panic!()
                    }
                }
            }
        }
        true
    }
}
