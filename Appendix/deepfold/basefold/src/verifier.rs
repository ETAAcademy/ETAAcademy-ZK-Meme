use util::algebra::polynomial::{EqMultilinear, Polynomial};
use util::merkle_tree::MERKLE_ROOT_SIZE;
use util::random_oracle::RandomOracle;
use util::{
    algebra::{coset::Coset, field::MyField},
    merkle_tree::MerkleTreeVerifier,
    query_result::QueryResult,
};

#[derive(Clone)]
pub struct Verifier<T: MyField> {
    total_round: usize,
    interpolate_cosets: Vec<Coset<T>>,
    polynomial_roots: Vec<MerkleTreeVerifier>,
    oracle: RandomOracle<T>,
    final_poly: Option<Polynomial<T>>,
    sumcheck_values: Vec<(T, T, T)>,
    open_point: Vec<T>,
    evaluation: Option<T>,
    step: usize,
}

impl<T: MyField> Verifier<T> {
    pub fn new(
        total_round: usize,
        coset: &Vec<Coset<T>>,
        commit: [u8; MERKLE_ROOT_SIZE],
        oracle: &RandomOracle<T>,
        step: usize,
    ) -> Self {
        Verifier {
            total_round,
            interpolate_cosets: coset.clone(),
            oracle: oracle.clone(),
            polynomial_roots: vec![MerkleTreeVerifier::new(
                coset[0].size() / (1 << step),
                &commit,
            )],
            final_poly: None,
            sumcheck_values: vec![],
            open_point: (0..total_round).map(|_| T::random_element()).collect(),
            evaluation: None,
            step,
        }
    }

    pub fn get_open_point(&self) -> Vec<T> {
        self.open_point.clone()
    }

    pub fn receive_sumcheck_value(&mut self, value: (T, T, T)) {
        self.sumcheck_values.push(value);
    }

    pub fn receive_folding_root(
        &mut self,
        leave_number: usize,
        folding_root: [u8; MERKLE_ROOT_SIZE],
    ) {
        self.polynomial_roots.push(MerkleTreeVerifier {
            leave_number,
            merkle_root: folding_root,
        });
    }

    pub fn set_evalutation(&mut self, evaluation: T) {
        self.evaluation = Some(evaluation);
    }

    pub fn set_final_poly(&mut self, poly: Polynomial<T>) {
        self.final_poly = Some(poly);
    }

    pub fn verify(&self, polynomial_proof: &Vec<QueryResult<T>>) -> bool {
        let mut leaf_indices = self.oracle.query_list.clone();
        let mut sum = self.sumcheck_values[0].0 + self.sumcheck_values[0].1;
        for i in 0..self.total_round / self.step {
            let domain_size = self.interpolate_cosets[i * self.step].size();
            leaf_indices = leaf_indices
                .iter_mut()
                .map(|v| *v % (domain_size / (1 << self.step)))
                .collect();
            leaf_indices.sort();
            leaf_indices.dedup();

            polynomial_proof[i].verify_merkle_tree(
                &leaf_indices,
                1 << self.step,
                &self.polynomial_roots[i],
            );
            let folding_value = &polynomial_proof[i].proof_values;

            for k in 0..self.step {
                let challenge = self.oracle.folding_challenges[i * self.step + k];
                sum =
                    self.process_sumcheck(challenge, sum, self.sumcheck_values[i * self.step + k]);
            }

            for k in &leaf_indices {
                let mut x;
                let mut nx;
                let mut verify_values = vec![];
                let mut verify_inds = vec![];
                for j in 0..(1 << self.step) {
                    // Init verify values, which is the total values in the first step
                    let ind = k + j * domain_size / (1 << self.step);
                    verify_values.push(folding_value[&ind]);
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
                            (x + nx
                                + challenge
                                    * (x - nx)
                                    * self.interpolate_cosets[i * self.step + j]
                                        .element_inv_at(verify_inds[l]))
                                * T::inverse_2(),
                        );
                        tmp_inds.push(verify_inds[l]);
                    }
                    verify_values = tmp_values;
                    verify_inds = tmp_inds;
                }
                // // ----------------------
                // for k in 0..self.step {
                //     let challenge = self.oracle.folding_challenges[i*self.step + k];
                //     let x = folding_value[j];
                //     let nx = folding_value[&(j + domain_size / 2)];
                //     let v =
                //         x + nx + challenge * (x - nx) * self.interpolate_cosets[i].element_inv_at(*j);
                // }

                assert_eq!(verify_values.len(), 1);
                let v = verify_values[0];

                if i == self.total_round / self.step {
                    let mut j = 0;
                    while i * self.step + j < self.total_round {
                        let challenge = self.oracle.folding_challenges[i * self.step + k];
                        sum = self.process_sumcheck(
                            challenge,
                            sum,
                            self.sumcheck_values[i * self.step + k],
                        );
                        j += 1;
                    }
                    let eq_poly = EqMultilinear::new(self.open_point.clone());
                    assert_eq!(
                        v,
                        sum * T::inverse(&eq_poly.evaluate(
                            &self.oracle.folding_challenges[0..self.total_round].to_vec()
                        ))
                    )
                } else if i == self.total_round / self.step - 1 {
                    let point = self.interpolate_cosets[(i + 1) * self.step].element_at(*k);
                    assert_eq!(v, self.final_poly.clone().unwrap().evaluation_at(point));
                } else {
                    assert_eq!(v, polynomial_proof[i + 1].proof_values[k]);
                }
            }
        }
        true
    }

    fn process_sumcheck(&self, challenge: T, last_sum: T, sumcheck_values: (T, T, T)) -> T {
        let x_0 = sumcheck_values.0;
        let x_1 = sumcheck_values.1;
        let x_2 = sumcheck_values.2;
        assert_eq!(last_sum, x_0 + x_1);
        let sum =
            x_0 * (T::from_int(1) - challenge) * (T::from_int(2) - challenge) * T::inverse_2()
                + x_1 * challenge * (T::from_int(2) - challenge)
                + x_2 * challenge * (challenge - T::from_int(1)) * T::inverse_2();
        sum
    }
}
