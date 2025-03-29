use util::algebra::polynomial::{MultilinearPolynomial, Polynomial};
use util::merkle_tree::MERKLE_ROOT_SIZE;
use util::random_oracle::RandomOracle;
use util::{
    algebra::{coset::Coset, field::MyField},
    merkle_tree::MerkleTreeVerifier,
    query_result::QueryResult,
};

#[derive(Clone)]
pub struct One2ManyVerifier<T: MyField> {
    total_round: usize,
    log_max_degree: usize,
    interpolate_cosets: Vec<Coset<T>>,
    function_root: Vec<MerkleTreeVerifier>,
    folding_root: Vec<MerkleTreeVerifier>,
    oracle: RandomOracle<T>,
    final_value: Option<Polynomial<T>>,
    evaluation: Option<MultilinearPolynomial<T>>,
    open_point: Vec<T>,
}

impl<T: MyField> One2ManyVerifier<T> {
    pub fn new(
        total_round: usize,
        log_max_degree: usize,
        coset: &Vec<Coset<T>>,
        commit: [u8; MERKLE_ROOT_SIZE],
        oracle: &RandomOracle<T>,
    ) -> Self {
        One2ManyVerifier {
            total_round,
            log_max_degree,
            interpolate_cosets: coset.clone(),
            function_root: vec![MerkleTreeVerifier {
                merkle_root: commit,
                leave_number: coset[0].size() / 2,
            }],
            folding_root: vec![],
            oracle: oracle.clone(),
            final_value: None,
            evaluation: None,
            open_point: (0..log_max_degree)
                .into_iter()
                .map(|_| T::random_element())
                .collect(),
        }
    }

    pub fn get_open_point(&self) -> Vec<T> {
        self.open_point.clone()
    }

    pub fn set_evaluation(&mut self, evaluation: MultilinearPolynomial<T>) {
        self.evaluation = Some(evaluation);
    }

    pub fn set_function(&mut self, leave_number: usize, function_root: &[u8; MERKLE_ROOT_SIZE]) {
        self.function_root.push(MerkleTreeVerifier {
            merkle_root: function_root.clone(),
            leave_number,
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

    pub fn set_final_value(&mut self, value: &Polynomial<T>) {
        assert!(value.degree() <= 1 << (self.log_max_degree - self.total_round));
        self.final_value = Some(value.clone());
    }

    pub fn verify(
        &self,
        folding_proof: &Vec<QueryResult<T>>,
        function_proof: &Vec<QueryResult<T>>,
    ) -> bool {
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
                function_proof[i].verify_merkle_tree(&leaf_indices, 2, &self.function_root[i]);
            } else {
                folding_proof[i - 1].verify_merkle_tree(
                    &leaf_indices,
                    2,
                    &self.folding_root[i - 1],
                );
            }

            let challenge = self.oracle.folding_challenges[i];
            let get_folding_value = if i == 0 {
                &function_proof[i].proof_values
            } else {
                &folding_proof[i - 1].proof_values
            };

            let function_values = if i != 0 {
                let function_query_result = &function_proof[i];
                function_query_result.verify_merkle_tree(&leaf_indices, 2, &self.function_root[i]);
                Some(&function_query_result.proof_values)
            } else {
                None
            };
            for j in &leaf_indices {
                let x = get_folding_value[j];
                let nx = get_folding_value[&(j + domain_size / 2)];
                let v =
                    x + nx + challenge * (x - nx) * self.interpolate_cosets[i].element_inv_at(*j);
                if i != 0 {
                    let x = function_values.as_ref().unwrap()[j];
                    let nx = function_values.as_ref().unwrap()[&(j + domain_size / 2)];
                    let v = (v * challenge + (x + nx)) * challenge
                        + (x - nx) * self.interpolate_cosets[i].element_inv_at(*j);
                    if i == self.total_round - 1 {
                        let x = self.interpolate_cosets[i + 1].element_at(*j);
                        if v != self.final_value.as_ref().unwrap().evaluation_at(x) {
                            return false;
                        }
                    } else if v != folding_proof[i].proof_values[j] {
                        return false;
                    }
                } else {
                    if v != folding_proof[i].proof_values[j] {
                        return false;
                    }
                }
                let x = function_proof[i].proof_values[j];
                let nx = function_proof[i].proof_values[&(j + domain_size / 2)];
                let v = x
                    + nx
                    + self.open_point[i] * (x - nx) * self.interpolate_cosets[i].element_inv_at(*j);
                if i < self.total_round - 1 {
                    assert_eq!(v, function_proof[i + 1].proof_values[j] * T::from_int(2));
                } else {
                    let x = self.interpolate_cosets[i + 1].element_at(*j);
                    let poly_v = self.evaluation.as_ref().unwrap().evaluate_as_polynomial(x);
                    assert_eq!(v, poly_v * T::from_int(2));
                }
            }
        }
        true
    }
}
