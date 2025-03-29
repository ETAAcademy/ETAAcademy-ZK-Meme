use util::algebra::polynomial::Polynomial;
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
    interpolation_roots: Vec<MerkleTreeVerifier>,
    oracle: RandomOracle<T>,
    final_poly: Option<Polynomial<T>>, // used for multi-step verifier
    open_point: T,
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
            interpolation_roots: vec![MerkleTreeVerifier::new(
                coset[0].size() / (1 << step),
                &commit,
            )],
            final_poly: None,
            open_point: T::random_element(),
            step: step,
        }
    }

    pub fn get_open_point(&self) -> T {
        self.open_point
    }

    pub fn receive_interpolation_root(
        &mut self,
        leave_number: usize,
        interpolation_root: [u8; MERKLE_ROOT_SIZE],
    ) {
        self.interpolation_roots.push(MerkleTreeVerifier {
            leave_number,
            merkle_root: interpolation_root,
        });
    }

    pub fn set_final_poly(&mut self, poly: Polynomial<T>) {
        self.final_poly = Some(poly);
    }

    pub fn verify(&self, interpolation_proof: &Vec<QueryResult<T>>, evaluation: T) -> bool {
        let mut leaf_indices = self.oracle.query_list.clone();
        for i in 0..self.total_round / self.step - 1 {
            let domain_size = self.interpolate_cosets[i * self.step].size();
            leaf_indices = leaf_indices
                .iter_mut()
                .map(|v| *v % (domain_size >> self.step))
                .collect();
            leaf_indices.sort();
            leaf_indices.dedup();

            // Cauchy: verify mt
            interpolation_proof[i].verify_merkle_tree(
                &leaf_indices,
                1 << self.step,
                &self.interpolation_roots[i],
            );

            let mut challenge = vec![];
            for j in 0..self.step {
                challenge.push(self.oracle.folding_challenges[i * self.step + j]);
            }

            let get_folding_value: Box<dyn Fn(&usize) -> T> = if i == 0 {
                Box::new(|x| {
                    (interpolation_proof[0].proof_values[x] - evaluation)
                        * (self.interpolate_cosets[0].element_at(*x) - self.open_point).inverse()
                })
            } else {
                Box::new(|x| interpolation_proof[i].proof_values[x])
            };
            for k in &leaf_indices {
                let mut x;
                let mut nx;
                let mut verify_values = vec![];
                let mut verify_inds = vec![];
                for j in 0..(1 << self.step) {
                    // Init verify values, which is the total values in the first step
                    let ind = k + j * domain_size / (1 << self.step);
                    verify_values.push(get_folding_value(&ind));
                    verify_inds.push(ind);
                }
                for j in 0..self.step {
                    let size = verify_values.len();
                    let mut tmp_values = vec![];
                    let mut tmp_inds = vec![];
                    for l in 0..size / 2 {
                        x = verify_values[l];
                        nx = verify_values[l + size / 2];
                        tmp_values.push(
                            x + nx
                                + challenge[j]
                                    * (x - nx)
                                    * self.interpolate_cosets[i * self.step + j]
                                        .element_inv_at(verify_inds[l]),
                        );
                        tmp_inds.push(verify_inds[l]);
                    }
                    verify_values = tmp_values;
                    verify_inds = tmp_inds;
                }
                assert_eq!(verify_values[0], interpolation_proof[i + 1].proof_values[k]);
            }
        }

        // Cauchy: the final round
        // let i = self.total_round / self.step;

        // interpolation_proof[i].verify_merkle_tree(
        //     &(0..self.interpolation_roots[i].leave_number).collect(),
        //     usize::pow(2, self.step as u32),
        //     &self.interpolation_roots[i],
        // );

        // let coset = self.interpolate_cosets[i * self.step].clone();
        // for x in 0..coset.size() {
        //     assert_eq!(
        //         self.final_poly
        //             .clone()
        //             .unwrap()
        //             .evaluation_at(coset.element_at(x)),
        //         interpolation_proof[i].proof_values[&x]
        //     )
        // }
        true
    }
}
