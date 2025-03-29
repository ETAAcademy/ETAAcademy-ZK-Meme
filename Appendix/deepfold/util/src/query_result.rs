use crate::algebra::field::{as_bytes_vec, MyField};
use crate::merkle_tree::MerkleTreeVerifier;
use std::collections::HashMap;
use std::mem::size_of;

#[derive(Clone)]
pub struct QueryResult<T: MyField> {
    pub proof_bytes: Vec<u8>,
    // Cauchy: Why use hashmap rather than Vec here?
    pub proof_values: HashMap<usize, T>,
}

impl<T: MyField> QueryResult<T> {
    pub fn verify_merkle_tree(
        &self,
        leaf_indices: &Vec<usize>,
        leaf_size: usize,
        merkle_verifier: &MerkleTreeVerifier,
    ) -> bool {
        let len = merkle_verifier.leave_number;

        let leaves: Vec<Vec<u8>> = leaf_indices
            .iter()
            .map(|x| {
                as_bytes_vec(
                    &(0..leaf_size)
                        .map(|j| {
                            self.proof_values
                                .get(&(x.clone() + j * len))
                                .unwrap()
                                .clone()
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let res = merkle_verifier.verify(self.proof_bytes.clone(), leaf_indices, &leaves);
        assert!(res);
        res
    }

    pub fn proof_size(&self) -> usize {
        self.proof_bytes.len() + self.proof_values.len() * size_of::<T>()
    }
}
