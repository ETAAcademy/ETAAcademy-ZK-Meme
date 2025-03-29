pub mod prover;
pub mod verifier;

#[cfg(test)]
mod tests {
    use crate::{prover::FriProver, verifier::FriVerifier};
    use csv::Writer;
    use std::mem::size_of;
    use util::{
        algebra::{
            coset::Coset, field::mersenne61_ext::Mersenne61Ext, field::MyField,
            polynomial::MultilinearPolynomial,
        },
        merkle_tree::MERKLE_ROOT_SIZE,
        random_oracle::RandomOracle,
    };

    use util::{CODE_RATE, SECURITY_BITS, SIZE, STEP};
    fn output_proof_size(variable_num: usize) -> usize {
        let total_round = variable_num;
        let polynomial = MultilinearPolynomial::random_polynomial(variable_num);
        let mut interpolate_cosets = vec![Coset::new(
            1 << (variable_num + CODE_RATE),
            Mersenne61Ext::random_element(),
        )];
        for i in 1..variable_num + 1 {
            interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
        }
        let random_oracle = RandomOracle::new(total_round, SECURITY_BITS / CODE_RATE);
        let vector_interpolation_coset =
            Coset::new(1 << variable_num, Mersenne61Ext::random_element());
        let mut prover = FriProver::new(
            total_round,
            &interpolate_cosets,
            &vector_interpolation_coset,
            polynomial,
            &random_oracle,
            STEP,
        );
        let commit = prover.commit_first_polynomial();
        let mut verifier = FriVerifier::new(
            total_round,
            &interpolate_cosets,
            &vector_interpolation_coset,
            commit,
            &random_oracle,
            STEP,
        );
        // cauchy: why vector of points rather than a single point?
        let open_point = verifier.get_open_point();
        prover.commit_functions(&mut verifier, &open_point);
        prover.prove();
        prover.commit_foldings(&mut verifier);
        let (folding_proofs, function_proofs, v_value) = prover.query();
        assert!(verifier.verify(&folding_proofs, &v_value, &function_proofs));
        folding_proofs.iter().map(|x| x.proof_size()).sum::<usize>()
            + (variable_num + 1) * MERKLE_ROOT_SIZE
            + size_of::<Mersenne61Ext>()
            + function_proofs
                .iter()
                .map(|x| x.proof_size())
                .sum::<usize>()
    }

    #[test]
    fn test_virgo_proof_size() {
        let mut wtr = Writer::from_path("virgo.csv").unwrap();
        let range = 10..SIZE;
        for i in range.clone() {
            let proof_size = output_proof_size(i);
            wtr.write_record([i.to_string(), proof_size.to_string()])
                .unwrap();
        }
    }
}
