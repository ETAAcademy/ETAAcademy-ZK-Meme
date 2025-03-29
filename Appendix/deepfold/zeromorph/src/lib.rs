pub mod prover;
pub mod verifier;

#[cfg(test)]
mod tests {
    use crate::{prover::Prover, verifier::Verifier};
    use csv::Writer;
    use util::{
        algebra::{
            coset::Coset,
            field::{mersenne61_ext::Mersenne61Ext, MyField},
            polynomial::MultilinearPolynomial,
        },
        merkle_tree::MERKLE_ROOT_SIZE,
        random_oracle::RandomOracle,
    };
    use util::{CODE_RATE, SECURITY_BITS, SIZE};

    fn output_proof_size(variable_num: usize) -> usize {
        let polynomial = MultilinearPolynomial::random_polynomial(variable_num);
        let mut interpolate_cosets = vec![Coset::new(
            1 << (variable_num + CODE_RATE),
            Mersenne61Ext::random_element(),
        )];
        for i in 1..variable_num {
            interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
        }
        let oracle = RandomOracle::new(variable_num, SECURITY_BITS / CODE_RATE);
        let mut prover = Prover::new(variable_num, &interpolate_cosets, polynomial, &oracle);
        let commit = prover.commit_polynomial();
        let mut verifier = Verifier::new(variable_num, &interpolate_cosets, commit, &oracle);
        let open_point = verifier.get_open_point();

        prover.commit_functions(&open_point, &mut verifier);
        prover.batch_iopp();
        prover.commit_foldings(&mut verifier);
        let (folding_proof, function_proof) = prover.query();
        verifier.verify(&folding_proof, &function_proof);
        folding_proof.iter().map(|x| x.proof_size()).sum::<usize>()
            + function_proof.iter().map(|x| x.proof_size()).sum::<usize>()
            + variable_num * MERKLE_ROOT_SIZE * 2
    }

    #[test]
    fn test_proof_size() {
        let mut wtr = Writer::from_path("zeromorph.csv").unwrap();
        let range = 18..SIZE;
        for i in range.clone() {
            let proof_size = output_proof_size(i);
            wtr.write_record(&[i.to_string(), proof_size.to_string()])
                .unwrap();
        }
    }
}
