pub mod prover;
pub mod verifier;

#[cfg(test)]
mod tests {
    use std::mem::size_of;

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
    use util::{CODE_RATE, SECURITY_BITS, SIZE, STEP};

    fn output_proof_size(variable_num: usize) -> usize {
        let total_round = variable_num;
        let polynomial = MultilinearPolynomial::random_polynomial(variable_num);
        let mut interpolate_cosets = vec![Coset::new(
            1 << (variable_num + CODE_RATE),
            Mersenne61Ext::from_int(1),
        )];
        for i in 1..variable_num + 1 {
            interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
        }
        let oracle = RandomOracle::new(
            total_round,
            (SECURITY_BITS as f32 / (2.0 / (1.0 + 0.5_f32.powi(CODE_RATE as i32))).log2()).ceil()
                as usize,
        );
        let mut prover = Prover::new(total_round, &interpolate_cosets, polynomial, &oracle, STEP);
        let commit = prover.commit_polynomial();
        let mut verifier = Verifier::new(total_round, &interpolate_cosets, commit, &oracle, STEP);
        let point = verifier.get_open_point();
        prover.send_evaluation(&mut verifier, &point);
        prover.prove(&point);
        prover.commit_foldings(&mut verifier);
        let proof = prover.query();
        assert!(verifier.verify(&proof));
        proof
            .iter()
            .map(|x: &util::query_result::QueryResult<Mersenne61Ext>| x.proof_size())
            .sum::<usize>()
            + variable_num * (MERKLE_ROOT_SIZE + size_of::<Mersenne61Ext>() * 3)
    }

    #[test]
    fn test_proof_size() {
        let mut wtr = Writer::from_path("basefold.csv").unwrap();
        let range = 10..SIZE;
        for i in range.clone() {
            let proof_size = output_proof_size(i);
            wtr.write_record(&[i.to_string(), proof_size.to_string()])
                .unwrap();
        }
    }
}
