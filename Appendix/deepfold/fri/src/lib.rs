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
            polynomial::Polynomial,
        },
        merkle_tree::MERKLE_ROOT_SIZE,
        random_oracle::RandomOracle,
    };
    use util::{CODE_RATE, SECURITY_BITS, SIZE, STEP};

    fn output_proof_size(variable_num: usize) -> usize {
        let degree = 1 << variable_num;
        let polynomial = Polynomial::random_polynomial(degree);
        let mut interpolate_cosets = vec![Coset::new(
            1 << (variable_num + CODE_RATE),
            Mersenne61Ext::from_int(1),
        )];
        for i in 1..variable_num + 1 {
            interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
        }
        let oracle = RandomOracle::new(variable_num, SECURITY_BITS / CODE_RATE);
        let mut prover = Prover::new(variable_num, &interpolate_cosets, polynomial, &oracle, STEP);
        let commits = prover.commit_polynomial();
        let mut verifier = Verifier::new(variable_num, &interpolate_cosets, commits, &oracle, STEP);
        let point = verifier.get_open_point();

        let evaluation = prover.prove(point);
        prover.commit_foldings_multi_step(&mut verifier);
        let interpolation_proof = prover.query();
        assert!(verifier.verify(&interpolation_proof, evaluation));
        interpolation_proof
            .iter()
            .map(|x| x.proof_size())
            .sum::<usize>()
            + variable_num * MERKLE_ROOT_SIZE
    }

    #[test]
    fn test_proof_size() {
        let mut wtr = Writer::from_path("fri.csv").unwrap();
        for i in 10..SIZE {
            let proof_size = output_proof_size(i);
            wtr.write_record([i.to_string(), proof_size.to_string()])
                .unwrap();
        }
    }
}
