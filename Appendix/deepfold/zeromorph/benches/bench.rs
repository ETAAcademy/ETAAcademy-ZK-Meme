use csv::Writer;
use std::time::Instant;
use util::{
    algebra::{
        coset::Coset,
        field::{mersenne61_ext::Mersenne61Ext, MyField},
        polynomial::MultilinearPolynomial,
    },
    merkle_tree::MERKLE_ROOT_SIZE,
    random_oracle::RandomOracle,
};
use zeromorph::{prover::Prover, verifier::Verifier};

use util::{CODE_RATE, SECURITY_BITS};

fn main() {
    let rep = 10;
    let mut wtr = Writer::from_path("zeromorph.csv").unwrap();
    for variable_num in 18..=26 {
        let polynomial = MultilinearPolynomial::random_polynomial(variable_num);
        let mut interpolate_cosets = vec![Coset::new(
            1 << (variable_num + CODE_RATE),
            Mersenne61Ext::random_element(),
        )];
        for i in 1..variable_num {
            interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
        }
        let oracle = RandomOracle::new(variable_num, SECURITY_BITS / CODE_RATE);
        let now = Instant::now();
        for _ in 1..rep {
            let prover = Prover::new(
                variable_num,
                &interpolate_cosets,
                polynomial.clone(),
                &oracle,
            );
            let _ = prover.commit_polynomial();
        }
        let mut prover = Prover::new(variable_num, &interpolate_cosets, polynomial, &oracle);
        let commit = prover.commit_polynomial();
        let commit_time = now.elapsed().as_micros() as usize / rep;
        let mut verifier = Verifier::new(variable_num, &interpolate_cosets, commit, &oracle);
        let open_point = verifier.get_open_point();

        let now = Instant::now();
        for _ in 1..rep {
            let mut p = prover.clone();
            let mut v = verifier.clone();
            p.commit_functions(&open_point, &mut v);
            p.batch_iopp();
            p.commit_foldings(&mut v);
        }
        prover.commit_functions(&open_point, &mut verifier);
        prover.batch_iopp();
        prover.commit_foldings(&mut verifier);
        let eval_time = now.elapsed().as_micros() as usize / rep;

        let (folding_proof, function_proof) = prover.query();
        let now = Instant::now();
        for _ in 1..rep {
            verifier.verify(&folding_proof, &function_proof);
        }
        verifier.verify(&folding_proof, &function_proof);
        let verify_time = now.elapsed().as_micros() as usize / rep;
        let proof_size = folding_proof.iter().map(|x| x.proof_size()).sum::<usize>()
            + function_proof.iter().map(|x| x.proof_size()).sum::<usize>()
            + variable_num * MERKLE_ROOT_SIZE * 2;
        wtr.write_record(
            &[
                variable_num,
                commit_time,
                eval_time,
                verify_time,
                proof_size,
            ]
            .map(|x| x.to_string()),
        )
        .unwrap();
    }
}
