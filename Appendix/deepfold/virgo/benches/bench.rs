extern crate criterion;

use criterion::*;

use util::{
    algebra::{
        coset::Coset, field::mersenne61_ext::Mersenne61Ext, field::MyField,
        polynomial::MultilinearPolynomial,
    },
    random_oracle::RandomOracle,
};
use virgo::{prover::FriProver, verifier::FriVerifier};

use util::{CODE_RATE, SECURITY_BITS, SIZE, STEP};

fn commit(criterion: &mut Criterion, variable_num: usize) {
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
    let vector_interpolation_coset = Coset::new(1 << variable_num, Mersenne61Ext::random_element());
    criterion.bench_function(&format!("bench virgo commit {}", variable_num), move |b| {
        b.iter_batched(
            || polynomial.clone(),
            |p| {
                let prover = FriProver::new(
                    total_round,
                    &interpolate_cosets,
                    &vector_interpolation_coset,
                    p,
                    &random_oracle,
                    STEP,
                );
                prover.commit_first_polynomial();
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_commit(c: &mut Criterion) {
    for i in 10..SIZE {
        commit(c, i);
    }
}

fn open(criterion: &mut Criterion, variable_num: usize) {
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
    let vector_interpolation_coset = Coset::new(1 << variable_num, Mersenne61Ext::random_element());
    let prover = FriProver::new(
        variable_num,
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
    let open_point = verifier.get_open_point();
    criterion.bench_function(&format!("virgo prove {}", variable_num), |b| {
        b.iter_batched(
            || (prover.clone(), verifier.clone()),
            |(mut p, mut v)| {
                p.commit_functions(&mut v, &open_point);
                p.prove();
                p.commit_foldings(&mut v);
                p.query();
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_open(c: &mut Criterion) {
    for i in 10..SIZE {
        open(c, i);
    }
}

fn verify(criterion: &mut Criterion, variable_num: usize) {
    let total_round = variable_num;
    let polynomial = MultilinearPolynomial::random_polynomial(variable_num);
    let mut interpolate_cosets = vec![Coset::new(
        1 << (variable_num + CODE_RATE),
        Mersenne61Ext::random_element(),
    )];
    for i in 1..variable_num + 1 {
        interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
    }
    let random_oracle = RandomOracle::new(variable_num, SECURITY_BITS / CODE_RATE);
    let vector_interpolation_coset = Coset::new(1 << variable_num, Mersenne61Ext::random_element());
    let mut prover = FriProver::new(
        variable_num,
        &interpolate_cosets,
        &vector_interpolation_coset,
        polynomial,
        &random_oracle,
        STEP,
    );
    let commit = prover.commit_first_polynomial();
    let mut verifier = FriVerifier::new(
        variable_num,
        &interpolate_cosets,
        &vector_interpolation_coset,
        commit,
        &random_oracle,
        STEP,
    );
    let open_point = verifier.get_open_point();
    prover.commit_functions(&mut verifier, &open_point);
    prover.prove();
    prover.commit_foldings(&mut verifier);
    let (folding_proofs, function_proofs, v_value) = prover.query();
    criterion.bench_function(&format!("virgo verify {}", variable_num), |b| {
        b.iter(|| {
            assert!(verifier.verify(&folding_proofs, &v_value, &function_proofs));
        })
    });
}

fn bench_verify(c: &mut Criterion) {
    for i in 10..SIZE {
        verify(c, i);
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_commit, bench_open, bench_verify
}

criterion_main!(benches);
