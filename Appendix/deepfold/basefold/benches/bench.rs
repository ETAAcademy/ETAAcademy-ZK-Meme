extern crate criterion;
use criterion::*;

use basefold::{prover::Prover, verifier::Verifier};
use util::{
    algebra::{
        coset::Coset,
        field::{mersenne61_ext::Mersenne61Ext, MyField},
        polynomial::MultilinearPolynomial,
    },
    random_oracle::RandomOracle,
};

use util::{CODE_RATE, SECURITY_BITS, SIZE, STEP};
fn commit<T: MyField>(criterion: &mut Criterion, variable_num: usize) {
    let polynomial = MultilinearPolynomial::random_polynomial(variable_num);
    let mut interpolate_cosets = vec![Coset::new(1 << (variable_num + CODE_RATE), T::from_int(1))];
    for i in 1..variable_num + 1 {
        interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
    }
    let oracle = RandomOracle::new(
        variable_num,
        (SECURITY_BITS as f32 / (2.0 / (1.0 + 0.5_f32.powi(CODE_RATE as i32))).log2()).ceil()
            as usize,
    );

    criterion.bench_function(
        &format!("basefold {} commit {}", T::FIELD_NAME, variable_num),
        move |b| {
            b.iter_batched(
                || polynomial.clone(),
                |p| {
                    let prover = Prover::new(variable_num, &interpolate_cosets, p, &oracle, STEP);
                    let _commit = prover.commit_polynomial();
                },
                BatchSize::SmallInput,
            )
        },
    );
}

fn bench_commit(c: &mut Criterion) {
    for i in 10..SIZE {
        commit::<Mersenne61Ext>(c, i);
        // commit::<Ft255>(c, i);
    }
}

fn open<T: MyField>(criterion: &mut Criterion, variable_num: usize) {
    let polynomial = MultilinearPolynomial::random_polynomial(variable_num);
    let mut interpolate_cosets = vec![Coset::new(1 << (variable_num + CODE_RATE), T::from_int(1))];
    for i in 1..variable_num + 1 {
        interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
    }
    let oracle = RandomOracle::new(
        variable_num,
        (SECURITY_BITS as f32 / (2.0 / (1.0 + 0.5_f32.powi(CODE_RATE as i32))).log2()).ceil()
            as usize,
    );
    let prover = Prover::new(variable_num, &interpolate_cosets, polynomial, &oracle, STEP);
    let commit = prover.commit_polynomial();
    let verifier = Verifier::new(variable_num, &interpolate_cosets, commit, &oracle, STEP);
    let point = verifier.get_open_point();

    criterion.bench_function(
        &format!("basefold {} open {}", T::FIELD_NAME, variable_num),
        move |b| {
            b.iter_batched(
                || (prover.clone(), verifier.clone()),
                |(mut p, mut v)| {
                    p.send_evaluation(&mut v, &point);
                    p.prove(&point);
                    p.commit_foldings(&mut v);
                    let _ = p.query();
                },
                BatchSize::SmallInput,
            )
        },
    );
}

fn bench_open(c: &mut Criterion) {
    for i in 10..SIZE {
        open::<Mersenne61Ext>(c, i);
        // open::<Ft255>(c, i);
    }
}

fn verify<T: MyField>(criterion: &mut Criterion, variable_num: usize) {
    let polynomial = MultilinearPolynomial::random_polynomial(variable_num);
    let mut interpolate_cosets = vec![Coset::new(1 << (variable_num + CODE_RATE), T::from_int(1))];
    for i in 1..variable_num + 1 {
        interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
    }
    let oracle = RandomOracle::new(
        variable_num,
        (SECURITY_BITS as f32 / (2.0 / (1.0 + 0.5_f32.powi(CODE_RATE as i32))).log2()).ceil()
            as usize,
    );
    let mut prover = Prover::new(variable_num, &interpolate_cosets, polynomial, &oracle, STEP);
    let commit = prover.commit_polynomial();
    let mut verifier = Verifier::new(variable_num, &interpolate_cosets, commit, &oracle, STEP);
    let point = verifier.get_open_point();
    prover.send_evaluation(&mut verifier, &point);
    prover.prove(&point);
    prover.commit_foldings(&mut verifier);
    let proof = prover.query();

    criterion.bench_function(
        &format!("basefold {} verify {}", T::FIELD_NAME, variable_num),
        move |b| {
            b.iter(|| {
                assert!(verifier.verify(&proof));
            })
        },
    );
}

fn bench_verify(c: &mut Criterion) {
    for i in 10..SIZE {
        verify::<Mersenne61Ext>(c, i);
        // verify::<Ft255>(c, i);
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_commit, bench_open, bench_verify
}

criterion_main!(benches);
