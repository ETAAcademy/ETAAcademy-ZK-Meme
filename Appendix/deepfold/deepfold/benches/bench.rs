extern crate criterion;
use criterion::*;

use deepfold::{prover::Prover, verifier::Verifier};
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
    let oracle = RandomOracle::new(variable_num, SECURITY_BITS / CODE_RATE);

    criterion.bench_function(&format!("deepfold commit {:02}", variable_num), move |b| {
        b.iter_batched(
            || polynomial.clone(),
            |p| {
                let prover = Prover::new(variable_num, &interpolate_cosets, p, &oracle, STEP);
                let _commit = prover.commit_polynomial();
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_commit(c: &mut Criterion) {
    for i in 10..SIZE {
        commit::<Mersenne61Ext>(c, i);
    }
}

fn open<T: MyField>(criterion: &mut Criterion, variable_num: usize) {
    let polynomial = MultilinearPolynomial::random_polynomial(variable_num);
    let mut interpolate_cosets = vec![Coset::new(1 << (variable_num + CODE_RATE), T::from_int(1))];
    for i in 1..variable_num + 1 {
        interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
    }
    let oracle = RandomOracle::new(variable_num, SECURITY_BITS / CODE_RATE);
    let prover = Prover::new(variable_num, &interpolate_cosets, polynomial, &oracle, STEP);
    let commit = prover.commit_polynomial();
    let verifier = Verifier::new(variable_num, &interpolate_cosets, commit, &oracle, STEP);
    let point = verifier.get_open_point();

    criterion.bench_function(&format!("deepfold open {:02}", variable_num), move |b| {
        b.iter_batched(
            || (prover.clone(), point.clone()),
            |(p, x)| {
                let _proof = p.generate_proof(x);
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_open(c: &mut Criterion) {
    for i in 10..SIZE {
        open::<Mersenne61Ext>(c, i);
    }
}

fn verify<T: MyField>(criterion: &mut Criterion, variable_num: usize) {
    let polynomial = MultilinearPolynomial::random_polynomial(variable_num);
    let mut interpolate_cosets = vec![Coset::new(1 << (variable_num + CODE_RATE), T::from_int(1))];
    for i in 1..variable_num + 1 {
        interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
    }
    let oracle = RandomOracle::new(variable_num, SECURITY_BITS / CODE_RATE);
    let prover = Prover::new(variable_num, &interpolate_cosets, polynomial, &oracle, STEP);
    let commit = prover.commit_polynomial();
    let verifier = Verifier::new(variable_num, &interpolate_cosets, commit, &oracle, STEP);
    let point = verifier.get_open_point();
    let proof = prover.generate_proof(point);

    criterion.bench_function(&format!("deepfold verify {:02}", variable_num), move |b| {
        b.iter_batched(
            || (verifier.clone(), proof.clone()),
            |(v, pi)| {
                assert!(v.verify(pi));
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_verify(c: &mut Criterion) {
    for i in 10..SIZE {
        verify::<Mersenne61Ext>(c, i);
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_commit, bench_open, bench_verify
}

criterion_main!(benches);
