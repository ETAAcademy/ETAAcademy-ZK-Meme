extern crate criterion;
use criterion::*;

use batch::{prover::Prover, verifier::Verifier};
use util::{
    algebra::{
        coset::Coset,
        field::{mersenne61_ext::Mersenne61Ext, MyField},
        polynomial::MultilinearPolynomial,
    },
    random_oracle::RandomOracle,
};

use util::{CODE_RATE, SECURITY_BITS, SIZE};

fn open<T: MyField>(criterion: &mut Criterion, variable_num: usize) {
    let mut interpolate_cosets = vec![Coset::new(1 << (variable_num + CODE_RATE), T::from_int(1))];
    for i in 1..variable_num {
        interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
    }
    let oracle = RandomOracle::new(variable_num, SECURITY_BITS / CODE_RATE);
    let polynomials = (0..variable_num)
        .rev()
        .map(|x| MultilinearPolynomial::random_polynomial(x + 1))
        .collect();
    let prover = Prover::new(variable_num, &interpolate_cosets, polynomials, &oracle);
    let commit = prover.commit_polynomial();
    let verifier = Verifier::new(variable_num, &interpolate_cosets, commit, &oracle);
    let point = verifier.get_open_point();

    criterion.bench_function(&format!("batch open {}", variable_num), move |b| {
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
    let mut interpolate_cosets = vec![Coset::new(1 << (variable_num + CODE_RATE), T::from_int(1))];
    for i in 1..variable_num {
        interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
    }
    let oracle = RandomOracle::new(variable_num, SECURITY_BITS / CODE_RATE);
    let polynomials = (0..variable_num)
        .rev()
        .map(|x| MultilinearPolynomial::random_polynomial(x + 1))
        .collect();
    let prover = Prover::new(variable_num, &interpolate_cosets, polynomials, &oracle);
    let commit = prover.commit_polynomial();
    let verifier = Verifier::new(variable_num, &interpolate_cosets, commit, &oracle);
    let point = verifier.get_open_point();
    let proof = prover.generate_proof(point);

    criterion.bench_function(&format!("batch verify {}", variable_num), move |b| {
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
    targets = bench_open, bench_verify
}

criterion_main!(benches);
