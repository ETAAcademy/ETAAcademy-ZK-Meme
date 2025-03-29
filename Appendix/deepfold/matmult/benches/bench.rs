extern crate criterion;
use criterion::*;

use criterion::Criterion;
use matmult::matmult::{mat_mult, naive_mat_mult, naive_opening, Matrix};
use util::algebra::field::mersenne61_ext::Mersenne61Ext;

fn bench_naive_mat_mult(c: &mut Criterion) {
    let row_size = vec![150, 300, 600, 900, 1200];
    for r in row_size {
        let mat_a = Matrix::<Mersenne61Ext>::sample(r, 768);
        let mat_b = Matrix::<Mersenne61Ext>::sample(768, 2304);
        let mat_c = mat_a.clone() * mat_b.clone();

        c.bench_function(&format!("naive mat mult for size {}", r), move |b| {
            b.iter_batched(
                || (mat_a.clone(), mat_b.clone(), mat_c.clone()),
                |(a, b, c)| {
                    naive_mat_mult(&a, &b, &c);
                },
                BatchSize::SmallInput,
            )
        });
    }
}

fn bench_mat_mult(c: &mut Criterion) {
    let row_size = vec![150, 300, 600, 900, 1200];
    for r in row_size {
        let mat_a = Matrix::<Mersenne61Ext>::sample(r, 768);
        let mat_b = Matrix::<Mersenne61Ext>::sample(768, 2304);
        let mat_c = mat_a.clone() * mat_b.clone();

        c.bench_function(
            &format!("batch-deepfold mat mult for size {}", r),
            move |b| {
                b.iter_batched(
                    || (mat_a.clone(), mat_b.clone(), mat_c.clone()),
                    |(a, b, c)| {
                        mat_mult(&a, &b, &c);
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

fn bench_naive_opening(c: &mut Criterion) {
    let row_size = vec![150, 300, 600, 900, 1200];
    for r in row_size {
        let mat_a = Matrix::<Mersenne61Ext>::sample(r, 768);
        let mat_b = Matrix::<Mersenne61Ext>::sample(768, 2304);
        let mat_c = mat_a.clone() * mat_b.clone();

        c.bench_function(
            &format!("naive opening mat mult for size {}", r),
            move |b| {
                b.iter_batched(
                    || (mat_a.clone(), mat_b.clone(), mat_c.clone()),
                    |(a, b, c)| {
                        naive_opening(&a, &b, &c);
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_naive_mat_mult, bench_mat_mult, bench_naive_opening
    // targets = bench_naive_opening
}

criterion_main!(benches);
