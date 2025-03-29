pub mod algebra {
    pub mod coset;
    pub mod field;
    pub mod polynomial;
}
pub mod sumcheck {
    pub mod prover;
    pub mod verifier;
}

pub mod interpolation;
pub mod merkle_tree;
pub mod query_result;
pub mod random_oracle;

pub const CODE_RATE: usize = 3;
pub const SECURITY_BITS: usize = 100;
pub const STEP: usize = 1;
pub const SIZE: usize = 27;
pub fn split_n(mut n: usize) -> Vec<usize> {
    let mut res = vec![];
    let mut i = 1;
    while i < n {
        res.push(i);
        n -= i;
        i <<= 1;
    }
    if n > 0 {
        res.push(n);
    }
    res.sort_by(|x, y| y.trailing_zeros().cmp(&x.trailing_zeros()));
    res
}

fn batch_bit_reverse(log_n: usize) -> Vec<usize> {
    let n = 1 << log_n;
    let mut res = (0..n).into_iter().map(|_| 0).collect::<Vec<usize>>();
    for i in 0..n {
        res[i] = (res[i >> 1] >> 1) | ((i & 1) << (log_n - 1));
    }
    res
}
