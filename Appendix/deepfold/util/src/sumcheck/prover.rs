use crate::{
    algebra::{field::MyField, polynomial::MultilinearPolynomial},
    random_oracle::RandomOracle,
};

use super::verifier::SumcheckVerifier;

pub struct SumcheckProver<T: MyField> {
    total_round: usize,
    polynomial: MultilinearPolynomial<T>,
    oracle: RandomOracle<T>,
    hypercube_interpolation: Vec<T>,
    sumcheck_values: Vec<(T, T, T)>,
}

impl<T: MyField> SumcheckProver<T> {
    pub fn new(
        total_round: usize,
        polynomial: MultilinearPolynomial<T>,
        oracle: &RandomOracle<T>,
    ) -> Self {
        assert_eq!(total_round as u32, polynomial.coefficients().len().ilog2());
        SumcheckProver {
            total_round,
            polynomial: polynomial.clone(),
            oracle: oracle.clone(),
            hypercube_interpolation: polynomial.evaluate_hypercube(),
            sumcheck_values: vec![],
        }
    }

    pub fn prove(&mut self) {
        let mut poly_hypercube = self.hypercube_interpolation.clone();
        // let mut eq_hypercube = EqMultilinear::new(point.clone()).evaluate_hypercube();
        for i in 0..self.total_round {
            let m = 1 << (self.total_round - i - 1);
            let challenge = self.oracle.folding_challenges[i];
            let (sum_0, sum_1, sum_2) = (0..m).into_iter().fold(
                (T::from_int(0), T::from_int(0), T::from_int(0)),
                |acc, x| {
                    let p_0 = poly_hypercube[x];
                    let p_1 = poly_hypercube[x + m];
                    // let e_0 = eq_hypercube[x];
                    // let e_1 = eq_hypercube[x + m];
                    (
                        // acc.0 + p_0 * e_0,
                        // acc.1 + p_1 * e_1,
                        // acc.2 + (p_1 + p_1 - p_0) * (e_1 + e_1 - e_0),
                        acc.0 + p_0,
                        acc.1 + p_1,
                        acc.2 + (p_1 + p_1 - p_0),
                    )
                },
            );
            self.sumcheck_values.push((sum_0, sum_1, sum_2));

            Self::sumcheck_next_domain(&mut poly_hypercube, m, challenge);
            // Self::sumcheck_next_domain(&mut eq_hypercube, m, challenge);
        }
    }

    fn sumcheck_next_domain(hypercube_interpolation: &mut Vec<T>, m: usize, challenge: T) {
        for i in 0..m {
            hypercube_interpolation[i] *= T::from_int(1) - challenge;
            let tmp = hypercube_interpolation[i + m] * challenge;
            hypercube_interpolation[i] += tmp;
        }
    }

    pub fn send_sumcheck_values(&mut self, verifier: &mut SumcheckVerifier<T>) {
        for value in &self.sumcheck_values {
            verifier.receive_sumcheck_value(value.clone());
        }
    }
}
