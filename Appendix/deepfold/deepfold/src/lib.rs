use std::mem::size_of;

use util::{
    algebra::{field::MyField, polynomial::Polynomial},
    merkle_tree::MERKLE_ROOT_SIZE,
    query_result::QueryResult,
};

pub mod prover;
pub mod verifier;

#[derive(Clone)]
pub struct DeepEval<T: MyField> {
    point: Vec<T>,
    first_eval: T,
    else_evals: Vec<T>,
}

impl<T: MyField> DeepEval<T> {
    pub fn new(point: Vec<T>, poly_hypercube: Vec<T>) -> Self {
        DeepEval {
            point: point.clone(),
            first_eval: Self::evaluatioin_at(point, poly_hypercube),
            else_evals: vec![],
        }
    }

    fn evaluatioin_at(point: Vec<T>, mut poly_hypercube: Vec<T>) -> T {
        let mut len = poly_hypercube.len();
        assert_eq!(len, 1 << point.len());
        for v in point.into_iter() {
            len >>= 1;
            for i in 0..len {
                poly_hypercube[i] *= T::from_int(1) - v;
                let tmp = poly_hypercube[i + len] * v;
                poly_hypercube[i] += tmp;
            }
        }
        poly_hypercube[0]
    }

    pub fn append_else_eval(&mut self, poly_hypercube: Vec<T>) {
        let mut point = self.point[self.else_evals.len()..].to_vec();
        point[0] += T::from_int(1);
        self.else_evals
            .push(Self::evaluatioin_at(point, poly_hypercube));
    }

    pub fn verify(&self, challenges: &Vec<T>) -> T {
        let (_, challenges) = challenges.split_at(challenges.len() - self.point.len());
        let mut y_0 = self.first_eval;
        assert_eq!(self.point.len(), self.else_evals.len());
        for ((x, eval), challenge) in self
            .point
            .iter()
            .zip(self.else_evals.iter())
            .zip(challenges.into_iter())
        {
            let y_1 = eval.clone();
            y_0 += (y_1 - y_0) * (challenge.clone() - x.clone());
        }
        y_0
    }
}

pub struct Commit<T: MyField> {
    merkle_root: [u8; MERKLE_ROOT_SIZE],
    deep: T,
}

#[derive(Clone)]
pub struct Proof<T: MyField> {
    merkle_root: Vec<[u8; MERKLE_ROOT_SIZE]>,
    query_result: Vec<QueryResult<T>>,
    deep_evals: Vec<(T, Vec<T>)>,
    shuffle_evals: Vec<T>,
    evaluation: T,
    final_value: T,
    final_poly: Polynomial<T>,
}

impl<T: MyField> Proof<T> {
    pub fn size(&self) -> usize {
        self.merkle_root.len() * MERKLE_ROOT_SIZE
            + self
                .query_result
                .iter()
                .fold(0, |acc, x| acc + x.proof_size())
            + (self.deep_evals.iter().fold(0, |acc, x| acc + x.1.len())
                + self.shuffle_evals.len()
                + 2)
                * size_of::<T>()
    }
}

#[cfg(test)]
mod tests {
    use crate::{prover::Prover, verifier::Verifier};
    use csv::Writer;
    use util::{
        algebra::{
            coset::Coset,
            field::{m31_ext::M31ext, MyField},
            polynomial::MultilinearPolynomial,
        },
        random_oracle::RandomOracle,
    };
    use util::{CODE_RATE, SECURITY_BITS, SIZE, STEP};

    fn output_proof_size<T: MyField>(variable_num: usize) -> usize {
        let polynomial = MultilinearPolynomial::random_polynomial(variable_num);
        let mut interpolate_cosets =
            vec![Coset::new(1 << (variable_num + CODE_RATE), T::from_int(1))];
        for i in 1..variable_num + 1 {
            interpolate_cosets.push(interpolate_cosets[i - 1].pow(2));
        }
        let oracle = RandomOracle::new(variable_num, SECURITY_BITS / CODE_RATE);
        let prover = Prover::new(variable_num, &interpolate_cosets, polynomial, &oracle, STEP);
        let commit = prover.commit_polynomial();
        let verifier = Verifier::new(variable_num, &interpolate_cosets, commit, &oracle, STEP);
        let point = verifier.get_open_point();
        let proof = prover.generate_proof(point);
        let size = proof.size();
        assert!(verifier.verify(proof));
        size
    }

    #[test]
    fn test_proof_size() {
        let mut wtr = Writer::from_path("deepfold.csv").unwrap();
        let range = 10..SIZE;
        for i in range.clone() {
            let proof_size = output_proof_size::<M31ext>(i);
            wtr.write_record(&[i.to_string(), proof_size.to_string()])
                .unwrap();
        }
    }
}
