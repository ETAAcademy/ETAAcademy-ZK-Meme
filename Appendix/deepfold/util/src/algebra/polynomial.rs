use std::ops::{Add, Mul};

use crate::batch_bit_reverse;

use super::coset::Coset;
use super::field::MyField;

#[derive(Debug, Clone)]
pub struct Polynomial<T: MyField> {
    pub coefficients: Vec<T>,
}

impl<T: MyField> Polynomial<T> {
    pub fn new(mut coefficients: Vec<T>) -> Polynomial<T> {
        let zero = T::from_int(0);
        while *coefficients.last().unwrap() == zero {
            coefficients.pop();
        }
        Polynomial { coefficients }
    }

    pub fn coefficients(&self) -> &Vec<T> {
        &self.coefficients
    }

    pub fn random_polynomial(degree: usize) -> Polynomial<T> {
        Polynomial {
            coefficients: (0..degree).map(|_| MyField::random_element()).collect(),
        }
    }

    pub fn degree(&self) -> usize {
        let n = self.coefficients.len();
        if n == 0 {
            0
        } else {
            n - 1
        }
    }

    pub fn evaluation_at(&self, x: T) -> T {
        let mut res = MyField::from_int(0);
        for i in self.coefficients.iter().rev() {
            res *= x;
            res += *i;
        }
        res
    }

    pub fn evaluation_over_coset(&self, coset: &Coset<T>) -> Vec<T> {
        coset.fft(self.coefficients.clone())
    }

    pub fn over_vanish_polynomial(
        &self,
        vanishing_polynomial: &VanishingPolynomial<T>,
    ) -> Polynomial<T> {
        let degree = vanishing_polynomial.degree;
        let low_term = vanishing_polynomial.shift;
        let mut coeff = vec![];
        let mut remnant = self.coefficients.clone();
        for i in (degree..self.coefficients.len()).rev() {
            let tmp = remnant[i] * low_term;
            coeff.push(remnant[i]);
            remnant[i - degree] += tmp;
        }
        coeff.reverse();
        Polynomial::new(coeff)
    }
}

#[derive(Debug, Clone)]
pub struct VanishingPolynomial<T: MyField> {
    degree: usize,
    shift: T,
}

impl<T: MyField> VanishingPolynomial<T> {
    pub fn new(coset: &Coset<T>) -> VanishingPolynomial<T> {
        let degree = coset.size();
        VanishingPolynomial {
            degree,
            shift: coset.shift().pow(degree),
        }
    }

    pub fn evaluation_at(&self, x: T) -> T {
        x.pow(self.degree) - self.shift
    }
}

#[derive(Debug, Clone)]
pub struct MultilinearPolynomial<T: MyField> {
    coefficients: Vec<T>,
}

impl<T: MyField> Add for MultilinearPolynomial<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut l = self.coefficients().clone();
        let mut r = rhs.coefficients.clone();
        if r.len() < l.len() {
            (l, r) = (r, l);
        }

        while l.len() < r.len() {
            l.extend(vec![T::from_int(0); l.len()]);
        }

        assert_eq!(l.len(), r.len());
        Self::new(
            l.iter()
                .zip(r.iter())
                .map(|(&f, &g)| f + g)
                .collect::<Vec<T>>(),
        )
    }
}

impl<T: MyField> Mul<T> for MultilinearPolynomial<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        Self::new(
            self.coefficients()
                .iter()
                .map(|&f| f * rhs)
                .collect::<Vec<T>>(),
        )
    }
}

impl<T: MyField> MultilinearPolynomial<T> {
    pub fn coefficients(&self) -> &Vec<T> {
        &self.coefficients
    }

    pub fn evaluate_from_hypercube(point: Vec<T>, mut poly_hypercube: Vec<T>) -> T {
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

    pub fn evaluate_hypercube(&self) -> Vec<T> {
        let log_n = self.variable_num();
        let n = self.coefficients.len();
        let rank = batch_bit_reverse(log_n);
        let mut res = self.coefficients.clone();
        for i in 0..n {
            if i < rank[i] {
                (res[i], res[rank[i]]) = (res[rank[i]], res[i]);
            }
        }
        for i in 0..log_n {
            let m = 1 << i;
            for j in (0..n).step_by(m * 2) {
                for k in 0..m {
                    let tmp = res[j + k];
                    res[j + k + m] += tmp;
                }
            }
        }
        res
    }

    pub fn new(coefficients: Vec<T>) -> Self {
        let len = coefficients.len();
        assert_eq!(len & (len - 1), 0);
        MultilinearPolynomial { coefficients }
    }

    pub fn folding(&self, parameter: T) -> Self {
        let coefficients = Self::folding_vector(&self.coefficients, parameter);
        MultilinearPolynomial { coefficients }
    }

    pub fn fold_self(&mut self, parameter: T) {
        self.coefficients = Self::folding_vector(&self.coefficients, parameter);
    }

    pub fn add_mult(&mut self, poly: &MultilinearPolynomial<T>, k: T) {
        assert_eq!(self.coefficients.len(), poly.coefficients.len());
        for (i, j) in self.coefficients.iter_mut().zip(poly.coefficients.iter()) {
            *i += k * j.clone();
        }
    }

    fn folding_vector(v: &Vec<T>, parameter: T) -> Vec<T> {
        let len = v.len();
        assert_eq!(len & (len - 1), 0);
        let mut res = vec![];
        for i in (0..v.len()).step_by(2) {
            res.push(v[i] + parameter * v[i + 1]);
        }
        res
    }

    pub fn random_polynomial(variable_num: usize) -> Self {
        MultilinearPolynomial {
            coefficients: (0..(1 << variable_num))
                .map(|_| MyField::random_element())
                .collect(),
        }
    }

    pub fn evaluate(&self, point: &Vec<T>) -> T {
        let len = self.coefficients.len();
        assert_eq!(1 << point.len(), self.coefficients.len());
        let mut res = self.coefficients.clone();
        for (index, coeff) in point.iter().enumerate() {
            for i in (0..len).step_by(2 << index) {
                let x = *coeff * res[i + (1 << index)];
                res[i] += x;
            }
        }
        res[0]
    }

    pub fn evaluate_as_polynomial(&self, point: T) -> T {
        let mut res = MyField::from_int(0);
        for i in self.coefficients.iter().rev() {
            res *= point;
            res += *i;
        }
        res
    }

    pub fn variable_num(&self) -> usize {
        self.coefficients.len().ilog2() as usize
    }
}

pub struct EqMultilinear<T: MyField> {
    b: Vec<T>,
}

impl<T: MyField> EqMultilinear<T> {
    pub fn evaluate_hypercube(&self) -> Vec<T> {
        let mut stack = vec![T::from_int(1)];
        for b in self.b.iter() {
            let new_stack = stack
                .iter()
                .flat_map(|prod| {
                    [
                        prod.clone() * (T::from_int(1) - b.clone()),
                        prod.clone() * b.clone(),
                    ]
                })
                .collect();
            stack = new_stack;
        }
        stack
    }

    pub fn new(b: Vec<T>) -> Self {
        EqMultilinear { b }
    }

    pub fn evaluate(&self, point: &Vec<T>) -> T {
        let mut res = T::from_int(1);
        for (x, b) in point.iter().zip(self.b.iter()) {
            res *=
                b.clone() * x.clone() + (T::from_int(1) - b.clone()) * (T::from_int(1) - x.clone());
        }
        res
    }
}

#[cfg(test)]
mod test {
    use crate::algebra::field::{ft255::Ft255, mersenne61_ext::Mersenne61Ext};

    use super::*;

    #[test]
    fn evaluation() {
        let coset = Coset::new(32, Ft255::random_element());
        let all_elements = coset.all_elements();
        let poly = Polynomial::random_polynomial(32);
        let eval = poly.evaluation_over_coset(&coset);
        for i in 0..coset.size() {
            assert_eq!(eval[i], poly.evaluation_at(all_elements[i]));
        }
        let poly = VanishingPolynomial::new(&coset);
        for i in coset.all_elements() {
            assert_eq!(Ft255::from_int(0), poly.evaluation_at(i));
        }
    }

    #[test]
    fn multilinear() {
        let poly = MultilinearPolynomial::random_polynomial(8);
        let point = (0..8).map(|_| Mersenne61Ext::random_element()).collect();
        let v = poly.evaluate(&point);
        let mut folding_poly = poly.clone();
        for parameter in point {
            folding_poly = folding_poly.folding(parameter);
        }
        assert_eq!(folding_poly.coefficients.len(), 1);
        assert_eq!(folding_poly.coefficients[0], v);
        let z = Mersenne61Ext::random_element();
        let beta = Mersenne61Ext::random_element();
        let folding_poly = poly.folding(z);
        let a = poly.evaluate_as_polynomial(beta);
        let b = poly.evaluate_as_polynomial(-beta);
        let c = folding_poly.evaluate_as_polynomial(beta * beta);
        let v = a + b + z * (a - b) * beta.inverse();
        assert_eq!(v * Mersenne61Ext::from_int(2).inverse(), c);

        let poly = MultilinearPolynomial::random_polynomial(8);
        let v = poly.evaluate_hypercube();
        for i in v.into_iter().enumerate() {
            let mut point = vec![];
            let mut t = i.0;
            for _j in 0..8 {
                point.push(Mersenne61Ext::from_int(t as u64 % 2));
                t /= 2;
            }
            point.reverse();
            assert_eq!(i.1, poly.evaluate(&point));
        }

        let point = (0..8).map(|_| Mersenne61Ext::random_element()).collect();
        let v = poly.evaluate(&point);
        let eq = EqMultilinear::new(point);
        let eq_hyper = eq.evaluate_hypercube();
        let mut sum = Mersenne61Ext::from_int(0);
        let poly_hyper = poly.evaluate_hypercube();
        for b in 0..(1 << 8) {
            sum += poly_hyper[b] * eq_hyper[b];
        }
        assert_eq!(v, sum);
    }
}
