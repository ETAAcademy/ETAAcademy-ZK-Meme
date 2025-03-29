use crate::merkle_tree::MERKLE_ROOT_SIZE;
use rand::RngCore;
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub mod bn254;
pub mod ft255;
pub mod m31_ext;
pub mod mersenne61_ext;

pub trait MyField:
    Sized
    + Clone
    + Copy
    + std::ops::Neg<Output = Self>
    + std::ops::Add<Output = Self>
    + std::ops::AddAssign
    + std::ops::Sub<Output = Self>
    + std::ops::SubAssign
    + std::ops::Mul<Output = Self>
    + std::ops::MulAssign
    + std::cmp::PartialEq
    + std::fmt::Display
    + std::fmt::Debug
    + std::marker::Send
    + std::marker::Sync
    + 'static
{
    const FIELD_NAME: &'static str;
    const LOG_ORDER: u64;
    fn from_int(x: u64) -> Self;
    fn random_element() -> Self;
    fn inverse(&self) -> Self;
    fn is_zero(&self) -> bool;
    fn to_bytes(&self) -> Vec<u8>;
    fn from_hash(hash: [u8; MERKLE_ROOT_SIZE]) -> Self;
    fn root_of_unity() -> Self;
    fn inverse_2() -> Self;
    #[inline(always)]
    fn get_generator(order: usize) -> Self {
        if (order & (order - 1)) != 0 || order > (1 << Self::LOG_ORDER) {
            panic!("invalid order");
        }
        let mut res = Self::root_of_unity();
        let mut i = 1u64 << Self::LOG_ORDER;
        while i > order as u64 {
            res *= res;
            i >>= 1;
        }
        res
    }
    #[inline(always)]
    fn pow(&self, mut n: usize) -> Self {
        let mut ret = Self::from_int(1);
        let mut base = self.clone();
        while n != 0 {
            if n % 2 == 1 {
                ret *= base;
            }
            base *= base;
            n >>= 1;
        }
        ret
    }
}

// Cauchy: define AnotherField, FftField trait for goldilocks and goldilocks64ext from DeepFold-Hyperplonk

pub trait AnotherField:
    Copy
    + Clone
    + Debug
    + Default
    + PartialEq
    + From<u32>
    + From<Self::BaseField>
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + MyField
{
    const NAME: &'static str;
    const SIZE: usize;
    const INV_2: Self;
    const ZERO: Self;
    const UNIT: Self;
    type BaseField: AnotherField;

    fn zero() -> Self;
    // fn is_zero(&self) -> bool;
    fn one() -> Self;
    fn random(rng: impl RngCore) -> Self;
    fn square(&self) -> Self {
        self.clone() * self.clone()
    }
    fn double(&self) -> Self {
        self.clone() + self.clone()
    }
    fn exp(&self, exponent: usize) -> Self;
    fn inv(&self) -> Option<Self>;
    fn add_base_elem(&self, rhs: Self::BaseField) -> Self;
    fn add_assign_base_elem(&mut self, rhs: Self::BaseField);
    fn mul_base_elem(&self, rhs: Self::BaseField) -> Self;
    fn mul_assign_base_elem(&mut self, rhs: Self::BaseField);
    fn from_uniform_bytes(bytes: &[u8; 32]) -> Self;
    fn serialize_into(&self, buffer: &mut [u8]);
    fn deserialize_from(buffer: &[u8]) -> Self;
}

pub trait FftField: AnotherField + From<Self::FftBaseField> {
    const LOG_ORDER: u32;
    const ROOT_OF_UNITY: Self;
    type FftBaseField: FftField<BaseField = Self::BaseField>;
}

#[inline]
pub fn as_bytes_vec<T: MyField>(s: &[T]) -> Vec<u8> {
    let mut res = vec![];
    for i in s {
        res.append(&mut i.to_bytes());
    }
    res
}

pub fn batch_inverse<T: MyField>(v: &Vec<T>) -> Vec<T> {
    let len = v.len();
    let mut res = v.clone();
    for i in 1..len {
        let x = res[i - 1];
        res[i] *= x;
    }
    let mut inv = res[len - 1].inverse();
    for i in (1..len).rev() {
        res[i] = inv * res[i - 1];
        inv *= v[i]
    }
    res[0] = inv;
    res
}

mod field_tests {
    use super::*;

    pub fn add_and_sub<T: MyField>() {
        for _i in 0..100 {
            let a = T::random_element();
            let b = T::random_element();
            let c = a + b - a;
            assert!(b == c)
        }
    }

    pub fn mult_and_inverse<T: MyField>() {
        for _i in 0..100 {
            let a = T::random_element();
            let b = a.inverse();
            assert_eq!(a * b, T::from_int(1));
            assert_eq!(b * a, T::from_int(1));
        }
        assert_eq!(T::inverse_2() * T::from_int(2), T::from_int(1));
    }

    pub fn assigns<T: MyField>() {
        for _i in 0..10 {
            let mut a = T::random_element();
            let aa = a;
            let b = T::random_element();
            a += b;
            assert_eq!(a, aa + b);
            a -= b;
            assert_eq!(a, aa);
            a *= b;
            assert_eq!(a, aa * b);
            a *= b.inverse();
            assert_eq!(a, aa);
            assert!((-a + a).is_zero());
        }
    }

    pub fn pow_and_generator<T: MyField>() {
        assert_eq!(T::get_generator(1), T::from_int(1));
        let x = T::get_generator(1 << 28);
        assert_eq!(x.pow(1 << 28), T::from_int(1));
        assert_ne!(x.pow(1 << 27), T::from_int(1));
    }
}
