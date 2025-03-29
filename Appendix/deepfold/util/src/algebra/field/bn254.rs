use halo2curves::{
    bn256::Fr,
    ff::{Field, PrimeField},
};

use super::MyField;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Bn254Fr(Fr);

impl std::fmt::Display for Bn254Fr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl std::ops::Neg for Bn254Fr {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl std::ops::Add for Bn254Fr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Bn254Fr(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for Bn254Fr {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::ops::Sub for Bn254Fr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Bn254Fr(self.0 - rhs.0)
    }
}

impl std::ops::SubAssign for Bn254Fr {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl std::ops::Mul for Bn254Fr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Bn254Fr(self.0 * rhs.0)
    }
}

impl std::ops::MulAssign for Bn254Fr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
const ROOT_OF_UNITY: Bn254Fr = Bn254Fr(Fr::ROOT_OF_UNITY);
const INVERSE_2: Bn254Fr = Bn254Fr(Fr::TWO_INV);

impl MyField for Bn254Fr {
    const FIELD_NAME: &'static str = "Ft255";
    const LOG_ORDER: u64 = 41;

    fn inverse_2() -> Self {
        INVERSE_2
    }

    fn root_of_unity() -> Self {
        ROOT_OF_UNITY
    }

    #[inline]
    fn from_int(x: u64) -> Self {
        Self(x.into())
    }

    #[inline]
    fn from_hash(hash: [u8; crate::merkle_tree::MERKLE_ROOT_SIZE]) -> Self {
        Self(Fr::from_bytes(hash[..].try_into().unwrap()).unwrap())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == Fr::zero()
    }

    #[inline]
    fn random_element() -> Self {
        let rng = rand::thread_rng();
        Bn254Fr(Fr::random(rng))
    }

    fn inverse(&self) -> Self {
        Self(self.0.invert().unwrap())
    }

    #[inline]
    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_bytes().into()
    }
}
