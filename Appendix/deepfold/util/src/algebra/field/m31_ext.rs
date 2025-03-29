use once_cell::sync::OnceCell;
use p3_field::{
    extension::Complex, AbstractExtensionField, AbstractField, Field, PrimeField32, TwoAdicField,
};
use rand::Rng;

use super::MyField;

type F = p3_field::extension::BinomialExtensionField<Complex<p3_mersenne_31::Mersenne31>, 2>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct M31ext(F);
impl std::ops::Neg for M31ext {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl std::ops::Add for M31ext {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for M31ext {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::ops::Sub for M31ext {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::SubAssign for M31ext {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl std::ops::Mul for M31ext {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl std::ops::MulAssign for M31ext {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl std::fmt::Display for M31ext {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

static ROOT_OF_UNITY: OnceCell<M31ext> = OnceCell::new();
static INVERSE_2: OnceCell<M31ext> = OnceCell::new();

impl MyField for M31ext {
    const FIELD_NAME: &'static str = "M31ext";
    const LOG_ORDER: u64 = F::TWO_ADICITY as u64 - 1;
    #[inline(always)]
    fn inverse_2() -> Self {
        INVERSE_2
            .get_or_init(|| Self::from_int(2).inverse())
            .clone()
    }
    #[inline(always)]
    fn root_of_unity() -> Self {
        ROOT_OF_UNITY
            .get_or_init(|| Self(F::two_adic_generator(F::TWO_ADICITY - 1)))
            .clone()
    }

    #[inline]
    fn from_int(x: u64) -> Self {
        Self(F::from_canonical_u64(x))
    }

    #[inline]
    fn from_hash(hash: [u8; crate::merkle_tree::MERKLE_ROOT_SIZE]) -> Self {
        todo!()
        // Self(F::deserialize(&hash.into()))
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    #[inline]
    fn random_element() -> Self {
        let mut rng = rand::thread_rng();
        Self(rng.gen::<F>())
    }
    #[inline(always)]
    fn inverse(&self) -> Self {
        Self(self.0.inverse())
    }

    #[inline]
    fn to_bytes(&self) -> Vec<u8> {
        let s: &[Complex<p3_mersenne_31::Mersenne31>] = self.0.as_base_slice();
        s.iter()
            .flat_map(|x| {
                x.real()
                    .as_canonical_u32()
                    .to_be_bytes()
                    .into_iter()
                    .chain(x.imag().as_canonical_u32().to_be_bytes().into_iter())
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::super::field_tests::*;
    use super::*;

    #[test]
    fn test() {
        println!("{}", M31ext::LOG_ORDER);
        add_and_sub::<M31ext>();
        mult_and_inverse::<M31ext>();
        assigns::<M31ext>();
        pow_and_generator::<M31ext>();
    }
}
