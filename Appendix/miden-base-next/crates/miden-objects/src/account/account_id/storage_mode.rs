use alloc::string::String;
use core::{fmt, str::FromStr};

use crate::errors::AccountIdError;

// ACCOUNT STORAGE MODE
// ================================================================================================

pub(super) const PUBLIC: u8 = 0b00;
pub(super) const PRIVATE: u8 = 0b10;

/// Describes where the state of the account is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AccountStorageMode {
    /// The account's full state is stored on-chain.
    Public = PUBLIC,
    /// The account's state is stored off-chain, and only a commitment to it is stored on-chain.
    Private = PRIVATE,
}

impl AccountStorageMode {
    /// Returns `true` if the storage mode is [`Self::Public`], `false` otherwise.
    pub fn is_public(&self) -> bool {
        matches!(self, Self::Public)
    }

    /// Returns `true` if the storage mode is [`Self::Private`], `false` otherwise.
    pub fn is_private(&self) -> bool {
        matches!(self, Self::Private)
    }
}

impl fmt::Display for AccountStorageMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccountStorageMode::Public => write!(f, "public"),
            AccountStorageMode::Private => write!(f, "private"),
        }
    }
}

impl TryFrom<&str> for AccountStorageMode {
    type Error = AccountIdError;

    fn try_from(value: &str) -> Result<Self, AccountIdError> {
        match value.to_lowercase().as_str() {
            "public" => Ok(AccountStorageMode::Public),
            "private" => Ok(AccountStorageMode::Private),
            _ => Err(AccountIdError::UnknownAccountStorageMode(value.into())),
        }
    }
}

impl TryFrom<String> for AccountStorageMode {
    type Error = AccountIdError;

    fn try_from(value: String) -> Result<Self, AccountIdError> {
        AccountStorageMode::from_str(&value)
    }
}

impl FromStr for AccountStorageMode {
    type Err = AccountIdError;

    fn from_str(input: &str) -> Result<AccountStorageMode, AccountIdError> {
        AccountStorageMode::try_from(input)
    }
}

#[cfg(any(feature = "testing", test))]
impl rand::distr::Distribution<AccountStorageMode> for rand::distr::StandardUniform {
    /// Samples a uniformly random [`AccountStorageMode`] from the given `rng`.
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> AccountStorageMode {
        match rng.random_range(0..2) {
            0 => AccountStorageMode::Public,
            1 => AccountStorageMode::Private,
            _ => unreachable!("gen_range should not produce higher values"),
        }
    }
}
