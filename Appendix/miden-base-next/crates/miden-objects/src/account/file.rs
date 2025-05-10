#[cfg(feature = "std")]
use std::{
    fs::{self, File},
    io::{self, Read},
    path::Path,
    vec::Vec,
};

use miden_crypto::utils::SliceReader;

use super::{
    super::utils::serde::{
        ByteReader, ByteWriter, Deserializable, DeserializationError, Serializable,
    },
    Account, AuthSecretKey, Word,
};

const MAGIC: &str = "acct";

// ACCOUNT FILE
// ================================================================================================

/// Account file contains a complete description of an account, including the [Account] struct as
/// well as account seed and account authentication info.
///
/// The intent of this struct is to provide an easy way to serialize and deserialize all
/// account-related data as a single unit (e.g., to/from files).
#[derive(Debug, Clone)]
pub struct AccountFile {
    pub account: Account,
    pub account_seed: Option<Word>,
    pub auth_secret_key: AuthSecretKey,
}

impl AccountFile {
    pub fn new(account: Account, account_seed: Option<Word>, auth: AuthSecretKey) -> Self {
        Self {
            account,
            account_seed,
            auth_secret_key: auth,
        }
    }
}

#[cfg(feature = "std")]
impl AccountFile {
    /// Serializes and writes binary [AccountFile] to specified file
    pub fn write(&self, filepath: impl AsRef<Path>) -> io::Result<()> {
        fs::write(filepath, self.to_bytes())
    }

    /// Reads from file and tries to deserialize an [AccountFile]
    pub fn read(filepath: impl AsRef<Path>) -> io::Result<Self> {
        let mut file = File::open(filepath)?;
        let mut buffer = Vec::new();

        file.read_to_end(&mut buffer)?;
        let mut reader = SliceReader::new(&buffer);

        Ok(AccountFile::read_from(&mut reader).map_err(|_| io::ErrorKind::InvalidData)?)
    }
}

// SERIALIZATION
// ================================================================================================

impl Serializable for AccountFile {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        target.write_bytes(MAGIC.as_bytes());
        let AccountFile {
            account,
            account_seed,
            auth_secret_key: auth,
        } = self;

        account.write_into(target);
        account_seed.write_into(target);
        auth.write_into(target);
    }
}

impl Deserializable for AccountFile {
    fn read_from<R: ByteReader>(source: &mut R) -> Result<Self, DeserializationError> {
        let magic_value = source.read_string(4)?;
        if magic_value != MAGIC {
            return Err(DeserializationError::InvalidValue(format!(
                "invalid account file marker: {magic_value}"
            )));
        }
        let account = Account::read_from(source)?;
        let account_seed = <Option<Word>>::read_from(source)?;
        let auth_secret_key = AuthSecretKey::read_from(source)?;

        Ok(Self::new(account, account_seed, auth_secret_key))
    }

    fn read_from_bytes(bytes: &[u8]) -> Result<Self, DeserializationError> {
        Self::read_from(&mut SliceReader::new(bytes))
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use miden_crypto::{
        dsa::rpo_falcon512::SecretKey,
        utils::{Deserializable, Serializable},
    };
    use storage::AccountStorage;
    #[cfg(feature = "std")]
    use tempfile::tempdir;

    use super::AccountFile;
    use crate::{
        account::{Account, AccountCode, AccountId, AuthSecretKey, Felt, Word, storage},
        asset::AssetVault,
        testing::account_id::ACCOUNT_ID_REGULAR_PUBLIC_ACCOUNT_IMMUTABLE_CODE,
    };

    fn build_account_file() -> AccountFile {
        let id = AccountId::try_from(ACCOUNT_ID_REGULAR_PUBLIC_ACCOUNT_IMMUTABLE_CODE).unwrap();
        let code = AccountCode::mock();

        // create account and auth
        let vault = AssetVault::new(&[]).unwrap();
        let storage = AccountStorage::new(vec![]).unwrap();
        let nonce = Felt::new(0);
        let account = Account::from_parts(id, vault, storage, code, nonce);
        let account_seed = Some(Word::default());
        let auth_secret_key = AuthSecretKey::RpoFalcon512(SecretKey::new());

        AccountFile::new(account, account_seed, auth_secret_key)
    }

    #[test]
    fn test_serde() {
        let account_file = build_account_file();
        let serialized = account_file.to_bytes();
        let deserialized = AccountFile::read_from_bytes(&serialized).unwrap();
        assert_eq!(deserialized.account, account_file.account);
        assert_eq!(deserialized.account_seed, account_file.account_seed);
        assert_eq!(
            deserialized.auth_secret_key.to_bytes(),
            account_file.auth_secret_key.to_bytes()
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_serde_file() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("account_file.mac");

        let account_file = build_account_file();
        account_file.write(filepath.as_path()).unwrap();
        let deserialized = AccountFile::read(filepath.as_path()).unwrap();

        assert_eq!(deserialized.account, account_file.account);
        assert_eq!(deserialized.account_seed, account_file.account_seed);
        assert_eq!(
            deserialized.auth_secret_key.to_bytes(),
            account_file.auth_secret_key.to_bytes()
        );
    }
}
