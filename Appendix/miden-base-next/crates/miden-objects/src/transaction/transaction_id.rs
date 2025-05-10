use alloc::string::String;
use core::fmt::{Debug, Display};

use super::{Digest, ExecutedTransaction, Felt, Hasher, ProvenTransaction, WORD_SIZE, Word, ZERO};
use crate::utils::serde::{
    ByteReader, ByteWriter, Deserializable, DeserializationError, Serializable,
};

// TRANSACTION ID
// ================================================================================================

/// A unique identifier of a transaction.
///
/// Transaction ID is computed as:
///
/// hash(init_account_commitment, final_account_commitment, input_notes_commitment,
/// output_notes_commitment)
///
/// This achieves the following properties:
/// - Transactions are identical if and only if they have the same ID.
/// - Computing transaction ID can be done solely from public transaction data.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TransactionId(Digest);

impl TransactionId {
    /// Returns a new [TransactionId] instantiated from the provided transaction components.
    pub fn new(
        init_account_commitment: Digest,
        final_account_commitment: Digest,
        input_notes_commitment: Digest,
        output_notes_commitment: Digest,
    ) -> Self {
        let mut elements = [ZERO; 4 * WORD_SIZE];
        elements[..4].copy_from_slice(init_account_commitment.as_elements());
        elements[4..8].copy_from_slice(final_account_commitment.as_elements());
        elements[8..12].copy_from_slice(input_notes_commitment.as_elements());
        elements[12..].copy_from_slice(output_notes_commitment.as_elements());
        Self(Hasher::hash_elements(&elements))
    }

    /// Returns the elements representation of this transaction ID.
    pub fn as_elements(&self) -> &[Felt] {
        self.0.as_elements()
    }

    /// Returns the byte representation of this transaction ID.
    pub fn as_bytes(&self) -> [u8; 32] {
        self.0.as_bytes()
    }

    /// Returns a big-endian, hex-encoded string.
    pub fn to_hex(&self) -> String {
        self.0.to_hex()
    }

    /// Returns the digest defining this transaction ID.
    pub fn inner(&self) -> Digest {
        self.0
    }
}

impl Debug for TransactionId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl Display for TransactionId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

// CONVERSIONS INTO TRANSACTION ID
// ================================================================================================

impl From<&ProvenTransaction> for TransactionId {
    fn from(tx: &ProvenTransaction) -> Self {
        Self::new(
            tx.account_update().initial_state_commitment(),
            tx.account_update().final_state_commitment(),
            tx.input_notes().commitment(),
            tx.output_notes().commitment(),
        )
    }
}

impl From<&ExecutedTransaction> for TransactionId {
    fn from(tx: &ExecutedTransaction) -> Self {
        let input_notes_commitment = tx.input_notes().commitment();
        let output_notes_commitment = tx.output_notes().commitment();
        Self::new(
            tx.initial_account().init_commitment(),
            tx.final_account().commitment(),
            input_notes_commitment,
            output_notes_commitment,
        )
    }
}

impl From<Word> for TransactionId {
    fn from(value: Word) -> Self {
        Self(value.into())
    }
}

impl From<Digest> for TransactionId {
    fn from(value: Digest) -> Self {
        Self(value)
    }
}

// CONVERSIONS FROM TRANSACTION ID
// ================================================================================================

impl From<TransactionId> for Word {
    fn from(id: TransactionId) -> Self {
        id.0.into()
    }
}

impl From<TransactionId> for [u8; 32] {
    fn from(id: TransactionId) -> Self {
        id.0.into()
    }
}

impl From<&TransactionId> for Word {
    fn from(id: &TransactionId) -> Self {
        id.0.into()
    }
}

impl From<&TransactionId> for [u8; 32] {
    fn from(id: &TransactionId) -> Self {
        id.0.into()
    }
}

// SERIALIZATION
// ================================================================================================

impl Serializable for TransactionId {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        target.write_bytes(&self.0.to_bytes());
    }
}

impl Deserializable for TransactionId {
    fn read_from<R: ByteReader>(source: &mut R) -> Result<Self, DeserializationError> {
        let id = Digest::read_from(source)?;
        Ok(Self(id))
    }
}
