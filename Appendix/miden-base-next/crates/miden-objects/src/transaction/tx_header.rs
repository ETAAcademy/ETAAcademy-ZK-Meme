use alloc::vec::Vec;

use vm_processor::{DeserializationError, Digest};

use crate::{
    note::NoteId,
    transaction::{
        AccountId, InputNoteCommitment, Nullifier, OutputNote, ProvenTransaction, TransactionId,
    },
    utils::{ByteReader, ByteWriter, Deserializable, Serializable},
};

/// A transaction header derived from a
/// [`ProvenTransaction`](crate::transaction::ProvenTransaction).
///
/// The header is essentially a direct copy of the transaction's commitments, in particular the
/// initial and final account state commitment as well as all nullifiers of consumed notes and all
/// note IDs of created notes. While account updates may be aggregated and notes may be erased as
/// part of batch and block building, the header retains the original transaction's data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransactionHeader {
    id: TransactionId,
    account_id: AccountId,
    initial_state_commitment: Digest,
    final_state_commitment: Digest,
    input_notes: Vec<Nullifier>,
    output_notes: Vec<NoteId>,
}

impl TransactionHeader {
    // CONSTRUCTORS
    // --------------------------------------------------------------------------------------------

    /// Constructs a new [`TransactionHeader`] from the provided parameters.
    ///
    /// Note that the nullifiers of the input notes and note IDs of the output notes must be in the
    /// same order as they appeared in the transaction. This is ensured when constructing this type
    /// from a proven transaction, but cannot be validated during deserialization, hence additional
    /// validation is necessary.
    pub(crate) fn new(
        id: TransactionId,
        account_id: AccountId,
        initial_state_commitment: Digest,
        final_state_commitment: Digest,
        input_notes: Vec<Nullifier>,
        output_notes: Vec<NoteId>,
    ) -> Self {
        Self {
            id,
            account_id,
            initial_state_commitment,
            final_state_commitment,
            input_notes,
            output_notes,
        }
    }

    /// Constructs a new [`TransactionHeader`] from the provided parameters for testing purposes.
    #[cfg(any(feature = "testing", test))]
    pub fn new_unchecked(
        id: TransactionId,
        account_id: AccountId,
        initial_state_commitment: Digest,
        final_state_commitment: Digest,
        input_notes: Vec<Nullifier>,
        output_notes: Vec<NoteId>,
    ) -> Self {
        Self::new(
            id,
            account_id,
            initial_state_commitment,
            final_state_commitment,
            input_notes,
            output_notes,
        )
    }

    // PUBLIC ACCESSORS
    // --------------------------------------------------------------------------------------------

    /// Returns the unique identifier of this transaction.
    pub fn id(&self) -> TransactionId {
        self.id
    }

    /// Returns the ID of the account against which this transaction was executed.
    pub fn account_id(&self) -> AccountId {
        self.account_id
    }

    /// Returns a commitment to the state of the account before this update is applied.
    ///
    /// This is equal to [`Digest::default()`] for new accounts.
    pub fn initial_state_commitment(&self) -> Digest {
        self.initial_state_commitment
    }

    /// Returns a commitment to the state of the account after this update is applied.
    pub fn final_state_commitment(&self) -> Digest {
        self.final_state_commitment
    }

    /// Returns a reference to the nullifiers of the consumed notes.
    ///
    /// Note that the note may have been erased at the batch or block level, so it may not be
    /// present there.
    pub fn input_notes(&self) -> &[Nullifier] {
        &self.input_notes
    }

    /// Returns a reference to the notes created by the transaction.
    ///
    /// Note that the note may have been erased at the batch or block level, so it may not be
    /// present there.
    pub fn output_notes(&self) -> &[NoteId] {
        &self.output_notes
    }
}

impl From<&ProvenTransaction> for TransactionHeader {
    fn from(tx: &ProvenTransaction) -> Self {
        TransactionHeader::new(
            tx.id(),
            tx.account_id(),
            tx.account_update().initial_state_commitment(),
            tx.account_update().final_state_commitment(),
            tx.input_notes().iter().map(InputNoteCommitment::nullifier).collect(),
            tx.output_notes().iter().map(OutputNote::id).collect(),
        )
    }
}

// SERIALIZATION
// ================================================================================================

impl Serializable for TransactionHeader {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        self.id.write_into(target);
        self.account_id.write_into(target);
        self.initial_state_commitment.write_into(target);
        self.final_state_commitment.write_into(target);
        self.input_notes.write_into(target);
        self.output_notes.write_into(target);
    }
}

impl Deserializable for TransactionHeader {
    fn read_from<R: ByteReader>(source: &mut R) -> Result<Self, DeserializationError> {
        let id = <TransactionId>::read_from(source)?;
        let account_id = <AccountId>::read_from(source)?;
        let initial_state_commitment = <Digest>::read_from(source)?;
        let final_state_commitment = <Digest>::read_from(source)?;
        let input_notes = <Vec<Nullifier>>::read_from(source)?;
        let output_notes = <Vec<NoteId>>::read_from(source)?;

        Ok(Self::new(
            id,
            account_id,
            initial_state_commitment,
            final_state_commitment,
            input_notes,
            output_notes,
        ))
    }
}
