use alloc::vec::Vec;

use crate::{
    Digest, Felt, Hasher, ZERO,
    block::BlockNumber,
    utils::serde::{ByteReader, ByteWriter, Deserializable, DeserializationError, Serializable},
};

/// The header of a block. It contains metadata about the block, commitments to the current
/// state of the chain and the hash of the proof that attests to the integrity of the chain.
///
/// A block header includes the following fields:
///
/// - `version` specifies the version of the protocol.
/// - `prev_block_commitment` is the hash of the previous block header.
/// - `block_num` is a unique sequential number of the current block.
/// - `chain_commitment` is a commitment to an MMR of the entire chain where each block is a leaf.
/// - `account_root` is a commitment to account database.
/// - `nullifier_root` is a commitment to the nullifier database.
/// - `note_root` is a commitment to all notes created in the current block.
/// - `tx_commitment` is a commitment to the set of transaction IDs which affected accounts in the
///   block.
/// - `tx_kernel_commitment` a commitment to all transaction kernels supported by this block.
/// - `proof_commitment` is the commitment of the block's STARK proof attesting to the correct state
///   transition.
/// - `timestamp` is the time when the block was created, in seconds since UNIX epoch. Current
///   representation is sufficient to represent time up to year 2106.
/// - `sub_commitment` is a sequential hash of all fields except the note_root.
/// - `commitment` is a 2-to-1 hash of the sub_commitment and the note_root.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct BlockHeader {
    version: u32,
    prev_block_commitment: Digest,
    block_num: BlockNumber,
    chain_commitment: Digest,
    account_root: Digest,
    nullifier_root: Digest,
    note_root: Digest,
    tx_commitment: Digest,
    tx_kernel_commitment: Digest,
    proof_commitment: Digest,
    timestamp: u32,
    sub_commitment: Digest,
    commitment: Digest,
}

impl BlockHeader {
    /// Creates a new block header.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        version: u32,
        prev_block_commitment: Digest,
        block_num: BlockNumber,
        chain_commitment: Digest,
        account_root: Digest,
        nullifier_root: Digest,
        note_root: Digest,
        tx_commitment: Digest,
        tx_kernel_commitment: Digest,
        proof_commitment: Digest,
        timestamp: u32,
    ) -> Self {
        // compute block sub commitment
        let sub_commitment = Self::compute_sub_commitment(
            version,
            prev_block_commitment,
            chain_commitment,
            account_root,
            nullifier_root,
            tx_commitment,
            tx_kernel_commitment,
            proof_commitment,
            timestamp,
            block_num,
        );

        // The sub commitment is merged with the note_root - hash(sub_commitment, note_root) to
        // produce the final hash. This is done to make the note_root easily accessible
        // without having to unhash the entire header. Having the note_root easily
        // accessible is useful when authenticating notes.
        let commitment = Hasher::merge(&[sub_commitment, note_root]);

        Self {
            version,
            prev_block_commitment,
            block_num,
            chain_commitment,
            account_root,
            nullifier_root,
            note_root,
            tx_commitment,
            tx_kernel_commitment,
            proof_commitment,
            timestamp,
            sub_commitment,
            commitment,
        }
    }

    // ACCESSORS
    // --------------------------------------------------------------------------------------------

    /// Returns the protocol version.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns the commitment of the block header.
    pub fn commitment(&self) -> Digest {
        self.commitment
    }

    /// Returns the sub commitment of the block header.
    ///
    /// The sub commitment is a sequential hash of all block header fields except the note root.
    /// This is used in the block commitment computation which is a 2-to-1 hash of the sub
    /// commitment and the note root [hash(sub_commitment, note_root)]. This procedure is used to
    /// make the note root easily accessible without having to unhash the entire header.
    pub fn sub_commitment(&self) -> Digest {
        self.sub_commitment
    }

    /// Returns the commitment to the previous block header.
    pub fn prev_block_commitment(&self) -> Digest {
        self.prev_block_commitment
    }

    /// Returns the block number.
    pub fn block_num(&self) -> BlockNumber {
        self.block_num
    }

    /// Returns the epoch to which this block belongs.
    ///
    /// This is the block number shifted right by [`BlockNumber::EPOCH_LENGTH_EXPONENT`].
    pub fn block_epoch(&self) -> u16 {
        self.block_num.block_epoch()
    }

    /// Returns the chain commitment.
    pub fn chain_commitment(&self) -> Digest {
        self.chain_commitment
    }

    /// Returns the account database root.
    pub fn account_root(&self) -> Digest {
        self.account_root
    }

    /// Returns the nullifier database root.
    pub fn nullifier_root(&self) -> Digest {
        self.nullifier_root
    }

    /// Returns the note root.
    pub fn note_root(&self) -> Digest {
        self.note_root
    }

    /// Returns the commitment to all transactions in this block.
    ///
    /// The commitment is computed as sequential hash of (`transaction_id`, `account_id`) tuples.
    /// This makes it possible for the verifier to link transaction IDs to the accounts which
    /// they were executed against.
    pub fn tx_commitment(&self) -> Digest {
        self.tx_commitment
    }

    /// Returns the transaction kernel commitment.
    ///
    /// The transaction kernel commitment is computed as a sequential hash of all transaction kernel
    /// hashes.
    pub fn tx_kernel_commitment(&self) -> Digest {
        self.tx_kernel_commitment
    }

    /// Returns the proof commitment.
    pub fn proof_commitment(&self) -> Digest {
        self.proof_commitment
    }

    /// Returns the timestamp at which the block was created, in seconds since UNIX epoch.
    pub fn timestamp(&self) -> u32 {
        self.timestamp
    }

    /// Returns the block number of the epoch block to which this block belongs.
    pub fn epoch_block_num(&self) -> BlockNumber {
        BlockNumber::from_epoch(self.block_epoch())
    }

    // HELPERS
    // --------------------------------------------------------------------------------------------

    /// Computes the sub commitment of the block header.
    ///
    /// The sub commitment is computed as a sequential hash of the following fields:
    /// `prev_block_commitment`, `chain_commitment`, `account_root`, `nullifier_root`, `note_root`,
    /// `tx_commitment`, `tx_kernel_commitment`, `proof_commitment`, `version`, `timestamp`,
    /// `block_num` (all fields except the `note_root`).
    #[allow(clippy::too_many_arguments)]
    fn compute_sub_commitment(
        version: u32,
        prev_block_commitment: Digest,
        chain_commitment: Digest,
        account_root: Digest,
        nullifier_root: Digest,
        tx_commitment: Digest,
        tx_kernel_commitment: Digest,
        proof_commitment: Digest,
        timestamp: u32,
        block_num: BlockNumber,
    ) -> Digest {
        let mut elements: Vec<Felt> = Vec::with_capacity(32);
        elements.extend_from_slice(prev_block_commitment.as_elements());
        elements.extend_from_slice(chain_commitment.as_elements());
        elements.extend_from_slice(account_root.as_elements());
        elements.extend_from_slice(nullifier_root.as_elements());
        elements.extend_from_slice(tx_commitment.as_elements());
        elements.extend_from_slice(tx_kernel_commitment.as_elements());
        elements.extend_from_slice(proof_commitment.as_elements());
        elements.extend([block_num.into(), version.into(), timestamp.into(), ZERO]);
        Hasher::hash_elements(&elements)
    }
}

// SERIALIZATION
// ================================================================================================

impl Serializable for BlockHeader {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        self.version.write_into(target);
        self.prev_block_commitment.write_into(target);
        self.block_num.write_into(target);
        self.chain_commitment.write_into(target);
        self.account_root.write_into(target);
        self.nullifier_root.write_into(target);
        self.note_root.write_into(target);
        self.tx_commitment.write_into(target);
        self.tx_kernel_commitment.write_into(target);
        self.proof_commitment.write_into(target);
        self.timestamp.write_into(target);
    }
}

impl Deserializable for BlockHeader {
    fn read_from<R: ByteReader>(source: &mut R) -> Result<Self, DeserializationError> {
        let version = source.read()?;
        let prev_block_commitment = source.read()?;
        let block_num = source.read()?;
        let chain_commitment = source.read()?;
        let account_root = source.read()?;
        let nullifier_root = source.read()?;
        let note_root = source.read()?;
        let tx_commitment = source.read()?;
        let tx_kernel_commitment = source.read()?;
        let proof_commitment = source.read()?;
        let timestamp = source.read()?;

        Ok(Self::new(
            version,
            prev_block_commitment,
            block_num,
            chain_commitment,
            account_root,
            nullifier_root,
            note_root,
            tx_commitment,
            tx_kernel_commitment,
            proof_commitment,
            timestamp,
        ))
    }
}

#[cfg(test)]
mod tests {
    use vm_core::Word;
    use winter_rand_utils::rand_array;

    use super::*;

    #[test]
    fn test_serde() {
        let chain_commitment: Word = rand_array();
        let note_root: Word = rand_array();
        let tx_kernel_commitment: Word = rand_array();
        let header = BlockHeader::mock(
            0,
            Some(chain_commitment.into()),
            Some(note_root.into()),
            &[],
            tx_kernel_commitment.into(),
        );
        let serialized = header.to_bytes();
        let deserialized = BlockHeader::read_from_bytes(&serialized).unwrap();

        assert_eq!(deserialized, header);
    }
}
