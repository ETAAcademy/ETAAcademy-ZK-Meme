use std::{collections::BTreeMap, vec::Vec};

use miden_lib::transaction::TransactionKernel;
use miden_objects::{
    Digest,
    account::AccountId,
    block::{
        AccountUpdateWitness, BlockAccountUpdate, BlockHeader, BlockNoteIndex, BlockNoteTree,
        BlockNumber, NullifierWitness, OutputNoteBatch, PartialAccountTree, PartialNullifierTree,
        ProposedBlock, ProvenBlock,
    },
    note::Nullifier,
    transaction::PartialBlockchain,
};

use crate::errors::ProvenBlockError;

// LOCAL BLOCK PROVER
// ================================================================================================

/// A local prover for blocks, proving a [`ProposedBlock`] and returning a [`ProvenBlock`].
#[derive(Clone)]
pub struct LocalBlockProver {}

impl LocalBlockProver {
    /// Creates a new [`LocalBlockProver`] instance.
    pub fn new(_proof_security_level: u32) -> Self {
        // TODO: This will eventually take the security level as a parameter, but until we verify
        // batches it is ignored.
        Self {}
    }

    /// Proves the provided [`ProposedBlock`] into a [`ProvenBlock`].
    ///
    /// For now this does not actually verify the batches or create a block proof, but will be added
    /// in the future.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - the account witnesses provided in the proposed block result in a different account tree
    ///   root than the contained previous block header commits to.
    /// - the nullifier witnesses provided in the proposed block result in a different nullifier
    ///   tree root than the contained previous block header commits to.
    /// - the account tree root in the previous block header does not match the root of the tree
    ///   computed from the account witnesses.
    /// - the nullifier tree root in the previous block header does not match the root of the tree
    ///   computed from the nullifier witnesses.
    pub fn prove(&self, proposed_block: ProposedBlock) -> Result<ProvenBlock, ProvenBlockError> {
        self.prove_without_batch_verification_inner(proposed_block)
    }

    /// Proves the provided [`ProposedBlock`] into a [`ProvenBlock`], **without verifying batches
    /// and proving the block**.
    ///
    /// This is exposed for testing purposes.
    #[cfg(any(feature = "testing", test))]
    pub fn prove_without_batch_verification(
        &self,
        proposed_block: ProposedBlock,
    ) -> Result<ProvenBlock, ProvenBlockError> {
        self.prove_without_batch_verification_inner(proposed_block)
    }

    /// Proves the provided [`ProposedBlock`] into a [`ProvenBlock`].
    ///
    /// The assumptions of this method are that the checks made by construction of a
    /// [`ProposedBlock`] are enforced.
    ///
    /// See [`Self::prove`] for more details.
    fn prove_without_batch_verification_inner(
        &self,
        proposed_block: ProposedBlock,
    ) -> Result<ProvenBlock, ProvenBlockError> {
        // Get the block number and timestamp of the new block and compute the tx commitment.
        // --------------------------------------------------------------------------------------------

        let block_num = proposed_block.block_num();
        let timestamp = proposed_block.timestamp();

        // Split the proposed block into its parts.
        // --------------------------------------------------------------------------------------------

        let (
            batches,
            account_updated_witnesses,
            output_note_batches,
            created_nullifiers,
            partial_blockchain,
            prev_block_header,
        ) = proposed_block.into_parts();

        let prev_block_commitment = prev_block_header.commitment();

        // Compute the root of the block note tree.
        // --------------------------------------------------------------------------------------------

        let note_tree = compute_block_note_tree(&output_note_batches);
        let note_root = note_tree.root();

        // Insert the created nullifiers into the nullifier tree to compute its new root.
        // --------------------------------------------------------------------------------------------

        let (created_nullifiers, new_nullifier_root) =
            compute_nullifiers(created_nullifiers, &prev_block_header, block_num)?;

        // Insert the state commitments of updated accounts into the account tree to compute its new
        // root.
        // --------------------------------------------------------------------------------------------

        let new_account_root =
            compute_account_root(&account_updated_witnesses, &prev_block_header)?;

        // Insert the previous block header into the block partial blockchain to get the new chain
        // commitment.
        // --------------------------------------------------------------------------------------------

        let new_chain_commitment = compute_chain_commitment(partial_blockchain, prev_block_header);

        // Transform the account update witnesses into block account updates.
        // --------------------------------------------------------------------------------------------

        let updated_accounts = account_updated_witnesses
            .into_iter()
            .map(|(account_id, update_witness)| {
                let (
                    _initial_state_commitment,
                    final_state_commitment,
                    // Note that compute_account_root took out this value so it should not be used.
                    _initial_state_proof,
                    details,
                ) = update_witness.into_parts();
                BlockAccountUpdate::new(account_id, final_state_commitment, details)
            })
            .collect();

        // Aggregate the verified transactions of all batches.
        // --------------------------------------------------------------------------------------------

        let txs = batches.into_transactions();
        let tx_commitment = txs.commitment();

        // Construct the new block header.
        // --------------------------------------------------------------------------------------------

        // Currently undefined and reserved for future use.
        // See miden-base/1155.
        let version = 0;
        let tx_kernel_commitment = TransactionKernel::kernel_commitment();

        // For now, we're not actually proving the block.
        let proof_commitment = Digest::default();

        let header = BlockHeader::new(
            version,
            prev_block_commitment,
            block_num,
            new_chain_commitment,
            new_account_root,
            new_nullifier_root,
            note_root,
            tx_commitment,
            tx_kernel_commitment,
            proof_commitment,
            timestamp,
        );

        // Construct the new proven block.
        // --------------------------------------------------------------------------------------------

        let proven_block = ProvenBlock::new_unchecked(
            header,
            updated_accounts,
            output_note_batches,
            created_nullifiers,
            txs,
        );

        Ok(proven_block)
    }
}

/// Computes the new nullifier root by inserting the nullifier witnesses into a partial nullifier
/// tree and marking each nullifier as spent in the given block number. Returns the list of
/// nullifiers and the new nullifier tree root.
fn compute_nullifiers(
    created_nullifiers: BTreeMap<Nullifier, NullifierWitness>,
    prev_block_header: &BlockHeader,
    block_num: BlockNumber,
) -> Result<(Vec<Nullifier>, Digest), ProvenBlockError> {
    // If no nullifiers were created, the nullifier tree root is unchanged.
    if created_nullifiers.is_empty() {
        return Ok((Vec::new(), prev_block_header.nullifier_root()));
    }

    let nullifiers: Vec<Nullifier> = created_nullifiers.keys().copied().collect();

    let mut partial_nullifier_tree = PartialNullifierTree::new();

    // First, reconstruct the current nullifier tree with the merkle paths of the nullifiers we want
    // to update.
    // Due to the guarantees of ProposedBlock we can safely assume that each nullifier is mapped to
    // its corresponding nullifier witness, so we don't have to check again whether they match.
    for witness in created_nullifiers.into_values() {
        partial_nullifier_tree
            .track_nullifier(witness)
            .map_err(ProvenBlockError::NullifierWitnessRootMismatch)?;
    }

    // Check the nullifier tree root in the previous block header matches the reconstructed tree's
    // root.
    if prev_block_header.nullifier_root() != partial_nullifier_tree.root() {
        return Err(ProvenBlockError::StaleNullifierTreeRoot {
            prev_block_nullifier_root: prev_block_header.nullifier_root(),
            stale_nullifier_root: partial_nullifier_tree.root(),
        });
    }

    // Second, mark each nullifier as spent in the tree. Note that checking whether each nullifier
    // is unspent is checked as part of the proposed block.

    // SAFETY: As mentioned above, we can safely assume that each nullifier's witness was
    // added and every nullifier should be tracked by the partial tree and
    // therefore updatable.
    partial_nullifier_tree.mark_spent(nullifiers.iter().copied(), block_num).expect(
      "nullifiers' merkle path should have been added to the partial tree and the nullifiers should be unspent",
    );

    Ok((nullifiers, partial_nullifier_tree.root()))
}

/// Adds the commitment of the previous block header to the partial blockchain to compute the new
/// chain commitment.
fn compute_chain_commitment(
    mut partial_blockchain: PartialBlockchain,
    prev_block_header: BlockHeader,
) -> Digest {
    // SAFETY: This does not panic as long as the block header we're adding is the next one in the
    // chain which is validated as part of constructing a `ProposedBlock`.
    partial_blockchain.add_block(prev_block_header, true);
    partial_blockchain.peaks().hash_peaks()
}

/// Computes the new account tree root after the given updates.
///
/// It uses a PartialMerkleTree for now while we use a SimpleSmt for the account tree. Once that is
/// updated to an Smt, we can use a PartialSmt instead.
fn compute_account_root(
    updated_accounts: &[(AccountId, AccountUpdateWitness)],
    prev_block_header: &BlockHeader,
) -> Result<Digest, ProvenBlockError> {
    // If no accounts were updated, the account tree root is unchanged.
    if updated_accounts.is_empty() {
        return Ok(prev_block_header.account_root());
    }

    // First reconstruct the current account tree from the provided merkle paths.
    // If a witness points to a leaf where multiple account IDs share the same prefix, this will
    // return an error.
    let mut partial_account_tree = PartialAccountTree::with_witnesses(
        updated_accounts.iter().map(|(_, update_witness)| update_witness.to_witness()),
    )
    .map_err(|source| ProvenBlockError::AccountWitnessTracking { source })?;

    // Check the account tree root in the previous block header matches the reconstructed tree's
    // root.
    if prev_block_header.account_root() != partial_account_tree.root() {
        return Err(ProvenBlockError::StaleAccountTreeRoot {
            prev_block_account_root: prev_block_header.account_root(),
            stale_account_root: partial_account_tree.root(),
        });
    }

    // Second, update the account tree by inserting the new final account state commitments to
    // compute the new root of the account tree.
    // If an account ID's prefix already exists in the tree, this will return an error.
    // Note that we have inserted all witnesses that we want to update into the partial account
    // tree, so we should not run into the untracked key error.
    partial_account_tree
        .upsert_state_commitments(updated_accounts.iter().map(|(account_id, update_witness)| {
            (*account_id, update_witness.final_state_commitment())
        }))
        .map_err(|source| ProvenBlockError::AccountIdPrefixDuplicate { source })?;

    Ok(partial_account_tree.root())
}

/// Compute the block note tree from the output note batches.
fn compute_block_note_tree(output_note_batches: &[OutputNoteBatch]) -> BlockNoteTree {
    let output_notes_iter =
        output_note_batches.iter().enumerate().flat_map(|(batch_idx, notes)| {
            notes.iter().map(move |(note_idx_in_batch, note)| {
                (
                    // SAFETY: The proposed block contains at most the max allowed number of
                    // batches and each batch is guaranteed to contain at most
                    // the max allowed number of output notes.
                    BlockNoteIndex::new(batch_idx, *note_idx_in_batch)
                        .expect("max batches in block and max notes in batches should be enforced"),
                    note.id(),
                    *note.metadata(),
                )
            })
        });

    // SAFETY: We only construct proposed blocks that:
    // - do not contain duplicates
    // - contain at most the max allowed number of batches and each batch is guaranteed to contain
    //   at most the max allowed number of output notes.
    BlockNoteTree::with_entries(output_notes_iter)
        .expect("the output notes of the block should not contain duplicates and contain at most the allowed maximum")
}
