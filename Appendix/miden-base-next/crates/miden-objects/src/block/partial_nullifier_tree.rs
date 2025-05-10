use crate::{
    Digest,
    block::{BlockNumber, NullifierTree, NullifierWitness},
    crypto::merkle::PartialSmt,
    errors::NullifierTreeError,
    note::Nullifier,
};

/// The partial sparse merkle tree containing the nullifiers of consumed notes.
///
/// A nullifier can only ever be spent once and its value in the tree is the block number at which
/// it was spent.
///
/// The tree guarantees that once a nullifier has been inserted into the tree, its block number does
/// not change. Note that inserting the nullifier multiple times with the same block number is
/// valid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialNullifierTree(PartialSmt);

impl PartialNullifierTree {
    /// Creates a new, empty partial nullifier tree.
    pub fn new() -> Self {
        PartialNullifierTree(PartialSmt::new())
    }

    /// Returns a new [`PartialNullifierTree`] instantiated with the provided entries.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - the merkle paths of the witnesses do not result in the same tree root.
    pub fn with_witnesses(
        witnesses: impl IntoIterator<Item = NullifierWitness>,
    ) -> Result<Self, NullifierTreeError> {
        let mut tree = Self::new();

        for witness in witnesses {
            tree.track_nullifier(witness)?;
        }

        Ok(tree)
    }

    /// Adds the given nullifier witness to the partial tree and tracks it. Once a nullifier has
    /// been added to the tree, it can be marked as spent using [`Self::mark_spent`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - after the witness' merkle path was added, the partial nullifier tree has a different root
    ///   than before it was added.
    pub fn track_nullifier(&mut self, witness: NullifierWitness) -> Result<(), NullifierTreeError> {
        let (path, leaf) = witness.into_proof().into_parts();
        self.0.add_path(leaf, path).map_err(NullifierTreeError::TreeRootConflict)
    }

    /// Marks the given nullifiers as spent at the given block number.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - a nullifier was already spent.
    /// - a nullifier is not tracked by this partial nullifier tree, that is, its
    ///   [`NullifierWitness`] was not added to the tree previously.
    pub fn mark_spent(
        &mut self,
        nullifiers: impl IntoIterator<Item = Nullifier>,
        block_num: BlockNumber,
    ) -> Result<(), NullifierTreeError> {
        for nullifier in nullifiers {
            self.mark_spent_single(nullifier, block_num)?;
        }

        Ok(())
    }

    /// Returns the root of the tree.
    pub fn root(&self) -> Digest {
        self.0.root()
    }

    /// Marks the given nullifier as spent at the given block number.
    ///
    /// # Errors
    ///
    /// See [`Self::mark_spent`] for the possible error conditions.
    fn mark_spent_single(
        &mut self,
        nullifier: Nullifier,
        block_num: BlockNumber,
    ) -> Result<(), NullifierTreeError> {
        let prev_nullifier_value = self
            .0
            .insert(nullifier.inner(), NullifierTree::block_num_to_leaf_value(block_num))
            .map_err(|source| NullifierTreeError::UntrackedNullifier { nullifier, source })?;

        if prev_nullifier_value != NullifierTree::UNSPENT_NULLIFIER {
            Err(NullifierTreeError::NullifierAlreadySpent(nullifier))
        } else {
            Ok(())
        }
    }
}

impl Default for PartialNullifierTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use miden_crypto::merkle::Smt;
    use winter_rand_utils::rand_array;

    use super::*;
    use crate::{EMPTY_WORD, Word};

    /// Test that using a stale nullifier witness together with a current one results in a different
    /// tree root and thus an error.
    #[test]
    fn partial_nullifier_tree_root_mismatch() {
        let key0 = Digest::from(Word::from(rand_array()));
        let key1 = Digest::from(Word::from(rand_array()));
        let key2 = Digest::from(Word::from(rand_array()));

        let value0 = EMPTY_WORD;
        let value1 = Word::from(rand_array());
        let value2 = EMPTY_WORD;

        let kv_pairs = vec![(key0, value0)];

        let mut full = Smt::with_entries(kv_pairs).unwrap();
        let stale_proof0 = full.open(&key0);
        // Insert a non-empty value so the nullifier tree's root changes.
        full.insert(key1, value1);
        full.insert(key2, value2);
        let proof2 = full.open(&key2);

        assert_ne!(stale_proof0.compute_root(), proof2.compute_root());

        let mut partial = PartialNullifierTree::new();

        partial.track_nullifier(NullifierWitness::new(stale_proof0)).unwrap();
        let error = partial.track_nullifier(NullifierWitness::new(proof2)).unwrap_err();

        assert_matches!(error, NullifierTreeError::TreeRootConflict(_));
    }

    #[test]
    fn nullifier_already_spent() {
        let nullifier1 = Nullifier::dummy(1);

        let block1 = BlockNumber::from(1);
        let block2 = BlockNumber::from(2);

        let tree = NullifierTree::with_entries([(nullifier1, block1)]).unwrap();

        let witness = tree.open(&nullifier1);

        let mut partial_tree = PartialNullifierTree::new();
        partial_tree.track_nullifier(witness).unwrap();

        // Attempt to insert nullifier 1 again at a different block number.
        let err = partial_tree.mark_spent([nullifier1], block2).unwrap_err();

        assert_matches!(err, NullifierTreeError::NullifierAlreadySpent(nullifier) if nullifier == nullifier1);
    }

    #[test]
    fn full_and_partial_nullifier_tree_consistency() {
        let nullifier1 = Nullifier::dummy(1);
        let nullifier2 = Nullifier::dummy(2);
        let nullifier3 = Nullifier::dummy(3);

        let block1 = BlockNumber::from(1);
        let block2 = BlockNumber::from(2);
        let block3 = BlockNumber::from(3);

        let mut tree =
            NullifierTree::with_entries([(nullifier1, block1), (nullifier2, block2)]).unwrap();

        let mut partial_tree = PartialNullifierTree::new();

        for nullifier in [nullifier1, nullifier2, nullifier3] {
            let witness = tree.open(&nullifier);
            partial_tree.track_nullifier(witness).unwrap();
        }

        assert_eq!(tree.root(), partial_tree.root());

        // Insert a new value into partial and full tree and assert the root is the same.
        tree.mark_spent(nullifier3, block3).unwrap();
        partial_tree.mark_spent([nullifier3], block3).unwrap();

        assert_eq!(tree.root(), partial_tree.root());
    }
}
