use alloc::vec::Vec;

use crate::{
    Digest, Felt, Hasher, MAX_ASSETS_PER_NOTE, WORD_SIZE, Word, ZERO,
    asset::Asset,
    errors::NoteError,
    utils::serde::{ByteReader, ByteWriter, Deserializable, DeserializationError, Serializable},
};

// NOTE ASSETS
// ================================================================================================
/// An asset container for a note.
///
/// A note must contain at least 1 asset and can contain up to 256 assets. No duplicates are
/// allowed, but the order of assets is unspecified.
///
/// All the assets in a note can be reduced to a single commitment which is computed by
/// sequentially hashing the assets. Note that the same list of assets can result in two different
/// commitments if the asset ordering is different.
#[derive(Debug, Default, Clone)]
pub struct NoteAssets {
    assets: Vec<Asset>,
    hash: Digest,
}

impl NoteAssets {
    // CONSTANTS
    // --------------------------------------------------------------------------------------------

    /// The maximum number of assets which can be carried by a single note.
    pub const MAX_NUM_ASSETS: usize = MAX_ASSETS_PER_NOTE;

    // CONSTRUCTOR
    // --------------------------------------------------------------------------------------------

    /// Returns new [NoteAssets] constructed from the provided list of assets.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The list contains more than 256 assets.
    /// - There are duplicate assets in the list.
    pub fn new(assets: Vec<Asset>) -> Result<Self, NoteError> {
        if assets.len() > Self::MAX_NUM_ASSETS {
            return Err(NoteError::TooManyAssets(assets.len()));
        }

        // make sure all provided assets are unique
        for (i, asset) in assets.iter().enumerate().skip(1) {
            // for all assets except the first one, check if the asset is the same as any other
            // asset in the list, and if so return an error
            if assets[..i].iter().any(|a| a.is_same(asset)) {
                return Err(match asset {
                    Asset::Fungible(asset) => NoteError::DuplicateFungibleAsset(asset.faucet_id()),
                    Asset::NonFungible(asset) => NoteError::DuplicateNonFungibleAsset(*asset),
                });
            }
        }

        let hash = compute_asset_commitment(&assets);
        Ok(Self { assets, hash })
    }

    // PUBLIC ACCESSORS
    // --------------------------------------------------------------------------------------------

    /// Returns a commitment to the note's assets.
    pub fn commitment(&self) -> Digest {
        self.hash
    }

    /// Returns the number of assets.
    pub fn num_assets(&self) -> usize {
        self.assets.len()
    }

    /// Returns true if the number of assets is 0.
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    /// Returns an iterator over all assets.
    pub fn iter(&self) -> core::slice::Iter<Asset> {
        self.assets.iter()
    }

    /// Returns all assets represented as a vector of field elements.
    ///
    /// The vector is padded with ZEROs so that its length is a multiple of 8. This is useful
    /// because hashing the returned elements results in the note asset commitment.
    pub fn to_padded_assets(&self) -> Vec<Felt> {
        // if we have an odd number of assets with pad with a single word.
        let padded_len = if self.assets.len() % 2 == 0 {
            self.assets.len() * WORD_SIZE
        } else {
            (self.assets.len() + 1) * WORD_SIZE
        };

        // allocate a vector to hold the padded assets
        let mut padded_assets = Vec::with_capacity(padded_len * WORD_SIZE);

        // populate the vector with the assets
        padded_assets.extend(self.assets.iter().flat_map(|asset| <[Felt; 4]>::from(*asset)));

        // pad with an empty word if we have an odd number of assets
        padded_assets.resize(padded_len, ZERO);

        padded_assets
    }

    // STATE MUTATORS
    // --------------------------------------------------------------------------------------------

    /// Adds the provided asset to this list of note assets.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The same non-fungible asset is already in the list.
    /// - A fungible asset issued by the same faucet exists in the list and adding both assets
    ///   together results in an invalid asset.
    /// - Adding the asset to the list will push the list beyond the [Self::MAX_NUM_ASSETS] limit.
    pub fn add_asset(&mut self, asset: Asset) -> Result<(), NoteError> {
        // check if the asset issued by the faucet as the provided asset already exists in the
        // list of assets
        if let Some(own_asset) = self.assets.iter_mut().find(|a| a.is_same(&asset)) {
            match own_asset {
                Asset::Fungible(f_own_asset) => {
                    // if a fungible asset issued by the same faucet is found, try to add the
                    // the provided asset to it
                    let new_asset = f_own_asset
                        .add(asset.unwrap_fungible())
                        .map_err(NoteError::AddFungibleAssetBalanceError)?;
                    *own_asset = Asset::Fungible(new_asset);
                },
                Asset::NonFungible(nf_asset) => {
                    return Err(NoteError::DuplicateNonFungibleAsset(*nf_asset));
                },
            }
        } else {
            // if the asset is not in the list, add it to the list
            self.assets.push(asset);
            if self.assets.len() > Self::MAX_NUM_ASSETS {
                return Err(NoteError::TooManyAssets(self.assets.len()));
            }
        }

        self.hash = compute_asset_commitment(&self.assets);

        Ok(())
    }
}

impl PartialEq for NoteAssets {
    fn eq(&self, other: &Self) -> bool {
        self.assets == other.assets
    }
}

impl Eq for NoteAssets {}

// HELPER FUNCTIONS
// ================================================================================================

/// Returns a commitment to a note's assets.
///
/// The commitment is computed as a sequential hash of all assets (each asset represented by 4
/// field elements), padded to the next multiple of 2. If the asset list is empty, a default digest
/// is returned.
fn compute_asset_commitment(assets: &[Asset]) -> Digest {
    if assets.is_empty() {
        return Digest::default();
    }

    // If we have an odd number of assets we pad the vector with 4 zero elements. This is to
    // ensure the number of elements is a multiple of 8 - the size of the hasher rate.
    let word_capacity = if assets.len() % 2 == 0 {
        assets.len()
    } else {
        assets.len() + 1
    };
    let mut asset_elements = Vec::with_capacity(word_capacity * WORD_SIZE);

    for asset in assets.iter() {
        // convert the asset into field elements and add them to the list elements
        let asset_word: Word = (*asset).into();
        asset_elements.extend_from_slice(&asset_word);
    }

    // If we have an odd number of assets we pad the vector with 4 zero elements. This is to
    // ensure the number of elements is a multiple of 8 - the size of the hasher rate. This
    // simplifies hashing inside of the virtual machine when ingesting assets from a note.
    if assets.len() % 2 == 1 {
        asset_elements.extend_from_slice(&Word::default());
    }

    Hasher::hash_elements(&asset_elements)
}

// SERIALIZATION
// ================================================================================================

impl Serializable for NoteAssets {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        const _: () = assert!(NoteAssets::MAX_NUM_ASSETS <= u8::MAX as usize);
        debug_assert!(self.assets.len() <= NoteAssets::MAX_NUM_ASSETS);
        target.write_u8(self.assets.len().try_into().expect("Asset number must fit into `u8`"));
        target.write_many(&self.assets);
    }
}

impl Deserializable for NoteAssets {
    fn read_from<R: ByteReader>(source: &mut R) -> Result<Self, DeserializationError> {
        let count = source.read_u8()?;
        let assets = source.read_many::<Asset>(count.into())?;
        Self::new(assets).map_err(|e| DeserializationError::InvalidValue(format!("{e:?}")))
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::{NoteAssets, compute_asset_commitment};
    use crate::{
        Digest,
        account::AccountId,
        asset::{Asset, FungibleAsset},
        testing::account_id::ACCOUNT_ID_PRIVATE_FUNGIBLE_FAUCET,
    };

    #[test]
    fn add_asset() {
        let faucet_id = AccountId::try_from(ACCOUNT_ID_PRIVATE_FUNGIBLE_FAUCET).unwrap();

        let asset1 = Asset::Fungible(FungibleAsset::new(faucet_id, 100).unwrap());
        let asset2 = Asset::Fungible(FungibleAsset::new(faucet_id, 50).unwrap());

        // create empty assets
        let mut assets = NoteAssets::default();

        assert_eq!(assets.hash, Digest::default());

        // add asset1
        assert!(assets.add_asset(asset1).is_ok());
        assert_eq!(assets.assets, vec![asset1]);
        assert_eq!(assets.hash, compute_asset_commitment(&[asset1]));

        // add asset2
        assert!(assets.add_asset(asset2).is_ok());
        let expected_asset = Asset::Fungible(FungibleAsset::new(faucet_id, 150).unwrap());
        assert_eq!(assets.assets, vec![expected_asset]);
        assert_eq!(assets.hash, compute_asset_commitment(&[expected_asset]));
    }
}
