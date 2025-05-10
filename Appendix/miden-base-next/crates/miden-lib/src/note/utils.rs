use miden_objects::{
    NoteError, Word,
    account::AccountId,
    asset::Asset,
    note::{NoteExecutionMode, NoteInputs, NoteRecipient, NoteTag, NoteType},
};

use super::well_known_note::WellKnownNote;

/// Creates a [NoteRecipient] for the P2ID note.
///
/// Notes created with this recipient will be P2ID notes consumable by the specified target
/// account.
pub fn build_p2id_recipient(
    target: AccountId,
    serial_num: Word,
) -> Result<NoteRecipient, NoteError> {
    let note_script = WellKnownNote::P2ID.script();
    let note_inputs = NoteInputs::new(vec![target.suffix(), target.prefix().as_felt()])?;

    Ok(NoteRecipient::new(serial_num, note_script, note_inputs))
}

/// Returns a note tag for a swap note with the specified parameters.
///
/// Use case ID for the returned tag is set to 0.
///
/// Tag payload is constructed by taking asset tags (8 bits of each faucet ID) and concatenating
/// them together as offered_asset_tag + requested_asset tag.
///
/// Network execution hint for the returned tag is set to `Local`.
pub fn build_swap_tag(
    note_type: NoteType,
    offered_asset: &Asset,
    requested_asset: &Asset,
) -> Result<NoteTag, NoteError> {
    const SWAP_USE_CASE_ID: u16 = 0;

    // Get bits 0..8 from the faucet IDs of both assets which will form the tag payload.
    let offered_asset_id: u64 = offered_asset.faucet_id_prefix().into();
    let offered_asset_tag = (offered_asset_id >> 56) as u8;

    let requested_asset_id: u64 = requested_asset.faucet_id_prefix().into();
    let requested_asset_tag = (requested_asset_id >> 56) as u8;

    let payload = ((offered_asset_tag as u16) << 8) | (requested_asset_tag as u16);

    let execution = NoteExecutionMode::Local;
    match note_type {
        NoteType::Public => NoteTag::for_public_use_case(SWAP_USE_CASE_ID, payload, execution),
        _ => NoteTag::for_local_use_case(SWAP_USE_CASE_ID, payload),
    }
}

#[cfg(test)]
mod tests {
    use miden_objects::{
        self,
        account::{AccountIdVersion, AccountStorageMode, AccountType, NetworkAccount},
        asset::{FungibleAsset, NonFungibleAsset, NonFungibleAssetDetails},
    };

    use super::*;

    #[test]
    fn swap_tag() {
        // Construct an ID that starts with 0xcdb1.
        let mut fungible_faucet_id_bytes = [0; 15];
        fungible_faucet_id_bytes[0] = 0xcd;
        fungible_faucet_id_bytes[1] = 0xb1;

        // Construct an ID that starts with 0xabec.
        let mut non_fungible_faucet_id_bytes = [0; 15];
        non_fungible_faucet_id_bytes[0] = 0xab;
        non_fungible_faucet_id_bytes[1] = 0xec;

        let offered_asset = Asset::Fungible(
            FungibleAsset::new(
                AccountId::dummy(
                    fungible_faucet_id_bytes,
                    AccountIdVersion::Version0,
                    AccountType::FungibleFaucet,
                    AccountStorageMode::Public,
                    NetworkAccount::Disabled,
                ),
                2500,
            )
            .unwrap(),
        );

        let requested_asset = Asset::NonFungible(
            NonFungibleAsset::new(
                &NonFungibleAssetDetails::new(
                    AccountId::dummy(
                        non_fungible_faucet_id_bytes,
                        AccountIdVersion::Version0,
                        AccountType::NonFungibleFaucet,
                        AccountStorageMode::Public,
                        NetworkAccount::Disabled,
                    )
                    .prefix(),
                    vec![0xaa, 0xbb, 0xcc, 0xdd],
                )
                .unwrap(),
            )
            .unwrap(),
        );

        // The fungible ID starts with 0xcdb1.
        // The non fungible ID starts with 0xabec.
        // The expected tag payload is thus 0xcdab.
        let expected_tag_payload = 0xcdab;

        let actual_tag =
            build_swap_tag(NoteType::Public, &offered_asset, &requested_asset).unwrap();

        // 0 is the SWAP use case ID.
        let expected_tag =
            NoteTag::for_public_use_case(0, expected_tag_payload, NoteExecutionMode::Local)
                .unwrap();

        assert_eq!(actual_tag, expected_tag);
    }
}
