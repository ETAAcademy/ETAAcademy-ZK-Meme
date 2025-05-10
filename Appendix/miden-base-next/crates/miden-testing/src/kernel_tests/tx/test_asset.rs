use miden_lib::utils::word_to_masm_push_string;
use miden_objects::{
    account::AccountId,
    asset::NonFungibleAsset,
    testing::{
        account_id::ACCOUNT_ID_PUBLIC_FUNGIBLE_FAUCET,
        constants::{
            FUNGIBLE_ASSET_AMOUNT, FUNGIBLE_FAUCET_INITIAL_BALANCE, NON_FUNGIBLE_ASSET_DATA,
        },
    },
};
use vm_processor::ProcessState;

use super::{Felt, Hasher, ONE, Word};
use crate::TransactionContextBuilder;

#[test]
fn test_create_fungible_asset_succeeds() {
    let tx_context = TransactionContextBuilder::with_fungible_faucet(
        ACCOUNT_ID_PUBLIC_FUNGIBLE_FAUCET,
        ONE,
        Felt::new(FUNGIBLE_FAUCET_INITIAL_BALANCE),
    )
    .build();

    let code = format!(
        "
        use.kernel::prologue
        use.miden::asset

        begin
            exec.prologue::prepare_transaction

            # create fungible asset
            push.{FUNGIBLE_ASSET_AMOUNT}
            exec.asset::create_fungible_asset

            # truncate the stack
            swapw dropw
        end
        "
    );

    let process = &tx_context.execute_code(&code).unwrap();
    let process_state: ProcessState = process.into();

    let faucet_id = AccountId::try_from(ACCOUNT_ID_PUBLIC_FUNGIBLE_FAUCET).unwrap();
    assert_eq!(
        process_state.get_stack_word(0),
        Word::from([
            Felt::new(FUNGIBLE_ASSET_AMOUNT),
            Felt::new(0),
            faucet_id.suffix(),
            faucet_id.prefix().as_felt(),
        ])
    );
}

#[test]
fn test_create_non_fungible_asset_succeeds() {
    let tx_context = TransactionContextBuilder::with_non_fungible_faucet(
        NonFungibleAsset::mock_issuer().into(),
        ONE,
        false,
    )
    .build();

    let non_fungible_asset = NonFungibleAsset::mock(&NON_FUNGIBLE_ASSET_DATA);

    let code = format!(
        "
        use.kernel::prologue
        use.miden::asset

        begin
            exec.prologue::prepare_transaction

            # push non-fungible asset data hash onto the stack
            push.{non_fungible_asset_data_hash}
            exec.asset::create_non_fungible_asset

            # truncate the stack
            swapw dropw
        end
        ",
        non_fungible_asset_data_hash =
            word_to_masm_push_string(&Hasher::hash(&NON_FUNGIBLE_ASSET_DATA)),
    );

    let process = &tx_context.execute_code(&code).unwrap();
    let process_state: ProcessState = process.into();

    assert_eq!(process_state.get_stack_word(0), Word::from(non_fungible_asset));
}

#[test]
fn test_validate_non_fungible_asset() {
    let tx_context = TransactionContextBuilder::with_non_fungible_faucet(
        NonFungibleAsset::mock_issuer().into(),
        ONE,
        false,
    )
    .build();

    let non_fungible_asset = NonFungibleAsset::mock(&[1, 2, 3]);
    let encoded = Word::from(non_fungible_asset);

    let code = format!(
        "
        use.kernel::asset

        begin
            push.{asset} 
            exec.asset::validate_non_fungible_asset

            # truncate the stack
            swapw dropw
        end
        ",
        asset = word_to_masm_push_string(&encoded)
    );

    let process = &tx_context.execute_code(&code).unwrap();
    let process_state: ProcessState = process.into();

    assert_eq!(process_state.get_stack_word(0), encoded);
}
