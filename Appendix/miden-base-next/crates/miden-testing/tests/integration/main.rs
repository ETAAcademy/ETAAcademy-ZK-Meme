extern crate alloc;

mod scripts;
mod wallet;

use miden_lib::transaction::TransactionKernel;
use miden_objects::{
    Felt, Word, ZERO,
    account::AccountId,
    asset::FungibleAsset,
    crypto::utils::Serializable,
    note::{Note, NoteAssets, NoteInputs, NoteMetadata, NoteRecipient, NoteScript, NoteType},
    testing::account_id::ACCOUNT_ID_SENDER,
    transaction::{ExecutedTransaction, ProvenTransaction},
};
use miden_tx::{
    LocalTransactionProver, ProvingOptions, TransactionProver, TransactionVerifier,
    TransactionVerifierError,
};
use vm_processor::utils::Deserializable;

// HELPER FUNCTIONS
// ================================================================================================

#[macro_export]
macro_rules! assert_transaction_executor_error {
    ($execution_result:expr, $expected_err_code:expr) => {
        match $execution_result {
            Err(miden_tx::TransactionExecutorError::TransactionProgramExecutionFailed(
                vm_processor::ExecutionError::FailedAssertion { clk: _, err_code, err_msg: _ }
            )) => {
                assert!(
                    err_code == $expected_err_code,
                    "Execution failed on assertion with an unexpected error code (Actual err_code: {}, expected {}).",
                    err_code, $expected_err_code
                );
            },
            Ok(_) => panic!("Execution was unexpectedly successful"),
            Err(other) => panic!("Execution error was not as expected: {}", other),
        }
    };
}

#[cfg(test)]
pub fn prove_and_verify_transaction(
    executed_transaction: ExecutedTransaction,
) -> Result<(), TransactionVerifierError> {
    let executed_transaction_id = executed_transaction.id();
    // Prove the transaction

    let proof_options = ProvingOptions::default();
    let prover = LocalTransactionProver::new(proof_options);
    let proven_transaction = prover.prove(executed_transaction.into()).unwrap();

    assert_eq!(proven_transaction.id(), executed_transaction_id);

    // Serialize & deserialize the ProvenTransaction
    let serialised_transaction = proven_transaction.to_bytes();
    let proven_transaction = ProvenTransaction::read_from_bytes(&serialised_transaction).unwrap();

    // Verify that the generated proof is valid
    let verifier = TransactionVerifier::new(miden_objects::MIN_PROOF_SECURITY_LEVEL);

    verifier.verify(&proven_transaction)
}

#[cfg(test)]
pub fn get_note_with_fungible_asset_and_script(
    fungible_asset: FungibleAsset,
    note_script: &str,
) -> Note {
    use miden_objects::note::NoteExecutionHint;

    let assembler = TransactionKernel::assembler().with_debug_mode(true);
    let note_script = NoteScript::compile(note_script, assembler).unwrap();
    const SERIAL_NUM: Word = [Felt::new(1), Felt::new(2), Felt::new(3), Felt::new(4)];
    let sender_id = AccountId::try_from(ACCOUNT_ID_SENDER).unwrap();

    let vault = NoteAssets::new(vec![fungible_asset.into()]).unwrap();
    let metadata =
        NoteMetadata::new(sender_id, NoteType::Public, 1.into(), NoteExecutionHint::Always, ZERO)
            .unwrap();
    let inputs = NoteInputs::new(vec![]).unwrap();
    let recipient = NoteRecipient::new(SERIAL_NUM, note_script, inputs);

    Note::new(vault, metadata, recipient)
}
