use alloc::{string::ToString, vec::Vec};

use miden_lib::{
    errors::tx_kernel_errors::{
        ERR_ACCOUNT_NONCE_DID_NOT_INCREASE_AFTER_STATE_CHANGE,
        ERR_EPILOGUE_TOTAL_NUMBER_OF_ASSETS_MUST_STAY_THE_SAME, ERR_TX_INVALID_EXPIRATION_DELTA,
    },
    transaction::{
        TransactionKernel,
        memory::{NOTE_MEM_SIZE, OUTPUT_NOTE_ASSET_COMMITMENT_OFFSET, OUTPUT_NOTE_SECTION_OFFSET},
    },
};
use miden_objects::{
    account::Account,
    transaction::{OutputNote, OutputNotes},
};
use vm_processor::{Felt, ONE, ProcessState};

use super::{ZERO, output_notes_data_procedure};
use crate::{
    TransactionContextBuilder, assert_execution_error, kernel_tests::tx::read_root_mem_word,
};

#[test]
fn test_epilogue() {
    let tx_context = TransactionContextBuilder::with_standard_account(ONE)
        .with_mock_notes_preserved()
        .build();

    let output_notes_data_procedure =
        output_notes_data_procedure(tx_context.expected_output_notes());

    let code = format!(
        "
        use.kernel::prologue
        use.kernel::account
        use.kernel::epilogue

        {output_notes_data_procedure}

        begin
            exec.prologue::prepare_transaction

            exec.create_mock_notes

            push.1
            exec.account::incr_nonce

            exec.epilogue::finalize_transaction

            # truncate the stack
            movupw.3 dropw movupw.3 dropw movup.9 drop
        end
        "
    );

    let process = tx_context
        .execute_code_with_assembler(
            &code,
            TransactionKernel::testing_assembler_with_mock_account(),
        )
        .unwrap();

    let final_account = Account::mock(
        tx_context.account().id().into(),
        tx_context.account().nonce() + ONE,
        TransactionKernel::assembler(),
    );

    let output_notes = OutputNotes::new(
        tx_context
            .expected_output_notes()
            .iter()
            .cloned()
            .map(OutputNote::Full)
            .collect(),
    )
    .unwrap();

    let mut expected_stack = Vec::with_capacity(16);
    expected_stack.extend(output_notes.commitment().as_elements().iter().rev());
    expected_stack.extend(final_account.commitment().as_elements().iter().rev());
    expected_stack.push(Felt::from(u32::MAX)); // Value for tx expiration block number
    expected_stack.extend((9..16).map(|_| ZERO));

    assert_eq!(
        *process.stack.build_stack_outputs().unwrap(),
        expected_stack.as_slice(),
        "Stack state after finalize_transaction does not contain the expected values"
    );

    assert_eq!(
        process.stack.depth(),
        16,
        "The stack must be truncated to 16 elements after finalize_transaction"
    );
}

#[test]
fn test_compute_output_note_id() {
    let tx_context = TransactionContextBuilder::with_standard_account(ONE)
        .with_mock_notes_preserved()
        .build();

    let output_notes_data_procedure =
        output_notes_data_procedure(tx_context.expected_output_notes());

    for (note, i) in tx_context.expected_output_notes().iter().zip(0u32..) {
        let code = format!(
            "
            use.kernel::prologue
            use.kernel::epilogue

            {output_notes_data_procedure}

            begin
                exec.prologue::prepare_transaction
                exec.create_mock_notes
                exec.epilogue::finalize_transaction

                # truncate the stack
                movupw.3 dropw movupw.3 dropw movup.9 drop
            end
            "
        );

        let process = &tx_context
            .execute_code_with_assembler(
                &code,
                TransactionKernel::testing_assembler_with_mock_account(),
            )
            .unwrap();

        assert_eq!(
            note.assets().commitment().as_elements(),
            read_root_mem_word(
                &process.into(),
                OUTPUT_NOTE_SECTION_OFFSET
                    + i * NOTE_MEM_SIZE
                    + OUTPUT_NOTE_ASSET_COMMITMENT_OFFSET
            ),
            "ASSET_COMMITMENT didn't match expected value",
        );

        assert_eq!(
            note.id().as_elements(),
            &read_root_mem_word(&process.into(), OUTPUT_NOTE_SECTION_OFFSET + i * NOTE_MEM_SIZE),
            "NOTE_ID didn't match expected value",
        );
    }
}

#[test]
fn test_epilogue_asset_preservation_violation_too_few_input() {
    let tx_context = TransactionContextBuilder::with_standard_account(ONE)
        .with_mock_notes_too_few_input()
        .build();

    let output_notes_data_procedure =
        output_notes_data_procedure(tx_context.expected_output_notes());

    let code = format!(
        "
        use.kernel::prologue
        use.test::account
        use.kernel::epilogue

        {output_notes_data_procedure}

        begin
            exec.prologue::prepare_transaction
            exec.create_mock_notes
            push.1
            call.account::incr_nonce
            exec.epilogue::finalize_transaction
            
            # truncate the stack
            movupw.3 dropw movupw.3 dropw movup.9 drop
        end
        "
    );

    let process = tx_context.execute_code_with_assembler(
        &code,
        TransactionKernel::testing_assembler_with_mock_account(),
    );
    assert_execution_error!(process, ERR_EPILOGUE_TOTAL_NUMBER_OF_ASSETS_MUST_STAY_THE_SAME);
}

#[test]
fn test_epilogue_asset_preservation_violation_too_many_fungible_input() {
    let tx_context = TransactionContextBuilder::with_standard_account(ONE)
        .with_mock_notes_too_many_fungible_input()
        .build();

    let output_notes_data_procedure =
        output_notes_data_procedure(tx_context.expected_output_notes());

    let code = format!(
        "
        use.kernel::prologue
        use.test::account
        use.kernel::epilogue

        {output_notes_data_procedure}

        begin
            exec.prologue::prepare_transaction
            exec.create_mock_notes
            push.1
            call.account::incr_nonce
            exec.epilogue::finalize_transaction
                        
            # truncate the stack
            movupw.3 dropw movupw.3 dropw movup.9 drop
        end
        "
    );

    let process = tx_context.execute_code_with_assembler(
        &code,
        TransactionKernel::testing_assembler_with_mock_account(),
    );

    assert_execution_error!(process, ERR_EPILOGUE_TOTAL_NUMBER_OF_ASSETS_MUST_STAY_THE_SAME);
}

#[test]
fn test_block_expiration_height_monotonically_decreases() {
    let tx_context = TransactionContextBuilder::with_standard_account(ONE).build();

    let test_pairs: [(u64, u64); 3] = [(9, 12), (18, 3), (20, 20)];
    let code_template = "
        use.kernel::prologue
        use.kernel::tx
        use.kernel::epilogue

        begin
            exec.prologue::prepare_transaction
            push.{value_1}
            exec.tx::update_expiration_block_num
            push.{value_2}
            exec.tx::update_expiration_block_num

            push.{min_value} exec.tx::get_expiration_delta assert_eq

            exec.epilogue::finalize_transaction
                        
            # truncate the stack
            movupw.3 dropw movupw.3 dropw movup.9 drop
        end
        ";

    for (v1, v2) in test_pairs {
        let code = &code_template
            .replace("{value_1}", &v1.to_string())
            .replace("{value_2}", &v2.to_string())
            .replace("{min_value}", &v2.min(v1).to_string());

        let process = &tx_context
            .execute_code_with_assembler(
                code,
                TransactionKernel::testing_assembler_with_mock_account(),
            )
            .unwrap();
        let process_state: ProcessState = process.into();

        // Expiry block should be set to transaction's block + the stored expiration delta
        // (which can only decrease, not increase)
        let expected_expiry =
            v1.min(v2) + tx_context.tx_inputs().block_header().block_num().as_u64();
        assert_eq!(process_state.get_stack_item(8).as_int(), expected_expiry);
    }
}

#[test]
fn test_invalid_expiration_deltas() {
    let tx_context = TransactionContextBuilder::with_standard_account(ONE).build();

    let test_values = [0u64, u16::MAX as u64 + 1, u32::MAX as u64];
    let code_template = "
        use.kernel::tx

        begin
            push.{value_1}
            exec.tx::update_expiration_block_num
        end
        ";

    for value in test_values {
        let code = &code_template.replace("{value_1}", &value.to_string());
        let process = tx_context.execute_code_with_assembler(
            code,
            TransactionKernel::testing_assembler_with_mock_account(),
        );

        assert_execution_error!(process, ERR_TX_INVALID_EXPIRATION_DELTA);
    }
}

#[test]
fn test_no_expiration_delta_set() {
    let tx_context = TransactionContextBuilder::with_standard_account(ONE).build();

    let code_template = "
    use.kernel::prologue
    use.kernel::epilogue
    use.kernel::tx

    begin
        exec.prologue::prepare_transaction

        exec.tx::get_expiration_delta assertz

        exec.epilogue::finalize_transaction
                    
        # truncate the stack
        movupw.3 dropw movupw.3 dropw movup.9 drop
    end
    ";

    let process = &tx_context
        .execute_code_with_assembler(
            code_template,
            TransactionKernel::testing_assembler_with_mock_account(),
        )
        .unwrap();
    let process_state: ProcessState = process.into();

    // Default value should be equal to u32::max, set in the prologue
    assert_eq!(process_state.get_stack_item(8).as_int() as u32, u32::MAX);
}

#[test]
fn test_epilogue_increment_nonce_success() {
    let tx_context = TransactionContextBuilder::with_standard_account(ONE)
        .with_mock_notes_preserved()
        .build();

    let output_notes_data_procedure =
        output_notes_data_procedure(tx_context.expected_output_notes());

    let code = format!(
        "
        use.kernel::prologue
        use.test::account
        use.kernel::epilogue

        {output_notes_data_procedure}

        begin
            exec.prologue::prepare_transaction

            exec.create_mock_notes

            push.1.2.3.4
            push.0
            call.account::set_item
            dropw

            push.1
            call.account::incr_nonce

            exec.epilogue::finalize_transaction

            # clean the stack
            dropw dropw dropw dropw
        end
        "
    );

    tx_context
        .execute_code_with_assembler(
            &code,
            TransactionKernel::testing_assembler_with_mock_account(),
        )
        .unwrap();
}

#[test]
fn test_epilogue_increment_nonce_violation() {
    let tx_context = TransactionContextBuilder::with_standard_account(ONE)
        .with_mock_notes_preserved()
        .build();

    let output_notes_data_procedure =
        output_notes_data_procedure(tx_context.expected_output_notes());

    let code = format!(
        "
        use.kernel::prologue
        use.test::account
        use.kernel::epilogue

        {output_notes_data_procedure}

        begin
            exec.prologue::prepare_transaction

            exec.create_mock_notes

            push.91.92.93.94
            push.0
            call.account::set_item
            dropw

            exec.epilogue::finalize_transaction

            # truncate the stack
            movupw.3 dropw movupw.3 dropw movup.9 drop
        end
        "
    );

    let process = tx_context.execute_code_with_assembler(
        &code,
        TransactionKernel::testing_assembler_with_mock_account(),
    );
    assert_execution_error!(process, ERR_ACCOUNT_NONCE_DID_NOT_INCREASE_AFTER_STATE_CHANGE)
}
