use miden_lib::{
    account::interface::NoteAccountCompatibility, note::well_known_note::WellKnownNote,
};
use miden_objects::{
    account::AccountId,
    block::BlockNumber,
    note::NoteId,
    transaction::{InputNote, InputNotes, TransactionArgs},
};
use winter_maybe_async::{maybe_async, maybe_await};

use super::{NoteAccountExecution, TransactionExecutor, TransactionExecutorError};

/// This struct performs input notes check against provided target account.
///
/// The check is performed using the [NoteConsumptionChecker::check_notes_consumability] procedure.
/// Essentially runs the transaction to make sure that provided input notes could be consumed by the
/// account.
pub struct NoteConsumptionChecker<'a>(&'a TransactionExecutor);

impl<'a> NoteConsumptionChecker<'a> {
    /// Creates a new [`NoteConsumptionChecker`] instance with the given transaction executor.
    pub fn new(tx_executor: &'a TransactionExecutor) -> Self {
        NoteConsumptionChecker(tx_executor)
    }

    /// Checks whether the provided input notes could be consumed by the provided account.
    ///
    /// This check consists of two main steps:
    /// - Statically check the notes: if all notes are either `P2ID` or `P2IDR` notes with correct
    ///   inputs, return `NoteAccountExecution::Success`.
    /// - Execute the transaction:
    ///   - Returns `NoteAccountExecution::Success` if the execution was successful.
    ///   - Returns `NoteAccountExecution::Failure` if some note returned an error. The fields
    ///     associated with `Failure` variant contains the ID of the failed note, a vector of IDs of
    ///     the notes, which were successfully executed, and the [TransactionExecutorError] if the
    ///     check failed durning the execution stage.
    #[maybe_async]
    pub fn check_notes_consumability(
        &self,
        target_account_id: AccountId,
        block_ref: BlockNumber,
        input_notes: InputNotes<InputNote>,
        tx_args: TransactionArgs,
    ) -> Result<NoteAccountExecution, TransactionExecutorError> {
        // Check input notes
        // ----------------------------------------------------------------------------------------

        let mut successful_notes = vec![];
        for note in input_notes.iter() {
            if let Some(well_known_note) = WellKnownNote::from_note(note.note()) {
                if let WellKnownNote::SWAP = well_known_note {
                    // if we encountered a SWAP note, then we have to execute the transaction
                    // anyway, but we should continue iterating to make sure that there are no
                    // P2ID(R) notes which return a `No`
                    continue;
                }

                match well_known_note.check_note_inputs(note.note(), target_account_id, block_ref) {
                    NoteAccountCompatibility::No => {
                        // if the check failed, return a `Failure` with the vector of successfully
                        // checked `P2ID` and `P2IDR` notes
                        return Ok(NoteAccountExecution::Failure {
                            failed_note_id: note.id(),
                            successful_notes,
                            error: None,
                        });
                    },
                    // this branch is unreachable, since we are handling the SWAP note separately,
                    // but as an extra precaution continue iterating over the notes and run the
                    // transaction to make sure the note which returned "Maybe" could be consumed
                    NoteAccountCompatibility::Maybe => continue,
                    NoteAccountCompatibility::Yes => {
                        // put the successfully checked `P2ID` or `P2IDR` note to the vector
                        successful_notes.push(note.id());
                    },
                }
            } else {
                // if we encountered not a well known note, then we have to execute the transaction
                // anyway, but we should continue iterating to make sure that there are no
                // P2ID(R) notes which return a `No`
                continue;
            }
        }

        // if all checked notes turned out to be either `P2ID` or `P2IDR` notes and all of them
        // passed, then we could safely return the `Success`
        if successful_notes.len() == input_notes.num_notes() {
            return Ok(NoteAccountExecution::Success);
        }

        // Execute transaction
        // ----------------------------------------------------------------------------------------
        maybe_await!(self.0.try_execute_notes(target_account_id, block_ref, input_notes, tx_args))
    }
}

/// Helper enum for getting a result of the well known note inputs check.
#[derive(Debug, PartialEq)]
pub enum NoteInputsCheck {
    Maybe,
    No { failed_note_id: NoteId },
}
