use core::fmt;

use super::TransactionEventError;

// CONSTANTS
// ================================================================================================

// TRANSACTION EVENT
// ================================================================================================

const ACCOUNT_VAULT_BEFORE_ADD_ASSET: u32 = 0x2_0000; // 131072
const ACCOUNT_VAULT_AFTER_ADD_ASSET: u32 = 0x2_0001; // 131073

const ACCOUNT_VAULT_BEFORE_REMOVE_ASSET: u32 = 0x2_0002; // 131074
const ACCOUNT_VAULT_AFTER_REMOVE_ASSET: u32 = 0x2_0003; // 131075

const ACCOUNT_STORAGE_BEFORE_SET_ITEM: u32 = 0x2_0004; // 131076
const ACCOUNT_STORAGE_AFTER_SET_ITEM: u32 = 0x2_0005; // 131077

const ACCOUNT_STORAGE_BEFORE_SET_MAP_ITEM: u32 = 0x2_0006; // 131078
const ACCOUNT_STORAGE_AFTER_SET_MAP_ITEM: u32 = 0x2_0007; // 131079

const ACCOUNT_BEFORE_INCREMENT_NONCE: u32 = 0x2_0008; // 131080
const ACCOUNT_AFTER_INCREMENT_NONCE: u32 = 0x2_0009; // 131081

const ACCOUNT_PUSH_PROCEDURE_INDEX: u32 = 0x2_000a; // 131082

const NOTE_BEFORE_CREATED: u32 = 0x2_000b; // 131083
const NOTE_AFTER_CREATED: u32 = 0x2_000c; // 131084

const NOTE_BEFORE_ADD_ASSET: u32 = 0x2_000d; // 131085
const NOTE_AFTER_ADD_ASSET: u32 = 0x2_000e; // 131086

const FALCON_SIG_TO_STACK: u32 = 0x2_000f; // 131087

const PROLOGUE_START: u32 = 0x2_0010; // 131088
const PROLOGUE_END: u32 = 0x2_0011; // 131089

const NOTES_PROCESSING_START: u32 = 0x2_0012; // 131090
const NOTES_PROCESSING_END: u32 = 0x2_0013; // 131091

const NOTE_EXECUTION_START: u32 = 0x2_0014; // 131092
const NOTE_EXECUTION_END: u32 = 0x2_0015; // 131093

const TX_SCRIPT_PROCESSING_START: u32 = 0x2_0016; // 131094
const TX_SCRIPT_PROCESSING_END: u32 = 0x2_0017; // 131095

const EPILOGUE_START: u32 = 0x2_0018; // 131096
const EPILOGUE_END: u32 = 0x2_0019; // 131097

/// Events which may be emitted by a transaction kernel.
///
/// The events are emitted via the `emit.<event_id>` instruction. The event ID is a 32-bit
/// unsigned integer which is used to identify the event type. For events emitted by the
/// transaction kernel, the event_id is structured as follows:
/// - The upper 16 bits of the event ID are set to 2.
/// - The lower 16 bits represent a unique event ID within the transaction kernel.
#[repr(u32)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TransactionEvent {
    AccountVaultBeforeAddAsset = ACCOUNT_VAULT_BEFORE_ADD_ASSET,
    AccountVaultAfterAddAsset = ACCOUNT_VAULT_AFTER_ADD_ASSET,

    AccountVaultBeforeRemoveAsset = ACCOUNT_VAULT_BEFORE_REMOVE_ASSET,
    AccountVaultAfterRemoveAsset = ACCOUNT_VAULT_AFTER_REMOVE_ASSET,

    AccountStorageBeforeSetItem = ACCOUNT_STORAGE_BEFORE_SET_ITEM,
    AccountStorageAfterSetItem = ACCOUNT_STORAGE_AFTER_SET_ITEM,

    AccountStorageBeforeSetMapItem = ACCOUNT_STORAGE_BEFORE_SET_MAP_ITEM,
    AccountStorageAfterSetMapItem = ACCOUNT_STORAGE_AFTER_SET_MAP_ITEM,

    AccountBeforeIncrementNonce = ACCOUNT_BEFORE_INCREMENT_NONCE,
    AccountAfterIncrementNonce = ACCOUNT_AFTER_INCREMENT_NONCE,

    AccountPushProcedureIndex = ACCOUNT_PUSH_PROCEDURE_INDEX,

    NoteBeforeCreated = NOTE_BEFORE_CREATED,
    NoteAfterCreated = NOTE_AFTER_CREATED,

    NoteBeforeAddAsset = NOTE_BEFORE_ADD_ASSET,
    NoteAfterAddAsset = NOTE_AFTER_ADD_ASSET,

    FalconSigToStack = FALCON_SIG_TO_STACK,

    PrologueStart = PROLOGUE_START,
    PrologueEnd = PROLOGUE_END,

    NotesProcessingStart = NOTES_PROCESSING_START,
    NotesProcessingEnd = NOTES_PROCESSING_END,

    NoteExecutionStart = NOTE_EXECUTION_START,
    NoteExecutionEnd = NOTE_EXECUTION_END,

    TxScriptProcessingStart = TX_SCRIPT_PROCESSING_START,
    TxScriptProcessingEnd = TX_SCRIPT_PROCESSING_END,

    EpilogueStart = EPILOGUE_START,
    EpilogueEnd = EPILOGUE_END,
}

impl TransactionEvent {
    /// Value of the top 16 bits of a transaction kernel event ID.
    pub const ID_PREFIX: u32 = 2;
}

impl fmt::Display for TransactionEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl TryFrom<u32> for TransactionEvent {
    type Error = TransactionEventError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value >> 16 != TransactionEvent::ID_PREFIX {
            return Err(TransactionEventError::NotTransactionEvent(value));
        }

        match value {
            ACCOUNT_VAULT_BEFORE_ADD_ASSET => Ok(TransactionEvent::AccountVaultBeforeAddAsset),
            ACCOUNT_VAULT_AFTER_ADD_ASSET => Ok(TransactionEvent::AccountVaultAfterAddAsset),

            ACCOUNT_VAULT_BEFORE_REMOVE_ASSET => {
                Ok(TransactionEvent::AccountVaultBeforeRemoveAsset)
            },
            ACCOUNT_VAULT_AFTER_REMOVE_ASSET => Ok(TransactionEvent::AccountVaultAfterRemoveAsset),

            ACCOUNT_STORAGE_BEFORE_SET_ITEM => Ok(TransactionEvent::AccountStorageBeforeSetItem),
            ACCOUNT_STORAGE_AFTER_SET_ITEM => Ok(TransactionEvent::AccountStorageAfterSetItem),

            ACCOUNT_STORAGE_BEFORE_SET_MAP_ITEM => {
                Ok(TransactionEvent::AccountStorageBeforeSetMapItem)
            },
            ACCOUNT_STORAGE_AFTER_SET_MAP_ITEM => {
                Ok(TransactionEvent::AccountStorageAfterSetMapItem)
            },

            ACCOUNT_BEFORE_INCREMENT_NONCE => Ok(TransactionEvent::AccountBeforeIncrementNonce),
            ACCOUNT_AFTER_INCREMENT_NONCE => Ok(TransactionEvent::AccountAfterIncrementNonce),

            ACCOUNT_PUSH_PROCEDURE_INDEX => Ok(TransactionEvent::AccountPushProcedureIndex),

            NOTE_BEFORE_CREATED => Ok(TransactionEvent::NoteBeforeCreated),
            NOTE_AFTER_CREATED => Ok(TransactionEvent::NoteAfterCreated),

            NOTE_BEFORE_ADD_ASSET => Ok(TransactionEvent::NoteBeforeAddAsset),
            NOTE_AFTER_ADD_ASSET => Ok(TransactionEvent::NoteAfterAddAsset),

            FALCON_SIG_TO_STACK => Ok(TransactionEvent::FalconSigToStack),

            PROLOGUE_START => Ok(TransactionEvent::PrologueStart),
            PROLOGUE_END => Ok(TransactionEvent::PrologueEnd),

            NOTES_PROCESSING_START => Ok(TransactionEvent::NotesProcessingStart),
            NOTES_PROCESSING_END => Ok(TransactionEvent::NotesProcessingEnd),

            NOTE_EXECUTION_START => Ok(TransactionEvent::NoteExecutionStart),
            NOTE_EXECUTION_END => Ok(TransactionEvent::NoteExecutionEnd),

            TX_SCRIPT_PROCESSING_START => Ok(TransactionEvent::TxScriptProcessingStart),
            TX_SCRIPT_PROCESSING_END => Ok(TransactionEvent::TxScriptProcessingEnd),

            EPILOGUE_START => Ok(TransactionEvent::EpilogueStart),
            EPILOGUE_END => Ok(TransactionEvent::EpilogueEnd),

            _ => Err(TransactionEventError::InvalidTransactionEvent(value)),
        }
    }
}
