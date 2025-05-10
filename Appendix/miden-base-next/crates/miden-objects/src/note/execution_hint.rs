// NOTE EXECUTION HINT
// ================================================================================================

use crate::{Felt, NoteError, block::BlockNumber};

/// Specifies the conditions under which a note is ready to be consumed.
/// These conditions are meant to be encoded in the note script as well.
///
/// This struct can be represented as the combination of a tag, and a payload.
/// The tag specifies the variant of the hint, and the payload encodes the hint data.
///
/// # Felt layout
///
/// [`NoteExecutionHint`] can be encoded into a [`Felt`] with the following layout:
///
/// ```text
/// [26 zero bits | payload (32 bits) | tag (6 bits)]
/// ```
///
/// This way, hints such as [NoteExecutionHint::Always], are represented by `Felt::new(1)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoteExecutionHint {
    /// Unspecified note execution hint. Implies it is not known under which conditions the note
    /// is consumable.
    None,
    /// The note's script can be executed at any time.
    Always,
    /// The note's script can be executed after the specified block number.
    ///
    /// The block number cannot be [`u32::MAX`] which is enforced through the [`AfterBlockNumber`]
    /// type.
    AfterBlock { block_num: AfterBlockNumber },
    /// The note's script can be executed in the specified slot within the specified round.
    ///
    /// The slot is defined as follows:
    /// - First we define the length of the round in powers of 2. For example, round_len = 10 is a
    ///   round of 1024 blocks.
    /// - Then we define the length of a slot within the round also using powers of 2. For example,
    ///   slot_len = 7 is a slot of 128 blocks.
    /// - Lastly, the offset specifies the index of the slot within the round - i.e., 0 is the first
    ///   slot, 1 is the second slot etc.
    ///
    /// For example: { round_len: 10, slot_len: 7, slot_offset: 1 } means that the note can
    /// be executed in any second 128 block slot of a 1024 block round. These would be blocks
    /// 128..255, 1152..1279, 2176..2303 etc.
    OnBlockSlot {
        round_len: u8,
        slot_len: u8,
        slot_offset: u8,
    },
}

impl NoteExecutionHint {
    // CONSTANTS
    // ------------------------------------------------------------------------------------------------

    pub(crate) const NONE_TAG: u8 = 0;
    pub(crate) const ALWAYS_TAG: u8 = 1;
    pub(crate) const AFTER_BLOCK_TAG: u8 = 2;
    pub(crate) const ON_BLOCK_SLOT_TAG: u8 = 3;

    // CONSTRUCTORS
    // ------------------------------------------------------------------------------------------------

    /// Creates a [NoteExecutionHint::None] variant
    pub fn none() -> Self {
        NoteExecutionHint::None
    }

    /// Creates a [NoteExecutionHint::Always] variant
    pub fn always() -> Self {
        NoteExecutionHint::Always
    }

    /// Creates a [NoteExecutionHint::AfterBlock] variant based on the given `block_num`
    ///
    /// # Errors
    ///
    /// Returns an error if `block_num` is equal to [`u32::MAX`].
    pub fn after_block(block_num: BlockNumber) -> Result<Self, NoteError> {
        AfterBlockNumber::new(block_num)
            .map(|block_number| NoteExecutionHint::AfterBlock { block_num: block_number })
    }

    /// Creates a [NoteExecutionHint::OnBlockSlot] for the given parameters. See the variants
    /// documentation for details on the parameters.
    pub fn on_block_slot(round_len: u8, slot_len: u8, slot_offset: u8) -> Self {
        NoteExecutionHint::OnBlockSlot { round_len, slot_len, slot_offset }
    }

    pub fn from_parts(tag: u8, payload: u32) -> Result<NoteExecutionHint, NoteError> {
        match tag {
            Self::NONE_TAG => {
                if payload != 0 {
                    return Err(NoteError::InvalidNoteExecutionHintPayload(tag, payload));
                }
                Ok(NoteExecutionHint::None)
            },
            Self::ALWAYS_TAG => {
                if payload != 0 {
                    return Err(NoteError::InvalidNoteExecutionHintPayload(tag, payload));
                }
                Ok(NoteExecutionHint::Always)
            },
            Self::AFTER_BLOCK_TAG => NoteExecutionHint::after_block(payload.into()),
            Self::ON_BLOCK_SLOT_TAG => {
                let remainder = ((payload >> 24) & 0xff) as u8;
                if remainder != 0 {
                    return Err(NoteError::InvalidNoteExecutionHintPayload(tag, payload));
                }

                let round_len = ((payload >> 16) & 0xff) as u8;
                let slot_len = ((payload >> 8) & 0xff) as u8;
                let slot_offset = (payload & 0xff) as u8;
                let hint = NoteExecutionHint::OnBlockSlot { round_len, slot_len, slot_offset };

                Ok(hint)
            },
            _ => Err(NoteError::NoteExecutionHintTagOutOfRange(tag)),
        }
    }

    /// Returns whether the note execution conditions validate for the given `block_num`
    ///
    /// # Returns
    /// - `None` if we don't know whether the note can be consumed.
    /// - `Some(true)` if the note is consumable for the given `block_num`
    /// - `Some(false)` if the note is not consumable for the given `block_num`
    pub fn can_be_consumed(&self, block_num: BlockNumber) -> Option<bool> {
        let block_num = block_num.as_u32();
        match self {
            NoteExecutionHint::None => None,
            NoteExecutionHint::Always => Some(true),
            NoteExecutionHint::AfterBlock { block_num: hint_block_num } => {
                Some(block_num >= hint_block_num.as_u32())
            },
            NoteExecutionHint::OnBlockSlot { round_len, slot_len, slot_offset } => {
                let round_len_blocks: u32 = 1 << round_len;
                let slot_len_blocks: u32 = 1 << slot_len;

                let block_round_index = block_num / round_len_blocks;

                let slot_start_block =
                    block_round_index * round_len_blocks + (*slot_offset as u32) * slot_len_blocks;
                let slot_end_block = slot_start_block + slot_len_blocks;

                let can_be_consumed = block_num >= slot_start_block && block_num < slot_end_block;
                Some(can_be_consumed)
            },
        }
    }

    /// Encodes the [`NoteExecutionHint`] into a 6-bit tag and a 32-bit payload.
    ///
    /// # Guarantees
    ///
    /// Since the tag has at most 6 bits, the returned byte is guaranteed to have its two most
    /// significant bits set to `0`.
    ///
    /// The payload is guaranteed to contain at least one `0` bit to make encoding it into
    /// [`NoteMetadata`](crate::note::NoteMetadata) safely possible.
    pub fn into_parts(&self) -> (u8, u32) {
        match self {
            NoteExecutionHint::None => (Self::NONE_TAG, 0),
            NoteExecutionHint::Always => (Self::ALWAYS_TAG, 0),
            NoteExecutionHint::AfterBlock { block_num } => {
                (Self::AFTER_BLOCK_TAG, block_num.as_u32())
            },
            NoteExecutionHint::OnBlockSlot { round_len, slot_len, slot_offset } => {
                let payload: u32 =
                    ((*round_len as u32) << 16) | ((*slot_len as u32) << 8) | (*slot_offset as u32);
                (Self::ON_BLOCK_SLOT_TAG, payload)
            },
        }
    }
}

/// Converts a [`NoteExecutionHint`] into a [`Felt`] with the layout documented on the type.
impl From<NoteExecutionHint> for Felt {
    fn from(value: NoteExecutionHint) -> Self {
        let int_representation: u64 = value.into();
        Felt::new(int_representation)
    }
}

/// Tries to convert a `u64` into a [`NoteExecutionHint`] with the expected layout documented on the
/// type.
///
/// Note: The upper 26 bits are not enforced to be zero.
impl TryFrom<u64> for NoteExecutionHint {
    type Error = NoteError;
    fn try_from(value: u64) -> Result<Self, Self::Error> {
        let tag = (value & 0b111111) as u8;
        let payload = ((value >> 6) & 0xffffffff) as u32;

        Self::from_parts(tag, payload)
    }
}

/// Converts a [`NoteExecutionHint`] into a `u64` with the layout documented on the type.
impl From<NoteExecutionHint> for u64 {
    fn from(value: NoteExecutionHint) -> Self {
        let (tag, payload) = value.into_parts();
        ((payload as u64) << 6) | (tag as u64)
    }
}

// AFTER BLOCK NUMBER
// ================================================================================================

/// A wrapper around a block number which enforces that it is not `u32::MAX`.
///
/// Used for the [`NoteExecutionHint::AfterBlock`] variant where this constraint is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AfterBlockNumber(BlockNumber);

impl AfterBlockNumber {
    /// Creates a new [`AfterBlockNumber`] from the given `block_number`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `block_number` is equal to `u32::MAX`.
    pub fn new(block_number: BlockNumber) -> Result<Self, NoteError> {
        if block_number.as_u32() == u32::MAX {
            Err(NoteError::NoteExecutionHintAfterBlockCannotBeU32Max)
        } else {
            Ok(Self(block_number))
        }
    }

    /// Returns the block number as a `u32`.
    pub fn as_u32(&self) -> u32 {
        self.0.as_u32()
    }
}

impl From<AfterBlockNumber> for u32 {
    fn from(block_number: AfterBlockNumber) -> Self {
        block_number.0.as_u32()
    }
}

impl TryFrom<u32> for AfterBlockNumber {
    type Error = NoteError;

    fn try_from(block_number: u32) -> Result<Self, Self::Error> {
        Self::new(block_number.into())
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use super::*;

    fn assert_hint_serde(note_execution_hint: NoteExecutionHint) {
        let (tag, payload) = note_execution_hint.into_parts();
        let deserialized = NoteExecutionHint::from_parts(tag, payload).unwrap();
        assert_eq!(deserialized, note_execution_hint);
    }

    #[test]
    fn test_serialization_round_trip() {
        assert_hint_serde(NoteExecutionHint::None);
        assert_hint_serde(NoteExecutionHint::Always);
        assert_hint_serde(NoteExecutionHint::after_block(15.into()).unwrap());
        assert_hint_serde(NoteExecutionHint::OnBlockSlot {
            round_len: 9,
            slot_len: 12,
            slot_offset: 18,
        });
    }

    #[test]
    fn test_encode_round_trip() {
        let hint = NoteExecutionHint::after_block(15.into()).unwrap();
        let hint_int: u64 = hint.into();
        let decoded_hint: NoteExecutionHint = hint_int.try_into().unwrap();
        assert_eq!(hint, decoded_hint);

        let hint = NoteExecutionHint::OnBlockSlot {
            round_len: 22,
            slot_len: 33,
            slot_offset: 44,
        };
        let hint_int: u64 = hint.into();
        let decoded_hint: NoteExecutionHint = hint_int.try_into().unwrap();
        assert_eq!(hint, decoded_hint);

        let always_int: u64 = NoteExecutionHint::always().into();
        assert_eq!(always_int, 1u64);
    }

    #[test]
    fn test_can_be_consumed() {
        let none = NoteExecutionHint::none();
        assert!(none.can_be_consumed(100.into()).is_none());

        let always = NoteExecutionHint::always();
        assert!(always.can_be_consumed(100.into()).unwrap());

        let after_block = NoteExecutionHint::after_block(12345.into()).unwrap();
        assert!(!after_block.can_be_consumed(12344.into()).unwrap());
        assert!(after_block.can_be_consumed(12345.into()).unwrap());

        let on_block_slot = NoteExecutionHint::on_block_slot(10, 7, 1);
        assert!(!on_block_slot.can_be_consumed(127.into()).unwrap()); // Block 127 is not in the slot 128..255
        assert!(on_block_slot.can_be_consumed(128.into()).unwrap()); // Block 128 is in the slot 128..255
        assert!(on_block_slot.can_be_consumed(255.into()).unwrap()); // Block 255 is in the slot 128..255
        assert!(!on_block_slot.can_be_consumed(256.into()).unwrap()); // Block 256 is not in the slot 128..255
        assert!(on_block_slot.can_be_consumed(1152.into()).unwrap()); // Block 1152 is in the slot 1152..1279
        assert!(on_block_slot.can_be_consumed(1279.into()).unwrap()); // Block 1279 is in the slot 1152..1279
        assert!(on_block_slot.can_be_consumed(2176.into()).unwrap()); // Block 2176 is in the slot 2176..2303
        assert!(!on_block_slot.can_be_consumed(2175.into()).unwrap()); // Block 1279 is in the slot
        // 2176..2303
    }

    #[test]
    fn test_parts_validity() {
        NoteExecutionHint::from_parts(NoteExecutionHint::NONE_TAG, 1).unwrap_err();
        NoteExecutionHint::from_parts(NoteExecutionHint::ALWAYS_TAG, 12).unwrap_err();
        // 4th byte should be blank for tag 3 (OnBlockSlot)
        NoteExecutionHint::from_parts(NoteExecutionHint::ON_BLOCK_SLOT_TAG, 1 << 24).unwrap_err();
        NoteExecutionHint::from_parts(NoteExecutionHint::ON_BLOCK_SLOT_TAG, 0).unwrap();

        NoteExecutionHint::from_parts(10, 1).unwrap_err();
    }

    #[test]
    fn test_after_block_fails_on_u32_max() {
        assert_matches!(
            NoteExecutionHint::after_block(u32::MAX.into()).unwrap_err(),
            NoteError::NoteExecutionHintAfterBlockCannotBeU32Max
        );
    }
}
