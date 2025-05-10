use alloc::vec::Vec;

use crate::{
    Digest, Felt, Hasher, MAX_INPUTS_PER_NOTE, WORD_SIZE, ZERO,
    errors::NoteError,
    utils::serde::{ByteReader, ByteWriter, Deserializable, DeserializationError, Serializable},
};

// NOTE INPUTS
// ================================================================================================

/// A container for note inputs.
///
/// A note can be associated with up to 128 input values. Each value is represented by a single
/// field element. Thus, note input values can contain up to ~1 KB of data.
///
/// All inputs associated with a note can be reduced to a single commitment which is computed by
/// first padding the inputs with ZEROs to the next multiple of 8, and then by computing a
/// sequential hash of the resulting elements.
#[derive(Clone, Debug)]
pub struct NoteInputs {
    values: Vec<Felt>,
    commitment: Digest,
}

impl NoteInputs {
    // CONSTRUCTOR
    // --------------------------------------------------------------------------------------------

    /// Returns [NoteInputs] instantiated from the provided values.
    ///
    /// # Errors
    /// Returns an error if the number of provided inputs is greater than 128.
    pub fn new(values: Vec<Felt>) -> Result<Self, NoteError> {
        if values.len() > MAX_INPUTS_PER_NOTE {
            return Err(NoteError::TooManyInputs(values.len()));
        }

        Ok(pad_and_build(values))
    }

    // PUBLIC ACCESSORS
    // --------------------------------------------------------------------------------------------

    /// Returns a commitment to these inputs.
    pub fn commitment(&self) -> Digest {
        self.commitment
    }

    /// Returns the number of input values.
    ///
    /// The returned value is guaranteed to be smaller than or equal to 128.
    pub fn num_values(&self) -> u8 {
        const _: () = assert!(MAX_INPUTS_PER_NOTE <= u8::MAX as usize);
        debug_assert!(
            self.values.len() < MAX_INPUTS_PER_NOTE,
            "The constructor should have checked the number of inputs"
        );
        self.values.len() as u8
    }

    /// Returns a reference to the input values.
    pub fn values(&self) -> &[Felt] {
        &self.values
    }

    /// Returns the note's input formatted to be used with the advice map.
    ///
    /// The format is `input_len || INPUTS || PADDING`, where:
    ///
    /// - input_len is the number of inputs
    /// - INPUTS is the variable inputs for the note
    /// - PADDING is the optional padding to align the data with a 2WORD boundary
    pub fn format_for_advice(&self) -> Vec<Felt> {
        // NOTE: keep map in sync with the `note::get_inputs` API procedure
        let mut padded = pad_inputs(&self.values);
        padded.insert(0, self.num_values().into());
        padded
    }
}

impl Default for NoteInputs {
    fn default() -> Self {
        pad_and_build(vec![])
    }
}

impl PartialEq for NoteInputs {
    fn eq(&self, other: &Self) -> bool {
        let NoteInputs { values: inputs, commitment: _ } = self;
        inputs == &other.values
    }
}

impl Eq for NoteInputs {}

// CONVERSION
// ================================================================================================

impl From<NoteInputs> for Vec<Felt> {
    fn from(value: NoteInputs) -> Self {
        value.values
    }
}

impl TryFrom<Vec<Felt>> for NoteInputs {
    type Error = NoteError;

    fn try_from(value: Vec<Felt>) -> Result<Self, Self::Error> {
        NoteInputs::new(value)
    }
}

// HELPER FUNCTIONS
// ================================================================================================

/// Returns a vector with built from the provided inputs and padded to the next multiple of 8.
fn pad_inputs(inputs: &[Felt]) -> Vec<Felt> {
    const BLOCK_SIZE: usize = WORD_SIZE * 2;

    let padded_len = inputs.len().next_multiple_of(BLOCK_SIZE);
    let mut padded_inputs = Vec::with_capacity(padded_len);
    padded_inputs.extend(inputs.iter());
    padded_inputs.resize(padded_len, ZERO);

    padded_inputs
}

/// Pad `values` and returns a new `NoteInputs`.
fn pad_and_build(values: Vec<Felt>) -> NoteInputs {
    let commitment = {
        let padded_values = pad_inputs(&values);
        Hasher::hash_elements(&padded_values)
    };

    NoteInputs { values, commitment }
}

// SERIALIZATION
// ================================================================================================

impl Serializable for NoteInputs {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        let NoteInputs { values, commitment: _commitment } = self;
        target.write_u8(values.len().try_into().expect("inputs len is not a u8 value"));
        target.write_many(values);
    }
}

impl Deserializable for NoteInputs {
    fn read_from<R: ByteReader>(source: &mut R) -> Result<Self, DeserializationError> {
        let num_values = source.read_u8()? as usize;
        let values = source.read_many::<Felt>(num_values)?;
        Self::new(values).map_err(|v| DeserializationError::InvalidValue(format!("{v}")))
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use miden_crypto::utils::Deserializable;

    use super::{Felt, NoteInputs, Serializable};

    #[test]
    fn test_input_ordering() {
        // inputs are provided in reverse stack order
        let inputs = vec![Felt::new(1), Felt::new(2), Felt::new(3)];
        // we expect the inputs to be padded to length 16 and to remain in reverse stack order.
        let expected_ordering = vec![Felt::new(1), Felt::new(2), Felt::new(3)];

        let note_inputs = NoteInputs::new(inputs).expect("note created should succeed");
        assert_eq!(&expected_ordering, &note_inputs.values);
    }

    #[test]
    fn test_input_serialization() {
        let inputs = vec![Felt::new(1), Felt::new(2), Felt::new(3)];
        let note_inputs = NoteInputs::new(inputs).unwrap();

        let bytes = note_inputs.to_bytes();
        let parsed_note_inputs = NoteInputs::read_from_bytes(&bytes).unwrap();
        assert_eq!(note_inputs, parsed_note_inputs);
    }
}
