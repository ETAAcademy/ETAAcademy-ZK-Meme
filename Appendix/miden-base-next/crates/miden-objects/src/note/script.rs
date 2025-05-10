use alloc::{sync::Arc, vec::Vec};
use core::fmt::Display;

use super::{Digest, Felt};
use crate::{
    NoteError, PrettyPrint,
    assembly::{
        Assembler, Compile,
        mast::{MastForest, MastNodeId},
    },
    utils::serde::{ByteReader, ByteWriter, Deserializable, DeserializationError, Serializable},
    vm::Program,
};

// NOTE SCRIPT
// ================================================================================================

/// An executable program of a note.
///
/// A note's script represents a program which must be executed for a note to be consumed. As such
/// it defines the rules and side effects of consuming a given note.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoteScript {
    mast: Arc<MastForest>,
    entrypoint: MastNodeId,
}

impl NoteScript {
    // CONSTRUCTORS
    // --------------------------------------------------------------------------------------------

    /// Returns a new [NoteScript] instantiated from the provided program.
    pub fn new(code: Program) -> Self {
        Self {
            entrypoint: code.entrypoint(),
            mast: code.mast_forest().clone(),
        }
    }

    /// Returns a new [NoteScript] compiled from the provided source code using the specified
    /// assembler.
    ///
    /// # Errors
    /// Returns an error if the compilation of the provided source code fails.
    pub fn compile(source_code: impl Compile, assembler: Assembler) -> Result<Self, NoteError> {
        let program = assembler
            .assemble_program(source_code)
            .map_err(NoteError::NoteScriptAssemblyError)?;
        Ok(Self::new(program))
    }

    /// Returns a new [NoteScript] deserialized from the provided bytes.
    ///
    /// # Errors
    /// Returns an error if note script deserialization fails.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NoteError> {
        Self::read_from_bytes(bytes).map_err(NoteError::NoteScriptDeserializationError)
    }

    /// Returns a new [NoteScript] instantiated from the provided components.
    ///
    /// # Panics
    /// Panics if the specified entrypoint is not in the provided MAST forest.
    pub fn from_parts(mast: Arc<MastForest>, entrypoint: MastNodeId) -> Self {
        assert!(mast.get_node_by_id(entrypoint).is_some());
        Self { mast, entrypoint }
    }

    // PUBLIC ACCESSORS
    // --------------------------------------------------------------------------------------------

    /// Returns the commitment of this note script (i.e., the script's MAST root).
    pub fn root(&self) -> Digest {
        self.mast[self.entrypoint].digest()
    }

    /// Returns a reference to the [MastForest] backing this note script.
    pub fn mast(&self) -> Arc<MastForest> {
        self.mast.clone()
    }

    /// Returns an entrypoint node ID of the current script.
    pub fn entrypoint(&self) -> MastNodeId {
        self.entrypoint
    }
}

// CONVERSIONS INTO NOTE SCRIPT
// ================================================================================================

impl From<&NoteScript> for Vec<Felt> {
    fn from(script: &NoteScript) -> Self {
        let mut bytes = script.mast.to_bytes();
        let len = bytes.len();

        // Pad the data so that it can be encoded with u32
        let missing = if len % 4 > 0 { 4 - (len % 4) } else { 0 };
        bytes.resize(bytes.len() + missing, 0);

        let final_size = 2 + bytes.len();
        let mut result = Vec::with_capacity(final_size);

        // Push the length, this is used to remove the padding later
        result.push(Felt::from(script.entrypoint.as_u32()));
        result.push(Felt::new(len as u64));

        // A Felt can not represent all u64 values, so the data is encoded using u32.
        let mut encoded: &[u8] = &bytes;
        while encoded.len() >= 4 {
            let (data, rest) =
                encoded.split_first_chunk::<4>().expect("The length has been checked");
            let number = u32::from_le_bytes(*data);
            result.push(Felt::new(number.into()));

            encoded = rest;
        }

        result
    }
}

impl From<NoteScript> for Vec<Felt> {
    fn from(value: NoteScript) -> Self {
        (&value).into()
    }
}

// CONVERSIONS FROM NOTE SCRIPT
// ================================================================================================

impl TryFrom<&[Felt]> for NoteScript {
    type Error = DeserializationError;

    fn try_from(elements: &[Felt]) -> Result<Self, Self::Error> {
        if elements.len() < 2 {
            return Err(DeserializationError::UnexpectedEOF);
        }

        let entrypoint: u32 = elements[0].try_into().map_err(DeserializationError::InvalidValue)?;
        let len = elements[1].as_int();
        let mut data = Vec::with_capacity(elements.len() * 4);

        for &felt in &elements[2..] {
            let v: u32 = felt.try_into().map_err(DeserializationError::InvalidValue)?;
            data.extend(v.to_le_bytes())
        }
        data.shrink_to(len as usize);

        let mast = MastForest::read_from_bytes(&data)?;
        let entrypoint = MastNodeId::from_u32_safe(entrypoint, &mast)?;
        Ok(NoteScript::from_parts(Arc::new(mast), entrypoint))
    }
}

impl TryFrom<Vec<Felt>> for NoteScript {
    type Error = DeserializationError;

    fn try_from(value: Vec<Felt>) -> Result<Self, Self::Error> {
        value.as_slice().try_into()
    }
}

// SERIALIZATION
// ================================================================================================

impl Serializable for NoteScript {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        self.mast.write_into(target);
        target.write_u32(self.entrypoint.as_u32());
    }
}

impl Deserializable for NoteScript {
    fn read_from<R: ByteReader>(source: &mut R) -> Result<Self, DeserializationError> {
        let mast = MastForest::read_from(source)?;
        let entrypoint = MastNodeId::from_u32_safe(source.read_u32()?, &mast)?;

        Ok(Self::from_parts(Arc::new(mast), entrypoint))
    }
}

// PRETTY-PRINTING
// ================================================================================================

impl PrettyPrint for NoteScript {
    fn render(&self) -> vm_core::prettier::Document {
        use vm_core::prettier::*;
        let entrypoint = self.mast[self.entrypoint].to_pretty_print(&self.mast);

        indent(4, const_text("begin") + nl() + entrypoint.render()) + nl() + const_text("end")
    }
}

impl Display for NoteScript {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.pretty_print(f)
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::{Assembler, Felt, NoteScript, Vec};
    use crate::testing::note::DEFAULT_NOTE_CODE;

    #[test]
    fn test_note_script_to_from_felt() {
        let assembler = Assembler::default();
        let tx_script_src = DEFAULT_NOTE_CODE;
        let note_script = NoteScript::compile(tx_script_src, assembler).unwrap();

        let encoded: Vec<Felt> = (&note_script).into();
        let decoded: NoteScript = encoded.try_into().unwrap();

        assert_eq!(note_script, decoded);
    }
}
