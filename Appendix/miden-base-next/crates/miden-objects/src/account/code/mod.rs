use alloc::{collections::BTreeSet, sync::Arc, vec::Vec};

use vm_core::{mast::MastForest, prettier::PrettyPrint};

use super::{
    AccountError, ByteReader, ByteWriter, Deserializable, DeserializationError, Digest, Felt,
    Hasher, Serializable,
};
use crate::account::{AccountComponent, AccountType};

pub mod procedure;
use procedure::{AccountProcedureInfo, PrintableProcedure};

// ACCOUNT CODE
// ================================================================================================

/// A public interface of an account.
///
/// Account's public interface consists of a set of account procedures, each procedure being a
/// Miden VM program. Thus, MAST root of each procedure commits to the underlying program.
///
/// Each exported procedure is associated with a storage offset and a storage size.
///
/// We commit to the entire account interface by building a sequential hash of all procedure MAST
/// roots and associated storage_offset's. Specifically, each procedure contributes exactly 8 field
/// elements to the sequence of elements to be hashed. These elements are defined as follows:
///
/// ```text
/// [PROCEDURE_MAST_ROOT, storage_offset, 0, 0, storage_size]
/// ```
#[derive(Debug, Clone)]
pub struct AccountCode {
    mast: Arc<MastForest>,
    procedures: Vec<AccountProcedureInfo>,
    commitment: Digest,
}

impl AccountCode {
    /// The maximum number of account interface procedures.
    pub const MAX_NUM_PROCEDURES: usize = 256;

    // CONSTRUCTORS
    // --------------------------------------------------------------------------------------------

    /// Creates a new [`AccountCode`] from the provided components' libraries.
    ///
    /// For testing use only.
    #[cfg(any(feature = "testing", test))]
    pub fn from_components(
        components: &[AccountComponent],
        account_type: AccountType,
    ) -> Result<Self, AccountError> {
        super::validate_components_support_account_type(components, account_type)?;
        Self::from_components_unchecked(components, account_type)
    }

    /// Creates a new [`AccountCode`] from the provided components' libraries.
    ///
    /// # Warning
    ///
    /// This does not check whether the provided components are valid when combined.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of procedures in all merged libraries is 0 or exceeds
    ///   [`AccountCode::MAX_NUM_PROCEDURES`].
    /// - Two or more libraries export a procedure with the same MAST root.
    /// - The number of [`StorageSlot`](crate::account::StorageSlot)s of a component or of all
    ///   components exceeds 255.
    /// - [`MastForest::merge`] fails on all libraries.
    pub(super) fn from_components_unchecked(
        components: &[AccountComponent],
        account_type: AccountType,
    ) -> Result<Self, AccountError> {
        let (merged_mast_forest, _) =
            MastForest::merge(components.iter().map(|component| component.mast_forest()))
                .map_err(AccountError::AccountComponentMastForestMergeError)?;

        let mut procedures = Vec::new();
        let mut proc_root_set = BTreeSet::new();

        // Slot 0 is globally reserved for faucet accounts so the accessible slots begin at 1 if
        // there is a faucet component present.
        let mut component_storage_offset = if account_type.is_faucet() { 1 } else { 0 };

        for component in components {
            let component_storage_size = component.storage_size();

            for module in component.library().module_infos() {
                for proc_mast_root in module.procedure_digests() {
                    // We cannot support procedures from multiple components with the same MAST root
                    // since storage offsets/sizes are set per MAST root. Setting them again for
                    // procedures where the offset has already been inserted would cause that
                    // procedure of the earlier component to write to the wrong slot.
                    if !proc_root_set.insert(proc_mast_root) {
                        return Err(AccountError::AccountComponentDuplicateProcedureRoot(
                            proc_mast_root,
                        ));
                    }

                    // Components that do not access storage need to have offset and size set to 0.
                    let (storage_offset, storage_size) = if component_storage_size == 0 {
                        (0, 0)
                    } else {
                        (component_storage_offset, component_storage_size)
                    };

                    // Note: Offset and size are validated in `AccountProcedureInfo::new`.
                    procedures.push(AccountProcedureInfo::new(
                        proc_mast_root,
                        storage_offset,
                        storage_size,
                    )?);
                }
            }

            component_storage_offset = component_storage_offset.checked_add(component_storage_size)
              .expect("account procedure info constructor should return an error if the addition overflows");
        }

        // make sure the number of procedures is between 1 and 256 (both inclusive)
        if procedures.is_empty() {
            return Err(AccountError::AccountCodeNoProcedures);
        } else if procedures.len() > Self::MAX_NUM_PROCEDURES {
            return Err(AccountError::AccountCodeTooManyProcedures(procedures.len()));
        }

        Ok(Self {
            commitment: build_procedure_commitment(&procedures),
            procedures,
            mast: Arc::new(merged_mast_forest),
        })
    }

    /// Returns a new [AccountCode] deserialized from the provided bytes.
    ///
    /// # Errors
    /// Returns an error if account code deserialization fails.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AccountError> {
        Self::read_from_bytes(bytes).map_err(AccountError::AccountCodeDeserializationError)
    }

    /// Returns a new definition of an account's interface instantiated from the provided
    /// [MastForest] and a list of [AccountProcedureInfo]s.
    ///
    /// # Panics
    /// Panics if:
    /// - The number of procedures is smaller than 1 or greater than 256.
    /// - If some any of the provided procedures does not have a corresponding root in the provided
    ///   MAST forest.
    pub fn from_parts(mast: Arc<MastForest>, procedures: Vec<AccountProcedureInfo>) -> Self {
        assert!(!procedures.is_empty(), "no account procedures");
        assert!(procedures.len() <= Self::MAX_NUM_PROCEDURES, "too many account procedures");

        Self {
            commitment: build_procedure_commitment(&procedures),
            procedures,
            mast,
        }
    }

    // PUBLIC ACCESSORS
    // --------------------------------------------------------------------------------------------

    /// Returns a commitment to an account's public interface.
    pub fn commitment(&self) -> Digest {
        self.commitment
    }

    /// Returns a reference to the [MastForest] backing this account code.
    pub fn mast(&self) -> Arc<MastForest> {
        self.mast.clone()
    }

    /// Returns a reference to the account procedures.
    pub fn procedures(&self) -> &[AccountProcedureInfo] {
        &self.procedures
    }

    /// Returns an iterator over the procedure MAST roots of this account code.
    pub fn procedure_roots(&self) -> impl Iterator<Item = Digest> + '_ {
        self.procedures().iter().map(|procedure| *procedure.mast_root())
    }

    /// Returns the number of public interface procedures defined in this account code.
    pub fn num_procedures(&self) -> usize {
        self.procedures.len()
    }

    /// Returns true if a procedure with the specified MAST root is defined in this account code.
    pub fn has_procedure(&self, mast_root: Digest) -> bool {
        self.procedures.iter().any(|procedure| procedure.mast_root() == &mast_root)
    }

    /// Returns information about the procedure at the specified index.
    ///
    /// # Panics
    /// Panics if the provided index is out of bounds.
    pub fn get_procedure_by_index(&self, index: usize) -> &AccountProcedureInfo {
        &self.procedures[index]
    }

    /// Returns the procedure index for the procedure with the specified MAST root or None if such
    /// procedure is not defined in this [AccountCode].
    pub fn get_procedure_index_by_root(&self, root: Digest) -> Option<usize> {
        self.procedures
            .iter()
            .map(|procedure| procedure.mast_root())
            .position(|r| r == &root)
    }

    /// Converts procedure information in this [AccountCode] into a vector of field elements.
    ///
    /// This is done by first converting each procedure into 8 field elements as follows:
    /// ```text
    /// [PROCEDURE_MAST_ROOT, storage_offset, storage_size, 0, 0]
    /// ```
    /// And then concatenating the resulting elements into a single vector.
    pub fn as_elements(&self) -> Vec<Felt> {
        procedures_as_elements(self.procedures())
    }

    /// Returns an iterator of printable representations for all procedures in this account code.
    ///
    /// # Returns
    /// An iterator yielding [`PrintableProcedure`] instances for all procedures in this account
    /// code.
    pub fn printable_procedures(&self) -> impl Iterator<Item = PrintableProcedure> {
        self.procedures()
            .iter()
            .filter_map(move |procedure_info| self.printable_procedure(procedure_info).ok())
    }

    // HELPER FUNCTIONS
    // --------------------------------------------------------------------------------------------

    /// Returns a printable representation of the procedure with the specified MAST root.
    ///
    /// # Errors
    /// Returns an error if no procedure with the specified root exists in this account code.
    fn printable_procedure(
        &self,
        proc_info: &AccountProcedureInfo,
    ) -> Result<PrintableProcedure, AccountError> {
        let node_id = self
            .mast
            .find_procedure_root(*proc_info.mast_root())
            .expect("procedure root should be present in the mast forest");

        Ok(PrintableProcedure::new(self.mast.clone(), *proc_info, node_id))
    }
}

// EQUALITY
// ================================================================================================

impl PartialEq for AccountCode {
    fn eq(&self, other: &Self) -> bool {
        // TODO: consider checking equality based only on the set of procedures
        self.mast == other.mast && self.procedures == other.procedures
    }
}

impl Ord for AccountCode {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.commitment.cmp(&other.commitment)
    }
}

impl PartialOrd for AccountCode {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for AccountCode {}

// SERIALIZATION
// ================================================================================================

impl Serializable for AccountCode {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        self.mast.write_into(target);
        // since the number of procedures is guaranteed to be between 1 and 256, we can store the
        // number as a single byte - but we do have to subtract 1 to store 256 as 255.
        target.write_u8((self.procedures.len() - 1) as u8);
        target.write_many(self.procedures());
    }

    fn get_size_hint(&self) -> usize {
        // TODO: Replace with proper calculation.
        let mut mast_forest_target = Vec::new();
        self.mast.write_into(&mut mast_forest_target);

        // Size of the serialized procedures length.
        let u8_size = 0u8.get_size_hint();
        let mut size = u8_size + mast_forest_target.len();

        for procedure in self.procedures() {
            size += procedure.get_size_hint();
        }

        size
    }
}

impl Deserializable for AccountCode {
    fn read_from<R: ByteReader>(source: &mut R) -> Result<Self, DeserializationError> {
        let module = Arc::new(MastForest::read_from(source)?);
        let num_procedures = (source.read_u8()? as usize) + 1;
        let procedures = source.read_many::<AccountProcedureInfo>(num_procedures)?;

        Ok(Self::from_parts(module, procedures))
    }
}

// PRETTY PRINT
// ================================================================================================

impl PrettyPrint for AccountCode {
    fn render(&self) -> vm_core::prettier::Document {
        use vm_core::prettier::*;
        let mut partial = Document::Empty;
        let len_procedures = self.num_procedures();

        for (index, printable_procedure) in self.printable_procedures().enumerate() {
            partial += indent(
                0,
                indent(
                    4,
                    text(format!("proc.{}", printable_procedure.mast_root()))
                        + nl()
                        + text(format!(
                            "storage.{}.{}",
                            printable_procedure.storage_offset(),
                            printable_procedure.storage_size()
                        ))
                        + nl()
                        + printable_procedure.render(),
                ) + nl()
                    + const_text("end"),
            );
            if index < len_procedures - 1 {
                partial += nl();
            }
        }
        partial
    }
}

// HELPER FUNCTIONS
// ================================================================================================

/// Converts given procedures into field elements
fn procedures_as_elements(procedures: &[AccountProcedureInfo]) -> Vec<Felt> {
    procedures.iter().flat_map(|procedure| <[Felt; 8]>::from(*procedure)).collect()
}

/// Computes the commitment to the given procedures
fn build_procedure_commitment(procedures: &[AccountProcedureInfo]) -> Digest {
    let elements = procedures_as_elements(procedures);
    Hasher::hash_elements(&elements)
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {

    use assembly::Assembler;
    use assert_matches::assert_matches;
    use vm_core::Word;

    use super::{AccountCode, Deserializable, Serializable};
    use crate::{
        AccountError,
        account::{AccountComponent, AccountType, StorageSlot, code::build_procedure_commitment},
    };

    #[test]
    fn test_serde_account_code() {
        let code = AccountCode::mock();
        let serialized = code.to_bytes();
        let deserialized = AccountCode::read_from_bytes(&serialized).unwrap();
        assert_eq!(deserialized, code)
    }

    #[test]
    fn test_account_code_procedure_root() {
        let code = AccountCode::mock();
        let procedure_root = build_procedure_commitment(code.procedures());
        assert_eq!(procedure_root, code.commitment())
    }

    #[test]
    fn test_account_code_procedure_offset_out_of_bounds() {
        let code1 = "export.foo add end";
        let library1 = Assembler::default().assemble_library([code1]).unwrap();
        let code2 = "export.bar sub end";
        let library2 = Assembler::default().assemble_library([code2]).unwrap();

        let component1 =
            AccountComponent::new(library1, vec![StorageSlot::Value(Word::default()); 250])
                .unwrap()
                .with_supports_all_types();
        let mut component2 =
            AccountComponent::new(library2, vec![StorageSlot::Value(Word::default()); 5])
                .unwrap()
                .with_supports_all_types();

        // This is fine as the offset+size for component 2 is <= 255.
        AccountCode::from_components(
            &[component1.clone(), component2.clone()],
            AccountType::RegularAccountUpdatableCode,
        )
        .unwrap();

        // Push one more slot so offset+size exceeds 255.
        component2.storage_slots.push(StorageSlot::Value(Word::default()));

        let err = AccountCode::from_components(
            &[component1, component2],
            AccountType::RegularAccountUpdatableCode,
        )
        .unwrap_err();

        assert_matches!(err, AccountError::StorageOffsetPlusSizeOutOfBounds(256))
    }
}
