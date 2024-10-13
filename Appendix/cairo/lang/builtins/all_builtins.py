"""
A list of all builtins that can be used in various situations where one is required, to reduce
code dependence on builtin names.
"""

from collections import UserList

OUTPUT_BUILTIN = "output"
PEDERSEN_BUILTIN = "pedersen"
RANGE_CHECK_BUILTIN = "range_check"
ECDSA_BUILTIN = "ecdsa"
BITWISE_BUILTIN = "bitwise"
EC_OP_BUILTIN = "ec_op"
KECCAK_BUILTIN = "keccak"
POSEIDON_BUILTIN = "poseidon"
RANGE_CHECK96_BUILTIN = "range_check96"
ADD_MOD_BUILTIN = "add_mod"
MUL_MOD_BUILTIN = "mul_mod"

BUILTIN_NAME_SUFFIX = "_builtin"


def with_suffix(builtin_name: str) -> str:
    """
    Adds `BUILTIN_NAME_SUFFIX` to the builtin name.
    """
    return builtin_name + BUILTIN_NAME_SUFFIX


def remove_builtin_suffix(builtin_name: str) -> str:
    """
    Removes the `BUILTIN_NAME_SUFFIX` suffix from the builtin name.
    """
    assert builtin_name.endswith(BUILTIN_NAME_SUFFIX)
    return builtin_name[: -len(BUILTIN_NAME_SUFFIX)]


class BuiltinList(UserList):
    def except_for(self, *builtins_to_remove):
        """
        Returns a `BuiltinList` of all the builtins except for those specified in
        `builtins_to_remove`.
        """
        return BuiltinList(builtin for builtin in self if builtin not in builtins_to_remove)

    def with_suffix(self):
        """
        Returns a list of the builtins with the suffix as defined by the function `with_suffix`.
        """
        return [with_suffix(builtin) for builtin in self]


ALL_BUILTINS = BuiltinList(
    [
        OUTPUT_BUILTIN,
        PEDERSEN_BUILTIN,
        RANGE_CHECK_BUILTIN,
        ECDSA_BUILTIN,
        BITWISE_BUILTIN,
        EC_OP_BUILTIN,
        KECCAK_BUILTIN,
        POSEIDON_BUILTIN,
        RANGE_CHECK96_BUILTIN,
        ADD_MOD_BUILTIN,
        MUL_MOD_BUILTIN,
    ]
)
# The list of builtins which may have ratio < 1 (and implement BuiltinInstanceDefWithLowRatio).
LOW_RATIO_BUILTINS = BuiltinList(
    [
        ADD_MOD_BUILTIN,
        MUL_MOD_BUILTIN,
        RANGE_CHECK96_BUILTIN,
    ]
)

SUPPORTED_DYNAMIC_BUILTINS = ALL_BUILTINS
