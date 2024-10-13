from starkware.cairo.common.cairo_builtins import ModBuiltin, UInt384
from starkware.cairo.common.math import safe_div, unsigned_div_rem
from starkware.cairo.common.registers import get_label_location

const BATCH_SIZE = 1;

// Returns the smallest felt 0 <= q < rc_bound such that x <= q * y.
func div_ceil{range_check_ptr}(x: felt, y: felt) -> felt {
    let (q, r) = unsigned_div_rem(x, y);
    if (r != 0) {
        return q + 1;
    } else {
        return q;
    }
}

// Fills the first instance of the add_mod and mul_mod builtins and calls the fill_memory hint to
// fill the rest of the instances and the missing values in the values table.
//
// This function uses a hardcoded value of batch_size=8, and asserts the instance definitions use
// the same value.
func run_mod_p_circuit_with_large_batch_size{
    range_check_ptr, add_mod_ptr: ModBuiltin*, mul_mod_ptr: ModBuiltin*
}(
    p: UInt384,
    values_ptr: UInt384*,
    add_mod_offsets_ptr: felt*,
    add_mod_n: felt,
    mul_mod_offsets_ptr: felt*,
    mul_mod_n: felt,
) {
    const BATCH_SIZE = 8;
    alloc_locals;
    local add_mod_n_instances = div_ceil(add_mod_n, BATCH_SIZE);
    local mul_mod_n_instances = div_ceil(mul_mod_n, BATCH_SIZE);
    local range_check_ptr = range_check_ptr;

    if (add_mod_n != 0) {
        assert add_mod_ptr[0] = ModBuiltin(
            p=p,
            values_ptr=values_ptr,
            offsets_ptr=add_mod_offsets_ptr,
            n=add_mod_n_instances * BATCH_SIZE,
        );
    }

    if (mul_mod_n != 0) {
        assert mul_mod_ptr[0] = ModBuiltin(
            p=p,
            values_ptr=values_ptr,
            offsets_ptr=mul_mod_offsets_ptr,
            n=mul_mod_n_instances * BATCH_SIZE,
        );
    }

    %{
        from starkware.cairo.lang.builtins.modulo.mod_builtin_runner import ModBuiltinRunner
        assert builtin_runners["add_mod_builtin"].instance_def.batch_size == ids.BATCH_SIZE
        assert builtin_runners["mul_mod_builtin"].instance_def.batch_size == ids.BATCH_SIZE

        ModBuiltinRunner.fill_memory(
            memory=memory,
            add_mod=(ids.add_mod_ptr.address_, builtin_runners["add_mod_builtin"], ids.add_mod_n),
            mul_mod=(ids.mul_mod_ptr.address_, builtin_runners["mul_mod_builtin"], ids.mul_mod_n),
        )
    %}

    let add_mod_ptr = &add_mod_ptr[add_mod_n_instances];
    let mul_mod_ptr = &mul_mod_ptr[mul_mod_n_instances];
    return ();
}

// Fills the first instance of the add_mod and mul_mod builtins and calls the fill_memory hint to
// fill the rest of the instances and the missing values in the values table.
//
// This function uses a hardcoded value of batch_size=1, and asserts the instance definitions use
// the same value.
func run_mod_p_circuit{add_mod_ptr: ModBuiltin*, mul_mod_ptr: ModBuiltin*}(
    p: UInt384,
    values_ptr: UInt384*,
    add_mod_offsets_ptr: felt*,
    add_mod_n: felt,
    mul_mod_offsets_ptr: felt*,
    mul_mod_n: felt,
) {
    if (add_mod_n != 0) {
        assert add_mod_ptr[0] = ModBuiltin(
            p=p, values_ptr=values_ptr, offsets_ptr=add_mod_offsets_ptr, n=add_mod_n
        );
    }

    if (mul_mod_n != 0) {
        assert mul_mod_ptr[0] = ModBuiltin(
            p=p, values_ptr=values_ptr, offsets_ptr=mul_mod_offsets_ptr, n=mul_mod_n
        );
    }

    %{
        from starkware.cairo.lang.builtins.modulo.mod_builtin_runner import ModBuiltinRunner
        assert builtin_runners["add_mod_builtin"].instance_def.batch_size == 1
        assert builtin_runners["mul_mod_builtin"].instance_def.batch_size == 1

        ModBuiltinRunner.fill_memory(
            memory=memory,
            add_mod=(ids.add_mod_ptr.address_, builtin_runners["add_mod_builtin"], ids.add_mod_n),
            mul_mod=(ids.mul_mod_ptr.address_, builtin_runners["mul_mod_builtin"], ids.mul_mod_n),
        )
    %}

    let add_mod_ptr = &add_mod_ptr[add_mod_n];
    let mul_mod_ptr = &mul_mod_ptr[mul_mod_n];
    return ();
}
