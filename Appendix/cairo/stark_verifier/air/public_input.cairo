from starkware.cairo.common.alloc import alloc
from starkware.cairo.common.builtin_poseidon.poseidon import poseidon_hash_many
from starkware.cairo.common.cairo_builtins import PoseidonBuiltin
from starkware.cairo.common.hash import HashBuiltin
from starkware.cairo.common.hash_state import hash_finalize, hash_init, hash_update
from starkware.cairo.common.math import assert_le, assert_nn, assert_nn_le, split_felt
from starkware.cairo.common.pow import pow
from starkware.cairo.common.uint256 import Uint256
from starkware.cairo.stark_verifier.air.layout import AirWithLayout
from starkware.cairo.stark_verifier.air.public_memory import (
    AddrValue,
    ContinuousPageHeader,
    get_continuous_pages_product,
    get_page_product,
)
from starkware.cairo.stark_verifier.core.config_instances import StarkConfig
from starkware.cairo.stark_verifier.core.serialize_utils import append_felt, append_felts

struct PublicInput {
    // Base 2 log of the number of steps.
    log_n_steps: felt,
    // Minimum value of range check component.
    range_check_min: felt,
    // Maximum value of range check component.
    range_check_max: felt,
    // Layout ID.
    layout: felt,
    // Dynamic layout params.
    dynamic_params: felt*,
    // Memory segment infos array.
    n_segments: felt,
    segments: SegmentInfo*,

    // Public memory section.
    // Address and value of the padding memory access.
    padding_addr: felt,
    padding_value: felt,

    // Main page.
    main_page_len: felt,
    main_page: AddrValue*,

    // Page header array.
    n_continuous_pages: felt,
    continuous_page_headers: ContinuousPageHeader*,
}

struct SegmentInfo {
    // Start address of the memory segment.
    begin_addr: felt,
    // Stop pointer of the segment - not necessarily the end of the segment.
    stop_ptr: felt,
}

// Computes the hash of the public input, which is used as the initial seed for the Fiat-Shamir
// heuristic.
func public_input_hash{range_check_ptr, pedersen_ptr: HashBuiltin*, poseidon_ptr: PoseidonBuiltin*}(
    air: AirWithLayout*, public_input: PublicInput*, config: StarkConfig*
) -> (res: felt) {
    alloc_locals;

    // Main page hash.
    let (hash_state_ptr) = hash_init();
    let (hash_state_ptr) = hash_update{hash_ptr=pedersen_ptr}(
        hash_state_ptr=hash_state_ptr,
        data_ptr=public_input.main_page,
        data_length=public_input.main_page_len * AddrValue.SIZE,
    );
    let (main_page_hash) = hash_finalize{hash_ptr=pedersen_ptr}(hash_state_ptr=hash_state_ptr);

    let (data: felt*) = alloc();
    local data_start: felt* = data;
    with data {
        append_felt(elem=config.n_verifier_friendly_commitment_layers);
        append_felt(elem=public_input.log_n_steps);
        append_felt(elem=public_input.range_check_min);
        append_felt(elem=public_input.range_check_max);
        append_felt(elem=public_input.layout);

        append_felts(len=air.air.n_dynamic_params, arr=public_input.dynamic_params);

        // n_segments is not written, it is assumed to be fixed.
        append_felts(len=public_input.n_segments * SegmentInfo.SIZE, arr=public_input.segments);
        append_felt(elem=public_input.padding_addr);
        append_felt(elem=public_input.padding_value);
        append_felt(elem=1 + public_input.n_continuous_pages);

        // Main page.
        append_felt(elem=public_input.main_page_len);
        append_felt(elem=main_page_hash);

        // Add the rest of the pages.
        add_continuous_page_headers(
            n_pages=public_input.n_continuous_pages, pages=public_input.continuous_page_headers
        );
    }

    let (res) = poseidon_hash_many(n=data - data_start, elements=data_start);
    return (res=res);
}

func add_continuous_page_headers{range_check_ptr, data: felt*}(
    n_pages: felt, pages: ContinuousPageHeader*
) {
    if (n_pages == 0) {
        return ();
    }

    assert data[0] = pages.start_address;
    assert data[1] = pages.size;
    assert data[2] = pages.hash;
    let data = data + 3;

    return add_continuous_page_headers(n_pages=n_pages - 1, pages=&pages[1]);
}

// Returns the product of all public memory cells.
func get_public_memory_product(public_input: PublicInput*, z: felt, alpha: felt) -> (
    res: felt, total_length: felt
) {
    alloc_locals;
    // Compute total product.
    let (main_page_prod) = get_page_product(
        z=z, alpha=alpha, data_len=public_input.main_page_len, data=public_input.main_page
    );
    let (continuous_pages_prod, continuous_pages_total_length) = get_continuous_pages_product(
        n_pages=public_input.n_continuous_pages, page_headers=public_input.continuous_page_headers
    );
    return (
        res=main_page_prod * continuous_pages_prod,
        total_length=continuous_pages_total_length + public_input.main_page_len,
    );
}

// Returns the ratio between the product of all public memory cells and z^|public_memory|.
// This is the value that needs to be at the memory__multi_column_perm__perm__public_memory_prod
// member expression.
func get_public_memory_product_ratio{range_check_ptr}(
    public_input: PublicInput*, z: felt, alpha: felt, public_memory_column_size: felt
) -> (res: felt) {
    alloc_locals;

    // Compute total product.
    let (pages_product, total_length) = get_public_memory_product(
        public_input=public_input, z=z, alpha=alpha
    );

    // Pad and divide.
    let (numerator) = pow(z, public_memory_column_size);
    tempvar padded_value = z - (public_input.padding_addr + alpha * public_input.padding_value);
    assert_le(total_length, public_memory_column_size);
    let (denominator_pad) = pow(padded_value, public_memory_column_size - total_length);

    return (res=numerator / pages_product / denominator_pad);
}
