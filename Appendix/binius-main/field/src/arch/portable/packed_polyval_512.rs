// Copyright 2024 Irreducible Inc.

use super::packed_scaled::packed_scaled_field;
use crate::arch::packed_polyval_256::PackedBinaryPolyval2x128b;

// 512 bit value that just contains four 128-bit integers.
// Is used for portable implementation of 512-bit packed fields.
packed_scaled_field!(PackedBinaryPolyval4x128b = [PackedBinaryPolyval2x128b; 2]);
