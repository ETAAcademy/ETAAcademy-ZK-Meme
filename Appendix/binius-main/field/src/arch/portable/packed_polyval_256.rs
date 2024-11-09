// Copyright 2024 Irreducible Inc.

use super::packed_scaled::packed_scaled_field;
use crate::arch::packed_polyval_128::PackedBinaryPolyval1x128b;

// 256 bit value that just contains two 128-bit integers.
// Is used for portable implementation of 512-bit packed fields.
packed_scaled_field!(PackedBinaryPolyval2x128b = [PackedBinaryPolyval1x128b; 2]);
