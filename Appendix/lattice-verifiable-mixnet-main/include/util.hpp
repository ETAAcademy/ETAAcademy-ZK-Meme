#pragma once

#include <cstddef>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <nfl.hpp>

namespace util {
/// Helper functions for messages conversion

/**
 * Center op1 modulo op2
 * @param rop     result
 * @param op1     number op1 already reduced modulo op2, i.e. such that 0 <= op1
 * < op2-1
 * @param op2     modulus
 * @param op2Div2 floor(modulus/2)
 */
inline void center(mpz_t & rop, mpz_t const &op1, mpz_t const &op2,
		mpz_t const &op2Div2) {
	mpz_set(rop, op1);
	if (mpz_cmp(op1, op2Div2) > 0) {
		mpz_sub(rop, rop, op2);
	}
}
} // namespace util
