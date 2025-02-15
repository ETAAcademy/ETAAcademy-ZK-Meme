#include "blake3.h"
#include "test.h"
#include "bench.h"
#include "common.h"
#include "sample_z_small.h"
#include "sample_z_large.h"

#define R       (HEIGHT+1)
#define V       (HEIGHT+3)
#define TAU     1000
#define NTI     130

static void pibnd_hash(uint8_t h[BLAKE3_OUT_LEN], params::poly_q A[R][V],
		params::poly_q t[TAU][V], params::poly_q W[R][NTI]) {
	blake3_hasher hasher;

	blake3_hasher_init(&hasher);
	/* Hash public key. */
	for (size_t i = 0; i < R; i++) {
		for (int j = 0; j < V; j++) {
			blake3_hasher_update(&hasher, (const uint8_t *)A[i][j].data(),
					16 * DEGREE);
		}
	}
	for (size_t i = 0; i < TAU; i++) {
		for (int j = 0; j < V; j++) {
			blake3_hasher_update(&hasher, (const uint8_t *)t[i][j].data(),
					16 * DEGREE);
		}
	}
	for (size_t i = 0; i < R; i++) {
		for (int j = 0; j < NTI; j++) {
			blake3_hasher_update(&hasher, (const uint8_t *)W[i][j].data(),
					16 * DEGREE);
		}
	}

	blake3_hasher_finalize(&hasher, h, BLAKE3_OUT_LEN);
}

static int pibnd_rej_sampling(params::poly_q Z[V][NTI],
		params::poly_q SC[V][NTI], uint64_t s2) {
	array < mpz_t, params::poly_q::degree > coeffs0, coeffs1;
	params::poly_q t;
	mpz_t dot, norm, qDivBy2, tmp;
	double r, M = 3.0;
	int64_t seed;
	mpf_t u;
	uint8_t buf[8];
	gmp_randstate_t state;
	int result;

	/// Constructors
	mpf_init(u);
	gmp_randinit_mt(state);
	mpz_inits(dot, norm, qDivBy2, tmp, nullptr);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs0[i], (params::poly_q::bits_in_moduli_product() << 2));
		mpz_init2(coeffs1[i], (params::poly_q::bits_in_moduli_product() << 2));
	}

	mpz_fdiv_q_2exp(qDivBy2, params::poly_q::moduli_product(), 1);
	mpz_set_ui(norm, 0);
	mpz_set_ui(dot, 0);
	for (int i = 0; i < V - 1; i++) {
		for (int j = 0; j < NTI; j++) {
			t = Z[i][j];
			t.invntt_pow_invphi();
			t.poly2mpz(coeffs0);
			t = SC[i][j];
			t.invntt_pow_invphi();
			t.poly2mpz(coeffs1);
			for (size_t i = 0; i < params::poly_q::degree; i++) {
				util::center(coeffs0[i], coeffs0[i],
						params::poly_q::moduli_product(), qDivBy2);
				util::center(coeffs1[i], coeffs1[i],
						params::poly_q::moduli_product(), qDivBy2);
				mpz_mul(tmp, coeffs0[i], coeffs1[i]);
				mpz_add(dot, dot, tmp);
				mpz_mul(tmp, coeffs1[i], coeffs1[i]);
				mpz_add(norm, norm, tmp);
			}
		}
	}

	getrandom(buf, sizeof(buf), 0);
	memcpy(&seed, buf, sizeof(buf));
	gmp_randseed_ui(state, seed);
	mpf_urandomb(u, state, mpf_get_default_prec());

	result = mpz_get_d(dot) < 0;
	r = -2.0 * mpz_get_d(dot) + mpz_get_d(norm);
	r = r / (2.0 * s2);
	r = exp(r) / M;
	result |= mpf_get_d(u) > (1 / M);

	mpf_clear(u);
	mpz_clears(dot, norm, qDivBy2, tmp, nullptr);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs0[i]);
		mpz_clear(coeffs1[i]);
	}
	return result;
}

/**
 * Test if the l2-norm is within bounds (4 * sigma * sqrt(N)).
 *
 * @param[in] r 			- the polynomial to compute the l2-norm.
 * @return the computed norm.
 */
bool pibnd_test_norm1(params::poly_q r, uint64_t sigma_sqr) {
	array < mpz_t, params::poly_q::degree > coeffs;
	mpz_t norm, qDivBy2, tmp;
	params::poly_q t;

	/// Constructors
	mpz_inits(norm, qDivBy2, tmp, nullptr);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], (params::poly_q::bits_in_moduli_product() << 2));
	}

	r.poly2mpz(coeffs);
	mpz_fdiv_q_2exp(qDivBy2, params::poly_q::moduli_product(), 1);
	mpz_set_ui(norm, 0);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		util::center(coeffs[i], coeffs[i],
				params::poly_q::moduli_product(), qDivBy2);
		mpz_mul(tmp, coeffs[i], coeffs[i]);
		mpz_add(norm, norm, tmp);
	}

	// Compare to (sigma * sqrt(2N))^2 = sigma^2 * 2 * N.
	uint64_t bound = 2 * sigma_sqr * params::poly_q::degree;
	int result = mpz_cmp_ui(norm, bound) <= 0;

	mpz_clears(norm, qDivBy2, tmp, nullptr);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
	}

	return result;
}

bool pibnd_test_norm2(params::poly_q r) {
	array < mpz_t, params::poly_q::degree > coeffs;
	mpz_t norm, qDivBy2, tmp, q;
	params::poly_q t;

	/// Constructors
	mpz_inits(norm, qDivBy2, tmp, q, nullptr);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], (params::poly_q::bits_in_moduli_product() << 2));
	}

	r.poly2mpz(coeffs);
	mpz_fdiv_q_2exp(qDivBy2, params::poly_q::moduli_product(), 1);
	mpz_set_ui(norm, 0);
	mpz_set_str(q, PRIMEQ, 10);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_mod(coeffs[i], coeffs[i], q);
		util::center(coeffs[i], coeffs[i],
				params::poly_q::moduli_product(), qDivBy2);
		mpz_mul(tmp, coeffs[i], coeffs[i]);
		mpz_add(norm, norm, tmp);
	}

	mpz_set_str(tmp, BOUND_B, 10);
	mpz_mul(tmp, tmp, tmp);
	int result = mpz_cmp(norm, tmp) <= 0;

	mpz_clears(norm, qDivBy2, tmp, q, nullptr);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
	}

	return !result;
}

// Sample a challenge.
void pibnd_sample_chall(params::poly_q & f) {
	f = nfl::ZO_dist();
	f.ntt_pow_phi();
}

static void pibnd_prover(uint8_t h[BLAKE3_OUT_LEN], params::poly_q Z[V][NTI],
		params::poly_q A[R][V], params::poly_q t[TAU][V],
		params::poly_q s[TAU][V]) {
	params::poly_q W[R][NTI], C[TAU][NTI], SC[V][NTI];
	std::array < mpz_t, params::poly_q::degree > coeffs;
	mpz_t qDivBy2;
	int64_t coeff;

	mpz_init(qDivBy2);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], (params::poly_q::bits_in_moduli_product() << 2));
	}
	mpz_fdiv_q_2exp(qDivBy2, params::poly_q::moduli_product(), 1);

	do {
		/* Prover samples Y from Gaussian. */
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < NTI; j++) {
				for (size_t k = 0; k < params::poly_q::degree; k++) {
					if (i < V - 1) {
						coeff = sample_z(0.0, SIGMA_B1);
					} else {
						coeff = sample_z((__float128) 0.0,
								(__float128) SIGMA_B2);
					}
					mpz_set_si(coeffs[k], coeff);
				}
				Z[i][j].mpz2poly(coeffs);
				Z[i][j].ntt_pow_phi();
			}
		}

		/* Prover computes W = AY. */
		for (int i = 0; i < R; i++) {
			for (int j = 0; j < NTI; j++) {
				W[i][j] = 0;
				for (int k = 0; k < V; k++) {
					W[i][j] = W[i][j] + A[i][k] * Z[k][j];
				}
			}
		}

		pibnd_hash(h, A, t, W);

		/* Sample challenge from RNG seeded with hash. */
		nfl::fastrandombytes_seed(h);

		/* Verifier samples challenge matrix C. */
		for (int i = 0; i < TAU; i++) {
			for (int j = 0; j < NTI; j++) {
				C[i][j] = nfl::ZO_dist();
				C[i][j].ntt_pow_phi();
			}
		}

		nfl::fastrandombytes_reseed();

		/* Prover computes Z = Y + SC and performs rejection sampling. */
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < NTI; j++) {
				SC[i][j] = 0;
				for (int k = 0; k < TAU; k++) {
					SC[i][j] = SC[i][j] + s[k][i] * C[k][j];
				}
				Z[i][j] = Z[i][j] + SC[i][j];
			}
		}
	} while (pibnd_rej_sampling(Z, SC, SIGMA_B1 * SIGMA_B1) == 1);

	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
	}
	mpz_clear(qDivBy2);
}

int pibnd_verifier(uint8_t h1[BLAKE3_OUT_LEN], params::poly_q Z[V][NTI],
		params::poly_q A[R][V], params::poly_q t[TAU][V]) {
	params::poly_q W[R][NTI], C[TAU][NTI];
	uint8_t h2[BLAKE3_OUT_LEN];
	int result;

	/* Sample challenge from RNG seeded with hash. */
	nfl::fastrandombytes_seed(h1);

	/* Verifier samples challenge matrix C. */
	for (int i = 0; i < TAU; i++) {
		for (int j = 0; j < NTI; j++) {
			C[i][j] = nfl::ZO_dist();
			C[i][j].ntt_pow_phi();
		}
	}

	/* Verifier checks that W = AZ - TC. */
	for (int i = 0; i < R; i++) {
		for (int j = 0; j < NTI; j++) {
			W[i][j] = 0;
			for (int k = 0; k < V; k++) {
				W[i][j] = W[i][j] + A[i][k] * Z[k][j];
			}
		}
	}
	for (int i = 0; i < R; i++) {
		for (int j = 0; j < NTI; j++) {
			for (int k = 0; k < TAU; k++) {
				W[i][j] = W[i][j] - t[k][i] * C[k][j];
			}
		}
	}

	pibnd_hash(h2, A, t, W);

	result = memcmp(h1, h2, BLAKE3_OUT_LEN) == 0;
	for (int i = 0; i < V; i++) {
		for (int j = 0; j < NTI; j++) {
			Z[i][j].invntt_pow_invphi();
			if (i < V - 1) {
				result &= pibnd_test_norm1(Z[i][j], SIGMA_B1 * SIGMA_B1);
			} else {
				result &= pibnd_test_norm2(Z[i][j]);
			}
		}
	}
	return result;
}

static void test() {
	params::poly_q A[R][V], s[TAU][V], t[TAU][V], Z[V][NTI];
	uint8_t h1[BLAKE3_OUT_LEN];
	std::array < mpz_t, params::poly_q::degree > coeffs;
	gmp_randstate_t prng;
	mpz_t q;

	mpz_init(q);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], (params::poly_q::bits_in_moduli_product() << 2));
	}

	/* Create instances. */
	gmp_randinit_default(prng);
	mpz_set_str(q, PRIMEQ, 10);
	for (int i = 0; i < R; i++) {
		for (int j = 0; j < V; j++) {
			for (size_t k = 0; k < params::poly_q::degree; k++) {
				mpz_urandomb(coeffs[k], prng, LEVEL);
				mpz_mod(coeffs[k], coeffs[k], q);
			}
			A[i][j].mpz2poly(coeffs);
			A[i][j].ntt_pow_phi();
		}
	}

	/* Create a total of TAU relations t_i = A * s_i */
	for (int i = 0; i < TAU; i++) {
		for (int j = 0; j < V; j++) {
			pibnd_sample_chall(s[i][j]);
		}
		for (int j = 0; j < R; j++) {
			t[i][j] = 0;
			for (int k = 0; k < V; k++) {
				t[i][j] = t[i][j] + A[j][k] * s[i][k];
			}
		}
	}

	TEST_ONCE("BND proof is consistent") {
		pibnd_prover(h1, Z, A, t, s);
		TEST_ASSERT(pibnd_verifier(h1, Z, A, t) == 1, end);
	} TEST_END;

  end:

	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
	}
	mpz_clear(q);
	gmp_randclear(prng);
	return;
}

static void bench() {
	params::poly_q A[R][V], s[TAU][V], t[TAU][V], Z[V][NTI];
	uint8_t h1[BLAKE3_OUT_LEN];
	std::array < mpz_t, params::poly_q::degree > coeffs;
	gmp_randstate_t prng;
	mpz_t q;

	mpz_init(q);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], (params::poly_q::bits_in_moduli_product() << 2));
	}

	/* Create instances. */
	gmp_randinit_default(prng);
	mpz_set_str(q, PRIMEQ, 10);
	for (int i = 0; i < R; i++) {
		for (int j = 0; j < V; j++) {
			for (size_t k = 0; k < params::poly_q::degree; k++) {
				mpz_urandomb(coeffs[k], prng, LEVEL);
				mpz_mod(coeffs[k], coeffs[k], q);
			}
			A[i][j].mpz2poly(coeffs);
			A[i][j].ntt_pow_phi();
		}
	}

	/* Create a total of TAU relations t_i = A * s_i */
	for (int i = 0; i < TAU; i++) {
		for (int j = 0; j < V; j++) {
			pibnd_sample_chall(s[i][j]);
		}
		for (int j = 0; j < R; j++) {
			t[i][j] = 0;
			for (int k = 0; k < V; k++) {
				t[i][j] = t[i][j] + A[j][k] * s[i][k];
			}
		}
	}

	BENCH_SMALL("BND prover (N relations)", pibnd_prover(h1, Z, A, t, s));
	BENCH_SMALL("BND verifier (N relations)", pibnd_verifier(h1, Z, A, t));

	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
	}
	mpz_clear(q);
	gmp_randclear(prng);
	return;
}

int main(int argc, char *argv[]) {
	printf("\n** Tests for lattice-based BND proof:\n\n");
	test();

	printf("\n** Benchmarks for lattice-based BND proof:\n\n");
	bench();
}
