#include <math.h>
#include <stdlib.h>

#include <flint/flint.h>
#include <flint/fmpz_mod_poly.h>

#include "common.h"
#include "test.h"
#include "bench.h"
#include "assert.h"
#include "blake3.h"

#define ETA         325
#define R           (HEIGHT+2)
#define V           (WIDTH+3)
#define TAU         1000

/* Had to move those to global to avoid overflowing the stack. */
params::poly_q A[R][V], s[TAU][V], t[TAU][R];
params::poly_big H0[V], _H[V], H[TAU][3];

static void pismall_hash(fmpz_t x, fmpz_t beta0, fmpz_t beta[TAU][3], fmpz_t q,
		commit_t & com) {
	uint8_t hash[BLAKE3_OUT_LEN];
	blake3_hasher hasher;
	flint_rand_t rand;
	ulong seed[2];

	flint_randinit(rand);

	blake3_hasher_init(&hasher);
	blake3_hasher_update(&hasher, (const uint8_t *)com.c1.data(), 16 * DEGREE);
	for (size_t i = 0; i < com.c2.size(); i++) {
		blake3_hasher_update(&hasher, (const uint8_t *)com.c2[i].data(),
				16 * DEGREE);
	}

	blake3_hasher_finalize(&hasher, hash, BLAKE3_OUT_LEN);

	memcpy(&seed[0], hash, sizeof(ulong));
	memcpy(&seed[1], hash + BLAKE3_OUT_LEN / 2, sizeof(ulong));
	flint_randseed(rand, seed[0], seed[1]);
	fmpz_randm(x, rand, q);
	fmpz_randm(beta0, rand, q);
	for (size_t i = 0; i < TAU; i++) {
		for (size_t j = 0; j < 3; j++) {
			fmpz_randm(beta[i][j], rand, q);
		}
	}
}

static void poly_to(params::poly_q & out, fmpz_mod_poly_t & in,
		const fmpz_mod_ctx_t ctx) {
	array < mpz_t, params::poly_q::degree > coeffs;

	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], (params::poly_q::bits_in_moduli_product() << 2));
	}

	for (size_t i = 0; i < params::poly_q::degree; i++) {
		fmpz_mod_poly_get_coeff_mpz(coeffs[i], in, i, ctx);
	}

	out.mpz2poly(coeffs);
	out.ntt_pow_phi();

	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
	}
}

static void poly_encode(params::poly_big & out, fmpz_mod_poly_t in0,
		fmpz_mod_poly_t in1, fmpz_t in[ETA], const fmpz_mod_ctx_t ctx) {
	array < mpz_t, params::poly_big::degree > coeffs;
	size_t i;

	for (i = 0; i < params::poly_big::degree; i++) {
		mpz_init2(coeffs[i], (params::poly_big::bits_in_moduli_product() << 2));
	}

	for (i = 0; i < params::poly_q::degree; i++) {
		fmpz_mod_poly_get_coeff_mpz(coeffs[i], in0, i, ctx);
	}
	for (; i < 2 * params::poly_q::degree; i++) {
		fmpz_mod_poly_get_coeff_mpz(coeffs[i], in1, i, ctx);
	}
	for (; i < 2 * params::poly_q::degree + ETA; i++) {
		fmpz_get_mpz(coeffs[i], in[i - 2 * params::poly_q::degree]);
	}

	out.mpz2poly(coeffs);
	out.ntt_pow_phi();

	for (size_t i = 0; i < params::poly_big::degree; i++) {
		mpz_clear(coeffs[i]);
	}
}

static void poly_from(fmpz_mod_poly_t & out, params::poly_q & in,
		const fmpz_mod_ctx_t ctx) {
	array < mpz_t, params::poly_q::degree > coeffs;

	in.invntt_pow_invphi();

	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], (params::poly_q::bits_in_moduli_product() << 2));
	}

	in.poly2mpz(coeffs);

	fmpz_mod_poly_zero(out, ctx);
	fmpz_mod_poly_fit_length(out, params::poly_q::degree, ctx);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		fmpz_mod_poly_set_coeff_mpz(out, i, coeffs[i], ctx);
	}

	in.ntt_pow_phi();

	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
	}
}

static void pismall_setup(fmpz_mod_poly_t lag[], fmpz_t a[TAU], fmpz_t & q,
		flint_rand_t prng, const fmpz_mod_ctx_t ctx) {
	array < mpz_t, params::poly_q::degree > coeffs;
	fmpz_t t, u;
	fmpz xs[TAU];

	fmpz_init(t);
	fmpz_init(u);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], (params::poly_q::bits_in_moduli_product() << 2));
	}
	for (size_t i = 0; i < TAU; i++) {
		fmpz_init(&xs[i]);
		fmpz_set(&xs[i], a[i]);
	}

	/* Compute l_0(X) = \Prod(X - a_i) */
	fmpz_mod_poly_product_roots_fmpz_vec(lag[0], xs, TAU, ctx);
	/* Compute l_i(X) = \Prod(X - a_j)(a_i - a_j) */
	for (size_t i = 1; i <= TAU; i++) {
		fmpz_set(t, &xs[i - 1]);
		fmpz_set(&xs[i - 1], &xs[TAU - 1]);
		fmpz_mod_poly_product_roots_fmpz_vec(lag[i], xs, TAU - 1, ctx);

		fmpz_set(&xs[i - 1], t);
		fmpz_set_ui(u, 1);
		for (size_t j = 0; j < TAU; j++) {
			if (i - 1 != j) {
				fmpz_sub(t, a[i - 1], a[j]);
				fmpz_mul(u, u, t);
			}
		}
		fmpz_invmod(u, u, q);
		fmpz_mod_poly_scalar_mul_fmpz(lag[i], lag[i], u, ctx);
	}

	fmpz_clear(t);
	fmpz_clear(u);
	for (size_t i = 0; i < TAU; i++) {
		fmpz_clear(&xs[i]);
	}
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
	}
}

static int pismall_prover(commit_t & com, fmpz_t x, fmpz_mod_poly_t f[V],
		fmpz_t rf[ETA], fmpz_mod_poly_t h[2][V], fmpz_t rh[ETA],
		vector < params::poly_q > rd, comkey_t & key,
		fmpz_mod_poly_t lag[TAU + 1], flint_rand_t prng,
		const fmpz_mod_ctx_t ctx) {
	array < mpz_t, params::poly_q::degree > coeffs, coeffs0, v[3][TAU][V];
	fmpz_mod_poly_t poly, zero;
	fmpz_t t, u, q, y[TAU], beta0, beta[TAU][3], r0[ETA], r[TAU][3][ETA];
	fmpz_mod_ctx_t ctx_q;
	vector < params::poly_q > d;
	params::poly_q s0[V];

	fmpz_init(t);
	fmpz_init(u);
	fmpz_init(q);
	fmpz_set_mpz(q, params::poly_q::moduli_product());
	fmpz_mod_ctx_init(ctx_q, q);

	fmpz_set_str(q, PRIMEQ, 10);
	fmpz_mod_poly_init(poly, ctx);
	fmpz_mod_poly_init(zero, ctx);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init(coeffs[i]);
		mpz_init(coeffs0[i]);
	}
	for (size_t i = 0; i < TAU; i++) {
		fmpz_init(y[i]);
		for (size_t j = 0; j < 3; j++) {
			for (size_t k = 0; k < V; k++) {
				for (size_t l = 0; l < params::poly_q::degree; l++) {
					mpz_init(v[j][i][k][l]);
				}
			}
		}
	}
	fmpz_init(beta0);
	for (size_t i = 0; i < TAU; i++) {
		for (size_t j = 0; j < 3; j++) {
			fmpz_init(beta[i][j]);
			for (size_t k = 0; k < ETA; k++) {
				fmpz_init(r[i][j][k]);
				fmpz_randm(r[i][j][k], prng, q);
			}
		}
	}
	for (size_t i = 0; i < ETA; i++) {
		fmpz_init(r0[i]);
		fmpz_randm(r0[i], prng, q);
		fmpz_randm(rh[i], prng, q);
	}
	for (size_t i = 0; i < V; i++) {
		fmpz_mod_poly_randtest(h[0][i], prng, params::poly_q::degree, ctx);
		fmpz_mod_poly_randtest(h[1][i], prng, params::poly_q::degree, ctx);
	}

	/* Generate s_0 = Z_q^vN, h = Z_q^2vn. */
	for (size_t i = 0; i < V; i++) {
		fmpz_mod_poly_randtest(poly, prng, params::poly_q::degree, ctx);
		poly_to(s0[i], poly, ctx);
	}

	/* Compute d = -A * s_0 and convert back from NTT due to commitment. */
	d.resize(R);
	for (int j = 0; j < R; j++) {
		d[j] = 0;
		for (int k = 0; k < V; k++) {
			d[j] = d[j] - A[j][k] * s0[k];
		}
		d[j].invntt_pow_invphi();
	}
	bdlop_commit(com, d, key, rd);

	/* Compute f \circ (f + 1) \circ (f - 1) explicitly by computing the v_i,j. */
	for (size_t k = 0; k < V; k++) {
		s0[k].invntt_pow_invphi();
		s0[k].poly2mpz(coeffs0);
		s0[k].ntt_pow_phi();
		for (size_t i = 0; i < TAU; i++) {
			s[i][k].invntt_pow_invphi();
			s[i][k].poly2mpz(coeffs);
			s[i][k].ntt_pow_phi();
			for (size_t l = 0; l < params::poly_q::degree; l++) {
				fmpz_set_mpz(t, coeffs[l]);
				fmpz_mod_sub_ui(u, t, 1, ctx_q);
				fmpz_mod_mul(t, t, u, ctx_q);
				fmpz_mod_add_ui(u, u, 2, ctx_q);
				fmpz_mod_mul(t, t, u, ctx_q);
				fmpz_get_mpz(v[0][i][k][l], t);

				fmpz_set_mpz(t, coeffs[l]);
				fmpz_set_mpz(u, coeffs0[l]);
				fmpz_mod_sub_ui(t, t, 1, ctx_q);
				fmpz_mod_mul(u, u, t, ctx_q);
				fmpz_mod_add_ui(t, t, 1, ctx_q);
				fmpz_mod_mul(u, u, t, ctx_q);
				fmpz_get_mpz(v[1][i][k][l], u);

				fmpz_set_mpz(t, coeffs[l]);
				fmpz_set_mpz(u, coeffs0[l]);
				fmpz_mod_sub_ui(u, u, 1, ctx_q);
				fmpz_mod_mul(u, u, t, ctx_q);
				fmpz_mod_add_ui(t, t, 1, ctx_q);
				fmpz_mod_mul(u, u, t, ctx_q);
				fmpz_set_mpz(t, v[1][i][k][l]);
				fmpz_mod_add_fmpz(u, u, t, ctx_q);
				fmpz_get_mpz(v[1][i][k][l], u);

				fmpz_set_mpz(t, coeffs[l]);
				fmpz_set_mpz(u, coeffs0[l]);
				fmpz_mod_add_ui(u, u, 1, ctx_q);
				fmpz_mod_mul(u, u, t, ctx_q);
				fmpz_mod_sub_ui(t, t, 1, ctx_q);
				fmpz_mod_mul(u, u, t, ctx_q);
				fmpz_set_mpz(t, v[1][i][k][l]);
				fmpz_mod_add_fmpz(u, u, t, ctx_q);
				fmpz_get_mpz(v[1][i][k][l], u);

				fmpz_set_mpz(t, coeffs[l]);
				fmpz_set_mpz(u, coeffs0[l]);
				fmpz_mod_sub_ui(u, u, 1, ctx_q);
				fmpz_mod_mul(t, t, u, ctx_q);
				fmpz_mod_add_ui(u, u, 2, ctx_q);
				fmpz_mod_mul(t, t, u, ctx_q);
				fmpz_get_mpz(v[2][i][k][l], t);

				fmpz_set_mpz(u, coeffs0[l]);
				fmpz_mod_sub_ui(t, u, 1, ctx_q);
				fmpz_mod_mul(u, u, t, ctx_q);
				fmpz_set_mpz(t, coeffs[l]);
				fmpz_mod_add_ui(t, t, 1, ctx_q);
				fmpz_mod_mul(u, u, t, ctx_q);
				fmpz_set_mpz(t, v[1][i][k][l]);
				fmpz_mod_add_fmpz(u, u, t, ctx_q);
				fmpz_get_mpz(v[2][i][k][l], u);

				fmpz_set_mpz(u, coeffs0[l]);
				fmpz_mod_add_ui(t, u, 1, ctx_q);
				fmpz_mod_mul(u, u, t, ctx_q);
				fmpz_set_mpz(t, coeffs[l]);
				fmpz_mod_sub_ui(t, t, 1, ctx_q);
				fmpz_mod_mul(u, u, t, ctx_q);
				fmpz_set_mpz(t, v[1][i][k][l]);
				fmpz_mod_add_fmpz(u, u, t, ctx_q);
				fmpz_get_mpz(v[2][i][k][l], u);
			}
		}
	}

	/* Encode H's as larger NTT. */
	for (size_t k = 0; k < V; k++) {
		poly_from(poly, s0[k], ctx);
		poly_encode(H0[k], poly, zero, r0, ctx);
		poly_encode(_H[k], h[0][k], h[1][k], rh, ctx);
		for (size_t i = 0; i < TAU; i++) {
			for (size_t j = 0; j < 3; j++) {
				for (size_t l = 0; l < params::poly_q::degree; l++) {
					fmpz_mod_poly_set_coeff_mpz(poly, l, v[j][i][k][l], ctx_q);
				}
				if (j == 0) {
					poly_from(zero, s[i][j], ctx_q);
					poly_encode(H[i][j], zero, poly, r[i][j], ctx_q);
				}
				fmpz_mod_poly_zero(zero, ctx);
				poly_encode(H[i][j], zero, poly, r[i][j], ctx_q);
			}
		}
	}

	pismall_hash(x, beta0, beta, q, com);

	/* Compute f = s_0 * l_0(x). */
	fmpz_mod_poly_evaluate_fmpz(y[0], lag[0], x, ctx_q);
	for (int i = 0; i < V; i++) {
		poly_from(poly, s0[i], ctx_q);
		fmpz_mod_poly_scalar_mul_fmpz(f[i], poly, y[0], ctx_q);
	}
	/* Compute remaining f = f(x) = \Sum s_i * l_i(x). */
	for (size_t i = 1; i <= TAU; i++) {
		fmpz_mod_poly_evaluate_fmpz(y[i - 1], lag[i], x, ctx_q);
		for (int j = 0; j < V; j++) {
			poly_from(poly, s[i - 1][j], ctx_q);
			fmpz_mod_poly_scalar_mul_fmpz(poly, poly, y[i - 1], ctx_q);
			fmpz_mod_poly_add(f[j], f[j], poly, ctx_q);
		}
	}

	/* Compute _rf = r0 * l_0(x) + Sum r_i,j * l_i(x) * l_0(x)^j */
	/* and _rh = r0 * beta_0 + Sum r_i,j * beta_i,j */
	for (size_t k = 0; k < ETA; k++) {
		fmpz_mod_mul(rf[k], r0[k], y[0], ctx);
		fmpz_mod_mul(t, r0[k], beta0, ctx);
		fmpz_mod_add(rh[k], rh[k], t, ctx_q);
		for (size_t i = 1; i <= TAU; i++) {
			for (size_t j = 0; j < 3; j++) {
				fmpz_mod_mul(t, r[i - 1][j][k], y[i - 1], ctx_q);
				for (size_t l = 0; l < j; l++) {
					fmpz_mod_mul(t, t, y[i], ctx_q);
				}
				fmpz_mod_add(rf[k], rf[k], t, ctx_q);
				fmpz_mod_mul(t, r[i - 1][j][k], beta[i - 1][j], ctx_q);
				fmpz_mod_add(rh[k], rh[k], t, ctx_q);
			}
		}
	}

	/* Compute _h = h + s_0 * beta_0 + \sum beta_i,j * (\delta_i * si, v_i,j) */
	for (size_t k = 0; k < V; k++) {
		poly_from(poly, s0[k], ctx);
		fmpz_mod_poly_scalar_mul_fmpz(poly, poly, beta0, ctx);
		fmpz_mod_poly_add(h[0][k], h[0][k], poly, ctx);
		for (size_t i = 1; i <= TAU; i++) {
			for (size_t j = 0; j < 3; j++) {
				if (j == 0) {
					poly_from(poly, s[i][k], ctx_q);
					fmpz_mod_poly_scalar_mul_fmpz(poly, poly, beta[i - 1][j],
							ctx_q);
					fmpz_mod_poly_add(h[0][k], h[0][k], poly, ctx);
				}
				for (size_t l = 0; l < params::poly_q::degree; l++) {
					fmpz_mod_poly_set_coeff_mpz(poly, l, v[j][i - 1][k][l],
							ctx_q);
				}
				fmpz_mod_poly_scalar_mul_fmpz(poly, poly, beta[i - 1][j],
						ctx_q);
				fmpz_mod_poly_add(h[1][k], h[1][k], poly, ctx);
			}
		}
	}

	fmpz_clear(t);
	fmpz_clear(u);
	fmpz_clear(q);
	fmpz_mod_poly_clear(poly, ctx);
	fmpz_mod_poly_clear(zero, ctx);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
		mpz_clear(coeffs0[i]);
	}
	for (size_t i = 0; i < TAU; i++) {
		fmpz_clear(y[i]);
		for (size_t j = 0; j < 3; j++) {
			for (size_t k = 0; k < V; k++) {
				for (size_t l = 0; l < params::poly_q::degree; l++) {
					mpz_clear(v[j][i][k][l]);
				}
			}
		}
	}
	fmpz_clear(beta0);
	for (size_t i = 0; i < TAU; i++) {
		for (size_t j = 0; j < 3; j++) {
			fmpz_clear(beta[i][j]);
			for (size_t k = 0; k < ETA; k++) {
				fmpz_clear(r[i][j][k]);
			}
		}
	}
	for (size_t i = 0; i < ETA; i++) {
		fmpz_clear(r0[i]);
	}
	return 1;
}

static int pismall_verifier(commit_t & com, fmpz_mod_poly_t f[V],
		fmpz_t rf[ETA], vector < params::poly_q > rd, comkey_t & key,
		fmpz_mod_poly_t lag[TAU + 1], flint_rand_t prng,
		const fmpz_mod_ctx_t ctx) {
	array < mpz_t, params::poly_q::degree > coeffs;
	fmpz_mod_poly_t poly, r[R], _d[R];
	fmpz_mod_ctx_t ctx_q;
	fmpz_t q, y, x, beta0, beta[TAU][3];
	params::poly_q one = 1;
	vector < params::poly_q > m;

	fmpz_init(q);
	fmpz_init(x);
	fmpz_init(y);
	fmpz_init(beta0);
	for (size_t i = 0; i < TAU; i++) {
		for (size_t j = 0; j < 3; j++) {
			fmpz_init(beta[i][j]);
		}
	}

	fmpz_set_str(q, PRIMEQ, 10);
	pismall_hash(x, beta0, beta, q, com);

	fmpz_set_mpz(q, params::poly_q::moduli_product());
	fmpz_mod_ctx_init(ctx_q, q);
	fmpz_mod_poly_init(poly, ctx_q);

	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], (params::poly_q::bits_in_moduli_product() << 2));
	}
	for (int i = 0; i < R; i++) {
		fmpz_mod_poly_init(r[i], ctx_q);
		fmpz_mod_poly_zero(r[i], ctx_q);
		fmpz_mod_poly_init(_d[i], ctx);
	}

	for (int i = 1; i <= TAU; i++) {
		fmpz_mod_poly_evaluate_fmpz(y, lag[i], x, ctx_q);
		for (int j = 0; j < R; j++) {
			poly_from(poly, t[i - 1][j], ctx_q);
			fmpz_mod_poly_scalar_mul_fmpz(poly, poly, y, ctx_q);
			fmpz_mod_poly_add(r[j], r[j], poly, ctx_q);
		}
	}

	/* Compute d = A * _f. */
	for (int j = 0; j < V; j++) {
		for (int i = 0; i < R; i++) {
			poly_to(one, f[j], ctx_q);
			one = one * A[i][j];
			poly_from(poly, one, ctx_q);
			fmpz_mod_poly_sub(r[i], r[i], poly, ctx_q);
		}
	}

	fmpz_mod_poly_evaluate_fmpz(y, lag[0], x, ctx_q);
	fmpz_invmod(y, y, q);
	m.resize(R);
	for (int i = 0; i < R; i++) {
		fmpz_mod_poly_scalar_mul_fmpz(r[i], r[i], y, ctx_q);
		poly_to(m[i], r[i], ctx_q);
		m[i].invntt_pow_invphi();
	}
	one = 1;
	one.ntt_pow_phi();
	int result = bdlop_open(com, m, key, rd, one);

	fmpz_clear(q);
	fmpz_clear(x);
	fmpz_clear(y);
	fmpz_mod_poly_clear(poly, ctx);
	for (int i = 0; i < R; i++) {
		fmpz_mod_poly_clear(r[i], ctx);
		fmpz_mod_poly_clear(_d[i], ctx);
	}
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
	}
	fmpz_clear(beta0);
	for (size_t i = 0; i < TAU; i++) {
		for (size_t j = 0; j < 3; j++) {
			fmpz_clear(beta[i][j]);
		}
	}
	return result;
}

static void test(flint_rand_t rand) {
	fmpz_t q, x, a[TAU], rf[ETA], rh[ETA];
	fmpz_mod_poly_t poly;
	fmpz_mod_ctx_t ctx, ctx_q;
	comkey_t key;
	commit_t com;
	vector < params::poly_q > rd;
	fmpz_mod_poly_t f[V], lag[TAU + 1], h[2][V];

	fmpz_init(q);
	fmpz_init(x);

	fmpz_set_mpz(q, params::poly_q::moduli_product());
	fmpz_mod_ctx_init(ctx_q, q);

	fmpz_set_str(q, PRIMEQ, 10);
	fmpz_mod_ctx_init(ctx, q);
	fmpz_mod_poly_init(poly, ctx);
	for (int i = 0; i < V; i++) {
		fmpz_mod_poly_init(f[i], ctx);
		fmpz_mod_poly_init(h[0][i], ctx);
		fmpz_mod_poly_init(h[1][i], ctx);
	}
	for (size_t i = 0; i < TAU; i++) {
		fmpz_init(a[i]);
		fmpz_randm(a[i], rand, q);
	}

	for (int i = 0; i < R; i++) {
		for (int j = 0; j < V; j++) {
			fmpz_mod_poly_randtest(poly, rand, params::poly_q::degree, ctx);
			poly_to(A[i][j], poly, ctx);
		}
	}
	for (size_t i = 0; i < ETA; i++) {
		fmpz_init(rf[i]);
		fmpz_init(rh[i]);
	}

	/* Create a total of TAU relations t_i = A * s_i */
	for (int i = 0; i < TAU; i++) {
		fmpz_mod_poly_init(lag[i], ctx);
		for (int j = 0; j < V; j++) {
			s[i][j] = nfl::hwt_dist {
			2 *DEGREE / 3};
			s[i][j].ntt_pow_phi();
		}
		for (int j = 0; j < R; j++) {
			t[i][j] = 0;
			for (int k = 0; k < V; k++) {
				t[i][j] = t[i][j] + A[j][k] * s[i][k];
			}
		}
	}
	fmpz_mod_poly_init(lag[TAU], ctx);

	rd.resize(R);
	TEST_BEGIN("conversion is correct") {
		for (int i = 0; i < R; i++) {
			for (int j = 0; j < V; j++) {
				poly_from(f[j], A[i][j], ctx);
				poly_to(rd[0], f[j], ctx);
				TEST_ASSERT(rd[0] == A[i][j], end);
			}
		}
		fmpz_mod_poly_zero(poly, ctx_q);
		fmpz_mod_poly_set_coeff_ui(poly, params::poly_q::degree, 1, ctx_q);
		fmpz_mod_poly_set_coeff_ui(poly, 0, 1, ctx_q);

		for (int i = 0; i < R; i++) {
			for (int j = 0; j < V; j++) {
				rd[0] = A[i][j] * A[i][j];
				poly_from(f[j], A[i][j], ctx_q);
				fmpz_mod_poly_mulmod(f[j], f[j], f[j], poly, ctx_q);
				poly_to(rd[1], f[j], ctx_q);
				TEST_ASSERT(rd[0] == rd[1], end);
			}
		}
	} TEST_END;

	pismall_setup(lag, a, q, rand, ctx);

	bdlop_keygen(key);
	bdlop_sample_rand(rd);

	TEST_ONCE("AEX proof is consistent") {
		pismall_prover(com, x, f, rf, h, rh, rd, key, lag, rand, ctx);
		TEST_ASSERT(pismall_verifier(com, f, rf, rd, key, lag, rand, ctx) == 1,
				end);
	} TEST_END;

	printf("\n** Benchmarks for lattice-based AEX proof:\n\n");
	BENCH_SMALL("pismall_setup", pismall_setup(lag, a, q, rand, ctx));
	BENCH_SMALL("pismall_prover", pismall_prover(com, x, f, rf, h, rh, rd, key,
					lag, rand, ctx));
	BENCH_SMALL("pismall_verifier", pismall_verifier(com, f, rf, rd, key, lag,
					rand, ctx));

  end:
	for (int i = 0; i < V; i++) {
		fmpz_mod_poly_clear(f[i], ctx);
		fmpz_mod_poly_clear(h[0][i], ctx);
		fmpz_mod_poly_clear(h[1][i], ctx);
	}
	for (int i = 0; i <= TAU; i++) {
		fmpz_mod_poly_clear(lag[i], ctx);
	}
	for (size_t i = 0; i < TAU; i++) {
		fmpz_clear(a[i]);
	}
	for (size_t i = 0; i < ETA; i++) {
		fmpz_clear(rf[i]);
		fmpz_clear(rh[i]);
	}
	fmpz_mod_poly_clear(poly, ctx);
	fmpz_mod_ctx_clear(ctx);
	fmpz_clear(q);
	fmpz_clear(x);
}

int main() {
	flint_rand_t rand;
	flint_randinit(rand);

	printf("\n** Tests for lattice-based AEX proof:\n\n");
	test(rand);
	flint_randclear(rand);
}
