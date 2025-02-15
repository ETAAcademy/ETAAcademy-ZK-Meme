#include "test.h"
#include "bench.h"
#include "common.h"
#include <sys/random.h>

void ghl_sample_message(params::poly_q & m) {
	// Sample a short polynomial.
	std::array < mpz_t, DEGREE > coeffs;
	uint64_t buf;

	size_t bits_in_moduli_product = params::poly_p::bits_in_moduli_product();
	for (size_t i = 0; i < params::poly_p::degree; i++) {
		mpz_init2(coeffs[i], bits_in_moduli_product << 2);
	}

	for (size_t j = 0; j < params::poly_p::degree / 2; j += 32) {
		getrandom(&buf, sizeof(buf), 0);
		for (size_t k = 0; k < 64; k += 2) {
			mpz_set_ui(coeffs[j + k / 2], (buf >> k) % PRIMEP);
		}
	}
	m.mpz2poly(coeffs);

	for (size_t i = 0; i < params::poly_p::degree; i++) {
		mpz_clear(coeffs[i]);
	}
}

void bgv_sample_message(params::poly_p & m) {
	// Sample a short polynomial.
	std::array < mpz_t, DEGREE > coeffs;
	uint64_t buf;

	size_t bits_in_moduli_product = params::poly_p::bits_in_moduli_product();
	for (size_t i = 0; i < params::poly_p::degree; i++) {
		mpz_init2(coeffs[i], bits_in_moduli_product << 2);
	}

	for (size_t j = 0; j < params::poly_p::degree; j += 32) {
		getrandom(&buf, sizeof(buf), 0);
		for (size_t k = 0; k < 64; k += 2) {
			mpz_set_ui(coeffs[j + k / 2], (buf >> k) % PRIMEP);
		}
	}
	m.mpz2poly(coeffs);

	for (size_t i = 0; i < params::poly_p::degree; i++) {
		mpz_clear(coeffs[i]);
	}
}

void bgv_rdc(params::poly_p & m) {
	// Sample a short polynomial.
	std::array < mpz_t, DEGREE > coeffs;

	size_t bits_in_moduli_product = params::poly_p::bits_in_moduli_product();
	for (size_t i = 0; i < params::poly_p::degree; i++) {
		mpz_init2(coeffs[i], bits_in_moduli_product << 2);
	}

	m.poly2mpz(coeffs);
	for (size_t j = 0; j < params::poly_p::degree; j++) {
		mpz_mod_ui(coeffs[j], coeffs[j], PRIMEP);
	}
	m.mpz2poly(coeffs);

	for (size_t i = 0; i < params::poly_p::degree; i++) {
		mpz_clear(coeffs[i]);
	}
}

// Generate a key pair.
void bgv_keygen(bgvkey_t & pk, params::poly_q & sk) {
	params::poly_q e = nfl::ZO_dist();
	pk.a = nfl::uniform();

	sk = nfl::ZO_dist();
	sk.ntt_pow_phi();
	e.ntt_pow_phi();

	pk.b = pk.a * sk;
	for (size_t i = 0; i < PRIMEP; i++) {
		pk.b = pk.b + e;
	}
}

void bgv_keyshare(params::poly_q s[], size_t shares, params::poly_q & sk) {
	params::poly_q t = sk;
	for (size_t i = 1; i < shares; i++) {
		s[i] = nfl::uniform();
		s[i].ntt_pow_phi();
		t = t - s[i];
	}
	s[0] = t;
}

void bgv_encrypt(bgvenc_t & c, bgvkey_t & pk, params::poly_p & m) {
	std::array < mpz_t, DEGREE > coeffs;
	params::poly_q e1 = nfl::ZO_dist();
	params::poly_q e2 = nfl::ZO_dist();
	params::poly_q r = nfl::ZO_dist();

	e1.ntt_pow_phi();
	e2.ntt_pow_phi();
	r.ntt_pow_phi();

	c.u = pk.a * r;
	c.v = pk.b * r;
	for (size_t i = 0; i < PRIMEP; i++) {
		c.u = c.u + e1;
		c.v = c.v + e2;
	}

	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], params::poly_q::bits_in_moduli_product() << 2);
	}

	m.poly2mpz(coeffs);
	r.mpz2poly(coeffs);
	r.ntt_pow_phi();
	c.v = c.v + r;

	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
	}
}

void bgv_add(bgvenc_t & c, bgvenc_t & d, bgvenc_t & e) {
	c.u = d.u + e.u;
	c.v = d.v + e.v;
}

void bgv_decrypt(params::poly_p & m, bgvenc_t & c, params::poly_q & sk) {
	std::array < mpz_t, DEGREE > coeffs;
	params::poly_q t = c.v - sk * c.u;
	mpz_t qDivBy2;

	mpz_init(qDivBy2);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], params::poly_q::bits_in_moduli_product() << 2);
	}
	mpz_fdiv_q_2exp(qDivBy2, params::poly_q::moduli_product(), 1);

	// Reduce the coefficients
	t.invntt_pow_invphi();
	t.poly2mpz(coeffs);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		util::center(coeffs[i], coeffs[i], params::poly_q::moduli_product(),
				qDivBy2);
		mpz_mod_ui(coeffs[i], coeffs[i], PRIMEP);
	}
	m.mpz2poly(coeffs);

	mpz_clear(qDivBy2);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
	}
}

void bgv_distdec(params::poly_q & tj, bgvenc_t & c, params::poly_q & sj) {
	std::array < mpz_t, DEGREE > coeffs;
	params::poly_q mj, Ej;
	mpz_t qDivBy2, bound;

	mpz_init(qDivBy2);
	mpz_init(bound);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], params::poly_q::bits_in_moduli_product() << 2);
	}
	mpz_fdiv_q_2exp(qDivBy2, params::poly_q::moduli_product(), 1);
	mpz_set_str(bound, BOUND_D, 10);

	Ej = nfl::uniform();
	Ej.poly2mpz(coeffs);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		util::center(coeffs[i], coeffs[i], params::poly_q::moduli_product(),
				qDivBy2);
		mpz_mod(coeffs[i], coeffs[i], bound);
	}
	Ej.mpz2poly(coeffs);
	Ej.ntt_pow_phi();
	mj = sj * c.u;
	tj = mj;
	for (size_t i = 0; i < PRIMEP; i++) {
		tj = tj + Ej;
	}
}

void bgv_comb(params::poly_p & m, bgvenc_t & c, params::poly_q t[],
		size_t shares) {
	std::array < mpz_t, DEGREE > coeffs;
	params::poly_q v;
	mpz_t qDivBy2;

	mpz_init(qDivBy2);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_init2(coeffs[i], params::poly_q::bits_in_moduli_product() << 2);
	}
	mpz_fdiv_q_2exp(qDivBy2, params::poly_q::moduli_product(), 1);

	v = c.v - t[0];
	for (size_t i = 1; i < shares; i++) {
		v = v - t[i];
	}
	v.invntt_pow_invphi();
	// Reduce the coefficients modulo p
	v.poly2mpz(coeffs);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		util::center(coeffs[i], coeffs[i], params::poly_q::moduli_product(),
				qDivBy2);
		mpz_mod_ui(coeffs[i], coeffs[i], PRIMEP);
	}
	m.mpz2poly(coeffs);

	mpz_clear(qDivBy2);
	for (size_t i = 0; i < params::poly_q::degree; i++) {
		mpz_clear(coeffs[i]);
	}
}

#ifdef MAIN
static void test() {
	bgvkey_t pk;
	params::poly_q sk, s[PARTIES], t[PARTIES], acc, mq;
	params::poly_p m, _m;
	bgvenc_t c1, c2;

	bgv_keygen(pk, sk);
	bgv_sample_message(m);

	TEST_BEGIN("BGV encryption is consistent") {
		bgv_encrypt(c1, pk, m);
		bgv_decrypt(_m, c1, sk);
		TEST_ASSERT(m - _m == 0, end);
		bgv_sample_message(m);
		bgv_decrypt(_m, c1, sk);
		TEST_ASSERT(m - _m != 0, end);
		bgv_encrypt(c1, pk, m);
		bgv_keygen(pk, sk);
		bgv_decrypt(_m, c1, sk);
		TEST_ASSERT(m - _m != 0, end);
	} TEST_END;

	TEST_BEGIN("BGV encryption is additively homomorphic") {
		bgv_sample_message(m);
		bgv_sample_message(_m);
		bgv_encrypt(c1, pk, m);
		bgv_encrypt(c2, pk, _m);
		bgv_add(c1, c1, c2);
		m = m + _m;
		bgv_rdc(m);
		bgv_decrypt(_m, c1, sk);
		TEST_ASSERT(m - _m == 0, end);
	} TEST_END;

	TEST_BEGIN("BGV distributed decryption is consistent") {
		bgv_keygen(pk, sk);
		bgv_encrypt(c1, pk, m);
		bgv_keyshare(s, PARTIES, sk);
		acc = s[0];
		for (size_t j = 1; j < PARTIES; j++) {
			acc = acc + s[j];
		}
		TEST_ASSERT(sk - acc == 0, end);
		for (size_t j = 0; j < PARTIES; j++) {
			bgv_distdec(t[j], c1, s[j]);
		}
		bgv_comb(_m, c1, t, PARTIES);
		TEST_ASSERT(m - _m == 0, end);
	} TEST_END;

  end:
	return;
}

static void bench() {
	bgvkey_t pk;
	params::poly_q sk, s[PARTIES], t[PARTIES], acc;
	params::poly_p m, _m;
	bgvenc_t c;

	BENCH_SMALL("bgv_keygen", bgv_keygen(pk, sk));

	BENCH_BEGIN("bgv_sample_message") {
		BENCH_ADD(bgv_sample_message(m));
	} BENCH_END;

	BENCH_BEGIN("bgv_encrypt") {
		BENCH_ADD(bgv_encrypt(c, pk, m));
	} BENCH_END;

	BENCH_BEGIN("bgv_decrypt") {
		bgv_encrypt(c, pk, m);
		BENCH_ADD(bgv_decrypt(_m, c, sk));
	} BENCH_END;

	BENCH_BEGIN("bgv_add") {
		BENCH_ADD(bgv_add(c, c, c));
	} BENCH_END;

	bgv_keyshare(t, PARTIES, sk);
	BENCH_BEGIN("bgv_distdec") {
		bgv_encrypt(c, pk, m);
		BENCH_ADD(bgv_distdec(t[0], c, s[0]));
	} BENCH_END;

	for (size_t i = 1; i < PARTIES; i++) {
		bgv_distdec(t[i], c, s[i]);
	}

	BENCH_BEGIN("bgv_comb") {
		BENCH_ADD(bgv_comb(_m, c, t, PARTIES));
	} BENCH_END;
}

int main(int argc, char *arv[]) {
	printf("\n** Tests for BGV encryption:\n\n");
	test();

	printf("\n** Benchmarks for BGV encryption:\n\n");
	bench();

	return 0;
}
#endif
