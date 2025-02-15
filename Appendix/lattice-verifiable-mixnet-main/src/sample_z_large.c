/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Integer samplers               *
 * ****************************** */

#include <stdint.h>
#include "sample_z_large.h"
#include "param.h"
#include "fastrandombytes.h"
#include "poly.h"

#include <x86intrin.h>

using namespace nfl;

/* Box-Muller sampler precision: 96 bits */
#define FP_PRECISION 96
static const __float128 FP_FACTOR = 1.0Q / (((__uint128_t)1) << FP_PRECISION);
#define BOX_MULLER_BYTES (12 * 2)

/* Constants used by COSAC */
static const __float128 pi2 = 2 * M_PIq;
static const __float128 sqrt_pi2 = 2 / (M_SQRT1_2q * M_2_SQRTPIq);

/* Parameters used by FACCT comparison */
#define COMP_ENTRY_SIZE 21
#define COSAC_EXP_MANTISSA_PRECISION 112
static const __uint128_t COSAC_EXP_MANTISSA_MASK = (((__uint128_t)1) << COSAC_EXP_MANTISSA_PRECISION) - 1;
#define COSAC_R_MANTISSA_PRECISION (COSAC_EXP_MANTISSA_PRECISION + 1)
static const __uint128_t COSAC_R_MANTISSA_MASK = (((__uint128_t)1) << COSAC_R_MANTISSA_PRECISION) - 1;
#define COSAC_R_EXPONENT_L (8 * COMP_ENTRY_SIZE - COSAC_R_MANTISSA_PRECISION)
static const __uint128_t COSAC_FLOAT128_ONE = ((__uint128_t)0x3fff) << 112;

#define DISCRETE_BYTES (2 * COMP_ENTRY_SIZE)

/* Constants used by FACCT */
#define CDT_ENTRY_SIZE 16
#define CDT_LOW_MASK 0x7fffffffffffffff
#define CDT_LENGTH 9 /* [0..tau*sigma]=[0..9] */

#define BERNOULLI_ENTRY_SIZE 9 /* 72bit randomness */

/* the closest integer k such that k*sigma_0=sigma */
#define BINARY_SAMPLER_K 11284

/* -1/k^2 */
#define BINARY_SAMPLER_K_2_INV (-1.0/(BINARY_SAMPLER_K * BINARY_SAMPLER_K))

#define EXP_MANTISSA_PRECISION 52
#define EXP_MANTISSA_MASK ((1LL << EXP_MANTISSA_PRECISION) - 1)
#define R_MANTISSA_PRECISION (EXP_MANTISSA_PRECISION + 1)
#define R_MANTISSA_MASK ((1LL << R_MANTISSA_PRECISION) - 1)
#define R_EXPONENT_L (8 * BERNOULLI_ENTRY_SIZE - R_MANTISSA_PRECISION)

#define DOUBLE_ONE (1023LL << 52)

#define UNIFORM_SIZE 4
#define UNIFORM_REJ 23
#define BARRETT_BITSHIFT (UNIFORM_SIZE * 8)

#define BARRETT_FACTOR ((1LL << BARRETT_BITSHIFT) / BINARY_SAMPLER_K)
#define UNIFORM_Q (BINARY_SAMPLER_K * BARRETT_FACTOR)

#define BASE_TABLE_SIZE (4 * CDT_ENTRY_SIZE)
#define BERNOULLI_TABLE_SIZE (4 * BERNOULLI_ENTRY_SIZE)

/* CDT table */
static const __m256i V_CDT[][2] = {{{2200310400551559144, 2200310400551559144, 2200310400551559144, 2200310400551559144}, {3327841033070651387, 3327841033070651387, 3327841033070651387, 3327841033070651387}},
{{7912151619254726620, 7912151619254726620, 7912151619254726620, 7912151619254726620}, {380075531178589176, 380075531178589176, 380075531178589176, 380075531178589176}},
{{5167367257772081627, 5167367257772081627, 5167367257772081627, 5167367257772081627}, {11604843442081400, 11604843442081400, 11604843442081400, 11604843442081400}},
{{5081592746475748971, 5081592746475748971, 5081592746475748971, 5081592746475748971}, {90134450315532, 90134450315532, 90134450315532, 90134450315532}},
{{6522074513864805092, 6522074513864805092, 6522074513864805092, 6522074513864805092}, {175786317361, 175786317361, 175786317361, 175786317361}},
{{2579734681240182346, 2579734681240182346, 2579734681240182346, 2579734681240182346}, {85801740, 85801740, 85801740, 85801740}},
{{8175784047440310133, 8175784047440310133, 8175784047440310133, 8175784047440310133}, {10472, 10472, 10472, 10472}},
{{2947787991558061753, 2947787991558061753, 2947787991558061753, 2947787991558061753}, {0, 0, 0, 0}},
{{22489665999543, 22489665999543, 22489665999543, 22489665999543}, {0, 0, 0, 0}}};

static const __m256i V_CDT_LOW_MASK = {CDT_LOW_MASK, CDT_LOW_MASK, CDT_LOW_MASK, CDT_LOW_MASK};

static const __m256i V_K_K_K_K = {BINARY_SAMPLER_K, BINARY_SAMPLER_K, BINARY_SAMPLER_K, BINARY_SAMPLER_K};

/* coefficients of the exp evaluation polynomial */
static const __m256i EXP_COFF[] = {{0x3e833b70ffa2c5d4, 0x3e833b70ffa2c5d4, 0x3e833b70ffa2c5d4, 0x3e833b70ffa2c5d4},
								   {0x3eb4a480fda7e6e1, 0x3eb4a480fda7e6e1, 0x3eb4a480fda7e6e1, 0x3eb4a480fda7e6e1},
								   {0x3ef01b254493363f, 0x3ef01b254493363f, 0x3ef01b254493363f, 0x3ef01b254493363f},
								   {0x3f242e0e0aa273cc, 0x3f242e0e0aa273cc, 0x3f242e0e0aa273cc, 0x3f242e0e0aa273cc},
								   {0x3f55d8a2334ed31b, 0x3f55d8a2334ed31b, 0x3f55d8a2334ed31b, 0x3f55d8a2334ed31b},
								   {0x3f83b2aa56db0f1a, 0x3f83b2aa56db0f1a, 0x3f83b2aa56db0f1a, 0x3f83b2aa56db0f1a},
								   {0x3fac6b08e11fc57e, 0x3fac6b08e11fc57e, 0x3fac6b08e11fc57e, 0x3fac6b08e11fc57e},
								   {0x3fcebfbdff556072, 0x3fcebfbdff556072, 0x3fcebfbdff556072, 0x3fcebfbdff556072},
								   {0x3fe62e42fefa7fe6, 0x3fe62e42fefa7fe6, 0x3fe62e42fefa7fe6, 0x3fe62e42fefa7fe6},
								   {0x3ff0000000000000, 0x3ff0000000000000, 0x3ff0000000000000, 0x3ff0000000000000}};

static const __m256d V_INT64_DOUBLE = {0x0010000000000000, 0x0010000000000000, 0x0010000000000000, 0x0010000000000000};
static const __m256d V_DOUBLE_INT64 = {0x0018000000000000, 0x0018000000000000, 0x0018000000000000, 0x0018000000000000};

static const __m256i V_EXP_MANTISSA_MASK = {EXP_MANTISSA_MASK, EXP_MANTISSA_MASK, EXP_MANTISSA_MASK, EXP_MANTISSA_MASK};
static const __m256i V_RES_MANTISSA = {1LL << EXP_MANTISSA_PRECISION, 1LL << EXP_MANTISSA_PRECISION, 1LL << EXP_MANTISSA_PRECISION, 1LL << EXP_MANTISSA_PRECISION};
static const __m256i V_RES_EXPONENT = {R_EXPONENT_L - 1023 + 1, R_EXPONENT_L - 1023 + 1, R_EXPONENT_L - 1023 + 1, R_EXPONENT_L - 1023 + 1};
static const __m256i V_R_MANTISSA_MASK = {R_MANTISSA_MASK, R_MANTISSA_MASK, R_MANTISSA_MASK, R_MANTISSA_MASK};
static const __m256i V_1 = {1, 1, 1, 1};
static const __m256i V_DOUBLE_ONE = {DOUBLE_ONE, DOUBLE_ONE, DOUBLE_ONE, DOUBLE_ONE};

static const __m256d V_K_2_INV = {BINARY_SAMPLER_K_2_INV, BINARY_SAMPLER_K_2_INV, BINARY_SAMPLER_K_2_INV, BINARY_SAMPLER_K_2_INV};

static inline uint64_t load_40(const unsigned char *x)
{
	return ((uint64_t)(*x)) | (((uint64_t)(*(x + 1))) << 8) | (((uint64_t)(*(x + 2))) << 16) | (((uint64_t)(*(x + 3))) << 24) | (((uint64_t)(*(x + 4))) << 32);
}

static inline __uint128_t load_96(const unsigned char *x)
{
	return ((__uint128_t)(*x)) | (((__uint128_t)(*(x + 1))) << 8) | (((__uint128_t)(*(x + 2))) << 16) | (((__uint128_t)(*(x + 3))) << 24) | (((__uint128_t)(*(x + 4))) << 32) | (((__uint128_t)(*(x + 5))) << 40) | (((__uint128_t)(*(x + 6))) << 48) | (((__uint128_t)(*(x + 7))) << 56) | (((__uint128_t)(*(x + 8))) << 64) | (((__uint128_t)(*(x + 9))) << 72) | (((__uint128_t)(*(x + 10))) << 80) | (((__uint128_t)(*(x + 11))) << 88);
}

static inline int64_t cosac_comp(const unsigned char *r, const __float128 x)
{
	__uint128_t res = *((__uint128_t *)(&x));
	__uint128_t res_mantissa;
	uint64_t res_exponent;
	__uint128_t r1;
	uint64_t r2;
	__uint128_t r_mantissa;
	uint64_t r_exponent;

	res_mantissa = (res & COSAC_EXP_MANTISSA_MASK) | (((__uint128_t)1) << COSAC_EXP_MANTISSA_PRECISION);
	res_exponent = COSAC_R_EXPONENT_L - 0x3fff + 1 + (res >> COSAC_EXP_MANTISSA_PRECISION);

	r1 = *((__uint128_t *)r);
	r2 = load_40(r + 16);

	r_mantissa = r1 & COSAC_R_MANTISSA_MASK;
	r_exponent = (r1 >> COSAC_R_MANTISSA_PRECISION) | (r2 << (128 - COSAC_R_MANTISSA_PRECISION));

	return (res == COSAC_FLOAT128_ONE) || ((r_mantissa < res_mantissa) && (r_exponent < (1LL << res_exponent)));
}

/* New COSAC sampler.
 * This is the sampling algorithm from:
 * Shuo Sun, Yongbin Zhou, Yunfeng Ji, Rui Zhang, & Yang Tao. (2021). Generic, Efficient and Isochronous Gaussian Sampling over the Integers.
 * https://eprint.iacr.org/2021/199 */
int64_t sample_z(const __float128 center, const __float128 sigma)
{
	unsigned char r[DISCRETE_BYTES];

	__float128 c, cr, rc;
	__float128 yr, rej;
	__float128 sigma2;
	__float128 yrc;

	uint64_t i;
	int64_t cmp1;
	uint64_t head = 2;

	unsigned char r_bm[BOX_MULLER_BYTES];

	__float128 r1, r2;
	__float128 norm[2];

	cr = roundq(center);
	c = center - cr;

	sigma2 = -sigma * sigma * 2;

	rc = expq(c * c / sigma2) / (sigma * sqrt_pi2);

	fastrandombytes(r, COMP_ENTRY_SIZE);

	if (cosac_comp(r, rc))
	{
		return cr;
	}

	while (1)
	{
		fastrandombytes(r, DISCRETE_BYTES);

		for (i = 0; i < 2; i++, head++)
		{
			if (head >= 2)
			{
				head = 0;

				fastrandombytes((unsigned char *)r_bm, BOX_MULLER_BYTES);

				r1 = (load_96(r) + 1) * FP_FACTOR;
				r2 = (load_96(r + 12) + 1) * FP_FACTOR;

				r1 = sqrtq(-2 * logq(r1)) * sigma;
				r2 = r2 * pi2;

				sincosq(r2, norm, norm + 1);

				norm[0] = r1 * norm[0];
				norm[1] = r1 * norm[1];
			}

			yr = norm[head] + c;

			cmp1 = yr >= 0;

			yr = floorq(yr) + cmp1;

			yrc = yr - c;
			rej = yrc + norm[head];
			yrc = yrc - norm[head];
			rej = expq(rej * yrc / sigma2);

			if (cosac_comp(r + i * COMP_ENTRY_SIZE, rej))
			{
				return yr + cr;
			}
		}
	}
}
