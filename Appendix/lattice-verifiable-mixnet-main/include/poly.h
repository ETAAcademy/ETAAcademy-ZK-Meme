/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Polynomial arithmetic          *
 * ****************************** */
 
#ifndef _POLY_H
#define _POLY_H

#include "param.h"

#include <stdint.h>

#include <gmp.h>
#include <mpc.h>

#include <complex.h>
#include <quadmath.h>

typedef struct 
{
	__complex128 poly[N];
} POLY_FFT;

typedef struct 
{
	mpc_t poly[N];
} POLY_FFT_HIGH;

typedef struct
{
	mpz_t poly[N];
} POLY_Z; 

typedef struct
{
	int64_t poly[N];
} POLY_64;

typedef struct
{
	__float128 poly[N];
} POLY_R;

static inline void poly_z_init(POLY_Z *a, const uint64_t n)
{
	uint64_t i;
	
	for (i = 0; i < n; i++)
	{
		mpz_init(a->poly[i]);
	}
}

static inline void poly_z_clear(POLY_Z *a, const uint64_t n)
{
	uint64_t i;
	
	for (i = 0; i < n; i++)
	{
		mpz_clear(a->poly[i]);
	}
}

static inline void poly_fft_init_high(POLY_FFT_HIGH *a, const uint64_t n, const uint64_t prec)
{
	uint64_t i;
	
	for (i = 0; i < n; i++)
	{
		mpc_init2(a->poly[i], prec);
	}
}

static inline void poly_fft_clear_high(POLY_FFT_HIGH *a, const uint64_t n)
{
	uint64_t i;
	
	for (i = 0; i < n; i++)
	{
		mpc_clear(a->poly[i]);
	}
}

void poly_mul_zz(POLY_Z *out, const POLY_Z *a, const POLY_Z *b, const uint64_t n);
void poly_mul_6464(POLY_64 *out, const POLY_64 *a, const POLY_64 *b, const uint64_t n);

#endif
