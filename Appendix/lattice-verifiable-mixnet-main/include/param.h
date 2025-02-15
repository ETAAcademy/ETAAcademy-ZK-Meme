#ifndef _PARAM_H
#define _PARAM_H

#include "common.h"
#include <stdint.h>
#include <quadmath.h>

#define N DEGREE

#define Q 274810798081LL /* 2^38 - 2^26 + 1 */

#define L 2

static const __float128 sigma_l[L + 1] = {9583.7471944500068780694380Q, 713170.79131244259284534655Q, 65489528.122674807214409517Q};

static const __float128 norm_l[L] = {376210269336.07783606747078Q, 3124915676658988.2977945632Q};

static const uint64_t reduce_k_prec[L] = {512, 1536};
#define FFT_PREC 1536

#endif
