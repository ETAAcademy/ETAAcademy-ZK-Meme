#include <cstddef>

#include <gmpxx.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <nfl.hpp>

#include "bench.h"
#include <sys/random.h>

using namespace std;

#ifndef COMMON_H
#define COMMON_H

/* Parameter v in the commitment  scheme (laximum l1-norm of challs). */
#define NONZERO     36
/* Security level to attain. */
#define LEVEL       128
/* The \infty-norm bound of certain elements. */
#define BETA 	    1
/* Width k of the comming matrix. */
#define WIDTH 	    4
/* Height of the commitment matrix. */
#define HEIGHT 	    1
/* Dimension of the committed messages. */
#ifndef SIZE
#define SIZE        2
#endif
/* Large modulus. */
#define PRIMEQ      "302231454903657293688833"
/* Small modulus. */
#define PRIMEP      2
/* Degree of the irreducible polynomial. */
#define DEGREE      4096
/* Sigma for the commitment gaussian distribution. */
#define SIGMA_C     (1u << 12)
/* Sigma for the boundness proof. */
#define SIGMA_B1     (11585u)
/* Sigma for the boundness proof. */
#define SIGMA_B2     (1e66l)
/* Norm bound for boundness proof. */
#define BOUND_B     "6678434726570384949248"
/* Parties that run the distributed decryption protocol. */
#define PARTIES     4
/* Security level for Distributed Decryption. */
#define BGVSEC      40
/* Bound for Distributed Decryption = 2^BGVSEC * q/(2 * p * PARTIES). */
#define BOUND_D     "750837175903336127688539820910095018"

namespace params {
    using poly_p = nfl::poly_from_modulus<uint32_t, DEGREE, 30>;
    using poly_q = nfl::poly_from_modulus<uint64_t, DEGREE, 124>;
    using poly_big = nfl::poly_from_modulus<uint64_t, 4 * DEGREE, 124>;
}

/*============================================================================*/
/* Type definitions                                                           */
/*============================================================================*/

/* Class that represents a commitment key pair. */
class comkey_t {
    public:
       params::poly_q A1[HEIGHT][WIDTH - HEIGHT];
       params::poly_q A2[SIZE][WIDTH];
};

/* Class that represents a commitment in CRT representation. */
class commit_t {
    public:
      params::poly_q c1;
      vector<params::poly_q> c2;
};

/* Class that represents a BGV key pair. */
class bgvkey_t {
    public:
       params::poly_q a;
       params::poly_q b;
};

class bgvenc_t {
    public:
       params::poly_q u;
       params::poly_q v;
};

#include "util.hpp"

void bdlop_sample_rand(vector<params::poly_q>& r);
void bdlop_sample_chal(params::poly_q& f);
bool bdlop_test_norm(params::poly_q r, uint64_t sigma_sqr);
void bdlop_commit(commit_t& com, vector<params::poly_q> m, comkey_t& key, vector<params::poly_q> r);
int  bdlop_open(commit_t& com, vector<params::poly_q> m, comkey_t& key, vector<params::poly_q> r, params::poly_q& f);
void bdlop_keygen(comkey_t& key);

void bgv_sample_message(params::poly_p& r);
void bgv_sample_short(params::poly_q& r);
void bgv_keygen(bgvkey_t& pk, params::poly_q& sk);
void bgv_encrypt(bgvenc_t &c, bgvkey_t& pk, params::poly_p& m);
void bgv_decrypt(params::poly_p& m, bgvenc_t& c, params::poly_q & sk);

#endif
