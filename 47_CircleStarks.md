# ETAAcademy-ZKMeme: 47. Circle Starks

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>47. Circle Starks</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Circle Starks</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

Circle STARKs, a novel construction based on the circle curve over a Mersenne prime ($p = 2^{31} - 1$), aim to address limitations in field smoothness and improve efficiency. Circle STARKs leverage the Circle FFT, a Fourier transform based on the structure of the circle curve, allowing polynomial arithmetic without the need for elliptic curve-based setups or algebraic geometry expertise. The Circle FFT’s cyclic group structure and compatibility with polynomial transformations allow circle STARKs to be as efficient as classical STARKs while eliminating specific conversion complexities and dimension gaps. By adapting traditional STARK components to this setup, circle STARKs achieve similar proof efficiency with streamlined arithmetic on arbitrary primes with smooth properties, promising practical advantages in both performance and simplicity.

## 1. Mersenne31

**Mersenne31** refers to the prime field defined by the modulus $2^{31} - 1$. This modulus is very close to the upper limit of 32-bit integers, $2^{32} - 1$ (i.e., 4,294,967,295), making arithmetic operations on 32-bit CPUs and GPUs highly efficient. The special form of this number allows for optimization through bitwise operations, particularly in binary arithmetic. By leveraging bit manipulation and specialized opcodes, both addition and multiplication in the Mersenne31 field can be reduced quickly, offering superior performance compared to other fields, such as BabyBear, especially when handling large integers requiring modular arithmetic.

In addition, addition operations may result in values exceeding the modulus $2^{31} - 1$. However, due to the unique form of $2^{31} - 1$, rapid modular reduction can be achieved through simple bitwise operations. Specifically, one can utilize the operation $x \rightarrow x + (x >> 31)$, where $x >> 31$ shifts $x$ right by 31 bits, effectively dividing it by $2^{31}$. This allows for quick reduction of results exceeding $2^{31} - 1$ back into the valid modular range. Such bit manipulation is highly efficient, avoiding the need for complex multiplication or division.

For standard multiplication, the product of two large integers $x$ and $y$ can exceed the 32-bit limit, necessitating modular reduction of the product. In the context of Mersenne31, calculating $xy \mod (2^{31} - 1)$ can yield very large results, so effective modularization is essential. Modern CPUs often feature high-order opcodes that can directly return the "high order bits" of the multiplication result, effectively providing $\text{floor}( \frac{xy}{2^{32}} )$. This capability allows for rapid handling of large integer multiplication without requiring multiple multiplication and modular reduction operations. Given that the modulus for Mersenne31 is close to $2^{31}$, this operation facilitates quick calculations of multiplication results under the modulus.

On the other hand, the modulus for BabyBear is $15 \cdot 2^{27} + 1$. The size of the multiplicative group of this modulus is always $p - 1$, allowing for the selection of a subgroup of size $2^{27}$ or the entire group. The FRI protocol can continuously reduce polynomials to a degree of 15, followed by a direct degree check at the end, making it suitable for FFT and similar optimization algorithms.

The Circle FFT is applicable when the prime $p$ is a CFFT-friendly prime that supports a domain size of $2^n$, where $n \geq 1$. In this context, $G_n$ is the unique proper subgroup of the circle curve $C(F_p)$ with order $N = 2^n$. Circle FFT is utilized for interpolating functions from twin-cosets $D = Q \cdot G_{n-1} \cup Q^{-1} \cdot G_{n-1}$, based on polynomials from the space $L_N(F)$.

In contrast, Mersenne31 is a prime field with a modulus of $2^{31} - 1$, resulting in a multiplicative group size of $2^{31} - 2$. This means that the size of this group can only be divided by 2 once (allowing for a subgroup of size $2^{30}$), but it cannot be further divided into smaller subgroups of powers of 2. This limitation restricts the application of Fast Fourier Transform (FFT) techniques, which typically require a group size that is a power of 2 for efficient computation. Consequently, the multiplicative group of Mersenne31 does not meet this criterion, which in turn restricts the use of the FRI algorithm, of which FFT is a crucial component.

## 2. Circle FRI

The ingenuity of Circle STARKs lies in their construction over finite fields $F_p$ where $p \equiv 3 \mod 4$. Here, we define the circle curve $C$ with the equation $x^2 + y^2 = 1$. This curve has two points at infinity in the extended field $F_p(i)$, specifically $\infty = (1 : i : 0)$ and $\bar{\infty} = (1 : -i : 0)$, which play a crucial role in the circle STARK framework. We establish an isomorphism between the circle curve and the projective line $P^1(F_p)$, allowing us to simplify bivariate function fields to univariate ones, thereby facilitating computation. This isomorphism is defined through a mapping known as stereographic projection, which projects points on the circle curve onto the y-axis with the point $(-1, 0)$ as the center of projection. The mapping can be expressed as follows:

$$
t = \frac{y}{x + 1}, \quad (x, y) = \left( \frac{1 - t^2}{1 + t^2}, \frac{2t}{1 + t^2} \right).
$$

Under this mapping, the point \((-1, 0)\) corresponds to a point at infinity on the projective line, while the points at infinity $\infty = (1 : +i : 0)$ and $\bar{\infty} = (1 : -i : 0)$ correspond to $t = \pm i$. This isomorphism, realized through stereographic projection, links points on the circle with those on the y-axis, forming the algebraic foundation for circle STARKs.

In the context of a prime $p \equiv 3 \mod 4$, we can define a circle group using the circle curve $C(F_p)$. The set of points $C(F_p)$ forms a cyclic group under a specific operation defined as:

$$
(x_0, y_0) \cdot (x_1, y_1) := (x_0 \cdot x_1 - y_0 \cdot y_1, x_0 \cdot y_1 + y_0 \cdot x_1).
$$

The identity element of this group is $(1, 0)$, and the inverse of any element $(x, y)$ is given by $J(x, y) = (x, -y)$, indicating a symmetric operation within the group. We also define an isomorphic mapping $(x, y) \mapsto x + i \cdot y$, which maps $C(F_p)$ into a multiplicative subgroup of the extended field $F_p(i)$. The resulting cyclic group structure enables the definition of what are referred to as "standard position cosets" and "twin cosets," corresponding to the subgroup $G_n$ and its associated twin cosets, respectively. These cosets provide an evaluation domain for implementing Circle FFT, particularly benefiting from the ability of twin cosets to decompose into smaller cosets, facilitating rotation-invariant FFT processing for non-smooth structures. Lemmas also indicate that for primes $p$ that support FFT, the standard position coset can be further decomposed into smaller twin cosets, simplifying calculations to accommodate the varying dimensional requirements of FFT algorithms.

The doubling form is as follows: given a prime $p$, we can construct a point set $G$ where $G = \{1, 2, \ldots, p-1\}$. This leads to a group of size $p + 1$ that exhibits a similar two-to-one property, represented by the set of points $(x, y)$ satisfying $x^2 + y^2 = 1$. These point sets obey an addition law akin to those found in trigonometry or complex multiplication. Within this point set $G$, there exists a special "doubling" mapping that takes a point $(x, y)$ and maps it to $(2x, 2y)$, which is known as the "doubling form." On the unit circle, a point $(x, y)$ can be expressed as $x = \cos(\theta)$ and $y = \sin(\theta)$. The purpose of this doubling form is to map one point to another, geometrically equivalent to doubling the angle.

According to the double angle formulas for trigonometric functions:

- $\cos(2\theta) = 2\cos^2(\theta) - 1$
- $\sin(2\theta) = 2\sin(\theta)\cos(\theta)$

If we replace \((x, y)\) with \((\cos(\theta), \sin(\theta))\), the results of the doubling mapping can be expressed as:

- $x' = \cos(2\theta) = 2\cos^2(\theta) - 1 = 2x^2 - 1$
- $y' = \sin(2\theta) = 2\sin(\theta)\cos(\theta) = 2xy$

Focusing on points on the circle that occupy "odd" positions, this forms a structure akin to a Fast Fourier Transform (FFT). This doubling mapping can be applied to FFT computations by merging point sets into a one-dimensional array, then taking random linear combinations, and continuously reducing the problem size through the doubling mapping to achieve efficient calculations.

Now, this represents our FFT. First, we consolidate all points aligned vertically. The formulas equivalent to those used in standard FRI, $B(x)^2$ and $C(x)^2$, yield:

$$
f_0(x) = \frac{F(x,y) + F(x,-y)}{2}
$$

$$
f_1(x) = \frac{F(x,y) + F(x,-y)}{2y}.
$$

Next, we can take a random linear combination, resulting in a one-dimensional function $F$ located on a subset of the x-axis:

From the second round onward, the mapping alters:

$$
f_0(2x^2 - 1) = \frac{F(x) + F(-x)}{2}
$$

$$
f_1(2x^2 - 1) = \frac{F(x) - F(-x)}{2x}.
$$

Through this process, Circle FRI elegantly unites geometry, algebra, and number theory, offering powerful tools for efficient computation in cryptographic settings.

<details><summary><b> Code</b></summary>

<details><summary><b> Python </b></summary>

```python

# This is the folding step in FRI, where you combine the evaluations at two
# sets of N coordinates each into evaluations at one set of N coordinates.
# We do three rounds of folding at a time, so each FRI step drops the degree
# by 8x
def fold(values, coeff, first_round):
    for i in range(FOLDS_PER_ROUND):
        full_len, half_len = values.shape[-1], values.shape[-1]//2
        left, right = values[::2], values[1::2]
        f0 = (left + right) * HALF
        if i == 0 and first_round:
            twiddle = (
                invy[full_len: full_len * 2]
                [folded_rbos[full_len:full_len*2:2]]
            )
        else:
            twiddle = (
                invx[full_len*2: full_len * 3]
                [folded_rbos[full_len:full_len*2:2]]
            )
        twiddle_box = zeros_like(left)
        twiddle_box[:] = twiddle.reshape((half_len,) + (1,) * (left.ndim-1))
        f1 = (left - right) * HALF * twiddle_box
        values = f0 + f1 * coeff
    return values

```

</details>

<details><summary><b> C++ </b></summary>

```C++


hashDigest_t addSubproof(
    state_t<unique_ptr<dataWithCommitment>>& currState,
    const vector<FieldElement>& evaluationBasis,
    const subproofLocation_t& pathToProof,
    const subproofLocation_t& pathFromRoot,
    const bool L0isMSB
    ){

    //
    // Base case of recursion
    //
    if(pathToProof.size() == 0){
        return currState.localState->getCommitment();
    }

    //
    // Updating paths
    //
    const auto& currWay = pathToProof[0];
    const subproofLocation_t nextPathToProof(pathToProof.begin()+1, pathToProof.end());
    subproofLocation_t nextPathFromRoot(1,currWay);
    nextPathFromRoot.insert(nextPathFromRoot.end(),pathFromRoot.begin(),pathFromRoot.end());

    //
    // Basis of next evaluation
    //
    const vector<FieldElement> basisForColumnsProof(getColumnBasis(evaluationBasis,L0isMSB));

    //
    // Check if current univariate already evaluated, and evaluate if needed
    //
    if(currState.subproofs.count(currWay) == 0){
        const vector<FieldElement> BasisL0 = getL0Basis(evaluationBasis,L0isMSB);
        const vector<FieldElement> BasisL1 = getL1Basis(evaluationBasis,L0isMSB);

        const unsigned short logSigmentLen = getL0Basis(basisForColumnsProof,L0isMSB).size();
        const unsigned short logNumSigments = getL1Basis(basisForColumnsProof,L0isMSB).size();
        const unsigned short logSigmentsInBlock = std::min((unsigned short)10,logNumSigments);

        const size_t sigmentLen = POW2(logSigmentLen);
        const size_t sigmentsInBlock = POW2(logSigmentsInBlock);

        ///
        /// The following is a trick for faster evaluation
        ///
        /// We have : values of a polynomial over a space L_0
        /// We want : the polynomials value over another point x not in L_0
        ///
        /// the Lagrange polynomial for \alpha \in L_0 is:
        ///
        /// l_\alpha (x) =
        /// \frac{ \prod_{\beta \ne \alpha \in L_0} (x - \beta) }{ \prod_{\beta \ne \alpha \in L_0} (\alpha - \beta) } =
        /// \frac{ Z_{L_0}(x) }{ (x-\alpha) \cdot \prod_{\beta \ne \alpha \in L_0} (\alpha - \beta) }
        ///
        /// We Define:
        ///
        /// C_\alpha := \prod_{\beta \ne \alpha \in L_0} (\alpha - \beta)
        ///
        /// Thus, given values p(\alpha) for any \alpha in L_0, the value over $x$ is:
        ///
        /// p(x) =
        /// \sum_{\alpha \in L_0} p(\alpha) \cdot l_\alpha (x) =
        /// Z_{L_0} (x) \cdot \sum_{\alpha \in L_0} \frac{ p(\alpha) }{(x-\alpha) \cdot C_\alpha}
        ///
        /// In this formula many factors are independent of $x$ and can be precomputed, and this is what used bellow
        ///

        // global auxiliary values
        const size_t L0_size = POW2(BasisL0.size());
        vector<FieldElement> spaceElements(L0_size);
        for(unsigned int i=0; i<L0_size; i++){
            spaceElements[i] = getSpaceElementByIndex(BasisL0,Algebra::zero(),i);
        }


        //compute Z_{L_0}
        const Algebra::SubspacePolynomial Z_L0(Algebra::elementsSet_t(BasisL0.begin(),BasisL0.end()));

        //compute C_\alpha vector
        vector<FieldElement> C_alpha(L0_size,Algebra::one());
        {

            for(unsigned int i=0; i<L0_size; i++){
                const FieldElement& alpha = spaceElements[i];
                for(unsigned int j=0; j<L0_size; j++){
                    if(i==j)continue;
                    const FieldElement& beta = spaceElements[j];
                    C_alpha[i] *= (alpha - beta);
                }
            }
        }


        const auto sigmentConstructor = [&](const size_t sigmentsBlockIdx, FieldElement* res){

            vector<FieldElement> vecToInveresePointwise(sigmentsInBlock*sigmentLen*L0_size);

            for(unsigned int localSigmentIdx = 0; localSigmentIdx < sigmentsInBlock; localSigmentIdx++){
                const size_t sigmentIdx = sigmentsBlockIdx*sigmentsInBlock + localSigmentIdx;

                for(unsigned int i=0; i< sigmentLen; i++){
                    const size_t globalIndex = sigmentIdx * sigmentLen + i;
                    const FieldElement currOffset = getSpaceElementByIndex(BasisL1,zero(),globalIndex);

                    for(size_t j=0; j< L0_size; j++){
                        const FieldElement alpha = spaceElements[j];
                        const size_t elementIndex = localSigmentIdx*sigmentLen*L0_size + i*L0_size + j;
                        vecToInveresePointwise[elementIndex] = ((currWay+currOffset)-alpha)*C_alpha[j];
                    }
                }
            }

            const vector<FieldElement> denuminators = Algebra::invertPointwise(vecToInveresePointwise);

            for(unsigned int localSigmentIdx = 0; localSigmentIdx < sigmentsInBlock; localSigmentIdx++){
                const size_t sigmentIdx = sigmentsBlockIdx*sigmentsInBlock + localSigmentIdx;
                FieldElement* currSigRes = res + localSigmentIdx*sigmentLen;

                for(unsigned int i=0; i< sigmentLen; i++){
                    const size_t globalIndex = sigmentIdx * sigmentLen + i;
                    const FieldElement currOffset = getSpaceElementByIndex(BasisL1,zero(),globalIndex);

                    currSigRes[i] = Algebra::zero();
                    for(size_t j=0; j< L0_size; j++){
                        const size_t currElemIdx = getBasisLIndex_byL0L1indices(evaluationBasis,j,globalIndex,L0isMSB);
                        const FieldElement alpha = spaceElements[j];
                        const FieldElement currVal = currState.localState->getElement(currElemIdx);

                        const size_t elementIndex = localSigmentIdx*sigmentLen*L0_size + i*L0_size + j;
                        currSigRes[i] += currVal * denuminators[elementIndex];
                    }
                    currSigRes[i] *= Z_L0.eval(currWay+currOffset);
                }
            }
        };

        currState.subproofs[currWay].localState =
            unique_ptr<dataWithCommitment>(
                new dataWithCommitment(
                        logSigmentLen + logSigmentsInBlock,
                        logNumSigments - logSigmentsInBlock,
                        sigmentConstructor
                    )
                );
    }


    //
    // Continue recursively
    //
    return addSubproof(currState.subproofs[currWay], basisForColumnsProof, nextPathToProof, nextPathFromRoot,L0isMSB);
    }

```

</details>

</details>

## 3. Circle FFTs

Circle FFT is an algorithm closely related to the Fast Reed-Solomon Interactive Oracle Proof of Proximity (FRI) and is specifically designed for polynomial operations in Riemann-Roch spaces.

### 3.1 Relationship Between FFT and FRI

To start, the ordinary Fast Fourier Transform (FFT) is an algorithm that converts polynomials from point-value representation to coefficient representation. Given the point values of $n$ polynomials, the FFT can transform them into their corresponding coefficients. The principles behind FRI and FFT are quite similar, with the main distinction being that FRI generates a random linear combination at each step, while FFT recursively performs half-size FFTs on the polynomials $f_0$ and $f_1$, ultimately using $FFT(f_0)$ for the even terms and $FFT(f_1)$ for the odd terms.

The Circle FFT algorithm recursively decomposes the problem into smaller subproblems. Specifically, it breaks down the problem into "even" and "odd" components and progressively reduces the subdomains being processed to efficiently compute the Fourier coefficients.

The double coset $D$ of subgroup $G_{n-1}$, which has size $|D| = 2^n$, is transformed via the mapping chain:

$$
D = D*n \xrightarrow{\phi_J} S_n \xrightarrow{\pi} S_{n-1} \xrightarrow{\pi} \ldots \xrightarrow{\pi} S_1
$$

Here, $S_j = D_j / J$ is the subset obtained through the quotient mapping and is viewed as a collection of points on the x-axis. This recursive step reduces the number of points at each level by half until $S_1$ contains only one point.

**Initial Decomposition (Even and Odd Terms)**

In the initial step, the function $f$ is decomposed into "even" and "odd" components concerning the involution $J$. Using the reference function $t_0(x, y) = y$ (which satisfies $t_0 \circ J = -t_0$, making it an odd function under the $J$ transformation), we can express $f \in F_{D_n}$ as two functions $f_0$ and $f_1$ defined on $S_n$:

$$
f_0(x) = \frac{f(x, y) + f(x, -y)}{2}, \quad
f_1(x) = \frac{f(x, y) - f(x, -y)}{2} \cdot y, \quad
$$

This satisfies the relationship:

$$
f(x, y) = f_0(x) + y \cdot f_1(x).
$$

This decomposition allows $f$ to be represented as a combination of $f_0$ and $f_1$, which can then be processed independently.

**Subsequent Steps (Smaller Subproblems)**

In the following steps, the algorithm recursively handles the functions obtained from the previous step $f_{k_0, \ldots, k_{n-j}}$, where $2 \leq j \leq n$. By selecting the reference odd function $t_1(x, y) = x$, we can further decompose the function:

$$
f_0(\pi(x)) = \frac{f(x) + f(-x)}{2}, \quad
f_1(\pi(x)) = \frac{f(x) - f(-x)}{2} \cdot x.
$$

Thus, we have:

$$
f(x) = f_0(\pi(x)) + x \cdot f_1(\pi(x)),
$$

where $\pi(x) = 2 \cdot x^2 - 1$ is the form of the mapping. This decomposition process enables the algorithm to derive functions $f_0$ and $f_1$ on new, smaller subsets, continuing until a constant function is defined at the single point $S_1$.

**Output of Circle FFT**

The final output of the algorithm is a set of coefficients $c_k$, corresponding to the Fourier basis $b_k$, satisfying:

$$
\sum_{k=0}^{2^n-1} c_k \cdot b_k = f.
$$

This indicates that the algorithm generates a set of coefficients that can represent the original function $f$ as a combination of these coefficients and basis functions.

**Simplification in Practical Applications**

In practical implementations, the factor of 2 in the decomposition of even and odd terms is often omitted, yielding an FFT result scaled against the basis. This simplification reduces the number of multiplication and addition operations, thereby enhancing the algorithm's efficiency.

### 3.2 Distinct Features of Circle FFT

Circle FFT is similar to traditional FFT but operates on what are known as **Riemann-Roch spaces**. This mathematical concept means that the polynomials processed in Circle FFT are those modulo the unit circle defined by the equation $x^2 + y^2 - 1 = 0$. Consequently, whenever a $y^2$ term appears, it is replaced with $1 - x^2$.

In Circle FFT, the coefficients of the polynomials are not typical monomials (e.g., $x, x^2, x^3$). Instead, they form a specific basis set, which may include:

$$
\{1, y, x, xy, 2x^2 - 1, 2x^2y - y, 2x^3 - x, 2x^3y - xy, 8x^4 - 8x^2 + 1, \ldots\}
$$

Initially, the polynomial space $L_N(F)$ contains all double-variable polynomials with coefficients in $F$ and a total degree not exceeding $N/2$. This space is defined as:

$$
L_N(F) = \{ p(x, y) \in F[x, y] / (x^2 + y^2 - 1) : \deg p \leq \frac{N}{2} \}
$$

Here, the ideal $(x^2 + y^2 - 1)$ is generated by the equation of the unit circle.

From an algebraic geometry perspective, $L_N(F)$ corresponds to the $F_p$-rational divisor $\frac{3N}{2} \cdot \infty + \frac{N}{2} \cdot \bar{\infty}$ of the Riemann-Roch space. This implies that the space includes all rational functions with poles located only at $\infty$ and $\bar{\infty}$, with the order of poles not exceeding $N/2$. Unlike elliptic curves, circle curves can utilize polynomial space as a substitute for rational function space, which is invariant under rotations—crucial for efficient coding and constructing distance-separating codes.

**Proposition 2: Twin Cosets and Standard Position Cosets** highlights two important properties of $L_N(F)$:

1. **Rotation Invariance**: For any $f \in L_N(F)$ and $P \in C(F_p)$, $f \circ T_P \in L_N(F)$.
2. **Dimension**: $L_N(F) = N + 1$, with any non-trivial $f \in L_N(F)$ having no more than $N$ roots on $C(F)$.

Through definitions and proofs, a basis for $L_N(F)$ can be represented as the following monomial set:

$$
1, x, \ldots, x^{N/2}, y, y \cdot x, \ldots, y \cdot x^{N/2 - 1}.
$$

Subsequently, **Circle Codes** are a type of algebraic geometry code, akin to generalized Reed-Solomon codes, generated from the linear codes of polynomial space $L_N(F)$. Specifically, for any subset $D$ on the circle curve $C(F_p)$ (like standard position cosets or twin cosets), the codewords of Circle Codes are defined by the evaluations of this polynomial space at $D$:

$$
C_N(F, D) = \{ f(P) : P \in D, f \in L_N(F) \}.
$$

### 3.3 Low-Degree Extensions and Circle FFT

The primary application of Circle FFT is to perform low-degree extensions, which means generating $k \times N$ point values from $N$ point values. The process begins with FFT to produce coefficients, followed by appending $(k-1) \times N$ zeros to these coefficients, and finally applying inverse FFT to obtain a larger set of point values.

### 3.4 Comparison with Other Heterogeneous FFTs

The definition introduces a special basis known as $B_n$, which depends on the size of the field and represents the transformation from functions in $F_D$ to coefficients in $B_n$. The polynomials of the FFT basis are constructed from the vanishing polynomials of a set of standard position cosets and possess linear independence. By leveraging Circle FFT, one can efficiently compute the FFT transformation coefficients of a given function over the twin cosets, with a computational complexity of $N \cdot n$ additions and $N \cdot n^2$ multiplications. The inverse transformation process ( i.e., recovering function values from coefficients in $B_n$) carries the same computational complexity.

Circle FFT is not the only type of "heterogeneous" FFT. **Elliptic Curve FFT (ECFFT)** is another, more complex, and powerful variant capable of functioning over arbitrary finite fields (such as prime fields and binary fields). However, the implementation of ECFFT is more intricate and less efficient. Therefore, in specific scenarios (like when using $p = 2^{31} - 1$), Circle FFT serves as a suitable substitute for ECFFT.

Similar to ECFFT, Circle FFT is a non-harmonic FFT. The group structure of the circle curve plays a central role in its construction, relying on two crucial self-maps: the group squaring map $\pi$ and the inversion map $J$. These mappings construct the space of low-degree interpolation functions, approximating the dimensionality to match the size of the field. Unlike the perfect match in elliptic curves, Circle FFT experiences a dimensional discrepancy between the result space and polynomial space, resulting in a transformation outcome space dimension $N$ that is one less than the original space.

<details><summary><b> Code</b></summary>

<details><summary><b> Python </b></summary>

```python
# Converts a list of evaluations to a list of coefficients. Note that the
# coefficients are in a "weird" basis: 1, y, x, xy, 2x^2-1...
def fft(vals, is_top_level=True):
    vals = vals.copy()
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    for i in range(log2(size)):
        vals = vals.reshape((1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[1]
        half_len = full_len >> 1
        L = vals[:, :half_len]
        R = vals[:, half_len:][:, ::-1, ...] # flip along axis 1
        f0 = L + R
        if i==0 and is_top_level:
            twiddle = invy[full_len: full_len + half_len]
        else:
            twiddle = invx[full_len*2: full_len*2 + half_len]
        twiddle_box = twiddle.reshape((1, half_len) + (1,) * (L.ndim - 2))
        f1 = (L - R) * twiddle_box
        vals[:, :half_len] = f0
        vals[:, half_len:] = f1
    return (
        (vals.reshape((size,) + shape_suffix))[rbos[size:size*2]] / size
    )

```

</details>

<details><summary><b> C++ </b></summary>

```c++
#include "algebraLib/novelFFT.hpp"
#include "algebraLib/ErrorHandling.hpp"
#include <FFT.h>
#include <string.h>

namespace Algebra{

using std::vector;

namespace{
    vector<SubspacePolynomial> calc_X_exp(const vector<FieldElement>& orderedBasis){
        vector<SubspacePolynomial> X_exp;
        {
            for(unsigned int i=0; i<orderedBasis.size(); i++){
                const elementsSet_t currBasis(orderedBasis.begin(),orderedBasis.begin()+i);
                X_exp.push_back(SubspacePolynomial(currBasis));
                SubspacePolynomial& currPoly = X_exp[X_exp.size()-1];
                const FieldElement currElem = orderedBasis[i];
                const FieldElement factor = one()/currPoly.eval(currElem);
                currPoly.multiplyByConstant(factor);
            }
        }
...


                for(size_t polyIdx = 0; polyIdx < numPolys; polyIdx++){
                    FieldElement* currPoly = &polysVals[polyIdx];
                    //handle case (18)
                    {
                        const size_t mc_case18 = m | ((c|1UL)<<currShift_c);
                        const size_t innerIdx = mc_case18 ^ oddMask_c;

                        currPoly[width*mc_case18] = currPoly[width*innerIdx] + currPoly[width*mc_case18];
                    }

                    //handle case (17)
                    {
                        const size_t mc_case17 = m | (c<<currShift_c);
                        const size_t prevIdx2 = mc_case17 ^ oddMask_c;

                        currPoly[width*mc_case17] = currPoly[width*mc_case17] + X_exp_precomp[c>>1] * currPoly[width*prevIdx2];
                    }
                }
            }
        }

        return polysVals;
    }

    vector<FieldElement> convertPolysBasis(const vector<FieldElement> orderedBasis, const vector<SubspacePolynomial> X_exp, vector<vector<FieldElement>>&& polysCoeffs, const size_t width, const FieldElement& pad_value){

    ALGEBRALIB_ASSERT(width >= polysCoeffs.size(), "Width must be at least as the number of polys");
    const size_t numPolys = polysCoeffs.size();
    const size_t spaceSize = 1UL<<orderedBasis.size();

    //evaluate using standard FFT on the space
    vector<FieldElement> res(width * spaceSize, pad_value);
    {
        FFF::Element* basis_vec = (FFF::Element*)(&(orderedBasis[0]));
        FFF::Element shift_fff(zero());
        FFF::Basis basis(basis_vec, orderedBasis.size(),shift_fff);
        FFF::FFT fftInstance(basis,FFF::FFT_OP);

#pragma omp parallel for
        for(unsigned int i=0; i< numPolys; i++){

            vector<FieldElement>& currCoeffs = polysCoeffs[i];

            ALGEBRALIB_ASSERT(currCoeffs.size() <= spaceSize, "FFT is supported only for evaluation spaces of size at least as the polynomial degree");

            if(currCoeffs.size() < spaceSize){
                currCoeffs.resize(spaceSize,zero());
            }

            auto c_poly = (FFF::Element*)(&currCoeffs[0]);
            fftInstance.AlgFFT(&c_poly,spaceSize);
...

novelFFT::novelFFT(const vector<FieldElement>& orderedBasis, vector<FieldElement>&& srcEval) :
    novelFFT(orderedBasis, calc_X_exp(orderedBasis), std::move(srcEval)){};

novelFFT::novelFFT(const vector<FieldElement>& orderedBasis, vector<vector<FieldElement>>&& polysCoeffs, const size_t width, const FieldElement& pad_value) :
    novelFFT(orderedBasis, calc_X_exp(orderedBasis), std::move(polysCoeffs), width, pad_value){};

void novelFFT::FFT(const vector<FieldElement>& affineShift, FieldElement* dst, const size_t diff_coset)const{

    const unsigned short basisSize = orderedBasis_.size();
    const size_t spaceSize = 1UL<<basisSize;

    //copy coefficient to destination
    {
        const unsigned int max_threads_machine = omp_get_max_threads();
        const size_t bufferBytesLen = polys_.size() * sizeof(FieldElement);
        const size_t blockBytesLen = bufferBytesLen / max_threads_machine;
        const size_t blockRemeinder = bufferBytesLen % max_threads_machine;
#pragma omp parallel for
        for(long long blockIdx = 0; blockIdx < max_threads_machine; blockIdx++){
            for(unsigned int cosetIdx = 0; cosetIdx < affineShift.size(); cosetIdx++){
                memcpy((char*)(dst + (cosetIdx*diff_coset)) + (blockIdx*blockBytesLen), ((char*)&polys_[0]) + (blockIdx*blockBytesLen), blockBytesLen);
            }
        }

        if(blockRemeinder > 0){
            for(unsigned int cosetIdx = 0; cosetIdx < affineShift.size(); cosetIdx++){
                memcpy((char*)(dst + (cosetIdx*diff_coset)) + (bufferBytesLen - blockRemeinder), (char*)(&polys_[0]) + (bufferBytesLen - blockRemeinder), blockRemeinder);
            }
        }

    }

    //execute the FFT
    {
        for (int i=basisSize-1; i >= 0; i--){
            const unsigned short currShift_c = i;
            const size_t currMask_m = (1UL<<currShift_c)-1;
            const size_t oddMask_c = 1UL<<currShift_c;
            const unsigned short len_c = basisSize-currShift_c;
            const unsigned short logNumPrecompute = len_c-1;
            const size_t numPrecompute = 1UL<<(logNumPrecompute);
            const size_t numPrecomputeMask = numPrecompute-1;
            const size_t numShifts = affineShift.size();

            vector<vector<FieldElement>> X_exp_precomp(affineShift.size(),vector<FieldElement>(numPrecompute));
#pragma omp parallel for
            for(unsigned long long expIdx=0; expIdx < (numShifts<<logNumPrecompute); expIdx++){
                const size_t c = expIdx & numPrecomputeMask;
                const size_t cosetId = expIdx >> logNumPrecompute;
                const FieldElement c_elem = getSpaceElementByIndex(orderedBasis_,affineShift[cosetId],c<<(currShift_c+1));
                X_exp_precomp[cosetId][c] = X_exp_[i].eval(c_elem);
            }
...

                    const size_t mc_case17index = width_*mc_case17;
                    const size_t mc_case18index = width_*mc_case18;
                    const long long cs1 = c>>1;
                    for(size_t cosetId=0; cosetId < affineShift.size(); cosetId++){

                        const FieldElement Xpre = X_exp_precomp[cosetId][cs1];
                        const FFF::Element XpreElem = *(FFF::Element*)(&Xpre);

                        const size_t cosetOffset = diff_coset*cosetId;
                        FieldElement* baseVec = dst + cosetOffset;
                        FFF::Element* currVecFFF = (FFF::Element*)(baseVec);

                        //
                        // Irreducible specific implementation
                        //
                        FFF::Element::do_FFT_step(XpreElem, &currVecFFF[mc_case18index], &currVecFFF[mc_case17index],numPolys_);


                        /*
                         * General code - less field specific optimizations
                         */

                        //const size_t ub = polys_.size() * diff_poly_fixed;
                        //for(size_t polyOffset=0; polyOffset < ub; polyOffset+=(diff_poly_fixed*2)){
                            //FieldElement* currVec = baseVec + polyOffset;

                            //handle case (17)
                            //currVec[mc_case17index] = currVec[mc_case17index] + Xpre * currVec[mc_case18index];

                            //handle case (18)
                            //currVec[mc_case18index] = currVec[mc_case17index] + currVec[mc_case18index];
                        //}
                    }
                }
            }
        }
    }
}

} // namespace Algebra

```

</details>

</details>

## 4.Circle STARKs

### 4.1 Quotienting

In STARK protocols, the operation of quotienting plays a crucial role in demonstrating the value of a polynomial at specific points. To prove that a polynomial $P(x)$ equals $y$ at a point $x_0$, we introduce a new polynomial $Q(x) = \frac{P(x) - y}{x - x_0}$ and verify the validity of $Q(x)$ to indirectly establish that $P(x_0) = y$. In Circle STARKs, due to the geometric constraints of the circular group, traditional linear functions cannot pass through a single point. Therefore, a two-point proving strategy is employed. By introducing an interpolation function $I(x)$, we construct a polynomial that takes specific values at two points, which we then divide by a linear function $L(x)$ to complete the proof. This ensures both the computational efficiency and security of Circle STARKs while overcoming the limitations imposed by the geometric structure.

For instance, if we have a polynomial $P$ such that it equals $v_1$ at $P_1$ and $v_2$ at $P_2$, we select an interpolation function $I$, which can be expressed as a linear function equal to $v_1$ at $P_1$ and $v_2$ at $P_2$. This can be simplified as $I(x) = v_1 + (v_2 - v_1) \cdot \frac{y - y_1}{y_2 - y_1}$. Next, we subtract $I(x)$ from $P(x)$ to ensure that $P(x) - I(x)$ equals zero at both points, then divide by the linear function $L(x)$ corresponding to $P_1$ and $P_2$ and prove that the quotient $\frac{P(x) - I(x)}{L(x)}$ is a polynomial. This ultimately demonstrates that $P$ equals $v_1$ at $P_1$ and $v_2$ at $P_2$.

In essence, every quotient $q$ can be uniquely decomposed into a set of smaller polynomials $q_k$ and an additional scalar $\lambda$. This decomposition is crucial as it simplifies polynomial computations, aiding us in finding suitable polynomials over a specific subset $\bar{H}$. The decomposition can be formulated as:

$$
q = \lambda \cdot v_{\bar{H}} + \sum_{k=1}^{d-1} \frac{v_{\bar{H}}}{v_{H_k}} \cdot q_k
$$

where $v_{\bar{H}}$ is the vanishing polynomial for the entire union, and $v_{H_k}$ is the vanishing polynomial for each dual co-domain $H_k$. This structure ensures that points on $H_k$ can be represented by $q_k$, while also maintaining control over the overall vanishing properties.

### 4.2 Vanishing Polynomials

Given $D$ as a subset of the circle curve $C(F_p)$ with even size $N$ such that $2 \leq N < p + 1$, we define a non-zero polynomial that vanishes on the set $D$ as the **vanishing polynomial** for $D$. Specifically, for a set of point pairs $\{P_k, Q_k\}$ in $D$, we can construct a vanishing polynomial expressed as the product of linear functions through these point pairs:

$$
v_D(x, y) = \prod_{k=1}^{N/2} \left( (x - P_{k,x})(Q_{k,y} - P_{k,y}) - (y - P_{k,y})(Q_{k,x} - P_{k,x}) \right).
$$

This construction guarantees the existence of a vanishing polynomial and proves its uniqueness under degree constraints in the polynomial space $L_N(F_p)$.

To simplify and optimize the computation of vanishing polynomials, we focus on those defined in the FFT domain, particularly the standard position co-sets and dual co-sets. In this context, the construction of the vanishing polynomial can be executed quickly with low computational complexity concerning field operations.

In STARK proofs, the polynomial equation to be verified is akin to $C(P(x), P(next(x))) = Z(x) \cdot H(x)$, where the vanishing polynomial $Z(x)$ is utilized to confirm that the polynomial evaluates to zero at specific points (typically at all points in the evaluation domain). In conventional STARK protocols, the roots of unity satisfying $x^n = 1$ are represented by the vanishing polynomial $x^n - 1$, ensuring it takes values of zero at the domain composed of these roots. In Circle STARKs, the construction of the vanishing polynomial is based on a different recursive relationship, starting with $Z_1(x, y) = y$ and $Z_2(x, y) = x$, and subsequently generated through the folding function $x \rightarrow 2x^2 - 1$ in a recursive manner, expressed as:

$$
Z\_{n+1}(x, y) = (2 \cdot Z_n(x, y)^2) - 1.
$$

This illustrates that the vanishing polynomial emerges from the folding function. In conventional STARKs, the folding function takes the form $x \rightarrow x^2$, while in Circle STARKs, it is adapted to suit the circular geometry. This structure guarantees that the vanishing polynomial has roots throughout the entire circular group, thus completing a vanishing operation akin to that in conventional STARKs, yet tailored to the unique properties of the geometric structure.

Moreover, the introduction of the quotient polynomial $q$ allows any polynomial $f$ that vanishes on $D$ to be expressed as $f = q \cdot v$, where $q$ is another polynomial derived from the lower-degree space $L_{M-N}(F)$.

These vanishing and quotient polynomials are integral to the STARK architecture, particularly in the algebraic linking steps of the proof structure (DEEP), ensuring the correctness and symmetry of polynomial verifications.

### 4.3 Bit-Reversed Order

In STARK protocols, evaluating polynomials is conducted in a "bit-reversed order" rather than a natural order (e.g., $P(1)$, $P(\omega)$, $P(\omega^2)$, etc.). This approach optimizes the grouping of adjacent values during the FRI evaluation process. Such ordering ensures that neighboring values can be effectively grouped together for evaluation. For instance, when $n=16$, the bit-reversed order would be $\{0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15\}$. A key property of this order is that adjacent values will be grouped together in the initial steps of FRI evaluation. For example, with $n=16$, since $\omega^8 = -1$, it implies that $P(\omega^i)$ and $P(-\omega^i) = P(\omega^{i+8})$ will be arranged adjacently. As FRI progresses, the grouping of values allows for a Merkle proof to be provided for two values simultaneously, enhancing spatial efficiency.

In Circle STARKs, the folding structure is slightly different. In the initial step, we pair the points $(x, y)$ with $(x, -y)$, reflecting the symmetry of the circle; in the second step, we pair $x$ with $-x$, emphasizing the symmetry of the base points. In subsequent steps, the folding structure extends beyond specific point pairings to introduce broader mapping relations, selecting $p$ and $q$ such that $Q^i(p) = -Q^i(q)$, where $Q^i$ represents the repeated application of the mapping $x \rightarrow 2x^2 - 1$ for $i$ iterations. To ensure that the bit-reversed order reflects this folding structure, we retain the last bit during the reversal of each bit and use it to determine whether to flip other bits. For instance, for a size of 16, the bit-reversed folding order would be $\{0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1\}$. This adjustment guarantees that the arrangement of points on the circle aligns with the required pairing structure, thereby making the FRI evaluation and folding process more efficient. By employing this methodology, we can efficiently manage polynomial evaluations at each step, thereby reducing computational complexity.

<details><summary><b> Code</b></summary>

<details><summary><b> Python</b></summary>

```python
# Generate a STARK proof for a given claim
#
# check_constraint(state, next_state, constraints, is_extended)
#
# Verifies that the constraint is satisfied at two adjacent rows of the trace.
# Must be degree <= H_degree+1
#
# trace: the computation trace
#
# constants: Constants that check_constraint has access to. This typically
#            includes opcodes and other constaints that differ row-by-row
#
# public_args: the rows of the trace that are revealed publicly
#
# prebuilt_constants_tree: The constants only need to be put into a Merkle tree
#                          once, so if you already did it, you can reuse the
#                          tree.
#
# H-degree: this plus one is the max degree of check_constraint. Must be a
# power of two
def mk_stark(check_constraint,
             trace,
             constants,
             public_args,
             prebuilt_constants_tree=None,
             H_degree=2):
    import time
    rounds, constants_width = constants.shape[:2]
    trace_length = trace.shape[0]
    trace_width = trace.shape[1]
    print('Trace length: {}'.format(trace_length))
    START = time.time()
    constants = pad_to(constants, trace_length)

    # Trace must satisfy
    # C(T(x), T(x+G), K(x), A(x)) = Z(x) * H(x)
    # Note that we multiply L[F1, F2] into C, to ensure that C does not
    # have to satisfy at the last coordinate in the set

    # We use the last row of the trace to make its degree N-1 rather than N.
    # This keeps deg(H) later on comfortably less than N*H_degree-1, which
    # makes it more efficient to bary-evaluate
    trace = tweak_last_row(trace)
    # The larger domain on which we run our polynomial math
    ext_degree = H_degree * 2
    ext_domain = sub_domains[
        trace_length*ext_degree:
        trace_length*ext_degree*2
    ]
    trace_ext = inv_fft(pad_to(fft(trace), trace_length*ext_degree))
    print('Generated trace extension', time.time() - START)
    # Decompose the trace into the public part and the private part:
    # trace = public * V + private. We commit to the private part, and show
    # the public part in the clear
    V, I = public_args_to_vanish_and_interp(
        trace_length,
        public_args,
        trace[cp.array(public_args)],
    )
    V_ext = inv_fft(pad_to(fft(V), trace_length*ext_degree))
    I_ext = inv_fft(pad_to(fft(I), trace_length*ext_degree))
    print('Generated V,I', time.time() - START)
    trace_quotient_ext = (
        (trace_ext - I_ext) / V_ext.reshape(V_ext.shape+(1,))
    )
    constants_ext = inv_fft(pad_to(fft(constants), trace_length*ext_degree))
    rolled_trace_ext = M31.append(
        trace_ext[ext_degree:],
        trace_ext[:ext_degree]
    )
    # Zero on the last two columns of the trace. We multiply this into C
    # to make it zero across the entire trace

    # We Merkelize the trace quotient (CPU-dominant) and compute C
    # (GPU-dominant) in parallel
    def f1():
        return merkelize_top_dimension(trace_quotient_ext)

    def f2():
        return compute_H(
            ext_domain,
            trace_ext,
            rolled_trace_ext,
            constants_ext,
            check_constraint,
            trace_length,
        )

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the tasks to the executor
        future1 = executor.submit(f1)
        future2 = executor.submit(f2)

        # Get the results
        TQ_tree = future1.result()
        H_ext = future2.result()
    #TQ_tree = f1()
    #H_ext = f2()

    print('Generated tree and C_ext!', time.time() - START)

    #H_coeffs = fft(H_ext)
    #print(cp.where(H_coeffs[trace_length * H_degree:].value % modulus != 0), H_ext.shape)
    #assert confirm_max_degree(H_coeffs, trace_length * H_degree)

    output_width = H_ext.shape[1]
    if prebuilt_constants_tree is not None:
        K_tree = prebuilt_constants_tree
    else:
        K_tree = merkelize_top_dimension(constants_ext)

    # Now, we generate a random point w, at which we evaluate our polynomials
    G = sub_domains[trace_length//2]
    w = projective_to_point(
        ExtendedM31(get_challenges(TQ_tree[1], modulus, 4))
    )
    w_plus_G = w + G

    # Trace quotient, at w and w+G
    TQ_bary = (bary_eval(trace, w) - bary_eval(I, w)) / bary_eval(V, w)
    TQ_bary2 = (
        (bary_eval(trace, w_plus_G) - bary_eval(I, w_plus_G))
        / bary_eval(V, w_plus_G)
    )
    # Constants, at w and w+G
    K_bary = bary_eval(constants, w)
    K_bary2 = bary_eval(constants, w_plus_G)

    # H, at w and w+G. We _could_ also compute it with compute_H, but
    # somehow a bary-evaluation is faster (!!) than calling the function
    bump = sub_domains[trace_length*ext_degree].to_extended()
    w_bump = w + bump
    wpG_bump = w_plus_G + bump
    H_ef = H_ext[::2]
    H_bary = bary_eval(H_ef, w_bump)
    H_bary2 = bary_eval(H_ef, wpG_bump)
    stack_ext = M31.append(
        trace_quotient_ext,
        constants_ext,
        H_ext,
        axis=1
    )
    stack_width = trace_width + constants_width + output_width
    S_at_w = ExtendedM31.append(TQ_bary, K_bary, H_bary)
    S_at_w_plus_G = ExtendedM31.append(TQ_bary2, K_bary2, H_bary2)
    # Compute a random linear combination of everything in stack_ext4, using
    # S_at_w and S_at_w_plus_G as entropy
    print("Computed evals at w and w+G", time.time() - START)
    entropy = TQ_tree[1] + K_tree[1] + S_at_w.tobytes() + S_at_w_plus_G.tobytes()
    fold_factors = ExtendedM31(
        get_challenges(entropy, modulus, stack_width * 4)
        .reshape((stack_width, 4))
    )
    #assert eq(
    #    bary_eval(stack_ext, w, True, True),
    #    S_at_w
    #)
    merged_poly = fold(stack_ext, fold_factors)
    #merged_poly_coeffs = fft(merged_poly)
    #assert confirm_max_degree(merged_poly_coeffs, trace_length * H_degree)
    print('Generated merged poly!', time.time() - START)
    # Do the quotient trick, to prove that the evaluation we gave is
    # correct. Namely, prove: (random_linear_combination(S) - I) / L is a
    # polynomial, where L is 0 at w w+G, and I is S_at_w and S_at_w_plus_G
    L3 = line_function(w, w_plus_G, ext_domain)
    I3 = interpolant(
        w,
        fold(S_at_w, fold_factors),
        w_plus_G,
        fold(S_at_w_plus_G, fold_factors),
        ext_domain
    )
    #assert eq(
    #    fold_ext(S_at_w, fold_factors),
    #    bary_eval(merged_poly, w, True)
    #)
    master_quotient = (merged_poly - I3) / L3
    print('Generated master_quotient!', time.time() - START)
    #master_quotient_coeffs = fft(master_quotient)
    #assert confirm_max_degree(master_quotient_coeffs, trace_length * H_degree)

    # Generate a FRI proof of (random_linear_combination(S) - I) / L
    fri_proof = prove_low_degree(master_quotient, extra_entropy=entropy)
    fri_entropy = (
        entropy +
        b''.join(fri_proof["roots"]) +
        fri_proof["final_values"].tobytes()
    )
    challenges_raw = get_challenges(
        fri_entropy, trace_length*ext_degree, NUM_CHALLENGES
    )
    fri_top_leaf_count = trace_length*ext_degree >> FOLDS_PER_ROUND
    challenges_top = challenges_raw % fri_top_leaf_count
    challenges_bottom = challenges_raw >> log2(fri_top_leaf_count)
    challenges = rbo_index_to_original(
        trace_length*ext_degree,
        challenges_top * FOLD_SIZE_RATIO + challenges_bottom
    )
    challenges_next = (challenges+ext_degree) % (trace_length*ext_degree)

    return {
        "fri": fri_proof,
        "TQ_root": TQ_tree[1],
        "TQ_branches": [get_branch(TQ_tree, c) for c in challenges],
        "TQ_leaves": trace_quotient_ext[challenges],
        "TQ_next_branches": [get_branch(TQ_tree, c) for c in challenges_next],
        "TQ_next_leaves": trace_quotient_ext[challenges_next],
        "K_root": K_tree[1],
        "K_branches": [get_branch(K_tree, c) for c in challenges],
        "K_leaves": constants_ext[challenges],
        "S_at_w": S_at_w,
        "S_at_w_plus_G": S_at_w_plus_G,
    }

```

</details>

<details><summary><b> C++</b></summary>

```c++

while (!verifier.doneInteracting()) {
    std::cout << "communication iteration #" << msgNum++ << ":";
    bool doStatusLoop = true;
    Timer roundTimer;
    std::thread barManager([&]() {
        unsigned int sleepInterval = 10;
        unsigned int sleepTime = 10;
        while (doStatusLoop) {
            std::cout << "." << std::flush;
            for (unsigned int i = 0; (i < sleepTime) && doStatusLoop; i++) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(sleepInterval));
            }
            sleepTime *= 2;
        }
    });

    startVerifier();
    const auto vMsg = verifier.sendMessage();
    verifierTime += t.getElapsed();

    startProver();
    t.restart();
    const auto pMsg = prover.receiveMessage(vMsg);
    proverTime += t.getElapsed();

    startVerifier();
    t.restart();
    verifier.receiveMessage(pMsg);
    verifierTime += t.getElapsed();

    doStatusLoop = false;
    barManager.join();
    std::cout << "(" << roundTimer.getElapsed() << " seconds)" << std::endl;
}

...

void printSpecs(const double proverTime, const double verifierTime,
                const size_t proofGeneratedBytes, const size_t proofSentBytes,
                const size_t queriedDataBytes) {
    startSpecs();
    specsPrinter specs("Protocol execution measurements");
    specs.addLine("Prover time", secondsToString(proverTime));
    specs.addLine("Verifier time", secondsToString(verifierTime));
    specs.addLine("Total IOP length", numBytesToString(proofGeneratedBytes));
    specs.addLine("Total communication complexity (STARK proof size)",
                  numBytesToString(proofSentBytes));
    specs.addLine("Query complexity", numBytesToString(queriedDataBytes));
    specs.print();

    resetColor();
}

...

bool executeProtocol(const BairInstance& instance, const BairWitness& witness,
                     const unsigned short securityParameter, bool testBair,
                     bool testAcsp, bool testPCP) {
    const bool noWitness = !(testBair || testAcsp || testPCP);

    prn::printBairInstanceSpec(instance);
    unique_ptr<AcspInstance> acspInstance = CBairToAcsp::reduceInstance(
        instance,
        vector<FieldElement>(instance.constraintsPermutation().numMappings(),
                             one()),
        vector<FieldElement>(instance.constraintsAssignment().numMappings(),
                             one()));

    prn::printAcspInstanceSpec(*acspInstance);
    prn::printAprInstanceSpec(*acspInstance);

    ...
}

void simulateProtocol(const BairInstance& instance,
                      const unsigned short securityParameter) {
    BairWitness* witness_dummy = nullptr;
    Protocols::executeProtocol(instance, *witness_dummy, securityParameter,
                               false, false, false);
}


```

</details>

</details>

## Conclusion

Circle STARKs enhance zero-knowledge proofs by efficiently handling polynomials in geometric structures through innovative techniques like quotienting and two-point proving. Their use of interpolation functions and specialized vanishing polynomials improves computational efficiency and verification accuracy. Furthermore, the adoption of bit-reversed order in polynomial evaluation optimizes the FRI process, allowing for effective value pairing and increased spatial efficiency. Overall, Circle STARKs significantly advance the security and applicability of zero-knowledge proofs in decentralized systems and privacy-preserving technologies.

[Circle Stark](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/circlestark)

[libSTARK](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/libSTARK)

<div  align="center"> 
<img src="images/47_circleStark.gif" width="50%" />
</div>
