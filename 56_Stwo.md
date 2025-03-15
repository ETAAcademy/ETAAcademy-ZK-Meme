# ETAAcademy-ZKMeme: 56. Stwo

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>56. Stwo</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Stwo</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

---

# Stwo: The Next-Generation STARK Prover

Stwo ("STARK Two") is Starknet’s next-generation ultra-fast prover, designed to be 100 times more efficient than the current prover, Stone ("STARK One"). By significantly reducing verification costs and latency, Stwo enhances user experience and increases transactions per second (TPS), making it ideal for applications such as off-chain gaming, smart wallets, high-frequency low-slippage trading, and large-scale payment systems like Visa and Alipay. Stwo leverages **Circle STARKs**, built on the Mersenne prime **M31**, incorporating advanced mathematical techniques known as "moon math" to drive innovation. Additionally, it implements the **log-up** and **sum-check** protocols while supporting mixed-degree polynomials. Stwo also introduces new infrastructure to optimize circuit encoding and virtual machine execution.

Stwo will integrate into StarkWare’s **SHARP (Shared Proving) framework**, becoming a key component of **Starknet** and **StarkEx** on the mainnet. It will enable writing Stwo **AIR constraints** in Cairo Assembly (CASM) and implementing recursive verifiers in Cairo.

---

#### Finite Fields in STARK Proof Systems

All STARK proof systems rely on **finite fields**, mathematical structures containing a **finite number of elements**, ensuring that computations do not involve floating points, fractions, or excessively large integers. The simplest finite fields are **prime fields**, where the number of elements is a prime number **p**, and arithmetic operations follow modulo p rules. If $p < 2^{32}$, each number can be efficiently stored in a **32-bit word**, making computation significantly faster.

Traditional STARK proof systems like Stone, BSBHR18b, BSCR+19, and COS20 use large prime numbers $(~2^{252}),$ requiring **eight 32-bit words** for storage and averaging **38 CPU cycles** per multiplication operation, leading to high computational costs. Stwo improves efficiency by utilizing the **Mersenne prime** $M31 = 2^{31} - 1$, allowing each number to fit in a single 32-bit word. Thanks to **SIMD vectorization**, multiplication operations in M31 achieve an unprecedented speed of **0.3 CPU cycles**, making it **125 times faster** than Stone.

M31 is particularly suited for STARK proofs due to its mathematical properties. The congruence relation **2³¹ ≡ 1 mod M31** enables computational optimizations. Compared to **Babybear** $(2^{31} - 2^{27} + 1)$, the prime used by **Risc0**, M31 achieves **1.3x faster** multiplication under vectorized computation.

#### Challenges in Finite Field Arithmetic for STARK

Finite field arithmetic exhibits **cyclic structures** in both addition and multiplication. In a prime field $F_p$, addition cycles through all elements modulo **p**. However, multiplication is more complex, as its periodicity depends on the chosen generator’s order, ideally a power of 2 for efficient **FFT (Fast Fourier Transform)** and **FRI (Fast Recursive Interpolation)** computations.

Most traditional STARK systems operate in prime fields where a **cyclic group of smooth order** exists, enabling **efficient FFT-based interpolation** and constraint writing. However, Mersenne prime fields like $F_{2^{31}-1}$ present a unique challenge: the maximum power-of-2 factor in $p - 1 = 2^{31} - 2$ is just **2**, making it impossible to find a generator **g** with the required $2^{20}$ or higher periodicity for STARK computations. This limitation necessitates alternative solutions.

#### Alternative Solutions: ECFFT and Circle STARKs

One proposed alternative is **Elliptic Curve FFT (ECFFT)**, designed to support prime fields corresponding to widely used elliptic curves like **secp256k1** and **secp256r1**. ECFFT employs elliptic curves' smooth subgroups as a basis for FFT, bypassing finite field limitations. It operates through a **2-to-1 mapping**, progressively halving the computational domain size, enabling efficient interpolation in rational function spaces. Unlike traditional FFT, ECFFT does not require smooth-order cyclic groups, offering flexibility in prime selection.

However, ECFFT introduces complexity, requiring **algebraic geometry techniques** such as **divisor class groups and Riemann-Roch spaces**. To avoid this complexity, researchers developed **Circle STARKs**, leveraging **algebraic cyclic groups** based on circular structures.

#### Circle STARKs: A Novel Approach

Traditional STARK systems generate sequences via simple multiplicative operations. Circle STARKs, in contrast, define sequences based on **points on a unit circle** satisfying $x^2 + y^2 = 1$. Given a generator $(g_x, g_y)$ and an initial point $(x_0, y_0)$, the sequence follows:

$(x_{i+1}, y_{i+1}) = (gₓ \cdot x_i - gᵧ \cdot y_i, gₓ \cdot y_i + gᵧ \cdot x_i)$

For instance, in $F_5$, choosing **g = 4** generates the cyclic sequence **(1,0), (0,1), (-1,0), (0,-1),...**. Applying this concept to **M31** with an optimal generator **(2, 1268011823)** yields a cycle of **2³¹ elements**, facilitating efficient recursive FFT and NTT (Number Theoretic Transform) computations.

Compared to elliptic curve-based STARKs, Circle STARKs offer key advantages:

- **Quadratic mapping (2-to-1)** avoids complex elliptic curve chains.
- **Direct application of FFT to polynomial bases**, eliminating rational function dependencies.
- **Simpler mathematical structure**, avoiding deep algebraic geometry concepts.

Though Circle STARKs introduce minor challenges in global quotient constraints, these are efficiently mitigated without affecting security.

#### Performance Gains with Stwo

Stwo introduces **Poseidon2**, a STARK-friendly hash function optimized for M31. At **128-bit security**, its proof verification throughput exceeds **500,000 Poseidon hashes per second** on a **4-core Intel 7 CPU**, and **600,000 hashes per second** on a **12-core M3 Pro chipset**.

---

## 1. Circle Curve

In the prime field $F_p$ where $p \equiv 3 \mod 4$, the Circle Curve is defined by the equation $x^2 + y^2 = 1$. Its projective version takes the form $X^2 + Y^2 = Z^2$. Due to the condition $p \equiv 3 \mod 4$, this curve has no points at infinity in $F_p$; however, in the algebraic closure $K$ (which contains all possible roots), there exist two special points at infinity, $(1:i:0)$ and $(1:-i:0)$, which belong to the quadratic extension field $F_{p(i)}$ rather than $F_p$, where $i^2 = -1$. A key property of this curve is its isomorphism to the projective line $P^1(F_p)$. This isomorphism extends to **any finite field extension** $F$ and its algebraic closure $K$, implying that **the number of points on the Circle Curve equals the number of elements in the finite field plus one, i.e., $|C(F)| = |F| + 1$.**

This isomorphism is realized via **stereographic projection**, mapping the Circle Curve onto the projective line with the center at $(-1,0)$. The point $(-1,0)$ on the circle is mapped to the point at infinity on $P^1(F_p)$. By defining the transformation $t = \frac{y}{x+1}$ and its inverse $(x, y) = \left( \frac{1 - t^2}{1 + t^2}, \frac{2t}{1 + t^2} \right)$, every point on the Circle Curve can be parametrized using a single variable $t$. This allows a reduction from a two-variable polynomial constraint problem (where constraints are imposed within a particular algebraic structure) to a single-variable problem, avoiding the need for advanced algebraic geometry tools like the Riemann-Roch theorem.

Moreover, the fact that the number of points on the Circle Curve over any finite field extension $F$ is $|F| + 1$ is crucial for the design of efficient computational methods such as **Circle Fast Fourier Transform (CFFT)**.

### Circle Group

The Circle Curve $C(F_p)$ forms a cyclic group of order $p+1$ over the finite field $F_p$, known as the **Circle Group**. The group operation is defined as:

$(x_0, y_0) \cdot (x_1, y_1) = (x_0 x_1 - y_0 y_1, x_0 y_1 + y_0 x_1).$

This structure implies that each element $P = (P_x, P_y)$ corresponds to a **rotation on the unit circle**, a crucial property for Circle FFT computations. The identity element is $(1,0)$, and the group inverse is given by $J(x,y) = (x,-y)$, meaning the inverse of any point is obtained by negating the $y$-coordinate. The squaring operation (multiplication of a point by itself) follows the rule:

$\pi(x,y) = (2x^2 - 1, 2xy).$

By mapping $(x, y) \to x + iy$, the Circle Group can be viewed as a multiplicative subgroup of the field extension $F_p(i)$, showcasing its cyclic nature. From the perspective of the projective line $P^1(F_p)$, it is isomorphic to a particular linear automorphism group:

$(t_1 : s_1) \oplus (t_2 : s_2) = (t_1 s_2 + t_2 s_1 : s_1 s_2 - t_1 t_2).$

In affine coordinates, this simplifies to:

$t_1 \oplus t_2 = \frac{t_1 + t_2}{1 - t_1 t_2}.$

This property is essential in Galois-FFT and Circle FFT computations. To construct efficient FFT domains, we leverage the cyclic structure of this group and define **twin-cosets** and **standard position cosets**.

For a binary order $N = 2^n$, there exists a unique cyclic subgroup $G_n$ of size $2^n$, composed of smaller subgroups $G_{n-1}$. The **twin-coset** is defined as:

$D = Q \cdot G_{n-1} \cup Q^{-1} \cdot G_{n-1},$

where $Q \cdot G_{n−1}$ and $Q^{-1} \cdot G_{n-1}$ are disjoint. If a twin-coset itself forms a new subgroup $G_n$, it is called a **standard position coset**. The existence of standard position cosets is closely related to whether the prime $p$ supports CFFT. Specifically, when $p \equiv 3 \mod 4$ and $2^{n+1}$ divides $p+1$, a unique standard position coset of size $2^n$ exists.

The structure of standard position cosets is defined by:

$Q \cdot G_{n-1} \cup Q^{-1} \cdot G_{n-1},$

where $Q$ is an element of order $2^{n+1}$. Additionally, the group inverse mapping $J(x,y) = (x,-y)$ may preserve some elements, meaning that $Q^{-1} \cdot G_{n-1} = Q \cdot G_{n-1}$ when $Q^2 \in G_{n-1}$. These fixed points distinguish the Circle Group structurally from elliptic curves and influence encoding and decoding processes in STARK computations.

While standard position cosets provide a structured computational domain, **twin-cosets** offer a more natural domain for CFFT calculations. Particularly in rotation operations, twin-cosets effectively mitigate computational irregularities. Since twin-cosets can be recursively decomposed into smaller twin-cosets, similar to how evaluation domains in single-variable FFTs are decomposed, they optimize computation further. In recursive computations, standard position cosets map to new standard position cosets, facilitating efficient CFFT computation.

A key result states that under the squaring map $\pi$, twin-coset size is halved while its structure remains unchanged. Specifically, if a coset is a standard position coset, then after the squaring map, it remains a standard position coset. If $D$ is a standard position coset of size $M = 2^m$, it decomposes into smaller twin-cosets as:

$D = Q \cdot G_m = \bigcup_{k=0}^{M/N - 1} (Q^{4k+1} \cdot G_{n-1} \cup Q^{-4k-1} \cdot G_{n-1}),$

where $Q$ is an element of order $2^{m+1}$ in $C(F_p)$ and $Q^4$ generates the subgroup $G_{m-1}$. The squaring map $\pi(x,y) = (2x^2-1, 2xy)$ maintains the twin-coset structure while recursively reducing the computational domain size. If $D$ is a twin-coset of size $N=2^n$, then its image under $\pi$ remains a twin-coset of size $N/2$. More importantly, if $D$ is a standard position coset, then $\pi(D)$ remains a standard position coset.

---

### Circle Codes

On the circular curve $C(F_p)$, an important polynomial space $L_N(F)$ is defined, where $F$ is an extension of the finite field $F_p$, and $N \geq 0$ is an even integer. This space consists of all bivariate polynomials that satisfy the constraint $x^2 + y^2 = 1$, with coefficients from $F$ and a total degree not exceeding $N/2$. In other words, $L_N(F)$ is the set of all polynomials in the quotient ring $F[x, y]/(x^2 + y^2 - 1)$ that meet the degree condition. The dimension of this space is $N+1$, and any nontrivial polynomial has at most $N$ roots.

From an algebraic geometry perspective, this structure corresponds to the Riemann-Roch space on the circular curve, describing all rational functions that have poles only at the points $\infty$ and $\bar{\infty}$, with pole orders at most $N/2$. Unlike elliptic curves, the geometry of the circular curve allows its function space to be represented directly using polynomials without requiring more complex rational functions. This characteristic makes computations more intuitive and is particularly critical for the implementation of STARK proof systems. In the context of STARK proofs, $L_N(F)$ plays a role similar to the low-degree extension (LDE) in traditional univariate STARK proofs. One of its key properties is **rotation invariance**: for any $f \in L_N(F)$ and any point $P \in C(F_p)$, the function $f \circ T_P$ still belongs to $L_N(F)$. This property ensures computational stability and is essential for efficient encoding and adjacency relations. Additionally, $L_N(F)$ exhibits strong divisibility properties, enabling the construction of Maximum Distance Separable (MDS) codes, which are crucial for data validation, error correction, and efficient proof systems.

#### Properties of $L_N(F)$

The polynomial space $L_N(F)$, defined on the circular curve, has several key properties:

- **Fractional Representation:** Through an isomorphic transformation, the space $L_N(F)$ can be mapped to a fractional representation of univariate polynomials. Specifically, a polynomial $p(t)$ is normalized as $p(t)/(1+t^2)^{N/2}$, where $\deg p(t) \leq N$. This representation is particularly useful in subsequent proofs and computations, facilitating coding theory on circular curves.

- **Dimension Calculation:** The dimension of $L_N(F)$ is precisely $N+1$. This result is derived constructively by demonstrating how bivariate polynomials $(x, y)$ can be converted into a univariate representation $(t)$ and analyzed using fractional forms. The correctness of this dimension computation is further verified using the **Riemann-Roch theorem**. Unlike elliptic curves, the circular curve has genus zero, meaning its Riemann-Roch space’s dimension depends solely on the divisor degree without any additional genus-dependent terms.

- **Canonical Form:** Every polynomial $p(x, y) \in L_N(F)$ can be uniquely expressed in the standard form:

  $p(x, y) = p_0(x) + y \cdot p_1(x),$

  where $p_0(x)$ is a polynomial of degree at most $N/2$, and $p_1(x)$ has a degree at most $N/2 -1$. This formulation arises from the constraint $x^2 + y^2 = 1$, allowing any polynomial to be rewritten by substituting $y^2 = 1 - x^2$.

- **Monomial Basis:** The monomial basis of $L_N(F)$ is given by:

  $1, x, x^2, ..., x^{N/2}, y, yx, yx^2, ..., yx^{N/2-1}.$

  The construction of this basis demonstrates the completeness of the space and provides a concrete framework for computational implementations, such as FFT-based calculations.

**Circle Codes** are linear codes derived from the space $L_N(F)$. These codes are particularly significant in **STARK proof systems**, serving as a bivariate counterpart to the **low-degree extension (LDE)** in univariate STARK schemes, analogous to Lagrange interpolation-based extensions in traditional settings. Due to their **rotation invariance** and **MDS properties**, Circle Codes exhibit exceptional characteristics in efficient encoding and error detection.

A **Circle Code** is an algebraic geometry code defined on a prime field $F_p$ where $p \equiv 3 \mod 4$. It is constructed by selecting a subset $D$ of points on the circular curve $C(F_p)$, with size $|D| > N$, and evaluating functions in $L_N(F)$ at these points. The resulting linear code $C_N(F, D)$ belongs to the class of **Generalized Reed-Solomon (GRS) codes**, with parameters:

- **Dimension:** $k = N + 1$
- **Minimum distance:** $d = |D| - N$

This code satisfies the Singleton bound, meaning it is an **MDS code** with optimal error-correcting capabilities. Furthermore, due to the unique structure of the circular curve, Circle Codes exhibit both **rotational and reflectional symmetry**—i.e., for any rotation $Q$ or reflection $J$ that maps $D$ onto itself, the code remains invariant. This symmetry enables more efficient FFT computations.

#### Computational Complexity and Applications

The theoretical foundation of Circle Codes establishes an isomorphic mapping to the classical **Reed-Solomon code** $RS_{N+1}[F, S]$, preserving distance properties. This mapping allows direct application of fast decoding techniques from Reed-Solomon codes. The computational complexity of encoding and decoding Circle Codes is:

- **Encoding Complexity:** $O(|D| \log N)$
- **Decoding Complexity:** The **Guruswami-Sudan decoder** can perform list decoding in $O(|D|^{1.5})$ complexity, efficiently recovering codewords close to a given received sequence.

In **STARK proof systems**, Circle Codes play a pivotal role in efficient **low-degree extensions (LDE)**. By leveraging FFT computations over standard cosets of order $2^k$, Circle Codes enable highly efficient multivariate LDE schemes. While Circle Codes introduce additional dimensional constraints compared to classical Reed-Solomon codes—limiting coverage to a subspace of the entire encoding space—this restriction has minimal impact on computational performance while still ensuring excellent error correction and efficiency.

---

### Vanishing Polynomials and Quotients

In the STARK proof system, vanishing polynomials play a crucial role, particularly in the Low-Degree Extension (LDE) and Algebraic Linking (DEEP) steps. These polynomials ensure encoding correctness and efficient polynomial interpolation. A vanishing polynomial is a nonzero polynomial that evaluates to zero over a specific set of points, denoted as $D$. In constructing these polynomials, the point set $D$ is often decomposed into paired points $(P_k, Q_k)$, forming a product representation:

$v_D(x, y) = \prod_{k=1}^{N/2} ((x - P_{k,x}) (Q_{k,y} - P_{k,y}) - (y - P_{k,y}) (Q_{k,x} - P_{k,x}))$

where $D$ is a set of points on the unit circle $C(F_p)$, with size $N$ (an even number), satisfying $2 \leq N < p+1$.

For an even-sized set $D$, these vanishing polynomials form a one-dimensional subspace, meaning they are unique up to scalar multiplication. Specifically, for twin-coset structured point sets $D$, a power mapping $\pi_{n-1}$ can be used to obtain a subset of size 2, leading to the definition of the vanishing polynomial as:

$v_D(x, y) = v_n(x, y) - x_D$

where $v_n(x, y) = \pi_x \circ \pi_{n-1} (x, y)$. Recursively constructed vanishing polynomials can be computed efficiently with complexity $O(n)$, reducing redundancy and enhancing STARK proof performance. One key property of these polynomials is their symmetry under group transformations. In particular, under standard cosets, vanishing polynomials alternate under the action of a generator, satisfying:

$v_n \circ T_P = -v_n$

This symmetry can be effectively leveraged to optimize computations in the STARK proof process.

#### Domain Quotients and Single-Point Quotients

The construction of domain quotients and single-point quotients plays a significant role in the algebraic linking step of STARK proofs over a finite field $F_p$. Given a point set $D$ of even size $N$, a quotient polynomial is formed using the vanishing polynomial $v$:

$f = q \cdot v$

where $q$ belongs to a lower-degree space $L_{M-N}(F)$, ensuring that the quotient polynomial exhibits specific pole structures while maintaining computational efficiency.

Next, vanishing polynomials on the unit circle $v_P$ are introduced to handle single-point quotient polynomials. Here, $v_P$ has a simple root at $P = (P_x, P_y)$ and a simple pole at the point at infinity $\bar{\infty}$. This leads to the quotient polynomial:

$q = \frac{f - f(P)}{v_P}$

which belongs to the space $L_N(F)$ and can be decomposed into real and imaginary parts, providing an effective computational framework for operations on the unit circle.

#### DEEP Quotients and Their Extensions

The concept of Deep Quotients (DEEP) further extends this approach. By utilizing vanishing polynomials $v_P$ on the unit circle, not only can single-point quotient polynomials be efficiently represented, but their decomposition into real and imaginary components is also facilitated. By extending computations to the field $F_p(i)$, these quotient polynomials remain both theoretically sound and computationally efficient. Additionally, the distribution of poles and zeros within the computation is well-regulated, preventing redundancy and ensuring optimal execution in STARK proofs.

---

## 2. Circle FFT

The theoretical framework of the Circular Fast Fourier Transform (CFFT) and its applications in the STARK proving system are explored in this section. Let $p$ be a CFFT-friendly prime that supports a subgroup $G_n$ of order $N = 2^n$ on a circular curve. Given a twin-coset set $D$, CFFT maps functions from an extended field $F$ to the polynomial space $L_N(F)$ through polynomial interpolation. The FFT basis $B_n$ is an $N$-dimensional basis formed by polynomials structured according to the field size. Each basis polynomial is given by:

$b_j^{(n)}(x, y) = y^{j_0} \cdot v_1(x)^{j_1} \cdots v_{n-1}(x)^{j_{n-1}}$

where $v_k(x)$ represents the vanishing polynomials corresponding to the standard position coset. The theorem states that coefficients of a given function with respect to $B_n$ can be efficiently computed using a specific algorithm with relatively low computational cost. CFFT is a non-harmonic FFT whose construction is closely related to two group homomorphisms on the circular curve: the group squaring map $\pi$ and the inversion map $J$. To achieve interpolation in low-degree function spaces, the dimensionality must closely match the field size. While this adaptation is perfect for elliptic curves, a slight dimensional discrepancy exists in the case of Circular FFT.

The recursive structure of the _twin-coset_ domain and its quotient space mappings simplify computation. The twin-coset domain $D = Q \cdot G_{n-1} \cup Q^{-1} \cdot G_{n-1}$ consists of two cosets, $Q \cdot G_{n-1}$ and $Q^{-1} \cdot G_{n-1}$, and remains invariant under the involution $J(D) = D$. Each $J$-orbit contains exactly two points. This leads to a **2-to-1 quotient mapping**:

$\varphi_J : D \to D/J, \quad P \mapsto \{P, J(P)\}$

which collapses each orbit into a single point, meaning that each point in the quotient space corresponds to two points in $D$. Recursively applying this process from the initial twin-coset domain $D_n$, each projection step reduces the domain size:

$D_j = \pi(D_{j+1}), \quad j = n-1, n-2, ..., 1$

until it shrinks to the final set $D_1$, which contains only a single $J$-orbit (i.e., $|D_1 / \varphi_J| = 1$). Each projection $\pi$ is a **2-to-1 mapping**, exponentially reducing the set size from $2^n$ down to $2^1$. This structure can be represented as a **hierarchical projection diagram**:

$\begin{array}{cccccc} D_n & \to & D_{n-1} & \to & D_{n-2} & \to \dots \to D_1 \\ \downarrow & & \downarrow & & \downarrow & \\ D_n/J & \to & D_{n-1}/J & \to & D_{n-2}/J & \to \dots \to D_1/J \end{array}$

In the quotient space $S_j = D_j / J$, this recursive mapping simplifies to an **x-axis transformation**, ultimately equivalent to a **quadratic transformation**:

$x \mapsto 2x^2 - 1$

which holds special significance in the Circular FFT. Due to the twin-coset structure and the 2-to-1 projection mappings, this framework provides an **efficient data structure reduction method**. It significantly reduces computational costs in FFT calculations while maintaining mathematical integrity and algebraic properties. This framework is not only applicable to Circular FFT but may also be useful in other computations involving twin-coset domains, projection mappings, and recursive reduction, facilitating optimized polynomial computation, Fourier transforms, and algebraic calculations.

### Recursive Structure of Circle FFT

Circle FFT is a recursive FFT algorithm based on the twin-coset structure, particularly suited for finite field computations under CFFT-friendly primes. Its core idea is to reduce an interpolation problem from a two-dimensional domain step by step via projection mappings, ultimately transforming it into a one-dimensional computation. The algorithm employs an "even-odd decomposition" to split the input function $f$ into an even part $f_0$ and an odd part $f_1$. Using the reference odd function $t_0(x, y) = y$, which satisfies the self-invariance property $t_0(J(x, y)) = -t_0(x, y)$, the function $f$ is decomposed as follows:

$f_0(x) = \frac{f(x, y) + f(x, -y)}{2}, \quad f_1(x) = \frac{f(x, y) - f(x, -y)}{2 \cdot y}$

so that the original function can be reconstructed as:

$f(x,y) = f_0(x) + y \cdot f_1(x)$

Following this, at each projection step $\pi(x)$, the functions $f_0$ and $f_1$ are further decomposed recursively until they yield coefficients associated with the FFT basis $B_n$. The computational complexity is comparable to classical FFT, i.e., $O(N \log N)$, but it adapts better to circular domain structures such as twin-cosets in Riemann-Roch spaces. The inverse Circular FFT (Inverse CFFT) follows a symmetric recursive structure in the opposite direction, starting from the minimal single-point domain $S_1$ and iteratively merging computations into larger domains to reconstruct the original function $f(x, y)$. This symmetry ensures efficient data recovery, making CFFT computationally balanced and numerically stable in certain special domains compared to traditional FFT methods.

### FFT Space Properties in Circle STARK

The decomposition of FFT space and its behavior at infinity are crucial in the Interactive Oracle Proof (IOP) system of Circle STARK. Specifically, the FFT space is decomposed as:

$L_N(F) = L'_N(F) + \langle v_n \rangle$

where $L'_N(F)$ is a special subspace and $v_n$ is a vanishing polynomial. Despite Circle FFT being non-harmonic, $L'_N(F)$ remains invariant under rotational actions. This is demonstrated using the vanishing polynomial $v_G(x, y)$ of subgroup $G_n$ and its linear independence properties. Moreover, $v_n$ is orthogonal within $L'_N(F)$, a result derived from FFT decomposition and $J$-invariance.

At infinity, defined as $\infty = (1 : i : 0)$ and $\bar{\infty} = (1 : -i : 0)$, polynomials in $L'_N(F)$ exhibit antisymmetric limits, whereas $v_n$ has identical limits at both points. This leads to the definition of two subspaces:

$L^-_N(F) = \{ f \in L_N(F) \mid f(\infty) = - f(\bar{\infty}) \}, \quad L^+_N(F) = \{ f \in L_N(F) \mid f(\infty) = f(\bar{\infty}) \}$

Further analysis reveals that products of polynomials in $L^-$ and $L^+$ behave predictably, ensuring that $L'_N(F) = L^-_N(F)$, meaning it consists entirely of polynomials with antisymmetric limits at infinity. This property is essential in Circle STARK proofs, which frequently involve polynomials with rotational symmetry and controlled vanishing behavior.

---

## 3. Circle STARK

Circle STARK is a zero-knowledge proof protocol based on circular curves. Unlike traditional one-dimensional STARKs, Circle STARK operates within a two-dimensional polynomial space defined over a circular curve, specifically $F_p[x, y] / (x^2 + y^2 - 1)$, rather than a one-dimensional polynomial space. This protocol is particularly advantageous for arithmetic circuits over large prime fields $F_p$ where $p-1$ lacks smoothness, but $p+1$ meets the conditions required for Circular Fast Fourier Transform (CFFT). Under such conditions, witness data is encoded as a two-dimensional polynomial, and constraints are transformed into algebraic relations via CFFT. The interactive proof process in Circle STARK follows a structure similar to that of traditional STARKs. However, due to the dimensional gap between the output space of the circular FFT and the polynomial space, constraints must be adjusted accordingly.

In this protocol, the trace domain consists of a standard position cosine set within a cyclic subgroup. The trace sequence is interpolated using two-dimensional polynomials of degree at most $N/2$, adhering to specific constraints that are typically periodic and enforced over a subdomain. These constraints and polynomials are committed to by the prover within an extended evaluation domain. By leveraging efficient Algebraic Intermediate Representations (AIR) and constraint satisfaction techniques, Circle STARK ensures computational security while enhancing verification efficiency, particularly in complex algebraic relations.

#### Constraint Selectors in Circle STARK

Constraint selectors play a crucial role in constructing polynomial constraints in zero-knowledge proof systems. The subdomain selector polynomial $s_{H'} = \frac{v_H}{v_{H'}}$ enables constraints on a coset $H'$ of a subgroup $G'$ within the standard position coset $H$. This polynomial is proven to belong to the FFT space $L'_N(F_p)$ and alternates under the action of $G'$. Additionally, the singleton domain selector polynomial $s_P = \frac{v_n}{v_0 \circ T_P^{-1}}$ allows for the selection of individual points within the FFT space, taking a nonzero value at target point $P$ while remaining zero elsewhere. This feature is particularly useful for efficiently computing function values in the FFT space.

To further optimize Circle STARK, a tangent function $s_P = x_P \cdot x + y_P \cdot y - 1$ replaces the selector polynomial. This function has a double root at $P$ but no other roots on the unit circle. When $P = (1,0)$, $s_P$ belongs to the space $L_2^+(F_p)$, reducing computational complexity.

#### Algebraic Intermediate Representation (AIR) and the IOP Protocol

The interactive oracle proof (IOP) protocol in Circle STARK proves the existence of low-degree polynomials $p_1, ..., p_i \in F_p[X, Y] / (X^2+Y^2-1)$ that satisfy all constraints on the trace domain $H$. The protocol unfolds in several stages:

1. **Trace Polynomial Computation**: The prover evaluates trace polynomials over the evaluation domain $D$ and shares these values with the verifier.
2. **Constraint Aggregation**: The verifier sends a random challenge $\beta$, which is used to aggregate multiple constraints into a single identity.
3. **Quotient Polynomial Computation**: The prover computes a quotient polynomial $q$ and decomposes it into $q_1, ..., q_{(d-1)}$ along with a parameter $\lambda$.
4. **DEEP Algebraic Linking**: The verifier selects a random point $\gamma$, and the prover declares polynomial evaluations at $\gamma$ and $T(\gamma)$.
5. **Low-Degree Testing**: The verifier checks the real and imaginary parts of the DEEP quotient polynomial to ensure correctness.

The reliability error $\varepsilon_{AIR}$ consists of constraint batching errors, DEEP linking errors, and proximity testing errors. By using real and imaginary components, the protocol avoids excessively large fields. Although it doubles the number of test functions, the computational overhead remains minimal. Moreover, the protocol can be transformed into a SNARK within the random oracle model. A key innovation is the introduction of parameter $\lambda$ to handle dimensional discrepancies, distinguishing it from traditional univariate and elliptic curve-based methods.

#### Quotient Polynomial Decomposition and Dimension Adjustment

A crucial step in the AIR protocol is computing the overall quotient polynomial $q \in L(d-1) \cdot N(F)$. This polynomial is derived by evaluating the trace polynomials $f_1, ..., f_w$ over the disjoint union $\bar{H}$ of cosets $H_k$. The "Decomposition Lemma" ensures that any $q \in L(d-1) \cdot N(F)$ can be uniquely expressed as:

$q = \lambda \cdot v_{\bar{H}} + \sum_{k=1}^{d-1} \frac{v_{\bar{H}}}{v_{H_k}} \cdot q_k,$
where $\lambda \in F$ and $q_k \in L'_N(F)$. This decomposition is necessary because $\bar{H}$ contains one fewer point than needed to uniquely determine polynomials in $L(d-1) \cdot N$. The parameter $\lambda$ resolves this "dimension gap."

To uniquely determine $q$, "infinity limit computation" is introduced, where the overall identity is divided by the highest-degree monomial and evaluated at infinity. It is shown that:

- If $d$ is odd, $q \in L^-(d-1) \cdot N(F)$ and $\lambda = 0$.
- If $d$ is even, $q \in L^+(d-1) \cdot N(F)$ and satisfies specific conditions.

This method extends to selectors based on linear tangent polynomials, allowing for constraints of maximum degree $d$ over punctured domains $H'$ with minimal computational overhead.

### Low-Degree Testing in Circle STARK

The low-degree test in Circle STARK employs an interactive oracle proof of proximity (IOP) to verify whether a function $f$ is close to a codeword in the circular code. Specifically, the goal is to show that the relative Hamming distance $d(f, C)$ is below a threshold $\theta$. This is achieved by decomposing $f$ into an FFT-space component $g$ and a vanishing polynomial component $v_n$, followed by successive folding steps that reduce the function space dimension.

The key to circular low-degree testing is a series of projection steps that iteratively reduce the polynomial space from high-degree polynomials to constant functions. The final objective is to efficiently verify that $f$ adheres to the low-degree polynomial proximity standard. Circular codes closely relate to Reed-Solomon codes, and their two-variable representation enables folding mechanisms that simplify polynomial spaces, making proximity testing more efficient. The robustness of circular proximity proofs relies on the geometric structure of the folding process, ensuring the reliability of the low-degree test.

---

## 4. LogUp & GKR

The Goldwasser-Kalai-Rothblum (GKR) protocol plays a crucial role in optimizing fractional sumcheck verification in lookup arguments based on logarithmic derivatives. This approach significantly reduces the computational cost for the prover. Specifically, when looking up multiple table columns, the prover only needs to commit to one additional column, representing the multiplicities of table entries. Furthermore, transitioning from univariate polynomial commitment schemes to multilinear commitment schemes opens new avenues for optimizing polynomial commitment applications and proofs.

#### LogUp

The "logUp" protocol (lookup argument) focuses on proving that multiple polynomial values belong to a predefined table. More concretely, the prover presents a set of polynomials $w_1(\vec{X}), \dots, w_M(\vec{X})$ defined over a Boolean hypercube $H_n$ and aims to convince the verifier that their values align with another polynomial $t(\vec{X})$ over the same hypercube. To accomplish this, the prover provides a polynomial $m(\vec{X})$, which represents the frequency of each element in the table, defined through Lagrange representation. This polynomial establishes the relationship between $w_i(\vec{X})$ and $t(\vec{X})$. By transforming logarithmic derivatives into sumcheck verifications, the logUp protocol simplifies the otherwise complex problem of polynomial accumulation into an efficient summation check, eliminating the need for large-scale multiplications.

The verifier supplies a random value $\alpha$ and employs evaluation checks, while the prover demonstrates equality via summation checks. Compared to traditional multivariate plookup protocols, logUp reduces the number of auxiliary columns the prover must provide and enables performance optimization by adjusting the number of helper functions and algebraic degrees. Additionally, the logUp protocol integrates the GKR protocol to further optimize the sumcheck process, eliminating the need to provide multiple auxiliary columns for each fractional term. This combined approach significantly lowers computational complexity, making logUp more advantageous for polynomial identity verification.

#### The Role of the GKR Protocol

The Goldwasser-Kalai-Rothblum (GKR) protocol is an interactive proof system designed to efficiently verify relationships between polynomials, particularly over structured domains such as finite fields or Boolean domains. It systematically reduces proof complexity to enhance efficiency. Given two polynomials $p(\vec{X})$ and $q(\vec{X})$, both defined over a finite field $F$ and composed of $n$-variable multilinear polynomials, the GKR protocol verifies the following relationship over the Boolean hypercube $H_n$:

$\sum_{\vec{x} \in H_n} p(\vec{x}) \cdot q(\vec{x}) = 0.$

Here, $H_n$ is the $n$-dimensional Boolean hypercube consisting of all vectors formed from $\pm1$. Although this equation can be generalized to higher-degree polynomials and more variables, we focus on the fundamental case for simplicity.

Applying the GKR protocol to logUp simplifies the fractional sumcheck verification process, significantly reducing the prover’s workload—especially when handling large lookup columns. Assuming the number of lookup columns $M$ is $2^k - 1$, the total number of columns, including the table column, becomes $2^k$. The prover must validate the following fractional sumcheck formula:

$\sum_{\vec{x} \in H_n} \left( \alpha \cdot m(\vec{x}) - \left( \alpha - t(\vec{x}) \right) - \sum_{i=1}^{M} \left( \frac{1}{\alpha - w_i(\vec{x})} \right) \right) = 0,$

where $\alpha$ is a random value supplied by the verifier. This formula establishes the relationship between the table polynomial, multiplication table polynomial, and witness column polynomials.

To prove this relationship, the prover constructs a multilinear polynomial $p(\vec{X}, \vec{Y})$ and $q(\vec{X}, \vec{Y})$ with $n+k$ variables and utilizes the GKR protocol. The verifier queries random points $\vec{r} \in F^n$ to validate evaluations of the table polynomial $t(\vec{X})$, multiplication table polynomial $m(\vec{X})$, and witness polynomials $w_1(\vec{X}), \dots, w_M(\vec{X})$. Consequently, the prover needs only one additional column to store table value frequencies, $m(\vec{X})$.

Regarding security, transforming fractional identities into fractional sumchecks introduces a soundness error, known as the logUp-GKR soundness error. To ensure protocol integrity, the random points must be chosen from $F \setminus H$.

#### Performance Analysis and Single-Variable Adaptation

Leveraging the GKR protocol reduces computational costs significantly for the prover. Compared to the basic logUp variant, using the GKR protocol roughly doubles the cost. Specifically, the computational cost per lookup element is $43 \times M + 29 \times A$, where $A$ represents the algebraic degree of computations. In contrast, the basic variant incurs a cost of $19 \times M + 16 \times A$. Different commitment schemes impact prover overhead differently: elliptic curve-based schemes may provide a tenfold advantage, whereas Reed-Solomon encoding-based schemes exhibit a smaller performance gain, depending on the hash function used.

The interactive oracle proof (IOP) method converts univariate polynomial commitments into multilinear commitments and enables fractional sumcheck GKR verification in univariate logUp. Unlike existing approaches, this method consistently employs Lagrange representations—including multilinear and univariate representations—mapping hypercube values to univariate domain values for a straightforward transformation.

1. Define a univariate ring domain $H = \{x \in F : x^{2n} = 1\}$ and establish a bijective mapping with the Boolean hypercube $H^n = \{ \pm1 \}^n$ via bit decomposition. This mapping translates polynomial values from the Boolean hypercube into univariate polynomial representations. Evaluations at query points $\vec{t} = (t_0, ..., t_{n-1})$ facilitate polynomial value verification.
2. During the protocol, the verifier employs Lagrange interpolation for querying, and the prover extends column values using a univariate polynomial $c(X)$ under periodicity constraints. The prover then applies a univariate sumcheck protocol to verify the polynomial $f(t_0, ..., t_{n-1})$.
3. Security analysis establishes error bounds and extends the protocol to batch evaluations for multiple polynomials. The protocol remains zero-knowledge and supports verification of multiple polynomial evaluations at given points.

The univariate logUp/GKR protocol applies the GKR protocol to verifying univariate logUp computations. Given a single-column lookup table $t$ and $M = 2^k - 1$ proof columns $w_1, \dots, w_M(X)$ where $k \geq 1$, the protocol maps univariate domain values to Boolean hypercube values $\{\pm1\}^n$ and employs the GKR protocol for fractional sumchecks. Ultimately, multilinear polynomial evaluations $t(\vec{X})$, $w_1(\vec{X})$, etc., at a random point $\vec{r}$ transform into a linear combination evaluation statement, verified via a univariate IOP protocol. The security error comprises the multivariate protocol's error and the univariate IOP conversion error, with the final security error being the sum of both components.

<details><summary><b> Code </b></summary>

<details><summary><b> prover/src/core/air/mod.rs </b></summary>

```rust

/// Arithmetic Intermediate Representation (AIR).
///
/// An Air instance is assumed to already contain all the information needed to evaluate the
/// constraints. For instance, all interaction elements are assumed to be present in it. Therefore,
/// an AIR is generated only after the initial trace commitment phase.
pub trait Air {
    fn components(&self) -> Vec<&dyn Component>;
}

...

/// A component is a set of trace columns of various sizes along with a set of
/// constraints on them.
pub trait Component {
    fn n_constraints(&self) -> usize;

    fn max_constraint_log_degree_bound(&self) -> u32;

    /// Returns the degree bounds of each trace column. The returned TreeVec should be of size
    /// `n_interaction_phases`.
    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>>;

    /// Returns the mask points for each trace column. The returned TreeVec should be of size
    /// `n_interaction_phases`.
    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>;

    fn preproccessed_column_indices(&self) -> ColumnVec<usize>;

    /// Evaluates the constraint quotients combination of the component at a point.
    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    );
}

pub trait ComponentProver<B: Backend>: Component {
    /// Evaluates the constraint quotients of the component on the evaluation domain.
    /// Accumulates quotients in `evaluation_accumulator`.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
    );
}

```

</details>

<details><summary><b> pcs/prover.rs </b></summary>

```rust

    pub fn prove_values(
        self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        channel: &mut MC::C,
    ) -> CommitmentSchemeProof<MC::H> {
        // Evaluate polynomials on open points.
        let span = span!(Level::INFO, "Evaluate columns out of domain").entered();
        let samples = self
            .polynomials()
            .zip_cols(&sampled_points)
            .map_cols(|(poly, points)| {
                points
                    .iter()
                    .map(|&point| PointSample {
                        point,
                        value: poly.eval_at_point(point),
                    })
                    .collect_vec()
            });
        span.exit();
        let sampled_values = samples
            .as_cols_ref()
            .map_cols(|x| x.iter().map(|o| o.value).collect());
        channel.mix_felts(&sampled_values.clone().flatten_cols());

        // Compute oods quotients for boundary constraints on the sampled points.
        let columns = self.evaluations().flatten();
        let quotients = compute_fri_quotients(
            &columns,
            &samples.flatten(),
            channel.draw_felt(),
            self.config.fri_config.log_blowup_factor,
        );

        // Run FRI commitment phase on the oods quotients.
        let fri_prover =
            FriProver::<B, MC>::commit(channel, self.config.fri_config, &quotients, self.twiddles);

        // Proof of work.
        let span1 = span!(Level::INFO, "Grind").entered();
        let proof_of_work = B::grind(channel, self.config.pow_bits);
        span1.exit();
        channel.mix_u64(proof_of_work);

        // FRI decommitment phase.
        let (fri_proof, query_positions_per_log_size) = fri_prover.decommit(channel);

        // Decommit the FRI queries on the merkle trees.
        let decommitment_results = self
            .trees
            .as_ref()
            .map(|tree| tree.decommit(&query_positions_per_log_size));

        let queried_values = decommitment_results.as_ref().map(|(v, _)| v.clone());
        let decommitments = decommitment_results.map(|(_, d)| d);

        CommitmentSchemeProof {
            commitments: self.roots(),
            sampled_values,
            decommitments,
            queried_values,
            proof_of_work,
            fri_proof,
            config: self.config,
        }
    }

```

</details>

<details><summary><b> pcs/verifier.rs </b></summary>

```rust

    pub fn verify_values(
        &self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        proof: CommitmentSchemeProof<MC::H>,
        channel: &mut MC::C,
    ) -> Result<(), VerificationError> {
        channel.mix_felts(&proof.sampled_values.clone().flatten_cols());
        let random_coeff = channel.draw_felt();

        let bounds = self
            .column_log_sizes()
            .flatten()
            .into_iter()
            .sorted()
            .rev()
            .dedup()
            .map(|log_size| {
                CirclePolyDegreeBound::new(log_size - self.config.fri_config.log_blowup_factor)
            })
            .collect_vec();

        // FRI commitment phase on OODS quotients.
        let mut fri_verifier =
            FriVerifier::<MC>::commit(channel, self.config.fri_config, proof.fri_proof, bounds)?;

        // Verify proof of work.
        channel.mix_u64(proof.proof_of_work);
        if channel.trailing_zeros() < self.config.pow_bits {
            return Err(VerificationError::ProofOfWork);
        }

        // Get FRI query positions.
        let query_positions_per_log_size = fri_verifier.sample_query_positions(channel);

        // Verify merkle decommitments.
        self.trees
            .as_ref()
            .zip_eq(proof.decommitments)
            .zip_eq(proof.queried_values.clone())
            .map(|((tree, decommitment), queried_values)| {
                tree.verify(&query_positions_per_log_size, queried_values, decommitment)
            })
            .0
            .into_iter()
            .collect::<Result<(), _>>()?;

        // Answer FRI queries.
        let samples = sampled_points.zip_cols(proof.sampled_values).map_cols(
            |(sampled_points, sampled_values)| {
                zip(sampled_points, sampled_values)
                    .map(|(point, value)| PointSample { point, value })
                    .collect_vec()
            },
        );

        let n_columns_per_log_size = self.trees.as_ref().map(|tree| &tree.n_columns_per_log_size);

        let fri_answers = fri_answers(
            self.column_log_sizes(),
            samples,
            random_coeff,
            &query_positions_per_log_size,
            proof.queried_values,
            n_columns_per_log_size,
        )?;

        fri_verifier.decommit(fri_answers)?;

        Ok(())
    }

```

</details>

<details><summary><b> prover/src/core/fields/m31.rs </b></summary>

````rust

impl M31 {
    /// Returns `val % P` when `val` is in the range `[0, 2P)`.
    ///
    /// ```
    /// use stwo_prover::core::fields::m31::{M31, P};
    ///
    /// let val = 2 * P - 19;
    /// assert_eq!(M31::partial_reduce(val), M31::from(P - 19));
    /// ```
    pub fn partial_reduce(val: u32) -> Self {
        Self(val.checked_sub(P).unwrap_or(val))
    }

    /// Returns `val % P` when `val` is in the range `[0, P^2)`.
    ///
    /// ```
    /// use stwo_prover::core::fields::m31::{M31, P};
    ///
    /// let val = (P as u64).pow(2) - 19;
    /// assert_eq!(M31::reduce(val), M31::from(P - 19));
    /// ```
    pub const fn reduce(val: u64) -> Self {
        Self((((((val >> MODULUS_BITS) + val + 1) >> MODULUS_BITS) + val) & (P as u64)) as u32)
    }

    pub const fn from_u32_unchecked(arg: u32) -> Self {
        Self(arg)
    }

    pub fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "0 has no inverse");
        pow2147483645(*self)
    }
}

````

</details>

<details><summary><b> prover/src/core/fields/qm31.rs </b></summary>

```rust

impl QM31 {
    pub const fn from_u32_unchecked(a: u32, b: u32, c: u32, d: u32) -> Self {
        Self(
            CM31::from_u32_unchecked(a, b),
            CM31::from_u32_unchecked(c, d),
        )
    }

    pub const fn from_m31(a: M31, b: M31, c: M31, d: M31) -> Self {
        Self(CM31::from_m31(a, b), CM31::from_m31(c, d))
    }

    pub const fn from_m31_array(array: [M31; SECURE_EXTENSION_DEGREE]) -> Self {
        Self::from_m31(array[0], array[1], array[2], array[3])
    }

    pub const fn to_m31_array(self) -> [M31; SECURE_EXTENSION_DEGREE] {
        [self.0 .0, self.0 .1, self.1 .0, self.1 .1]
    }

    /// Returns the combined value, given the values of its composing base field polynomials at that
    /// point.
    pub fn from_partial_evals(evals: [Self; SECURE_EXTENSION_DEGREE]) -> Self {
        let mut res = evals[0];
        res += evals[1] * Self::from_u32_unchecked(0, 1, 0, 0);
        res += evals[2] * Self::from_u32_unchecked(0, 0, 1, 0);
        res += evals[3] * Self::from_u32_unchecked(0, 0, 0, 1);
        res
    }

    // Note: Adding this as a Mul impl drives rust insane, and it tries to infer Qm31*Qm31 as
    // QM31*CM31.
    pub fn mul_cm31(self, rhs: CM31) -> Self {
        Self(self.0 * rhs, self.1 * rhs)
    }
}

```

</details>

<details><summary><b> prover/src/core/fri.rs </b></summary>

```rust

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FriConfig {
    pub log_blowup_factor: u32,
    pub log_last_layer_degree_bound: u32,
    pub n_queries: usize,
    // TODO(andrew): fold_steps.
}

/// A FRI prover that applies the FRI protocol to prove a set of polynomials are of low degree.
pub struct FriProver<'a, B: FriOps + MerkleOps<MC::H>, MC: MerkleChannel> {
    config: FriConfig,
    first_layer: FriFirstLayerProver<'a, B, MC::H>,
    inner_layers: Vec<FriInnerLayerProver<B, MC::H>>,
    last_layer_poly: LinePoly,
}

impl<'a, B: FriOps + MerkleOps<MC::H>, MC: MerkleChannel> FriProver<'a, B, MC> {
    /// Commits to multiple circle polynomials.
    ///
    /// `columns` must be provided in descending order by size with at most one column per size.
    ///
    /// This is a batched commitment that handles multiple mixed-degree polynomials, each
    /// evaluated over domains of varying sizes. Instead of combining these evaluations into
    /// a single polynomial on a unified domain for commitment, this function commits to each
    /// polynomial on its respective domain. The evaluations are then efficiently merged in the
    /// FRI layer corresponding to the size of a polynomial's domain.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `columns` is empty or not sorted in descending order by domain size.
    /// * An evaluation is not from a sufficiently low degree circle polynomial.
    /// * An evaluation's domain is smaller than the last layer.
    /// * An evaluation's domain is not a canonic circle domain.
    #[instrument(skip_all)]
    pub fn commit(
        channel: &mut MC::C,
        config: FriConfig,
        columns: &'a [SecureEvaluation<B, BitReversedOrder>],
        twiddles: &TwiddleTree<B>,
    ) -> Self {
        assert!(!columns.is_empty(), "no columns");
        assert!(columns.iter().all(|e| e.domain.is_canonic()), "not canonic");
        assert!(
            columns.array_windows().all(|[a, b]| a.len() > b.len()),
            "column sizes not decreasing"
        );

        let first_layer = Self::commit_first_layer(channel, columns);
        let (inner_layers, last_layer_evaluation) =
            Self::commit_inner_layers(channel, config, columns, twiddles);
        let last_layer_poly = Self::commit_last_layer(channel, config, last_layer_evaluation);

        Self {
            config,
            first_layer,
            inner_layers,
            last_layer_poly,
        }
    }

    /// Returns a FRI proof and the query positions.
    ///
    /// Returned query positions are mapped by column commitment domain log size.
    pub fn decommit(self, channel: &mut MC::C) -> (FriProof<MC::H>, BTreeMap<u32, Vec<usize>>) {
        let max_column_log_size = self.first_layer.max_column_log_size();
        let queries = Queries::generate(channel, max_column_log_size, self.config.n_queries);
        let column_log_sizes = self.first_layer.column_log_sizes();
        let query_positions_by_log_size =
            get_query_positions_by_log_size(&queries, column_log_sizes);
        let proof = self.decommit_on_queries(&queries);
        (proof, query_positions_by_log_size)
    }

...

pub struct FriVerifier<MC: MerkleChannel> {
    config: FriConfig,
    // TODO(andrew): The first layer currently commits to all input polynomials. Consider allowing
    // flexibility to only commit to input polynomials on a per-log-size basis. This allows
    // flexibility for cases where committing to the first layer, for a specific log size, isn't
    // necessary. FRI would simply return more query positions for the "uncommitted" log sizes.
    first_layer: FriFirstLayerVerifier<MC::H>,
    inner_layers: Vec<FriInnerLayerVerifier<MC::H>>,
    last_layer_domain: LineDomain,
    last_layer_poly: LinePoly,
    /// The queries used for decommitment. Initialized when calling
    /// [`FriVerifier::sample_query_positions()`].
    queries: Option<Queries>,
}

impl<MC: MerkleChannel> FriVerifier<MC> {
    /// Verifies the commitment stage of FRI.
    ///
    /// `column_bounds` should be the committed circle polynomial degree bounds in descending order.
    ///
    /// # Errors
    ///
    /// An `Err` will be returned if:
    /// * The proof contains an invalid number of FRI layers.
    /// * The degree of the last layer polynomial is too high.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * There are no degree bounds.
    /// * The degree bounds are not sorted in descending order.
    /// * A degree bound is less than or equal to the last layer's degree bound.
    pub fn commit(
        channel: &mut MC::C,
        config: FriConfig,
        proof: FriProof<MC::H>,
        column_bounds: Vec<CirclePolyDegreeBound>,
    ) -> Result<Self, FriVerificationError> {
        assert!(column_bounds.is_sorted_by_key(|b| Reverse(*b)));

        MC::mix_root(channel, proof.first_layer.commitment);

        let max_column_bound = column_bounds[0];
        let column_commitment_domains = column_bounds
            .iter()
            .map(|bound| {
                let commitment_domain_log_size = bound.log_degree_bound + config.log_blowup_factor;
                CanonicCoset::new(commitment_domain_log_size).circle_domain()
            })
            .collect();

        let first_layer = FriFirstLayerVerifier {
            column_bounds,
            column_commitment_domains,
            proof: proof.first_layer,
            folding_alpha: channel.draw_felt(),
        };

        let mut inner_layers = Vec::new();
        let mut layer_bound = max_column_bound.fold_to_line();
        let mut layer_domain = LineDomain::new(Coset::half_odds(
            layer_bound.log_degree_bound + config.log_blowup_factor,
        ));

        for (layer_index, proof) in proof.inner_layers.into_iter().enumerate() {
            MC::mix_root(channel, proof.commitment);

            inner_layers.push(FriInnerLayerVerifier {
                degree_bound: layer_bound,
                domain: layer_domain,
                folding_alpha: channel.draw_felt(),
                layer_index,
                proof,
            });

            layer_bound = layer_bound
                .fold(FOLD_STEP)
                .ok_or(FriVerificationError::InvalidNumFriLayers)?;
            layer_domain = layer_domain.double();
        }

        if layer_bound.log_degree_bound != config.log_last_layer_degree_bound {
            return Err(FriVerificationError::InvalidNumFriLayers);
        }

        let last_layer_domain = layer_domain;
        let last_layer_poly = proof.last_layer_poly;

        if last_layer_poly.len() > (1 << config.log_last_layer_degree_bound) {
            return Err(FriVerificationError::LastLayerDegreeInvalid);
        }

        channel.mix_felts(&last_layer_poly);

        Ok(Self {
            config,
            first_layer,
            inner_layers,
            last_layer_domain,
            last_layer_poly,
            queries: None,
        })
    }

    /// Verifies the decommitment stage of FRI.
    ///
    /// The query evals need to be provided in the same order as their commitment.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The queries were not yet sampled.
    /// * The queries were sampled on the wrong domain size.
    /// * There aren't the same number of decommitted values as degree bounds.
    // TODO(andrew): Finish docs.
    pub fn decommit(
        mut self,
        first_layer_query_evals: ColumnVec<Vec<SecureField>>,
    ) -> Result<(), FriVerificationError> {
        let queries = self.queries.take().expect("queries not sampled");
        self.decommit_on_queries(&queries, first_layer_query_evals)
    }



```

</details>

<details><summary><b> prover/src/core/backend/simd/mod.rs </b></summary>

```rust

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct SimdBackend;

impl Backend for SimdBackend {}
impl BackendForChannel<Blake2sMerkleChannel> for SimdBackend {}
#[cfg(not(target_arch = "wasm32"))]
impl BackendForChannel<Poseidon252MerkleChannel> for SimdBackend {}

```

</details>

<details><summary><b> prover/src/examples/plonk/mod.rs </b></summary>

```rust

#[derive(Clone)]
pub struct PlonkEval {
    pub log_n_rows: u32,
    pub lookup_elements: PlonkLookupElements,
    pub claimed_sum: SecureField,
    pub base_trace_location: TreeSubspan,
    pub interaction_trace_location: TreeSubspan,
    pub constants_trace_location: TreeSubspan,
}

impl FrameworkEval for PlonkEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let a_wire = eval.get_preprocessed_column(Plonk::new("wire_a".to_string()).id());
        let b_wire = eval.get_preprocessed_column(Plonk::new("wire_b".to_string()).id());
        // Note: c_wire could also be implicit: (self.eval.point() - M31_CIRCLE_GEN.into_ef()).x.
        //   A constant column is easier though.
        let c_wire = eval.get_preprocessed_column(Plonk::new("wire_c".to_string()).id());
        let op = eval.get_preprocessed_column(Plonk::new("op".to_string()).id());

        let mult = eval.next_trace_mask();
        let a_val = eval.next_trace_mask();
        let b_val = eval.next_trace_mask();
        let c_val = eval.next_trace_mask();

        eval.add_constraint(
            c_val.clone() - op.clone() * (a_val.clone() + b_val.clone())
                + (E::F::one() - op) * a_val.clone() * b_val.clone(),
        );

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one(),
            &[a_wire, a_val],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one(),
            &[b_wire, b_val],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            (-mult).into(),
            &[c_wire, c_val],
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

```

</details>

<details><summary><b> prover/src/constraint_framework/logup.rs </b></summary>

```rust

/// Interaction elements for the logup protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupElements<const N: usize> {
    pub z: SecureField,
    pub alpha: SecureField,
    pub alpha_powers: [SecureField; N],
}
impl<const N: usize> LookupElements<N> {
    pub fn draw(channel: &mut impl Channel) -> Self {
        let [z, alpha] = channel.draw_felts(2).try_into().unwrap();
        let mut cur = SecureField::one();
        let alpha_powers = std::array::from_fn(|_| {
            let res = cur;
            cur *= alpha;
            res
        });
        Self {
            z,
            alpha,
            alpha_powers,
        }
    }
    pub fn combine<F: Clone, EF>(&self, values: &[F]) -> EF
    where
        EF: Clone + Zero + From<F> + From<SecureField> + Mul<F, Output = EF> + Sub<EF, Output = EF>,
    {
        assert!(
            self.alpha_powers.len() >= values.len(),
            "Not enough alpha powers to combine values"
        );
        values
            .iter()
            .zip(self.alpha_powers)
            .fold(EF::zero(), |acc, (value, power)| {
                acc + EF::from(power) * value.clone()
            })
            - EF::from(self.z)
    }

    pub fn dummy() -> Self {
        Self {
            z: SecureField::one(),
            alpha: SecureField::one(),
            alpha_powers: [SecureField::one(); N],
        }
    }
}

// SIMD backend generator for logup interaction trace.
pub struct LogupTraceGenerator {
    log_size: u32,
    /// Current allocated interaction columns.
    trace: Vec<SecureColumnByCoords<SimdBackend>>,
    /// Denominator expressions (z + sum_i alpha^i * x_i) being generated for the current lookup.
    denom: SecureColumn,
}
impl LogupTraceGenerator {
    pub fn new(log_size: u32) -> Self {
        let trace = vec![];
        let denom = SecureColumn::zeros(1 << log_size);
        Self {
            log_size,
            trace,
            denom,
        }
    }

    /// Allocate a new lookup column.
    pub fn new_col(&mut self) -> LogupColGenerator<'_> {
        let log_size = self.log_size;
        LogupColGenerator {
            gen: self,
            numerator: SecureColumnByCoords::<SimdBackend>::zeros(1 << log_size),
        }
    }

    /// Finalize the trace. Returns the trace and the total sum of the last column.
    /// The last column is shifted by the cumsum_shift.
    pub fn finalize_last(
        mut self,
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        SecureField,
    ) {
        let mut last_col_coords = self.trace.pop().unwrap().columns;

        // Compute cumsum_shift.
        let coordinate_sums = last_col_coords.each_ref().map(|c| {
            c.data
                .iter()
                .copied()
                .sum::<PackedBaseField>()
                .pointwise_sum()
        });
        let claimed_sum = SecureField::from_m31_array(coordinate_sums);
        let cumsum_shift = claimed_sum / BaseField::from_u32_unchecked(1 << self.log_size);
        let packed_cumsum_shift = PackedSecureField::broadcast(cumsum_shift);

        last_col_coords.iter_mut().enumerate().for_each(|(i, c)| {
            c.data
                .iter_mut()
                .for_each(|x| *x -= packed_cumsum_shift.into_packed_m31s()[i])
        });
        let coord_prefix_sum = last_col_coords.map(inclusive_prefix_sum);
        let secure_prefix_sum = SecureColumnByCoords {
            columns: coord_prefix_sum,
        };
        self.trace.push(secure_prefix_sum);
        let trace = self
            .trace
            .into_iter()
            .flat_map(|eval| {
                eval.columns.map(|col| {
                    CircleEvaluation::new(CanonicCoset::new(self.log_size).circle_domain(), col)
                })
            })
            .collect_vec();
        (trace, claimed_sum)
    }
}

```

</details>

</details>

[Stwo-dev](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/stwo-dev)

<div  align="center">
<img src="https://github.com/ETAAcademy/ETAAcademy-Images/blob/main/ETAAcademy-ZKmeme/56_stwo.gif?raw=true" width="50%" />
</div>
