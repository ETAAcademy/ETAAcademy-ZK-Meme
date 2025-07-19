# ETAAcademy-ZKMeme: 64. Lattice & NIST PQC

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>64. Lattice & NIST PQC</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Lattice & NIST PQC</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# NTT-Driven Lattice Cryptography: The Efficiency Behind Kyber, Dilithium, and Falcon

Modern lattice-based cryptography has become the mainstream technology for post-quantum security. Its development began in the 1990s with Ajtai’s SIS problem and the NTRU system, and advanced through theoretical breakthroughs such as LWE and its ring variants. The adoption of Number Theoretic Transform (NTT) techniques enabled efficient polynomial multiplication with $O(n \log n)$ complexity. As a result, leading schemes like Kyber (encryption), Dilithium (signatures), and Falcon (signatures) have been standardized by NIST. The security of these schemes is based on the hardness of the SIS and LWE lattice problems. By leveraging polynomial ring LWE constructions, they achieve batch encryption and high efficiency, and each scheme adopts a different NTT implementation strategy (full NTT, truncated NTT, or large modulus methods) to suit their parameters. For encryption, Kyber uses Ring-LWE over a polynomial ring and the Fujisaki-Okamoto (FO) transform to achieve CCA-secure key encapsulation. For signatures, Dilithium is built on Σ-protocols and the Fiat-Shamir transform to construct zero-knowledge proofs, compressing signature size through high/low bit decomposition and hint mechanisms. Falcon, based on the GPV framework and NTRU lattices, uses trapdoor Gaussian sampling to achieve even shorter signatures, but requires advanced techniques such as Rényi divergence to analyze security under aggressive parameters, with its unforgeability ultimately reduced to the hardness of the ring ISIS problem.

---

## 1. NTT in Lattice-Based Cryptography

Lattice-based schemes have become dominant in the NIST post-quantum cryptography standardization process, with three out of four final standards being lattice-based. The mathematical structures commonly used in lattice cryptography include standard lattices, ideal lattices, NTRU lattices, and module lattices. The core operation for standard lattices is matrix (or vector) multiplication over finite fields, while for ideal, module, and NTRU lattices, the core operation is polynomial multiplication in a polynomial ring such as $Z_q[x]/(\varphi(x))$. Leading schemes like Kyber, Dilithium, Saber, Falcon, and NTRU all rely on efficient polynomial multiplication.

The Fast Fourier Transform (FFT) and its number-theoretic variant, the Number Theoretic Transform (NTT), are the most efficient algorithms for polynomial multiplication, reducing the complexity from $O(n^2)$ to $O(n \log n)$. While FFT operates over the complex numbers and is fast, it suffers from floating-point errors, making it unsuitable for integer-coefficient polynomials. NTT, on the other hand, works over integers modulo a prime $p$, ensuring all operations are exact and making it ideal for lattice cryptography, where polynomial coefficients are integers.

NTT requires certain parameter conditions to be met, such as the length $n$ being a power of two and the modulus $q$ satisfying $q \equiv 1 \pmod{2n}$. Some schemes (e.g., Kyber, Saber, NTRU) do not fully meet these ideal NTT conditions and thus cannot use the "full-featured" NTT directly. In such cases, alternative algorithms like Karatsuba or Toom-Cook are used, which are less efficient. Therefore, extending NTT to "NTT-unfriendly" rings (where parameters are not ideal) is of great practical interest, as it can further improve the efficiency of polynomial multiplication.

### What is NTT?

The Number Theoretic Transform (NTT) is a variant of the FFT that operates under modular arithmetic, specifically designed for efficient multiplication of integer-coefficient polynomials. The core idea is to convert polynomial multiplication into pointwise multiplication in the value domain, and then use the inverse transform to recover the coefficient representation. NTT’s efficiency and suitability for hardware implementation make it the method of choice for polynomial multiplication in lattice-based cryptography. NIST PQC schemes such as Kyber, NTRU, and Saber make extensive use of NTT.

- **Polynomial Multiplication and Convolution:** At its core, polynomial multiplication is a convolution operation. Direct computation is $O(n^2)$, but NTT reduces this to $O(n \log n)$. There are three common types of polynomial multiplication: linear convolution (ordinary multiplication), cyclic convolution (modulo $x^n-1$), and negative wrapped convolution (NWC, modulo $x^n+1$), each corresponding to different polynomial ring structures.

- **Chinese Remainder Theorem (CRT):** Efficient NTT implementations often rely on CRT, which decomposes computations modulo a large modulus into several smaller moduli, enabling parallelism and efficient hardware implementation.

- **Radix-2 Fast NTT:** Similar to FFT, NTT can be implemented efficiently using radix-2 algorithms, greatly speeding up computations.

- **Hardware and Parameter Constraints:** NTT is not only fast but also hardware-friendly and resistant to side-channel attacks. Even in rings that are not NTT-friendly (i.e., where modulus and polynomial degree do not meet certain conditions), recent research has found ways to apply NTT techniques. Choosing the optimal NTT implementation for a given ring structure and parameter set is a key issue in practical deployment.

### Types of NTT

There are two main types of NTT:

- **Cyclic Convolution-based NTT (CC-based NTT):** Suitable for polynomial rings $Z_q[x]/(x^n-1)$. The parameters require $n$ to be a power of two and $q$ to be a prime such that $q \equiv 1 \pmod{n}$, ensuring the existence of a primitive $n$-th root of unity $\omega_n$ modulo $q$, and $n^{-1}$ is the modular inverse of $n$ modulo $q$.

  - **Forward NTT:** For a polynomial $a(x)$, the $j$-th coefficient after NTT is $a_j' = \sum_{i=0}^{n-1} a_i \omega_n^{ij} \mod q$, for $j = 0, 1, ..., n-1$.
  - **Inverse NTT:** Replace $\omega_n$ with its inverse and multiply by $n^{-1}$: $a_i = n^{-1} \sum_{j=0}^{n-1} a_j' \omega_n^{-ij} \mod q$.

- **Negative Wrapped Convolution-based NTT (NWC-based NTT):** Suitable for polynomial rings $Z_q[x]/(x^n+1)$, common in schemes like NTRU and Kyber. Here, $x^n = -1$, so higher powers of $x$ can be represented using lower powers and a negative sign. The parameters are stricter: $q \equiv 1 \pmod{2n}$, ensuring the existence of a primitive $2n$-th root of unity $\psi_{2n}$ modulo $q$. By introducing powers of $\psi$ for pre- and post-processing, the standard NTT formula can be applied to $x^n+1$ rings.
  - **Preprocessing:** Multiply each coefficient $a_i$ by $\psi^i$.
  - **Standard NTT:** Apply the usual NTT formula to the preprocessed polynomial.
  - **Postprocessing:** After the inverse NTT, multiply each coefficient by $\psi^{-i}$ to recover the original ring.

### Tricks for Efficient Polynomial Multiplication

- **Karatsuba Algorithm:** A divide-and-conquer method for efficient polynomial (or large integer) multiplication. For two polynomials $(a + bx)$ and $(c + dx)$, the naive method requires four multiplications, but Karatsuba reduces this to three, at the cost of extra additions and subtractions. Recursively applied, it reduces complexity from $O(n^2)$ to $O(n^{1.58})$.

- **Good’s Trick:** For polynomial rings $Z_q[x]/(x^{h \cdot 2^k} - 1)$ with odd $h$, this method decomposes a large ring into two smaller rings, enabling parallelism and NTT acceleration. The polynomial coefficients are arranged in an $h \times 2^k$ matrix, NTT is applied to each row, polynomial multiplication to each column, and then the inverse NTT and CRT are used to map results back.

- **Schönhage’s Trick:** For $Z_q[x]/(x^{2mn} - 1)$, this method maps large polynomial multiplication to NTT-friendly smaller rings using bivariate polynomials, enabling NTT acceleration.

- **Nussbaumer’s Trick:** Similar to Schönhage’s, but for $Z_q[x]/(x^{2mn}+1)$, again mapping to NTT-friendly structures.

- **Radix-2 Fast NTT:** The most efficient NTT implementation, using recursive divide-and-conquer and butterfly operations. For $n$ a power of two, the polynomial can be repeatedly split in half, ideal for recursion. FFT and CRT techniques can further decompose large polynomial multiplications into smaller, NTT-friendly ones. Two main algorithms are:
  - **Cooley-Tukey Algorithm (CT):** Commonly used for forward NTT, with input in natural order and output in bit-reversed order. The butterfly operation is $A = a + \omega b \mod q$, $B = a - \omega b \mod q$.
  - **Gentleman-Sande Algorithm (GS):** Commonly used for inverse NTT, with input in bit-reversed order and output in natural order. The butterfly operation is $a = (A + B) \mod q$, $b = \omega^{-1}(A - B) \mod q$.

---

### Types of Polynomial Rings and NTT Adaptations in Lattice Cryptography

In lattice-based cryptography, three main types of polynomial rings are commonly used: $Z_q[x]/(\varphi(x))$, where $\varphi(x)$ is typically $x^n \pm 1$ with $n$ a power of two (the most common and NTT-friendly structure), $n$ not a power of two, or $\varphi(x)$ being a general polynomial of degree $n$.

**Traditional NTT requirements** dictate that $n$ should be a power of two and $q$ a prime such that $q \equiv 1 \pmod{n}$ (or $2n$), ensuring the existence of a primitive root of unity and efficient NTT implementation. However, in practice, parameters do not always meet these ideal conditions (as in Kyber, Saber, and other lattice schemes), necessitating alternative approaches to extend NTT applicability. The essence of these methods is a “divide-map-merge” strategy: through clever mathematical transformations, the original NTT problem is broken down into several smaller, NTT-friendly subproblems, which are then recombined to yield the final result. There are three main classes of such relaxation techniques:

#### Incomplete FFT Trick

When $q$ is an NTT-friendly prime (of the form $q = q' \cdot 2^e + 1$) but does not satisfy $q \equiv 1 \pmod{2n}$, the FFT decomposition tree does not need to be fully expanded to the linear term; it can be “truncated” earlier. As long as $q \equiv 1 \pmod{2n/2^\beta}$ (where $\beta$ is the truncation level), a truncated NTT can be performed. This relaxes the requirement on $q$ and broadens the range of applicable parameters, enabling efficient implementations in schemes like Kyber and Dilithium.

#### Splitting Polynomial Ring

This approach decomposes a large ring $Z_q[x]/(x^n \pm 1)$ into several smaller rings (e.g., $Z_q[y]$ / $(y^{n/2^α} \pm 1)$ ), allowing NTT to be performed in each smaller ring. This is well-suited for hardware and parallel implementations, with variants such as Pt-NTT, K-NTT, and H-NTT differing in the details of splitting and multiplication.

- **Splitting:** Group polynomial coefficients by variable, forming a bivariate polynomial.
- **NTT in Small Rings:** Perform NTT/inverse NTT in each small ring.
- **Merging:** Use inverse transforms to recombine results into the original ring.

#### Large Modulus Techniques

When $q$ is not an NTT-friendly prime (or even not a prime at all), several strategies can be used:

- **NTT-Friendly Large Prime:** Choose a large modulus $N$ that is itself NTT-friendly.
- **Residue Number System (RNS):** Let $N$ be the product of several NTT-friendly primes; perform NTT in each small modulus and recombine results using the Chinese Remainder Theorem (CRT).
- **Composite Modulus Ring:** Perform NTT directly in a composite modulus ring, suitable for certain special cases. This approach is the most general but requires more storage and computational resources.

  - Select a large modulus $N$ (with $N > n \cdot q^2$), which is either an NTT-friendly prime or a product of such primes.
  - Perform NTT in $Z_N[x]/(x^n \pm 1)$, then reduce modulo $q$ to recover the result in the original ring.

### NTT Implementations in Leading Lattice Schemes

Efficient polynomial multiplication is critical in all major NIST PQC lattice schemes, directly impacting the speed and security of encryption, decryption, and signature operations. However, each scheme’s parameters and ring structures differ, requiring tailored NTT implementations:

- **Kyber:** $n = 256$, $q = 7681$ (Round 1) or $q = 3329$ (Rounds 2/3), polynomial ring $Z_q[x]/(x^n+1)$. Hardware implementations (e.g., AVX2) may use different NTT variants depending on the platform.

  - _Round 1:_ $q \equiv 1 \pmod{2n}$, allowing full-featured NWC-based NTT.
  - _Rounds 2/3:_ $q \equiv 1 \pmod{n}$ but not $2n$; full NTT is not possible, so truncated NTT or ring-splitting methods are used.

- **Dilithium:** $n = 256$, $q = 8380417$, $Z_q[x]/(x^n+1)$.

  - $q \equiv 1 \pmod{2n}$, enabling direct use of standard NWC-based NTT for both forward and inverse transforms.

- **Falcon:** $n = 512/1024$, $q = 12289$, $Z_q[x]/(x^n+1)$.

  - $q \equiv 1 \pmod{2n}$, so standard NTT can be used, similar to Dilithium.

- **Saber:** $n = 256$, $q = 8192$ (a power of two), $Z_q[x]/(x^n+1)$.

  - $q$ is not a prime and does not meet NTT requirements, so NTT cannot be used directly.
  - _Implementation:_ Early versions used Toom-Cook and Karatsuba algorithms; later versions use large modulus techniques (e.g., RNS, composite modulus) to indirectly enable NTT.

- **NTRU:** $n$ is a prime (e.g., 509, 677, 821), $q$ is a power of two, $Z_q[x]/(x^n-1)$.

  - Both $q$ and $n$ are not NTT-friendly, so direct NTT is not possible.
  - _Method 1:_ Extend $n$ to the nearest power of two $n'$, perform NTT in $Z_N[x]/(x^{n'}-1)$, then reduce modulo $x^n-1$.
  - _Method 2:_ Use Good’s trick to decompose the problem into several small NTTs and polynomial multiplications.

- **NTRU Prime:** Both $n$ and $q$ are primes, $Z_q[x]/(x^n-x-1)$.
  - The structure is more general, making NTT implementation even more challenging.
  - _Implementation:_ Either extend to a larger ring and use large modulus NTT, or use advanced tricks (Good’s, Schönhage’s, Nussbaumer’s) to map the problem to an NTT-friendly structure.

#### Key Points

- **Full-featured NTT** is suitable for ideal parameters (where $n$ is a power of two and $q$ is NTT-friendly), using NWC-based or CC-based NTT for maximum efficiency (e.g., Dilithium, Falcon).
- **Truncated NTT or ring-splitting methods** are used for non-ideal parameters (e.g., $n$ is a power of two but $q$ is not NTT-friendly), as in Kyber Rounds 2/3 and H-NTT.
- **Large modulus techniques** are used when parameters are far from ideal (e.g., $n$ is not a power of two, or $q$ is not NTT-friendly), as in Saber, NTRU, and NTRU Prime, at the cost of higher resource consumption.
- **Advanced tricks** (Good’s, Schönhage’s, Nussbaumer’s) further enhance flexibility and efficiency for more general polynomial rings.

These diverse NTT adaptation strategies ensure that efficient polynomial multiplication remains feasible across the wide range of parameters encountered in modern lattice-based cryptography.

---

## 2. Lattice-Based Encryption Schemes

In cryptography, lattices modulo $q$ are commonly used, defined as $\Lambda_q^\perp(A) = \{ v \in \mathbb{Z}^m : Av \equiv 0 \pmod{q} \}$, where $A$ is an $n \times m$ matrix. Two foundational hard problems underpinning lattice-based cryptography are the Short Integer Solution (SIS) problem and the Learning With Errors (LWE) problem.

- **SIS Problem:** Given a random matrix $A$, the goal is to find a nonzero vector $z$ such that $Az \equiv 0 \pmod{q}$ and $\|z\|_\infty \leq \beta$.
- **LWE Problem:** Given samples of the form $(A, As + e)$ and $(A, u)$, distinguish between the two distributions. Here, $s$ is a secret vector, $e$ is a noise vector, and $u$ is a random vector. The presence of noise $e$ makes it infeasible to recover $s$ using linear algebra. The security of LWE is reducible to worst-case lattice problems, making it a strong foundation for modern lattice-based cryptography.

**Note:** SIS algorithms can be used to attack LWE; if one can find short vectors $r_1, r_2$ such that $r_1^T A + r_2^T = 0$, they can distinguish LWE samples from random ones. The security of LWE depends on the parameters $q, \beta, m, n$, especially the ratio $q/\beta$.

### LWE Encryption Scheme

- **Key Generation:**

  - Secret key $s$ is sampled from $[\beta]^m$, i.e., each entry is randomly chosen from $\{-\beta, ..., 0, ..., \beta\}$. All operations are performed in $\mathbb{Z}_q$. The sampling distribution (uniform, binomial, or Gaussian) is chosen based on security and efficiency requirements.
  - Public key: $A \in \mathbb{Z}_q^{n \times m}$ is a random matrix, and $t = As + e_1$, where $e_1$ is a noise vector sampled from $[\beta]^n$.

- **Encryption:** Choose random vectors $r, e_2 \in [\beta]^m$ and $e_3 \in [\beta]$. Compute $u^T = r^T A + e_2^T$ and $v = r^T t + e_3 + \lceil q/2 \rceil \cdot \mu$, where $\mu$ is the message bit. The ciphertext is $(u, v)$.

- **Decryption:** Compute $v - u^T s$, which yields $q/2 \cdot \mu$ plus a small error. By checking whether the result is closer to 0 or $q/2$, the original bit $\mu$ can be recovered. In basic LWE encryption, $\mu$ is typically a single bit (0 or 1), but it can be generalized to larger values in $\mathbb{Z}_q$.

### Lattice Reduction Algorithms

Lattice reduction algorithms are used to find short vectors in high-dimensional lattices and are the main tools for attacking LWE and SIS-based schemes. The most notable are the LLL (Lenstra–Lenstra–Lovász) and BKZ (Block Korkine-Zolotarev) algorithms. The **Hermite factor** $\delta$ (often written as $\delta_0$ or $\delta$) describes the relationship between the length of the shortest vector found and the lattice determinant. Formally, for a lattice of dimension $d$ and determinant $\det(\Lambda)$, the shortest vector found is approximately $\delta^d \cdot (\det(\Lambda))^{1/d}$. The closer $\delta$ is to 1, the stronger the algorithm and the lower the security; in practice, $\delta$ is typically between 1.01 and 1.005.

### From LWE to Ring-LWE

Traditional LWE encryption is inefficient, as encrypting a single bit results in large ciphertexts and public keys. **Ring-LWE** addresses this by replacing numbers with polynomials, allowing batch processing of multiple bits per operation and greatly improving efficiency and space utilization. The core idea is to generalize operations from $Z_q$ to the polynomial ring $Z_q[X]/(f(X))$, where elements are polynomials $a(X) = a_0 + a_1 X + \cdots + a_{d-1} X^{d-1}$ with coefficients in $\mathbb{Z}_q$, and $f(X)$ is typically $X^d \pm 1$ (as in Kyber, NTRU, Dilithium, etc.). Polynomials can be viewed as vectors, and polynomial multiplication can be represented as matrix-vector multiplication, allowing the Ring-LWE problem to be reduced to a high-dimensional lattice problem. The security analysis is similar to standard LWE.

Compared to standard LWE, Ring-LWE encryption can encrypt $d$ bits at once (where $d$ is the polynomial degree), and both public key and ciphertext sizes are significantly reduced, resulting in much higher efficiency.

- **Ring-LWE:** Given $(A, t = As + e)$ in a polynomial ring, the problem is to distinguish it from a random distribution.
- **Module-LWE:** A generalization of Ring-LWE, supporting more flexible parameters and structures (used in many NIST standard schemes).
- **NTRU:** The earliest efficient lattice-based encryption scheme using polynomial rings. It constructs a "trapdoor one-way function" in the polynomial ring, with security similar to Ring-LWE. NTRU key recovery and decryption leverage the invertibility of polynomials and modular arithmetic.

### Efficient Polynomial Multiplication

In Ring-LWE schemes (such as Kyber), the most computationally intensive operation is polynomial multiplication. The naive algorithm has complexity $O(d^2)$, while Karatsuba and Toom-Cook algorithms can reduce this to $O(d^{1.5})$. The **Number Theoretic Transform (NTT)**, which is the finite field analogue of FFT, reduces the complexity to $O(d \log d)$, greatly improving efficiency. NTT is applicable to polynomial rings $\mathbb{Z}_q[X]/(X^d+\alpha)$ when $q$ has a primitive $2d$-th root of unity (i.e., $q \equiv 1 \pmod{2d}$). Schemes like Kyber choose $q$ and $d$ (e.g., $q=3329, d=256$) specifically to enable efficient NTT-based multiplication.

Kyber’s polynomial ring is $\mathbb{Z}_{3329}[X]/(X^{256}+1)$, which supports efficient NTT. The public key matrix $A$, secret key $s$, noise $e$, and all multiplications are performed using NTT for acceleration. For further optimization, $A$ can be generated and stored directly in NTT form, avoiding repeated transformations. Kyber supports different security levels (512/768/1024), with parameters such as $k, \eta_1, \eta_2, d_u, d_v$ controlling security and key/ciphertext sizes. Kyber uses **binomial distribution** for noise sampling, which is both efficient and secure. Both ciphertexts and public keys are compressed (only high bits are kept) to further reduce size. During decryption, as long as the total error (from noise and compression) is less than $q/4$, the plaintext can be correctly recovered. Kyber’s parameters ensure an extremely low decryption error probability.

### Security: CPA and CCA

- **CPA Security (Chosen Plaintext Attack):** The attacker only sees public keys and ciphertexts and cannot distinguish whether a key is random.
- **CCA Security (Chosen Ciphertext Attack):** The attacker can also access a decryption oracle but still cannot distinguish the key. CCA security is stronger and required for real-world deployments (e.g., TLS).

Kyber is inherently CPA-secure. By applying the **Fujisaki-Okamoto (FO) transform**, it can be upgraded to a CCA-secure Key Encapsulation Mechanism (KEM). The FO transform ensures that the decryption oracle cannot be abused: during decapsulation, the ciphertext is re-encrypted and compared, preventing oracle misuse. The CCA security of Kyber’s KEM ultimately relies on the CPA security of the underlying encryption and the correct implementation of the FO transform. As long as the CPA encryption is secure, the hash function is secure, and the FO process is correctly implemented, the KEM is CCA-secure.

- **Encapsulation:** Randomly select a message $m$, encrypt it with the public key to obtain ciphertext $c$, and use a hash function $H(m, pk)$ to derive the shared key $K$ and encryption randomness $\rho$.
- **Decapsulation:** Use the secret key to decrypt $c$ and recover $m'$, then recompute $K'$ and $\rho'$ using $H(m', pk)$, re-encrypt $m'$ with the public key and $\rho'$ to get $c'$. Only if $c = c'$ is $K'$ output; otherwise, output $\perp$ (or a pseudorandom key). This ensures that even with access to the decapsulation oracle, an attacker cannot distinguish the key or attack the protocol.

### Engineering Optimizations in Kyber

- **Public Key Hash Preprocessing:** Kyber’s public key is large (about 1KB), and hashing it is time-consuming. In practice, the hash can be precomputed and stored, so only 32 bytes need to be hashed during decapsulation, improving efficiency.
- **Pseudorandom Key on Decapsulation Failure:** The standard FO transform requires outputting $\perp$ on failure, but Kyber outputs a pseudorandom key (derived from the ciphertext and secret key hash) to prevent side-channel attacks and information leakage.
- **NTT Optimization:** All polynomial operations are accelerated using NTT, greatly improving performance.

---

## Dilithium: From Σ-Protocols to Lattice-Based Digital Signatures

A **Σ-protocol** is a three-step interactive zero-knowledge proof (ZKPoK) that can be used to prove knowledge of a secret $s$ such that $As + e = t$. By applying the **Fiat-Shamir transform**, this interactive protocol can be converted into a non-interactive signature scheme, where the challenge $c$ is generated via a hash function. This approach is analogous to the Schnorr signature in the discrete logarithm setting, but here the underlying hard problem is based on lattice assumptions such as LWE or SIS. Thus, lattice-based digital signatures are essentially non-interactive zero-knowledge proofs of knowledge (NIZKPoK).

### Basic Structure

- **Public Key:** $A, t = As_1 + s_2$
- **Secret Key:** $s_1, s_2$ (with small coefficients)
- The goal is to prove knowledge of $s_1, s_2$ such that $As_1 + s_2 = t$, with both $s_1$ and $s_2$ having small coefficients.

Directly proving knowledge of small $s_1, s_2$ such that $As_1 + s_2 = t$ is much harder on lattices than in the discrete log setting, mainly due to the need to simultaneously prove range constraints. The solution is to **relax the conditions**: it suffices to prove knowledge of $s_1, s_2$ with slightly larger coefficients (e.g., in $[\bar{\beta}]$), and a small $\bar{c}$, such that $A\bar{s}_1 + \bar{s}_2 = \bar{c}$. If one can construct such a tuple, the problem reduces to the Ring-LWE or Ring-SIS problem, ensuring the signature scheme’s security is still based on hard lattice problems.

In the Σ-protocol, the challenge $c$ affects the distribution of the response $z$, which in turn impacts the security and efficiency of the signature. Each challenge $c$ is essentially a sparse $\pm1$ polynomial, with only $\eta$ nonzero entries, ensuring the challenge space $C$ is large ($2^\eta \cdot \binom{d}{\eta} > 2^{256}$), which provides security while keeping the response $z$ compact for efficiency. This design is widely adopted in leading lattice-based signature schemes such as Dilithium.

### Basic Σ-Protocol Flow

- **Secret Key:** $s_1 \in [\beta]^m, s_2 \in [\beta]^n$
- **Public Key:** $A \in R_{q,f}^{n \times m}, t = As_1 + s_2$
- **Interactive Steps:**
  1. **Commitment:** The prover samples $y_1, y_2 \in [\gamma+\bar{\beta}]$, computes $w = Ay_1 + y_2$, and sends $w$.
  2. **Challenge:** The verifier randomly selects $c \in C$ (the challenge space).
  3. **Response:** The prover computes $z_1 = cs_1 + y_1, z_2 = cs_2 + y_2$.
  4. **Rejection Sampling:** To ensure the response $(z_1, z_2)$ is independent of the secret, if any coefficient of $z_1, z_2$ exceeds $[\bar{\beta}]$, output $\perp$ (abort); otherwise, send $(z_1, z_2)$. The prover repeats until a valid response is found.
  5. **Verification:** The verifier checks the ranges of $z_1, z_2$ and that $Az_1 + z_2 - ct = w$. As long as the protocol does not abort, the distribution of $(z_1, z_2)$ is independent of the secret. One can also simulate a real conversation by randomly sampling $(z_1, z_2) \in [\bar{\beta}]$, a random $c$, and setting $w = Az_1 + z_2 - ct$.

**Proof of Knowledge:** If an attacker can answer two different challenges $c, c'$ for the same $w$, the secret can be extracted. Specifically, from $Az_1 + z_2 - ct = Az_1' + z_2' - c't$, it follows that $A(z_1 - z_1') + (z_2 - z_2') = (c - c')t$, which is a relaxed version of the secret equation. Thus, forging a signature or proof reduces to solving a hard lattice problem (Ring-LWE/SIS), ensuring security.

The choice of $\bar{\beta}$ must balance the probability of rejection sampling (not too high for efficiency, not too low for security). Typically, $\bar{\beta} \approx \gamma d(m+n)$, where $\gamma$ depends on the sparsity $\eta$ of the challenge and $\beta$. A larger $\bar{\beta}$ reduces rejection but weakens security; a smaller $\bar{\beta}$ increases security but reduces efficiency.

### Signature Compression: Bit-Dropping and Hints

Protocols like Dilithium further reduce signature size through **bit-dropping (high/low decomposition)**, removing $z_2$ and/or $w$, and careful analysis of correctness and zero-knowledge properties. In the original protocol (e.g., Fiat-Shamirized Schnorr/lattice protocols), the prover must send $z_1, z_2$ (or $w$), resulting in long signatures. The question is: can we send only $z_1$ (or $z$) and a short hash (e.g., $ρ = H(w)$ ), omitting $z_2$ or $w$ to **significantly reduce signature length**?

**High/Low Decomposition (HIGHS/LOWS):** Each ring element $w \in Z_q$ is split into a high part (HIGHS($w$) in a set $S$ of size $2^\kappa$) and a low part (LOWS($w$) = $w$ - HIGHS($w$) in $[q/2^{\kappa+1}]$). Only the high part is kept, discarding the low part (bit-dropping), greatly reducing data size. In the protocol, $w = \text{HIGHS}(Ay)$, and only the high part is sent; verification only checks the high bits. For $Az - ct = Ay - cs_2$, as long as LOWS($Ay - cs_2$) is within range, HIGHS($Ay$) = HIGHS($Ay - cs_2$), so verification only needs to check HIGHS($Az - ct$) = $w$, without needing $z_2$.

**Further Compression of Public Key and Signature:** The public key is usually a large matrix $A$ and vector $t = As_1 + s_2$, with each coefficient requiring $\log q$ bits, making the key large. To compress, only the high part of $t$ ( $t_1 = HIGHT(t)$ ) is published, discarding the low part ( $t_0 = LOWT(t)$ ). Thus, the public key size drops from $nd \cdot \log q$ to $nd \cdot \ell$ (with $\ell \ll \log q$). The challenge is that verification needs $t$, but only $t_1$ is available. The key equation is $Az - ct = Az - c(t_1 + t_0) = (Az - ct_1) - ct_0$. As long as $ct_0$ is small, HIGHS($Az - ct$) = HIGHS($Az - ct_1$). A **hint mechanism** is used: if $ct_0 \in [\delta_S]^n$, HIGHS($Az - ct_1$) can differ from HIGHS($Az - ct$) by at most one interval. The prover provides a 1-bit hint per coefficient, indicating "left" or "right." The verifier uses the hint and $Az - ct_1$ to recover HIGHS($Az - ct$). Modern lattice-based signatures (e.g., CRYSTALS-Dilithium, ML-DSA) essentially convert these zero-knowledge protocols (with high/low decomposition and hints) into non-interactive digital signatures via the Fiat-Shamir transform.

### Signature and Verification Flow

- **Signing:**

  - Randomly sample $y$, compute the high part $w = \text{HIGHS}(Ay)$
  - Compute challenge $c = H(w, \mu, A, t)$ ($\mu$ is the message digest, $A, t$ are the public key)
  - Compute $z = cs_1 + y$
  - Check if $z$, LOWS($Ay - cs_2$), and $ct_0$ are in range; otherwise, resample
  - Compute hint $h = \text{HINT}(Az - ct_1, ct_0)$
  - Output signature $(z, h, c)$

- **Verification:**
  - Check $z \in [\bar{\beta}]^m$
  - Use $h$ and $Az - ct_1$ to recover $w' = \text{USEHINT}(Az - ct_1, h)$
  - Check $H(w', \mu, A, t) = c$

#### Example Parameters and Their Meaning (Dilithium Level 3)

To illustrate the structure and parameter choices in a modern lattice-based signature scheme, consider Dilithium Level 3 as an example:

- **Ring Parameters:** The scheme operates over the ring $R_{q,f}$, where the modulus polynomial is $f(X) = X^{256} + 1$ and the modulus $q = 2^{23} - 2^{13} + 1$. This choice of $q$ is NTT-friendly, enabling efficient polynomial multiplication.
- **Secret Vectors:** The secret keys $s_1$ and $s_2$ are vectors with coefficients in the range $[-\beta, \beta]$, where $\beta = 4$.
- **Public Key:** The public key matrix $A$ is deterministically generated from a 256-bit seed $\rho$, and the public key vector is $t_1 = \text{HIGH}(As_1 + s_2)$, where only the high bits are retained for compression.
- **High/Low Bit Sets:**
  - Set $S$: The set of high bits for certain protocol values, consisting of 16 points with spacing of $(q-1)/16$.
  - Set $T$: The set of high bits for public key compression, consisting of $2^{10}$ points with spacing of $2^{13}$.
- **Challenge Space $C$:** The challenge polynomials are sparse, with 49 coefficients set to $\pm1$ and the remaining 207 set to 0.
- **Other Parameters:**
  - $\gamma = 196$
  - $\bar{\beta} = 2^{19} - \gamma - 1$
  - $\delta_S = (q-1)/32 - 1$
- **Public Key Size:** 1952 bytes, consisting of the 256-bit seed and $6 \times 256 \times 10$ bits for $t_1$.
- **Signature Size:** Approximately 3424 bytes, including $z_1$ (256 coefficients, each 5 polynomials of 20 bits), the challenge $c$ (256 bits), and the hint $h$ (256 coefficients, each 6 bits).

This parameter set demonstrates how Dilithium achieves a balance between security, efficiency, and compactness, leveraging NTT-friendly rings, compressed public keys, and carefully chosen challenge spaces.

---

## 3. Falcon: A Compact Lattice-Based Signature Scheme

Falcon is a NIST PQC standard signature scheme built upon the GPV framework and NTRU lattices. Its main advantage is its extremely small signature size—significantly shorter than Dilithium—though this comes at the cost of more complex security proofs. The GPV (Gentry-Peikert-Vaikuntanathan, 2008) framework is a general paradigm for lattice-based signatures, whose core idea is to use a lattice trapdoor to efficiently generate short vectors (signatures). By combining full-domain hash (FDH) and Gaussian sampling, the signature distribution is made close to the ideal, preventing information leakage. Falcon’s security relies on the ability of the trapdoor Gaussian sampler to generate sufficiently random short vectors.

The original GPV security proof uses statistical distance to measure how close the signature distribution is to the ideal. However, for Falcon’s practical parameters, the statistical distance (e.g., 2⁻³⁴) is much larger than the theoretical requirement (which should be close to zero), making the original GPV proof inapplicable. To address this, Falcon’s security analysis adopts Rényi divergence—a more relaxed measure of distribution closeness—allowing for a broader range of parameters and a more practical security analysis.

### The GPV Framework

The GPV framework is the theoretical foundation for lattice-based signatures. Its core is the **lattice trapdoor**: constructing a lattice (such as an SIS or NTRU lattice) with a "good basis" (the trapdoor), which enables efficient sampling of short lattice vectors.

- **Gaussian Sampling:** Using the trapdoor, given a "target point," one can efficiently sample a short lattice vector from a discrete Gaussian distribution, ensuring the signature distribution is close to ideal and preventing information leakage.
- **Full-Domain Hash (FDH):** A hash function maps the message to a "target point" in the lattice; the signature is a short vector (sampled using the trapdoor) that maps to this target point under a linear relation.
- The security of the signature can be tightly reduced to hard lattice problems (such as SIS or LWE), providing theoretical quantum resistance.

**In summary:**

- The public key is a "trapdoor function" $f$, and the secret key is its inverse.
- **Signing:** For a message $m$, hash it to $y = H(m)$, then use the trapdoor to sample a short preimage $f^{-1}(y)$ as the signature.
- **Verification:** Check that $f(\sigma) = H(m)$ and that $\sigma$ is sufficiently short.

### Falcon’s Construction

Falcon applies the GPV framework to NTRU lattices and uses a Gaussian sampler to generate short vectors, resulting in highly compact signatures. Thanks to the NTRU ring structure, the public key is just a single polynomial, greatly reducing size and speeding up operations. During signing, if the sampled vector is not short enough, the process is repeated (on average, 1–2 times). To prevent information leakage, Falcon introduces a random salt $r$ for each signature; Falcon+ recommends using a new salt for every resampling, which also aids in security proofs. Falcon’s parameters are chosen aggressively to minimize signature size, which complicates theoretical security proofs.

#### Key Generation (Gen)

- Use TpdGen to generate a trapdoor basis $(f, g, F, G)$.
- Construct the NTRU lattice basis $B$.
- Compute the public key $h = g \cdot f^{-1} \mod q$.
- Output $(\text{sk} = B, \text{pk} = h)$.

#### Signing (Sgn+, CoreFalcon+)

- Repeat:
  - Randomly sample a salt $r$.
  - Compute $c = H(\text{pk}, r, m)$.
  - Use the trapdoor basis $B$ and Gaussian sampler PreSmp to sample $(s_1, s_2)$, a short preimage of $c$.
  - If $\|(s_1, s_2)\|_2 \leq \beta$, output the signature $\sigma = (r, s_2)$.
  - Otherwise, repeat.

#### Verification (Ver)

- Compute $c = H(\text{pk}, r, m)$.
- Compute $s_1 = c - s_2 \cdot h \mod q$.
- Check that $\|(s_1, s_2)\|_2 \leq \beta$.

### Security: UF-CMA and Reduction to R-ISIS

**UF-CMA Security:** An attacker $A$ who has seen up to $Q_s$ signatures should not be able to forge a valid signature on a new message. If $A$ can forge, it can be used to solve the (QH+1)-R-ISIS (multi-target ring ISIS) problem.

- **Original UF-CMA Security:** $A$ interacts with a signing oracle and a hash oracle, and finally outputs $(m*, σ*)$. If the signature verifies and $m^*$ was not previously queried, $A$ wins.
- **Limiting Resampling:** If the sampling success rate is low and many resamplings are needed, security degrades. If the total number of samplings exceeds $C_s$, the oracle aborts. If $p$ is the probability of successful sampling, the security loss is bounded by a function of $Q_s, C_s, p$.
- **Preventing Salt Collisions:** The salt $r$ is chosen to be sufficiently long (e.g., 320 bits), making collisions extremely unlikely. If the same $(\text{pk}, r, m)$ is queried multiple times, the oracle aborts.
- **Simulating the Hash Oracle:** The ideal (uniform) distribution is replaced by the actual distribution (Gaussian sampling + NTRU structure), and Rényi divergence is used to quantify the difference. The hash oracle no longer returns a uniformly random $c$, but one generated via Gaussian sampling. The security loss is controlled by the Rényi divergence.
- **Ideal Sampler:** The signing oracle samples directly from the ideal Gaussian distribution, and the Rényi divergence again quantifies the difference from the actual sampler.

**Reduction to R-ISIS:** To prove security, a reduction constructs an attacker $B$ that uses $A$ to solve the (QH+1)-R-ISIS problem. $B$ embeds its own ISIS challenge into the hash oracle queries. If $A$ forges a signature, it must have used one of $B$'s challenge points, allowing $B$ to solve the ISIS instance. Thus, if $A$ can forge with non-negligible probability, so can $B$, contradicting the assumed hardness of R-ISIS.

**In summary:** Falcon’s UF-CMA security is bounded by the sum of sampling loss, Rényi divergence loss, salt collision loss, and the hardness of the R-ISIS problem. This reflects the delicate balance between aggressive parameter optimization for compactness and the need for rigorous security proofs in modern lattice-based signatures.
