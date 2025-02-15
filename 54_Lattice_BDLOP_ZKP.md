# ETAAcademy-ZKMeme: 54. Lattice & BDLOP Based ZKP

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>54. Lattice & BDLOP ZKP</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Lattice & BDLOP ZKP</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Zero-Knowledge Proofs Based on Lattice Cryptography and BDLOP Commitment Scheme

Lattice-based cryptography and the BDLOP commitment scheme are at the forefront of quantum-resistant cryptographic protocols. By utilizing the hardness of lattice problems such as SIS and LWE, BDLOP offers an efficient, secure, and compact solution for quantum-resistant zero-knowledge proofs and related cryptographic applications.

---

Lattice cryptography is based on the computational difficulty of finding short non-zero vectors in random Euclidean lattices. The core problems in lattice cryptography include the Short Integer Solution (SIS) problem and the Learning With Errors (LWE) problem, both of which provide robust security foundations for cryptographic protocols. In this context, a "lattice" refers to a structure in an _n_-dimensional Euclidean space $R^n$, where the points of the lattice are integer linear combinations of a set of basis vectors $B = (\vec{b_1}, \dots, \vec{b_n}) \in \mathbb{R}^{n \times n}$, and the lattice points are of the form $a_1 \vec{b_1} + \dots + a_n \vec{b_n} \in \mathbb{R}^n$, with $a_1, \dots, a_n \in \mathbb{Z}$.

In the past, many early lattice-based cryptographic schemes were proven insecure because the underlying random lattice distributions were vulnerable to attacks. However, by using structured variants of SIS and LWE, such as **Module-SIS** and **Module-LWE**, where the matrix is not a simple integer matrix but instead is an extension over a ring, more efficient computations can be achieved. While classical cryptographic schemes, such as RSA and elliptic curve cryptography (ECC), can be broken by Shor’s algorithm in polynomial time due to their reliance on discrete logarithm and integer factorization problems, lattice-based cryptography is considered resistant to quantum computing. Therefore, it plays a crucial role in quantum-resistant cryptographic schemes in the post-quantum era.

Moreover, by utilizing ring structures and number-theoretic transforms (NTT) to compute polynomial coefficients, encryption schemes like **NTRU** offer significantly faster encryption and decryption speeds compared to traditional elliptic curve-based systems. These efficiencies make them highly valuable in zero-knowledge proof (ZKP) protocols and encryption schemes, particularly in those that do not rely on complex hash functions but instead leverage the hardness of lattice problems to ensure privacy.

#### **The Short Integer Solution (SIS) Problem**

The SIS problem involves a matrix $A$, and the challenge is to find a short, non-zero solution vector $s$ such that $A \cdot s = 0 \mod q$, where $q$ is a modulus. When the matrix rows are randomly selected, this problem becomes computationally difficult. The SIS problem is easier when the required vector length ($\beta$) is relatively small compared to the modulus $q$.

#### **The Learning With Errors (LWE) Problem**

LWE is a generalization of the SIS problem, where noise is added to the lattice problem to increase its difficulty. LWE is more complex than SIS and typically solved by searching for the kernel of $s$. LWE is a critical building block in lattice-based cryptography and offers strong security guarantees for encryption and proof systems.

#### **BDLOP Commitment Scheme**

The **BDLOP (Baum et al.)** commitment scheme is an additive homomorphic commitment scheme widely used in quantum-resistant zero-knowledge proofs (ZKPs). The BDLOP commitment scheme ensures **statistical binding** and **hiding**, and it is constructed based on the **MLWE** (Matrix Learning With Errors) and **MSIS** (Minimum Short Independent Vectors) problems. The BDLOP scheme is designed to provide smaller commitment sizes and lower **robustness errors**, addressing the shortcomings of earlier schemes.

Prior to BDLOP, many cryptographic schemes were based on protocols using the ** syndrome decoding problem** (a generalization of SIS). However, these protocols suffered from large robustness errors, leading to large proof sizes and inefficiencies. Later, researchers introduced improvements by incorporating **Learning Parity with Noise (LPN)** and **Ring-LWE** problems, but these schemes still exhibited significant error probabilities. The BDLOP scheme, however, overcame these challenges and became the leading lattice-based commitment scheme.

BDLOP is particularly effective in quantum-resistant applications, providing strong security while maintaining efficient proof sizes. It supports applications such as **set membership proofs**, **verifiable decryption**, and **zero-knowledge proofs for short solutions to linear equations**.

#### **Advantages of BDLOP and Lattice-Based Zero-Knowledge Proofs**

Lattice-based cryptographic schemes like BDLOP offer several advantages:

1. **Quantum Resistance**: Unlike classical cryptographic schemes based on factorization or discrete logarithms, lattice-based systems are believed to be secure against quantum attacks, which makes them ideal for post-quantum cryptography.

2. **Efficiency**: The use of **NTT** (Number-Theoretic Transform) in lattice-based schemes such as NTRU and BDLOP significantly enhances the speed of encryption and decryption operations. This makes them highly efficient for large-scale applications, especially in zero-knowledge proofs, where computational efficiency is critical.

3. **Compact Proofs**: The BDLOP commitment scheme provides compact proofs that scale well, making it suitable for scenarios where proof size is a key concern, such as privacy-preserving protocols and secure authentication systems.

4. **Robust Security**: The **robustness error** in BDLOP is significantly lower than in earlier schemes, ensuring stronger security guarantees for the commitment and proof processes.

---

## 1. Core Concepts

In lattice-based cryptographic protocols and zero-knowledge proofs (ZKPs), **polynomial rings** (such as $R$ and $R_q$) are fundamental structures used to handle polynomial operations and vector calculations. For example, the polynomial rings $R = Z[X] / (X^d + 1)$ and $R_q = Z_q[X] / (X^d + 1)$, where $d$ is a power of 2, are used in lattice cryptography. Here, $q$ is an odd prime number, and $Z$ represents the set of integers, while $Z_q = Z / qZ$ is the ring of integers modulo $q$, where elements are in the range $[-\frac{q-1}{2}, \frac{q-1}{2}]$. Vectors and matrices are represented as $v \in Z_q^m$ and $A \in Z_q^{m \times n}$, respectively. A polynomial $a = \sum_{i=0}^{d-1} a_i X^i \in R_q$ can be represented by its coefficient vector $\vec{a} = (a_0, a_1, ..., a_{d-1})^T$.

**Toeplitz matrices** are often used in $R_q$ to efficiently represent and compute polynomial coefficients, optimizing the computational process. A Toeplitz matrix transforms polynomial coefficients into a matrix form, where multiplying the matrix by a vector is equivalent to the polynomial operation. **Norms** (such as $l_{\infty}$ and $l_p$) are used to quantify the size of vectors, much like conventional vector norms. These norms help measure error and noise, contributing to the security and efficiency of algorithms. The **automorphism group** (including automorphisms like $\sigma_i(X) = X^i$) is used to optimize mappings and symmetry, improving the computational efficiency of encryption and proof systems. Together, these concepts form the foundation for building efficient and secure lattice-based cryptographic protocols.

**Number-Theoretic Transform (NTT)** improves computational efficiency by mapping polynomials in $R_q$ to smaller modular rings defined by $l$ moduli, $(X^{d/l} - \zeta^{2i-1})$, using the Chinese Remainder Theorem to ensure the mapping is invertible. Specifically, given a polynomial $f$, its NTT representation is $\hat{f} = (\hat{f_1}, \hat{f_2}, ..., \hat{f_l})$, where each $\hat{f_i}$ is the remainder of $f$ modulo the corresponding factor. NTT has additivity, multiplicativity, and invertibility, meaning that the inverse transformation can recover the original polynomial: $NTT^{-1}(NTT(f)) = f$. This property is essential for encryption and decryption processes. Additionally, NTT provides a way to verify consistency by ensuring that the sum of the NTT coefficients equals $d \times f_0$, where $f_0$ is the constant term of the polynomial:

$$
\sum_{i=0}^{d-1} \hat{f_i} = d \cdot f_0
$$

In lattice-based encryption and quantum-resistant protocols, NTT plays a crucial role in optimizing computations and enhancing efficiency.

#### Challenge Space

In lattice-based zero-knowledge proofs, the **challenge space** refers to the set of challenges (elements) from which the verifier selects, using it to inquire about a commitment or a specific calculation from the prover. The prover then responds with information that can validate the truth of certain claims while maintaining privacy. In ZKPs, the verifier typically selects a challenge from a set where the difference between any two elements is invertible. However, in the ring $R_q$, not all elements have an inverse, and the existence of an inverse is related to the way the ring is decomposed.

Lyubashevsky et al. proposed a method to relax the conditions for invertibility, defining the challenge space as a set of ternary polynomials $C = \{-1, 0, 1\}^d$, where the coefficients follow a specific probability distribution: the probability of 0 is $1/2$, and the probabilities of $\pm 1$ are each $1/4$. For example, if an element $c$ is randomly selected from the challenge space $C$, the distribution of its coefficients modulo $(X - \zeta)$ is close to uniform, and the maximum probability for coefficients in $\mathbb{Z}_q$ is limited to $\epsilon$. When $d = 128$ and $q \approx 2^{32}$, $\epsilon$ is calculated as $2^{-31.44}$. If $c$ is an element in the partially decomposed ring $R_q$, and $c \mod (X - \zeta^{2i-1}) = 0$, then $c$ is non-invertible. Therefore, the difference between any two challenges $\bar{c} = c - c'$ is non-invertible with a probability no greater than $\epsilon^{d/l}$, where $l$ is the decomposition factor of the ring. This method ensures that most challenges selected are invertible, and the probability of their differences being non-invertible is constrained. This approach avoids the complexity of uniformly sampling from a large set and ensures security effectively.

#### Error Distributions and Their Impact on Security

Error distributions play a crucial role in the security of lattice-based encryption protocols. Early encryption schemes often required errors to be sampled from discrete Gaussian distributions, but many practical schemes have since adopted centered binomial distributions or uniform distributions over small intervals. The discrete Gaussian distribution is normalized from a continuous normal distribution over the real numbers and is defined as:

$$
D_{a,\sigma}(x) = \frac{\rho_{a,\sigma}(x)}{\rho_\sigma(\mathbb{Z})}
$$

where $\rho_{a,\sigma}(x)$ is the probability density of the normal distribution, and $\rho_\sigma(\mathbb{Z})$ is the normalization constant over the integer set $\mathbb{Z}$. Sampling from a discrete Gaussian distribution is typically denoted as $x \leftarrow D_{a,\sigma}$. If other distributions are used, they are either specified in advance or represented by the symbol $\chi$.

Lattice-based zero-knowledge proofs employ **rejection sampling** to hide the randomness of secret commitments $\mathbf{r}$. Rejection sampling involves introducing a combination of masking vectors and challenge vectors to ensure no information about the randomness is leaked.

1. **Uniform Distribution Rejection Sampling**: The prover samples a masking vector $\mathbf{y}$ from the interval $[-\delta + 1, \delta]$, and then computes the masked vector $\mathbf{z} = \mathbf{y} + c\mathbf{r}$ based on the challenge $c$. If $|\vec{z}|_∞$ is less than a threshold (e.g., $|\vec{z}|_∞ < \delta - \beta$), the prover proceeds; otherwise, the process is rejected and restarted. The expected retry count $M$ is related to $\delta$ and $\beta$ and is approximately given by:

$$
\frac{1}{M} = \frac{(2(\delta - \beta) - 1)}{(2\delta - 1)}^{\kappa d} \approx e^{-\kappa d \beta / \delta}
$$

2. **Discrete Gaussian Distribution Rejection Sampling**: When the masking vector $\mathbf{y}$ is sampled from a discrete Gaussian distribution, two types of rejection sampling methods exist: $Rej_0$ and $Rej_1$. In $Rej_0$, a condition is checked:

$$
u > \frac{1}{M} \exp\left( -2 \langle \vec{z}, c\vec{r} \rangle + \frac{\| c\vec{r} \|^2_2}{2\sigma^2} \right)
$$

If this condition is not satisfied, the process is rejected. $Rej_1$ adds an additional step to check whether the inner product $\langle \mathbf{z}, c\mathbf{r} \rangle$ is negative. If either condition fails, the prover rejects and restarts. By adjusting parameters, the expected retry count can be controlled.

---

## 2. Lattice-Based Cryptography

The security of lattice-based cryptographic protocols relies on the computational complexity of problems related to **Learning with Errors (LWE)** and **Short Integer Solution (SIS)**, such as **Ring-LWE (RLWE)**, **Module-LWE (MLWE)**, and **Module-SIS (MSIS)**. These problems involve finding short integer solutions in high-dimensional lattice spaces, which are considered computationally difficult in both classical and quantum computing. The difficulty of these problems makes breaking the cryptographic schemes nearly impossible, providing strong security guarantees for lattice-based protocols. Additionally, because these problems are resistant to quantum algorithms, lattice-based cryptographic schemes are considered a promising candidate for quantum-safe encryption.

The **BDLOP scheme** (an extension of the traditional SIS commitment scheme) combines high-dimensional random vectors $\mathbf{r}$ with message vectors $\mathbf{s}$ to generate a commitment. Specifically, the commitment is generated by first committing to a part related to the random vector, $t_0 = B_0 \mathbf{r}$, and then hiding the message using a second part, $t_1 = B_1 \mathbf{r} + \mathbf{s}$. Due to the indistinguishability of $t_0$ and $t_1$ under the **Module-LWE** assumption, the scheme ensures both **hiding** and **binding** properties. The binding property guarantees that, for different message vectors $\mathbf{s_0}$, different random vectors $\mathbf{r_0}$ are required, which leads to a solution of **Module-SIS**, ensuring that the commitment cannot be forged without solving this difficult problem.

For example, in a variant of the BDLOP commitment scheme, let $B_0 \in R^{\mu \times (\mu + \lambda + \epsilon)}_q$ be a matrix, and let $\vec{b}_1, \dots, \vec{b}_ε \in R_q^{\mu + \lambda + \epsilon}$
be vectors and $\vec{r} \leftarrow \chi^{(\mu + \lambda + \epsilon)d}$ where $\chi$ is a discrete Gaussian distribution. For a polynomial vector $m \in R^\epsilon_q$, its commitment is a tuple $(\vec{t}_0, t_1, \dots, t_ε)$, where $\vec{t}_0 = B_0 \vec{r}$, and for each $i = 1, \dots, \epsilon$, $t_i = \langle \vec{b}_i, \vec{r} \rangle + m_i$. This commitment scheme is both **computationally binding** and **hiding**, with binding derived from the difficulty of **$MSIS_μ$** and hiding from the difficulty of **$MLWE_λ$**. In essence, without solving these hard problems, one cannot find the committed message or alter the committed value.

The **BGV Encryption Scheme** (Brakerski-Gentry-Vaikuntanathan) is a lattice-based encryption scheme that ensures encryption security through noise and the hardness of the RLWE problem, while also supporting **additive homomorphic operations**. The key steps in the BGV encryption scheme are as follows:

1. **Encryption Process:**
   The scheme is based on a plaintext space (ring $R_p$) and a noise distribution $\chi$. Encryption uses the public key and secret key to encrypt the message. The public key, $pk$, consists of a matrix $A = (\frac{a\ p \0}{b\ 0\ p})$,
   where $a$ and $b$ are generated by multiplying random vectors with the noise distribution $\chi$. When encrypting a message $m$ with a random vector $\vec{r} = (r_1, r_2, r_3) \leftarrow \chi^{3d}$, the ciphertext is computed as: $\vec{c} = \vec{m} + A\vec{r} = \frac{a r_1 + p r_2}{b r_1 + p r_3 + m}$
2. **Decryption Process:**
   During decryption, the ciphertext is dot-multiplied with the secret key, then reduced modulo $q$ and $p$, resulting in the original message: $[[\langle \mathbf{c}, \mathbf{sk} \rangle]_q ]_p = m + p (r_1 e + r_3 - s r_2) \mod p = m$
3. **Security:**
   The security of the BGV scheme is based on the difficulty of the **$RLWE_{\chi}$** problem, which ensures **CPA (Chosen Plaintext Attack) security**. This means that breaking the ciphertext is very difficult even with knowledge of the public key.

4. **Homomorphic Properties:**
   The BGV scheme is **additively homomorphic**, meaning that two encrypted messages can be added together to obtain a new valid ciphertext. If the noise in the ciphertext is below a certain threshold, the decryption of the sum will correctly yield the sum of the original messages.

5. **Correctness:**
   The BGV encryption scheme is **$\tau$-correct**, meaning that the sum of multiple honestly generated ciphertexts will, with very high probability, decrypt to the correct sum of the original messages.

---

## 3. Zero-Knowledge Proofs Based on Commitment Schemes in Lattice Cryptography

A **Zero-Knowledge Proof (ZKP)** is a two-party protocol where the prover demonstrates to the verifier that they know the solution to a problem, without revealing any information beyond the solution itself. In lattice cryptography, zero-knowledge proofs are commonly used in encryption signatures and privacy-preserving protocols. By converting a zero-knowledge proof into a signature scheme (such as the **Fiat-Shamir transformation**), the verifier can confirm that the prover knows a secret key without disclosing the key itself.

However, lattice cryptography presents specific challenges when implementing zero-knowledge proofs. For example, the mathematical structure of lattice-based systems makes certain elements non-invertible, and the required property of "short vectors" does not always hold under arithmetic operations. This can make classic zero-knowledge proof schemes inefficient, often requiring multiple repetitions to ensure a low error probability. However, by leveraging the unique mathematical structure of lattice cryptography, the proof systems can be made more efficient, avoiding the need for repeated verification while maintaining low computational and memory demands. These optimizations make lattice-based zero-knowledge proofs more suitable for privacy-preserving and encryption protocols compared to other schemes based on hash functions.

#### Lattice-Based Zero-Knowledge Proof Systems

Lattice-based zero-knowledge proof systems utilize **linear random functions** (e.g., $f : \mathbb{R}_q^n \to \mathbb{R}_q^m$, where $A$ is a random matrix, $\mathbf{s}$ is a secret vector, and $\mathbf{t}$ is public) to implement efficient proof protocols. These systems incorporate techniques such as **approximate preimages** and **rejection sampling** to ensure zero-knowledge properties. Linear random functions in lattice cryptography are based on the hardness of **Module-LWE (Module Learning with Errors)** and **Module-SIS (Module Short Integer Solution)** problems, making them collision-resistant one-way functions. In this setting, the prover can demonstrate knowledge of a solution without revealing any additional information.

An **approximate preimage** refers to the situation where, for some vector $\mathbf{s_0}$, it is an approximate preimage of the function $f$ such that $A\mathbf{s_0} = c\mathbf{t}$, where $c$ is a short polynomial. Unlike traditional **SIS (Short Integer Solution)** problems, the preimage solutions here are longer, but they remain difficult to find for appropriately chosen parameters. In essence, this protocol is similar to the **Schnorr protocol** for proving knowledge, where the prover demonstrates they know an approximate preimage.

The protocol proceeds as follows:

1. **Commitment:** The prover sends the verifier a commitment vector $\mathbf{w} = A\mathbf{y}$, where $\mathbf{y}$ is a vector that is longer than $\mathbf{s}$ but still relatively short.
2. **Challenge and Response:** The verifier sends a short challenge polynomial $c$, and the prover computes $\mathbf{z} = \mathbf{y} + c\mathbf{s}$. Using **rejection sampling**, the prover ensures that the distribution of $\mathbf{z}$ does not leak any information about $\mathbf{s}$.
3. **Verification:** The verifier checks that $A\mathbf{z} = \mathbf{w} + c\mathbf{t}$ to ensure that $\mathbf{z}$ satisfies the corresponding verification equation. If the check fails, the protocol restarts.

This protocol guarantees both **soundness** and **correctness**. Additionally, it can be transformed into a **non-interactive signature scheme** using the **Fiat-Shamir transformation**, making it suitable for real-world applications. Signature schemes based on this protocol, such as **Dilithium**, have been nominated as candidates for the **NIST Post-Quantum Cryptography (PQC) standardization**.

#### Extending Zero-Knowledge Proofs for More Complex Functions

To prove more complex functions, methods for proving **extended linear relationships** and **multiplicative relations** can be employed. These extended proof methods can handle more complicated cryptographic tasks, such as **group signatures** and **ring signatures**, while also dealing with more intricate Boolean and arithmetic circuits, which can be represented and proven using **R1CS (Rank-One Constraint Systems)**.

---

### 3.1 Two Types of Zero-Knowledge Proof Protocols: Multiplicative and Linear Relations

Two fundamental types of zero-knowledge proof protocols—**Multiplicative Relations** and **Linear Relations**—are essential in lattice-based cryptographic systems. Here we explore these two types of protocols, along with the corresponding commitment schemes and optimizations.

#### 1) Multiplicative Relation Zero-Knowledge Proof

The **Multiplicative Relation Zero-Knowledge Proof** is designed to prove the relationship $m_3 = m_1 \cdot m_2$, where $m_1$, $m_2$, and $m_3$ are messages related through multiplication. Using a commitment scheme combined with the **Fiat-Shamir transformation**, this protocol can be converted into a non-interactive zero-knowledge proof (NIZK).

The size of the proof is given by the formula:

$$
\text{Proof Size} = d \cdot \lceil \log q \rceil + k(\lambda + \mu + 4) \cdot d \cdot \lceil \log q \rceil + 256 \text{ bits}
$$

where $d$ is the dimension, $\mu$ and $\lambda$ are the ranks of the **MSIS** and **MLWE** modules, and $k$ is a parameter used to enhance knowledge soundness. In cases involving multiple message triples, the proof size increases linearly, without introducing additional "garbage" terms.

The protocol is efficient in that it scales linearly with the number of message triples, making it more practical for larger systems. This approach avoids the overhead associated with non-linear growth in proof size, ensuring that the proof remains manageable in computationally demanding applications.

#### 2) Linear Relation Zero-Knowledge Proof

The **Linear Relation Zero-Knowledge Proof** is used to prove a linear relationship of the form $A \mathbf{s} = \mathbf{u}$, where $\mathbf{s}$ is a secret vector, and $A$ and $\mathbf{u}$ are public quantities. This proof leverages **NTT (Number Theoretic Transform)** techniques to demonstrate the **ternary nature** of the vector $\mathbf{s}$, i.e., $s \cdot (1-s) = 0$, and shows how challenges and inner product computations can be used to verify the linear relationship.

After the challenge phase, where the verifier sends a challenge vector $\mathbf{\gamma} \in \mathbb{Z}_q^m$, the prover demonstrates the following equation:

$$
\langle A \vec{s} - \vec{u}, \vec{\gamma} \rangle = \langle A \vec{s}, \vec{\gamma} \rangle - \langle \vec{u}, \vec{\gamma} \rangle = \langle \vec{s}, A^{T} \vec{\gamma} \rangle - \langle \vec{u}, \vec{\gamma} \rangle
$$

Through the application of **NTT transformations**, the prover calculates a constant term $f_0 = 0$. Once the challenge is computed, the proof is validated by the verifier, and the process ensures that the prover has demonstrated knowledge of $\mathbf{s}$ without revealing it directly.

The proof size for this linear relationship is given by:

$$
(\mu + n/d + 4) \cdot d \cdot \lceil \log q \rceil + k(\lambda + \mu + n/d + 3) \cdot d \cdot \lceil \log q \rceil
$$

This size does not include the commitment portion of the protocol. The efficiency of this proof arises from the use of **NTT** and the combination of **Fiat-Shamir transformation**, which allows the protocol to be converted into a non-interactive zero-knowledge proof, suitable for various cryptographic applications.

#### Commitment Scheme: BDLOP in the Linear Size Proof System

The commitment scheme employed in the **Linear Size Proof System** is the **BDLOP scheme**. Although the commitment size in BDLOP is linearly related to the message size, this does not pose a problem in the linear size proof system. In fact, the absence of length restrictions allows for greater optimization in real-world applications. These optimizations compensate for the increase in commitment size, resulting in a more efficient proof process.

For instance, in some multi-round protocols, the prover can submit the NTT coefficients gradually, rather than submitting the entire message at once. This incremental submission approach leads to a more efficient proof process, as it reduces the overall computational cost of proving the relationship. Thus, the combination of **NTT bases** and **BDLOP commitments** allows for the efficient proof of linear and multiplicative relationships in lattice-based cryptography.

#### Improvements and Optimizations in BDLOP Commitment Scheme

The opening proof process in BDLOP has been further improved to enhance its robustness and reduce costs. Initially, the opening proof required that the difference between challenge polynomials ($\bar{c} = c - c_0$) be invertible, which led to high proof costs. However, recent analysis indicates that the invertibility condition can be relaxed. Instead of requiring an invertible challenge difference, the protocol can operate by restricting the probability that certain NTT coefficients are non-zero.

When the success probability exceeds a certain threshold, the prover can recover the message by handling the NTT coefficients one by one, simplifying the proof process significantly.

Additionally, to further improve security and reduce costs, **automorphism** techniques have been introduced. Automorphisms, which involve transformations of the challenge polynomial within the ring $R_q$, are used to generate multiple images of the challenge. This technique forces the prover to provide correct responses across various transformations, making it more difficult for the prover to forge a valid proof.

Each time the protocol is repeated, the challenge polynomial $c$ undergoes a transformation by the automorphism group of the ring $R_q$, resulting in different versions of the challenge. As the NTT coefficients are "rotated," the prover must demonstrate knowledge across these varied coefficients, thereby lowering the chance of successful forgery.

---

### 3.2 Linear Size Proof Systems

In cryptographic protocols, **zero-knowledge proofs** (ZKPs) play a crucial role in verifying that a party possesses specific knowledge without revealing the knowledge itself. One of the essential challenges in ZKPs is efficiently proving relationships between cryptographic elements, especially when dealing with large-scale data or complex structures. In this article, we discuss a **linear-size proof system** for efficiently proving **multiplicative** and **linear relationships**, leveraging optimizations such as **masked openings**, **automorphisms**, and **NTT (Number Theoretic Transform)**.

#### Efficient Multiplicative Proof Construction with Masked Opening

A new approach to **product relation proofs** is presented, where the added cost of the proof is minimal—about 1 KB—regardless of the number of polynomial messages involved. The core idea is to use a fixed-form random vector mask (denoted as _masked opening_), where the masked opening is represented as:

$$
\vec{z} = \vec{y} + c \vec{r}
$$

Here, $\vec{y}$ is a known vector, $c$ is a challenge scalar, and $\vec{r}$ is a random value. The prover uses this mask to demonstrate that they can construct a valid masked version of the message vector, and prove this relationship in the context of the multiplicative relation.

This masked opening approach is used in quadratic equations, where the relationship to be proven appears in the main coefficient, while lower-order terms are referred to as "garbage terms". Consider the quadratic equation:

$$
\langle \vec{\alpha}, \vec{f} \circ \vec{f} + c \vec{f} \rangle - g_0 - c g_1 = \langle \vec{\alpha}, (B_1\vec{y}) \circ (B_1\vec{y}) \rangle - g_0 + c \left( \langle \vec{\alpha}, B_1 \vec{y} - 2(B_1\vec{y}) \circ \vec{s} \rangle - g_1 \right) + c^2 \langle \vec{\alpha}, \vec{s} \circ \vec{s} - \vec{s} \rangle = 0
$$

where $\circ$ represents pointwise polynomial multiplication, and $\vec{\alpha}$ is a random challenge vector. Here, $g_0$ and $g_1$ are the "garbage" polynomials. Before seeing the challenge $c$, the prover commits to these garbage polynomials, ensuring that the equation holds for all challenges, provided the message satisfies the product relation.

If $s_i s_i = s_i$ holds for some $i$, the main term $\langle \vec{\alpha}, \vec{s} \circ \vec{s} - \vec{s} \rangle$ has a probability of $(1 - 1/q)$ in $\alpha$, and for this term, the prover cannot use any garbage polynomials to compensate. Therefore, the equation only holds when the quadratic polynomial has a root under a random challenge $c$, with a probability of $2/q$. This provides a product relationship proof with a reliability of $3/q$.

#### Self-Automorphism and Masked Opening in Product Proofs

By combining **self-automorphism** and masked openings, we can construct a more robust quadratic equation to prove the product relation:

$$
\sum_{i=0}^{k-1} \langle \alpha_i, \sigma^{-i} (f^{(i)} \circ f^{(i)}) + \sigma_i(c) \vec{f^{(i)}} \rangle - g_0 - c g_1 = 0
$$

This equation proves the product relationship, with the reliability error of $(3/q)^k$, assuming that the $k$ challenge vectors $\vec{\gamma_0}, \dots, \vec{\gamma}_{k-1}$ are uniformly random. It's important to note that the equation still only involves two garbage polynomials. In the non-interactive variant of the protocol, one of the garbage terms is essentially free, allowing the proof of any number of quadratic product relationships between polynomials with only a commitment to a single garbage polynomial. These optimizations significantly reduce the cost of the proof and improve the protocol’s efficiency.

#### Efficient Linear Relation Proof System

To efficiently prove a **linear relationship** of the form $A \vec{s} = \vec{u}$, where $\vec{s}$ and $\vec{u}$ are committed NTT vectors, we employ a clever use of NTT transformations and self-automorphisms. The key idea is to transform the linear equation into a scalar product form, which allows efficient computation and verification.

The basic equation is:

$$
\langle \vec{\gamma}, A \vec{s} - \vec{u} \rangle = \langle A^T \vec{\gamma}, \vec{s} \rangle - \langle \vec{\gamma}, \vec{u} \rangle = 0
$$

where $\vec{\gamma}$ is a uniformly random challenge vector. If $A \vec{s} \neq \vec{u}$, the probability that the scalar product is zero is $1/q$.

Additionally, the product of two polynomials corresponds to the pointwise multiplication of their NTT vectors. Furthermore, the constant term $f_0 = f(0)$ of a polynomial is equal to the sum of its NTT coefficients, scaled appropriately. These facts allow us to compute the scalar product of the NTT vectors by calculating the polynomial product and reading the result from the constant term of the final polynomial.

To compute the scalar product, let $\vec{s} = \text{NTT}(\mathbf{s})$ and $\vec{\gamma} = \text{NTT}(\gamma)$, then:

$$
\langle \vec{\gamma}, \vec{s} \rangle = \sum_{i=0}^{d-1} \gamma_i s_i = \sum_{i=0}^{d-1} \gamma(\zeta_i) s(\zeta_i) = \sum_{i=0}^{d-1} (\gamma s)(\zeta_i) = \frac{1}{d} \sum_{i=0}^{d-1} f(\zeta_i) = f(0)
$$

where $f$ is the polynomial $d\gamma \mathbf{s}$, and $f(0)$ is its constant term.

The prover then needs to demonstrate that the constant term of $f = \text{NTT}^{-1}(dA^T \gamma) \mathbf{s} - \langle \vec{\gamma}, \vec{u} \rangle$ is zero. Given the commitment $t = \langle \vec{b}, \vec{r} \rangle + \mathbf{s}$, the verifier can compute the commitment to $f$ as:

$$
\tau = \text{NTT}^{-1}(dA^T \vec{\gamma}) t - \langle \vec{u}, \vec{\gamma} \rangle = \text{NTT}^{-1}(dA^T \vec{\gamma}) \langle \vec{b}, \vec{r} \rangle + f
$$

To prove that $f(0) = 0$, the prover commits to a uniformly random polynomial $g$, ensuring that $g(0) = 0$. The prover then sends $t' = \langle \vec{b'}, \vec{r} \rangle + g$ and uses this polynomial to mask the other coefficients of $f$. The verifier checks that $h(0) = f(0) + g(0) = 0$. If $A \vec{s} \neq \vec{u}$, the probability that $f(0) = \langle \gamma, A \vec{s} - \vec{u} \rangle$ is uniformly random for a random $\gamma$, making $h(0)$ equal to zero with probability $1/q$, which is the protocol's reliability error.

#### Improving Reliability with Self-Automorphisms

Finally, to further enhance the protocol’s reliability, we use **self-automorphisms**. Consider $k$ uniformly random vectors $\vec{\gamma_0}, \dots, \vec{\gamma_{k-1}}$, and define:

$$
f_i = \text{NTT}^{-1}(dA^T \gamma_i) \mathbf{s} - \langle \vec{\gamma}_i, \vec{u} \rangle
$$

The constant term of $f_i$ is $\langle \vec{γ_i}, A \vec{s} - \vec{u} \rangle$. By constructing a mapping $L_i: R_q → R_q$ (using trace maps to a subring of $R_q$), we ensure that when $A \vec{s} \neq \vec{u}$, the first $k$ coefficients of $f = L_0(f_0) + \dots + L_{k-1}(f_{k-1})$ are zero. This boosts the protocol’s reliability error to $1/q^k$ while requiring only a commitment to one mask polynomial $g$ and sending a single polynomial $h$ to prove the product relation.

---

### 3.3 Optimized Linear-Size Proof Systems

**Linear-Size Proof Systems**: In these systems, the size of the proof is linearly related to the size of the witness (the proof object), and they are commonly applied in areas such as **integer relations proofs**, **group signatures**, and **ring signatures**. Proof sizes in these systems typically range between 10KB and 100KB. Optimized **linear proof systems** are designed to handle very large witnesses (e.g., on the order of 10GB), where proof sizes can scale up to 10MB. While these systems come with higher computational and memory demands, they offer significantly improved proof efficiency.

Optimized linear-size proof systems are particularly useful for proving relationships among the preimages of multiple collision-resistant hashes. The goal is to prove relationships of the form $\mathbf{u}_i = A\mathbf{s}_i$, where each hash value corresponds to an m-bit value. The proof size is intended to be linearly related to the number of hashes $n$, such that the size of the proof remains efficient even as the number of hashes increases. To achieve this, a new protocol is proposed, where the proof size is proportional to the square root of the witness size.

#### Protocol Design

The protocol uses a variant of an **amortized approximate opening proof**. In this setup, the prover sends a masked approximate opening vector $\mathbf{z} = \mathbf{y} + x_1 \mathbf{s}_1 + \cdots + x_n \mathbf{s}_n$, where $x_i \in \mathbb{Z}_q$ are integer challenges. The protocol then constructs a polynomial $f(x_1, \dots, x_n)$ to prove that each $\mathbf{s}_i$ vector is binary, i.e., each element in the vector is from the set $\{0, 1\}$.

However, there are two key challenges in this approach. First, due to the presence of many "garbage terms" (a quadratic number of terms), directly committing to these terms would lead to excessive costs, making it impossible to meet the sublinear size requirement. Second, proving that the structure of $\mathbf{z}$ remains consistent and independent of the challenge vector $\mathbf{x}$ is a non-trivial task. To solve these problems, the protocol employs a multi-round interactive technique, similar to the **Schwartz-Zippel Lemma**, where the coefficients of each polynomial are committed to in successive rounds. This helps prevent an explosive growth of garbage terms and ensures the reliability of the protocol.

#### Ensuring Consistency of $\mathbf{z}$

The second challenge in the protocol is ensuring that the form of $\mathbf{z}$ remains consistent with expectations throughout the process. To address this, the protocol combines with a binary proof system to ensure that each $\mathbf{s}_i$ is a binary vector. Furthermore, a Merkle tree construction is introduced as an extension to prove the hash preimage relationships. In this extended protocol, the prover recursively traverses each level of the tree to prove that the vector corresponding to each hash value is valid. The proof expands across all levels of the tree, establishing how each hash value is related to the previous level's hash value, thereby creating a valid proof structure across the entire tree.

#### Efficiency and Security Considerations

This optimized linear-size proof system is able to handle multiple hash relationships and provides competitive proof sizes over a broader range of parameters. Specifically, in the context of hash tree constructions, the system achieves efficient proofs by recursively proving hash relationships at each level of the tree.

Additionally, current lattice-based cryptographic proof techniques such as **noise flooding** and **rejection sampling** use statistical analysis to ensure that the protocol transcript does not leak information about the message or the random numbers involved. These methods rely on extensive repeated computations and intricate mechanisms to maintain zero-knowledge properties. However, in many cases, these methods may be overly complex, as they require both the message and random numbers to be proven zero-knowledge.

In practical applications, it may not be necessary to fully hide all information about the randomness. In fact, some leakage of random number information can still maintain the security of the protocol. To address this, a new approach has been proposed, which does not use noise flooding or rejection sampling, allowing the proof to reveal some information about the randomness while still preserving the security of the message.

The advancements in lattice-based zero-knowledge proof systems have led to significant improvements in efficiency, particularly in reducing proof sizes while maintaining security. These improvements are especially relevant in the context of **amortized proofs**, **product proofs**, and **Stern-style protocols**. The integration of NTT (Number Theoretic Transform) in lattice-based encryption has shown promising results in optimizing these proofs. Ultimately, these developments make lattice-based zero-knowledge proofs more practical and applicable in real-world scenarios, particularly in situations where large-scale proofs and efficient verification are required.

<details><summary><b> Code </b></summary>

<details><summary><b> Sage </b></summary>

```python
def findMSISdelta(B, n, d, logq):
	logC = log(B, 2)
	logdelta = logC^2 / (4*n*d*logq)
	return 2^logdelta

# Function for estimating the MLWE hardness for a (m x n) matrix in \Rq and
# secret coefficients sampled uniformly at random between -nu and nu.
# We use the LWE-Estimator by [AlbPlaSco15] as a black-box.

def findMLWEdelta(nu, n, d, logq):
    load("https://bitbucket.org/malb/lwe-estimator/raw/HEAD/estimator.py")
    n = n * d
    q = 2^logq
    stddev = sqrt(((2*nu+1)^2 - 1)/12)
    alpha = alphaf(sigmaf(stddev),q)
    set_verbose(1)
    L = estimate_lwe(n, alpha, q, reduction_cost_model=BKZ.enum)
    delta_enum1 = L['usvp']['delta_0']
    delta_enum2 = L['dec']['delta_0']
    delta_enum3 = L['dual']['delta_0']
    L = estimate_lwe(n, alpha, q, reduction_cost_model=BKZ.sieve)
    delta_sieve1 = L['usvp']['delta_0']
    delta_sieve2 = L['dec']['delta_0']
    delta_sieve3 = L['dual']['delta_0']
    return max(delta_enum1,delta_enum2,delta_enum3,delta_sieve1,delta_sieve2,delta_sieve3)
```

</details>

<details><summary><b> C++ </b></summary>

```C++
// Commit to a message.
void bdlop_commit(commit_t & com, vector < params::poly_q > m, comkey_t & key,
		vector < params::poly_q > r) {
	params::poly_q _m;

	com.c1 = r[0];
	for (size_t i = 0; i < HEIGHT; i++) {
		for (size_t j = 0; j < r.size() - HEIGHT; j++) {
			com.c1 = com.c1 + key.A1[i][j] * r[j + HEIGHT];
		}
	}

	com.c2.resize(m.size());
	for (size_t i = 0; i < m.size(); i++) {
		com.c2[i] = 0;
		for (size_t j = 0; j < r.size(); j++) {
			com.c2[i] = com.c2[i] + key.A2[i][j] * r[j];
		}
		_m = m[i];
		_m.ntt_pow_phi();
		com.c2[i] = com.c2[i] + _m;
	}
}

// Open a commitment on a message, randomness, factor.
int bdlop_open(commit_t & com, vector < params::poly_q > m, comkey_t & key,
		vector < params::poly_q > r, params::poly_q & f) {
	params::poly_q c1, _c1, c2[SIZE], _c2[SIZE], _m;
	int result = true;

	c1 = r[0];
	for (size_t i = 0; i < HEIGHT; i++) {
		for (size_t j = 0; j < r.size() - HEIGHT; j++) {
			c1 = c1 + key.A1[i][j] * r[j + HEIGHT];
		}
	}

	for (size_t i = 0; i < m.size(); i++) {
		c2[i] = 0;
		for (size_t j = 0; j < r.size(); j++) {
			c2[i] = c2[i] + key.A2[i][j] * r[j];
		}
		_m = m[i];
		_m.ntt_pow_phi();
		c2[i] = c2[i] + f * _m;
	}

	_c1 = f * com.c1;
	if (_c1 != c1 || com.c2.size() != m.size()) {
		result = false;
		cout << "ERROR: Commit opening failed test for c1" << endl;
	} else {
		for (size_t i = 0; i < m.size(); i++) {
			_c2[i] = f * com.c2[i];
			if (_c2[i] != c2[i]) {
				cout << "ERROR: Commit opening failed test for c2 (postition "
						<< i << ")" << endl;
				result = false;
			}
		}

		// Compute sigma^2 = (0.954 * v * beta * sqrt(k * N))^2.
		for (size_t i = 0; i < r.size(); i++) {
			c1 = r[i];
			c1.invntt_pow_invphi();
			if (!bdlop_test_norm(c1, 16 * SIGMA_C)) {
				cout << "ERROR: Commit opening failed norm test" << endl;
				result = false;
				break;
			}
		}
	}

	return result;
}

```

</details>

</details>

---

[LBZKP-main](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/LBZKP-main)

[lattice-verifiable-mixnet-main](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/lattice-verifiable-mixnet-main)

<div  align="center"> 
<img src="images/54_Lattice_BDLOP_ZKP.gif" width="50%" />
</div>
