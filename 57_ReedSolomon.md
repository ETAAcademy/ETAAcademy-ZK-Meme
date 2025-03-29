# ETAAcademy-ZKMeme: 57. Reed Solomon Code

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>57. Reed Solomon Code</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Reed-Solomon</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

---

# Reed-Solomon Code

Reed-Solomon codes enable data recovery through polynomial interpolation even when portions are lost. These codes have become essential in zero-knowledge proof systems, particularly for Polynomial Commitment Schemes (PCS) and Persistent Computation Delegation (PCD). Current PCD approaches face limitations: homomorphic vector commitment schemes are computationally expensive and quantum-vulnerable, while hash-based schemes restrict accumulation depth. The ARC scheme addresses these challenges by combining Reed-Solomon encoding with non-homomorphic Merkle trees, enabling infinite-depth accumulation for both polynomial IOP and R1CS schemes. Multilinear PCS methods fall into two categories: cryptographic assumption-based and error-correction code-based (including tensor code approaches like Brakedown and Orion, and FRI-based methods). FRI-based Reed-Solomon constructions have emerged as preferred solutions due to their efficient provers, transparent setup, and polylogarithmic verification. Ongoing research in Basefold and Deepfold focuses on optimizing queries, handling multivariate cases, and supporting arbitrary-length inputs, advancing zero-knowledge proofs toward greater efficiency and security.

---

Communication consists of three primary processes: encoding, transmission, and decoding. The original data to be encoded is referred to as the **message**, while the encoded data is called the **codeword**. The encoding and decoding process can be categorized into four main layers: **source coding**, which compresses and decompresses data; **cryptographic coding**, which ensures data security; **channel coding** (also known as forward error correction or FEC, such as Reed-Solomon codes), which detects and corrects transmission errors; and **line coding**, which converts digital signals into physical signals for transmission.

Channel coding is generally divided into two types: **linear block codes** and **convolutional codes**. Linear block codes operate by dividing data into fixed-size blocks (groups) for processing. One of the simplest forms is **repetition codes**, where a message (e.g., "UFO") is repeated multiple times (e.g., "UUUfffooo"), but this method is inefficient since only 1/n of the transmitted data is useful. Another method is **XOR codes**, where parity data is generated using the XOR operation (e.g., "101" XOR "010" yields "111"). While XOR codes have lower redundancy, they can only recover one missing data unit. An ideal coding scheme should be capable of encoding a message of length k (where each symbol belongs to a finite field F) into a longer codeword of length n, such that the original message can still be fully recovered even if up to r = n - k symbols are lost. Reed-Solomon (RS) codes precisely address this challenge.

#### Reed-Solomon Encoding Principle

Reed-Solomon codes leverage the principles of **polynomial interpolation**. They encode a message of length k into a codeword of length n using a polynomial transformation $m \to f(x)$, ensuring that any k symbols in the codeword are sufficient to reconstruct the entire message. Specifically, given a message $m = (a_0, a_1, ..., a_{k-1})$, we construct a polynomial of degree at most $k-1$:

$y = f(x) = a_0 + a_1x + a_2x^2 + ... + a_{k-1}x^{k-1}$

The message coefficients define this polynomial. The codeword is then obtained by evaluating the polynomial at n predefined points:

$c = (f(x_0), f(x_1), ..., f(x_{n-1}))$

At the receiver’s end, any k received symbols $f(x_{i_1}), f(x_{i_2}), ..., f(x_{i_k})$ can be used to reconstruct the original polynomial coefficients via matrix inversion. This reconstruction is feasible because the corresponding Vandermonde matrix:

<div  align="center">
<img src="https://github.com/ETAAcademy/ETAAcademy-Images/blob/main/ETAAcademy-ZKmeme/57_Vandermonde.png?raw=true" width="30%" />
</div>

is invertible as long as all $x_i$ are distinct. The determinant of this Vandermonde matrix is:

$\det(V) = \prod_{0 \leq i < j \leq k} (x_j - x_i),$

which is nonzero when all $x_i$ are distinct, ensuring the system has a unique solution.

**Reed-Solomon Code as a Linear Code**

Since the encoding process involves matrix-vector multiplication, RS codes exhibit **linear properties**. The codewords form a linear space spanned by the columns of the Vandermonde matrix. If we define the basis vectors as $g_i$, the codeword can be expressed as:

$\vec{y} = m_0 g_0 + m_1 g_1 + ... + m_{k-1} g_{k-1}$

For any two codewords $\vec{y}_m$ and $\vec{y}_n$, their linear combination:

$k_1 \vec{y}_m + k_2 \vec{y}_n$

remains a valid codeword, confirming that RS codes form a linear code.

From a geometric perspective, each received symbol $(x_i, y_i)$ represents a constraint that defines a hyperplane in k-dimensional space. The interpolation process reconstructs the polynomial coefficients by finding the unique intersection point of these hyperplanes. The Vandermonde matrix rows define the normal vectors of these hyperplanes, ensuring that the intersection point (the original message) is uniquely determined when k equations (received symbols) are available.

In practical applications, it is often desirable for the first k symbols of the codeword to be the original message itself. This property, known as **systematic encoding**, eliminates the need for decoding if no errors occur in the first k symbols. In systematic RS encoding, the encoding matrix is structured such that the first k columns form an identity matrix, directly preserving the original message in the codeword.

#### Reed-Solomon Code (RS Code) and Its Applications

Reed-Solomon (RS) codes are powerful error-correcting codes, typically represented as RS[F, L, d], where **F** is the mathematical field, **L** is the set of evaluation points, and a smooth curve (a polynomial of degree **d**) is used to connect these points. The **code rate** $\rho = d / |L|$ indicates the information density. RS codes rely on two key concepts: **rational constraints** (similar to a fraction calculator, consisting of a numerator $p$ and a denominator $q$, ensuring the result is a valid RS codeword) and **list decoding** (identifying all valid codewords within a certain error tolerance $\gamma$, as bounded by the Johnson bound).

The strength of RS codes lies in their ability to reconstruct the original data even when some points are erroneous. As long as enough correct points remain, the original polynomial can be determined, enabling data recovery. RS codes are significant due to their:

- **Error tolerance** – They can handle a certain proportion of errors with theoretical guarantees.
- **Efficiency** – Encoding and decoding processes are computationally efficient.
- **Security** – They provide mathematical proof mechanisms, making them useful in cryptographic systems.

RS codes integrate with concepts such as **quotients** and **proximity gaps**, making them vital not only in data storage and transmission but also in modern cryptographic proof systems.

**Quotients** are an essential technique in RS codes, operating on original data **f**, a special point set **S**, an expected value function **Ans**, and a padding function **Fill**. For special points in **S**, the Fill function provides direct values; for others, the quotient operation computes the difference between the original and expected values, dividing by a vanishing polynomial.

A key property of this quotient operation is that if all sufficiently close codewords at special points satisfy the expected values, then the quotient result maintains a minimum distance from any lower-degree RS codeword. This ensures that after quotient processing, the result remains "complex" and cannot be easily forged using simpler polynomials. This feature makes quotient operations critical in:

- **Data compression** – Reducing redundancy while preserving essential information.
- **Error detection** – Identifying inconsistencies in transmitted or stored data.
- **Cryptographic proofs** – Establishing reliable verification mechanisms.

**Proximity gaps** leverage approximation errors (computed based on field **F**, degree **d**, code rate $\rho$, distance parameter $\delta$, and variable count **m**) and combination functions (which merge multiple functions $f_1, ..., f_m$ with specific weightings **r**).

The core idea is that if the combined function is sufficiently "complex" and cannot be closely approximated by simple RS codewords, then each individual function in the set must be close to a valid RS codeword within a sufficiently large subset. This method offers:

- **Improved verification efficiency** – Avoiding the need for individual function verification.
- **Enhanced security** – Ensuring robustness against forgery and approximation attacks.

**Linear Codes and Reed-Solomon Theory** provide a foundation for **DeepFold**. The **Hamming distance** $Ham(\vec{u}, \vec{v})$ measures the proportion of differing positions between two vectors. An **(n, k, $\delta$)-linear code** is a linear subspace defined by an injective mapping $E: F^k \to F^n$, where an RS code encodes a message as a polynomial of degree at most $\rho |L|$ and evaluates it over **L**.

RS decoding operates within two critical bounds:

- **Unique decoding radius**: $\delta \leq (1 - \rho)/2$
- **List decoding radius**: $\delta \in ((1 - \rho)/2, 1 - \rho)$

A widely accepted conjecture suggests that the list decoding radius can reach $1 - \rho - \varepsilon$. The **consistency principle** asserts that if a random linear combination of two vectors is $\delta$-close to an RS codeword with a non-negligible probability, then both vectors must be $\delta$-close to some RS codeword over the same position set.

Research on **distance preservation bounds** has evolved from the original $(1 - \rho)/2$ bound to the optimal $1 - \rho - \varepsilon$ bound achieved in **Deep-FRI**. This advancement allows **DeepFold** to significantly reduce the number of queries needed by leveraging the **list decoding radius** rather than the **unique decoding radius**.

---

## 1. Reed-Solomon PCD: A High-Efficiency Accumulation Scheme

Proof-Carrying Data (PCD) is a system designed to verify the correctness of distributed incremental computations. It serves as a validation mechanism for computational chains, such as blockchain transactions or image modification histories. PCD has found extensive applications in programming language semantics execution, succinct arguments, verifiable computation, image provenance, and blockchain technologies.

In a PCD system, **strings and words** represent data and proofs, while **Hamming distance** measures data similarity. **Polynomials** provide the foundation for constructing efficient proof systems, and **random oracles** ensure the system's security. The most advanced PCD constructions currently rely on accumulation or folding schemes, which allow a prover to efficiently aggregate NP statements into an accumulator, reducing verification costs and making them independent of the number of accumulation steps.

**Challenges in Existing PCD Schemes**

Despite their advancements, existing PCD schemes face two major limitations:

- **Homomorphic Vector Commitment-Based Schemes**
  - These schemes either rely on the discrete logarithm problem (which is computationally expensive and vulnerable to quantum attacks) or on lattice-based assumptions (which provide post-quantum security but remain inefficient).
  - They are unable to leverage the latest advances in IOP-based SNARKs.
- **Homomorphic Checker-Based Hash Schemes**
  - These schemes avoid public-key assumptions and offer post-quantum security.
  - However, they only support a limited number of consecutive accumulation steps, restricting the depth of the computational graph and decreasing efficiency as the limit increases.

The **ARC (Reed-Solomon-based Accumulation Scheme)** seeks to overcome these limitations through innovative techniques, aiming for **infinite accumulation depth and higher efficiency**.

**A Hash-Based Accumulation Scheme with Infinite Depth**

A novel high-efficiency hash-based accumulation scheme replaces homomorphic vector commitments with non-homomorphic **Merkle tree commitments** to handle Reed-Solomon-encoded NP proof accumulation. The core of this scheme is a new accumulation protocol based on **Reed-Solomon proximity statements**, achieving near-optimal efficiency parameters.

For an RS code with rate $\rho$, the scheme only requires $\frac{2\lambda}{\log(1/\rho)}$ Merkle tree openings, and under certain common assumptions, the factor of 2 can be eliminated. Compared to prior work by Bünz et al., which supports only finite-depth accumulation and requires $O(d\lambda)$ Merkle openings, this approach is significantly more efficient.

Moreover, this scheme leverages **distance-preserving techniques**, ensuring that output distances remain within the same bounds as input distances. This avoids the issue of progressively increasing distances found in prior schemes. By extending the accumulation to the **list-decoding radius** $1 - \sqrt{\rho}$, the scheme further improves efficiency, making it more practical and widely applicable.

**Reed-Solomon-Based NP Accumulation Schemes**

Two primary Reed-Solomon-based NP accumulation schemes have been developed:

- **Polynomial IOP Accumulation Scheme**
  - Inspired by previous SNARK approaches, this scheme simplifies NP proof verification to checking codeword proximity to an RS code.
  - It consists of three steps: Proximity IOP (PIOP), identity transformation, and replacing low-degree tests with an accumulation scheme.
  - To further enhance the design and security proof of accumulation schemes, a new model called **Interactive Oracle Reductions (IOR)** has been introduced. IOR simplifies the construction and security analysis of PCD systems, paving the way for more robust and efficient solutions.
- **R1CS-Based Accumulation Scheme**
  - This scheme transforms NP relations into single-variable polynomial identities and further converts them into RS proximity statements.
  - It employs an accumulator structure containing two codewords: one corresponding to the RS proximity accumulator and the other to the accumulated polynomial identity proof.
  - Each input requires only $t = \frac{2\lambda}{\log(1/\rho)}$ Merkle path openings while preserving distance properties.

Each approach offers unique advantages:

- The **first scheme** is well-suited for improving existing IOP-based SNARKs for PCD constructions.
- The **second scheme** is simpler and more efficient as it avoids the need for PIOPs, making it ideal for developing new systems.

By leveraging Reed-Solomon codes, the ARC scheme significantly improves PCD efficiency, scalability, and security. These advancements open the door for more practical implementations in blockchain technology, verifiable computing, and cryptographic proof systems.

---

### 1.1 Interactive Oracle Reductions (IOR) and Reed-Solomon Accumulation

Interactive Oracle Reductions (IOR) extend existing proof paradigms by incorporating oracle-based interactions to achieve more efficient and scalable cryptographic proofs. The three core types of protocols that contribute to IOR are:

- **Knowledge Arguments:** Where the prover submits an answer, and the verifier provides a binary pass/fail response.
- **Knowledge Reductions:** Where a problem instance and its proof are transformed into an equivalent new instance and proof.
- **Interactive Oracle Reductions:** Where the prover supplies additional information (e.g., oracle string instances), and the verifier can access the prover’s messages.

IOR’s key innovation lies in its ability to support oracle string instances and allow the verifier to interact with prover messages. This enables non-interactive conversion in the random oracle model via the BCS transformation. IOR is particularly useful in constructing accumulation schemes—these require reductions from an instance $R$ to $R_{ACC}$ and from $R_{ACC}^{\*}$ to $R_{ACC}$ in a many-to-one manner. When $R$ is NP-complete, IOR enables the construction of proof-carrying data (PCD) systems.

#### Reed-Solomon Accumulation for Approximate Proximity Statements

The fundamental goal of Reed-Solomon (RS) accumulation is to reduce two vector proximity claims $f_1$ and $f_2$ with respect to an RS code $C$ into a single proximity claim about a new vector $f$. This process combines the two proximity statements— $f_1$ close to an RS code and $f_2$ close to an RS code—into a unified claim without doubling the storage size.

Traditional methods employing random linear combinations suffer from an exponential growth in statement size, while prior work that mitigated size growth resulted in degraded proximity guarantees, reducing proximity from $\delta$ to $\delta - \varepsilon$, thereby restricting accumulation depth. To address this, a novel approach using quotient operations was introduced:

$Quotient(f, x, y)(X) := \frac{(f(X) - y)}{(X - x)}$

This solution involves the verifier sampling a random combination $r$ and a set of evaluation points $x_1, ..., x_t$, while the prover sends a new function $f$. The verifier then constructs a new statement using the quotient operation, ensuring that the new claim remains independent of previous statement sizes while preserving distance properties.

This method’s correctness relies on:

- Maintaining $\delta$-far proximity with high probability using randomized selection.
- Degree correction procedures ensuring consistency in statements.
- Out-of-domain sampling techniques that surpass traditional unique decoding radius limitations, supporting larger $\delta$ values and improving query complexity and efficiency.

This innovation not only enables infinite-depth accumulation but also establishes a robust foundation for building efficient PCD systems.

#### Reed-Solomon Approximate Proximity Accumulation System

The RS accumulation system is designed to handle RS proximity relations. It operates over a finite field $F$ with evaluation domain $L$, a maximum degree $d_{max}$, and a proximity parameter $\delta$. The system involves:

- Rational constraints $(c, d)$ and an interleaved word $f = (f_1, ..., f_k)$ as instances.
- Exact relations $R_{RS}$, requiring $c(f)$ to be an RS codeword.
- Approximate relations $\tilde{R}_{RS}$, allowing $c(f)$ to be within $\delta$ distance of an RS codeword.

For security, the system imposes constraints on field size $|F| \geq 2\lambda \cdot 10^7 \cdot m \cdot d_{max}^3 \cdot \rho^{-3.5}$ and specific proximity parameter limits. In terms of efficiency, it employs a three-round interactive process:

- **Query complexity:** $t \cdot \sum_{i=1}^{m} k_i$
- **Proof length:** $|L| + t + 1$ field elements
- **Time complexity:**
  - Prover: $O(|L| \cdot \sum_{i=1}^{m} |c_i| + d_{max} \log d_{max})$
  - Verifier: $O(t \cdot \sum_{i=1}^{m} |c_i|)$

Critically, this system enables the merging of multiple RS proximity claims into a single statement while maintaining a strict $2^{-\lambda}$ error bound. This makes it particularly useful for large-scale cryptographic proof systems.

#### Protocol Structure and Validation

The RS accumulation system is parameterized using out-of-domain repetition $s$ and in-domain repetition $t$. The protocol consists of two main phases:

- **Interactive Phase:** Six sequential steps where the verifier sends random values $r$, the prover responds with functions, and both parties iteratively refine the instance.
- **Query Phase:** Definition of virtual function $f'$, answer function $Ans$, and rational function $c$, culminating in the final instance output.

The integrity of the construction is ensured by:

- **Prover Honesty:** When honest, the combined $f'$ is a valid RS codeword.
- **Quotient Verification:** The quotient operation maintains correct polynomial degree and ensures proximity.
- **Efficient Interaction Design:** Optimized round complexity and computational overhead.

This design enhances the system’s ability to handle multiple RS proximity statements while maintaining efficiency and security through random sampling and iterative validation.

#### Security Analysis and Optimization

Security is ensured through a three-phase error analysis model. The protocol defines an accumulated round error bound $\varepsilon_{rbr}$, considering:

- **Approximate Error ($\varepsilon_{prox}$)** – Ensures proximity preservation.
- **Out-of-Domain Error ($\varepsilon_{ood}$)** – Guarantees consistency in sampled points.
- **Repetition Error ($(1-\delta)^t$)** – Controls proximity over accumulation depth.

Each phase enforces strict probability bounds and verifies the system’s transition states, maintaining verifier consistency across rounds.

Optimizations include:

- **FFT Delay Optimization:**
  - The prover initially transmits zero values, deferring actual FFT computations to only necessary instances.
  - Probabilistic analysis confirms this efficiency (e.g., at $|L|=2^{26}, t=80$, an FFT computation is needed only once every 100 steps).
- **Parameter Refinement for Speculative Security:**
  - Assumes minimal codeword density within $(1 - \rho - \eta')$ radius.
  - Reduces query complexity from $t \approx \frac{2\lambda}{\log(1/\rho)}$ to approximately $\frac{\lambda}{\log(1/\rho)}$.
  - With $s = 2$, field size requirements are further optimized: $|F| \geq 2\lambda \cdot \frac{(m-1) |L|}{\eta'}$.

---

### 1.2 NP Accumulation

The NP accumulation scheme primarily addresses the Rank-1 Constraint System (R1CS) circuit satisfiability problem. This problem is defined by matrices $A, B, C \in F^{(M \times N)}$ and an instance length $n$, requiring the existence of a witness $w$ such that the constraint $A_z \circ B_z = C_z$ holds. The accumulation scheme for the $R_{R1CS}$ relation transforms the matrix constraint system $A_z \circ B_z = C_z$ into a polynomial representation, facilitating an efficient verification system.

The system expresses R1CS constraints using the polynomial:

$\hat{p}(Y_1, \dots, Y_m, Z_1, \dots, Z_N) = \sum_{i=1}^{M} eq(i-1, Y_1, \dots, Y_m) \cdot (a_i^T \vec{Z} \cdot b_i^T \vec{Z} - c_i^T \vec{Z})$

and defines an accumulator relation containing two main components: index information (including the finite field $F$, evaluation domain $L$, maximum degree $d_{max}$, and distance parameters $\delta, \gamma$) and instance composition (including two rational constraints, vector $v$, error term $e$, and two oracle strings $f, g$).

The system establishes two relations: the exact relation $R_{ACC}$, which requires $f$ to be an RS codeword, $c_f(f)$ and $c_g(g)$ to be RS codewords, and $\hat{P}(v||\vec{f}) = e$; and the approximate relation $R̃_{ACC}$, which allows $c_g(g)$ to be close to an RS codeword while requiring the existence of $u$ within $f$'s list satisfying specific conditions. The core of the scheme is the construction of two components: the reduction from $R_{R1CS}$ to the intermediate relation $R_{ACC}$ and the many-to-one reduction within $R_{ACC}$.

#### Reduction from $R_{R1CS}$ to $R_{ACC}$

The core of the reduction from $R_{R1CS}$ to $R_{ACC}$ involves constructing specialized polynomial systems that transform R1CS problems (which verify satisfaction of multiplication constraints) into ACC problems (which verify proximity between functions and specific polynomials). It ensures that $f$ must be $\delta$-close to a codeword $u$ satisfying $P(v, \vec{u}) = e$, where $P$ is a multivariate polynomial of total degree $c$ with $k + d$ variables, and $\vec{u}$ represents the coefficient vector of $u$. The distance parameter $\delta$ does not exceed the unique decoding radius of the code.

The reduction procedure constructs special polynomial systems to convert R1CS constraints into an ACC problem. It starts by defining $M$ (a power of two) and $m = \log M$. For each index $i$, it constructs a multilinear polynomial:

$pow_i(Y_1, \dots, Y_m) = Y_1^{b_1} \cdot \dots \cdot Y_m^{b_m},$

where $(b_1, \dots, b_m)$ represents the binary expansion of $i$, ensuring that $pow_i(y, y^2, y^4, \dots, y^{2^{m-1}}) = y^i$. Then, it encodes the R1CS matrix constraints into a polynomial $P$ of total degree $m+2$. The reduction protocol involves the prover sending $f$ (which encodes the witness $w$ in the honest case), the verifier randomly sampling $r$, and generating a new statement $(e, v, f) \in L(R_{ACC})$, where $e := 0$, and $v := (r, r^2, r^4, \dots, r^{2^{m-1}}, x) \in F^k$.

The security of this reduction is ensured by bounding $\delta$ within the unique decoding radius, and by analyzing the constructed univariate polynomial $F(X)$. If $x$ is not a valid R1CS instance, the probability that the new statement belongs to $L(R_{ACC})$ is negligible.

#### Many-to-One Reduction in $R_{ACC}$

The many-to-one reduction aims to merge $m$ statements $(e_1,v_1,f_1),..., (e_m,v_m,f_m)$ into a single new statement $(e, v, f) \in R_{ACC}$. It utilizes Lagrange interpolation: selecting $m$ distinct points and constructing corresponding Lagrange polynomials $L$, which take value 1 at their respective points and 0 elsewhere. The protocol involves:

- The prover sending a polynomial $Q$ to handle statement combinations.
- The verifier choosing a random point $\alpha$.
- The prover sending a new function $f$ (a weighted combination of the original $f_i$).
- The verifier evaluating:

$F(X) := P\left(\sum_{i=1}^{m} L_i(X) \cdot (v_i, \vec{f_i})\right) - Q(X) \cdot V(X) - \sum_{i=1}^{m} L_i(X) \cdot e_i.$

The new statement ensures $R_{ACC}$ membership, proximity to an RS codeword, and quotient correctness. By leveraging off-domain sampling techniques, the scheme improves security and allows list decoding.

#### Proof-Carrying Data (PCD) System Construction

To build a PCD system, two components are employed:

- $RDX_{CAST}$: Converts an original relation $R$ into an accumulator relation $R_{ACC}$, akin to a "casting" step, proving correctness.
- $RDX_{FOLD}$: Merges multiple accumulator instances into one, akin to "folding."

The process follows two steps:

- **ARG Construction:** The prover generates $(\pi_{CAST}, acc.w)$, and the verifier checks whether $acc.x, acc.w$ belong to $R_{ACC}$.
- **ACC Construction:** It processes $m$ accumulator instances and $n$ predicate instances using $VCAST$, generating $m+n$ new accumulator instances, then merges them via $RDX_{FOLD}$ into a final accumulator $acc$ and proof $\pi_{FOLD}$.

Extending this to PCD and IVC requires $R$ to be NP-complete and the reduction verifier to be succinct. The framework unifies reduction, accumulation, and PCD, offering a flexible and efficient construction method for distributed computation verification.

Integrity is ensured by proving all components ( $f_i$, $c_f(i(f_i))$, $c_g(i(g_i))$ ) remain valid codewords. The reliability analysis bounds the total round error probability through:

$\kappa_{rbr} = \max\left(
  \varepsilon_{prox}(d_{max}, \rho, \delta, m \cdot (2t + 6)),
  m \cdot \frac{\ell^2}{2} \cdot \left(\frac{d_{max}}{|F|-|L|}\right)^s,
  \frac{(m-1) \cdot d_P}{F},
  (1-\delta)^t
\right).$

This guarantees robust security and efficiency, making the approach practical for real-world applications.

---

## 2. Reed-Solomon Polynomial Commitment Schemes (PCS)

In practical zero-knowledge proof (ZKP) systems, Polynomial Commitment Schemes (PCS) and Proof-Carrying Data (PCD) mechanisms are often used together. PCS enables efficient representation and verification of polynomials within individual computation steps, while PCD links these steps into a verifiable computation chain. As a fundamental primitive, single-variable PCS was initially proposed by KZG, offering constant-time verification and compact proof size. However, its reliance on a trusted setup and high computational overhead has driven researchers toward error-correcting code-based PCS. Among these, the FRI scheme, built upon Reed-Solomon codes, has become a preferred choice in practical applications due to its efficient prover, transparent setup, and polylogarithmic verification time and proof size.

Multivariate PCS schemes can generally be categorized into two main types: cryptographic-assumption-based and error-correcting-code-based approaches. The former includes schemes like mKZG, which provides logarithmic verification time and proof size but requires a trusted setup; Bulletproofs, which eliminates the need for a trusted setup but incurs high verification costs; Hyrax, which improves verification time but is constrained by product complexity; and DARK, which achieves logarithmic complexity but relies on strong assumptions. The latter category consists of tensor-code-based schemes, such as Brakedown and Orion, and FRI-based approaches like Virgo, PolyFRIM, ZeromorphFRI, and BaseFold.

Although FRI-based multivariate PCS schemes offer smaller proof sizes, they face three critical challenges:

- **Query Optimization Problem**: The study of query count in FRI has evolved from unique decoding radius (1-ρ/2) to Ben-Sasson et al.'s 1-⁴√ρ and 1-³√ρ bounds, and further to DEEP techniques enabling arbitrary list decoding radius. While BaseFold achieves efficient multivariate PCS via unconventional FRI usage, it remains constrained by unique decoding settings. As a result, it requires an excessive number of queries—exceeding 120 for 100-bit security at a code rate of 1/8—leading to a proof size of 619KB for μ=22-variable multilinear polynomials, significantly larger than DEEP-FRI’s approximately 200KB proof size for single-variable polynomials of similar scale.
- **Adapting Single-Variable to Multivariate PCS**: Multivariate polynomials, particularly multilinear polynomials, are widely used in constructing efficient SNARKs, driving research interest in extending FRI techniques from single-variable to multivariate PCS. However, this transition poses significant challenges since Reed-Solomon codes were originally designed for single-variable settings. Although Virgo, PolyFRIM, and BaseFold have attempted such adaptations, they all exhibit substantial increases in prover time or proof size compared to single-variable FRI.
- **Handling Arbitrary-Length Inputs**: FRI-based PCS requires extensive padding when handling input sizes that are not powers of 2. For example, encoding a vector of size $2^k+2^{(k/2)}$ requires padding up to $2^{(k+1)}$. Unlike multilinear KZG variants, which can skip zero elements, FRI-based PCS relies on Reed-Solomon encoding, making the entire padded vector nonzero and nearly doubling prover time. This issue is exacerbated in verifiable matrix multiplication, where overhead can increase up to fourfold.

**DeepFold’s Innovative Solution**

DeepFold introduces DEEP techniques into BaseFold’s unconventional FRI framework, marking the first adaptation of FRI-based multilinear PCS to list decoding radius settings. This significantly reduces redundant queries and introduces a batch evaluation scheme for non-power-of-2 polynomials, breaking the constraints of unique decoding. When the code rate is 1/8, DeepFold reduces the required number of queries from over 120 to approximately 34, achieving over a threefold reduction in proof size. Furthermore, DeepFold develops batch processing techniques to efficiently handle arbitrary-length inputs, maintaining optimal prover time while substantially reducing proof size. These advancements provide crucial tools for building more efficient zero-knowledge proof systems.

---

#### Fold & Batch in DeepFold

The primary contribution of DeepFold is the reduction of query numbers in BaseFold, significantly decreasing proof size. FRI (Fast Reed-Solomon Interactive Oracle Proof of Proximity) is a specialized Reed-Solomon interactive oracle proof (RS-IOPP) that enables a prover to demonstrate that a vector $\vec{v}$ is $\Delta$-close to the code $RS[F, L_0, \rho]$, where $L_0$ is a multiplicative subgroup of $F$. Here, $\Delta \in [0,1]$ represents the maximum relative Hamming distance allowed in the FRI protocol, while $\delta = Ham(\vec{v}, RS[F, L_0, \rho])$ denotes the actual distance between $\vec{v}$ and the Reed-Solomon code. The FRI protocol consists of two main phases:

- **Commitment phase**: The prover performs $\mu$ rounds of folding, reducing polynomial length by half in each round while sending Merkle commitments.
- **Query phase**: The verifier randomly queries the vectors at each round to check the consistency of the folding process. To ensure security, the queries must be repeated $s$ times.

BaseFold's key insight was that the last-round polynomial $f^{(\mu)}$ in FRI is equivalent to the evaluation of a multilinear polynomial $\tilde{f}$ at a random challenge point $(r_1, \dots, r_\mu)$. This property enables an efficient multilinear polynomial commitment scheme (PCS) by running FRI in parallel with the sumcheck protocol to verify multilinear polynomial evaluations. However, BaseFold operates only within the unique decoding radius $(1-\rho)/2$, leading to excessive queries—over 120 to ensure 100-bit security—resulting in a proof size exceeding 600KB.

DeepFold overcomes these limitations by extending query constraints from the unique decoding radius to the list decoding radius. This enhancement significantly reduces the required number of queries and addresses BaseFold’s security challenges under the list decoding setting. The core innovations enabling this improvement are:

- **Incorporation of DEEP (Domain Extension for Error Protection) technology**: DeepFold introduces out-of-domain query points $\alpha_i$ at each FRI round, preventing malicious provers from exploiting multiple valid codewords for cheating.
- **Development of "Deep Folding" methodology**: Instead of relying on quotient-based techniques in DEEP-FRI, DeepFold refines the folding equation:

  $f^{(i)}(X) = \frac{(f^{(i)}E(X) + r_i \cdot f^{(i)}_O(X)) - (f^{(i)}_E(\alpha^2_i) + r_i \cdot f^{(i)}_O(\alpha^2_i))}{X - \alpha^2_i}$

  This approach efficiently utilizes the twin polynomial relationship to simplify verification, ultimately reducing to FRI’s final polynomial $f^{(\mu)} = \tilde{f}(r_1, \dots, r_\mu)$.

**DeepFold Protocol Execution**

- The prover first commits to the vector $\vec{v} = f^{(0)}|L_0$ using a Merkle root and additional evaluation points.
- During the evaluation phase, $\mu$ rounds of interaction take place, where polynomials are folded, and additional evaluations are incorporated.
- Finally, $s$ repeated queries ensure the integrity of the folding process.

Unlike BaseFold, DeepFold does not explicitly invoke sumcheck but instead applies DEEP evaluation folding techniques directly to the original statement, leading to a more compact proof. The distance parameter $\Delta$ is improved from the unique decoding radius $(1-\rho)/2$ to the list decoding radius $1-\rho-\varepsilon$, significantly reducing the query count from over 120 to approximately 34 and decreasing proof size by a factor of about three.

**Batch Processing in DeepFold**

DeepFold further introduces an efficient batch processing technique to handle inputs of arbitrary lengths. This is achieved by decomposing inputs into multiple polynomials of varying sizes and verifying them simultaneously via random linear combinations, ensuring optimal prover complexity without substantially increasing proof size.

Traditional batch evaluation methods require padding smaller polynomials to the size of the largest polynomial, incurring significant computational and storage overhead. The core innovation of **DeepFold.BatchEvaluation** lies in leveraging the inherent structure of multilinear polynomials, allowing them to be randomly folded and evaluated in smaller segments, eliminating the need for explicit padding.

When processing multiple polynomials with different variable counts—e.g., $\tilde{f}_0$ with $\mu$ variables and $\tilde{f}_1$ with $\ell < \mu$ variables—the algorithm:

- First folds the larger polynomial $\tilde{f}_0$ down to the same dimension as $\tilde{f}_1$ in round $(\mu - \ell)$.
- Then, it combines them using a random challenge $\gamma_i$ to form a single polynomial $f_0^{(i)} + \gamma_i \cdot f_1^{(i)}$ for further processing.

This method is reflected in key modifications to the protocol:

- The prover submits additional evaluation values $c\vec{w} := f_j(\alpha^{2j-i+1})$.
- The verifier provides extra random challenges $\gamma_i$ for polynomial combination.
- The folding equation is extended to:

  $f^{(i)}(X) = f^{(i)}E(X) + r_i \cdot f^{(i)}_O(X) + \gamma_i \cdot f_i(X)$

- The verification step is adjusted to check:

  $g_{\vec{w}}(r_i) + \gamma_i \cdot c_{\vec{w}[2:]} = g_{\vec{w}[2:]}$

This batch processing method enables DeepFold to efficiently verify multiple multilinear polynomial evaluations of different sizes while keeping proof size and verification time nearly identical to the independent evaluation of the largest polynomial. Prover complexity scales only linearly with the total size of all polynomials, making it a practical solution for handling irregularly sized inputs (e.g., vectors of length $2^{2n} + 2^n$). Furthermore, these optimizations lay the foundation for a zero-knowledge version of DeepFold, making it a powerful tool for real-world applications requiring compact and efficient polynomial commitment schemes.

---

#### Zero-Knowledge Proof in DeepFold (zkDeepFold)

DeepFold’s zero-knowledge extension (zkDeepFold) addresses the unique challenges of achieving zero-knowledge properties in hash-based polynomial commitment schemes. Unlike homomorphic commitment schemes such as mKZG, DeepFold cannot directly implement zero-knowledge by applying random linear combinations. A thorough analysis reveals two primary sources of information leakage in DeepFold: (1) the verifier's ability to query specific polynomial values at each round, and (2) access to specific leaf nodes in the Merkle tree during the query phase. To mitigate these risks, zkDeepFold employs a dual strategy to ensure zero-knowledge properties while maintaining efficiency.

The zkDeepFold framework introduces two key modifications to prevent information leakage:

- **Extension of the Multilinear Polynomial:** The original multilinear polynomial $\tilde{f}$ with $\mu$ variables is extended to a new polynomial $f̃_{ext}$ with $\mu + 1$ variables. This extension introduces $2^u$ random coefficients such that $f̃_{ext}(\vec{z}||0) = \tilde{f}(\vec{z})$, while for any $x \neq 0$, $f_{ext}$ evaluates to a random value. This ensures that the original polynomial's information remains hidden during evaluations.

- **Batch Processing for Additional Masking:** Despite the polynomial extension, the last few evaluation rounds still pose a risk of leakage, as the verifier could interpolate the polynomial using the $s$ queried points and infer information about the original coefficients. To address this, zkDeepFold introduces batch evaluation of $f̃_{ext}$ alongside a smaller random polynomial $\tilde{g}$, which is designed to mask potential leakage in the final $\ell$ rounds. Specifically, during the commitment phase, a random vector $\vec{r}$ is sampled to construct $f̃_{ext}$ and $\tilde{g}$. During the evaluation phase, the BatchEval protocol is used to jointly verify $y = f̃_{ext}(\vec{z}||0)$ and $\tilde{g}(\vec{z}[\mu - \ell + 2:]||0)$. This design effectively guarantees zero-knowledge while minimizing computational overhead, as $\tilde{g}$ can remain significantly smaller than $\tilde{f}_{ext}$, ensuring that zkDeepFold does not incur excessive proof size or verification costs.

**Integration into zk-SNARK Systems**
Modern zk-SNARK frameworks such as Libra and HyperPlonk typically integrate multilinear polynomial commitment schemes (PCS) with polynomial interactive oracle proofs (PIOP) to construct efficient proof systems. DeepFold, with its transparent setup (requiring no trusted setup), compact proof sizes, and efficient proving process, presents a viable alternative to existing schemes such as mKZG and Virgo. By replacing these commitment schemes with DeepFold, zk-SNARK systems can immediately benefit from reduced proof sizes and improved performance.

DeepFold’s efficiency extends to verifiable computation, particularly in matrix multiplication proofs. When combined with Thaler’s $O(n^2)$ complexity proof method, DeepFold enables efficient verification of matrix multiplication correctness. In this approach, matrices $A, B,$ and $C$ are encoded as multilinear polynomials, and the sumcheck protocol is used to validate the relation:

$\sum_{\vec{y} \in \{0,1\}^{\log n}} \tilde{A}(r_1, \vec{y}) \cdot \tilde{B}(\vec{y}, r_2) = \tilde{C}(r_1, r_2)$

DeepFold facilitates efficient polynomial evaluations and commitments throughout this process. Notably, its batch processing capability addresses the challenge of handling matrix dimensions that are not powers of two, making it highly effective for real-world applications. This feature is particularly valuable in zero-knowledge machine learning (zkML), where large-scale, irregular data structures must be efficiently processed and verified.

<details><summary><b> Code </b></summary>

<details><summary><b> deepfold/src/lib.rs </b></summary>

```rust

#[derive(Clone)]
pub struct DeepEval<T: MyField> {
    point: Vec<T>,
    first_eval: T,
    else_evals: Vec<T>,
}

...

pub struct Commit<T: MyField> {
    merkle_root: [u8; MERKLE_ROOT_SIZE],
    deep: T,
}

#[derive(Clone)]
pub struct Proof<T: MyField> {
    merkle_root: Vec<[u8; MERKLE_ROOT_SIZE]>,
    query_result: Vec<QueryResult<T>>,
    deep_evals: Vec<(T, Vec<T>)>,
    shuffle_evals: Vec<T>,
    evaluation: T,
    final_value: T,
    final_poly: Polynomial<T>,
}

```

</details>

<details><summary><b> Code </b></summary>

```rust

#[derive(Clone)]
pub struct Prover<T: MyField> {
    total_round: usize,
    interpolate_cosets: Vec<Coset<T>>,
    interpolations: Vec<InterpolateValue<T>>,
    hypercube_interpolation: Vec<T>,
    deep_eval: Vec<DeepEval<T>>,
    shuffle_eval: Option<DeepEval<T>>,
    oracle: RandomOracle<T>,
    final_value: Option<T>,
    final_poly: Option<Polynomial<T>>,
    step: usize,
}
...

impl<T: MyField> Prover<T> {
...
    pub fn commit_polynomial(&self) -> Commit<T> {
        Commit {
            merkle_root: self.interpolations[0].commit(),
            deep: self.deep_eval[0].first_eval,
        }
    }

     fn evaluation_next_domain(&self, round: usize, challenges: &Vec<T>) -> Vec<T> {
        let mut get_folding_value = self.interpolations[round].value.clone();

        for j in 0..self.step {
            if round * self.step + j == self.total_round {
                break;
            }
            let len = self.interpolate_cosets[round * self.step + j].size();
            let coset = &self.interpolate_cosets[round * self.step + j];
            let challenge = challenges[j];
            let mut tmp_folding_value = vec![];
            for i in 0..(len / 2) {
                let x = get_folding_value[i];
                let nx = get_folding_value[i + len / 2];
                let new_v = (x + nx) + challenge * (x - nx) * coset.element_inv_at(i);
                tmp_folding_value.push(new_v * T::inverse_2());
            }
            get_folding_value = tmp_folding_value;
        }
        get_folding_value
    }

    ...

    pub fn generate_proof(mut self, point: Vec<T>) -> Proof<T> {
        self.prove(point);
        let query_result = self.query();
        Proof {
            merkle_root: (1..self.total_round / self.step)
                .into_iter()
                .map(|x| self.interpolations[x].commit())
                .collect(),
            query_result,
            deep_evals: self
                .deep_eval
                .iter()
                .map(|x| (x.first_eval, x.else_evals.clone()))
                .collect(),
            shuffle_evals: self.shuffle_eval.as_ref().unwrap().else_evals.clone(),
            final_value: self.final_value.unwrap(),
            final_poly: self.final_poly.unwrap(),
            evaluation: self.shuffle_eval.as_ref().unwrap().first_eval,
        }
    }

```

</details>

<details><summary><b> Code </b></summary>

```rust

#[derive(Clone)]
pub struct Verifier<T: MyField> {
    total_round: usize,
    interpolate_cosets: Vec<Coset<T>>,
    polynomial_roots: Vec<MerkleTreeVerifier>,
    first_deep: T,
    oracle: RandomOracle<T>,
    final_value: Option<T>,
    final_poly: Option<Polynomial<T>>,
    shuffle_eval: Option<DeepEval<T>>,
    deep_evals: Vec<DeepEval<T>>,
    open_point: Vec<T>,
    step: usize,
}

impl<T: MyField> Verifier<T> {
...
    pub fn verify(mut self, proof: Proof<T>) -> bool {
        self.final_value = Some(proof.final_value);
        self.final_poly = Some(proof.final_poly);
        let mut leave_number = self.interpolate_cosets[0].size() / (1 << self.step);
        for merkle_root in proof.merkle_root {
            leave_number /= 1 << self.step;
            self.polynomial_roots.push(MerkleTreeVerifier {
                merkle_root,
                leave_number,
            });
        }
        self.shuffle_eval = Some(DeepEval {
            point: self.open_point.clone(),
            first_eval: proof.evaluation,
            else_evals: proof.shuffle_evals,
        });
        assert_eq!(self.first_deep, proof.deep_evals[0].0);
        proof
            .deep_evals
            .into_iter()
            .enumerate()
            .for_each(|(idx, (first_eval, else_evals))| {
                self.deep_evals.push(DeepEval {
                    point: std::iter::successors(Some(self.oracle.deep[idx]), |&x| Some(x * x))
                        .take(self.total_round - idx)
                        .collect::<Vec<_>>(),
                    first_eval,
                    else_evals,
                });
            });
        self._verify(&proof.query_result)
    }

    fn _verify(&self, polynomial_proof: &Vec<QueryResult<T>>) -> bool {
        let mut leaf_indices = self.oracle.query_list.clone();
        for i in 0..self.total_round / self.step {
            let domain_size = self.interpolate_cosets[i * self.step].size();
            leaf_indices = leaf_indices
                .iter_mut()
                .map(|v| *v % (domain_size / (1 << self.step)))
                .collect();
            leaf_indices.sort();
            leaf_indices.dedup();

            polynomial_proof[i].verify_merkle_tree(
                &leaf_indices,
                1 << self.step,
                &self.polynomial_roots[i],
            );

            if i == self.total_round / self.step - 1 {
                let challenges = self.oracle.folding_challenges[0..self.total_round].to_vec();
                assert_eq!(
                    self.shuffle_eval.as_ref().unwrap().verify(&challenges),
                    self.final_value.unwrap()
                );
                for j in &self.deep_evals {
                    assert_eq!(j.verify(&challenges), self.final_value.unwrap());
                }
            }

            let folding_value = &polynomial_proof[i].proof_values;
            let mut challenge = vec![];
            for j in 0..self.step {
                challenge.push(self.oracle.folding_challenges[i * self.step + j]);
            }
            for k in &leaf_indices {
                // let x = folding_value[k];
                // let nx = folding_value[&(k + domain_size / 2)];
                // let v =
                //     x + nx + challenge * (x - nx) * self.interpolate_cosets[i].element_inv_at(*k);
                // if i == self.total_round - 1 {
                //     assert_eq!(v * T::inverse_2(), self.final_value.unwrap());
                // } else {
                //     assert_eq!(v * T::inverse_2(), polynomial_proof[i + 1].proof_values[k]);
                // }
                let mut x;
                let mut nx;
                let mut verify_values = vec![];
                let mut verify_inds = vec![];
                for j in 0..(1 << self.step) {
                    // Init verify values, which is the total values in the first step
                    let ind = k + j * domain_size / (1 << self.step);
                    verify_values.push(folding_value[&ind]);
                    verify_inds.push(ind);
                }
                for j in 0..self.step {
                    let size = verify_values.len();
                    let mut tmp_values = vec![];
                    let mut tmp_inds = vec![];
                    for l in 0..size / 2 {
                        x = verify_values[l];
                        nx = verify_values[l + size / 2];
                        tmp_values.push(
                            (x + nx
                                + challenge[j]
                                    * (x - nx)
                                    * self.interpolate_cosets[i * self.step + j]
                                        .element_inv_at(verify_inds[l]))
                                * T::inverse_2(),
                        );
                        tmp_inds.push(verify_inds[l]);
                    }
                    verify_values = tmp_values;
                    verify_inds = tmp_inds;
                }
                assert_eq!(verify_values[0], polynomial_proof[i + 1].proof_values[k]);
            }
        }
        true
    }

```

</details>

</details>

---

[Deepfold](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/deepfold)

<div  align="center">
<img src="https://github.com/ETAAcademy/ETAAcademy-Images/blob/main/ETAAcademy-ZKmeme/57_ReedSolomon.gif?raw=true" width="50%" />
</div>
