# ETAAcademy-ZKMeme: 58. Libra

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>58. Reed Solomon Code</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Libra</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

---

# Libra

Modern SNARKs employ modular designs combining theoretical primitives (PCPs, IOPs, PIOPs) with cryptographic compilers, categorized by their polynomial commitment schemes (univariate-based like Plonk or multilinear-based like Libra and Spartan). Recent research has shifted from optimizing proof size toward prover efficiency, exemplified by Libra's breakthrough in simultaneously achieving linear-time proving and succinct verification—resolving the longstanding tension between these competing goals. Libra revolutionizes zero-knowledge proofs by integrating GKR protocol with multilinear extensions and sumcheck, transforming circuit verification into efficient polynomial operations with linear complexity. It achieves linear-time proving by exploiting circuit sparsity, implementing dynamic programming techniques, and dividing sumcheck into independent phases—reducing complexity from O(C log C) to O(C). The system adds zero-knowledge through logarithmic-sized masking polynomials, random linear combinations, and low-degree extensions that prevent information leakage while enabling verification, making large-scale privacy-preserving computations practical while minimizing computational overhead.

---

**Libra** is a groundbreaking zero-knowledge proof (ZKP) system that achieves an optimal balance between prover time, proof size, and verification efficiency. It does so by elegantly combining the **Goldwasser-Kalai-Rothblum (GKR) protocol**, the **Sumcheck protocol**, and a **zero-knowledge verifiable polynomial delegation (zkVPD)** scheme. At its core, Libra is built on the GKR protocol—a highly efficient interactive proof system tailored for layered arithmetic circuits. GKR simplifies the verification of complex computations by decomposing them into layer-by-layer checks, where each gate receives input from the gates in the next layer. This structural design allows the verification process to focus on a single layer at a time.

To facilitate this, **Multilinear Extensions** are employed: the binary values of each layer’s gates (typically 0 or 1) are extended into multivariate polynomials. This transforms the task of verifying a circuit layer’s correctness into evaluating a multilinear extension at a randomly chosen point. Instead of computing the circuit directly, both prover and verifier operate over these polynomials, enabling much more efficient verification.

The GKR protocol, however, requires evaluating summation-style checks multiple times. Here, the **Sumcheck Protocol** plays a pivotal role. It allows the verifier to delegate the evaluation of large, high-dimensional polynomial sums to the prover. At each round of the protocol, the prover sends a univariate polynomial (focusing on one variable at a time), and the verifier performs a random spot-check. This approach enables verifying a sum over an exponential-size domain ( $O(2^n)$ ) with only linear complexity in the number of variables (O(n))—a significant efficiency improvement over traditional brute-force summation.

Libra also integrates a zkVPD scheme, giving it strong **zero-knowledge guarantees**: the prover can convince the verifier of the correctness of a computation without revealing anything about the underlying data. Libra’s innovations lie not just in efficiency, but also in applying zero-knowledge to the GKR framework with unprecedented performance.

#### Linear-Time Prover via Sparse Optimization

A key achievement of Libra is its **linear-time prover algorithm**, which is made possible by exploiting the **sparsity** inherent in arithmetic circuits—most gates in a circuit connect only to a few others, resulting in polynomials where most coefficients are zero. Libra focuses computation only on the non-zero coefficients.

Further efficiency is gained through a **memoization strategy** using a **bookkeeping table**. This avoids exponential recomputation during the Sumcheck protocol by storing intermediate polynomial evaluations. Libra also splits the Sumcheck process into **two independent stages**: one for each polynomial in a product-sum proof. This sidesteps complex interdependencies and improves modularity.

As a result, Libra reduces the prover’s complexity from superlinear O(C log C) (as in the original GKR) to optimal **O(C)**, where C is the size of the circuit—achieving the theoretical limit for scalability in ZKP systems.

#### Innovations in Adding Zero-Knowledge to GKR

Libra introduces several novel techniques to inject zero-knowledge into the GKR protocol with minimal overhead:

- **Lightweight Masking Polynomials**: Traditional zero-knowledge GKR approaches use large masking polynomials with size proportional to the circuit (O(n) coefficients). Libra replaces these with compact polynomials having only O(log n) coefficients, significantly reducing computational cost. This design is based on the insight that it’s unnecessary to mask every possible computation path to ensure privacy.

- **Random Linear Combinations**: The prover transmits a random weighted combination of the original and masking polynomials. This guarantees that the verifier cannot reconstruct the original data while still being able to validate correctness.

- **Low-Degree Extensions**: For circuit values defined over finite sets (e.g., gate outputs), Libra applies a low-degree extension to generate matching polynomials that align with the original values at designated points. These extensions are only seen through masked evaluations by the verifier, preserving confidentiality.

By combining these techniques, **Libra achieves a practical, scalable, and secure ZKP system**, suitable for large-scale privacy-preserving computations. Its optimizations bring the theoretical elegance of GKR into real-world applications, opening the door for efficient verification in privacy-sensitive domains like blockchain, secure multiparty computation, and confidential machine learning.

---

## 1. The Evolution and Foundations of Modern Succinct Argument Systems

The evolution of cryptographic proof systems traces back to the pioneering work of Goldreich, Micali, and Wigderson in the 1980s. Since then, the field has developed significantly, particularly in the direction of **succinct non-interactive arguments of knowledge (SNARKs)**—systems that enable a prover to convince a verifier of the correctness of a statement $R(x, w) = 1$ with **sublinear communication complexity** (proof size $O_λ(log t)$ ) and **efficient verification** ($O_λ(n + \log t)$). These systems are foundational to public-key cryptography, digital signatures, secure multiparty computation, and widely used privacy-preserving cryptocurrencies like ZCash and Monero.

Modern SNARKs typically follow a **modular design**, composed of information-theoretic components (e.g., PCPs, linear PCPs, IOPs, and PIOPs) and cryptographic compilers (e.g., polynomial commitment schemes, or PCS). Based on the underlying PCS design, SNARKs can be divided into:

- **Univariate PCS-based SNARKs**, such as Sonic, Marlin, and Plonk
- **Multilinear PCS-based SNARKs**, such as Hyrax, Libra, Spartan, Kopis, Xiphos, Gemini, Orion, Brakedown, and Hyperplonk

They can also be classified by setup requirements:

- **Structured Reference String (SRS)** models, which may be trusted, transparent, or updatable
- **Idealized models**, such as the **Random Oracle Model (ROM)**, **Generic Group Model (GGM)**, and **Algebraic Group Model (AGM)**

Recent research has shifted focus from minimizing proof size and verification time to **reducing prover time**, a critical factor for scaling real-world applications. Notably, systems like Libra, Spartan, and Gemini have achieved **linear-time provers (O(n))**, making them ideal for large-scale computation. Balancing **succinct proofs**, **efficient verification**, and **minimal prover overhead** is now the central challenge and ambition of next-generation SNARKs.

#### Succinct Arguments of Knowledge (SAKs)

Succinct Arguments of Knowledge form the foundation of zero-knowledge proof systems. In this model, a prover $P$ convinces a verifier $V$ that they know a valid witness $w$ for a statement $x$ in an NP relation $R$, without revealing $w$ itself. The system comprises three components:

- **setup algorithm** to generate public parameters
- **prover algorithm** that constructs the proof using the witness
- **verifier algorithm** that checks the proof’s validity

Two core properties are required for soundness and utility:

- **Completeness**: Honest provers always convince the verifier.
- **Knowledge Soundness**: If a malicious prover can convince the verifier, then it is possible to extract a valid witness.

The defining feature of _succinctness_ ensures that both communication and verification complexity are polylogarithmic in the size of the computation (poly(λ, log |w|)), making the proof system practical even for large computations. This "proofs much smaller than computation" property enables powerful applications in verifiable computation, privacy-preserving systems, and blockchain scalability.

#### Polynomial Commitment Schemes (PCS)

At the heart of modern SNARK constructions lies the **Polynomial Commitment Scheme**—a cryptographic primitive that allows a prover to commit to a polynomial and later prove the value of the polynomial at specific points, without revealing the entire polynomial.

A PCS typically includes four algorithms:

- **Setup**: Generates public parameters
- **Commit**: Commits to a polynomial
- **Open**: Opens the commitment at a point
- **Eval**: Proves and verifies evaluations interactively

PCS guarantees three critical properties:

- **Completeness**: Honest proofs are accepted
- **Binding**: A commitment binds to only one polynomial
- **Knowledge Soundness**: Acceptance implies knowledge of the committed polynomial

By applying the **Fiat-Shamir transform**, these interactive protocols can be made non-interactive in the **Random Oracle Model (ROM)**. The key advantage of PCS is its **succinctness**—the size of the commitment and proof depends only on the security parameter, not on the polynomial's degree. This makes PCS the perfect cryptographic compiler to transform idealized information-theoretic PIOPs into practical SNARKs.

The security of many PCS constructions—especially **KZG-based PCS**—relies on the **q-DLOG assumption**: given powers of a secret $\tau$ in two groups $g₁^τ, g₁^τ², ..., g₁^τᵍ, g₂^τ, g₂^τ², ..., g₂^τᵍ$, computing $\tau$ is infeasible.

#### Polynomial Interactive Oracle Proofs (PIOP)

PIOP is a modern, modular approach to building zero-knowledge proof systems. It decomposes the protocol design into two layers:

- **information-theoretic oracle proof** that is secure in an idealized model
- **cryptographic compiler**, usually a PCS, to convert it into a practical system

In a PIOP, the prover provides oracle access to polynomials, and the verifier checks identities over these polynomials by querying points without learning the full polynomials. The **PCS bridges the gap**, turning this oracle access into cryptographically secure commitments and evaluations, preserving succinctness and efficiency.

This decoupling allows researchers to separately optimize the proof protocol and the cryptographic backend, making the design process more flexible and scalable. PIOP-based SNARKs are now the standard in modern zero-knowledge system architecture.

#### Algebraic Foundations of Zero-Knowledge Proofs

The algebraic underpinnings of zkSNARKs rest on concepts like **multilinear polynomials**, **multilinear extensions (MLE)**, and **sumcheck protocols**.

- **Multilinear Extension (MLE)** maps a function defined on the Boolean hypercube $B^\mu$ to a polynomial over $\mathbb{F}^\mu$. A key tool here is the polynomial:

  $eq_\mu(x, y) = \prod_{i=1}^\mu (x_i y_i + (1 - x_i)(1 - y_i))$
  which evaluates to 1 if $x = y$ and 0 otherwise. This enables efficient construction of **Lagrange basis polynomials** and supports interpolation.

- **Sumcheck Protocols** allow the prover to convince the verifier of the sum of a polynomial over the hypercube, reducing exponential complexity $O(2^\mu)$ to linear interaction.

- **Polynomial Decomposition Lemmas** further allow any polynomial $G(X)$ and $q(X)$ to be expressed via a bivariate polynomial $Q(X, Y)$ such that $G(X) = Q(q(X), X)$, facilitating efficient proof composition.

These mathematical constructs serve as the backbone for SNARK systems like **Libra**, which achieves a historic milestone: simultaneously realizing **linear-time proving** and **succinct verification**. Libra addresses one of the longest-standing tensions in zero-knowledge proof systems—balancing prover efficiency with succinctness—through innovative algorithm design. This breakthrough points the way toward future systems capable of supporting truly scalable, verifiable computation.

---

## 2.The Linear-Time Prover Algorithm in the GKR Protocol

The GKR (Goldwasser–Kalai–Rothblum) protocol is an interactive proof system designed to verify the correctness of layered arithmetic circuit computations. It serves as a theoretical foundation for zero-knowledge proof systems like Libra. At the heart of GKR lies the concept of multilinear extension: any function $V: \{0,1\}^\ell \to \mathbb{F}$ can be uniquely extended to a multilinear polynomial $\widetilde{V}: \mathbb{F}^\ell \to \mathbb{F}$, preserving the original values on Boolean inputs.

The GKR protocol verifies circuit evaluations by ascending from the output layer (layer 0) to the input layer (layer $d$), encoding inter-layer relationships via the equation:

$V_i(z) = \sum_{x,y \in \{0,1\}^{s_{i+1}}} \big( add_{i+1}(z,x,y) \cdot (V_{i+1}(x) + V_{i+1}(y)) + mult_{i+1}(z,x,y) \cdot V_{i+1}(x) \cdot V_{i+1}(y) \big)$

Here, $\text{add}_i$ and $\text{mult}_i$ are predicate functions that describe the structure of the circuit connections.

Originally, GKR had a significant efficiency bottleneck. A naive implementation would require $O(S^2)$ time for the prover, where $S$ is the number of gates in a layer (with $s = \log S$). This inefficiency arises because a degree-2s multivariate polynomial over Boolean inputs has at least $2^{2s} = S^2$ monomials.

To improve efficiency, the protocol uses the **sum-check** subprotocol to verify layer relations. A key challenge is to merge multiple claims per round efficiently. The original GKR protocol used evaluation at online points, while later improvements, like those in Libra, adopted the use of **random linear combinations**, reducing prover time from $O(S \log S)$ to linear $O(S)$.

#### The Sum-Check Protocol

The sum-check protocol allows efficient verification of the sum of a multivariate polynomial $f(x_1, ..., x_\ell)$ over Boolean inputs:

$H = \sum_{b_1, ..., b_\ell \in \{0,1\}} f(b_1, ..., b_\ell)$

While the direct computation takes $O(2^\ell)$ time, the sum-check protocol reduces the verifier’s work to polynomial time via an $\ell$-round interactive process. In each round, the prover sends a univariate polynomial obtained by summing out all but one variable. The verifier checks consistency and sends a random challenge $r_i$. In the final round, the verifier evaluates the polynomial at a random point using an oracle.

This reduces the verification from exponential $O(2^\ell)$ to linear $O(\ell \cdot d)$, forming the backbone of the GKR verification process.

#### FunctionEvaluations: Core of the Linear-Time Prover

Libra introduces a **FunctionEvaluations** algorithm to optimize the prover’s work in the sum-check protocol. This algorithm computes evaluations of multilinear extensions at points of the form $(r_1, ..., r_{i-1}, t, b_{i+1}, ..., b_\ell)$, where the $r$’s are verifier’s challenges and $t \in \{0,1,2\}$. It uses dynamic programming and interpolation to incrementally compute all required values in linear time $O(2^\ell)$.

By storing intermediate results in an “accounting table,” the algorithm avoids redundant computation and prepares values for subsequent rounds, ensuring overall linear-time complexity in the input size.

#### Modular Sum-Check and Product Variants

The basic sum-check protocol can be extended to handle products of functions:

$\sum_{b \in \{0,1\}^{\ell-i}} f(r_1, ..., r_{i-1}, t, b) \cdot g(r_1, ..., r_{i-1}, t, b)$

This variant is called **SumCheckProduct**. It requires computing both $f$ and $g$ evaluations using FunctionEvaluations and performing pointwise multiplication to accumulate the sums for each $t \in \{0,1,2\}$.

Thanks to the multilinear structure, each round only needs evaluations at three points, making this process efficient and modular.

#### Handling GKR-Specific Sum-Check Structures

In GKR, the prover often needs to evaluate sums of the form:

$\sum_{x,y \in \{0,1\}^\ell} f_1(g,x,y) \cdot f_2(x) \cdot f_3(y)$

Where:

- $f_1$ is a sparse multilinear extension with $O(2^\ell)$ nonzero entries,
- $f_2$, $f_3$ are multilinear functions derived from arrays,
- $g \in \mathbb{F}^\ell$ is a fixed point.

The optimized GKR prover breaks this into **two phases**:

- **Phase One:** Sum over $x$, treating the inner sum over $y$ as part of a precomputed function $h_g(x)$. This reduces the double sum to a single sum:

$\sum_{x \in \{0,1\}^\ell} f_2(x) \cdot h_g(x)$

Initialization of $h_g(x)$ uses the sparsity of $f_1$ and the multilinearity of the identity function $I(g,z)$, enabling efficient construction in $O(2^\ell)$ time.

- **Phase Two:** After $x$ is fixed via random challenges $u_1, ..., u_\ell$, compute:

$\sum_{y \in \{0,1\}^\ell} f_1(g, u, y) \cdot f_3(y)$

The sparse structure and precomputation again enable this in linear time.

This two-phase approach avoids the naive $O(2^{2\ell})$ cost and achieves the desired linear time in $O(2^\ell)$, leveraging both the multilinear structure and the sparsity of the involved functions.

#### Generalization to k-ary Sum-Checks

The approach generalizes naturally to higher-arity versions of the form:

$\sum_{x_1, ..., x_k \in \{0,1\}^\ell} f_0(g,x_1,...,x_k) \cdot f_1(x_1) \cdot ... \cdot f_k(x_k)$

Where:

- $f_0$ is sparse,
- $f_1, ..., f_k$ are multilinear.

The general strategy is to break this into $k$ phases, each reducing one variable group while preserving linear time complexity. In each phase, initialization and SumCheckProduct leverage precomputed evaluations and the function's structure.

---

## 3. Zero-Knowledge GKR Protocol

The zkVPD (Zero-Knowledge Verifiable Polynomial Delegation) scheme is a key cryptographic component in the Libra system, designed to enable a prover to commit to a polynomial $f$ and prove a specific evaluation $f(t) = y$ at a given point without revealing any additional information about the polynomial. The scheme consists of five algorithms: KeyGen (key generation), Commit (commitment creation), CheckComm (commitment validation), Open (point value proof generation), and Verify (proof verification). It ensures three crucial security properties:

- **Perfect Completeness**: Honest execution always succeeds.
- **Binding**: The prover cannot claim different evaluations for the same commitment.
- **Zero-Knowledge**: The interaction is indistinguishable from a simulation, ensuring that the verifier cannot learn anything other than the claimed evaluation point.

In the Libra system, the zkVPD is used to implement the zero-knowledge feature of the GKR protocol, committing to polynomial values at different layers of the circuit and utilizing the Fiat-Shamir transform for non-interactive verification. The scheme allows for efficient verification while maintaining privacy, often implemented using bilinear pairings, such as the KZG commitment, and combined with small masking polynomials to ensure privacy while verifying the correctness of the computation.

#### Key Innovations in Libra's zkVPD

Libra innovatively addresses the potential information leakage issue in GKR protocols, where the prover sends evaluation values of polynomials at random points, which may reveal information about the circuit. Building on Chiesa et al.'s approach to random polynomial masking, Libra demonstrates that a small mask polynomial $g(x_1, ..., x_\ell) = a_0 + g_1(x_1) + g_2(x_2) + ... + g_\ell(x_\ell)$ is sufficient to achieve zero-knowledge, where each $g_i$ is a random degree $d$ univariate polynomial. The mask polynomial size is reduced to $O(d \cdot \ell)$ rather than the exponential size of the original polynomial $f$, providing a more efficient implementation.

This method uses a linear combination of a commitment $H + \rho G$, where $G$ is the masking polynomial, and verifies the evaluation values at specific points by opening the mask polynomial. The zero-knowledge property is proven by constructing a simulator $S$ that can produce a view indistinguishable from a real interaction.

#### Enhancements for Information Leakage

The GKR protocol still has potential information leakage points. At the end of the zero-knowledge and checking phases, the verifier (V) must query an oracle at random points to evaluate polynomials, which leads to the leakage of evaluation values for the polynomials $\tilde{V}_i$ defined by the i-th layer of two circuits, specifically $\tilde{V}_i(u)$ and $\tilde{V}_i(v)$. The Libra system’s zero-knowledge GKR protocol addresses the information leakage issue present in the GKR protocol through a series of innovative techniques. Even after applying the zero-knowledge and checking protocols, the verifier can still gain information about the circuits by querying polynomial evaluation values $\tilde{V}_i(u)$ and $\tilde{V}_i(v)$.

Libra builds on the low-degree extension idea proposed by Chiesa et al., where the polynomial $V̇ᵢ(z)=Ṽᵢ(z)+Zᵢ(z)∑_{w∈{0,1}^λ}Rᵢ(z,w)$, with $Z(z) = ∏{i=1}^{sᵢ} zᵢ(1-zᵢ)$, guarantees that $Z(z) = 0$ for all $z \in {0,1}^{s_i}$. Here, $R_i(z,w)$ is a random low-degree polynomial, and $\lambda$ is the security parameter. However, the key innovation in Libra’s approach is the proof that the mask polynomial $R_i$ can be drastically simplified to a small polynomial containing only two variables, each with degree 2, as opposed to the original approach that requires a polynomial with $s_i + 2s_{i+1} + \lambda$ variables.

The complete protocol flow includes initialization, output layer processing, zero-knowledge and checking execution, intermediate layer recursive verification, and final input layer verification. The protocol rigorously proves its zero-knowledge property by constructing a simulator $S$ that uses a zero-knowledge and checking simulator as a subroutine. It then proves that the view generated by this simulator is indistinguishable from the real protocol. The critical proof steps are based on row-reduction analysis of the matrix formed by the four evaluation values of $R_i$, ensuring that the matrix is full rank when $u_1 \neq v_1$ and $2c^2 - 1 \neq 0 \mod p$, which guarantees the linear independence and uniform distribution of the four evaluation values.

#### Zero-Knowledge GKR Protocol in Libra

The full protocol follows a layered verification process, from the output layer to the input layer, ensuring each layer of the circuit is validated. The key stages include:

- **Initialization Phase**: The prover (P) sends the circuit output `out` to the verifier (V), claiming the computed result.

- **Mask Polynomial Preparation**: The prover (P) randomly selects quadratic small mask polynomials $R_1(z_1, w), \ldots, R_k(z_1, w)$ for each layer of the circuit and sends commitments to the verifier (V).

- **Output Layer Processing**: The verifier (V) defines $\dot{V}_0(z) = \tilde{V}_0(z)$ as the multilinear extension of `out`, evaluates it at a random point $g^{(0)}$, and sends it to the prover (P).

- **First Layer Verification**:

  - The prover (P) and verifier (V) perform zero-knowledge checks and verify the relationship between the output layer and the first layer:

  $V̇₀(g^{(0)}) = ∑_{x,y∈{0,1}^s₁} m̃ult₁(g^{(0)},x,y)(V̇₁(x)·V̇₁(y)) + ãdd₁(g^{(0)},x,y)(V̇₁(x)+V̇₁(y))$

  - The verifier (V) receives the evaluation values for two points $\dot{V}_1(u^{(1)})$ and $\dot{V}_1(v^{(1)})$.
  - The verifier (V) computes the gate function values and verifies consistency:

  $\text{mult}_1(g^{(0)}, u^{(1)}, v^{(1)}) \cdot \dot{V}_1(u^{(1)}) \cdot \dot{V}_1(v^{(1)}) + \text{add}_1(g^{(0)}, u^{(1)}, v^{(1)})(\dot{V}_1(u^{(1)}) + \dot{V}_1(v^{(1)}))$

- **Intermediate Layer Recursion** (for $i = 1, \ldots, d-1$):

  - The verifier (V) selects random coefficients $\alpha^{(i)}, \beta^{(i)}$ for the linear combination.
  - Zero-knowledge checks are performed, and the relationship between the current layer and the next layer is verified:

  $α^{(i)}·V̇ᵢ(u^{(i)}) + β^{(i)}·V̇ᵢ(v^{(i)}) = ∑_{x,y∈{0,1}^sᵢ₊₁,w∈{0,1}} ( I(0̄,w)·Multᵢ₊₁(x,y)(V̇ᵢ₊₁(x)·V̇ᵢ₊₁(y)) + Addᵢ₊₁(x,y)(V̇ᵢ₊₁(x)+V̇ᵢ₊₁(y)) + I((x,y),0̄)(α⁽ⁱ⁾·Zᵢ(u^{(i)})·Rᵢ(u₁^{(i)},w) + β^{(i)}·Zᵢ(v^{(i)})·Rᵢ(v₁^{(i)},w)))$

  $\left. + I((x,y),0̄)(\alpha^{(i)} \cdot Z_i(u^{(i)}) \cdot R_i(u_1^{(i)},w) + \beta^{(i)} \cdot Z_i(v^{(i)}) \cdot R_i(v_1^{(i)},w)) \right)$

  - The prover (P) sends the evaluation values for the next layer's two points
    $\dot{V}_{i+1}(u^{(i+1)})$ and $\dot{V}_{i+1}(v^{(i+1)})$.

  - The verifier (V) computes necessary coefficients and performs local validation:

    - $aᵢ₊₁ = α^{(i)}·m̃ultᵢ₊₁(u^{(i)},u^{(i+1)},v^{(i+1)}) + β^{(i)}·m̃ultᵢ₊₁(v^{(i)},u^{(i+1)},v^{(i+1)})$
    - $bᵢ₊₁ = α^{(i)}·ãddᵢ₊₁(u^{(i)},u^{(i+1)},v^{(i+1)}) + β^{(i)}·ãddᵢ₊₁(v^{(i)},u^{(i+1)},v^{(i+1)})$
    - $Z_i(u^{(i)}), Z_i(v^{(i)}), I(0̄,c^{(i)}), I((u^{(i+1)}, v^{(i+1)}),0̄)$

  - The value of $R_i$ at specific two points is opened for verification:

  $I(0̄,c^{(i)})(aᵢ₊₁(V̇ᵢ₊₁(u^{(i+1)})·V̇ᵢ₊₁(v^{(i+1)}))+bᵢ₊₁(V̇ᵢ₊₁(u^{(i+1)})+V̇ᵢ₊₁(v^{(i+1)})))+
I((u^{(i+1)},v^{(i+1)}),0̄)(α^{(i)}·Zᵢ(u^{(i)})·Rᵢ(u₁^{(i)},c^{(i)}) + β^{(i)}·Zᵢ(v^{(i)})·Rᵢ(v₁^{(i)},c^{(i)}))$

  $+ I((u^{(i+1)}, v^{(i+1)}),0̄)(\alpha^{(i)} \cdot Z_i(u^{(i)}) \cdot R_i(u_1^{(i)}, c^{(i)}) + \beta^{(i)} \cdot Z_i(v^{(i)}) \cdot R_i(v_1^{(i)}, c^{(i)}))$

  - After completing the final consistency check, the process moves to the next layer.

- **Input Layer Verification**:

  - The verifier (V) receives the two declarations for the input layer: $\dot{V}_k(u^{(d)})$ and $\dot{V}_k(v^{(d)})$.
  - The values of $R_k$ at four points are opened.
  - The mask form is verified for consistency with the actual inputs:

  $V̇ₖ(u^{(d)}) = Ṽₖ(u^{(d)}) + Zₖ(u^{(d)})∑_{w∈{0,1}} Rₖ(u₁^{(d)},w)  V̇ₖ(v^{(d)}) = Ṽₖ(v^{(d)}) + Zₖ(v^{(d)})∑_{w∈{0,1}} Rₖ(v₁^{(d)},w)$

  - Based on the verification results, the proof is either accepted or rejected.

#### Input Layer zkVPD Protocol

The zero-knowledge Verifiable Polynomial Delegation (zkVPD) protocol of the Libra system is another key component in implementing the zero-knowledge GKR protocol. It is divided into two application parts: one provides simplified and efficient commitment and opening mechanisms for small mask polynomials ($g_i$ and $R_i$) in the intermediate layers (from layer 1 to layer d-1) with complexities of $O(s_i)$ and $O(1)$, respectively; the other provides customized zkVPD handling of commitments and openings for the input layer (layer d) polynomial $V̇_d(z)$. The input layer protocol cleverly optimizes the decomposition of $V̇_d(z)$ into a multilinear polynomial $Ṽ_d(z)$ and a masking term $Z_d(z)R_d(z_1)$, and these are committed and verified using the homomorphic properties together. The masking polynomial is simplified into a linear form $R_d(x_1) = a_0 + a_1x_1$.

The complete implementation includes key generation based on bilinear pairings, commitment calculation, opening protocols, and verification steps, ensuring its completeness, reliability, and zero-knowledge properties. Additionally, an algorithm for multilinear extension commitments and openings with $O(2^ℓ)$ linear time complexity and a verification algorithm with $O(ℓ)$ complexity are provided in the appendix. Through these targeted optimizations and efficient implementations, the zkVPD protocol ensures that the entire Libra system achieves efficient, feasible proof generation and verification while maintaining zero-knowledge properties.

The input layer zkVPD protocol specifically handles polynomials of the form $V̇(x) = Ṽ(x) + Z(x)R(x₁)$. This protocol is realized through bilinear pairing cryptography, and its main steps are as follows:

- **Key Generation (KeyGen)**:

  - Randomly choose parameters $\alpha$ and $t_1$ to $t_{\ell+1}$.
  - Generate bilinear pairing parameters $bp$.
  - Compute public parameters $pp$, which include group elements and bilinear pairing parameters:
    - Bilinear pairing parameters $bp$
    - $g^\alpha, g^{t_{\ell+1}}, g^{\alpha t_{\ell+1}}$
    - Set ${g^{\prod_{i \in W} t_i}, g^{\alpha \prod_{i \in W} t_i}}$, where $W \in W_\ell$ (with $W_\ell$ being the set of all subsets of {1, ..., $\ell$}).
  - Set verification parameters $vp = (bp, g^{t_1}, ..., g^{t_{\ell+1}}, g^\alpha)$.

- **Commitment Generation (Commit)**:

  - Compute commitments for the multilinear part $Ṽ$ and the linear part $R$ separately.
  - Use random masks $r_V$ and $r_R$ to hide the actual values.
  - Output four commitments: $com = (c_1, c_2, c_3, c_4)$, where each part contains the original values and $\alpha$-multiplied forms.

- **Commitment Check (CheckComm)**:

  - Use bilinear pairings to verify consistency between $c_1$ and $c_2$, and between $c_3$ and $c_4$.
  - Check $e(c_1, g^\alpha) = e(c_2, g)$ and $e(c_3, g^\alpha) = e(c_4, g)$.

- **Opening Protocol (Open)**:

  - Randomly choose auxiliary values $r_1$ to $r_\ell$.
  - Compute polynomial $q_i$, such that the polynomial difference can be expressed as a linear combination of variable differences.
  - Generate evaluation proof $\pi$, containing $\ell+1$ pairs of group elements.
  - Compute and output the evaluation value $y = Ṽ(u) + Z(u)R(u_1)$ along with its proof.

- **Verification (Verify)**:

  - Check the internal consistency of proof $\pi$.
  - Use bilinear pairings to verify the relationship between the evaluation value $y$ and the commitment $com$.
  - Verify the evaluation value using the equation:

    $e(c_1 \cdot c_3^{Z(u)}/g^y, g) = \prod_{i=1}^\ell e(\pi_i, g^{t_i - u_i}) \cdot e(\pi_{\ell+1}, g^{t_{\ell+1}})$

#### Optimizations and Complexity

The Libra system introduces several optimizations to reduce the complexity of zkVPD. The commitment and opening processes are efficient, with complexities of $O(s_i)$ for the mask polynomials and $O(1)$ for the opening mechanism. The input layer protocol is particularly optimized, leveraging bilinear pairings and homomorphic properties to validate the polynomial efficiently.

These innovations ensure that Libra's zkVPD maintains the desired zero-knowledge property while offering efficient proof generation and verification. The overall time complexity for the multi-linear commitment and opening algorithm is linear in $O(2^\ell)$, and the verification complexity is $O(\ell)$, ensuring that the system is both practical and secure for use in the Libra protocol.

The input layer zkVPD protocol ensures the validity of the polynomial evaluation without revealing any sensitive information, allowing Libra to achieve both privacy and efficiency in its zero-knowledge proofs.

---

<details><summary><b> Code </b></summary>

<details><summary><b> linear_gkr/zk_prover.cpp </b></summary>

```c

quadratic_poly zk_prover::sumcheck_phase1_update(prime_field::field_element previous_random, int current_bit)
{
	std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
	quadratic_poly ret = quadratic_poly(prime_field::field_element(0), prime_field::field_element(0), prime_field::field_element(0));
	for(int i = 0; i < (total_uv >> 1); ++i)
	{
		prime_field::field_element zero_value, one_value;
		int g_zero = i << 1, g_one = i << 1 | 1;
		if(current_bit == 0)
		{
			V_mult_add[i].b = V_mult_add[g_zero].b;
			V_mult_add[i].a = V_mult_add[g_one].b - V_mult_add[i].b;

			addV_array[i].b = addV_array[g_zero].b;
			addV_array[i].a = addV_array[g_one].b - addV_array[i].b;

			add_mult_sum[i].b = add_mult_sum[g_zero].b;
			add_mult_sum[i].a = add_mult_sum[g_one].b - add_mult_sum[i].b;

		}
		else
		{
			V_mult_add[i].b.value = (V_mult_add[g_zero].a.value * previous_random.value + V_mult_add[g_zero].b.value) % prime_field::mod;
			V_mult_add[i].a.value = (V_mult_add[g_one].a.value * previous_random.value + V_mult_add[g_one].b.value - V_mult_add[i].b.value + prime_field::mod_512) % prime_field::mod;

			addV_array[i].b.value = (addV_array[g_zero].a.value * previous_random.value + addV_array[g_zero].b.value) % prime_field::mod;
			addV_array[i].a.value = (addV_array[g_one].a.value * previous_random.value + addV_array[g_one].b.value - addV_array[i].b.value + prime_field::mod_512) % prime_field::mod;

			add_mult_sum[i].b.value = (add_mult_sum[g_zero].a.value * previous_random.value + add_mult_sum[g_zero].b.value) % prime_field::mod;
			add_mult_sum[i].a.value = (add_mult_sum[g_one].a.value * previous_random.value + add_mult_sum[g_one].b.value - add_mult_sum[i].b.value + prime_field::mod_512) % prime_field::mod;

		}
		ret.a.value = (ret.a.value + add_mult_sum[i].a.value * V_mult_add[i].a.value) % prime_field::mod;
		ret.b.value = (ret.b.value + add_mult_sum[i].a.value * V_mult_add[i].b.value + add_mult_sum[i].b.value * V_mult_add[i].a.value
									+ addV_array[i].a.value) % prime_field::mod;
		ret.c.value = (ret.c.value + add_mult_sum[i].b.value * V_mult_add[i].b.value
									+ addV_array[i].b.value) % prime_field::mod;
	}

	total_uv >>= 1;
	Iuv = Iuv * (prime_field::field_element(1) - previous_random);
	if(current_bit > 0){
		maskR_sumcu = maskR_sumcu * (prime_field::field_element(1) - previous_random);
		maskR_sumcv = maskR_sumcv * (prime_field::field_element(1) - previous_random);

		Zu = Zu * (prime_field::field_element(1) - previous_random) * previous_random;
	}

	ret.b = ret.b - maskR_sumcu - maskR_sumcv;
	ret.c = ret.c + maskR_sumcu + maskR_sumcv;



	//compute with sumcheck maskpol

	prime_field::field_element tmp1, tmp2;
	tmp1.value = maskpoly[current_bit << 1].value;
	tmp2.value = maskpoly[(current_bit << 1) + 1].value;

	for(int i = 0; i < length_u + length_v - current_bit; i++){
		tmp1 = tmp1 + tmp1;
		tmp2 = tmp2 + tmp2;
	}

	maskpoly_sumc = (maskpoly_sumc - tmp1 - tmp2) * inv_2;

	prime_field::field_element tmp3;
	if(current_bit > 0){
		maskpoly_sumr = maskpoly_sumr + maskpoly[(current_bit << 1) - 2] * previous_random * previous_random + maskpoly[(current_bit << 1) - 1] * previous_random;
		tmp3 = maskpoly_sumr;
		for(int i = 0; i < length_u + length_v - current_bit; i++)
			tmp3 = tmp3 + tmp3;
	}

	ret.a = ret.a + tmp1;
	ret.b = ret.b + tmp2;
	ret.c = ret.c + maskpoly_sumc + tmp3;
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	total_time += time_span.count();
	return ret;
}


quadratic_poly zk_prover::sumcheck_phase2_update(prime_field::field_element previous_random, int current_bit)
{
	std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
	quadratic_poly ret = quadratic_poly(prime_field::field_element(0), prime_field::field_element(0), prime_field::field_element(0));
	for(int i = 0; i < (total_uv >> 1); ++i)
	{
		int g_zero = i << 1, g_one = i << 1 | 1;
		if(current_bit == 0)
		{
			V_mult_add[i].b = V_mult_add[g_zero].b;
			V_mult_add[i].a = V_mult_add[g_one].b - V_mult_add[i].b;

			addV_array[i].b = addV_array[g_zero].b;
			addV_array[i].a = addV_array[g_one].b - addV_array[i].b;

			add_mult_sum[i].b = add_mult_sum[g_zero].b;
			add_mult_sum[i].a = add_mult_sum[g_one].b - add_mult_sum[i].b;
		}
		else
		{

			V_mult_add[i].b.value = (V_mult_add[g_zero].a.value * previous_random.value + V_mult_add[g_zero].b.value) % prime_field::mod;
			V_mult_add[i].a.value = (V_mult_add[g_one].a.value * previous_random.value + V_mult_add[g_one].b.value + prime_field::mod_512 - V_mult_add[i].b.value) % prime_field::mod;

			addV_array[i].b.value = (addV_array[g_zero].a.value * previous_random.value + addV_array[g_zero].b.value) % prime_field::mod;
			addV_array[i].a.value = (addV_array[g_one].a.value * previous_random.value + addV_array[g_one].b.value + prime_field::mod_512 - addV_array[i].b.value) % prime_field::mod;

			add_mult_sum[i].b.value = (add_mult_sum[g_zero].a.value * previous_random.value + add_mult_sum[g_zero].b.value) % prime_field::mod;
			add_mult_sum[i].a.value = (add_mult_sum[g_one].a.value * previous_random.value + add_mult_sum[g_one].b.value + prime_field::mod_512 - add_mult_sum[i].b.value) % prime_field::mod;
		}

		ret.a.value = (ret.a.value + add_mult_sum[i].a.value * V_mult_add[i].a.value) % prime_field::mod;
		ret.b.value = (ret.b.value + add_mult_sum[i].a.value * V_mult_add[i].b.value
									+	add_mult_sum[i].b.value * V_mult_add[i].a.value
									+ addV_array[i].a.value) % prime_field::mod;
		ret.c.value = (ret.c.value + add_mult_sum[i].b.value * V_mult_add[i].b.value
									+ addV_array[i].b.value) % prime_field::mod;
	}

	total_uv >>= 1;
	//maskR
	if(current_bit > 0)
		Iuv = Iuv * (prime_field::field_element(1) - previous_random);

	if(current_bit > 0){
		maskR_sumcu = maskR_sumcu * (prime_field::field_element(1) - previous_random);
		maskR_sumcv = maskR_sumcv * (prime_field::field_element(1) - previous_random);
		Zv = Zv * (prime_field::field_element(1) - previous_random) * previous_random;
	}
	ret.b = ret.b - maskR_sumcu - maskR_sumcv;
	ret.c = ret.c + maskR_sumcu + maskR_sumcv;


	//mask sumcheck
	int current = current_bit + length_u;

	prime_field::field_element tmp1, tmp2;
	tmp1.value = maskpoly[current << 1].value;
	tmp2.value = maskpoly[(current << 1) + 1].value;
	for(int i = 0; i < length_u + length_v - current; i++){
		tmp1 = tmp1 + tmp1;
		tmp2 = tmp2 + tmp2;
	}
	maskpoly_sumc = (maskpoly_sumc - tmp1 - tmp2) * inv_2;

	prime_field::field_element tmp3;
	maskpoly_sumr = maskpoly_sumr + maskpoly[(current << 1) - 2] * previous_random * previous_random + maskpoly[(current << 1) - 1] * previous_random;

	tmp3 = maskpoly_sumr;
	for(int i = 0; i < length_u + length_v - current; i++)
		tmp3 = tmp3 + tmp3;


	ret.a.value = (ret.a.value + tmp1.value) % prime_field::mod;
	ret.b.value = (ret.b.value + tmp2.value) % prime_field::mod;
	ret.c.value = (ret.c.value + maskpoly_sumc.value + tmp3.value) % prime_field::mod;
	ret.a.value = (ret.a.value + prime_field::mod) % prime_field::mod;
	ret.b.value = (ret.b.value + prime_field::mod) % prime_field::mod;
	ret.c.value = (ret.c.value + prime_field::mod) % prime_field::mod;
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	total_time += time_span.count();
	return ret;
}

```

</details>

<details><summary><b> linear_gkr/zk_verifier.cpp </b></summary>

```c


bool zk_verifier::verify(const char* output_path)
{
	for(int i = C.total_depth - 1; i >= 1; --i)
	{
	    // Initialization phase
	    auto rho = trans.random();
	    auto digest_maskR = p->sumcheck_init(i, C.circuit[i].bit_length, C.circuit[i - 1].bit_length, C.circuit[i - 1].bit_length, alpha, beta, r_0, r_1, one_minus_r_0, one_minus_r_1);

	    // First phase verification - sumcheck for u variables
	    p->sumcheck_phase1_init();
	    for(int j = 0; j < C.circuit[i - 1].bit_length; ++j) {
	        if(j == C.circuit[i - 1].bit_length - 1) {
	            quintuple_poly poly = p->sumcheck_phase1_updatelastbit(previous_random, j);
	            // Verify this polynomial
	            if(poly.eval(0) + poly.eval(1) != alpha_beta_sum) {
	                return false;  // Verification failed
	            }
	            alpha_beta_sum = poly.eval(r_u[j]);
	        } else {
	            quadratic_poly poly = p->sumcheck_phase1_update(previous_random, j);
	            // Verify this polynomial
	            if(poly.eval(0) + poly.eval(1) != alpha_beta_sum) {
	                return false;  // Verification failed
	            }
	            alpha_beta_sum = poly.eval(r_u[j]);
	        }
	    }

	    // Second phase verification - sumcheck for v variables
	    p->sumcheck_phase2_init(previous_random, r_u, one_minus_r_u);
	    // Similar verification loop...

	    // Calculate predicates and perform final verification
	    beta_init(i, alpha, beta, r_0, r_1, r_u, r_v, one_minus_r_0, one_minus_r_1, one_minus_r_u, one_minus_r_v);
	    auto predicates_value = predicates(i, r_0, r_1, r_u, r_v, alpha, beta);

	    // Prepare for next round
	    alpha = tmp_alpha[0];
	    beta = tmp_beta[0];
	    // ...
	 }
}

vector<prime_field::field_element> zk_verifier::predicates(int depth, prime_field::field_element *r_0, prime_field::field_element *r_1, prime_field::field_element *r_u, prime_field::field_element *r_v, prime_field::field_element alpha, prime_field::field_element beta)
{
    // ...
    for(int i = 0; i < (1 << C.circuit[depth].bit_length); ++i)
    {
        int g = i, u = C.circuit[depth].gates[i].u, v = C.circuit[depth].gates[i].v;
        switch(C.circuit[depth].gates[i].ty)
        {
            case 1: // Multiplication gate
            {
                int g_first_half = g & ((1 << first_half_g) - 1);
                int g_second_half = (g >> first_half_g);
                int u_first_half = u & ((1 << first_half_uv) - 1);
                int u_second_half = u >> first_half_uv;
                int v_first_half = v & ((1 << first_half_uv) - 1);
                int v_second_half = v >> first_half_uv;

                ret[1].value = ret[1].value +
                    (beta_g_r0_first_half[g_first_half].value * beta_g_r0_second_half[g_second_half].value +
                     beta_g_r1_first_half[g_first_half].value * beta_g_r1_second_half[g_second_half].value) % prime_field::mod *
                    (beta_u_first_half[u_first_half].value * beta_u_second_half[u_second_half].value % prime_field::mod) % prime_field::mod *
                    (beta_v_first_half[v_first_half].value * beta_v_second_half[v_second_half].value % prime_field::mod);

                ret[1].value = ret[1].value % prime_field::mod;
                break;
            }
            // Other gate types...
        }
    }
    // ...
    return ret;
}

```

</details>

<details><summary><b> linear_gkr/prime_field.cpp </b></summary>

```c

field_element field_element::operator + (const field_element &b) const
{
    field_element ret;
    ret.value = (b.value + value);
    if(ret.value >= mod_512)
        ret.value = ret.value + minus_mod_512;
    return ret;
}

field_element field_element::operator * (const field_element &b) const
{
    field_element ret;
    ret.value = (b.value * value) % mod;
    return ret;
}

field_element field_element::operator - (const field_element &b) const
{
    field_element ret;
    if(value >= b.value)
        ret.value = value - b.value;
    else
        ret.value = value + mod_512 - b.value;
    return ret;
}

u256b u256b::operator * (const u256b &x) const
{
    u256b ret;
    __uint128_t lolo = (__uint128_t)lo * (__uint128_t)x.lo;

    __uint128_t lomid1 = (__uint128_t)mid * (__uint128_t)x.lo;
    __uint128_t lomid2 = (__uint128_t)lo * (__uint128_t)x.mid;

    // ... More computation ...

    return ret;
}

u512b u512b::operator % (const u256b &x) const
{
    // Barrett reduction algorithm
    if(lo == 0 && mid == 0 && hi.lo == 0 && hi.mid == 0 && hi.hi == 0)
        return *this;
    u512b hi_factor = (u512b)hi * (u512b)my_factor;
    // ... More computation ...
    return result;
}

u512b u512b::operator + (const u512b &x) const
{
    u512b ret;
    __uint128_t carry, carry2;
    ret.lo = lo + x.lo;
    carry = ret.lo < lo;  // Detect overflow
    ret.mid = mid + x.mid + carry;
    // ... Handle more carries ...
    return ret;
}

```

</details>

<details><summary><b> linear_gkr/polynomial.cpp </b></summary>

```c

prime_field::field_element quadratic_poly::eval(const prime_field::field_element &x) const
{
    return ((a * x) + b) * x + c;
}

prime_field::field_element linear_poly::eval(const prime_field::field_element &x) const
{
    return a * x + b;
}

quadratic_poly quadratic_poly::operator + (const quadratic_poly &x) const
{
    return quadratic_poly(a + x.a, b + x.b, c + x.c);
}

linear_poly linear_poly::operator + (const linear_poly &x) const
{
    return linear_poly(a + x.a, b + x.b);
}

cubic_poly quadratic_poly::operator * (const linear_poly &x) const
{
    return cubic_poly(a * x.a, a * x.b + b * x.a, b * x.b + c * x.a, c * x.b);
}

quadratic_poly linear_poly::operator * (const linear_poly &x) const
{
    return quadratic_poly(a * x.a, a * x.b + b * x.a, b * x.b);
}

```

</details>

<details><summary><b> linear_gkr/prover_fast_track.cpp </b></summary>

```c

prime_field::field_element* prover::evaluate()
{
	std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
	circuit_value[0] = new prime_field::field_element[(1 << C.circuit[0].bit_length)];
	for(int i = 0; i < (1 << C.circuit[0].bit_length); ++i)
	{
		int g, u, v, ty;
		g = i;
		u = C.circuit[0].gates[g].u;
		v = C.circuit[0].gates[g].v;
		ty = C.circuit[0].gates[g].ty;
		assert(ty == 3 || ty == 2);
		circuit_value[0][g] = prime_field::field_element(u);
	}
	assert(C.total_depth < 1000000);
	for(int i = 1; i < C.total_depth; ++i)
	{
		circuit_value[i] = new prime_field::field_element[(1 << C.circuit[i].bit_length)];
		for(int j = 0; j < (1 << C.circuit[i].bit_length); ++j)
		{
			int g, u, v, ty;
			g = j;
			ty = C.circuit[i].gates[g].ty;
			u = C.circuit[i].gates[g].u;
			v = C.circuit[i].gates[g].v;
			if(ty == 0)
			{
				circuit_value[i][g] = circuit_value[i - 1][u] + circuit_value[i - 1][v];
			}
			else if(ty == 1)
			{
				circuit_value[i][g] = circuit_value[i - 1][u] * circuit_value[i - 1][v];
			}
			else if(ty == 2)
			{
				circuit_value[i][g] = prime_field::field_element(0);
			}
			else if(ty == 3)
			{
				circuit_value[i][g] = prime_field::field_element(u);
			}
			else if(ty == 4)
			{
				circuit_value[i][g] = circuit_value[i - 1][u];
			}
			else if(ty == 5)
			{
				circuit_value[i][g] = prime_field::field_element(0);
				for(int k = u; k < v; ++k)
					circuit_value[i][g] = circuit_value[i][g] + circuit_value[i - 1][k];
			}
			else if(ty == 6)
			{
				circuit_value[i][g] = prime_field::field_element(1) - circuit_value[i - 1][u];
			}
			else if(ty == 7)
			{
				circuit_value[i][g] = circuit_value[i - 1][u] - circuit_value[i - 1][v];
			}
			else if(ty == 8)
			{
				auto &x = circuit_value[i - 1][u], &y = circuit_value[i - 1][v];
				circuit_value[i][g] = x + y - prime_field::field_element(2) * x * y;
			}
			else if(ty == 9)
			{
				auto &x = circuit_value[i - 1][u], &y = circuit_value[i - 1][v];
				circuit_value[i][g] = y - x * y;
			}
			else if(ty == 10)
			{
				circuit_value[i][g] = circuit_value[i - 1][u];
			}
			else
			{
				assert(false);
			}
		}
	}

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cerr << "total evaluation time: " << time_span.count() << " seconds." << std::endl;
	return circuit_value[C.total_depth - 1];
}

quadratic_poly prover::sumcheck_phase1_update(prime_field::field_element previous_random, int current_bit)
{
	std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
	quadratic_poly ret = quadratic_poly(prime_field::field_element(0), prime_field::field_element(0), prime_field::field_element(0));
	for(int i = 0; i < (total_uv >> 1); ++i)
	{
		prime_field::field_element zero_value, one_value;
		int g_zero = i << 1, g_one = i << 1 | 1;
		if(current_bit == 0)
		{
			V_mult_add[i].b = V_mult_add[g_zero].b;
			V_mult_add[i].a = V_mult_add[g_one].b - V_mult_add[i].b;

			addV_array[i].b = addV_array[g_zero].b;
			addV_array[i].a = addV_array[g_one].b - addV_array[i].b;

			add_mult_sum[i].b = add_mult_sum[g_zero].b;
			add_mult_sum[i].a = add_mult_sum[g_one].b - add_mult_sum[i].b;

		}
		else
		{
			V_mult_add[i].b.value = (V_mult_add[g_zero].a.value * previous_random.value + V_mult_add[g_zero].b.value) % prime_field::mod;
			V_mult_add[i].a.value = (V_mult_add[g_one].a.value * previous_random.value + V_mult_add[g_one].b.value - V_mult_add[i].b.value + prime_field::mod) % prime_field::mod;

			addV_array[i].b.value = (addV_array[g_zero].a.value * previous_random.value + addV_array[g_zero].b.value) % prime_field::mod;
			addV_array[i].a.value = (addV_array[g_one].a.value * previous_random.value + addV_array[g_one].b.value - addV_array[i].b.value + prime_field::mod) % prime_field::mod;

			add_mult_sum[i].b.value = (add_mult_sum[g_zero].a.value * previous_random.value + add_mult_sum[g_zero].b.value) % prime_field::mod;
			add_mult_sum[i].a.value = (add_mult_sum[g_one].a.value * previous_random.value + add_mult_sum[g_one].b.value - add_mult_sum[i].b.value + prime_field::mod) % prime_field::mod;

		}
		ret.a.value = (ret.a.value + add_mult_sum[i].a.value * V_mult_add[i].a.value) % prime_field::mod;
		ret.b.value = (ret.b.value + add_mult_sum[i].a.value * V_mult_add[i].b.value + add_mult_sum[i].b.value * V_mult_add[i].a.value
									+ addV_array[i].a.value) % prime_field::mod;
		ret.c.value = (ret.c.value + add_mult_sum[i].b.value * V_mult_add[i].b.value
									+ addV_array[i].b.value) % prime_field::mod;
	}

	total_uv >>= 1;
	ret.a.value = ret.a.value % prime_field::mod;
	ret.b.value = ret.b.value % prime_field::mod;
	ret.c.value = ret.c.value % prime_field::mod;

	ret.a.value = (ret.a.value + prime_field::mod) % prime_field::mod;
	ret.b.value = (ret.b.value + prime_field::mod) % prime_field::mod;
	ret.c.value = (ret.c.value + prime_field::mod) % prime_field::mod;


	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	total_time += time_span.count();
	return ret;
}

quadratic_poly prover::sumcheck_phase2_update(prime_field::field_element previous_random, int current_bit)
{
	std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
	quadratic_poly ret = quadratic_poly(prime_field::field_element(0), prime_field::field_element(0), prime_field::field_element(0));
	for(int i = 0; i < (total_uv >> 1); ++i)
	{
		int g_zero = i << 1, g_one = i << 1 | 1;
		if(current_bit == 0)
		{
			V_mult_add[i].b = V_mult_add[g_zero].b;
			V_mult_add[i].a = V_mult_add[g_one].b - V_mult_add[i].b;

			addV_array[i].b = addV_array[g_zero].b;
			addV_array[i].a = addV_array[g_one].b - addV_array[i].b;

			add_mult_sum[i].b = add_mult_sum[g_zero].b;
			add_mult_sum[i].a = add_mult_sum[g_one].b - add_mult_sum[i].b;
		}
		else
		{

			V_mult_add[i].b.value = (V_mult_add[g_zero].a.value * previous_random.value + V_mult_add[g_zero].b.value) % prime_field::mod;
			V_mult_add[i].a.value = (V_mult_add[g_one].a.value * previous_random.value + V_mult_add[g_one].b.value + prime_field::mod - V_mult_add[i].b.value) % prime_field::mod;

			addV_array[i].b.value = (addV_array[g_zero].a.value * previous_random.value + addV_array[g_zero].b.value) % prime_field::mod;
			addV_array[i].a.value = (addV_array[g_one].a.value * previous_random.value + addV_array[g_one].b.value + prime_field::mod - addV_array[i].b.value) % prime_field::mod;

			add_mult_sum[i].b.value = (add_mult_sum[g_zero].a.value * previous_random.value + add_mult_sum[g_zero].b.value) % prime_field::mod;
			add_mult_sum[i].a.value = (add_mult_sum[g_one].a.value * previous_random.value + add_mult_sum[g_one].b.value + prime_field::mod - add_mult_sum[i].b.value) % prime_field::mod;
		}

		ret.a.value = (ret.a.value + add_mult_sum[i].a.value * V_mult_add[i].a.value) % prime_field::mod;
		ret.b.value = (ret.b.value + add_mult_sum[i].a.value * V_mult_add[i].b.value
									+ add_mult_sum[i].b.value * V_mult_add[i].a.value
									+ addV_array[i].a.value) % prime_field::mod;
		ret.c.value = (ret.c.value + add_mult_sum[i].b.value * V_mult_add[i].b.value
									+ addV_array[i].b.value) % prime_field::mod;
	}

	total_uv >>= 1;

	ret.a.value = ret.a.value % prime_field::mod;
	ret.b.value = ret.b.value % prime_field::mod;
	ret.c.value = ret.c.value % prime_field::mod;
	ret.a.value = (ret.a.value + prime_field::mod) % prime_field::mod;
	ret.b.value = (ret.b.value + prime_field::mod) % prime_field::mod;
	ret.c.value = (ret.c.value + prime_field::mod) % prime_field::mod;
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	total_time += time_span.count();
	return ret;
}

std::pair<prime_field::field_element, prime_field::field_element> prover::sumcheck_finalize(prime_field::field_element previous_random)
{
	v_v = V_mult_add[0].eval(previous_random);
	return std::make_pair(v_u, v_v);
}

```

</details>

<details><summary><b> linear_gkr/verifier_fast_track.cpp </b></summary>

```c

bool verifier::verify()
{
	prime_field::init_random();
	p -> proof_init();

	auto result = p -> evaluate();

	prime_field::field_element alpha, beta;
	alpha.value = 1;
	beta.value = 0;
	random_oracle oracle;
	//initial random value
	prime_field::field_element *r_0 = generate_randomness(C.circuit[C.total_depth - 1].bit_length), *r_1 = generate_randomness(C.circuit[C.total_depth - 1].bit_length);
	prime_field::field_element *one_minus_r_0, *one_minus_r_1;
	one_minus_r_0 = new prime_field::field_element[C.circuit[C.total_depth - 1].bit_length];
	one_minus_r_1 = new prime_field::field_element[C.circuit[C.total_depth - 1].bit_length];

	for(int i = 0; i < (C.circuit[C.total_depth - 1].bit_length); ++i)
	{
		one_minus_r_0[i] = prime_field::field_element(1) - r_0[i];
		one_minus_r_1[i] = prime_field::field_element(1) - r_1[i];
	}

	std::chrono::high_resolution_clock::time_point t_a = std::chrono::high_resolution_clock::now();
	std::cerr << "Calc V_output(r)" << std::endl;
	prime_field::field_element a_0 = p -> V_res(one_minus_r_0, r_0, result, C.circuit[C.total_depth - 1].bit_length, (1 << (C.circuit[C.total_depth - 1].bit_length)));

	std::chrono::high_resolution_clock::time_point t_b = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> ts = std::chrono::duration_cast<std::chrono::duration<double>>(t_b - t_a);
	std::cerr << "	Time: " << ts.count() << std::endl;
	a_0 = alpha * a_0;

	prime_field::field_element alpha_beta_sum = a_0; //+ a_1
	prime_field::field_element direct_relay_value;
	for(int i = C.total_depth - 1; i >= 1; --i)
	{
		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
		std::cerr << "Bound u start" << std::endl;
		p -> sumcheck_init(i, C.circuit[i].bit_length, C.circuit[i - 1].bit_length, C.circuit[i - 1].bit_length, alpha, beta, r_0, r_1, one_minus_r_0, one_minus_r_1);
		p -> sumcheck_phase1_init();
		prime_field::field_element previous_random = prime_field::field_element(0);
		//next level random
		auto r_u = generate_randomness(C.circuit[i - 1].bit_length);
		auto r_v = generate_randomness(C.circuit[i - 1].bit_length);
		direct_relay_value = alpha * direct_relay(i, r_0, r_u) + beta * direct_relay(i, r_1, r_u);
		prime_field::field_element *one_minus_r_u, *one_minus_r_v;
		one_minus_r_u = new prime_field::field_element[C.circuit[i - 1].bit_length];
		one_minus_r_v = new prime_field::field_element[C.circuit[i - 1].bit_length];

		for(int j = 0; j < C.circuit[i - 1].bit_length; ++j)
		{
			one_minus_r_u[j] = prime_field::field_element(1) - r_u[j];
			one_minus_r_v[j] = prime_field::field_element(1) - r_v[j];
		}
		for(int j = 0; j < C.circuit[i - 1].bit_length; ++j)
		{
			quadratic_poly poly = p -> sumcheck_phase1_update(previous_random, j);
			previous_random = r_u[j];
			if(poly.eval(0) + poly.eval(1) != alpha_beta_sum)
			{
				fprintf(stderr, "Verification fail, phase1, circuit %d, current bit %d\n", i, j);
				return false;
			}
			else
			{
			}
			alpha_beta_sum = poly.eval(r_u[j]);
		}
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
		std::cerr << "	Time: " << time_span.count() << std::endl;

		std::cerr << "Bound v start" << std::endl;
		t0 = std::chrono::high_resolution_clock::now();
		p -> sumcheck_phase2_init(previous_random, r_u, one_minus_r_u);
		previous_random = prime_field::field_element(0);
		for(int j = 0; j < C.circuit[i - 1].bit_length; ++j)
		{
			if(i == 1)
				r_v[j] = prime_field::field_element(0);
			quadratic_poly poly = p -> sumcheck_phase2_update(previous_random, j);
			previous_random = r_v[j];
			if(poly.eval(0) + poly.eval(1) + direct_relay_value * p -> v_u != alpha_beta_sum)
			{
				fprintf(stderr, "Verification fail, phase2, circuit level %d, current bit %d\n", i, j);
				return false;
			}
			else
			{
			}
			alpha_beta_sum = poly.eval(r_v[j]) + direct_relay_value * p -> v_u;
		}
		t1 = std::chrono::high_resolution_clock::now();
		time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
		std::cerr << "	Time: " << time_span.count() << std::endl;

		auto final_claims = p -> sumcheck_finalize(previous_random);
		auto v_u = final_claims.first;
		auto v_v = final_claims.second;

		beta_init(i, alpha, beta, r_0, r_1, r_u, r_v, one_minus_r_0, one_minus_r_1, one_minus_r_u, one_minus_r_v);
		auto mult_value = mult(i);
		auto add_value = add(i);
		auto not_value = not_gate(i);
		auto minus_value = minus_gate(i);
		auto xor_value = xor_gate(i);
		auto naab_value = NAAB_gate(i);
		auto sum_value = sum_gate(i);
		auto relay_value = relay_gate(i);

		if(alpha_beta_sum != add_value * (v_u + v_v) + mult_value * v_u * v_v + direct_relay_value * v_u + not_value * (prime_field::field_element(1) - v_u) + minus_value * (v_u - v_v) + xor_value * (v_u + v_v - prime_field::field_element(2) * v_u * v_v) + naab_value * (v_v - v_u * v_v) + sum_value * v_u + relay_value * v_u)
		{
			fprintf(stderr, "Verification fail, semi final, circuit level %d\n", i);
			return false;
		}
		else
		{
			fprintf(stderr, "Verification Pass, semi final, circuit level %d\n", i);
		}
		auto tmp_alpha = generate_randomness(1), tmp_beta = generate_randomness(1);
		alpha = tmp_alpha[0];
		beta = tmp_beta[0];
		delete[] tmp_alpha;
		delete[] tmp_beta;
		if(i != 1)
			alpha_beta_sum = alpha * v_u + beta * v_v;
		else
		{
			alpha_beta_sum = v_u;
		}

		delete[] r_0;
		delete[] r_1;
		delete[] one_minus_r_0;
		delete[] one_minus_r_1;
		r_0 = r_u;
		r_1 = r_v;
		one_minus_r_0 = one_minus_r_u;
		one_minus_r_1 = one_minus_r_v;
		std::cerr << "Prove Time " << p -> total_time << std::endl;
	}

	//post sumcheck
	prime_field::field_element* input;
	input = new prime_field::field_element[(1 << C.circuit[0].bit_length)];

	for(int i = 0; i < (1 << C.circuit[0].bit_length); ++i)
	{
		int g = i;
		if(C.circuit[0].gates[g].ty == 3)
		{
			input[g] = prime_field::field_element(C.circuit[0].gates[g].u);
		}
		else if(C.circuit[0].gates[g].ty == 2)
		{
			input[g] = prime_field::field_element(0);
		}
		else
			assert(false);
	}
	auto input_0 = V_in(r_0, one_minus_r_0, input, C.circuit[0].bit_length, (1 << C.circuit[0].bit_length));
		// input_1 = V_in(r_1, one_minus_r_1, input, C.circuit[0].bit_length, (1 << C.circuit[0].bit_length));

	delete[] input;
	delete[] r_0;
	delete[] r_1;
	delete[] one_minus_r_0;
	delete[] one_minus_r_1;
	if(alpha_beta_sum != input_0)
	{
		fprintf(stderr, "Verification fail, final input check fail.\n");
		return false;
	}
	else
	{
		fprintf(stderr, "Verification pass\n");
		std::cerr << "Prove Time " << p -> total_time << std::endl;
	}
	p -> delete_self();
	delete_self();
	return true;
}

```

</details>

<details><summary><b> VPD/vpdR.cpp </b></summary>

```c


std::pair<mpz_class, mpz_class> commit(Ec1& digest, Ec1& digesta, Ec1& digest2, Ec1& digest2a, vector<mpz_class>& input, vector<mpz_class>& input2){

	mpz_class r_f;
	digest = g1 * 0;
	mpz_urandomm(r_f.get_mpz_t(), r_state, p.get_mpz_t());
	pre_input(input);
	vector<mpz_class>& coeffs = input;
	assert(coeffs.size() >= 1);
	coeffs[(int)coeffs.size() - 1] = r_f;

	clock_t commit_t = clock();

	//compute digest pub

	for(int i = 0; i < coeffs.size(); i++)
		if(coeffs[i] < 0)
			coeffs[i] += p;

	vector<mpz_class> scalar_pow;
	scalar_pow.resize(multi_scalar_w);

	for(int i = 0; i < (1 << NumOfVar) / multi_scalar_w + 1; i++){
		for(int j = 0; j < multi_scalar_w; ++j)
		{
			int id = i * multi_scalar_w + j;
			if(id >= (1 << NumOfVar))
			{
				scalar_pow[j] = 0;
			}
			else
			{
				scalar_pow[j] = coeffs[id];
			}
		}

		digest = digest + multi_scalar_calc(i, (1 << NumOfVar), scalar_pow);
	}
	mie::Vuint temp(coeffs[1 << NumOfVar].get_str().c_str());
	digest += pub_g1[1 << NumOfVar] * temp;

	mpz_class r_f2;
	digest2 = g1 * 0;
	mpz_urandomm(r_f2.get_mpz_t(), r_state, p.get_mpz_t());

	vector<mpz_class> coeffs2 = input2;
	coeffs2.push_back(r_f2);
	for(int i = 0; i < coeffs.size(); i++)
		if(coeffs[i] < 0)
			coeffs[i] += p;
	mie::Vuint temp0(coeffs2[0].get_str().c_str());
	mie::Vuint temp1(coeffs2[1].get_str().c_str());
	mie::Vuint temprf2(coeffs2[2].get_str().c_str());
	digest2 = pub_g1[0] * temp0 + pub_g1[1 << (NumOfVar - 1)] * temp1 + pub_g1[1 << NumOfVar] * temprf2;

	const mie::Vuint tempa(a.get_str().c_str());

	digesta = digest * tempa;
	digest2a = digest2 * tempa;

	cout << "Input VPD commit time: " << (double)(clock() - commit_t) / CLOCKS_PER_SEC << endl;

	return make_pair(r_f, r_f2);
}

void prove(vector<mpz_class> r, mpz_class& ans, vector<mpz_class>& input, vector<mpz_class> &input2, vector<Ec1>& witness, vector<Ec1>& witnessa, mpz_class r_f, mpz_class r_f2, mpz_class Z){
	vector<mpz_class> coeffs = input;
	coeffs[0] = (coeffs[0] + (Z * input2[0]) % p) % p;
	coeffs[1 << (NumOfVar - 1)] = (coeffs[1 << (NumOfVar - 1)] + (Z * input2[1]) % p) % p;

	clock_t prove_t = clock();

	std::vector<mpz_class> t(NumOfVar);
	for(int i = 0; i < t.size(); i++)
		mpz_urandomm(t[i].get_mpz_t(), r_state, p.get_mpz_t());

	assert(coeffs.size() >= 1);
	coeffs[coeffs.size() - 1] = ((r_f + Z * r_f2) % p);


	for(int i = 0; i < coeffs.size(); i++)
		if(coeffs[i] < 0)
			coeffs[i] += p;

	std::vector<mpz_class> ans_pre;
	ans_pre.resize((int)pow(2, NumOfVar));
	ans_pre[0] = 1;
	ans = coeffs[0];
	for(int i = 0; i < NumOfVar; i++){
		for(int j = (int)pow(2, i); j < (int)pow(2, i+1); j++){
			ans_pre[j] = (r[i] * ans_pre[j - (int)pow(2, i)]) % p;
			ans = (ans + (ans_pre[j] * coeffs[j]) % p) % p;
		}
	}

	witness.resize(NumOfVar + 1);
	for(int i = 0; i < NumOfVar; i++)
		witness[i] = g1 * 0;
	witnessa.resize(NumOfVar + 1);
	vector<mpz_class> witness_coeffs((int)pow(2, NumOfVar)), temp_coeffs = coeffs;

	int start_index = 0;
	for(int i = 0; i < NumOfVar; i++){
		for(int j = 0; j < pow(2, r.size() - i - 1); j++){
			witness_coeffs[start_index + j] = temp_coeffs[pow(2, r.size() - i - 1) + j] % p;
			temp_coeffs[j] = (temp_coeffs[j] + (temp_coeffs[pow(2, r.size() - i - 1) + j] * r[NumOfVar - 1 - i]) % p) % p;

		}
		start_index += pow(2, r.size() - i - 1);
	}
	for(int i = 0; i < witness_coeffs.size(); i++)
		if(witness_coeffs[i] < 0)
			witness_coeffs[i] += p;

	const mie::Vuint tempa(a.get_str().c_str());

	std::vector<mpz_class> scalar_pow;
	scalar_pow.resize(multi_scalar_w);
	for(int i = 0; i < (1 << (NumOfVar - 1)) / multi_scalar_w + 1; ++i)
	{
		int temp = 0;
		for(int k = NumOfVar - 1; k >= 0; --k)
		{
			if(i >= ((1 << k) / multi_scalar_w + 1))
			{
				break;
			}
			for(int j = 0; j < multi_scalar_w; ++j)
			{
				int id = i * multi_scalar_w + j;
				if(id >= (1 << k))
				{
					scalar_pow[j] = 0;
				}
				else
				{
					scalar_pow[j] = witness_coeffs[id + temp];
				}
			}
			temp += (1 << k);
			witness[k] += multi_scalar_calc(i, (1 << k), scalar_pow);
		}
	}

	for(int k = NumOfVar - 1; k >= 0; --k)
	{
		mie::Vuint temptk(t[k].get_str().c_str());
		witness[k] += pub_g1[1 << NumOfVar] * temptk;
		witnessa[k] = witness[k] * tempa;
	}


	mpz_class tmp = coeffs[1 << NumOfVar];
	for(int i = 0; i < NumOfVar; i++)
		tmp += (t[i] * r[i]) % p;
	witness[NumOfVar] = pre_exp(g1_pre, tmp);
	clock_t zkt = clock();
	for(int i = 0; i < NumOfVar; i++){
		mie::Vuint tempti(t[i].get_str().c_str());
		witness[NumOfVar] -= pub_g1[1 << i] * tempti;
	}
	witnessa[NumOfVar] = witness[NumOfVar] * tempa;

	cout << "Input VPD prove time: " << (double)(clock() - prove_t) / CLOCKS_PER_SEC << endl;
}

bool verify(vector<mpz_class> r, Ec1 digest, Ec1 digest2, mpz_class Z, mpz_class& ans, vector<Ec1>& witness, vector<Ec1>& witnessa){
	clock_t verify_t = clock();

	Fp12 ea1, ea2;

	bool flag = 1;

	for(int i = 0; i < r.size(); i++){
		opt_atePairing(ea1, g2, witnessa[i]);
		opt_atePairing(ea2, g2a, witness[i]);
		if(ea1 != ea2){
			cout << "here error!" << endl;
			flag = 0;
		}
	}

	Fp12 ea3, ea4 = 1;

	std::vector<Fp12> temp(r.size() + 1);

	mie::Vuint tempz(Z.get_str().c_str());
	Ec1 temp2 = pre_exp(g1_pre, ans);

	opt_atePairing(ea3, g2, digest + digest2 * tempz - temp2);

	for(int i = 0; i < r.size() + 1; i++){
		if(i == r.size()){
			opt_atePairing(temp[i], pub_g2[i], witness[i]);
		}
		if(i < r.size()){
			Ec2 temp3 = pub_g2[i] - pre_exp(g2_pre, r[i]);

			opt_atePairing(temp[i], temp3, witness[i]);
		}
		ea4 *= temp[i];
	}





	if(ea3 != ea4) {
		cout << "final error" << endl;
		flag = 0;
	}

	cout << "Input VPD verify time: "<<(double)(clock() - verify_t) / CLOCKS_PER_SEC << endl;
	return flag;

}

void KeyGen(int d){
	NumOfVar = d;
	clock_t KeyGen_t = clock();
	mpz_urandomm(a.get_mpz_t(), r_state, p.get_mpz_t());

	precompute_g1();
	KeyGen_preprocessing(g1);
	g1a = pre_exp(g1_pre, a);
	g2a = pre_exp(g2_pre, a);

	s.resize(NumOfVar + 1);
	for(int i = 0; i < NumOfVar + 1; i++)
		mpz_urandomm(s[i].get_mpz_t(), r_state, p.get_mpz_t());
	pub_g1_exp.resize((int)pow(2, d));
	pub_g1.resize((int)pow(2, d) + 1);
	pub_g2.resize(d + 1);
	pub_g1_exp[0] = 1;
	pub_g1[0] = g1;
	pub_g2[0] = g2;
	for(int i = 0; i < d; i++){
		for(int j = 1 << i; j < (1 << (i + 1)); j++){
			pub_g1_exp[j] = (s[i] * pub_g1_exp[j - (1 << i)]) % p;
			pub_g1[j] = g1_exp(pub_g1_exp[j]);
		}
	}
	pub_g1[1 << d] = pre_exp(g1_pre, s[d]);
	//multi_scalar
	//assert(multi_scalar_w == 2); //to avoid some error
	vector<Ec1> scalars;
	scalars.resize(multi_scalar_w);
	multi_scalar_g1.resize((1 << d) / multi_scalar_w + 1);
	for(int i = 0; i < (1 << d) / multi_scalar_w + 1; ++i)
	{
		for(int j = 0; j < multi_scalar_w; ++j)
		{
			int id = i * multi_scalar_w + j;
			if(id >= (1 << d))
			{
				scalars[j] = g1 * 0;
			}
			else
			{
				scalars[j] = pub_g1[id];
			}
		}
		for(int j = 3; j < (1 << multi_scalar_w); ++j)
		{
			multi_scalar_g1[i].value[j - 3] = g1 * 0;
			for(int k = 0; k < multi_scalar_w; ++k)
			{
				if((j >> k) & 1)
					multi_scalar_g1[i].value[j - 3] += scalars[k];
			}
		}
	}
	for(int i = 0; i < d + 1; ++i)
	{
		pub_g2[i] = pre_exp(g2_pre, s[i]);
	}
	cout << "Input VPD KeyGen time: " << (double)(clock() - KeyGen_t) / CLOCKS_PER_SEC << endl;

	return;
}

```

</details>

</details>

[Libra](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/Libra-Libra)
