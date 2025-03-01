# ETAAcademy-ZKMeme: 55. Collaborative ZKPs

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>55. Collaborative ZKPs</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Collaborative_ZKPs</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Collaborative Zero-Knowledge Proofs (co-ZKPs)

Collaborative Zero-Knowledge Proofs (co-ZKPs) are an innovation over traditional Zero-Knowledge Proofs (ZKPs), aiming to address the excessive computational burden on a single prover inherent in traditional ZKPs. zk-SNARKs are a key technology in this evolution, and unlike conventional zk-SNARKs, co-ZKPs employ protocols such as GKR, Sumcheck, Zerocheck, Productcheck, mKZG, and Multi-Party Computation (MPC) techniques, which ensure that each party involved in verifying the authenticity of data does not leak private information. The applications of co-ZKPs are vast, including zkBridge cross-chain bridges, verifiable machine learning, public auditable multi-party computation (PA-MPC), and CoSNARKs.

---

The Collaborative zk-SNARK (collaborative zk-SNARK) is designed to address the problem of a single prover's computational burden in traditional zk-SNARKs when handling large-scale circuits. In collaborative zk-SNARKs, the proof generation process is distributed across multiple servers, with each server handling a portion of the computation. This distributed approach reduces the burden on individual servers and enhances scalability.

However, most existing research assumes that the servers involved are "benign," meaning they can directly access sensitive witness information. In real-world applications, this is not always viable, especially when the witness contains sensitive data. Secret-sharing techniques are used to ensure the privacy of the witness and prevent the leakage of sensitive information between servers.

One extension of this method is zk-SNARK-as-a-Service (zkSaaS), which primarily optimizes the efficiency of proof generation. However, this method still relies on a "master server" to handle the resource-intensive parts of computation and communication, which creates a scalability bottleneck. An improved version, based on Libra zk-SNARKs, has been proposed, offering a fully distributed proof generation process. In this scheme, all servers share nearly equal computational loads, enhancing scalability. Furthermore, by introducing certain verification mechanisms, this protocol can resist attacks from malicious actors.

The security model of this approach focuses on both the semi-honest and malicious adversarial models, with particular attention given to how the system can maintain its security in the presence of malicious adversaries.

## GKR, Sumcheck, Zerocheck, and Productcheck Protocols

The **GKR Protocol** (Goldwasser, Kalai, Rothblum) is used to prove the correctness of the outputs of hierarchical arithmetic circuits. It is based on the Sumcheck Protocol, which is fundamental for zk-SNARKs. **Sumcheck** is used to verify the sum of the values of a polynomial, forming the foundation of zk-SNARKs. **Zerocheck** is a special case of Sumcheck, used to check if a polynomial evaluates to zero over a hypercube. **Productcheck** is used to verify the product of polynomial values, applicable in scenarios like permutation checks.

#### GKR Protocol

Let $C$ be an arithmetic circuit of depth $d$ over a finite field $F$. Assume that each layer $i$ has $S$ gates, with $S = 2^m$, where each gate’s input comes from two gates from the previous layer, and outputs a value. Layer 0 is the output layer, and layer $d$ is the input layer. For each layer, a polynomial extension function $V_i$ is defined, which returns the output value of a gate based on its label. These polynomial extensions $V_i$ are then computed layer by layer recursively and verified using the Sumcheck Protocol to ensure the correctness of each layer’s calculations.

In Virgo, the GKR protocol is used to design a distributed proof generation algorithm, allowing proofs to be generated for multiple identical circuits simultaneously, thus improving efficiency. However, a security concern arises as all servers involved have access to the witness, which may compromise security. One of the core concepts in this protocol is **multilinear extension**, which extends a function defined over binary vectors to a broader field. Specifically, **multilinear extension combines each input value with linear interpolation functions to form a polynomial**, which enables the verification of the circuit's output and ensures computational correctness in higher dimensions. This method makes the proof process more efficient and is suitable for zk-SNARKs and other zero-knowledge proof systems.

Given a vector $V$, its multilinear extension $\tilde{V}$ is a function from $F^{\ell}$ to $F$, where $F$ is a finite field, and $\tilde{V}$ satisfies $\tilde{V}(x) = V(x)$ for all $x \in \{0,1\}^\ell$. The mathematical expression for the multilinear extension is:

$\tilde{V}(x) = \sum_{b \in \{0,1\}^\ell} \left( \prod_{i=1}^{\ell} \beta_{b_i}(x_i) \right) \cdot V(b)$

where $\beta_{b_i}(x_i) = (1 - x_i)(1 - b_i) + x_i b_i$ is the binary linear interpolation function. This extension serves to transform values defined over binary vectors into a polynomial, enabling computation and verification in higher dimensions.

#### Sumcheck Protocol

The Sumcheck Protocol is an interactive protocol between the prover and verifier to verify whether the sum of a polynomial $f$ over the $\ell$-dimensional hypercube is correct. Specifically, it verifies the correctness of the sum $H = \sum_{b_1, b_2, \dots, b_\ell} f(b_1, b_2, \dots, b_\ell)$ for all $b_i \in \{0,1\}$. This is an interactive protocol that requires $\ell$ rounds of communication. The communication and verification cost is $O(d \cdot \ell)$, where $d$ is the degree of the polynomial $f$. To construct a zero-knowledge Sumcheck Protocol, masking polynomials are added to the process.

The Sumcheck protocol is the core construction for multivariate interactive proofs, allowing the prover to convince the verifier that a multivariate polynomial $f(x)$ satisfies the equation $H = \sum_{x \in \{0,1\}^\ell} f(x)$. The execution of the protocol is divided into $\ell$ rounds, with the following steps:

1. In the $i$-th round, the prover sends a univariate polynomial $f_i(x) = \sum_{b_{i+1}, ..., b_\ell \in \{0,1\}} f(r_1, ..., r_{i-1}, x, b_{i+1}, ..., b_\ell)$.
2. The verifier checks if the equation holds using the previous round’s random challenge $r_{i-1}$:

$f_{i-1}(r_{i-1}) = f_i(0) + f_i(1)$

3. Finally, the verifier queries the polynomial $f$ to validate the correctness of the entire process.

In contrast to traditional univariate Sumcheck, the multivariate Sumcheck Protocol does not rely on Fast Fourier Transforms (FFT) for polynomial division but instead uses multivariate polynomials and interactive proofs over multiple rounds. This makes it more efficient, and currently, there is no research focused on the secret-sharing variant of the multivariate Sumcheck Protocol.

#### Zerocheck Protocol

The Zerocheck Protocol is used to verify if a multivariate polynomial evaluates to zero over a hypercube. Essentially, it is a special case of Sumcheck, where an equality constraint is multiplied by the polynomial to check if the result is zero. The Zerocheck Protocol allows the prover to convince the verifier that $\sum_{x \in \{0,1\}^\ell} f(x) = 0$. The prover constructs an auxiliary polynomial:

$\tilde{eq}(r, x) = \prod_{i=1}^{\ell} (1 - x_i)(1 - r_i) + x_i r_i$

where $r$ is the verifier’s random challenge. The prover and verifier then run the **Sumcheck Protocol** to verify the equation:

$0 = \sum_{x \in \{0,1\}^\ell} f(x) \cdot \tilde{eq}(r, x)$

#### Productcheck Protocol

The Productcheck Protocol is used to verify if the product of a multivariate polynomial over the hypercube equals a given value. It can be extended to other zk-SNARK tools, such as **Permutation-check** and **Multiset-check**. The primary purpose of Productcheck is to prove that the product of a multivariate polynomial $f(x)$ over $\{0,1\}^\ell$ equals a given value $H$, i.e.,

$H = \prod_{x \in \{0,1\}^\ell} f(x)$

The core task is to compute an auxiliary polynomial $v(x)$, which is related to the values of $f(x)$ over $\{0,1\}^\ell$. This protocol plays a critical role in verifying the integrity of polynomial products and has wide applications in zk-SNARK systems.

---

## Completely Distributed Sumcheck Protocol

In traditional sumcheck protocols, each round of computation requires elements from specific positions for operations. Directly packing and sharing all the elements at once (i.e., bundling everything together) is not ideal because each round of computation needs to operate on corresponding elements from specific positions, and bundled data is not conducive to such operations. A larger packing factor (i.e., bundling too many elements) can reduce fault tolerance, meaning it can lower the system’s ability to handle malicious servers. Therefore, directly packing all elements for sharing is not an optimal solution.

The **Completely Distributed Sumcheck Protocol**, utilizing Multi-Party Computation (MPC) and Packed Secret Sharing (PSS) technology, accelerates and distributes the computation tasks. The goal of this method is to allow multiple servers to collaborate in computing the **bookkeeping table** rather than a single server handling all the computations.

By removing the dependence on powerful servers during the proof generation process, this approach achieves even distribution of the workload across multiple servers, ensuring that the proof generation can scale efficiently to larger circuits. To achieve this, a SIMD-friendly zk-SNARKs scheme is used. Unlike some zk-SNARKs based on Fast Fourier Transform (FFT) (like Groth16 and Plonk), the focus here is on avoiding FFT and utilizing **GKR-based zk-SNARKs**. We note that the Sumcheck protocol forms the basis of GKR-based zk-SNARKs, which possesses a SIMD structure that can distribute computation tasks evenly across multiple servers. To implement this, the PSS technique is used to distribute the workload across servers, and through linear combinations in different rounds, a **bookkeeping table** is generated, significantly improving computational efficiency.

To design a completely distributed Sumcheck protocol, the core step is to have multiple servers collaboratively compute the **bookkeeping table**. This table records various evaluation results of a polynomial. Each server only receives polynomial evaluation values represented by ciphertexts (packed secret shares). Specifically, the evaluation values are the results of evaluating a polynomial $f(x)$ over the hypercube $\{0,1\}^\ell$.

1. **Constructing the Bookkeeping Table**: Let $f(x)$ be an $\ell$-variable multilinear polynomial. The first row of the bookkeeping table stores the evaluations of $f$ at all $2^\ell$ Boolean input points. In each round of computation, the number of elements in the table is halved, and eventually, only two values remain: $f(0)$ and $f(1)$.
2. **Core Formula (Recursive Computation)**: The key idea of the Sumcheck protocol is that the prover and verifier interact in multiple rounds, gradually reducing the dimensionality of the polynomial until the correctness of the sum is confirmed. In each round, the prover computes a new polynomial based on the results of the previous round and sends it to the verifier for validation.
   - In the $i$-th round of the Sumcheck interaction, the prover needs to compute:  
     $f(r_1, \dots, r_{i-1}, r_i, b^{(i)}) = (1 - r_i) f(r_1, \dots, r_{i-1}, 0, b^{(i)}) + r_i f(r_1, \dots, r_{i-1}, 1, b^{(i)})$
   - Where $r_i$ is the random challenge provided by the verifier, and $b^{(i)}$ represents the remaining Boolean values for the variables.
   - Since the evaluations $f(r_1, \dots, r_{i-1}, 0, b^{(i)})$ and $f(r_1, \dots, r_{i-1}, 1, b^{(i)})$ have already been computed in the previous round, the current round only needs to perform simple linear operations on these values.

#### Packed Secret Sharing (PSS)

PSS is an extension of Shamir’s Secret Sharing (SSS) that allows sharing multiple secrets at once, rather than just one. Suppose we have a vector of $k$ secrets: $x = \{x_1, x_2, ..., x_k\}$, where $k$ is the **packing factor**, representing how many secrets can be shared in a single bundle. PSS constructs a polynomial of degree at least $k-1$, $f(-i+1) = x_i$ for $i \in [k]$, such that $f(-i+1)$ equals the values of the $k$ secrets. Then, the polynomial's evaluations at these $k$ points are computed and distributed to $N$ participants. Any $d+1$ participants (where $d \geq k-1$) can use Lagrange interpolation to reconstruct all the secrets. PSS is linearly homomorphic, meaning the shared values can be directly added without redistribution; it also supports multiplication, although multiplication increases the polynomial degree and requires $2d+1 \leq N$ for safe execution.

Each round of computation involves the following steps:

- The current data is split into multiple vectors (each representing a portion of the computation), and each vector is represented by secret shares in ciphertext.
- Each server computes locally and sends the results to the verifier, who checks the correctness of the computation.

In each round, each server only needs to compute locally:
$J x_j^{(i+1)} K = (1 - r_i) \cdot J x_j^{(i)} K + r_i \cdot J x_{j + \frac{n_i}{2k}}^{(i)} K$
Where $J x_j^{(i)} K$ represents the secret-shared evaluation value, and the computation process does not require decryption, thus preserving the privacy of $f(x)$.

By using weighted linear combinations, multiple data can be processed in parallel, leveraging SIMD to speed up computations. The factors $(1 - r_i)$ and $r_i$ enable this computation to be weighted at each round. This approach allows multiple servers to simultaneously calculate these values, with each server only computing the linear combination of values $x_j$ and $x_{j + 2^{n/k}}$, rather than the entire polynomial. Additionally, ciphertexts are updated based on random challenges, ensuring that the computation remains parallelized and the server's load remains light.

At the final few rounds, each server only holds a single element, and special handling is needed. To ensure effective linear combinations of the remaining elements, a technique called **PSS Permutation** is used. This function rearranges the data via a permutation $p_i(j)$, ensuring the data can continue to be processed in the next round of secret-shared computation.

By combining all of these steps, a fully distributed Sumcheck protocol is achieved. In this protocol:

- Each server performs local computations and communication, without relying on a centralized server to control all computations.
- All operations and data sharing are accomplished through PSS and random challenges, ensuring the security and privacy of the protocol.

#### Extension to Polynomial Products

This completely distributed Sumcheck protocol can be extended to the product of multiple polynomials. For the product of two polynomials $f$ and $g$, we first compute the bookkeeping table for each polynomial separately using the Sumcheck protocol. Then, in each round, the servers multiply corresponding entries from the two tables and perform summation operations:
$\sum_{b_1, b_2, \dots, b_\ell \in \{0,1\}} f(r_1, \dots, r_{i-1}, x_i, b_{i+1}, \dots, b_\ell) \cdot g(r_1, \dots, r_{i-1}, x_i, b_{i+1}, \dots, b_\ell)$
To perform these multiplication operations, we use the **PSS multiplication** protocol, which allows for encrypted sharing of the product.

---

## mKZG: A Distributed Polynomial Commitment Scheme for Efficient zk-SNARKs

The GKR protocol combines the sumcheck protocol with polynomial expansion to allow the prover to efficiently prove the correctness of hierarchical arithmetic circuit outputs. It recursively verifies the computational results on each layer of the circuit, using randomly selected vectors and linear combinations for validation. However, the GKR protocol by itself does not possess zero-knowledge properties, making it unsuitable as a proof system (argument system). **To extend the GKR protocol into a zero-knowledge proof system (zk-SNARK), polynomial commitment (PC) schemes are introduced. The prover first commits to the polynomial expansion of the input layer, and the verifier checks the commitment by verifying the opening messages to ensure the proof’s validity.** Ultimately, the GKR protocol is extended into an efficient zero-knowledge proof system, such as the Libra zk-SNARK, which guarantees circuit correctness while maintaining zero-knowledge properties.

#### Commitment Generation

In the original GKR protocol, the prover \(P\) needs to submit two values, $\tilde{V}_d(u^{(d)})$ and $\tilde{V}_d(v^{(d)})$, to the verifier \(V\). Here, $u^{(d)}$ and $v^{(d)}$ are two random vectors chosen by the verifier, and $\tilde{V}_d$ is a **multilinear extension** of the polynomial representing the proof of the circuit's input layer. These values are validated using a **polynomial commitment** (PC) scheme. To achieve distributed polynomial commitment generation, the mKZG commitment scheme is employed, in conjunction with the distributed multi-scalar multiplication (dMSM) protocol, to avoid the challenges posed by FFTs.

Instead of relying on FFT, a novel approach utilizes SIMD (Single Instruction, Multiple Data) structures and the dMSM protocol for distributed computation. This ensures an efficient and scalable process for polynomial commitment generation. For a polynomial $f$, the commitment is given by:

$com_f = g^{f(s)} = \prod_{b \in \{0,1\}^\ell} g^{\prod_{i=1}^\ell \beta_{b_i}(s_i) \cdot V(b)}$

where:

- $\beta_{b_i}(s_i) = (1 - s_i)(1 - b_i) + s_i b_i$ is the Lagrange basis function.
- $V(b)$ is the value of the vector $x$ at the point $b$.
- $g^{\prod_{i=1}^{\ell} \beta_{b_i}(s_i)}$ is a group element.

This commitment can be computed using multi-scalar multiplication (MSM). However, in a collaborative environment, each server holds only a partial share of $x$, not the complete vector. To address this, **distributed MSM (dMSM)** technology allows multiple servers to work together and compute the MSM collaboratively. The approach is to precompute and store all possible values of $g^{\prod_{i=1}^{\ell} \beta_{b_i}(s_i)}$, and share these values across servers via secret sharing. The servers use the distributed MSM protocol (ΠdMSM) to compute the polynomial commitment collaboratively.

#### Opening Proof Generation

In a multivariate polynomial commitment scheme, to prove that $f(u) = z$, the prover needs to compute a series of quotient polynomials and remainder polynomials. The **polynomial division** process is as follows: through $\ell$ rounds of polynomial division, the quotient polynomials $Q_i$ and remainder polynomials $R_i$ are computed ($i \in [\ell]$):

$R_{i-1}(x_i, x_{i+1}, ..., x_\ell) = Q_i(x_{i+1}, ..., x_\ell) \cdot (x_i - u_i) + R_i(x_{i+1}, ..., x_\ell)$

where $R_0 = f$ is the original polynomial.

The servers then use these quotient and remainder polynomial evaluations to generate the proof. The evaluation values are calculated efficiently using the properties of polynomial algebra, particularly the SIMD structure. For each binary value $b \in \{0,1\}^{\ell-i}$, the evaluation of the quotient and remainder polynomials is computed recursively as follows:

$Q_i(b) = R_{i-1}(1, b) - R_{i-1}(0, b)$
$R_i(b) = (1 - u_i) R_{i-1}(0, b) + u_i R_{i-1}(1, b)$

These calculations are performed locally on each server, minimizing communication overhead. Each round of the quotient polynomial $Q_i$ and remainder polynomial $R_i$ is computed by linearly combining and subtracting previous polynomial evaluations, generating the final evaluation values efficiently.

After all quotient polynomial evaluations $Q_i$ are obtained, the servers use a distributed multi-scalar multiplication protocol (dMSM-Semi) to collaboratively compute and generate the final proof. This process is carried out via distributed computation and linear combinations, avoiding reliance on a powerful single server and reducing computational and communication costs, making the proof generation process more efficient and distributed.

In the dMVPC.Open process, $O(\ell) = O(\log n)$ rounds of distributed MSM (FdMSM) need to be executed, which involves **multiple communication rounds**. However, through **batching optimization (Batching Distributed MSMs)**, all MSM computations are merged into a single computation, reducing the number of communication rounds. Each server computes $O(\ell)$ values locally and sends them all at once. Server $S_1$ then processes all MSM calculations in parallel, reducing the round complexity of the protocol from $O(\log n)$ to $O(1)$. This **scalable collaborative multi-linear polynomial commitment scheme**, powered by **distributed MSM** and **batching optimization**, achieves high efficiency and security in zk-SNARK computations.

Finally, to ensure the protocol remains secure against malicious behavior, the protocol’s semi-honest secure PSS replacement, dMSM, and PSS multiplication protocols are upgraded to maliciously secure versions, with mechanisms for detecting malicious actions. These changes ensure the protocol can withstand up to $t$ compromised servers while maintaining efficiency. The maliciously secure protocol achieves a computational and space complexity of $O\left(\frac{|C|}{N}\right)$ and guarantees the correctness and resilience of the protocol against attacks.

---

## Applications of Collaborative zk-SNARK

Collaborative zk-SNARKs offer significant potential across several fields. First, in the cross-chain bridge domain, distributed proof generation is proposed to enhance the security and efficiency of zkBridge systems. Compared to traditional centralized zkBridge approaches, this method not only improves transaction privacy but also maintains high efficiency. It is well-suited for decentralized, privacy-first applications like zkRollup and zkEVM. Second, collaborative zk-SNARKs address scalability issues in verifiable machine learning by handling larger models and overcoming the scalability limitations of traditional zk-SNARK provers. Lastly, in Publicly Auditable Multi-Party Computation (PA-MPC), the use of collaborative zk-SNARKs boosts the efficiency and security of result verification, showcasing their wide potential in decentralized and privacy-preserving applications.

#### coSNARK

zkSNARKs are efficient zero-knowledge proofs (ZKP) that ensure data privacy. Traditionally, however, they are generated by a single prover, which limits their use in applications involving multiple participants.

Collaborative zk-SNARK (coSNARK) solves the limitation of traditional zk-SNARKs by allowing multiple parties to generate a zk-SNARK proof collectively. By combining Multi-Party Computation (MPC) with zk-SNARKs, multiple participants can collaboratively compute a proof without revealing their individual secrets. This method allows for the outsourcing of ZKP computation by distributing the calculation tasks across several servers, thereby alleviating the computational load while safeguarding secrecy. For example, the secret can be divided into parts and sent to different servers, which then collaborate to compute the proof. Combining MPC also ensures that each participant’s computational results are verifiable, increasing transparency.

SPDZ is an MPC protocol designed for dishonest majority settings, while GSZ is for honest majority settings, both utilizing linear secret sharing schemes. SPDZ uses additive secret sharing, while GSZ uses Shamir's secret sharing. Previous research has shown that SPDZ operates on elliptic curve groups, which significantly reduce communication costs when compared to field-based computation. By using SPDZ and GSZ linear secret sharing protocols, MPC computations are efficiently supported. Furthermore, optimization techniques, such as parallelized product checks and reducing the overhead of nonlinear operations, ensure computational efficiency. Experimental results show that when using an honest majority protocol (GSZ), collaborative proofs almost achieve the same execution time as single-party proofs, while dishonest majority protocols (SPDZ) only introduce a 2x delay.

To mitigate efficiency issues caused by multi-party computation, several optimization strategies can be adopted. For linear operations, such as FFT and MSM, these can be distributed without adding extra communication overhead. For complex nonlinear operations, like polynomial division, the use of public key values can help avoid additional computational cost. For product checks, specific cryptographic techniques can be employed to reduce the number of communication rounds.

#### coNoir

The coNoir project aims to implement a collaborative machine learning model training process within Noir, integrating Multi-Party Computation (MPC) and Zero-Knowledge Proofs (ZK). This approach ensures that machine learning models can be trained collaboratively without compromising data privacy, while also generating a ZK proof to verify the training process.

The project focuses on implementing fixed-point arithmetic and matrix operations, employing Newton-Raphson for maximum likelihood estimation to fit logistic regression models. Additionally, coNoir enables secure collaborative training, supporting both ZK and MPC modes. The testing phase will utilize the Iris plant dataset to evaluate the model's performance, with the accuracy of the training results validated using traditional decimal representations. The project's code and documentation will be open-sourced, further advancing the development of privacy-preserving machine learning in the future.

<details><summary><b> Code </b></summary>

<details><summary><b> sumcheck.rs </b></summary>

```rust

struct Delegator {
    // the 2^N evaluations of the polynomial
    x: Vec<Fr>,
}

impl Delegator {
    fn new(size: usize) -> Self {
        let rng = &mut ark_std::test_rng();
        let x: Vec<Fr> = (0..2usize.pow(size as u32))
            .into_iter()
            .map(|_| Fr::rand(rng))
            .collect();
        Self { x }
    }
    fn delegate(&self, l: usize) -> Vec<Vec<Fr>> {
        let pp = PackedSharingParams::<Fr>::new(l);
        transpose(
            self.x
                .par_chunks_exact(l)
                .map(|chunk| pp.pack_from_public(chunk.to_vec()))
                .collect(),
        )
    }
}

```

</details>

<details><summary><b> dgkr.rs </b></summary>

```rust

#[derive(Clone, Debug)]
pub struct PackedProvingParameters<E: Pairing> {
    // Packed shares of f1 and V
    pub f1s: Vec<SparseMultilinearExtension<E::ScalarField>>,
    pub poly_vs: Vec<PackedDenseMultilinearExtension<E::ScalarField>>,
    // Challenges
    pub challenge_g: Vec<E::ScalarField>,
    pub challenge_u: Vec<E::ScalarField>,
    pub challenge_v: Vec<E::ScalarField>,
    pub challenge_r: Vec<E::ScalarField>,
    pub commitment: PolynomialCommitment<E>
}

...

/// This is a proof-of-concept implementation of the distributed GKR function.
/// The following implementation is valid only for data-parallel circuits.
pub async fn d_gkr_function<F: FftField, Net: MPCSerializeNet>(
    shares_f1: &SparseMultilinearExtension<F>,
    shares_f2: &PackedDenseMultilinearExtension<F>,
    shares_f3: &PackedDenseMultilinearExtension<F>,
    challenge_g: &Vec<F>,
    challenge_u: &Vec<F>,
    challenge_v: &Vec<F>,
    pp: &PackedSharingParams<F>,
    net: &Net,
    sid: MultiplexedStreamID,
) -> Result<Vec<(F, F, F)>, MPCNetError> {
    // Init Phase 1
    let hg = d_initialize_phase_one(shares_f1, shares_f3, challenge_g, pp, net, sid).await?;
    // Sumcheck product 1
    let mut proof1 =
        d_sumcheck_product(&hg.shares, &shares_f2.shares, challenge_u, pp, net, sid).await?;
    // Init Phase 2
    let f1 = d_initialize_phase_two(shares_f1, challenge_g, challenge_v, pp, net, sid).await?;
    // Calculate f3*f2(u). Omitted for simplicity.
    // let f2_u = d_fix_variable(&shares_f2.shares, challenge_u, pp, net, sid).await?[0];
    // let f2_u = d_unpack_0(f2_u, pp, net, sid).await?;
    let f2_u = F::one();
    let shares_f3_f2u = shares_f3.mul(&f2_u);
    // Sumcheck product 2
    let proof2 =
        d_sumcheck_product(&f1.shares, &shares_f3_f2u.shares, challenge_v, pp, net, sid).await?;
    proof1.extend(proof2);
    Ok(proof1)
}

```

</details>

<details><summary><b> dhyperplonk.rs </b></summary>

```rust

#[derive(Clone, Debug)]
pub struct PackedProvingParameters<E: Pairing> {
    pub a_evals: Vec<E::ScalarField>,
    pub b_evals: Vec<E::ScalarField>,
    pub c_evals: Vec<E::ScalarField>,
    pub input: Vec<E::ScalarField>,
    pub q1: Vec<E::ScalarField>,
    pub q2: Vec<E::ScalarField>,
    pub sigma_a: Vec<E::ScalarField>,
    pub sigma_b: Vec<E::ScalarField>,
    pub sigma_c: Vec<E::ScalarField>,
    pub sid: Vec<E::ScalarField>,
    // Challenges
    pub eq: Vec<E::ScalarField>,
    pub challenge: Vec<E::ScalarField>,
    pub beta: E::ScalarField,
    pub gamma: E::ScalarField,
    pub commitment: PolynomialCommitment<E>,
    // Masks needed
    pub mask: Vec<E::ScalarField>,
    pub unmask0: Vec<E::ScalarField>,
    pub unmask1: Vec<E::ScalarField>,
    pub unmask2: Vec<E::ScalarField>,
    // Dummies
    pub reduce_target: Vec<E::ScalarField>,
}
...

```

</details>

<details><summary><b> mpc-net/lib.rs </b></summary>

```rust

#[async_trait]
#[auto_impl(&, &mut, Arc)]
pub trait MPCNet: Send + Sync {
    /// Am I the first party?

    fn is_leader(&self) -> bool {
        self.party_id() == 0
    }
    /// How many parties are there?
    fn n_parties(&self) -> usize;
    /// What is my party number (0 to n-1)?
    fn party_id(&self) -> u32;
    /// Is the network layer initalized?
    fn is_init(&self) -> bool;

    /// Get upload/download in bytes
    fn get_comm(&self) -> (usize, usize);

    fn add_comm(&self, up: usize, down: usize);
...

```

</details>

<details><summary><b> secret-sharing </b></summary>

```rust

pub struct PackedSharingParams<F>
where
    F: FftField,
{
    /// Corrupting threshold
    pub t: usize,
    /// Packing factor
    pub l: usize,
    /// Number of parties
    pub n: usize,
    /// Share domain
    pub share: Radix2EvaluationDomain<F>,
    /// Secrets domain
    pub secret: Radix2EvaluationDomain<F>,
    /// Secrets2 domain
    pub secret2: Radix2EvaluationDomain<F>,
}

...

    pub fn new(l: usize) -> Self {
        let n = l * 4;
        let t = l - 1;
        debug_assert_eq!(n, 2 * (t + l + 1));

        let share = Radix2EvaluationDomain::<F>::new(n).unwrap();
        let secret = Radix2EvaluationDomain::<F>::new(l + t + 1)
            .unwrap()
            .get_coset(F::GENERATOR)
            .unwrap();
        let secret2 = Radix2EvaluationDomain::<F>::new(2 * (l + t + 1))
            .unwrap()
            .get_coset(F::GENERATOR)
            .unwrap();
...

    /// Packs secrets into shares
    #[allow(unused)]
    pub fn pack_from_public<G: DomainCoeff<F>>(&self, mut secrets: Vec<G>) -> Vec<G> {
        // assert!(secrets.len() == self.l, "Secrets length mismatch");
        self.pack_from_public_in_place(&mut secrets);
        secrets
    }

    #[allow(unused)]
    pub fn pack_from_public_rand<G: DomainCoeff<F> + UniformRand>(
        &self,
        mut secrets: Vec<G>,
    ) -> Vec<G> {
        assert!(secrets.len() == self.l, "Secrets length mismatch");
        let mut rng = ark_std::test_rng();
        // Resize the secrets with t+1 random points
        let rand_points = (0..self.t + 1)
            .map(|_| G::rand(&mut rng))
            .collect::<Vec<G>>();
        secrets.extend_from_slice(&rand_points);
        self.pack_from_public_in_place(&mut secrets);
        secrets
    }

...

    /// Unpacks shares of degree t+l into secrets
    #[allow(unused)]
    pub fn unpack<G: DomainCoeff<F>>(&self, mut shares: Vec<G>) -> Vec<G> {
        self.unpack_in_place(&mut shares);
        shares
    }

    /// Unpacks shares of degree 2(t+l) into secrets
    #[allow(unused)]
    pub fn unpack2<G: DomainCoeff<F>>(&self, mut shares: Vec<G>) -> Vec<G> {
        debug_assert!(shares.len() == self.n, "Shares length mismatch");
        self.unpack2_in_place(&mut shares);
        shares
    }

```

</details>

</details>

---

[Scalable-Collaborative-zkSNARK-main](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/Scalable-Collaborative-zkSNARK-main)

<div  align="center"> 
<img src="images/55_CoZKPs.gif" width="50%" />
</div>
