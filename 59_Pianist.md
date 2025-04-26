# ETAAcademy-ZKMeme: 59. Pianist

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>59. Pianist</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Pianist</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

---

# Pianist: Distributed SNARKs Overcoming the Bottlenecks of Proof Generation

Distributed SNARK technology speeds up zero-knowledge proofs by enabling collaborative proving across multiple machines, with approaches falling into two categories **Privacy-preserving**(outsourcing computation to multiple servers) and **Computation-accelerating**(using many machines in parallel). Recent systems like deVirgo, HyperPianist, Cirrus, Pianist, and Hekaton use **distributed PIOP + distributed PCS**. Among these, **Pianist** stands out with constant proof size, communication and verification complexity，and efficiently handles large-scale tasks for blockchain applications like zkRollup and zkEVM by integrating with Libra.

Pianist combines a bivariate Plonk constraint system with distributed KZG polynomial commitment, introducing the Robust Collaborative Proving System (RCPS) to verify individual machine contributions. Its main limitation is the 10× overhead when converting R1CS circuits, which has prompted research into more efficient R1CS-native distributed SNARK designs, exploring enhanced **inner product arguments**, new **batchable bivariate KZG commitment variants**, and **lookup arguments for non-fixed tables** combined with **efficient preprocessing methods**.

---

**Pianist** is an innovative distributed zero-knowledge proof (ZKP) system built upon the arithmetic foundations of **Plonk**, designed to support arbitrary general-purpose circuits with massive parallelism. It enables scalable, high-performance proof generation across hundreds of machines and offers **four key advantages**:

- **Fully Distributed Proof Generation**: Pianist supports both **data parallelism** and **arbitrary circuits**, transforming the computing power of hundreds of machines into a single, unified prover. It achieves **linear scalability** with the number of machines, enabling virtually unbounded performance improvements while maintaining **constant communication overhead**, **constant proof size**, and **constant verification time**.

- **Updatable Trusted Setup**: The system employs **KZG-based polynomial commitments** with an **updatable trusted setup**, ensuring both security and flexibility.

- **Compatibility with Plonkish Systems**: By building directly on Plonk, Pianist is naturally compatible with all **Plonkish arithmetization systems**, making it a powerful and flexible backend for modern ZKP applications.

- **Robustness via RCPS**: Pianist introduces the concept of a **Robust Collaborative Proving System (RCPS)**, allowing the **master node** to verify the correctness of messages from each machine before aggregating them into the final proof. This ensures security and fault tolerance even in the presence of potentially malicious or faulty nodes.

**Core Technology: Bivariate Plonk Constraints + Distributed KZG Commitments**

At the heart of Pianist lies a novel technical foundation: the **bivariate extension of Plonk’s constraint system** coupled with **distributed KZG polynomial commitments**. The key innovation is the introduction of a second variable $Y$ and **Lagrange polynomials** $R_i(Y)$, which encode the identity of each machine in the distributed environment.

Each prover node $i$ starts with a **univariate constraint polynomial**:

$g_i(X) := q_{a,i}(X)a_i(X) + q_{b,i}(X)b_i(X) + q_{o,i}(X)o_i(X) + q_{ab,i}(X)a_i(X)b_i(X) + q_{c,i}(X)$

To integrate the outputs of all machines into a single system, Pianist constructs a **bivariate witness polynomial**:

$S(Y, X) = \sum_{i=0}^{M-1} R_i(Y) \cdot s_i(X)$

where:

- $M$ is the total number of machines,
- $s_i(X)$ is a witness column ( e.g., $a_i(X), b_i(X), o_i(X)$ ) from machine $i$,
- $R_i(Y)$ is the **Lagrange polynomial** associated with node $i$, defined over the $M$-th roots of unity.

This method is extended to all input/output/constraint polynomials:

$A(Y,X)=\sum_{i=0}^{M-1}R_i(Y)a_i(X)，B(Y,X)=\sum_{i=0}^{M-1}R_i(Y)b_i(X)， O(Y,X)=\sum_{i=0}^{M-1}R_i(Y)o_i(X)$

These are then combined into a global bivariate constraint polynomial:

$G(Y, X) := Q_a(Y, X)A(Y, X) + Q_b(Y, X)B(Y, X) + Q_{ab}(Y, X)A(Y, X)B(Y, X) + Q_o(Y, X)O(Y, X) + Q_c(Y, X)$

During proof generation:

- The system selects a random point $\alpha$ for the $X$-dimension,
- Each prover evaluates its witness polynomial at $\alpha$, computing $s_i(\alpha)$,
- Each machine sends $s_i(\alpha)$ and its **KZG opening proof** to the master node.

The brilliance lies in the properties of the Lagrange polynomials $R_i(Y)$:

- $R_i(\omega^i) = 1$ at the $i$-th root of unity,
- $R_i(\omega^j) = 0$ for $j \ne i$.

This guarantees that when $Y = \omega^i$, the global constraint $G(Y, X)$ simplifies **exactly** to the local constraint $g_i(X)$, ensuring correctness. It also means each node’s computations remain **independent**, while the **global proof remains unified and verifiable**.

By blending Plonk-style arithmetic constraints with bivariate polynomial composition and robust KZG commitments, Pianist creates a **highly scalable, efficient, and secure** distributed SNARK system. Its proof size and verifier complexity remain **constant**, even as the number of machines and the size of the circuit scale up.

Pianist’s design makes it an ideal ZKP engine for next-generation blockchain applications where performance, scalability, and security are non-negotiable.

---

## 1. Pianist(Plonk via Unlimited Distribution)

Despite rapid advances in zkSNARK technology, high-performance proof systems still face significant computational bottlenecks when handling large-scale circuits. To address this challenge, researchers have explored various **distributed zero-knowledge proof (ZKP)** architectures. Early efforts like **DIZK** marked an important step forward, enabling proofs for circuits 100 times larger than single-machine capacity by leveraging **128 distributed machines**. However, DIZK suffers from significant communication overhead. Its reliance on the **MapReduce framework** to perform distributed **Number Theoretic Transforms (NTT)** causes computation time to grow quickly with circuit size, while communication costs scale **linearly with the total circuit size** $O(N)$, even though it maintains a **constant proof size**.

**deVirgo** further reduced the per-machine computational time but retained DIZK’s key limitations: **linear communication complexity** and a **proof size that increases with the number of machines**. Moreover, deVirgo only supports **data-parallel circuits**, and does not fully support general-purpose computations.

In contrast, **Pianist** offers a fundamentally more scalable and robust approach. It simultaneously achieves:

- **Optimal linear scalability** of prover time with respect to the number of machines,
- **Minimal inter-machine communication**, with communication complexity depending only on the **number of machines** $O(M)$, not on the total circuit size,
- **Constant proof size** $O(1)$ regardless of circuit size or machine count,
- **Robustness**, allowing the detection and exclusion of malicious or faulty participants through partial proof verification.

**A New Paradigm for Scalable zkSNARKs**

Pianist breaks through the limitations of prior distributed ZKP systems by decoupling communication cost from the total circuit size. This enables near-perfect scalability with constant proof size, low communication overhead, and strong security guarantees.

By supporting **general-purpose circuits** and introducing a **robust collaborative proof architecture**, Pianist sets a new standard for distributed zkSNARK systems—ideal for applications like **zkRollups**, **zkEVM**, and any use case requiring **secure, large-scale, and distributed computation**.

### Beyond Recursion: Distributed ZKP Architecture vs. PCD and IVC

In addition to distributed proof systems, alternative techniques have been proposed for handling large-scale computations—particularly **Proof-Carrying Data (PCD)** and **Incrementally Verifiable Computation (IVC)**, which were discussed in earlier works involving **Reed-Solomon codes**.

PCD is a novel cryptographic framework that decomposes large computations into a sequence of steps. At each step, the prover not only demonstrates the correctness of the current computation but also ensures that all previous steps were valid. This structure is especially well-suited for **data-parallel circuits** running in **memory-constrained environments**.

There are two primary approaches to implementing PCD:

- **Succinct verification:** Each step generates a proof for the current computation and verifies the proof from the previous step.
- **Accumulation-based approach:** Instead of verifying at each step, SNARK verifications are postponed and accumulated, with all verifications performed at the final step.

While PCD and IVC do not correspond directly to general-purpose circuits, certain systems have claimed IVC capabilities. These methods focus on **incrementally verifying long-running computations**, such as proving the iterative application of a non-deterministic function $f$, as in $f^n(z_0)$, supported by the **Nova** system.

Building on proof aggregation, **aPlonk** introduced a **multi-polynomial commitment scheme**, enabling multiple provers to combine their polynomial commitments. It uses a **generalized inner-product argument (IPA)** to batch open proofs and shifts most verification responsibilities from the verifier to the prover. However, all prover nodes must share the same **Fiat–Shamir randomness**, which requires frequent synchronization and communication between nodes during proof generation. This reliance on **recursive proving and synchronized commitments** introduces both complexity and inefficiency, especially in high-throughput environments.

**Distributed ZKPs over Recursive Approaches**

These limitations form the basis of **Pianist’s architectural choice**: unlike recursive proof systems like PCD and IVC, Pianist avoids the overhead of generating and aggregating separate proofs for each subcircuit. While DIZK took a direct distributed approach, it was hampered by excessive communication costs from running **interleaved NTT operations** over large networks—scaling linearly with the circuit size $O(N)$.

Pianist addresses both of these issues with a novel combination of cryptographic design and distributed system engineering:

- It leverages a **knowledge-sound IOP constraint system**.
- It transforms this into an **interactive proof of knowledge** using a **bivariate polynomial commitment scheme**.
- This approach “**splits**” large NTT instances across machines to reduce communication, and “**merges**” the resulting subcircuit proofs into a unified proof without costly recursive aggregation.

By instantiating the system with a **bivariate KZG commitment**, Pianist enables a distributed computation model akin to "**independent work + efficient reporting**." The system supports **parallel processing** of large-scale circuits with **constant proof size and verification time**, even in massive computations.

**A Scalable Alternative for zkRollup and zkEVM**

Unlike aPlonk, Pianist avoids frequent synchronization, and unlike PCD, it does not follow a strictly sequential model. This makes it particularly effective for **blockchain scalability challenges**, especially in Layer-2 systems such as **zkRollups** and **zkEVMs**, where the burden of generating zk-proofs has become a bottleneck. These environments often require **high-memory machines (terabytes of RAM)** to process thousands of transactions per batch.

Pianist introduces two protocol variants, both based on Plonk:

- One optimized for **data-parallel circuits**,
- The other for **general-purpose circuits**.

Both follow a **mining pool-like model**, where the computation is distributed across $M$ machines. Each machine processes a subcircuit $C_i$ of size $T = N / M$, reducing prover time from $O(MT \log MT)$ to $O(T \log T + M \log M)$. Crucially, each machine only needs **$O(1)$ communication**, and the system retains the **original Plonk’s constant proof size and verification time**.

### Bivariate Polynomial IOP

Pianist is a fully distributed SNARK (Succinct Non-Interactive Argument of Knowledge) system that satisfies the critical properties of an interactive argument $(ARG = (G, P, V)):$ _completeness_ (the prover can always convince the verifier of a correct statement) and witness-extendable simulation (an extractor exists that can derive the witness from any successful prover). The system adopts a public-coin polynomial IOP (PIOP), a variant of interactive oracle proofs where all prover messages are encoded as polynomials. By applying the Fiat-Shamir transform, Pianist achieves non-interactive zero-knowledge while maintaining succinctness in proof size $(|\pi| = poly(\lambda, log|C|))$ and verification time $(poly(\lambda, |x|, log|C|)).$

Pianist's technical innovation lies in the combination of a bivariate KZG polynomial commitment scheme with a bivariate polynomial IOP. This enables distributed machines to independently commit to subcircuits and construct a complete distributed proof by exchanging only specific evaluation points and succinct proofs. Built atop the Plonk protocol, Pianist rigorously encodes each subcircuit into an arithmetic constraint system. Inputs and outputs of circuit gates are mapped into polynomials, and carefully crafted selector polynomials distinguish between different gate types to form gate constraint equations. To ensure circuit integrity, Pianist implements copy constraints using structured product arguments and running product polynomials. The prover constructs polynomial relations representing wiring constraints and engages in a structured three-round interaction to convince the verifier of constraint satisfaction, thereby guaranteeing the correctness of the full computation.

At its core, Pianist uses bivariate polynomials to build a distributed constraint system, aggregating polynomials from independently computed subcircuits into a single bivariate polynomial while preserving constraint structure. A naive aggregation approach such as $A(Y, X) = \sum_{i=0}^{M-1} Y^i a_i(X)$ introduces problematic cross-terms like $Y^i a_i(X) \cdot Y^j b_j(X)$. Inspired by Caulk, Pianist instead leverages Lagrange polynomials $R_i(Y)$ as aggregation tools. Each party i holds a local witness vector $\vec{a_i} = (a_{i,0}, \dots, a_{i,T-1})$, which is interpolated into a univariate polynomial $a_i(X)$ using Lagrange polynomials $L_j(X)$ over the T-th roots of unity. These are combined using a second set of Lagrange polynomials $R_i(Y)$ (over M-th roots of unity) to form a bivariate polynomial $A(Y, X) = \sum_{i=0}^{M-1} a_i(X) R_i(Y)$.

Lowercase letters (a, b, c) denote local univariate polynomials, while uppercase letters (A, B, C) represent aggregated bivariate polynomials. (x, y) refers to concrete evaluations, and (X, Y) are the symbolic polynomial variables.The key property of Lagrange polynomials — where $R_i(\omega_Y^j) = 1$ when $i = j$ and $0$ otherwise — enables $A(\omega_Y^i, X) = \sum_{j=0}^{M-1} a_j(X)R_j(\omega_Y^i) = a_i(X) \cdot 1 + \sum_{j \neq i} a_j(X) \cdot 0 = a_i(X)$, allowing precise extraction of each party's contribution. This property underpins Pianist's ability to encode global circuit structure while enabling efficient local computation.

Using this structure, Pianist defines aggregated Plonk-style gate constraints $G(Y, X)$, copy constraints $P_0(Y, X)$ and $P_1(Y, X)$, and a linear combination constraint:

$G(Y, X) + \lambda P_0(Y, X) + \lambda^2 P_1(Y, X) - V_X(X) H_X(Y, X) = V_Y(Y) H_Y(Y, X),$

where $V_Y(Y) = Y^M - 1$. Due to the selector property of Lagrange polynomials, this constraint reduces to checking the i-th subcircuit when Y is set to $\omega_Y^i$.

All polynomials except $H_Y(Y, X)$ can be maintained entirely in a distributed fashion. To handle $H_Y$, the prover sends only its evaluation at a random verifier-supplied point $\alpha$. This allows Pianist to operate in a truly distributed mode: each machine works independently and communicates minimally to produce a unified, succinct proof. The system remains efficient and scalable not only for fully data-parallel tasks, but also supports simple inter-circuit wiring through custom gates and Y-variable shifts.

### Distributed KZG Commitments and Extending to Arbitrary Circuits with Robust Coordination

While Pianist was initially designed to efficiently handle data-parallel subcircuits, it significantly advances beyond this limitation by supporting **arbitrary general circuits**, addressing a core challenge in distributed SNARK design: **ensuring correct wiring across independently computed subcircuits**.

**Generalizing Wire Connections Across Subcircuits**

Traditional Plonk protocols use permutation polynomials $\sigma_a(X)$, $\sigma_b(X)$, and $\sigma_c(X)$—to manage wiring within a single circuit. Pianist generalizes this concept by introducing a **two-dimensional coordinate system** for wire mappings across distributed subcircuits. Specifically, for each input/output type $\{(\sigma_{Y,s,i}(X), \sigma_{X,s,i}(X))\}_{s\in\{a,b,o\}}$, Pianist defines:

$(\sigma_{Y,s,i}(\omega_X^j), \sigma_{X,s,i}(\omega_X^j)) = (\omega_Y^{i'}, k'_{s} \cdot \omega_X^{j'})$

This mapping captures how the $j$-th entry of the $s$-type polynomial in the $i$-th machine's subcircuit connects to the $j'$-th entry in the $s'$-type polynomial of the $i'$-th machine. In essence, it accurately encodes the **cross-subcircuit wire routing** in the distributed circuit.

To verify the global correctness of all such cross-connections, Pianist ensures the following identity:

$\prod_{i=0}^{M-1}\prod_{j=0}^{T-1}\frac{f_i(\omega_X^j)}{f'_i(\omega_X^j)} = 1$

where $f_i$ and $f'_i$ are constructed to encapsulate the source and destination of each connection. However, due to the nature of distribution, **local running products $z_i(X)$ may not naturally close** at the subcircuit boundaries. Specifically:

$z_i^* = z_i(\omega_X^{T-1}) \cdot \frac{f_i(\omega_X^{T-1})}{f'_i(\omega_X^{T-1})} \neq 1$

To address this, Pianist introduces an **auxiliary polynomial $W(X)$**, which "bridges" the endpoints of subcircuit connections. With this, Pianist defines four core constraints:

- **Initial value constraint** ($P_0$)
- **Internal product consistency** ($P_1$)
- **Cross-subcircuit start point constraint** ($P_2$)
- **Cross-subcircuit endpoint constraint** ($P_3$)

These constraints are then aggregated with the arithmetic gate constraint $G(Y, X)$ to form a unified constraint system. This design enables Pianist to handle **arbitrary wiring structures** in a distributed SNARK setting with provable correctness.

**Distributed KZG Commitments and Efficient PCS Construction**

To bring these theoretical advancements to practical use, Pianist integrates a fully **distributed KZG polynomial commitment scheme**, enabling a **completely decentralized SNARK construction**. This realization answers a long-standing challenge: **Can we construct an efficient SNARK protocol entirely from a distributed IOP and a distributed PCS? Pianist answers yes.**

In this setting, a degree-N polynomial is split across $M$ machines, each holding a chunk of size $T = N/M$. The commitment scheme achieves three key goals:

- **Linear-time proof generation** across all machines
- **Minimal communication overhead**
- **Preserved succinctness** in proof size and verification time

The distributed PCS used in Pianist meets the standard requirements of **completeness** and **knowledge soundness** while achieving impressive efficiency:

- **Proof computation:**

  - Total: $O(N)$ group operations
  - Per-node: $O(N/M)$ group operations
  - Master node: $O(N/M + M)$ group operations

- **Communication between provers and the aggregator ($P_0$):**

  - Constant: $O(1)$

- **Proof size and verification cost:**
  - Constant: $O(1)$ group elements and operations

During operation, each prover independently computes its local commitments and subproofs. The master node aggregates these results with **only constant communication** and **almost perfectly balanced workloads**. The total proof generation involves $O(N \log T + M \log M)$ field operations and $O(N)$ group operations—achieving **true scalability and simplicity** in distributed proving.

**RCPS: Robust Collaboration in the Face of Malicious Actors**

One of Pianist’s most innovative contributions is its **Robust Collaborative Proving System (RCPS)**—a protocol designed to handle adversarial or unreliable provers in a distributed zero-knowledge setting.

Built atop the distributed PCS and IOP, RCPS introduces a complete framework capable of functioning even under malicious conditions. The system ensures that:

- Even if some prover nodes are compromised, the **final proof remains valid**
- Malicious contributions are **detected and filtered out** automatically

RCPS achieves this through:

- A structured protocol suite (e.g., `Setup`, `SplitCircuit`, `MasterKeyGen`)
- Four security properties:
  - **Partial completeness**
  - **Full completeness**
  - **Partial witness extension simulation**
  - **Full witness extension simulation**

A central verifier node ($P_0$) not only aggregates local proofs but also **verifies the validity of each node’s contribution**. The system outputs a **vector of accept/reject decisions** identifying trustworthy participants, and only these nodes’ inputs are used to generate the final proof.

---

## 2. Towards a Scalable Distributed SNARK for R1CS

In modern SNARK systems, the proving time remains a dominant bottleneck. Distributed SNARKs aim to overcome this by allowing multiple provers to collaboratively generate a proof, significantly enhancing efficiency and scalability for large-scale computations. Among them, Pianist is currently the state-of-the-art distributed SNARK for Plonk-style circuits, offering constant size proofs, average communication cost, and verifier complexity. However, when applied to Rank-1 Constraint Systems (R1CS), it requires converting R1CS to Plonk form—a transformation that dramatically increases the circuit size and incurs roughly 10× overhead.

This inefficiency stems from the fundamental differences between R1CS and Plonk. While Plonk excels at handling arbitrary nonlinear gates and supports custom constraints, R1CS offers free addition gates and benefits from a mature ecosystem, including widely adopted DSLs and toolchains such as Circom. Unfortunately, using Plonk-optimized SNARKs like Pianist to prove R1CS statements leads to significant performance penalties. For instance, converting a circuit with 261,833 R1CS constraints can result in over 2.5 million Plonk constraints, increasing the setup and proof cost substantially. On the other hand, existing distributed SNARK systems tailored for R1CS, such as DIZK and Hekaton, lag behind Pianist in asymptotic efficiency.

This tradeoff presents a critical challenge across applications like blockchains, zero-knowledge virtual machines, verifiable fully homomorphic encryption, and privacy-preserving machine learning. The technical difficulty arises from the structure of R1CS itself—it models constraints as sparse $n \times n$ matrices with $O(n)$ nonzero elements, making efficient distributed processing more complex than in Plonk. Existing approaches often suffer from linear communication costs (e.g., Ligero-based schemes) or logarithmic proof size and verifier time (e.g., methods using univariate and multilinear sum-checks).

A novel solution to this challenge lies at the intersection of two core technologies: inner product arguments (IPA) and distributed SNARKs. This approach introduces a highly efficient inner product argument, combined with a preprocessed variant of the KZG commitment scheme adapted to support batched bivariate polynomial commitments and lookup arguments without precomputed tables. As a result, it achieves constant verifier complexity and supports general computation beyond data-parallel circuits, while remaining compatible with a scalable distributed setup.

In terms of IPA, recent techniques span discrete logarithm-based constructions and Reed-Solomon code-based methods, offering logarithmic proof sizes and avoiding trusted setup. Alternatively, modular IPA constructions using PIOP + PCS frameworks allow flexible tradeoffs via the choice of the underlying PCS. Notably, KZG-based variants can achieve constant-sized proofs, particularly those leveraging univariate sum-checks and Laurent polynomials.

Distributed SNARKs generally fall into two categories. The first relies on secret-sharing the witness across multiple machines for privacy, though this offers limited acceleration as each share scales with the original witness size. The second—and more relevant—category uses multiple machines to accelerate a single prover's workload. Examples include DIZK, which extends Groth16 over R1CS using distributed algorithms for MSMs and FFTs. However, its reliance on distributed FFTs leads to high communication overhead and increases total proving time.

Recent work (e.g., deVirgo, HyperPianist, Cirrus, Pianist, Hekaton) shares a core strategy: decomposing a large statement into smaller substatements processed by multiple provers, then aggregating the subproofs into a final proof. These systems typically follow a "distributed PIOP + distributed PCS" design, with efficiency determined by the specific protocol details.

This new line of work brings four key innovations:

- **Distributed IOP Architecture**: A novel distributed interactive oracle proof design transforms R1CS constraints into a combination of inner product and Hadamard product problems, which can then be split into independently solvable subproblems.
- **Improved Inner Product Argument**: A refined IPA scheme combines the strengths of Laurent polynomials and univariate sum-checks, resulting in minimal proof size, fewer PCS commitments, and reduced FFT operations.
- **Efficient Preprocessing Protocol**: New techniques such as low-degree-encoded polynomial remapping, efficient online table validity checks, and table decomposition reduce verifier complexity from linear to constant.
- **Distributed Batched Bivariate KZG Commitments**: An optimized PCS protocol that supports batched commitments with constant proof size and verification cost, enabling high-performance, fully distributed SNARKs for R1CS.

These advances pave the way for a new generation of scalable, efficient, and general-purpose distributed SNARK systems that natively support R1CS without costly format conversions.

### A New Inner Product PIOP for Efficient Distributed SNARKs on R1CS

In the design of efficient distributed SNARK systems for Rank-1 Constraint Systems (R1CS), the inner product polynomial interactive oracle proof (PIOP) plays a crucial role. Two representative PIOP constructions have emerged: one based on _univariate sumcheck_ protocols (which reduces the number of oracles and queries), and another based on _Laurent polynomials_ (which minimizes FFT operations by working with coefficient-based representations). By combining the strengths of both approaches, researchers have proposed a new, highly efficient inner product PIOP that serves as the backbone for distributed SNARK protocols targeting R1CS.

**Novel Algebraic Insight**

The key innovation in this new PIOP is the discovery of a novel algebraic expression connecting inner products and univariate sumchecks:

$\langle \mathbf{f}, \mathbf{s} \rangle = y \iff \sum_{x \in H} f(x)s(x^{-1}) = m \cdot y$

Here, $\mathbf{f}$ and $\mathbf{s}$ are vectors, and $f(X)$, $s(X)$ are their corresponding polynomial encodings. $H$ is a multiplicative subgroup of size $m$. To overcome the challenge posed by evaluating $s(x^{-1})$ (i.e., at inverse points), a new polynomial $s'(X)$ is constructed such that it matches $s(X^{-1})$ on all elements of $H$. This transformation allows the inner product relation to be embedded into a standard sumcheck protocol.

To further enhance efficiency, the protocol multiplies the identity by a random linear term $(X - u)$, removing the need for low-degree testing and streamlining the proof. The resulting protocol consists of only four rounds:

- The prover commits to the polynomials $f(X)$ and $s'(X)$,
- The verifier sends a random challenge $u$,
- The prover responds with polynomials $g'(X)$ and $h(X)$,
- The verifier queries these polynomials at a single random point $\alpha$ and checks the final identity.

This construction results in:

- **Proof size**: only 4 polynomial oracles,
- **Soundness error**: $\frac{2m}{|\mathbb{F}|}$,
- **Prover complexity**: $6 \cdot \text{FFT}(m, m) + O(m)$,
- **Verifier complexity**: $O(\log m)$.

Compared to existing inner product arguments, this new approach significantly improves efficiency and lays the groundwork for high-performance SNARKs tailored to R1CS.

**Transforming R1CS Constraints into Distributed Inner Product PIOPs**

This new inner product PIOP framework also enables the efficient transformation of complex R1CS constraints into distributed PIOP-friendly formats. The process begins by decomposing R1CS into three linear constraints and one multiplicative (quadratic) constraint. Freivalds' technique is then applied to transform the linear constraints $Pw = a$ into randomized inner product form:

$\mathbf{r}^\top Pw = \mathbf{r}^\top a$

To enable parallel computation, the system partitions the large inner product of size $m\ell$ into $\ell$ smaller inner products of size $m$. This is achieved by splitting the vectors $\mathbf{p}' = \mathbf{r}^\top P$ and $\mathbf{w}$ into $\ell$ sub-vectors, and assembling them into matrices $P'$ and $W$. The resulting form becomes:

$\sum_{i \in [\ell]} \langle P'[i], W[i] \rangle = \sum_{i \in [\ell]} \langle A[i], R[i] \rangle$

For the quadratic constraint $\mathbf{a} \circ \mathbf{b} = \mathbf{c}$, a similar decomposition is applied, resulting in $\ell$ small element-wise product constraints $A[i] \circ B[i] = C[i]$. These are then randomized and combined into inner products using structured challenge vectors, further reducing the number of required proofs.

Additionally, to minimize communication, the system constructs virtual polynomials to implicitly compute $f_{B[i]}'$ from $f_{B[i]}$, allowing the removal of an entire polynomial oracle.

This hybrid approach—fusing advanced inner product PIOP design with strategic R1CS-to-inner-product conversion—provides a foundation for scalable and efficient distributed SNARK systems, opening up possibilities for real-world deployment in blockchain, zero-knowledge virtual machines, and privacy-preserving computation.

### Efficient Batched Bivariate KZG Polynomial Commitment

The development of a high-performance distributed SNARK system fundamentally relies on an efficient **batched bivariate KZG polynomial commitment scheme**. One of the key challenges is to prove multiple evaluations across several bivariate polynomials efficiently. Naive extensions of univariate batched KZG methods to the bivariate case result in significant computational and communication overhead. This work overcomes the core limitations through two key innovations:

- **Evaluation Point Padding**: By extending the set of evaluation points into a Cartesian product $R \times S$, the quotient equations involved in the proof can be unified under a more structured form, enabling more systematic handling of the polynomial divisions.

- **Introduction of Auxiliary Polynomial $s(X, Y)$**: This polynomial enables a decomposition of the original quotient equation into two logically independent parts. Each part can be subjected to a separate random linear combination, simplifying the proof structure and enhancing efficiency.

The protocol proceeds through a sequence of well-defined steps: structured reference string (SRS) generation, construction of the auxiliary polynomial $s(X, Y)$, definition of the quotient polynomials, and a robust verification mechanism. In particular, for the special case where all evaluation points share the same Y-coordinate, the bivariate problem is reduced to a univariate one. This optimization further reduces the proof size to just **4 group elements** and **$n + 1$** field elements. Since field elements are typically 3–4 times smaller than group elements, this reduction has major practical implications.

**Distributed Batched Bivariate Polynomial Commitments**

Building upon the Pianist system’s distributed polynomial commitment scheme (PCS) design, this work extends the batched bivariate KZG mechanism to the distributed setting. The core algebraic equation

$f(X, Y) - r(X, Y) = P(X, Y) \cdot Z_R(X) + q(X, Y) \cdot Z_S(Y)$

is generalized for the distributed environment. The transformation leverages the auxiliary polynomial $s(X, Y)$ and the interpolation properties of **Lagrange basis polynomials** $N_j$, enabling a structure where the quotient polynomial $P(X, Y) \cdot Z_R(X)$ can be split into components computed independently by multiple provers and later aggregated. Meanwhile, the remaining part $q(X, Y)$ can be computed centrally by the main prover in $O(|R| + \ell)$ time.

The complete distributed protocol includes:

- Generation of structured SRS parameters and polynomial commitments,
- Construction of helper polynomials,
- Computation of quotient polynomials, and
- A verification phase optimized for the demands of R1CS-based PIOP.

The system adapts the SRS structure specifically for compatibility with distributed R1CS PIOP. The computational complexity is as follows:

- **Each sub-prover** performs $O(m)$ group operations and $O(nm + nt)$ field operations,
- The **main prover** performs $O(\ell t)$ group operations and $O(\ell nt)$ field operations,
- Each prover communicates only $O(n)$ group elements.

Despite operating in a distributed setting, the system retains the same **proof size** and **verifier complexity** as its non-distributed counterpart.

By integrating all the innovations discussed—including:

- the improved inner product PIOP,
- the distributed PIOP transformation for R1CS,
- sublinear verifier optimizations,
- the batched bivariate KZG commitment scheme,
- and its distributed generalization—

the work successfully constructs a theoretically sound and practically efficient **distributed SNARK system for R1CS**. This represents a major step forward in scalable zero-knowledge proof systems and distributed cryptographic protocols.

<details><summary><b> Code </b></summary>

<details><summary><b> backend/backend.go </b></summary>

```go
// ID represent a unique ID for a proving scheme
type ID uint16

const (
    UNKNOWN ID = iota
    GROTH16
    PLONK
)

// ProverConfig is the configuration for the prover with the options applied.
type ProverConfig struct {
    Force         bool                      // defaults to false
    HintFunctions map[hint.ID]hint.Function // defaults to all built-in hint functions
    CircuitLogger zerolog.Logger            // defaults to gnark.Logger
}

// NewProverConfig returns a default ProverConfig with given prover options opts
// applied.
func NewProverConfig(opts ...ProverOption) (ProverConfig, error) {
    log := logger.Logger()
    opt := ProverConfig{CircuitLogger: log, HintFunctions: make(map[hint.ID]hint.Function)}
    for _, v := range hint.GetRegistered() {
        opt.HintFunctions[hint.UUID(v)] = v
    }
    for _, option := range opts {
        if err := option(&opt); err != nil {
            return ProverConfig{}, err
        }
    }
    return opt, nil
}
```

</details>

<details><summary><b> backend/piano/piano.go or gpiano.go </b></summary>

```go
// Setup prepares the public data associated to a circuit + public inputs.
func Setup(ccs frontend.CompiledConstraintSystem, publicWitness *witness.Witness) (ProvingKey, VerifyingKey, error) {

    switch tccs := ccs.(type) {
    case *cs_bn254.SparseR1CS:
        w, ok := publicWitness.Vector.(*witness_bn254.Witness)
        if !ok {
            return nil, nil, witness.ErrInvalidWitness
        }
        return piano_bn254.Setup(tccs, *w)
    default:
        panic("unimplemented")
    }

}

// Prove generates piano proof from a circuit, associated preprocessed public data, and the witness
// if the force flag is set:
// 	will executes all the prover computations, even if the witness is invalid
//  will produce an invalid proof
//	internally, the solution vector to the SparseR1CS will be filled with random values which may impact benchmarking
func Prove(ccs frontend.CompiledConstraintSystem, pk ProvingKey, fullWitness *witness.Witness, opts ...backend.ProverOption) (Proof, error) {

    // apply options
    opt, err := backend.NewProverConfig(opts...)
    if err != nil {
        return nil, err
    }

    switch tccs := ccs.(type) {
    case *cs_bn254.SparseR1CS:
        w, ok := fullWitness.Vector.(*witness_bn254.Witness)
        if !ok {
            return nil, witness.ErrInvalidWitness
        }
        return piano_bn254.Prove(tccs, pk.(*piano_bn254.ProvingKey), *w, opt)

    default:
        panic("unimplemented")
    }
}

// Verify verifies a piano proof, from the proof, preprocessed public data, and public witness.
func Verify(proof Proof, vk VerifyingKey, publicWitness *witness.Witness) error {

    switch _proof := proof.(type) {

    case *piano_bn254.Proof:
        w, ok := publicWitness.Vector.(*witness_bn254.Witness)
        if !ok {
            return witness.ErrInvalidWitness
        }
        return piano_bn254.Verify(_proof, vk.(*piano_bn254.VerifyingKey), *w)

    default:
        panic("unimplemented")
    }
}
```

</details>

<details><summary><b> internal/backend/r1cs.go </b></summary>

```go
// Solve sets all the wires and returns the a, b, c vectors.
// the cs system should have been compiled before. The entries in a, b, c are in Montgomery form.
// a, b, c vectors: ab-c = hz
// witness = [publicWires | secretWires] (without the ONE_WIRE !)
// returns  [publicWires | secretWires | internalWires ]
func (cs *R1CS) Solve(witness, a, b, c []fr.Element, opt backend.ProverConfig) ([]fr.Element, error) {
    log := logger.Logger().With().Str("curve", cs.CurveID().String()).Int("nbConstraints", len(cs.Constraints)).Str("backend", "groth16").Logger()

    nbWires := cs.NbPublicVariables + cs.NbSecretVariables + cs.NbInternalVariables
    solution, err := newSolution(nbWires, opt.HintFunctions, cs.MHintsDependencies, cs.MHints, cs.Coefficients)
    if err != nil {
        return make([]fr.Element, nbWires), err
    }
    start := time.Now()

    if len(witness) != int(cs.NbPublicVariables-1+cs.NbSecretVariables) { // - 1 for ONE_WIRE
        err = fmt.Errorf("invalid witness size, got %d, expected %d = %d (public) + %d (secret)", len(witness), int(cs.NbPublicVariables-1+cs.NbSecretVariables), cs.NbPublicVariables-1, cs.NbSecretVariables)
        log.Err(err).Send()
        return solution.values, err
    }

    // compute the wires and the a, b, c polynomials
    if len(a) != len(cs.Constraints) || len(b) != len(cs.Constraints) || len(c) != len(cs.Constraints) {
        err = errors.New("invalid input size: len(a, b, c) == len(Constraints)")
        log.Err(err).Send()
        return solution.values, err
    }

    solution.solved[0] = true // ONE_WIRE
    solution.values[0].SetOne()
    copy(solution.values[1:], witness)
    for i := 0; i < len(witness); i++ {
        solution.solved[i+1] = true
    }

    // keep track of the number of wire instantiations we do, for a sanity check to ensure
    // we instantiated all wires
    solution.nbSolved += uint64(len(witness) + 1)

    // now that we know all inputs are set, defer log printing once all solution.values are computed
    // (or sooner, if a constraint is not satisfied)
    defer solution.printLogs(opt.CircuitLogger, cs.Logs)

    if err := cs.parallelSolve(a, b, c, &solution); err != nil {
        if unsatisfiedErr, ok := err.(*UnsatisfiedConstraintError); ok {
            log.Err(errors.New("unsatisfied constraint")).Int("id", unsatisfiedErr.CID).Send()
        } else {
            log.Err(err).Send()
        }
        return solution.values, err
    }

    // sanity check; ensure all wires are marked as "instantiated"
    if !solution.isValid() {
        log.Err(errors.New("solver didn't instantiate all wires")).Send()
        panic("solver didn't instantiate all wires")
    }

    log.Debug().Dur("took", time.Since(start)).Msg("constraint system solver done")

    return solution.values, nil
}

// IsSolved returns nil if given witness solves the R1CS and error otherwise
// this method wraps cs.Solve() and allocates cs.Solve() inputs
func (cs *R1CS) IsSolved(witness *witness.Witness, opts ...backend.ProverOption) error {
    opt, err := backend.NewProverConfig(opts...)
    if err != nil {
        return err
    }

    a := make([]fr.Element, len(cs.Constraints))
    b := make([]fr.Element, len(cs.Constraints))
    c := make([]fr.Element, len(cs.Constraints))
    v := witness.Vector.(*bn254witness.Witness)
    _, err = cs.Solve(*v, a, b, c, opt)
    return err
}

// GetConstraints return a list of constraint formatted as L⋅R == O
// such that [0] -> L, [1] -> R, [2] -> O
func (cs *R1CS) GetConstraints() [][]string {
    r := make([][]string, 0, len(cs.Constraints))
    for _, c := range cs.Constraints {
        // for each constraint, we build a string representation of it's L, R and O part
        // if we are worried about perf for large cs, we could do a string builder + csv format.
        var line [3]string
        line[0] = cs.vtoString(c.L)
        line[1] = cs.vtoString(c.R)
        line[2] = cs.vtoString(c.O)
        r = append(r, line[:])
    }
    return r
}
```

</details>

<details><summary><b> internal/backend/bn254/piano or gpiano </b></summary>

<details><summary><b> setup.go </b></summary>

```go
// ProvingKey stores the data needed to generate a proof:
// * the commitment scheme
// * ql, prepended with as many ones as they are public inputs
// * qr, qm, qo prepended with as many zeroes as there are public inputs.
// * qk, prepended with as many zeroes as public inputs, to be completed by the prover
// with the list of public inputs.
// * sigma_1, sigma_2, sigma_3 in both basis
// * the copy constraint permutation
type ProvingKey struct {
	// Verifying Key is embedded into the proving key (needed by Prove)
	Vk *VerifyingKey

	// qr,ql,qm,qo (in canonical basis).
	Ql, Qr, Qm, Qo []fr.Element

	// qk in Lagrange basis (canonical basis), prepended with as many zeroes as public inputs.
	// Storing LQk in Lagrange basis saves a fft...
	Qk []fr.Element

	// Domains used for the FFTs.
	// Domain[0] = small Domain
	// Domain[1] = big Domain
	Domain [2]fft.Domain
	// Domain[0], Domain[1] fft.Domain

	// Permutation polynomials
	S1Canonical, S2Canonical, S3Canonical     []fr.Element

	// position -> permuted position (position in [0,3*sizeSystem-1])
	Permutation []int64
}

// VerifyingKey stores the data needed to verify a proof:
// * The commitment scheme
// * Commitments of ql prepended with as many ones as there are public inputs
// * Commitments of qr, qm, qo, qk prepended with as many zeroes as there are public inputs
// * Commitments to S1, S2, S3
type VerifyingKey struct {
	// Size circuit
	SizeY             uint64
	SizeX             uint64
	SizeYInv          fr.Element
	SizeXInv          fr.Element
	Generator         fr.Element
	NbPublicVariables uint64

	// Commitment scheme that is used for an instantiation of PLONK
	DKZGSRS *dkzg.SRS
	KZGSRS *kzg.SRS

	// cosetShift generator of the coset on the small domain
	CosetShift fr.Element

	// S commitments to S1, S2, S3
	S [3]kzg.Digest

	// Commitments to ql, qr, qm, qo prepended with as many zeroes (ones for l) as there are public inputs.
	// In particular Qk is not complete.
	Ql, Qr, Qm, Qo, Qk kzg.Digest
}

// Setup sets proving and verifying keys
func Setup(spr *cs.SparseR1CS, publicWitness bn254witness.Witness) (*ProvingKey, *VerifyingKey, error) {
	globalDomain[0] = fft.NewDomain(mpi.WorldSize)
	if mpi.WorldSize < 6 {
		globalDomain[1] = fft.NewDomain(8 * mpi.WorldSize)
	} else {
		globalDomain[1] = fft.NewDomain(4 * mpi.WorldSize)
	}

	one := fr.One()

	var pk ProvingKey
	var vk VerifyingKey

	// The verifying key shares data with the proving key
	pk.Vk = &vk

	nbConstraints := len(spr.Constraints)

	// fft domains
	sizeSystem := uint64(nbConstraints + spr.NbPublicVariables) // spr.NbPublicVariables is for the placeholder constraints
	pk.Domain[0] = *fft.NewDomain(sizeSystem)
	pk.Vk.CosetShift.Set(&pk.Domain[0].FrMultiplicativeGen)

	var t, s *big.Int
	var err error
	if mpi.SelfRank == 0 {
		for {
			t, err = rand.Int(rand.Reader, spr.CurveID().ScalarField())
			if err != nil {
				return nil, nil, err
			}
			var ele fr.Element
			ele.SetBigInt(t)
			if !ele.Exp(ele, big.NewInt(int64(globalDomain[0].Cardinality))).Equal(&one) {
				break
			}
		}
		for {
			s, err = rand.Int(rand.Reader, spr.CurveID().ScalarField())
			if err != nil {
				return nil, nil, err
			}
			var ele fr.Element
			ele.SetBigInt(s)
			if !ele.Exp(ele, big.NewInt(int64(pk.Domain[0].Cardinality))).Equal(&one) {
				break
			}
		}
		// send t and s to all other processes
		tByteLen := (t.BitLen() + 7) / 8
		sByteLen := (s.BitLen() + 7) / 8
		for i := uint64(1); i < mpi.WorldSize; i++ {
			if err := mpi.SendBytes([]byte{byte(tByteLen)}, i); err != nil {
				return nil, nil, err
			}
			if err := mpi.SendBytes(t.Bytes(), i); err != nil {
				return nil, nil, err
			}
			if err := mpi.SendBytes([]byte{byte(sByteLen)}, i); err != nil {
				return nil, nil, err
			}
			if err := mpi.SendBytes(s.Bytes(), i); err != nil {
				return nil, nil, err
			}
		}
		globalSRS, err = kzg.NewSRS(globalDomain[0].Cardinality, t)
		if err != nil {
			return nil, nil, err
		}
	} else {
		tByteLen, err := mpi.ReceiveBytes(1, 0)
		if err != nil {
			return nil, nil, err
		}
		tbytes, err := mpi.ReceiveBytes(uint64(tByteLen[0]), 0)
		if err != nil {
			return nil, nil, err
		}
		t = new(big.Int).SetBytes(tbytes)
		sByteLen, err := mpi.ReceiveBytes(1, 0)
		if err != nil {
			return nil, nil, err
		}
		sbytes, err := mpi.ReceiveBytes(uint64(sByteLen[0]), 0)
		if err != nil {
			return nil, nil, err
		}
		s = new(big.Int).SetBytes(sbytes)
	}
	vk.KZGSRS = globalSRS

	// h, the quotient polynomial is of degree 3(n+1)+2, so it's in a 3(n+2) dim vector space,
	// the domain is the next power of 2 superior to 3(n+2). 4*domainNum is enough in all cases
	// except when n<6.
	if sizeSystem < 6 {
		pk.Domain[1] = *fft.NewDomain(8 * sizeSystem)
	} else {
		pk.Domain[1] = *fft.NewDomain(4 * sizeSystem)
	}

	vk.SizeY = globalDomain[0].Cardinality
	vk.SizeYInv.SetUint64(vk.SizeY).Inverse(&vk.SizeYInv)
	vk.SizeX = pk.Domain[0].Cardinality
	vk.SizeXInv.SetUint64(vk.SizeX).Inverse(&vk.SizeXInv)
	vk.Generator.Set(&pk.Domain[0].Generator)
	vk.NbPublicVariables = uint64(spr.NbPublicVariables)

	dkzgSRS, err := dkzg.NewSRS(vk.SizeX+3, []*big.Int{t, s}, &globalDomain[0].Generator)
	if err != nil {
		return nil, nil, err
	}
	if err := pk.InitKZG(dkzgSRS); err != nil {
		return nil, nil, err
	}

	// public polynomials corresponding to constraints: [ placholders | constraints | assertions ]
	pk.Ql = make([]fr.Element, pk.Domain[0].Cardinality)
	pk.Qr = make([]fr.Element, pk.Domain[0].Cardinality)
	pk.Qm = make([]fr.Element, pk.Domain[0].Cardinality)
	pk.Qo = make([]fr.Element, pk.Domain[0].Cardinality)
	pk.Qk = make([]fr.Element, pk.Domain[0].Cardinality)

	for i := 0; i < spr.NbPublicVariables; i++ { // placeholders (-PUB_INPUT_i + qk_i = 0) TODO should return error is size is inconsistant
		pk.Ql[i].SetOne().Neg(&pk.Ql[i])
		pk.Qr[i].SetZero()
		pk.Qm[i].SetZero()
		pk.Qo[i].SetZero()
		pk.Qk[i].Set(&publicWitness[i])
	}
	offset := spr.NbPublicVariables
	for i := 0; i < nbConstraints; i++ { // constraints

		pk.Ql[offset+i].Set(&spr.Coefficients[spr.Constraints[i].L.CoeffID()])
		pk.Qr[offset+i].Set(&spr.Coefficients[spr.Constraints[i].R.CoeffID()])
		pk.Qm[offset+i].Set(&spr.Coefficients[spr.Constraints[i].M[0].CoeffID()]).
			Mul(&pk.Qm[offset+i], &spr.Coefficients[spr.Constraints[i].M[1].CoeffID()])
		pk.Qo[offset+i].Set(&spr.Coefficients[spr.Constraints[i].O.CoeffID()])
		pk.Qk[offset+i].Set(&spr.Coefficients[spr.Constraints[i].K])
	}

	pk.Domain[0].FFTInverse(pk.Ql, fft.DIF)
	pk.Domain[0].FFTInverse(pk.Qr, fft.DIF)
	pk.Domain[0].FFTInverse(pk.Qm, fft.DIF)
	pk.Domain[0].FFTInverse(pk.Qo, fft.DIF)
	pk.Domain[0].FFTInverse(pk.Qk, fft.DIF)
	fft.BitReverse(pk.Ql)
	fft.BitReverse(pk.Qr)
	fft.BitReverse(pk.Qm)
	fft.BitReverse(pk.Qo)
	fft.BitReverse(pk.Qk)

	// build permutation. Note: at this stage, the permutation takes in account the placeholders
	buildPermutation(spr, &pk)

	// set s1, s2, s3
	ccomputePermutationPolynomials(&pk)

	// Commit to the polynomials to set up the verifying key
	if vk.Ql, err = dkzg.Commit(pk.Ql, vk.DKZGSRS); err != nil {
		return nil, nil, err
	}
	if vk.Qr, err = dkzg.Commit(pk.Qr, vk.DKZGSRS); err != nil {
		return nil, nil, err
	}
	if vk.Qm, err = dkzg.Commit(pk.Qm, vk.DKZGSRS); err != nil {
		return nil, nil, err
	}
	if vk.Qo, err = dkzg.Commit(pk.Qo, vk.DKZGSRS); err != nil {
		return nil, nil, err
	}
	if vk.Qk, err = dkzg.Commit(pk.Qk, vk.DKZGSRS); err != nil {
		return nil, nil, err
	}
	if vk.S[0], err = dkzg.Commit(pk.S1Canonical, vk.DKZGSRS); err != nil {
		return nil, nil, err
	}
	if vk.S[1], err = dkzg.Commit(pk.S2Canonical, vk.DKZGSRS); err != nil {
		return nil, nil, err
	}
	if vk.S[2], err = dkzg.Commit(pk.S3Canonical, vk.DKZGSRS); err != nil {
		return nil, nil, err
	}

	return &pk, &vk, nil

}
```

</details>

<details><summary><b> prove.go </b></summary>

```go
// Proof denotes a Piano proof generated from M parties each with N rows.
type Proof struct {

    // Commitments to the solution vectors
    LRO [3]dkzg.Digest

    // Commitment to Z, the permutation polynomial
    Z dkzg.Digest

    // Commitments to Hx1, Hx2, Hx3 such that
    // Hx = Hx1 + (X**N) * Hx2 + (X**(2N)) * Hx3 and
    // commitments to Hy1, Hy2, Hy3 such that
    // Hy = Hy1 + (Y**M) * Hy2 + (Y**(2M)) * Hy3
    Hx [3]dkzg.Digest
    Hy [3]kzg.Digest

    // Batch partially opening proof of
    // foldedHx(Y, X) = Hx1(Y, X) + alpha*Hx2(Y, X) + (alpha**2)*Hx3(Y, X),
    // L(Y, X), R(Y, X), O(Y, X), Ql(Y, X), Qr(Y, X), Qm(Y, X), Qo(Y, X),
    // Qk(Y, X), S1(Y, X), S2(Y, X), S3(Y, X),
    // Z(Y, X) on X = alpha
    PartialBatchedProof dkzg.BatchOpeningProof

    // Opening partially proof of Z(Y, X) on X = omegaX*alpha
    PartialZShiftedProof dkzg.OpeningProof

    // Batch opening proof of FoldedHx(Y, alpha), L(Y, alpha), R(Y, alpha), O(Y, alpha),
    // Ql(Y, alpha), Qr(Y, alpha), Qm(Y, alpha), Qo(Y, alpha), Qk(Y, alpha),
    // S1(Y, alpha), S2(Y, alpha), S3(Y, alpha), Z(Y, alpha), z(Y, mu*alpha),
    // FoldedHy(Y) on Y = beta
    BatchedProof kzg.BatchOpeningProof
}

// Prove from the public data
func Prove(spr *cs.SparseR1CS, pk *ProvingKey, fullWitness bn254witness.Witness, opt backend.ProverConfig) (*Proof, error) {
    fmt.Println("Prover started")
    log := logger.Logger().With().Str("curve", spr.CurveID().String()).Int("nbConstraints", len(spr.Constraints)).Str("backend", "piano").Logger()
    start := time.Now()
    // pick a hash function that will be used to derive the challenges
    hFunc := sha256.New()

    // create a transcript manager to apply Fiat Shamir
    fs := fiatshamir.NewTranscript(hFunc, "gamma", "eta", "lambda", "alpha", "beta")

    // result
    proof := &Proof{}

    // compute the constraint system solution
    var solution []fr.Element
    var err error
    if solution, err = spr.Solve(fullWitness, opt); err != nil {
        if !opt.Force {
            return nil, err
        } else {
            // we need to fill solution with random values
            var r fr.Element
            _, _ = r.SetRandom()
            for i := spr.NbPublicVariables + spr.NbSecretVariables; i < len(solution); i++ {
                solution[i] = r
                r.Double(&r)
            }
        }
    }

    fmt.Println("Solution computed")

    // query L, R, O in Lagrange basis, not blinded
    lSmallX, rSmallX, oSmallX := evaluateLROSmallDomainX(spr, pk, solution)

    // save lL, lR, lO, and make a copy of them in
    // canonical basis note that we allocate more capacity to reuse for blinded
    // polynomials
    lCanonicalX, rCanonicalX, oCanonicalX := computeLROCanonicalX(
        lSmallX,
        rSmallX,
        oSmallX,
        &pk.Domain[0],
    )
    if err != nil {
        return nil, err
    }

    // compute kzg commitments of bcL, bcR and bcO
    if err := commitToLRO(lCanonicalX, rCanonicalX, oCanonicalX, proof, pk.Vk.DKZGSRS); err != nil {
        return nil, err
    }

    // The first challenge is derived using the public data: the commitments to the permutation,
    // the coefficients of the circuit, and the public inputs.
    // derive gamma from the Comm(cL), Comm(cR), Comm(cO)
    if err := bindPublicData(&fs, "gamma", *pk.Vk, fullWitness[:spr.NbPublicVariables]); err != nil {
        return nil, err
    }
    gamma, err := deriveRandomness(&fs, "gamma", false, &proof.LRO[0], &proof.LRO[1], &proof.LRO[2])
    if err != nil {
        return nil, err
    }

    // Fiat Shamir this
    eta, err := deriveRandomness(&fs, "eta", false)
    if err != nil {
        return nil, err
    }

    // compute Z, the permutation accumulator polynomial, in canonical basis
    // lL, lR, lO are NOT blinded
    zCanonicalX, err := computeZCanonicalX(
        lSmallX,
        rSmallX,
        oSmallX,
        pk, eta, gamma,
    )
    if err != nil {
        return nil, err
    }

    // commit to z
    // note that we explicitly double the number of tasks for the multi exp
    // in dkzg.Commit
    // this may add additional arithmetic operations, but with smaller tasks
    // we ensure that this commitment is well parallelized, without having a
    // "unbalanced task" making the rest of the code wait too long
    if proof.Z, err = dkzg.Commit(zCanonicalX, pk.Vk.DKZGSRS, runtime.NumCPU()*2); err != nil {
        return nil, err
    }

    // derive lambda from the Comm(L), Comm(R), Comm(O), Com(Z)
    lambda, err := deriveRandomness(&fs, "lambda", false, &proof.Z)
    if err != nil {
        return nil, err
    }

    hx1, hx2, hx3 := computeQuotientCanonicalX(pk, lCanonicalX, rCanonicalX, oCanonicalX, zCanonicalX, eta, gamma, lambda)

    // print vector of hx1, hx2, hx3

    // compute kzg commitments of Hx1, Hx2 and Hx3
    if err := commitToQuotientX(hx1, hx2, hx3, proof, pk.Vk.DKZGSRS); err != nil {
        return nil, err
    }

    // derive alpha
    alpha, err := deriveRandomness(&fs, "alpha", false, &proof.Hx[0], &proof.Hx[1], &proof.Hx[2])
    if err != nil {
        return nil, err
    }

    // open Z at mu*alpha
    var alphaShifted fr.Element
    alphaShifted.Mul(&alpha, &pk.Vk.Generator)
    var zShiftedAlpha []fr.Element
    proof.PartialZShiftedProof, zShiftedAlpha, err = dkzg.Open(
        zCanonicalX,
        alphaShifted,
        pk.Vk.DKZGSRS,
    )
    if err != nil {
        return nil, err
    }

    // foldedHDigest = Comm(Hx1) + (alpha**(N))*Comm(Hx2) + (alpha**(2(N)))*Comm(Hx3)
    var bAlphaPowerN, bSize big.Int
    bSize.SetUint64(pk.Domain[0].Cardinality)
    var alphaPowerN fr.Element
    alphaPowerN.Exp(alpha, &bSize)
    alphaPowerN.ToBigIntRegular(&bAlphaPowerN)
    foldedHxDigest := proof.Hx[2]
    foldedHxDigest.ScalarMultiplication(&foldedHxDigest, &bAlphaPowerN)
    foldedHxDigest.Add(&foldedHxDigest, &proof.Hx[1])
    foldedHxDigest.ScalarMultiplication(&foldedHxDigest, &bAlphaPowerN)
    foldedHxDigest.Add(&foldedHxDigest, &proof.Hx[0])

    // foldedHx = Hx1 + (alpha**(N))*Hx2 + (alpha**(2(N)))*Hx3
    foldedHx := hx3
    utils.Parallelize(len(foldedHx), func(start, end int) {
        for i := start; i < end; i++ {
            foldedHx[i].Mul(&foldedHx[i], &alphaPowerN)
            foldedHx[i].Add(&foldedHx[i], &hx2[i])
            foldedHx[i].Mul(&foldedHx[i], &alphaPowerN)
            foldedHx[i].Add(&foldedHx[i], &hx1[i])
        }
    })

    dkzgOpeningPolys := [][]fr.Element{
        foldedHx,
        lCanonicalX,
        rCanonicalX,
        oCanonicalX,
        pk.Ql,
        pk.Qr,
        pk.Qm,
        pk.Qo,
        pk.Qk,
        pk.S1Canonical,
        pk.S2Canonical,
        pk.S3Canonical,
        zCanonicalX,
    }
    dkzgDigests := []dkzg.Digest{
        foldedHxDigest,
        proof.LRO[0],
        proof.LRO[1],
        proof.LRO[2],
        pk.Vk.Ql,
        pk.Vk.Qr,
        pk.Vk.Qm,
        pk.Vk.Qo,
        pk.Vk.Qk,
        pk.Vk.S[0],
        pk.Vk.S[1],
        pk.Vk.S[2],
        proof.Z,
    }

    // Batch open the first list of polynomials
    var evalsXOnAlpha [][]fr.Element
    proof.PartialBatchedProof, evalsXOnAlpha, err = dkzg.BatchOpenSinglePoint(
        dkzgOpeningPolys,
        dkzgDigests,
        alpha,
        hFunc,
        pk.Vk.DKZGSRS,
    )

    if err != nil {
        return nil, err
    }

    if mpi.SelfRank != 0 {
        log.Debug().Dur("took", time.Since(start)).Msg("prover done")
        if err != nil {
            return nil, err
        }

        return proof, nil
    }

    // DBG check whether constraints are satisfied
    if err := checkConstraintX(
        pk,
        evalsXOnAlpha,
        zShiftedAlpha,
        gamma,
        eta,
        lambda,
        alpha,
    ); err != nil {
        return nil, err
    }

    polysCanonicalY := append(evalsXOnAlpha, zShiftedAlpha)
    for i := 0; i < len(polysCanonicalY); i++ {
        globalDomain[0].FFTInverse(polysCanonicalY[i], fft.DIF)
        fft.BitReverse(polysCanonicalY[i])
    }

    // compute Hy in canonical form
    hyCanonical1, hyCanonical2, hyCanonical3 := computeQuotientCanonicalY(pk,
        polysCanonicalY,
        eta,
        gamma,
        lambda,
        alpha,
    )

    // compute kzg commitments of Hy1, Hy2 and Hy3
    if err := commitToQuotientOnY(hyCanonical1, hyCanonical2, hyCanonical3, proof, globalSRS); err != nil {
        return nil, err
    }
    // derive beta
    ts := []*curve.G1Affine{
        &proof.PartialBatchedProof.H,
    }
    for _, digest := range proof.PartialBatchedProof.ClaimedDigests {
        ts = append(ts, &digest)
    }
    for _, digest := range proof.Hy {
        ts = append(ts, &digest)
    }
    beta, err := deriveRandomness(&fs, "beta", true, ts...)
    if err != nil {
        return nil, err
    }

    // foldedHy = Hy1 + (beta**M)*Hy2 + (beta**(2M))*Hy3
    var bBetaPowerM big.Int
    bSize.SetUint64(globalDomain[0].Cardinality)
    var betaPowerM fr.Element
    betaPowerM.Exp(beta, &bSize)
    betaPowerM.ToBigIntRegular(&bBetaPowerM)
    foldedHyDigest := proof.Hy[2]
    foldedHyDigest.ScalarMultiplication(&foldedHyDigest, &bBetaPowerM)
    foldedHyDigest.Add(&foldedHyDigest, &proof.Hy[1])
    foldedHyDigest.ScalarMultiplication(&foldedHyDigest, &bBetaPowerM)
    foldedHyDigest.Add(&foldedHyDigest, &proof.Hy[0])
    foldedHy := hyCanonical3
    utils.Parallelize(len(foldedHy), func(start, end int) {
        for i := start; i < end; i++ {
            foldedHy[i].Mul(&foldedHy[i], &betaPowerM)
            foldedHy[i].Add(&foldedHy[i], &hyCanonical2[i])
            foldedHy[i].Mul(&foldedHy[i], &betaPowerM)
            foldedHy[i].Add(&foldedHy[i], &hyCanonical1[i])
        }
    })

    polysCanonicalY = append(polysCanonicalY, foldedHy)

    // evalsOnBeta := evalPolynomialsAtPoint(openingPolysCanonicalY, beta)
    // DBG check whether constraints are satisfied
    // if err := checkConstraintY(pk.Vk,
    // 	evalsOnBeta,
    // 	gamma,
    // 	eta,
    // 	lambda,
    // 	alpha,
    // 	beta,
    // ); err != nil {
    // 	return nil, err
    // }

    var digestsY []curve.G1Affine
    digestsY = append(digestsY, proof.PartialBatchedProof.ClaimedDigests...)
    digestsY = append(digestsY, proof.PartialZShiftedProof.ClaimedDigest, foldedHyDigest)
    proof.BatchedProof, err = kzg.BatchOpenSinglePoint(
        polysCanonicalY,
        digestsY,
        beta,
        hFunc,
        globalSRS,
    )
    if err != nil {
        return nil, err
    }
    return proof, nil
}

func computeZCanonicalX(...) ([]fr.Element, error)
func computeQuotientCanonicalX(...) ([]fr.Element, []fr.Element, []fr.Element)
func computeQuotientCanonicalY(...) ([]fr.Element, []fr.Element, []fr.Element)
```

</details>

<details><summary><b> verify.go </b></summary>

```go
  func Verify(proof *Proof, vk *VerifyingKey, publicWitness bn254witness.Witness) error {
      log := logger.Logger().With().Str("curve", "bn254").Str("backend", "piano").Logger()
      start := time.Now()

      // pick a hash function to derive the challenge (the same as in the prover)
      hFunc := sha256.New()

      // transcript to derive the challenge
      fs := fiatshamir.NewTranscript(hFunc, "gamma", "eta", "lambda", "alpha", "beta")

      // The first challenge is derived using the public data: the commitments to the permutation,
      // the coefficients of the circuit, and the public inputs.
      // derive gamma from the Comm(blinded cl), Comm(blinded cr), Comm(blinded co)
      if err := bindPublicData(&fs, "gamma", *vk, publicWitness); err != nil {
          return err
      }
      gamma, err := deriveRandomness(&fs, "gamma", true, &proof.LRO[0], &proof.LRO[1], &proof.LRO[2])
      if err != nil {
          return err
      }
      // derive eta from Comm(l), Comm(r), Comm(o)
      eta, err := deriveRandomness(&fs, "eta", true)
      if err != nil {
          return err
      }

      // derive lambda from Comm(l), Comm(r), Comm(o), Com(Z)
      lambda, err := deriveRandomness(&fs, "lambda", true, &proof.Z)
      if err != nil {
          return err
      }

      // derive alpha, the point of evaluation
      alpha, err := deriveRandomness(&fs, "alpha", true, &proof.Hx[0], &proof.Hx[1], &proof.Hx[2])
      if err != nil {
          return err
      }

      // evaluation of Z=Xⁿ⁻¹ at α
      var alphaPowerN, zalpha fr.Element
      var bExpo big.Int
      one := fr.One()
      bExpo.SetUint64(vk.SizeX)
      alphaPowerN.Exp(alpha, &bExpo)
      zalpha.Sub(&alphaPowerN, &one)

      // compute the folded commitment to H: Comm(h₁) + αᵐ*Comm(h₂) + α²⁽ᵐ⁾*Comm(h₃)
      var alphaNBigInt big.Int
      alphaPowerN.ToBigIntRegular(&alphaNBigInt)
      foldedHxDigest := proof.Hx[2]
      foldedHxDigest.ScalarMultiplication(&foldedHxDigest, &alphaNBigInt)
      foldedHxDigest.Add(&foldedHxDigest, &proof.Hx[1])
      foldedHxDigest.ScalarMultiplication(&foldedHxDigest, &alphaNBigInt)
      foldedHxDigest.Add(&foldedHxDigest, &proof.Hx[0])

      foldedPartialProof, foldedPartialDigest, err := dkzg.FoldProof(
          []dkzg.Digest{
              foldedHxDigest,
              proof.LRO[0],
              proof.LRO[1],
              proof.LRO[2],
              vk.Ql,
              vk.Qr,
              vk.Qm,
              vk.Qo,
              vk.Qk,
              vk.S[0],
              vk.S[1],
              vk.S[2],
              proof.Z,
          },
          &proof.PartialBatchedProof,
          alpha,
          hFunc)

      if err != nil {
          return fmt.Errorf("failed to fold proof on X = alpha: %v", err)
      }
      // Batch verify
      var shiftedalpha fr.Element
      shiftedalpha.Mul(&alpha, &vk.Generator)
      err = dkzg.BatchVerifyMultiPoints(
          []dkzg.Digest{
              foldedPartialDigest,
              proof.Z,
          },
          []dkzg.OpeningProof{
              foldedPartialProof,
              proof.PartialZShiftedProof,
          },
          []fr.Element{
              alpha,
              shiftedalpha,
          },
          vk.DKZGSRS,
      )
      if err != nil {
          return fmt.Errorf("failed to batch verify on X = alpha: %v", err)
      }

      // derive beta
      ts := []*curve.G1Affine{
          &proof.PartialBatchedProof.H,
      }
      for _, digest := range proof.PartialBatchedProof.ClaimedDigests {
          ts = append(ts, &digest)
      }
      for _, digest := range proof.Hy {
          ts = append(ts, &digest)
      }
      beta, err := deriveRandomness(&fs, "beta", true, ts...)
      if err != nil {
          return err
      }

      if err := checkConstraintY(vk, proof.BatchedProof.ClaimedValues, gamma, eta, lambda, alpha, beta); err != nil {
          return err
      }
      // foldedHy = Hy1 + (beta**M)*Hy2 + (beta**(2M))*Hy3
      var bBetaPowerM, bSize big.Int
      bSize.SetUint64(vk.SizeY)
      var betaPowerM fr.Element
      betaPowerM.Exp(beta, &bSize)
      betaPowerM.ToBigIntRegular(&bBetaPowerM)
      foldedHyDigest := proof.Hy[2]                                      // Hy3
      foldedHyDigest.ScalarMultiplication(&foldedHyDigest, &bBetaPowerM) // (beta**M)*Hy3
      foldedHyDigest.Add(&foldedHyDigest, &proof.Hy[1])                  // (beta**M)*Hy3 + Hy2
      foldedHyDigest.ScalarMultiplication(&foldedHyDigest, &bBetaPowerM) // (beta**(2M))*Hy3 + (beta**M)*Hy2
      foldedHyDigest.Add(&foldedHyDigest, &proof.Hy[0])                  // (beta**(2M))*Hy3 + (beta**M)*Hy2 + Hy1

      if err := kzg.BatchVerifySinglePoint(
          append(proof.PartialBatchedProof.ClaimedDigests,
              proof.PartialZShiftedProof.ClaimedDigest,
              foldedHyDigest,
          ),
          &proof.BatchedProof,
          beta, // not consistent with the prover
          hFunc,
          vk.KZGSRS,
      ); err != nil {
          return err
      }

      log.Debug().Dur("took", time.Since(start)).Msg("verifier done")

      return err
  }
```

</details>

</details>

<details><summary><b> std/hash/mimc/mimc.go </b></summary>

```go
// MiMC contains the params of the Mimc hash func and the curves on which it is implemented
type MiMC struct {
    params []big.Int           // slice containing constants for the encryption rounds
    id     ecc.ID              // id needed to know which encryption function to use
    h      frontend.Variable   // current vector in the Miyaguchi–Preneel scheme
    data   []frontend.Variable // state storage. data is updated when Write() is called. Sum sums the data.
    api    frontend.API        // underlying constraint system
}

// NewMiMC returns a MiMC instance, than can be used in a gnark circuit
func NewMiMC(api frontend.API) (MiMC, error) {
    if constructor, ok := newMimc[api.Compiler().Curve()]; ok {
        return constructor(api), nil
    }
    return MiMC{}, errors.New("unknown curve id")
}
```

</details>

<details><summary><b> frontend/api.go </b></summary>

```go
// API represents the available functions to circuit developers
type API interface {
    // ---------------------------------------------------------------------------------------------
    // Arithmetic

    // Add returns res = i1+i2+...in
    Add(i1, i2 Variable, in ...Variable) Variable

    // Neg returns -i
    Neg(i1 Variable) Variable

    // Sub returns res = i1 - i2 - ...in
    Sub(i1, i2 Variable, in ...Variable) Variable

    // Mul returns res = i1 * i2 * ... in
    Mul(i1, i2 Variable, in ...Variable) Variable

    // DivUnchecked returns i1 / i2 . if i1 == i2 == 0, returns 0
    DivUnchecked(i1, i2 Variable) Variable

    // Div returns i1 / i2
    Div(i1, i2 Variable) Variable

    // Inverse returns res = 1 / i1
    Inverse(i1 Variable) Variable
```

</details>

<details><summary><b> frontend/cs/r1cs/api.go </b></summary>

```go
  // Add returns res = i1+i2+...in
  func (system *r1cs) Add(i1, i2 frontend.Variable, in ...frontend.Variable) frontend.Variable {

      // extract frontend.Variables from input
      vars, s := system.toVariables(append([]frontend.Variable{i1, i2}, in...)...)

      // allocate resulting frontend.Variable
      res := make(compiled.LinearExpression, 0, s)

      for _, v := range vars {
          l := v.Clone()
          res = append(res, l...)
      }

      res = system.reduce(res)

      return res
  }

  // Mul returns res = i1 * i2 * ... in
  func (system *r1cs) Mul(i1, i2 frontend.Variable, in ...frontend.Variable) frontend.Variable {
      vars, _ := system.toVariables(append([]frontend.Variable{i1, i2}, in...)...)

      mul := func(v1, v2 compiled.LinearExpression) compiled.LinearExpression {

          n1, v1Constant := system.ConstantValue(v1)
          n2, v2Constant := system.ConstantValue(v2)

          // v1 and v2 are both unknown, this is the only case we add a constraint
          if !v1Constant && !v2Constant {
              res := system.newInternalVariable()
              system.Constraints = append(system.Constraints, newR1C(v1, v2, res))
              return res
          }

          // v1 and v2 are constants, we multiply big.Int values and return resulting constant
          if v1Constant && v2Constant {
              n1.Mul(n1, n2).Mod(n1, system.CurveID.ScalarField())
              return system.toVariable(n1).(compiled.LinearExpression)
          }

          // ensure v2 is the constant
          if v1Constant {
              v1, v2 = v2, v1
          }

          return system.mulConstant(v1, v2)
      }

      res := mul(vars[0], vars[1])

      for i := 2; i < len(vars); i++ {
          res = mul(res, vars[i])
      }

      return res
  }

  // IsZero returns 1 if i1 is zero, 0 otherwise
  func (system *r1cs) IsZero(i1 frontend.Variable) frontend.Variable {
      vars, _ := system.toVariables(i1)
      a := vars[0]
      if c, ok := system.ConstantValue(a); ok {
          if c.IsUint64() && c.Uint64() == 0 {
              return system.toVariable(1)
          }
          return system.toVariable(0)
      }

      debug := system.AddDebugInfo("isZero", a)

      //m * (1 - m) = 0       // constrain m to be 0 or 1
      // a * m = 0            // constrain m to be 0 if a != 0
      // _ = inverse(m + a) 	// constrain m to be 1 if a == 0

      // m is computed by the solver such that m = 1 - a^(modulus - 1)
      res, err := system.NewHint(hint.IsZero, 1, a)
      if err != nil {
          // the function errs only if the number of inputs is invalid.
          panic(err)
      }
      m := res[0]
      system.addConstraint(newR1C(a, m, system.toVariable(0)), debug)

      system.AssertIsBoolean(m)
      ma := system.Add(m, a)
      _ = system.Inverse(ma)
      return m
  }

```

</details>

<details><summary><b> frontend/compile.go </b></summary>

```go
  // Compile will generate a ConstraintSystem from the given circuit
  //
  // 1. it will first allocate the user inputs (see type Tag for more info)
  // example:
  // 		type MyCircuit struct {
  // 			Y frontend.Variable `gnark:"exponent,public"`
  // 		}
  // in that case, Compile() will allocate one public variable with id "exponent"
  //
  // 2. it then calls circuit.Define(curveID, R1CS) to build the internal constraint system
  // from the declarative code
  //
  // 3. finally, it converts that to a ConstraintSystem.
  // 		if zkpID == backend.GROTH16	→ R1CS
  //		if zkpID == backend.PLONK 	→ SparseR1CS
  //
  // initialCapacity is an optional parameter that reserves memory in slices
  // it should be set to the estimated number of constraints in the circuit, if known.
  func Compile(curveID ecc.ID, newBuilder NewBuilder, circuit Circuit, opts ...CompileOption) (CompiledConstraintSystem, error) {
      log := logger.Logger()
      log.Info().Str("curve", curveID.String()).Msg("compiling circuit")
      // parse options
      opt := CompileConfig{}
      for _, o := range opts {
          if err := o(&opt); err != nil {
              log.Err(err).Msg("applying compile option")
              return nil, fmt.Errorf("apply option: %w", err)
          }
      }

      // instantiate new builder
      builder, err := newBuilder(curveID, opt)
      if err != nil {
          log.Err(err).Msg("instantiating builder")
          return nil, fmt.Errorf("new compiler: %w", err)
      }

      // parse the circuit builds a schema of the circuit
      // and call circuit.Define() method to initialize a list of constraints in the compiler
      if err = parseCircuit(builder, circuit); err != nil {
          log.Err(err).Msg("parsing circuit")
          return nil, fmt.Errorf("parse circuit: %w", err)

      }

      // compile the circuit into its final form
      return builder.Compile()

}

type CompileConfig struct {
    Capacity int
    IgnoreUnconstrainedInputs bool
}

```

</details>

<details><summary><b> examples/piano/main.go </b></summary>

```go
// In this example we show how to use PLONK with KZG commitments. The circuit that is
// showed here is the same as in ../exponentiate.

// Circuit y == x**e
// only the bitSize least significant bits of e are used
type Circuit struct {
    // tagging a variable is optional
    // default uses variable name and secret visibility.
    X frontend.Variable `gnark:",public"`
    Y frontend.Variable `gnark:",public"`

    E frontend.Variable
}

// Define declares the circuit's constraints
// y == x**e
func (circuit *Circuit) Define(api frontend.API) error {

    // number of bits of exponent
    const bitSize = 4000

    // specify constraints
    output := frontend.Variable(1)
    bits := api.ToBinary(circuit.E, bitSize)

    for i := 0; i < len(bits); i++ {
        // api.Println(fmt.Sprintf("e[%d]", i), bits[i]) // we may print a variable for testing and / or debugging purposes

        if i != 0 {
            output = api.Mul(output, output)
        }
        multiply := api.Mul(output, circuit.X)
        output = api.Select(bits[len(bits)-1-i], multiply, output)

    }

    api.AssertIsEqual(circuit.Y, output)

    return nil
}

func main() {

    var circuit Circuit

    // // building the circuit...
    ccs, err := frontend.Compile(ecc.BN254, scs.NewBuilder, &circuit)
    if err != nil {
        fmt.Println("circuit compilation error")
    }

    // Correct data: the proof passes
    {
        // Witnesses instantiation. Witness is known only by the prover,
        // while public w is a public data known by the verifier.
        var w Circuit
        w.X = 12
        w.E = 2
        //  + mpi.SelfRank
        tmp := 144
        // for i := 0; i < int(mpi.SelfRank); i++ {
        // 	tmp *= 12
        // }
        w.Y = tmp

        witnessFull, err := frontend.NewWitness(&w, ecc.BN254)
        if err != nil {
            log.Fatal(err)
        }

        witnessPublic, err := frontend.NewWitness(&w, ecc.BN254, frontend.PublicOnly())
        if err != nil {
            log.Fatal(err)
        }

        // public data consists the polynomials describing the constants involved
        // in the constraints, the polynomial describing the permutation ("grand
        // product argument"), and the FFT domains.
        pk, vk, err := piano.Setup(ccs, witnessPublic)
        if err != nil {
            log.Fatal(err)
        }

        proof, err := piano.Prove(ccs, pk, witnessFull)
        if err != nil {
            log.Fatal(err)
        }

        if mpi.SelfRank == 0 {
            err = piano.Verify(proof, vk, witnessPublic)
            if err != nil {
                log.Fatal(err)
            }
        }
    }

    fmt.Println("Done")
}
```

</details>

</details>

[Pianist](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/pianist-gnark-stable)
