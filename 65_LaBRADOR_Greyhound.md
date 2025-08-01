# ETAAcademy-ZKMeme: 65. LaBRADOR & Greyhound

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>65. LaBRADOR & Greyhound</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>LaBRADOR & Greyhound</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Lattice-Based Zero-Knowledge Proof Systems: MLE-PCS, LaBRADOR, and Greyhound

**Multilinear Polynomial Commitment Schemes (MLE-PCS)** are a class of efficient commitment mechanisms tailored for multilinear polynomials defined over Boolean hypercubes. They fall into **four main categories** based on the underlying commitment primitives—**KZG10**, **Merkle Tree**, **Pedersen**, and **Ajtai**—and employ **four principal proof techniques**: **quotient polynomial**, **inner product**, **sumcheck**, and **split-and-fold**. These designs enable proof generation with **$O(N)$** prover complexity, significantly improving over traditional schemes.

**LaBRADOR** is a post-quantum recursive zero-knowledge proof system built on the **Module-SIS** assumption. It employs a two-layer Ajtai commitment structure, **Johnson–Lindenstrauss (JL) projections**, norm **amortization and aggregation techniques**, and an efficient verification of **inner product constraints**. LaBRADOR also provides a reduction from **R1CS** (Rank-1 Constraint System) to its native constraint model, allowing it to support mixed binary and large-modulus constraints.

**Greyhound**, building upon LaBRADOR, proposes a **lattice-based polynomial commitment scheme**. It introduces a three-round protocol and a two-layer commitment structure that reduces polynomial evaluations to LaBRADOR-style inner product constraints. To enable compatibility with the LaBRADOR framework, it performs ring-switching between structured rings and finite fields. Greyhound ensures **zero-knowledge** via the **Module-LWE** assumption and forms a cohesive, post-quantum secure ecosystem for zero-knowledge proof systems.

---

## 1. Multilinear Polynomial Commitment Schemes (MLE-PCS): Foundations and Variants

Polynomial commitments are a foundational component in modern zero-knowledge proof systems. Among them, _Multilinear Polynomial Commitment Schemes_ (MLE-PCS) are specialized schemes designed for multilinear polynomials, which are polynomials where each variable appears with degree at most one. This makes them particularly suitable for protocols defined over Boolean hypercubes.

### What is a Multilinear Polynomial?

A **multilinear polynomial** over variables $X_0, X_1, \ldots, X_{n-1}$ is a polynomial where each monomial contains any variable at most once. For example, $f(X_0, X_1) = X_0X_1 + X_1$ is multilinear. These polynomials are naturally defined over the Boolean hypercube $\{0,1\}^n$, which contains $N = 2^n$ points. MLE-PCS focuses on committing to such polynomials efficiently.

There are two common representations of a multilinear polynomial $\tilde{f}$:

- **Coefficient form**:

  $\tilde{f}(X_0, X_1, \ldots, X_{n - 1}) = c_0 + c_1 X_0 + c_2 X_1 + \ldots + c_{2^n - 1} X_0 X_1 \cdots X_{n - 1}$

  where $c_i \in \mathbb{F}$ are the coefficients and $\text{monomial}_i$ is a multilinear monomial determined by the binary expansion of $i$.

- **Evaluation form**:

  $\tilde{f}(X_0, \ldots, X_{n-1}) = \sum_{i=0}^{2^n - 1} \tilde{f}(\text{bits}(i)) \cdot \tilde{eq}(\text{bits}(i), (X_0, \ldots, X_{n-1}))$

  where $\text{bits}(i)$ is the little-endian binary representation of $i$, and $\tilde{eq}$ is a multilinear Lagrange basis polynomial (used for interpolation).

### Why MLE-PCS?

MLE-PCS schemes exploit the structure of multilinear polynomials to achieve significantly improved performance. In particular, they can achieve **linear prover time complexity $O(N)$** compared to traditional polynomial commitment schemes that require $O(N \log N)$. This makes MLE-PCS attractive in settings that demand scalability, such as large-scale verifiable computation and SNARKs.

### Classification by Commitment Type

MLE-PCS schemes can be categorized based on the type of cryptographic commitment used under the hood. These underlying primitives determine critical properties like proof size, efficiency, quantum resistance, and whether a trusted setup is required.

| Commitment Type       | Algebraic Structure              | Example Protocols                                  |
| --------------------- | -------------------------------- | -------------------------------------------------- |
| **KZG10-based**       | Pairing-friendly elliptic curves | PST13, HyperKZG, Mercury, Samaritan                |
| **Merkle Tree-based** | Linear codes + hash functions    | Virgo, Basefold, Deepfold, Ligerito, FRI-Binius    |
| **Pedersen-based**    | Elliptic curves (no pairing)     | Hyrax, Σ-Check                                     |
| **Ajtai-based**       | Lattices                         | **Greyhound**, **Hyperwolf** (post-quantum SNARKs) |

Lattice-based schemes (e.g., Greyhound and Hyperwolf) rely on the hardness of lattice problems like the Short Integer Solution (SIS) or Learning With Errors (LWE), making them promising candidates for **post-quantum cryptography**. These schemes use tools like **LLL** (Lenstra–Lenstra–Lovász) or **BKZ** (Block Korkine–Zolotarev) algorithms to analyze or construct lattice-based commitments.

### Techniques for MLE-PCS Construction

Although all MLE-PCS constructions originate from univariate polynomial commitment schemes, their core difference lies in how they encode and verify evaluations. There are four major strategies:

| Approach           | Main Advantage                         | Trade-Offs                | Best Use Cases        |
| ------------------ | -------------------------------------- | ------------------------- | --------------------- |
| **Quotienting**    | Simple and compatible with various PCS | Requires structured SRS   | General purpose       |
| **Inner Product**  | High flexibility, batch-friendly       | Higher verifier cost      | Batched evaluations   |
| **Sumcheck**       | Highly efficient for verifier          | More complex to implement | Large computations    |
| **Split-and-Fold** | Conceptually simple (like FRI)         | More recursive rounds     | Recursive-friendly ZK |

#### **Quotienting**

This method rewrites the evaluation check $f(u) = v$ into a divisibility statement: $\tilde{f}(X) - v = \sum_{i=0}^{n-1} q_i(X) \cdot (X_i - u_i)$

The prover commits to the quotient polynomials $q_i$, and the verifier checks this identity at a random point. It's simple and generic but usually requires a **structured reference string (SRS)**.

#### **Inner Product**

Here, the multilinear polynomial is encoded as a vector, and the evaluation reduces to: $\langle \vec{f}, \bigotimes_{j=0}^{n-1}(1, u_j) \rangle = v$

This enables batch evaluation proofs, though it may be costly for the verifier due to the inner product computation.

#### **Sumcheck**

The proof reduces to checking:

$\sum_{\vec{b} \in \{0,1\}^n} \tilde{f}(\vec{b}) \cdot \tilde{eq}(\vec{b}, \vec{u}) = v$

This method has excellent verifier efficiency and is widely used in proof systems like GKR, but it requires more intricate implementation and arithmetic circuits.

#### **Split-and-Fold**

Inspired by FRI, this approach recursively folds the polynomial until it reduces to a constant:

$\tilde{f}(X_0, \ldots, X_{n-1}) = \langle \vec{f}, \bigotimes_{i=0}^{n-1}(1, X_i) \rangle$

The folding follows a recurrence:

$\langle \vec{f}, \bigotimes_{j=0}^{n-1}(1, u_j) \rangle = \langle \vec{f}_{\text{even}}, \bigotimes_{j=1}^{n-1}(1, u_j) \rangle + u_0 \cdot \langle \vec{f}_{\text{odd}}, \bigotimes_{j=1}^{n-1}(1, u_j) \rangle$

This method is particularly suitable for recursive proof systems and STARK-like designs.

Multilinear Polynomial Commitment Schemes (MLE-PCS) provide a powerful, structured approach for committing to and proving properties of multilinear polynomials. By tailoring the commitment method and proof strategy—whether quotienting, inner-product, sumcheck, or folding—designers can optimize for prover time, proof size, and quantum resistance depending on the application domain. As zero-knowledge systems continue to evolve, MLE-PCS will remain a foundational tool in their design space.

---

## 2. LaBRADOR: Lattice-Based Recursively Amortized Demonstration of R1CS

**LaBRADOR** is a cutting-edge lattice-based proof system that breaks away from traditional Probabilistically Checkable Proofs (PCPs) by leveraging stronger lattice assumptions—specifically, the **Module-SIS** (Short Integer Solution) problem. This allows LaBRADOR to achieve **more compact proofs** than known hash-based systems, both in theory and in practice.

LaBRADOR's core objective is to prove knowledge of a witness that satisfies a given constraint system—**without revealing the witness itself**. For example, the prover wishes to show they know a set of vectors $\vec{s}_1, \ldots, \vec{s}_r \in \mathbb{R}_q^n$ that satisfy certain **inner product equations and norm bounds**, without leaking the vectors' actual values.

This setup defines a **principal relation $R$**, composed of:

- A set of functions $F$ over the witness (called the _main constraints_),
- A set of auxiliary functions $F'$ (only concerned with constant terms),
- A norm bound $\beta$.

Each function in $F$ takes the form:

$f(\vec{s}_1, \ldots, \vec{s}_r) = \sum_{i,j=1}^{r} a_{ij} \langle \vec{s}_i, \vec{s}_j \rangle + \sum_{i=1}^{r} \langle \vec{\varphi}_i, \vec{s}_i \rangle - b$

Where:

- Each $\vec{s}_i \in \mathbb{R}_q^n$ (vectors over a ring modulo $q$)
- Coefficients $a_{ij}, b \in \mathbb{R}_q$
- Vectors $\vec{\varphi}_i \in \mathbb{R}_q^n$
- The inner product $\langle \cdot, \cdot \rangle$ is taken modulo $q$
- The ring is $\mathbb{R}_q = \mathbb{Z}_q[X]/(f(X))$

The **witness** is valid if the following three conditions are satisfied:

- **Main Constraint**: All $f \in F$ evaluate to zero
- **Constant Term Constraint**: For all $f' \in F'$, the constant term $\text{ct}(f') = 0$
- **Norm Bound**: $\sum_{i=1}^r |\vec{s}_i|_2^2 \leq \beta^2$

LaBRADOR represents a lattice-based translation of **Rank-1 Constraint Systems (R1CS)** into this structured relation $R$, with a small permissible slack (e.g., norm bound might increase by a factor of 2), which is tolerable and secure in lattice cryptography.

#### Example

Suppose $r = 2$ and the constraint is:

$f(\vec{s}_1, \vec{s}_2) = \langle \vec{s}_1, \vec{s}_1 \rangle + \langle \vec{\varphi}_1, \vec{s}_1 \rangle + \langle \vec{\varphi}_2, \vec{s}_2 \rangle - b$

Then, to prove knowledge of $\vec{s}_1, \vec{s}_2$ without revealing them, the prover must show:

- $f(\vec{s}_1, \vec{s}_2) = 0$
- Other functions in $F'$ have zero constant term
- Norm constraint $|\vec{s}_1|^2 + |\vec{s}_2|^2 \leq \beta^2$ holds

### Core Techniques in LaBRADOR

LaBRADOR achieves efficient zero-knowledge proofs using **four main cryptographic primitives**:

#### Commitment (Hiding the Witness)

To hide the witness vectors, LaBRADOR uses **Ajtai commitments**, which are lattice-based and post-quantum secure.

- **Inner Commitment**: For each vector $\vec{s}_i$, compute $\vec{t}_i = A \vec{s}_i \in \mathbb{R}_q^\kappa$ using a public matrix $A$.
- **Outer Commitment**: Since sending all $\vec{t}_i$ is expensive, each is decomposed by a radix-$b_1$ representation:

  $\vec{t}_i = \vec{t}_i^{(0)} + b_1 \vec{t}_i^{(1)} + \cdots + b_1^{t_1 - 1} \vec{t}_i^{(t_1 - 1)}$

  Then these are concatenated into a long vector $\vec{t}$ and committed via another matrix:

  $\vec{u}_1 = B \vec{t}$

  Additional “garbage” terms (intermediate decomposition data) are committed via $\vec{u}_2$ and will be revealed later in a recursive manner.

#### Norm Bound via Johnson-Lindenstrauss Projection

Directly checking the norm $\sum_i |\vec{s}_i|^2 \leq \beta^2$ would be costly. Instead, the protocol uses **random projections**:

- Verifier generates random matrices $\Pi_i \in {-1,0,1}^{256 \times nd}$
- Prover computes $\vec{p} = \sum_i \Pi_i \vec{s}_i$
- Verifier checks $|\vec{p}|^2 \leq 128 \cdot \beta^2$

This serves as a proxy for the true norm check with bounded slack. These projections are encoded into linear functions added to $F'$ for verification of constant terms.

#### Aggregation (Combining Multiple Constraints)

To reduce communication cost, the prover aggregates many constraints from $F$ and $F'$ into fewer functions:

- **Round 1**: The verifier samples random vectors $\vec{\psi}^{(k)}$ and $\vec{\omega}^{(k)}$.
  The prover computes for each $k$:

  $f''^{(k)} = \sum_l \psi^{(k)}_l f'^{(l)} + \sum_j \omega^{(k)}_j \left( \sum_i \langle \pi^{(j)}_i, \vec{s}_i \rangle - p_j \right)$

  Each resulting $f''^{(k)}$ is a linear combination of original $F'$-type functions and inner product constraints, and satisfies:

  $\text{ct}(f''^{(k)}) = 0$

  Thus, each $f''^{(k)}$ can be treated as an $F$-type function.

- **Round 2**: The verifier sends random scalars $\alpha\_k$ and $\beta\_k$.
  The prover combines all functions into a single constraint:

  $F(\vec{s}_1, \ldots, \vec{s}_r) = \sum_k \alpha_k f_k + \sum_k \beta_k f''^{(k)}$

Now, instead of proving multiple constraints, the prover only needs to prove that this single aggregated function satisfies:

$F = 0$

This transformation drastically reduces the number of proofs needed, improving overall efficiency.

#### Amortization (Random Linear Combinations)

To avoid revealing all $\vec{s}_i$, the verifier sends **random challenge polynomials** $c_1, \ldots, c_r \in \mathbb{R}_q$. The prover computes:

$\vec{z} = \sum_{i=1}^r c_i \vec{s}_i$

This **single vector** replaces the entire witness in the proof. All inner products in $F$ can be re-expressed as functions of $\vec{z}$:

- $\langle \vec{z}, \vec{z} \rangle = \sum_{i,j} g_{ij} c_i c_j$
- $\sum_i \langle \vec{\varphi}_i, \vec{z} \rangle c_i = \sum_{i,j} h_{ij} c_i c_j$
- Total sum checks against constant $b$: $\sum a_{ij} g_{ij} + \sum h_{ii} = b$

The prover commits to $g_{ij}, h_{ij}$ as “garbage terms” using the second outer commitment $\vec{u}_2$, which are then revealed with $\vec{z}$ during verification.

#### Verification

The verifier performs three critical checks to validate the proof:

- **Ajtai Commitment Consistency**: Verify that the amortized vector $\vec{z}$ and garbage terms $\vec{g}, \vec{h}$ are consistent with the previously sent commitments $\vec{u}_1, \vec{u}_2$. This ensures the prover cannot change their committed values after seeing the challenges.

- **Projection Norm Bound**: Check that the projection vector satisfies $\|\vec{p}\|^2 \leq 128 \cdot \beta^2$. Through the Johnson-Lindenstrauss lemma, this indirectly verifies that the original witness vectors satisfy their norm constraints without revealing the vectors themselves.

- **Aggregated Constraint Satisfaction**: Verify that the reconstructed aggregated function $F(\vec{z}) = 0$ holds. This ensures all dot product constraints and constant term constraints from the original relation $R$ are satisfied.

If all verification checks pass, the verifier is convinced that the prover possesses a valid witness satisfying the principal relation $R$, while maintaining complete zero-knowledge about the witness content.

---

### Recursive Composition in LaBRADOR

The core goal of the LaBRADOR protocol is to prove knowledge of a set of vectors $\vec{s}_1, \dots, \vec{s}_r$ satisfying certain **dot product constraints** and a **norm bound**, collectively called the _principal relation_ $R$. At the end of the main protocol, the prover sends several values—$\vec{z}, \vec{t}, \vec{g}, \vec{h}$—that themselves satisfy a new set of dot product and norm constraints structurally similar to $R$. This observation allows LaBRADOR to apply **recursion**: the main protocol can be executed _on itself_, enabling **recursive composition**. This forms the foundation for building highly efficient, recursive zero-knowledge proof systems, such as recursive SNARKs.

#### Merging Constraints for Recursion

To facilitate recursion, LaBRADOR reorganizes the output of the main protocol into a format compatible with another instance of the same protocol:

- **Unified Norm Constraint**:
  Initially, there are three separate norm bounds:

  $\|\vec{z}\|^2 \leq \gamma^2,\quad \|\vec{t}\|^2 \leq \gamma_1^2,\quad \|\vec{g}\|^2 + \|\vec{h}\|^2 \leq \gamma_2^2$

  These are merged into a single constraint:

  $\|\vec{z}\|^2 + \|\vec{t}\|^2 + \|\vec{g}\|^2 + \|\vec{h}\|^2 \leq \gamma^2 + \gamma_1^2 + \gamma_2^2$

  enabling the new relation to match the same structure as the original $R$: a set of dot products and one global norm bound.

- **Linear Constraint Consolidation**:
  Vectors $\vec{t}, \vec{g}, \vec{h}$ appear only in _linear_ inner product equations (not quadratically like $\vec{z}$). So they are concatenated into a single large vector:

  $\vec{v} = \vec{t} \,\|\, \vec{g} \,\|\, \vec{h}$

  All constraints can now be rephrased as inner product equations involving only $\vec{z}$ and $\vec{v}$.

#### Decomposition for Controlled Recursion

Direct recursion introduces a challenge: the bit-length and norm of $\vec{z}$ can grow exponentially across recursive layers. To prevent this blow-up, LaBRADOR applies **base decomposition** to $\vec{z}$:

- Decompose $\vec{z}$ with respect to a base $b$:

  $\vec{z} = \vec{z}^{(0)} + b \cdot \vec{z}^{(1)}$

  where $\vec{z}^{(0)}$ is the centered residue modulo $b$ (e.g., values in ${\pm1, \pm2}$ if $b=4$), and $\vec{z}^{(1)}$ contains the high bits.

- Inner product terms decompose accordingly:

  $\langle \vec{z}, \vec{z} \rangle = \langle \vec{z}^{(0)}, \vec{z}^{(0)} \rangle + 2b \langle \vec{z}^{(0)}, \vec{z}^{(1)} \rangle + b^2 \langle \vec{z}^{(1)}, \vec{z}^{(1)} \rangle$

  which can be handled by the same proof machinery.

#### Preparing the Recursive Input

To execute recursion, LaBRADOR constructs a new input instance of the protocol:

- **Vector Partitioning**:
  Vectors $\vec{z}^{(0)}$, $\vec{z}^{(1)}$, and $\vec{v}$ are each partitioned into chunks of fixed length $n'$, forming new vectors:

  $\vec{s}_1', \dots, \vec{s}_{r'}' \in \mathbb{R}_q^{n'}$

- **Constraint System Construction**:
  A new set of dot product constraints is defined over these new witness vectors:

  $g^{(k)}(\vec{s}_1', \dots, \vec{s}_{r'}') = \sum_{i,j} a_{ij}^{(k)} \langle \vec{s}_i', \vec{s}_j' \rangle + \sum_i \langle \vec{\varphi}_i^{(k)}, \vec{s}_i' \rangle - b^{(k)} = 0$

- **Norm Bound Update**:
  The combined norm constraint becomes:

  $$
  \|\vec{z}^{(0)}\|^2 + \|\vec{z}^{(1)}\|^2 + \|\vec{v}\|^2 \leq \underbrace{\frac{2}{b^2} \gamma^2 + \gamma_1^2 + \gamma_2^2}_{:= (\beta')^2}
  $$

  forming a new relation $R(n', r', \beta')$ suitable for the next recursive layer.

---

### Bridging R1CS and Lattice-Based Proofs: The LaBRADOR Approach

One of the central innovations of the LaBRADOR system is its ability to reduce **Rank-1 Constraint Systems (R1CS)**—a foundational constraint system in modern zero-knowledge proofs and SNARK/zkVM architectures—to a system of **dot product constraints** over lattice structures. This reduction forms a crucial bridge between **general-purpose computation** and **lattice-based zero-knowledge proof systems**, enabling the use of lattice commitments and proof protocols (as introduced earlier) to securely and efficiently prove R1CS satisfiability.

#### From R1CS to Dot Products

An R1CS instance is defined by three matrices $(A, B, C)$ over a ring $\mathbb{Z}_q$, along with a witness vector $\vec{w}$ satisfying:

$A\vec{w} \circ B\vec{w} = C\vec{w}$

where $\circ$ denotes the **Hadamard product** (element-wise multiplication).

Lattices natively support **linear** constraints (e.g., inner products), but not multiplicative ones. Therefore, to prove R1CS relations using lattice-based systems, the multiplication must be **embedded into a linear form**. This is achieved through a clever reduction based on **binary constraint encoding**.

#### Binary R1CS Over $\mathbb{F}_2$

LaBRADOR sidesteps multiplication by working modulo 2, where Boolean operations behave algebraically:

- **Commitment Phase**:
  Commit to the vectors $(\vec{a}, \vec{b}, \vec{c}, \vec{w})$ such that:

  $\vec{a} = A\vec{w} \mod 2,\quad \vec{b} = B\vec{w} \mod 2,\quad \vec{c} = C\vec{w} \mod 2$

- **Multiplication as Linear Constraints**:
  Use the identity:

  $ab = c \iff a + b - 2c \in \{0,1\}$

  for $a, b, c \in {0,1}$. This converts multiplication into a **binary constraint**. Define:

  $\vec{d} := \vec{a} + \vec{b} - 2\vec{c}$

  Then, proving $\vec{d}$ is a binary vector (i.e., all components are in ${0,1}$) ensures the original multiplication constraints hold. This binary proof can be handled using **quadratic constraints** of the form $x(x - 1) = 0$.

- **Efficient Linear Constraint Checking**:
  Rather than proving each of the $k$ linear constraints $\vec{a} = A\vec{w}$, etc., one can prove a **random linear combination** of them. The verifier selects random vectors $\vec{\alpha}, \vec{\beta}, \vec{\gamma} \in \mathbb{F}_2^k$, and the prover constructs:

  $\delta := \alpha^T A + \beta^T B + \gamma^T C$

  The verifier then checks:

  $\langle \vec{\alpha}, \vec{a} \rangle + \langle \vec{\beta}, \vec{b} \rangle + \langle \vec{\gamma}, \vec{c} \rangle - \langle \delta, \vec{w} \rangle = 0 \mod 2$

  This reduces the set of nonlinear Hadamard constraints into a few **linear inner products with binary coefficients**, which can be proven using lattice-based commitment systems.

#### Supporting Large Modulus Arithmetic: R1CS mod $2^d + 1$

Many real-world applications operate over large moduli, such as $q = 2^{64} + 1$, which are incompatible with the binary trick above. LaBRADOR extends its framework to support **modulo-$2^d + 1$ arithmetic**, enabling efficient constraints over large integers.

- **Embedding into Lattices via Ring Morphism**:
  Elements of $\mathbb{Z}_{2^d + 1}$ are embedded into the polynomial ring $R_q$ using the morphism:

  $\phi: R \to \mathbb{Z}_{2^d + 1}, \quad \sum a_i X^i \mapsto \sum a_i 2^i \mod 2^d + 1$

  Representations use **non-adjacent form (NAF)** to keep coefficients small, ensuring compatibility with **Ajtai-style lattice commitments**.

- **Proving Arithmetic Constraints**:
  Commit to encoded values $\mathrm{Enc}(a), \mathrm{Enc}(b), \mathrm{Enc}(c), \mathrm{Enc}(w)$ with small coefficients. To prove $a \cdot b = c \mod 2^d + 1$, the verifier sends $l$ random vectors $\phi_i$, and the prover computes:

  $d_i := \phi_i \circ a,\quad \text{then proves } \langle d_i, b \rangle = \langle \phi_i, c \rangle$

  These are again dot product constraints, reducing back to **linear inner products over the ring**, expressible and provable within the lattice proof system.

#### Composing Binary and Mod-$2^d + 1$ R1CS

Practical applications often involve both binary logic and large integer arithmetic. For example, proving the validity of a **Dilithium signature** requires:

- Binary constraints for the message hash and Fiat-Shamir challenge (e.g., SHA or Merkle hash logic),
- Large-modulus arithmetic for signature verification over $\mathbb{Z}_q$.

LaBRADOR supports **mixed R1CS encodings**—combining binary and mod-$2^d + 1$ constraints—in a unified system. All components share the same lattice-based commitment layer, and the final proof **merges constraints across both domains**, enabling powerful and flexible zero-knowledge applications.

---

## 3. Greyhound: A Lattice-Based Polynomial Commitment Scheme

**Greyhound** is a lattice-based **polynomial commitment scheme** built on standard assumptions such as **SIS (Short Integer Solution)** and **LWE (Learning With Errors)**. Its purpose is to allow a prover to convince a verifier that a committed polynomial $f(x)$, of degree at most $N$, evaluates to a claimed value $y$ at a point $z$—i.e., $f(z) = y$—**without revealing the polynomial itself**.

At its core, Greyhound is a **three-round interactive protocol**:

- The **prover** sends a structured commitment.
- The **verifier** sends a random challenge.
- The **prover** responds with an evaluation proof.

Verification requires only $O(\sqrt{N})$ work, making Greyhound highly efficient. It serves as a foundational building block for evaluation proofs, while **LaBRADOR** functions as a **general-purpose zero-knowledge proof system** (akin to SNARKs such as Groth16 and Plonk). LaBRADOR adds **succinctness** and **zero-knowledge**, and is capable of compressing complex constraint systems—including those derived from Greyhound—into compact, composable proofs.

Greyhound works by expressing its evaluation constraints as **dot product constraints**, which can be embedded into LaBRADOR’s lattice-based framework. The entire process leverages structured lattice commitments, with optimizations including **number-theoretic transforms (NTT)** and **AVX-512** (Intel’s SIMD instruction set) for efficient matrix-vector multiplication and polynomial operations over lattices.

### Algebraic Setting

Greyhound is built on the **Ring-SIS** assumption. Computations are performed over the ring:

$R_q = \mathbb{Z}_q[X]/(X^d + 1)$

Let $\delta = \lceil \log q \rceil$ be the bit-length required to represent elements in binary. A **gadget matrix** is used for binary expansion:

$G_n := I_n \otimes (1, 2, \ldots, 2^{\delta-1}) \in R_q^{n \times n\delta}$

This gadget enables conversion between ring elements and their binary encodings, which is crucial for constructing structured commitments.

### Commitment Construction

Greyhound builds a **two-layer commitment** to a set of polynomials $f_1, \ldots, f_r \in R_q^m$:

**Step 1: Inner Commitments (à la LaBRADOR)**
Each polynomial vector $f_i$ is binary-expanded via the inverse gadget:

$s_i = G_m^{-1}(f_i) \in R_q^{m\delta}$

Then compute:

$t_i = A \cdot s_i \in R_q^n$

where $A$ is a public matrix used in the lattice commitment.

**Step 2: Outer Commitment**
Each $t_i$ is again binary-expanded:

$\hat{t}_i = G_n^{-1}(t_i) \in R_q^{n\delta}$

These expanded vectors are **packed into a single outer commitment**:

$$
u := B \cdot
\begin{bmatrix}
\hat{t}_1 \\
\vdots \\
\hat{t}_r
\end{bmatrix}
\in R_q^n
$$

This nested structure enables compact, composable commitments with high linearity, which are suitable for efficient lattice-based proof systems.

### Commitment Opening and Binding

To open a commitment and prove its binding, the prover reveals:

- $s_i = G_m^{-1}(f_i)$
- $\hat{t}_i = G_n^{-1}(t_i)$

The verifier checks:

- $f_i = G_m s_i$
- $A s_i = G_n \hat{t}_i$
- $u = B [\hat{t}_1; \ldots; \hat{t}_r]$

These checks ensure consistency with the commitment. Binding security relies on the **Module-SIS** assumption: if a prover could find two different openings satisfying these equations, they would solve an SIS instance, which is assumed to be hard.

### Evaluation Proof for $f(z) = y$

To prove that the committed polynomial evaluates to a value $y$ at a point $x$, the prover demonstrates knowledge of $f_1, \ldots, f_r$ such that:

$a^\top (f_1 \| \ldots \| f_r) = y$

Here, $a \in R_q^{rm}$ is a vector determined by $x$—for example, using the evaluation basis $(1, x, x^2, \ldots, x^{m-1})$, structured into a Kronecker product.

#### Interactive Proof (3-Round Protocol)

This protocol can be transformed into a non-interactive proof using the **Fiat-Shamir heuristic**.

- **Prover Sends**:

  $w = a^\top (f_1 \| \ldots \| f_r) = a^\top G_m (s_1 \| \ldots \| s_r) \in R_q^r
  $

- **Verifier Sends**:
  A random challenge vector $c \in R_q^r$

- **Prover Responds**:

  - The binary-expanded $\hat{t}_i = G_n^{-1}(t_i)$
  - $z = \sum_i c_i s_i \in R_q^{m\delta}$

- **Verifier Checks**:

  - $w^\top b = y$
  - $w^\top c = a^\top G_m z$
  - $A z = \sum_i c_i G_n \hat{t}_i$
  - $u = B [\hat{t}_1; \ldots; \hat{t}_r]$

These equations verify that the committed polynomial evaluates to the claimed value, while preserving zero-knowledge and soundness under the SIS assumption.

---

### Efficient Evaluation Proofs via LaBRADOR: Linear Systems and Dot Product Structures

In the basic three-round evaluation proof protocol of Greyhound, the **prover reveals a large number of intermediate vectors**—such as $\hat{t}_i$, $z$, and $w$—which can become costly in terms of communication and verifier computation.

To mitigate this, **Greyhound uses a key technique**: rather than revealing these vectors directly, the prover demonstrates knowledge of a **short solution to a fixed linear system**:

$$
M \cdot \begin{bmatrix} \hat{w} \\ \hat{t} \\ z \end{bmatrix} = \text{rhs}
$$

This is a core use case for **LaBRADOR**, a lattice-based proof system that excels at proving knowledge of short vectors satisfying structured linear relations.

#### Matrix System Encoding

The linear system in question takes the following expanded form:

$$
\begin{bmatrix}
D & 0 & 0 \\
0 & B & 0 \\
b^\top G_r & 0 & 0 \\
c^\top G_r & 0 & -a^\top G_m \\
0 & c^\top \otimes G_n & -A
\end{bmatrix}
\cdot
\begin{bmatrix}
\hat{w} \\
\hat{t} \\
z
\end{bmatrix} = \begin{bmatrix}v \\u \\y \\0 \\0\end{bmatrix}
$$

Each block of this matrix encodes a structural constraint from the evaluation proof:

- The outer commitments,
- Gadget decompositions,
- Dot product evaluations,
- Linear transformations through public matrices like $A$, $B$,
- And constraint vectors like $a$, $b$, and the Fiat–Shamir challenge vector $c$.

This system can be proven in zero-knowledge using LaBRADOR by committing to the short vector $(\hat{w}, \hat{t}, z)$ and proving it satisfies the linear relation above.

#### Dot Product Formulation of Polynomial Evaluations

To express polynomial evaluations in a structure suitable for LaBRADOR, we break down the polynomial into blocks and reframe the evaluation as a **double dot product** (or bilinear form).

Let:

- $N = m \cdot r$ be the total degree bound of a univariate polynomial.
- $f(x) = \sum_{i=0}^{N-1} f_i x^i$ be the polynomial to be evaluated.
- Split the coefficients into $r$ blocks of size $m$:

  $$
  f_1 = (f_0, ..., f_{m-1}),\quad
  f_2 = (f_m, ..., f_{2m-1}),\quad \ldots,\quad
  f_r = (\ldots, f_{N-1})
  $$

Let:

- $a = (1, x, x^2, ..., x^{m-1}) \in \mathbb{Z}_q^m$
- $b = (1, x^m, x^{2m}, ..., x^{(r-1)m}) \in \mathbb{Z}_q^r$

Then the evaluation $f(x)$ can be rewritten as:

$$
f(x) = a^\top \cdot (f_1 \| f_2 \| \ldots \| f_r) \cdot b
$$

This **bilinear structure**—a dot product between two vectors across a concatenated matrix—maps naturally to LaBRADOR’s proof system, which is optimized for proving such multilinear relationships with short witnesses.

#### Extending to Bivariate Polynomials

This dot product structure generalizes to bivariate polynomials. Consider:

$$
f(X, Y) = \sum_{i=0}^{m-1} \sum_{j=0}^{r-1} f_{i,j} X^i Y^j
$$

This is a polynomial of degree $m-1$ in $X$ and $r-1$ in $Y$. It can be rewritten as a **matrix-vector-matrix product**:

$$
f(X, Y) =
\begin{bmatrix}
1 & X & X^2 & \cdots & X^{m-1}
\end{bmatrix}
\cdot
\underbrace{
\begin{bmatrix}
f_{0,0} & \cdots & f_{0,r-1} \\
f_{1,0} & \cdots & f_{1,r-1} \\
\vdots & & \vdots \\
f_{m-1,0} & \cdots & f_{m-1,r-1}
\end{bmatrix}
}_{\text{Coefficient Matrix } M}
\cdot
\begin{bmatrix}
1 \\
Y \\
Y^2 \\
\vdots \\
Y^{r-1}
\end{bmatrix}
$$

This again fits the structure:

$f(X, Y) = a^\top \cdot M \cdot b$

where:

- $a = [1, X, X^2, \ldots, X^{m-1}]$
- $b = [1, Y, Y^2, \ldots, Y^{r-1}]$
- $M$ is the matrix of coefficients, whose columns we denote as $f_1, \ldots, f_r$

Each column $f_i$ of the matrix $M$ is binary-decomposed via a gadget matrix:

$s_i := G_{b_0, m}^{-1}(f_i) \in \mathbb{R}_q^{\delta_0 m}$

The entire evaluation proof thus becomes a structured dot product over these gadget-expanded columns and basis vectors $a, b$, again suitable for LaBRADOR.

---

### Polynomial Commitments over $\mathbb{Z}_q$ via Ring Transformations in LaBRADOR

LaBRADOR operates over the ring $R_q = \mathbb{Z}_q[X]/(X^d + 1)$, whereas most traditional polynomial commitment schemes work directly over the finite field $\mathbb{Z}_q$. To bridge this gap, LaBRADOR leverages a technique from AFLN24 that transforms polynomial evaluation statements over $\mathbb{Z}_q$ into equivalent computations within $R_q$.

Suppose we wish to prove a polynomial evaluation statement of the form:

$f(x) = y \quad \text{over } \mathbb{Z}_q, \quad \deg(f) < N$

To do this using LaBRADOR, the polynomial $f$ is first partitioned into chunks of length $d$, resulting in:

$f = (f_0, f_1, \dots, f_{N/d - 1})$

Each chunk $f_i$ corresponds to a degree-$d$ polynomial over $\mathbb{Z}_q$, which can be naturally embedded into $R_q$.

Next, the input $x \in \mathbb{Z}_q$ is encoded as a polynomial in $R_q$ by defining:

$x(X) = \sum_{j=0}^{d-1} x_j X^j$

where $x_j \in \mathbb{Z}_q$ are components derived from the original scalar input $x$.

The evaluation then becomes:

$y = \text{constant term of } \sum_{i=0}^{N/d - 1} \sigma^{-1}(x) \cdot f_i \cdot x^{d^i}$

Here, $\sigma^{-1}$ denotes the Galois conjugate operator in $R_q$, which acts by mapping $X \mapsto X^{-1}$. LaBRADOR utilizes this transformation to express the result in a form suitable for zero-knowledge proof systems over rings. In particular, it rewrites the evaluation as:

$f(x^d) = \sigma^{-1}(x)^{-1} \cdot y$

This reformulation enables computations and commitments to remain entirely within $R_q$, while preserving correctness over $\mathbb{Z}_q$.

#### Hiding Polynomial Coefficients via Randomization

A common issue with polynomial commitment schemes is the potential leakage of information about the committed polynomial's coefficients. In naïve constructions, the commitment might directly include coefficient data, and during verification, some coefficients may be revealed explicitly.

To mitigate this, LaBRADOR incorporates a **randomization vector** to hide the original coefficients. The commitment is structured as:

$u = B\hat{t} + Er$

where:

- $B\hat{t}$ is the base commitment derived from the polynomial’s representation,
- $E$ is a random matrix,
- $r$ is a random masking vector.

This structure relies on the **Module-LWE (Learning With Errors)** assumption from lattice-based cryptography. Under this hardness assumption, the commitment $u$ reveals no useful information about $\hat{t}$, even if $u$ is publicly known. The use of random masking ensures statistical hiding, thereby preserving the zero-knowledge property of the commitment scheme.

[LaBRADOR_Greyhound_C](https://github.com/lattice-dogs/labrador)
[LaBRADOR_Greyhound_Python](https://wiki.lacom.io/wiki/cryptography/greyhound-poc)
