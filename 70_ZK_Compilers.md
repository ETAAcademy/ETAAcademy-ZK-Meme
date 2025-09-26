# ETAAcademy-ZKMeme: 70. ZK Compilers

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>70. ZK Compilers</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZK_Compilers</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# ZK Compiler Ecosystem: From Circuit Generation to Security Verification

ZK compilers serve as the core infrastructure of zero-knowledge proof ecosystems, enabling developers to build zero-knowledge proof applications without directly handling complex low-level circuit designs by transforming high-level program code into ZKP-friendly circuits or bytecode.

This technology stack encompasses a complete taxonomy ranging from general-purpose circuit compilers (Cairo, ZoKrates, Noir) to specialized compilers (zkSolc, Zinnia), addressing wire consistency challenges in SIMD compilers through linear mapping proofs and segment-by-segment proving strategies, while achieving efficient machine learning model verification in ZKML compilers through DAG representation and hierarchical optimization.

To ensure compiler security and reliability, the industry has developed specialized testing frameworks including fuzzing and mutation testing to detect completeness, correctness, and soundness bugs, while employing rigorous verification conditions (VCs) and lowering processes such as field-blasting/flattening to ensure that compilers correctly extend the soundness and completeness properties of ZK proof systems from high-level languages to underlying constraint systems, thereby providing a reliable technical foundation for privacy-preserving computation, blockchain applications, and verifiable machine learning domains.

---

## 1. Zero-Knowledge (ZK) Compilers

Compiler verification generally falls into two categories: **foundational verification** and **automatic verification**.

- **Foundational verification** relies on proof assistants such as **Coq** or **Isabelle**, where every compiler transformation is manually proven correct. A prominent example is **CompCert**, a C compiler verified in Coq, including its optimization passes. Similar efforts include backend verification, which proves correctness of the lowering process from intermediate representation (IR) to machine code—for example, CompCert’s verified lowering, and CakeML’s lowering to multiple ISAs. Foundational approaches provide strong guarantees but come with high proof-maintenance costs: CompCert, for instance, took a team of experts nearly eight years to complete, and any code update requires updates to the corresponding proofs.

- **Automatic verification**, in contrast, uses automated reasoning tools and avoids the burden of manual proof maintenance. Tools such as **Cobalt**, **Rhodium**, and **PEC** provide domain-specific languages (DSLs) for describing verified compiler optimizations and analyses. Of particular relevance to ZK compilers is **Alive**, a DSL for expressing verified peephole optimizations in LLVM. Alive translates snippets of LLVM IR into more efficient fragments while ensuring correctness even in the presence of undefined behavior. Alive2 extends this to translation validation for LLVM, and **VeRA** performs verified range analysis in the Firefox JavaScript engine.

Work closer to cryptography also exists. For example, the **Porcupine compiler** automatically synthesizes representations for fully homomorphic encryption, while **Gillar** proves the semantic preservation of optimization passes in the Qiskit quantum compiler. For zero-knowledge proofs (ZKPs), prior verification attempts have taken foundational approaches, such as **PinocchioQ**, which builds on CompCert. However, these focus on compiling high-level programs into circuit representations without aligning correctness guarantees with the requirements of ZKP systems.

### From Programs to Proofs: The Role of ZK Compilers

A zero-knowledge proof allows a prover to convince a verifier that a constraint C(u, w) holds for public input **u** and private input **w**, without revealing **w** itself. To achieve this, high-level constraints are transformed into **existentially quantified circuits (EQCs)**. Different proving systems (e.g., **SNARKs**, **STARKs**, **Halo2**) use different circuit formats—such as **R1CS**, **AIR**, or **PLONK**.

Because writing circuits by hand is cumbersome, DSLs such as **ZoKrates**, **Cairo**, and **Noir** provide higher-level programming abstractions. Their compilers transform developer-written code into EQCs:

- **Front-end phase** – Parses source code, analyzes syntax and semantics, marks variable visibility (public vs. private), performs type checking, and resolves control flow.
- **Intermediate representation (IR)** – Converts the abstract syntax tree (AST) into a front-end IR suitable for further analysis. Information-flow (taint) analysis ensures that all variables influenced by private inputs are properly marked.
- **Flattening phase** – Converts IR into a circuit-friendly representation by applying:

  - **Static single assignment (SSA)** transformation,
  - **Loop unrolling**,
  - **Function inlining**,
  - **Conditional rewriting** (if-statements into selectors).

- **Front-end optimizations** – Includes common subexpression elimination, constant propagation, and bit recombination.
- **Back-end phase** – Transforms the flattened IR into an algebraic circuit representation, applies algebraic optimizations (linear reductions, liveness analysis), and completes arithmeticization.

The result is an optimized **finite field constraint system** ready for proof generation and verification by a ZK protocol.

### Public vs. Private Inputs

Inputs to a ZK program are typed over the underlying field:

- **Public inputs (u):** Visible to both prover and verifier.
- **Private inputs (w):** Known only to the prover, hidden from the verifier.

In ZK DSLs, developers typically specify constraints using `assert` statements. Instead of hand-writing circuits, they define high-level conditions (e.g., `assert(x * y == z)`), which the compiler later translates into EQCs. The program output consists of return values and the Boolean outcomes of all assertions. Correctness is captured by the relation:

$$
P(u, w) = y \quad \Leftrightarrow \quad \exists w \; \text{s.t. } C(u, w, y) = \text{SAT}.
$$

### The Landscape of ZK Compilers

ZK compilers can be grouped into four categories:

- **General-purpose circuit compilers** – e.g., **ZoKrates**, **Noir**, **Cairo**, **Circom**, which translate programs into ZK-friendly circuits.
- **Smart contract compilers** – e.g., **zkSolc**, **Polygon zkEVM Compiler**, which lower Solidity contracts into zkEVM bytecode.
- **Research and specialized compilers** – e.g., **Ligero**, **CP-SNARK**, **Zinnia**, exploring performance optimization and new compilation methods.
- **Auxiliary tooling** – e.g., **zkSolc Source Mapping**, which bridges Solidity source code, LLVM IR, and zkEVM bytecode for better developer tooling.

Across all categories, the core mission of a ZK compiler remains the same: **translate high-level developer code into efficient, proof-friendly representations**—whether algebraic circuits or zkEVM bytecode—enabling secure and scalable zero-knowledge applications.

---

## 2. SIMD and ZKML Compilers

### 2.1 SIMD Compilers

Most efficient zero-knowledge (ZK) systems follow a **two-phase pattern**:

- **Batch commitment** – Witness values are packed and committed using vector or polynomial commitments.
- **Consistency proofs** – The prover must demonstrate both (a) correctness of Hadamard products for multiplication gates, and (b) _wire consistency_ across different committed vectors.

Many existing ZKP schemes can be seen as reducing **SIMD-ZK → General ZK**. For instance, **R1CS (rank-1 constraint system)** represents circuits as matrix–vector constraints:

$$
L_z \cdot R_z - O_z = 0.
$$

Other schemes verify wire consistency directly, but often incur large communication overhead (on the order of $B^2$).

#### From SIMD-ZK to General ZK

A **SIMD compiler** aims to transform arbitrary general circuits into SIMD-ZK-friendly form while maintaining sublinear communication costs.

- **Limitation of SIMD-ZK:** It efficiently proves repeated subcircuits (e.g., $B$ identical subcircuits $C$), but cannot directly express general circuits.
- **The key challenge:** **Wire consistency.** Once wires are batched, the same logical wire may appear multiple times across different batches. The prover must convince the verifier that these wire instances correspond to the same value.

A naïve approach (e.g., in **AntMan**) checks consistency pairwise, which costs $O(B^3)$. Even with optimizations, the best-known results still scale as $O(C^{3/4})$. Thus, a more efficient wire-consistency mechanism is needed.

#### Linear Map Proofs

A new approach reframes wire consistency as a **matrix–vector relation**:

- Define a global **wire vector** $w \in \mathbb{F}^{Bm}$, containing all wire values in the circuit.
- Each gate’s input vector $l \in \mathbb{F}^{Bn}$ is simply a projection of $w$.
- A sparse $0\!-\!1$ matrix $L \in \{0,1\}^{Bn \times Bm}$ encodes this relation, so that:

$$
l = Lw.
$$

Thus, checking wire consistency reduces to proving that $l = Lw$ holds.

But SIMD-ZK excels only at proving _identical batch operations_, not arbitrary matrix selection. To bridge this gap, we use the **random challenge vector technique**:

- The verifier samples a random vector $\hat{r} \in \mathbb{F}^{Bn}$.
- Both parties compute inner products:

$$
\hat{r}^T l \stackrel{?}{=} \hat{r}^T (Lw).
$$

If this equality holds, it implies wire consistency across the entire batch. This compresses all consistency checks into **a single inner-product verification**, reducing complexity from $O(B^3)$ to $O(B)$.

To preserve zero-knowledge, a **masked random vector** $\tilde{r}$ with $\sum \tilde{r}[i] = 0$ is introduced, ensuring that no sensitive linear combinations of witness values are leaked.

The resulting system, called **FeSIMDZK**, extends plain SIMD-ZK beyond proving gate computations $o = l * r$, to also proving **linear relations** $l = Lw, r = Rw, o = Ow$. This makes it possible to express _any circuit_ in SIMD-ZK form.

#### Memory-Efficient Proofs for Large Circuits

The above approach requires storing the global wire vector $w$, leading to memory cost $O(C)$. For very large circuits, this becomes infeasible.

The solution is **segmented proofs**:

- Partition the circuit into segments $C = (C_1, C_2, \dots, C_{n'})$.

- For each segment $C_j$:

  - Prove internal correctness using SIMD-ZK.
  - Prove that outputs of $C_j$ match the inputs of $C_{j+1}$ via a linear map proof: $M w_j = \tilde{w}_{j+1}$.

- After completing one segment, discard its witness data and move on to the next.

This reduces memory usage from $O(C)$ to $O(|w_j| + |\tilde{w}_{j+1}|)$, depending only on segment size. In this way, **very large circuits** can be proven without requiring full-circuit memory storage.

#### From AntMan to AntMan++

- **AntMan** was originally a VOLE-SIMD protocol, restricted to SIMD circuits.
- **AntMan++** extends this via the SIMD→General ZK compiler, enabling proofs for **arbitrary circuits** with **sublinear communication complexity** $O(|C|^{1/2})$.

---

### 2.2 ZKML Compilers

As machine learning (ML) models become increasingly valuable, developers frequently deploy them on cloud platforms such as Google Cloud and provide access through **Machine Learning as a Service (MLaaS)**. However, two persistent challenges arise:

- **Users** want guarantees that the model’s outputs are computed correctly.
- **Developers** want to protect their intellectual property (IP) by hiding model details.

In regulated industries, transparency requirements often clash with the need for IP protection. Traditional approaches to transparency typically expose sensitive information about proprietary models. **Zero-Knowledge Machine Learning (ZKML)** resolves this tension by leveraging **Zero-Knowledge Proofs (ZKPs)**: provers can demonstrate the correctness of model inference without revealing the model weights or sensitive inputs.

#### ZKML Compilation Pipeline

The challenge is that existing ZKML approaches often require deep cryptographic expertise, which most AI developers lack. ZK compilers close this gap by seamlessly integrating ML frameworks (e.g., **PyTorch**) with ZKP engines (e.g., **Expander**).

The compilation process follows these steps:

- **Model Extraction** – The ML model is exported to an intermediate representation, typically **ONNX (Open Neural Network Exchange)**.
- **Graph Representation** – The model is expressed as a **directed acyclic graph (DAG)**:

  - **Vertices (V):** ML operators such as matrix multiplications, convolutions, activations (ReLU, softmax), pooling, or attention layers.
  - **Edges (E):** Data dependencies indicating flow of activations through the network.
    This DAG representation captures complex, non-sequential architectures (e.g., residual connections in ResNets) that simple sequential lists cannot.

3. **Circuit Translation** – ML operators are compiled into ZKP-friendly circuits using additions, multiplications, and lookup gates.
4. **Optimization & Instrumentation** – Additional edges and nodes are injected to carry auxiliary information needed for proof generation (e.g., remainders for division, lookup tables for non-linear activations).

This pipeline enables automated transformation of PyTorch models into ZKP circuits, bridging the gap between AI development and cryptographic proof generation.

#### Handling ML–ZKP Mismatches

A major challenge in ZKML is reconciling **standard ML computation** with the stricter requirements of **ZKP circuits**.

- **Auxiliary information:** Many ML operations, especially non-linear ones, need additional constraints in ZK.

  - Division requires proving that both the quotient and remainder are correct, with the remainder smaller than the divisor.
  - ReLU, softmax, and normalization layers rely on **lookup constraints**, mapping inputs to outputs via precomputed tables.

- **Quantization for finite fields:** Traditional ML uses floating-point arithmetic, whereas ZKPs operate over finite fields.

  - Early approaches simulated floats using fixed-point numbers, requiring large bit-widths and inefficient large fields (e.g., BN254).
  - **ZKPyTorch** introduces **static quantization**, with integer-based tensor-wise scaling, designed specifically for ZK systems. This reduces bit-width and allows computation over smaller, more efficient fields (e.g., M61), balancing accuracy and efficiency.

#### Circuit-Level Optimizations

ZKPyTorch applies **layered optimizations** to reduce proof size and generation time:

- **Model-level batching:** Sequential token-dependent computations in large models (e.g., LLMs) are transformed into **parallel verification**.

  - Example: A Transformer layer involves $L$ independent matrix multiplications of shape $1 \times H$ (activation) × $H \times W$ (weights).
  - Naïve approach: prove $L$ separate multiplications → complexity $O(LHW)$.
  - Batched approach: reduce to one large multiplication with complexity $O(WH + LH)$.
  - This significantly reduces gate count for Transformer and CNN layers.

- **Operation-level integration:** Incorporates best practices like FFT-based convolution optimization (ZKCNN) and lookup-table tricks (ZKLLM).
- **Circuit-level parallelism:** Introduces parallel execution across tensor operations, enabling efficient use of multicore CPUs and GPUs. A **broadcast mechanism** shares data between parallel tasks, minimizing redundancy.

These optimizations allow ZK circuits to scale to modern deep learning architectures without sacrificing precision.

---

## 3. Vulnerabilities and Testing of ZK Compilers: Fuzzing, Mutation, and Verification Conditions

Zero-Knowledge Proofs (ZKPs) are becoming increasingly popular in privacy-preserving applications and blockchain systems. To make ZKPs accessible to developers, the industry has designed domain-specific languages (DSLs) and compilers that transform programs into ZK circuits. A ZK compiler takes a program written in a DSL, compiles it into a constraint system or circuit, and passes it to the prover and verifier for ZK execution.

However, ZK compilers face a fundamental challenge: for an arbitrary computation $P$, there is no decision procedure that can guarantee correctness and robustness of the compiled circuit. This makes ZK compilers error-prone, and subtle bugs may allow malicious users to generate invalid proofs that are nonetheless accepted by verifiers, leading to serious security flaws and even large-scale financial losses in cryptocurrencies.

ZK compilers often contain tens of thousands—or even hundreds of thousands—of lines of code, spanning a complex pipeline of compilation and optimization. These include frontend parsing, intermediate representations, backend optimizations, and circuit generation. Such complexity makes correctness difficult to guarantee. Since ZK compilers are already deployed in highly sensitive domains like finance and blockchain, even small errors can cause privacy leaks or multimillion-dollar economic damages.

### 3.1 Fuzzing in ZKP Contexts

Fuzzing is a widely used automated software testing technique that continuously generates random inputs and checks whether a program behaves unexpectedly. However, applying fuzzing to ZKP compilers introduces unique challenges:

- **Test oracle design**: In traditional fuzzing, the oracle checks for crashes or unexpected outputs. In ZKP, the oracle must determine whether a circuit behaves incorrectly under cryptographic constraints.
- **Witness generation**: ZK circuits require valid witnesses (inputs satisfying the constraints), not arbitrary random numbers. Generating meaningful witnesses is far more complex than generating random input data.

#### Bug Types in ZK Circuits

Three primary categories of bugs arise in ZKP compilers and circuits:

- **Completeness bugs** – A valid input fails to produce a proof (e.g., the circuit misses a valid case). These are relatively easy to detect.
- **Correctness bugs** – A proof is generated but does not correspond to the intended computation (e.g., the circuit implements addition instead of multiplication). Detecting these requires knowing the correct expected behavior.
- **Soundness bugs** – The circuit is under-constrained, allowing invalid inputs to generate proofs that the verifier incorrectly accepts. This is the hardest category to detect, as it requires finding “fake witnesses” that exploit missing constraints.

A further complication is that ZKP proving and verification are expensive—large circuits can take hours to evaluate—making traditional “high-volume fuzzing” infeasible.

#### DSLs, Intermediate Representations, and Trade-offs

ZK circuits can be written in DSLs such as Circom, Noir, ZoKrates, Cairo, or Halo2. Each DSL is tied to a specific proving system, which complicates the development of universal fuzzing tools.

- **DSL-level fuzzers** leverage semantic knowledge of high-level constructs but are system-specific.
- **Low-level (constraint system) fuzzers** operate directly on representations like R1CS, making them portable but losing structural information.

A promising approach is to use **intermediate representations (IRs)** such as CirC, which abstract away DSL differences while preserving enough structure for effective testing. However, mismatches between DSLs and IR layers can introduce false positives or negatives.

#### Fuzzing Framework for ZK Circuits

A ZKP fuzzing framework typically follows a closed-loop process:

- **Circuit Compilation** → Compile the target DSL program into a constraint system and witness generator.
- **Input Generator** → Produce candidate inputs, guided by a test oracle or reference implementation.
- **Witness Generation** → Construct valid witnesses for the chosen inputs.
- **Mock Prover** → Run a lightweight simulated prover instead of a full cryptographic proof system, reducing test cost.
- **Evaluator** → Check outputs against the oracle to detect bugs.
- **Witness Mutation** → Modify valid witnesses to create “near-boundary” cases, stressing the circuit’s constraints.
- **Feedback Loop** → Use results to refine input generation and improve coverage.

#### Handling Cryptographic Primitives

Many ZK circuits include cryptographic subroutines (e.g., hashes, signatures). These are particularly challenging:

- Standard fuzzing cannot easily generate meaningful inputs for cryptographic functions.
- Specialized **oracles** or **white-box hooks** are needed, comparing circuit outputs to trusted reference implementations.
- Differential testing can be applied, where the same input is tested across different circuits or backends.

However, fuzzer guidance is still necessary to produce structurally valid inputs (e.g., preimages for hash functions).

---

### 3.2 Mutation Testing for ZK Compilers (MTZK): Detecting Logic Errors Beyond Crashes

Mutation testing introduces deliberately designed _metamorphic relations (MRs)_ to modify the inputs of a ZK compiler. By comparing the outputs of the original and mutated inputs, the compiler’s correctness can be tested automatically, without manual intervention.

Two broad categories of vulnerabilities arise in ZK compilers:

- **Compilation Failures** – These occur when the compiler crashes, throws an exception, or otherwise fails to produce output. Such failures may expose denial-of-service (DoS) or memory exploitation vulnerabilities. They are relatively easy to detect, and fuzzing is generally effective for this category.
- **Logic Errors** – These are far more subtle and dangerous. The compiler successfully produces output, but the compiled circuit or program does not preserve the intended semantics. Logic errors may cause verifiers to accept invalid proofs, creating serious security vulnerabilities. Detecting such bugs requires more sophisticated methods.

#### The MTZK Approach

**MTZK (Mutation Testing for ZK Compilers)** introduces two novel metamorphic relations specifically designed to expose hidden logic errors in ZK compilers:

- **Satisfiability-Invariant Mutation (SIM)**

  - Inserts constraints into the source program that are _guaranteed to be satisfiable_.
  - After compilation, the output program should also remain satisfiable (SAT).
  - If the compiled program yields UNSAT, this indicates an error in the compiler’s constraint translation process.

- **Information-Visibility Mutation (IVM)**

  - Mutates the visibility of inputs by swapping public and private variables.
  - The functional behavior of the program should remain unchanged.
  - If the mutated program produces a different satisfiability result, this reveals a flaw in the compiler’s information flow analysis or optimization passes.

These metamorphic relations are particularly effective at detecting **non-crashing but semantically incorrect outputs**—the kind of subtle logic errors that fuzzing alone often misses. Such errors carry greater security implications than compilation failures, since they may enable invalid proofs to bypass verification undetected.

#### Three-Stage MTZK Testing Workflow

MTZK follows a structured, three-stage testing pipeline:

- **Random Program Generation**

  - Randomly generate source programs in a ZK DSL.
  - Programs are first created in a custom intermediate representation (IR), then concretized into the syntax of the target DSL.

- **Satisfiability-Invariant Mutation (MR_SIM)**

  - Apply SIM to insert always-satisfiable constraints into the generated program.
  - Compile and check whether the SAT property is preserved.

- **Information-Visibility Mutation (MR_IVM)**

  - Apply IVM to the program, mutating variable visibility (public ↔ private).
  - Assert that the program remains satisfiable (SAT).
  - If the proof produced by the prover is rejected (UNSAT), the discrepancy indicates a compiler error.

#### Type-Guided Program Generation

Naive random expression generation tends to produce many ill-typed statements, which are rejected by the compiler frontend and waste testing resources. MTZK mitigates this by adopting a **type-guided generation strategy**:

- Enforce type correctness during generation to ensure programs pass the frontend.
- Use a depth parameter $d$ (default = 2) to bound expression complexity.
- Randomly choose a target type $\tau$, then select operators compatible with $\tau$.
- Recursively generate well-typed operands.

This ensures that most generated programs are syntactically and semantically valid, maximizing the efficiency of mutation testing.

#### Black-Box Testing, Broad Applicability

A key strength of MTZK is that it is a **black-box testing method**:

- It does not rely on the internal implementation details of the compiler.
- It can be applied to any ZK compiler, regardless of the programming language or protocol it supports.
- Minimal engineering effort is required to adapt MTZK to new compilers.

This universality makes MTZK suitable for testing ZK compilers across ecosystems—from Circom and Halo2 to ZoKrates, Noir, and beyond.

<details><summary>Code</summary>

```Algorithm: Type-Guided Expression Generation
function GENEXPR(τ, d)
    if d = 0 then
        return GENCONST(τ)
    op ← SELECTOP(τ)
    n_opd ← NUMOPDS(op)
    for i ← 1 to n_opd do
        τ_opd ← OPDTYPE(op, τ, i)
        opd_i ← GENEXPR(τ_opd, d-1)
    expr ← op(opd_1, ..., opd_n_opd)
    return expr
```

</details>

---

### 3.3 Correctness of ZKP Compilers: Challenges and Verification

The correctness of zero-knowledge proof (ZKP) compilers is critical for security. A flaw in the compiler could allow a false statement to be proven, undermining the integrity of the system. Yet, verifying compiler correctness is highly challenging for three main reasons:

- **Defining correctness is non-trivial.**
  Unlike traditional compilers, a ZKP compiler’s notion of correctness must account for both logical soundness and cryptographic guarantees.

- **ZKP compilers span multiple domains.**
  The prover must convince the verifier that a predicate φ(x, w) holds. High-level predicates φ′ are typically expressed in a language with common data types (e.g., booleans, fixed-width integers). The compiled statement φ, however, must be expressed over a large prime field. Thus, any definition of compiler correctness must bridge these heterogeneous domains.

- **ZKP compilers evolve rapidly and are performance-sensitive.**
  Verification must not preclude future compiler optimizations or introduce performance regressions.

Formally, given a ZK proof system where statements are specified in a low-level language **L**, a compiler translates from a high-level language **H** into **L**. If the compiler is correct, it extends the soundness and completeness properties of the ZK proof system from **L** to statements written in **H**. Moreover, correctness definitions are preserved under sequential composition, meaning that proving correctness of each compilation pass suffices to establish correctness of the full compiler.

#### From IR to R1CS: Lowering and Field-Blasting

A central step in ZKP compilation is the _lowering_ process, where intermediate representations (IRs) are transformed into Rank-1 Constraint Systems (R1CS). This lowering often proceeds in two phases:

- **Field-blasting.**
  This step converts formulas written in a mixed logic—combining bit-vectors and field operations—into constraints purely over a finite field. For example, CirC lowers from a mixed signature $(Σ_{BV} ∪ Σ_F)$ into the finite field signature $Σ_F$. The idea is to “blast” away the bit-vector structure by:

  - Encoding booleans with the constraint $v(v-1)=0$, forcing values to be 0 or 1.
  - Representing bit-vectors either as single field elements or by decomposing them into individual bits.
  - Expanding bitwise operations into field multiplications and additions.
  - Preserving native field operations directly.

  The result is a set of polynomial equations entirely within $F_p$, ready for a ZK backend (e.g., R1CS, PLONK).

- **Flattening.**
  The finite-field constraints are further lowered into R1CS form, which expresses each constraint as a rank-1 quadratic equation suitable for proof generation.

This lowering pipeline can be understood as a “unified translation” process that maps all data types—booleans, bit-vectors, and field elements—into arithmetic constraints over a single field.

#### Calculus of Compilation Rules

The lowering process can be described as a **calculus of compilation rules**. Each rule translates mixed-type predicates step by step into finite field constraints. Rules may introduce fresh variables, enforce new constraints, or apply encoding transformations until the original predicate φ is fully expressed as field equations.

This process is **non-deterministic**: multiple translation paths are possible. However, all valid translations are semantically equivalent; they differ only in efficiency.

<details><summary>Code</summary>

```Algorithm: Field-blasting
fn variable(t, isInst) → Enc :
    if isInst:
        t' ← fresh(name(t) || 'u';
        ∑ᵢ ite(t[i] ≈ 1₍₁₎, 2ⁱ, 0), ⊤)
        return t, uint, t'
    else:
        for i in [0, size(sort(t)) - 1]:
            t'ᵢ ← fresh(name(t) || i,
                ite(t[i] ≈ 1₍₁₎, 1, 0), ⊥)
            assert(t'ᵢ(t'ᵢ - 1) = 0)
        return t, bits, t'₀, ..., t'ₛᵢᵤₑ₍ₛₒᵣₜ₍ₜ₎₎₋₁

fn const(t) → Enc :
    for i in [0, size(sort(t)) - 1]:
        t'ᵢ ← ite(t[i] ≈ 1₍₁₎, 1, 0)
    return t, bits, t'₀, ..., t'ₛᵢᵤₑ₍ₛₒᵣₜ₍ₜ₎₎₋₁

fn assertEq(e : Enc, e' : Enc) :
    if kind(e) = bits:
        for i in [0, size(terms(e)) - 1]:
            assert(terms(e)[i] ≈ terms(e')[i])
    elif kind(e) = uint:
        assert(terms(e)[0] ≈ terms(e')[0])

fn convert(e : Enc, kind' : Kind) → Enc :
    t ← encoded_term(e)
    if kind(e) = bits and kind' = uint:
        return t, uint, ∑ᵢ 2ⁱterms(e)[i]
    elif kind(e) = uint and kind' = bits:
        e' ← variable(t, ⊥)
        assert(terms(e)[0] ≈ ∑ᵢ 2ⁱterms(e')[i])
        return e'

fn bvZeroExt(t, o : Op, e : Enc) :
    if kind(e) = bits:
        w ← size(terms(e))
        for i in [0, w - 1]:
            t'ᵢ ← terms(e)[i]
        for i in [0, o.newBits - 1]:
            t'w+i ← 0
        return t, bits, t'₀, ..., t'w+o.newBits-1
    else:
        return t, kind(e), terms(e)

fn bvMulUint(t, o : Op, ē : [Enc]) :
    w ← size(sort(encoded_term(e[0])))
    W ← size(ē) × w
    assume W < ⌊log₂ p⌋
    s' ← ∏ᵢ terms(eᵢ)[0]
    b ← ff2bv(W, s')
    for i in [0, W - 1]:
        t'ᵢ ← fresh(i, ite(b[i], 1, 0), ⊥)
        assert(t'ᵢ(t'ᵢ - 1) ≈ 0)
    assert(s' ≈ ∑ᵢ₌₀^(W-1) 2ⁱt'ᵢ)
    return t, bits, t'₀, ..., t'W-1

```

</details>

#### Verification Conditions for Encoding Rules

To ensure correctness of the encoding process, **Verification Conditions (VCs)** are defined. These conditions check both _soundness_ (the encoding does not admit false statements) and _completeness_ (all valid statements are preserved).

- **Encoding uniqueness.** Each term must have a unique valid encoding within a chosen scheme. Functions `fromTerm` and `toTerm` establish a bijection between terms and encodings.
- **Operator rules.** Input encodings and assertions must imply validity of the output encoding (_soundness_), and conversely, the validity of the output must imply the input encodings and assertions (_completeness_).
- **Equality rules.** Equality constraints on encodings must correctly reflect equality of the underlying terms.
- **Conversion rules.** Transformations between different encodings must preserve both soundness and completeness.
- **Variable rules.** Witness variables and instance variables are treated separately, ensuring that assertions guarantee valid encodings.
- **Constant rules.** Constants always map to valid encodings, requiring only a minimal VC.

By satisfying these verification conditions, a ZKP compiler ensures that its translation from high-level predicates to field constraints preserves the essential soundness and completeness properties of the proof system.

---

[ONNX](https://github.com/onnx/onnx)
[Zkp-Compiler-Shootout](https://github.com/anoma/zkp-compiler-shootout)
[Cairo](https://github.com/starkware-libs/cairo/tree/main)
[Noir](https://github.com/noir-lang/noir)
[Zokrates](https://zokrates.github.io/)
[PyTorch](https://github.com/pytorch/pytorch)
[Picus](https://github.com/Veridise/Picus)
[EcneProject](https://github.com/franklynwang/EcneProject)
[Circomspect](https://github.com/trailofbits/circomspect)
