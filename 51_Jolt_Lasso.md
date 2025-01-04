# ETAAcademy-ZKMeme: 51. Jolt & Lasso

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>51. Jolt & Lasso</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Jolt_Lasso</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Exploring Zero-Knowledge Virtual Machines (ZKVMs): Innovations in Jolt, Lasso Lookup, and Efficient Memory Checking

Zero-Knowledge Virtual Machines (ZKVMs) are advanced computational frameworks designed to enable privacy-preserving computations on blockchain platforms using zero-knowledge proofs. The integration of Zero-Knowledge Virtual Machines (ZKVMs) such as Valida, Nexos, and ZKwasm, are focused on improving scalability, multi-chain support, and developer experience, helping to drive the adoption of zero-knowledge technology in blockchain ecosystems. Jolt is one such ZKVM optimized for the RISC-V architecture, utilizing techniques like Lasso Lookup, offline memory checking, and Rank-1 Constraint Systems (R1CS) to enhance proof generation efficiency and memory verification.

## Virtual Machines (VMs) and ZKVMs

Virtual Machines (VMs) act as interpreters that execute instructions, much like a CPU. They typically operate by executing bytecode and are designed to provide cross-platform compatibility, fulfilling the mantra "write once, run anywhere." Common examples of VMs include the **Java Virtual Machine (JVM)**, **Python Virtual Machine**, and **WebAssembly Virtual Machine (WASM VM)**. In blockchain, zero-knowledge virtual machines (ZKVMs) introduce the added benefit of privacy and verifiability through zero-knowledge proofs.

### ZKVM Workflow from a User’s Perspective

1. **Writing the Program**: Developers write the program in a high-level language such as Rust or Solidity.
2. **Compilation**: The program is compiled into bytecode optimized for the ZKVM.
3. **Execution**: The bytecode is executed within the ZKVM, producing an **execution trace** of the computations.
4. **Proof Generation**: Using the execution trace and predefined circuits, a **zero-knowledge proof** (ZK proof) is generated.
5. **Proof Verification**: The proof is verified, often on-chain, to ensure the correctness of the computation without revealing sensitive data.

### Components of ZKVM

ZKVMs can be divided into those major components:

1. **Frontend**: The frontend consists of the virtual machine and its instruction set, responsible for interpreting and executing programs. In a ZKVM, the frontend generates the necessary computational model for proof generation.

2. **Backend**: The backend handles proof generation and verification. It translates the VM’s computation into mathematical representations used in zero-knowledge proof systems.

3. **Arithmeticization**

This process converts computational logic into mathematical constraints. Commonly used proof systems include:

- **Halo2**: Optimized for recursive and adaptive circuits.
- **Plonk**: A general-purpose, efficient zero-knowledge proof system.
- **STARK**: Transparent and scalable, requiring no trusted setup.
- **AIR**: Abstracts computation as iterative steps for efficient validation.
- **HyperPlonk**: An improved version of Plonk with enhanced efficiency.
- **MLE-based**: Leverages Maximum Likelihood Estimation for proof generation.

4. **Polynomial Commitment Schemes (PCS)**

Proof systems often use **PCS** to aggregate data and improve scalability. Examples include:

- **KZG Commitments**: Efficient for polynomial verification.
- **Inner Product Arguments (IPA)**: Optimized for vector dot product proofs.
- **FRI**: A highly efficient method for reducing verification complexity.

5. **Recursive Proofs**

Recursive proofs compress multiple computations into a single zero-knowledge proof. This approach boosts efficiency by allowing smaller proofs to be integrated recursively into larger ones.

6. **Snarkified Verifiers**

This involves converting traditional computational verifiers into zero-knowledge proof-compatible verifiers, enabling private verification of results. These challenges highlight the difficulty of building efficient ZKEVMs while maintaining compatibility with Ethereum.

### Emerging ZKVM Projects

Several innovative ZKVM projects are shaping the future of zero-knowledge technology:

1. **Jolt**: Aims to reduce computational complexity for ZK proofs.
2. **Valida**: Focuses on optimizing ZKEVM operations and developer experience.
3. **Nexos**: Enhances ZK proofs for multi-chain and high-performance applications.
4. **SP1**: Prioritizes speed and efficiency in proof generation and verification.
5. **ZKwasm**: Combines WASM’s flexibility with zero-knowledge proof capabilities for cross-chain compatibility.

---

## **Overview of Jolt Technology**

Jolt is a zero-knowledge virtual machine (zkVM) tailored for the RISC-V architecture, developed by a16z. Its primary objective is to validate the correctness of virtual machine programs through zero-knowledge proofs. Highly modular in nature, Jolt leverages Lasso lookup, R1CS (Rank-1 Constraint Systems), and offline memory checking to achieve efficient program verification. However, challenges such as memory bottlenecks arise when handling complex operations like elliptic curve arithmetic or signature verification.

**Key Steps in Jolt's zkVM Workflow**

1. **Compilation**: Programs are compiled into RISC-V binary code.

2. **Execution**: Jolt’s RISC-V emulator runs the binary, generating execution traces.

3. **Proof Generation**: The prover uses these traces to produce a zero-knowledge proof, demonstrating the correctness of the program execution.

4. **Proof Verification**: The verifier validates the proof using Jolt's verification framework.

**Core Components of Jolt Proofs**

**Lasso** handles localized validation of simple operations, such as additions and comparisons.**Offline Memory Checking** optimizes memory verification across the program, minimizing proof-generation costs. **R1CS** builds global constraints to validate the overall program logic.

1. **Lasso Lookup**: This is a table-based lookup technique for efficiently verifying simple operations, such as equality and comparison. For instance, in the BEQ instruction, `EQ[x || y] = 1` verifies whether x and y are equal.

2. **Offline Memory Checking**: Ensures memory consistency during random access in zero-knowledge proofs. This approach uses commitments to verify memory access correctness, significantly reducing computational overhead during proof generation.

3. **R1CS (Rank-1 Constraint System)**: A universal constraint representation used in zkSNARKs. Jolt employs R1CS to encode program logic as polynomial equations, facilitating the efficient verification of intricate computation logic.

#### Jolt Instruction Lookup

Instruction correctness in Jolt relies on lookup tables and arguments to verify operations efficiently.

1. **Lasso Protocol:** Utilizes tables and index matrices to validate results. Multi-linear extensions (MLE) and sum-check protocols optimize this process.

2. **Optimization:** Large lookup tables are split into smaller subtables. Subtable values are committed as polynomials, enabling scalable validation.

3. **Verification:** A combination of commitments and offline memory checks ensures consistency between prover-submitted polynomials and lookup tables.

#### Offline Memory Checking in Jolt

Offline memory checking is critical for validating memory operations, including instruction lookup, bytecode verification, and general memory read/write.

1. **Core Mechanism:** Verifies consistency between initial, final, read, and write memory states using hash-based commitments. These are validated with sum-check protocols.

2. **Applications:**

   - **Instruction Lookup:** Memory checks validate addresses and values.
   - **Bytecode Verification:** Ensures execution traces align with bytecode using extended hash functions.
   - **Memory Read/Write:** Validates memory access across registers, RAM, and I/O.

3. **Optimization:** Memory validation leverages multi-variable polynomials and recursive protocols, reducing computational and storage overhead.

#### Rank-1 Constraint System (R1CS) in Jolt

Jolt employs a Rank-1 Constraint System (R1CS) as a unified verification mechanism to ensure seamless cooperation among the various components of its virtual machine (VM). Building on the foundation of instruction verification and memory operation validation, Jolt integrates R1CS to address the following challenges:

1. **Correct Execution Sequence of Instructions**: Ensuring that instructions are executed in the correct order according to the bytecode address and program counter (PC), including jumps and branching operations.
2. **Accurate Operand Indexing in Instruction Lookup**: Verifying that the instruction lookup references the correct operand indices (e.g., `rs1`, `rs2`, or `imm`).
3. **Consistency Between Instruction Lookup and Memory Operations**: Ensuring that the inputs and outputs of the instruction lookup align with the memory's read and write values.

The core function of R1CS in Jolt is to connect all VM processes—fetching, decoding, and executing instructions—through constraints, ensuring the correctness of each step. The complete set of R1CS constraints is available in the `constraints.rs` file.

**Fundamentals of R1CS**

R1CS is represented by three matrices, $A$, $B$, and $C$, in $\mathbb{F}^{m \times m}$. Its satisfiability is determined by the equation: $(A \cdot z) \circ (B \cdot z) - (C \cdot z) = 0$

Here, $z$ is the witness, which the prover must supply to demonstrate knowledge of a $z$ that satisfies the equation. Each individual constraint can be expressed as: $a \cdot b - c = 0$

In Jolt, the input for R1CS is derived from the VM's execution trace. Each step in this trace includes details about bytecode, instruction lookups, and memory operations. Additionally, **bitflags** are introduced to dynamically activate specific constraints, allowing precise control.

**Bitflags**

Bitflags in Jolt consist of 64 bits divided into two categories:

1. **Circuit Flags**: Multi-bit flags that describe the attributes of instructions (e.g., whether the instruction involves a jump or requires an update to the PC).
2. **Instruction Flags**: Single-bit flags (one-hot encoded) indicating specific instruction types. For example, the addition instructions `ADD` and `ADDI` share a common instruction flag.

**Example: Verifying the Consistency of `rd` Values**

To verify that the value written to `rd` matches the output of the instruction lookup, an auxiliary variable is defined as: $\text{condition} \gets rd\_bytecode \cdot flag_7$

The following constraint is added to ensure the validity of the condition: $rd\_bytecode \cdot flag_7 - \text{condition} = 0$

The main constraint then enforces consistency: $\text{condition} \cdot (rd\_write - lookup\_output) = 0$

Where:

- `rd_write` is the value written to memory.
- `lookup_output` is the result of the instruction lookup.

This guarantees that the value written to `rd` is identical to the output of the instruction lookup.

**Satisfiability of R1CS**

Once the R1CS constraints and witness are constructed, Jolt uses **Spartan** to prove their satisfiability. Spartan efficiently generates proofs by transforming the R1CS instance into a **Sumcheck protocol** instance. To ensure the integrity of witness computations, Spartan employs a multivariate polynomial commitment scheme, such as Hyrax or HyperKZG. This structured approach enables Jolt to establish a robust verification system, ensuring the correctness and reliability of its virtual machine operations.

---

## **Lasso Lookup: A Table-Based Verification Technique in Zero-Knowledge Proofs**

**Lasso Lookup** is a table-based verification technique designed to validate whether query vectors appear in a table, commonly used in zero-knowledge proofs (ZKPs). This method distinguishes between two types of query relationships: **Unindexed Lookup Arguments** and **Indexed Lookup Arguments**.

**1. Unindexed Lookup Argument**

In this approach, the goal is to verify that each element of a query vector $f$ exists within a table vector $t$, without considering the specific positions of the elements in $t$. The relationship to be verified can be expressed as:

$\forall i \in [0, m), f_i \in t$

This is particularly useful for operations where the order of table rows is irrelevant. For example, in an XOR operation table, the rows can be rearranged without affecting the results, as the computation is independent of row order.

**2. Indexed Lookup Argument**

For indexed lookups, the relationship between the query vector and the table is position-dependent. Here, a single-row table is used to represent query results, where the rows are indexed to encode the input operands for operations. For instance, consider an XOR operation table where each row corresponds to a specific operation:

- Row 0 encodes $00 \oplus 00 = 00$, with the row index $0$ represented in binary as $0000$. The high bits ($00$) denote input $A$, and the low bits ($00$) denote input $B$.
- Row 5 encodes $01 \oplus 01 = 00$, with the row index $5$ represented as $0101$, where $01$ corresponds to both inputs $A$ and $B$.

The indexed lookup argument ensures the relationship:

$\forall i \in [0, m), f_i = t_{a_i}$

Where:

- $f_i$ represents the query vector element at index $i$.
- $t$ is the table vector.
- $a_i$ denotes the index of the table row corresponding to $f_i$.

**Implementation of Indexed Lookup Arguments** involves commitments to three vectors: the table vector $t$, the query vector $f$, and the index vector $a$. The proof verifies the relationship:

$R_{indexed-lkup} = (cm(\vec{t}), cm(\vec{f}), cm(\vec{a}); \vec{t}, \vec{f}, \vec{a}) \mid \forall i \in [0, m), f_i = t_{a_i}$

**3. Methods for Constructing Indexed Lookup Proofs**

Two primary methods are used to construct proofs for indexed lookups:

1. **Using Additive Homomorphism**: If the table supports additive homomorphism (e.g., in protocols like Plookup or Caulk/Caulk+), an index column is added to the table, and a new table is generated by applying a random linear combination of the table and index columns.

2. **Without Additive Homomorphism**: If additive homomorphism is not supported, a large value $\kappa$ is introduced. The index column is merged with the table column, and the prover ensures that each table element is less than $\kappa$.

**4. Protocols and Table Types in Lasso**

The Lasso framework proposes four indexed lookup proof protocols, each designed to optimize verification efficiency:

1. **Offline Memory Check Proofs**: Validate the lookup relationship directly.
2. **Spark Protocol**: Uses matrix selectors to establish lookup relationships.
3. **Surge Protocol**: Decomposes large tables into smaller sub-tables to improve efficiency.
4. **Generalized Lasso Protocol**: Combines sparse and dense checks, leveraging lazy computation of table entries during queries to reduce proof complexity.

Lasso supports three types of tables:

1. **Unstructured but Small Tables**: Suitable for tables with no inherent structure but manageable size.
2. **Decomposable Tables**: Tables that can be split into smaller, independent components.
3. **Non-decomposable but MLE-Structured Tables**: Tables that cannot be split but have a structure conducive to Maximum Likelihood Estimation (MLE).

### Offline Memory Checking: A New Perspective on Indexed Lookup Arguments

While there are existing solutions like Plookup, Caulk/Caulk+, Baloo, Logup, and CQ that can directly handle lookup arguments, **Offline Memory Checking (OMC)** offers a novel and intuitive approach to validating indexed lookup relationships. It models the process of value lookups as memory reads in a virtual machine (VM), ensuring the validity of every read operation.

**Key Concepts of Offline Memory Checking**

**1. Problem Description**
Given a memory table `t` and an index vector `a`, Offline Memory Checking aims to verify the indexed lookup relationship $f_i = t_{a_i}$. This involves ensuring that each element of the query vector $f$ matches the value in the table $t$ at the position specified by $a$.

**2. Memory-in-the-Head Perspective**
The lookup process is reimagined as a VM executing memory read operations. Each index in the vector `a` corresponds to a memory access, treated as a sequential state transition in the VM.

- **VM Execution Sequence:** The VM starts from an initial state $S_0$ and transitions through a series of read operations $R_i$, ending in a final state $S_m$. Each read operation logs two events:
  - **Read Log ($R$)**: Captures the state of the memory at the time of the read.
  - **Write Log ($W$)**: Records the updated state of the memory, where a counter associated with the memory location is incremented.

**3. VM Memory State Representation**
The memory state $S$ is represented as a set of tuples $(i, v, c)$, where:

- $i$: Memory address.
- $v$: Memory value.
- $c$: Counter tracking the number of times the memory location has been read.

Each read operation generates a pair of logs:

- $R_j$: Records the memory state before the read.
- $W_j$: Records the memory state after the read, with $c$ incremented by one.

**4. Verification Conditions**
To validate the correctness of memory access, the VM execution must satisfy four key conditions:

1. **Initial State Consistency:** The initial memory state $S_0$ aligns with the table $t$, with all counters set to zero.
2. **Read Accuracy:** Each read operation retrieves the correct value from memory.
3. **Log Consistency:** For every read log $R_j$, there is a corresponding write log $W_j$, ensuring $W_j = R_j + 1$ for the counter value.
4. **Value Consistency:** Each read value matches the corresponding memory content, and subsequent reads are verified against the previous writes.

By adhering to these conditions, the verifier ensures that all memory reads are valid and cannot be falsified. The prover demonstrates the integrity of the initial state and sequentially proves the consistency of each read log.

**Constructing a Lookup Argument Protocol**

The protocol for verifying memory operations involves encoding memory states and logs as polynomials and proving their relationships.

**1. Matrix and Polynomial Encoding**

- Memory states ($S_0$ and $S_n$) and logs ($R_j$ and $W_j$) are represented as matrices.
- These matrices are encoded as polynomials, such as $R_c(X)$ for read logs and $W_c(X)$ for write logs, with constraints ensuring $W_c(X) = R_c(X) + 1$.

**2. Constraints**
The protocol enforces constraints to ensure:

- The initial state and write logs align with the final state and read logs.
- Multiset equivalence between the initial and final memory states, combined with the logs.

**3. Protocol Flow**

- **First Round:** The prover generates VM execution logs and commits to the counter values and memory states.
- **Second Round:** The verifier sends random challenges $\beta$ and $\gamma$. The prover calculates updated logs and state commitments and uses **Grand Product Argument** to prove consistency: $\prod_{i=0}^{n-1} S_i^{init} \cdot \prod_{j=0}^{m-1} R_j = \prod_{i=0}^{n-1} S_i^{final} \cdot \prod_{j=0}^{m-1} W_j$
- The verifier checks the commitments and ensures all equations hold.

**Comparison with Plookup**

In Plookup, the prover introduces an intermediate vector $\vec{s}$ to demonstrate multiset equivalence between $\vec{f}$ and $\vec{t}$: $\vec{s} = \vec{f} \cup \vec{t}$. This approach, while effective, requires additional sorting steps for $\vec{s}$, which can be computationally expensive.

**Advantages of Offline Memory Checking**

1. **Counter-Based Validation:** OMC eliminates the need for sorting by introducing counters $\vec{c}$, which are small, predictable values (e.g., $0, 1, \dots, m-1$).
2. **Simplified Verifier Workload:** The verifier can automatically generate necessary values, reducing dependency on the prover.

While Plookup provides a straightforward framework, OMC reduces computational overhead by leveraging counters and matrix commitments, offering a more efficient alternative for verifying indexed lookups.

### Sparse Vectors and the Spark Protocol

The Spark protocol introduces a polynomial commitment scheme that leverages the structure of sparse polynomials to significantly enhance the efficiency of the prover. First proposed within the [Spartan] proof system, the Spark protocol serves as a foundation for Lasso, an extended framework that further refines sparse vector handling.

A typical polynomial commitment scheme involves two primary phases:

1. **Commitment Phase:**  
   A commitment to the target polynomial $g$ is generated:

   $cm(g) \leftarrow \text{PCS.Commit}(g)$

2. **Evaluation Proof Phase:**  
   The prover computes the value $v$ of the polynomial $g$ at a specific point $u$ and generates a proof $\pi_{g,v}$:

   $\pi_{g,v} \leftarrow \text{PCS.Eval}(cm(g), u, v; g)$  
   The verifier then validates:

   $\text{Accept/Reject} \leftarrow \text{PCS.Verify}(cm(g), u, v, \pi_v)$

For sparse polynomials—those with only a few nonzero terms—traditional methods scale in complexity with the degree $N$ of the polynomial. Spark addresses this inefficiency by encoding sparse polynomials in a dense form, reducing computational overhead.

**Sparse Vectors in Dense Encoding**

A sparse vector can be represented as: $\vec{g} = (g_0, g_1, \dots, g_{N-1})$, where only $m$ of the $N$ entries are nonzero. Its dense representation is: $\text{DenseRepr}(\vec{g}) = \big((k_0, h_0), (k_1, h_1), \dots, (k_{m-1}, h_{m-1})\big)$. Here, $\vec{k}$ identifies the positions of the nonzero elements, and $\vec{h}$ stores their corresponding values.

Using this representation, evaluating the polynomial $g(X)$ at a set of points $\vec{u}$ is efficiently computed as: $\tilde{g}(\vec{u}) = \sum_{i=0}^{m-1} h_i \cdot \lambda_{k_i}$, where $\lambda_{k_i} = \tilde{eq}_{k_i}(\vec{u})$ represents the Lagrange basis function values at the specified points. This optimization reduces evaluation complexity from $O(N)$ to $O(m)$.

**Auxiliary Vectors and the Sumcheck Protocol**

To prove that $\tilde{g}(\vec{u}) = v$, the Spark protocol introduces an auxiliary vector:

$\vec{e} = (\lambda_{k_0}, \lambda_{k_1}, \dots, \lambda_{k_{m-1}})$

This vector satisfies the equation: $v = \sum_{i=0}^{m-1} h_i \cdot e_i$

**Proof Workflow**

1. **Sumcheck Reduction:**  
   The prover reduces the summation verification to an equation:
   
   $v' = h(\vec{\rho}) \cdot e(\vec{\rho})$  
   Here, $\vec{\rho}$ represents a folding point chosen by the verifier. The prover generates a proof for this equation, reducing the complexity to $O(m)$.

3. **Validation of Auxiliary Vector:**  
   The correctness of $\vec{e}$, where each $e_i$ corresponds to $\lambda_{k_i}$, is proven using a memory-checking approach. In this context, $\vec{\lambda}$ (calculated from $\vec{u}$) is treated as public memory. The prover demonstrates that $\vec{e}$ is derived correctly from $\vec{\lambda}$.

**Memory Checking: Linking Spark to Lookup Arguments**

Memory checking applies Indexed Lookup Argument principles to verify that the auxiliary vector $\vec{e}$ correctly derives its elements from the publicly accessible vector $\vec{\lambda}$. The verifier ensures: $\forall i \in [0, m), \; e_i = \lambda_{k_i}$. Here, $\lambda_i = \tilde{eq}_i(\vec{u})$, computed directly from the evaluation points $\vec{u}$.

To facilitate efficient verification, the Spark protocol employs the following steps:

1. **Public Commitment to Auxiliary Data:**  
   The prover commits to $\vec{\lambda}$ with $cm(\vec{\lambda})$, ensuring it encapsulates the evaluation data:

   $\lambda_i = \tilde{eq}_i(\vec{u})$

2. **Prover-Generated Proofs:**  
   The prover generates a commitment $cm(\vec{\lambda})$ and proves its correctness to the verifier using the memory-checking framework. This ensures that the auxiliary data aligns with the prescribed evaluation points.

### Efficient Sparse Query Proofs for Oversized Tables

The **Lasso protocol** addresses the challenge of proving sparse queries over oversized tables by leveraging **Sparse Polynomial Commitment** and the **Sumcheck Protocol**. This approach significantly optimizes the handling of large-scale sparse matrices, which are common in applications requiring efficient zero-knowledge proofs and cryptographic verifications.

**Sparse Polynomial Commitment and the Sumcheck Protocol**

Sparse polynomial commitment is a technique designed to efficiently handle proofs over sparse matrices. By encoding sparse data structures into dense representations, this method reduces computational and storage overhead without sacrificing accuracy or correctness.

The Sumcheck Protocol is a distributed verification technique that simplifies polynomial summation relationships into single-point verifications. This stepwise reduction makes it possible to validate complex relationships in sparse matrices with lower computational complexity.

**Indexed Lookup Argument**

Lasso introduces a straightforward Indexed Lookup Argument using Multivariate Lagrange Interpolation (MLE) polynomials. The relationship between the lookup vector and the table vector is expressed as:

$\sum_{y \in \{0,1\}^{\log N}} M^\sim(\vec{X}, \vec{y}) \cdot t^\sim(\vec{y}) = f^\sim(\vec{X})$

Using the Sumcheck Protocol, the summation is iteratively reduced to a single-point verification, drastically lowering the complexity, particularly for sparse matrices $M$.

**Table Decomposition Strategy**

For oversized tables, Lasso employs **table decomposition**, a technique that breaks a large sparse matrix into smaller, more manageable sub-tables. For example, a 32-bit RangeCheck table can be decomposed into:

- **Four 8-bit RangeCheck tables** or
- **Two 16-bit RangeCheck tables**

**Core Idea of Decomposition**

Table decomposition reduces the computational and storage complexity by leveraging the sparsity and decomposability of polynomials. This is particularly effective for validating lookup relationships, such as RangeCheck tables.

**RangeCheck Table Example**

RangeCheck tables verify whether a value $x$ lies within a specific range $[0, 2^k)$. For instance:

$T_{\text{range}, k} = (0, 1, 2, \ldots, 2^k - 1)$

can be decomposed into smaller sub-tables:

$T_{\text{range}, k}[i \cdot 2^{k/2} + j] = t_{\text{range}, k/2}[i] \cdot 2^{k/2} + t_{\text{range}, k/2}[j]$

For $k = 4$: $T_{\text{range, 4}} = [0, 1, \dots, 15]$ is decomposed into $t_{\text{range}, 2} = (0, 1, 2, 3)$.

This decomposition reduces a high-dimensional problem into low-dimensional sub-table computations, minimizing the prover's computational load. The complexity of generating proofs for a decomposed table is $O(m + m \cdot N^{1/c})$, where $c$ is the decomposition factor. Even for non-sparse tables, this method optimizes most common operations like XOR and AND.

**Efficient RangeCheck Lookup Argument**

Lasso enhances the efficiency of RangeCheck lookups using MLE polynomials, such as $\tilde{t_{range2}}(\vec{X})$ and $\tilde{T_{range4}}(\vec{X})$, to define relationships: $\tilde{T_{range4}}(X_0, X_1, X_2, X_3) = 4 \cdot \tilde{t_{range2}}(X_0, X_1) + \tilde{t_{range2}}(X_2, X_3)$

This structure enables efficient lookup arguments and reduces verification costs. Unlike traditional approaches that require precomputing the entire table, this method allows dynamic proof generation for large tables, making it ideal for applications in zero-knowledge proofs where flexibility is key.

**Optimized Proof Generation with Sumcheck**

Using the Sumcheck Protocol, the proof of a summation relationship is reduced to:

$v' = \Big( 4 \cdot \tilde{e}^{(x)}(\vec{\rho}) + \tilde{e}^{(y)}(\vec{\rho}) \Big) \cdot \tilde{eq_i}(\vec{\rho}, (r_0, r_1, r_2, r_3))$, where $\tilde{eq_i}$ is an indicator polynomial for verifying if the current $i$-th position is valid, and $\vec{\rho}$ is a random challenge vector generated by the verifier. The auxiliary vectors $\vec{e}^{(x)}$ and $\vec{e}^{(y)}$ can then be validated using **Offline Memory Checking**.

### Decomposing Common Tables Using the Lasso Framework

The **Lasso framework** provides a powerful method for decomposing complex tables, particularly those used to represent the **RISC-V instruction computation process**. By breaking down large logical tables, such as equality checks, comparison operations, and shift calculations, into smaller sub-tables, the framework enhances computational efficiency and avoids the storage challenges associated with oversized tables. This approach enables precise and efficient computation.

**EQ Table (Equality Check)**

The **EQ table** determines whether two numbers are equal. For instance, in a 2-bit example, the table returns 1 if numbers **A** and **B** are equal and 0 otherwise.

For larger bit-widths (e.g., $W$), the EQ table is decomposed into multiple smaller **EQ sub-tables**. Each sub-table processes a segment of the bits, and the results from these sub-tables are multiplied to produce the final equality result.

**Example:**

If $W = 4$, the EQ table is divided into two 2-bit sub-tables. Each sub-table verifies equality for its segment, and the final EQ result is obtained by multiplying the results from the sub-tables.

**LTU Table (Less-Than Relationship Check)**

The **LTU table** checks whether $X < Y$. This operation involves scanning the binary representation of $X$ and $Y$ in parallel, comparing each bit from the most significant to the least significant. The first differing bit determines the outcome of $X < Y$.

To construct the LTU table, auxiliary tables $\text{LTU}_i$ are introduced. Each $\text{LTU}_i$ represents the comparison result for the $i$-th bit and computes whether $X$ and $Y$ differ at that position. The final $\text{LTU}(W)$ result is obtained by aggregating the results from all $\text{LTU}_i$ tables.

**SLL Table (Shift Left Logical Operation)**

The **SLL table** represents the **Shift Left Logical** operation, where a binary vector is shifted to the left by a specified number of positions.

To handle variable shift amounts efficiently, the SLL operation is broken into sub-tables, each corresponding to a specific shift amount. The appropriate sub-table is selected based on the shift value. The results from these sub-tables are then aggregated to compute the final SLL operation.

During the shift process, overflow bits are discarded, retaining only the significant bits required for the computation.

### Generalized Lasso: Extending Efficiency for Large-Scale Table Queries

The **Generalized Lasso framework** extends the classic Lasso approach by enabling the direct handling of ultra-large tables without splitting them into smaller components. This approach leverages the Maximum Likelihood Estimation (MLE) structure and sparsity of table data to significantly enhance the efficiency of querying large tables. It also reduces the computational overhead for the Prover in the **Sumcheck protocol**, making it feasible to efficiently verify massive data tables, such as those encountered in cryptographic algorithms.

This method is particularly suited for scenarios involving complex computations over large data structures, where traditional approaches might be computationally prohibitive. A key feature of the Generalized Lasso is its use of the **Indexed Lookup Argument**, which enables the query process to be represented by the following equation:

$\tilde{f}(\vec{X}) = \sum_{\vec{y} \in \{0,1\}^{\log N}} \tilde{M}(\vec{X}, \vec{y}) \cdot \tilde{t}(\vec{y}),$

where:

- $\vec{f}$ represents the query vector,
- $\vec{t}$ is the table vector,
- $\tilde{M}$ is a table selection matrix, where each row corresponds to a unit vector.

The Prover and Verifier need to prove that each query $f_i$ matches a specific table entry $t_j$.

The **Sumcheck protocol** is used by the Verifier to confirm the correctness of queries through a series of challenges involving a random vector $\vec{r}$. The verification is simplified using the following equation:

$\sum_{\vec{y} \in \{0,1\}^{\log N}} \tilde{M}(\vec{r}, \vec{y}) \cdot \tilde{t}(\vec{y}).$

The protocol operates iteratively, where the Prover computes and sends polynomial evaluations while the Verifier challenges the results step by step. This multi-round process ensures correctness with reduced computational requirements for the Verifier.

One of the key advantages of the Generalized Lasso is its ability to exploit the sparsity of the selection matrix $\tilde{M}$. This allows the Prover to significantly reduce computational costs during the Sumcheck protocol. Instead of summing $N$ terms, the Prover needs to compute only $m$ terms, as most terms in the sparse polynomial are zero. This optimization saves both time and resources.

#### **Condensation for Enhanced Efficiency**

The **Condensation technique** is a specialized optimization within the sparse-dense Sumcheck protocol. It divides the Sumcheck computation into segments and leverages precomputed auxiliary vectors to reduce the Prover’s workload. By precomputing these vectors, the Condensation method reduces the original complexity of $O(m \cdot \log N)$ to $O(c \cdot m)$, where $c = \frac{\log N}{\log m}$. This results in a significant improvement in computational efficiency.

**Example: Binary Index-Based Table Entries**

Consider a table $t$, where each entry $t_i$ can be derived from the binary representation of its index $i = (i_0, i_1, \dots, i_{s-1})$ using the following linear combination:

$t(i_0, \dots, i_{s-1}) = d_0 i_0 + d_1 i_1 + \dots + d_{s-1} i_{s-1}.$

This structure associates each table entry with the binary bits of its index, allowing efficient computation of table values. For a given partial index, the remaining values can be calculated in constant time, further enhancing efficiency.

**Accelerating Sumcheck with Precomputed Vectors**

The Condensation method introduces two auxiliary vectors, $q$ and $z$, to optimize the initial $\log m$ rounds of the Sumcheck protocol. These vectors are defined as follows:

$q_k = \sum_{y \in \text{extend}(k, \log m, \log N)} \tilde{u}(\vec{y}) \cdot \tilde{t}(\vec{y}),$

$z_k = \sum_{y \in \text{extend}(k, \log m, \log N)} \tilde{u}(\vec{y}),$

where $\text{extend}(k, \log m, \log N)$ represents the set of binary strings with the first $\log m$ bits fixed to $k$. These vectors allow the Prover to precompute critical values that can be reused during later rounds, eliminating redundant calculations.

**Tree-Based Representation**

The summation process can be visualized as a binary tree of depth $\log N$, where each leaf corresponds to a term $u(b) \cdot t(b)$. The $q$ vector represents the computed values at a specific level of this tree, enabling the Prover to efficiently calculate results for each round without recalculating individual terms.

<details><summary><b> Code </b></summary>

<details><summary><b> memory_checking.rs </b></summary>

```rust

pub trait MemoryCheckingProver<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Self: Sync,
{
    type ReadWriteGrandProduct: BatchedGrandProduct<F, PCS, ProofTranscript> + Send + 'static =
        BatchedDenseGrandProduct<F>;
    type InitFinalGrandProduct: BatchedGrandProduct<F, PCS, ProofTranscript> + Send + 'static =
        BatchedDenseGrandProduct<F>;

    type Polynomials: StructuredPolynomialData<DensePolynomial<F>>;
    type Openings: StructuredPolynomialData<F> + Sync + Initializable<F, Self::Preprocessing>;
    type Commitments: StructuredPolynomialData<PCS::Commitment>;
    type ExogenousOpenings: ExogenousOpenings<F> + Sync = NoExogenousOpenings;

    type Preprocessing = NoPreprocessing;

    /// The data associated with each memory slot. A triple (a, v, t) by default.
    type MemoryTuple: Copy + Clone = (F, F, F);

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::prove_memory_checking")]
    /// Generates a memory checking proof for the given committed polynomials.
    fn prove_memory_checking(
        pcs_setup: &PCS::Setup,
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        jolt_polynomials: &JoltPolynomials<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> MemoryCheckingProof<F, PCS, Self::Openings, Self::ExogenousOpenings, ProofTranscript> {
        let (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ) = Self::prove_grand_products(
            preprocessing,
            polynomials,
            jolt_polynomials,
            opening_accumulator,
            transcript,
            pcs_setup,
        );

        let read_write_batch_size =
            multiset_hashes.read_hashes.len() + multiset_hashes.write_hashes.len();
        let init_final_batch_size =
            multiset_hashes.init_hashes.len() + multiset_hashes.final_hashes.len();

        // For a batch size of k, the first log2(k) elements of `r_read_write`/`r_init_final`
        // form the point at which the output layer's MLE is evaluated. The remaining elements
        // then form the point at which the leaf layer's polynomials are evaluated.
        let (_, r_read_write_opening) =
            r_read_write.split_at(read_write_batch_size.next_power_of_two().log_2());
        let (_, r_init_final_opening) =
            r_init_final.split_at(init_final_batch_size.next_power_of_two().log_2());

        let (openings, exogenous_openings) = Self::compute_openings(
            preprocessing,
            opening_accumulator,
            polynomials,
            jolt_polynomials,
            r_read_write_opening,
            r_init_final_opening,
            transcript,
        );

        MemoryCheckingProof {
            multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
            openings,
            exogenous_openings,
        }
    }

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::prove_grand_products")]
    /// Proves the grand products for the memory checking multisets (init, read, write, final).
    fn prove_grand_products(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        jolt_polynomials: &JoltPolynomials<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
        pcs_setup: &PCS::Setup,
    ) -> (
        BatchedGrandProductProof<PCS, ProofTranscript>,
        BatchedGrandProductProof<PCS, ProofTranscript>,
        MultisetHashes<F>,
        Vec<F>,
        Vec<F>,
    ) {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();

        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        let (read_write_leaves, init_final_leaves) =
            Self::compute_leaves(preprocessing, polynomials, jolt_polynomials, &gamma, &tau);
        let (mut read_write_circuit, read_write_hashes) =
            Self::read_write_grand_product(preprocessing, polynomials, read_write_leaves);
        let (mut init_final_circuit, init_final_hashes) =
            Self::init_final_grand_product(preprocessing, polynomials, init_final_leaves);

        let multiset_hashes =
            Self::uninterleave_hashes(preprocessing, read_write_hashes, init_final_hashes);
        Self::check_multiset_equality(preprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript(transcript);

        let (read_write_grand_product, r_read_write) = read_write_circuit.prove_grand_product(
            Some(opening_accumulator),
            transcript,
            Some(pcs_setup),
        );
        let (init_final_grand_product, r_init_final) = init_final_circuit.prove_grand_product(
            Some(opening_accumulator),
            transcript,
            Some(pcs_setup),
        );

        drop_in_background_thread(read_write_circuit);
        drop_in_background_thread(init_final_circuit);

        (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        )
    }

    fn compute_openings(
        preprocessing: &Self::Preprocessing,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        polynomials: &Self::Polynomials,
        jolt_polynomials: &JoltPolynomials<F>,
        r_read_write: &[F],
        r_init_final: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self::Openings, Self::ExogenousOpenings) {
        let mut openings = Self::Openings::initialize(preprocessing);
        let mut exogenous_openings = Self::ExogenousOpenings::default();

        let eq_read_write = EqPolynomial::evals(r_read_write);
        polynomials
            .read_write_values()
            .par_iter()
            .zip_eq(openings.read_write_values_mut().into_par_iter())
            .chain(
                Self::ExogenousOpenings::exogenous_data(jolt_polynomials)
                    .par_iter()
                    .zip_eq(exogenous_openings.openings_mut().into_par_iter()),
            )
            .for_each(|(poly, opening)| {
                let claim = poly.evaluate_at_chi_low_optimized(&eq_read_write);
                *opening = claim;
            });

        let read_write_polys: Vec<_> = [
            polynomials.read_write_values(),
            Self::ExogenousOpenings::exogenous_data(jolt_polynomials),
        ]
        .concat();
        let read_write_claims: Vec<_> =
            [openings.read_write_values(), exogenous_openings.openings()].concat();
        opening_accumulator.append(
            &read_write_polys,
            DensePolynomial::new(eq_read_write),
            r_read_write.to_vec(),
            &read_write_claims,
            transcript,
        );

        let eq_init_final = EqPolynomial::evals(r_init_final);
        polynomials
            .init_final_values()
            .par_iter()
            .zip_eq(openings.init_final_values_mut().into_par_iter())
            .for_each(|(poly, opening)| {
                let claim = poly.evaluate_at_chi_low_optimized(&eq_init_final);
                *opening = claim;
            });

        opening_accumulator.append(
            &polynomials.init_final_values(),
            DensePolynomial::new(eq_init_final),
            r_init_final.to_vec(),
            &openings.init_final_values(),
            transcript,
        );

        (openings, exogenous_openings)
    }

    /// Constructs a batched grand product circuit for the read and write multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::read_write_grand_product")]
    fn read_write_grand_product(
        _preprocessing: &Self::Preprocessing,
        _polynomials: &Self::Polynomials,
        read_write_leaves: <Self::ReadWriteGrandProduct as BatchedGrandProduct<
            F,
            PCS,
            ProofTranscript,
        >>::Leaves,
    ) -> (Self::ReadWriteGrandProduct, Vec<F>) {
        let batched_circuit = Self::ReadWriteGrandProduct::construct(read_write_leaves);
        let claims = batched_circuit.claimed_outputs();
        (batched_circuit, claims)
    }

    /// Constructs a batched grand product circuit for the init and final multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::init_final_grand_product")]
    fn init_final_grand_product(
        _preprocessing: &Self::Preprocessing,
        _polynomials: &Self::Polynomials,
        init_final_leaves: <Self::InitFinalGrandProduct as BatchedGrandProduct<
            F,
            PCS,
            ProofTranscript,
        >>::Leaves,
    ) -> (Self::InitFinalGrandProduct, Vec<F>) {
        let batched_circuit = Self::InitFinalGrandProduct::construct(init_final_leaves);
        let claims = batched_circuit.claimed_outputs();
        (batched_circuit, claims)
    }

    fn interleave<T: Copy + Clone>(
        _preprocessing: &Self::Preprocessing,
        read_values: &Vec<T>,
        write_values: &Vec<T>,
        init_values: &Vec<T>,
        final_values: &Vec<T>,
    ) -> (Vec<T>, Vec<T>) {
        let read_write_values = interleave(read_values, write_values).cloned().collect();
        let init_final_values = interleave(init_values, final_values).cloned().collect();

        (read_write_values, init_final_values)
    }

    fn uninterleave_hashes(
        _preprocessing: &Self::Preprocessing,
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        assert_eq!(read_write_hashes.len() % 2, 0);
        let num_memories = read_write_hashes.len() / 2;

        let mut read_hashes = Vec::with_capacity(num_memories);
        let mut write_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        let mut init_hashes = Vec::with_capacity(num_memories);
        let mut final_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            init_hashes.push(init_final_hashes[2 * i]);
            final_hashes.push(init_final_hashes[2 * i + 1]);
        }

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        }
    }

    fn check_multiset_equality(
        _preprocessing: &Self::Preprocessing,
        multiset_hashes: &MultisetHashes<F>,
    ) {
        let num_memories = multiset_hashes.read_hashes.len();
        assert_eq!(multiset_hashes.final_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.write_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.init_hashes.len(), num_memories);

        (0..num_memories).into_par_iter().for_each(|i| {
            let read_hash = multiset_hashes.read_hashes[i];
            let write_hash = multiset_hashes.write_hashes[i];
            let init_hash = multiset_hashes.init_hashes[i];
            let final_hash = multiset_hashes.final_hashes[i];
            assert_eq!(
                init_hash * write_hash,
                final_hash * read_hash,
                "Multiset hashes don't match"
            );
        });
    }

    /// Computes the MLE of the leaves of the read, write, init, and final grand product circuits,
    /// one of each type per memory.
    /// Returns: (interleaved read/write leaves, interleaved init/final leaves)
    fn compute_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        exogenous_polynomials: &JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> (
        <Self::ReadWriteGrandProduct as BatchedGrandProduct<F, PCS, ProofTranscript>>::Leaves,
        <Self::InitFinalGrandProduct as BatchedGrandProduct<F, PCS, ProofTranscript>>::Leaves,
    );

    /// Computes the Reed-Solomon fingerprint (parametrized by `gamma` and `tau`) of the given memory `tuple`.
    /// Each individual "leaf" of a grand product circuit (as computed by `read_leaves`, etc.) should be
    /// one such fingerprint.
    fn fingerprint(tuple: &Self::MemoryTuple, gamma: &F, tau: &F) -> F;
    /// Name of the memory checking instance, used for Fiat-Shamir.
    fn protocol_name() -> &'static [u8];
}



```

</details>

<details><summary><b> surge.rs </b></summary>

```rust
#[tracing::instrument(skip_all, name = "Surge::prove")]
    pub fn prove(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        generators: &PCS::Setup,
        ops: Vec<Instruction>,
    ) -> (Self, Option<ProverDebugInfo<F, ProofTranscript>>) {
        let mut transcript = ProofTranscript::new(b"Surge transcript");
        let mut opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript> =
            ProverOpeningAccumulator::new();
        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        let num_lookups = ops.len().next_power_of_two();
        let polynomials = Self::generate_witness(preprocessing, &ops);

        let mut commitments = SurgeCommitments::<PCS, ProofTranscript>::initialize(preprocessing);
        let trace_polys = polynomials.read_write_values();
        let trace_comitments =
            PCS::batch_commit_polys_ref(&trace_polys, generators, BatchType::SurgeReadWrite);
        commitments
            .read_write_values_mut()
            .into_iter()
            .zip(trace_comitments.into_iter())
            .for_each(|(dest, src)| *dest = src);
        commitments.final_cts = PCS::batch_commit_polys(
            &polynomials.final_cts,
            generators,
            BatchType::SurgeInitFinal,
        );

        let num_rounds = num_lookups.log_2();
        let instruction = Instruction::default();

        // TODO(sragss): Commit some of this stuff to transcript?

        // Primary sumcheck
        let r_primary_sumcheck = transcript.challenge_vector(num_rounds);
        let eq: DensePolynomial<F> = DensePolynomial::new(EqPolynomial::evals(&r_primary_sumcheck));
        let sumcheck_claim: F = Self::compute_primary_sumcheck_claim(&polynomials, &eq);

        transcript.append_scalar(&sumcheck_claim);
        let mut combined_sumcheck_polys = polynomials.E_polys.clone();
        combined_sumcheck_polys.push(eq);

        let combine_lookups_eq = |vals: &[F]| -> F {
            let vals_no_eq: &[F] = &vals[0..(vals.len() - 1)];
            let eq = vals[vals.len() - 1];
            instruction.combine_lookups(vals_no_eq, C, M) * eq
        };

        let (primary_sumcheck_proof, r_z, mut sumcheck_openings) =
            SumcheckInstanceProof::<F, ProofTranscript>::prove_arbitrary::<_>(
                &sumcheck_claim,
                num_rounds,
                &mut combined_sumcheck_polys,
                combine_lookups_eq,
                instruction.g_poly_degree(C) + 1, // combined degree + eq term
                &mut transcript,
            );

        // Remove EQ
        let _ = combined_sumcheck_polys.pop();
        let _ = sumcheck_openings.pop();
        opening_accumulator.append(
            &polynomials.E_polys.iter().collect::<Vec<_>>(),
            DensePolynomial::new(EqPolynomial::evals(&r_z)),
            r_z.clone(),
            &sumcheck_openings.iter().collect::<Vec<_>>(),
            &mut transcript,
        );

        let primary_sumcheck = SurgePrimarySumcheck {
            claimed_evaluation: sumcheck_claim,
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            E_poly_openings: sumcheck_openings,
            _marker: PhantomData,
        };

        let memory_checking = SurgeProof::prove_memory_checking(
            generators,
            preprocessing,
            &polynomials,
            &JoltPolynomials::default(), // Hack: required by the memory-checking trait, but unused in Surge
            &mut opening_accumulator,
            &mut transcript,
        );

        let proof = SurgeProof {
            _instruction: PhantomData,
            commitments,
            primary_sumcheck,
            memory_checking,
        };
        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript,
            opening_accumulator,
        });
        #[cfg(not(test))]
        let debug_info = None;

        (proof, debug_info)
    }


```

</details>

<details><summary><b> eq.rs </b></summary>

```rust
impl<F: JoltField> LassoSubtable<F> for EqSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        // Materialize table entries in order where (x | y) ranges 0..M
        // Below is the optimized loop for the condition:
        // table[x | y] = (x == y)
        let mut entries: Vec<F> = vec![F::zero(); M];
        let bits_per_operand = (log2(M) / 2) as usize;

        for idx in 0..(1 << bits_per_operand) {
            let concat_idx = idx | (idx << bits_per_operand);
            entries[concat_idx] = F::one();
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \prod_i x_i * y_i + (1 - x_i) * (1 - y_i)
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (x, y) = point.split_at(b);

        let mut result = F::one();
        for i in 0..b {
            result *= x[i] * y[i] + (F::one() - x[i]) * (F::one() - y[i]);
        }
        result
    }
}
```

</details>

</details>

---

[Jolt](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/jolt)

<div  align="center"> 
<img src="images/51_jolt_lasso.gif" width="50%" />
</div>
