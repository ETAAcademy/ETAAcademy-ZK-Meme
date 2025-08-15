# ETAAcademy-ZKMeme: 67. ZK Hardware

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>67. ZK Hardware</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZK Hardware</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Zero-Knowledge Proof Hardware: Acceleration, Security, and System Optimization

ZK hardware ecosystem can be divided into four main categories: **compute acceleration** (GPUs, FPGAs, ASICs, and specialized instruction sets), **security** (Trusted Execution Environments, or TEEs), **storage bandwidth** (High Bandwidth Memory, or HBM, and Processing-In-Memory, or PIM), and **network transmission** (SmartNICs).

By combining hardware and algorithm design, this ecosystem overcomes traditional limitations in acceleration, security, and system optimization. It does so by breaking down complex zero-knowledge proof (ZKP) algorithms into fundamental computational primitives and mapping them to dedicated hardware units.

- **GPUs** deliver end-to-end acceleration through full pipeline optimization.
- **FPGAs** create custom accelerators for tasks such as multi-scalar multiplication (MSM) and ZK-friendly hashing.
- **ASICs** use unified architectures capable of supporting multiple ZKP protocols.
- **Specialized instruction sets** improve performance of critical operations via SIMD extensions.
- **TEEs** provide hardware-isolated environments for secure, trusted computation.
- **HBM and PIM** boost memory bandwidth and enable in-memory processing.
- **SmartNICs** offload computation and data handling to the network layer.

Together, these advances can yield performance gains of several hundred times, enabling zero-knowledge proofs to be deployed effectively in large-scale, real-world applications.

---

## 1. Hardware Acceleration for Zero-Knowledge Proofs

Modern general-purpose zero-knowledge proof (ZKP) systems—particularly zk-SNARKs and STARKs—can be divided into three major stages:

- **Arithmetization** – Transforming the computation into an arithmetic circuit or constraint system.
- **Polynomial IOP** – Reducing the constraints to polynomial checks through interactive oracle proofs.
- **Polynomial Commitment Scheme (PCS)** – Allowing the prover to commit to polynomials and later open them at specific points.

For zk-SNARKs, three performance metrics are most critical:

- **Proof generation time** – How long the prover takes to produce the proof.
- **Verification time** – How quickly the verifier can check it.
- **Proof size** – How much data must be sent to the verifier.

Different zk-SNARK schemes trade off these metrics in different ways. **Groth16**, for example, produces extremely small proofs and offers fast verification, but relies on elliptic-curve-based KZG commitments, which are computationally heavy and difficult to accelerate in hardware. Groth16 also requires a **trusted setup** and is not post-quantum secure. Many variants—such as Pinocchio, Hyrax, Libra, Plonk, and Hyperplonk—replace the IOP layer but still use KZG as their PCS, retaining the same acceleration bottlenecks.

### The State of Hardware Acceleration

Attempts to accelerate Groth16 using ASICs, FPGAs, or GPUs have required enormous investment (over **\$100M** in some cases) but yielded limited improvements. For example, the state-of-the-art Groth16 ASIC accelerator **PipeZK** achieves only a **5–15× speedup**, insufficient for many large-scale applications.

While elliptic-curve-based SNARKs are efficient for verification and proof size (hundreds of bytes), their proof generation is inherently hard to speed up. In contrast, **hash-based ZKP constructions**—though producing larger proofs—are far more hardware-friendly. When combined with algorithm–hardware co-design, hash-based ZKPs can achieve significant **end-to-end acceleration**, enabling proofs for more complex computations and expanding the practical scope of ZKP technology.

### Decomposing ZKP Computations for Hardware

Although ZKP algorithms seem complex, their execution can be broken down into a small set of low-level computational tasks, such as:

- **NTT** (Number Theoretic Transform)
- Polynomial arithmetic
- Sparse matrix–vector multiplication (SpMV)
- Hashing (vector-based and Merkle tree)
- Sumcheck protocol operations

These tasks can be further decomposed into **primitive operations**—e.g., addition, multiplication, vector hashing—executed by functional units (FUs) on the chip.

A hardware accelerator arranges these functional units according to:

- **Task requirements** (operation type, input/output size)
- **Bandwidth constraints** (on-chip and off-chip memory)
- **Chip area constraints**

Pipelining and intra-task parallelism ensure that each FU is kept busy. Data layout (e.g., matrix transposition), data reuse (to avoid repeated memory fetches), and compute-instead-of-transfer strategies (e.g., recomputation in Sumcheck instead of storing intermediate states) help mitigate memory bandwidth bottlenecks.

### Task-Level Hardware Strategies

#### Reed–Solomon Codes in ZKP

In ZKP, Reed–Solomon (RS) codes are not used for error correction but to efficiently verify message integrity. Encoding is accelerated using NTT, and even when codewords exceed a single FU’s capacity, block processing can be used. The linearity of RS codes allows multiple codewords to be combined into a single proof. Vector-processing units can accelerate RS encoding efficiently in hardware.

#### Merkle Trees

Merkle trees provide compact integrity proofs for large datasets. The root hash represents the entire dataset. Dedicated **vector-hash units** can compute node hashes in parallel. Strategies vary by tree depth:

- **Upper levels** (fewer, larger nodes) → parallel hashing.
- **Lower levels** (many small nodes) → grouped and reordered hashing.

#### Sparse Matrix–Vector Multiplication (SpMV)

In ZKP, each constraint system (e.g., R1CS) produces three sparse matrices $A, B, C$, used for linear operations on circuit wire values. These matrices can be huge (up to 8 GB), making full on-chip storage impractical. Since the matrices are fixed and sparse, hardware can:

- Use **output-stationary** computation (keep outputs in place while streaming inputs).
- Apply Beneš networks to align vector elements.
- Avoid reloading matrix indices.
- Maximize on-chip bandwidth usage and minimize off-chip access.

#### Sumcheck Protocol

The sumcheck protocol verifies polynomial sums iteratively via dynamic programming. It is a major time consumer in ZKP. The bottleneck is memory access, not computation. Hardware can reduce off-chip data movement via:

- On-demand recomputation
- Streaming processing
- Compression of large vectors $A$
  This improves bandwidth utilization and avoids storing all intermediate states.

<details><summary>Code</summary>

```Algorithm
// The sumcheck dynamic programming algorithm
⏺ def sumcheckDP(A[0:(1<<L)]) -> result[0:L][0:2], rx[0:L]:
      for i in range(1, L+1):
          let s = (1<<(L-i))
          # sum of evaluations
          let y0, y1 = 0
          for b in range(s):
              if i > 1:
                  # Update DP arrays
                  A[b] = A[b] *(1-rx[i-1]) + A[b+2*s]*rx[i-1]
                  A[b+s] = A[b+s]*(1-rx[i-1]) + A[b+3*s]*rx[i-1]
              y0 += A[b]
              y1 += A[b+s]
          result[i][0:2] = [y0, y1]
          # Create random challenge
          rx[i] = HASH(result[i])
      return result, rx
```

</details>

#### Polynomial Arithmetic and NTT Acceleration

Polynomial addition is trivial in hardware (vector addition), but multiplication is more complex—it requires convolution. NTT accelerates polynomial multiplication by:

- Transforming coefficients to the NTT domain.
- Performing point-wise multiplication ($O(n)$ complexity).
- Applying the inverse NTT to return to coefficient form.

Dedicated NTT units are designed for modular multiplication and hierarchical butterfly operations. By executing **NTT → pointwise multiply → inverse NTT** entirely in hardware, these units avoid repeated CPU iterations, enabling large polynomial multiplications to complete with minimal overhead.

---

### 1.1 End-to-End ZKP Acceleration Platform on GPU

The computational workload of a Zero-Knowledge Proof (ZKP) consists of three main stages:

- **Setup** – a one-time cost to compile the circuit into a system of polynomial equations.
- **Witness Generation (Frontend)** – for each input, compute the solution to the polynomial equations.
- **Backend Proof Computation** – generate a proof that the equations hold, and verify it.

Existing work primarily focuses on optimizing _individual components_ (e.g., MSM or NTT) in isolation. However, there is a lack of **unified, flexible, end-to-end acceleration platforms**. In particular, witness generation is rarely considered as part of a complete pipeline, which limits overall throughput.

In modern systems using **hash-based Polynomial Commitment Schemes (PCS)**, the polynomial commitment phase (NTT + reverse order + Merkle tree construction) accounts for less than half of total proof generation time. As a result, accelerating only NTT or MSM yields limited benefits (at most \~2× speedup). Additionally, **data transfer between CPU and accelerators** can further bottleneck performance.

Some existing end-to-end ASIC systems (e.g., NoCap, SZKP) cover most proof-generation stages, but omit witness generation. Since the witness must be recomputed for every new input and cannot be reused, the backend can remain idle waiting for witness data—significantly limiting throughput.

#### Why Plonkish Arithmetic Is Key for GPU Acceleration

Current high-performance provers often rely on **Plonkish** circuit arithmetization, a method introduced by Gabizon et al. in 2019. Compared to the traditional **Rank-1 Constraint System (R1CS)**:

- **R1CS**

  - Each gate can only be addition or multiplication.
  - The wiring topology is represented as sparse matrices $A, B, C$.
  - Sparse matrix operations are expensive for hardware acceleration.

- **Plonkish**

  - Uses **copy constraints** to express arbitrary wiring topology.
  - The witness is stored as a **matrix** of inputs/outputs for each gate.
  - Supports **custom gates** – a single custom gate can replace many addition/multiplication gates, greatly reducing circuit size.
  - Hardware-friendly: avoids sparse matrix representation and maps well to parallel GPU/ASIC processing.

Custom gates act like CPU _custom instructions_, enabling significant reductions in circuit size for repeated sub-circuits and improving proof generation efficiency.

#### GPU as an End-to-End ZKP Engine

**General-Purpose GPUs (GPGPUs)** have become a cornerstone of machine learning and high-performance computing thanks to their ability to handle **massive parallel workloads**.

A typical CUDA programming model:

- **Kernel** – function executed on the GPU.
- **Thread → Block → Grid** hierarchy.
- **threadIdx**, **blockIdx** for data indexing.
- **SIMT** (Single Instruction Multiple Threads) allows thousands of threads to execute the same instruction across different data elements.

This architecture is ideal for:

- **Witness generation** (parallel evaluation of independent gates).
- **Polynomial commitments** (parallel hashing in FRI).

#### End-to-End GPU ZKP Workflow

- **CPU**: Prepares circuit and input data, then sends it to the GPU.
- **GPU**: Executes the entire proof pipeline, including Witness Generation and Proof Computation.
- **CPU**: Receives only the minimal final data needed to assemble the proof.

#### Witness Generation on GPU

Witness generation evaluates each gate in topological order to ensure inputs/outputs satisfy constraints. Gates with no dependency can be processed in parallel.

Optimizations:

- **Dependency layering** – gates are divided into levels (0, 1, 2, …) based on data dependencies.
- **Type grouping** – gates in the same level are grouped by type (add, mul, Poseidon hash, etc.) in a HashMap.
- **Serialization + Lookup Table** – gate parameters are stored in a large array; a lookup table stores start indices and lengths per gate type.

A CPU dispatcher launches corresponding GPU kernels for each gate type using this lookup table, updating the witness matrix efficiently.

#### FRI Polynomial Commitments

In **FRI PCS**, most time is spent on **leaf-node hashing**, not tree construction. Merkle trees are not the bottleneck, so optimization focuses on parallel hash scheduling.

**NTT Optimization**:

- In-place (inverse) NTT results in bit-reversed order.
- Reordering to natural order is expensive.
- Solution: Treat the problem as a **warp-level blockwise matrix transpose**, leveraging shared memory for efficient access.

#### Parallelizing Long Division in FRI’s Low-Degree Test

In the low-degree test, the prover must show $F(z) = v$, i.e., $F(x) - v$ is divisible by $x - z$. This is computed via polynomial long division.

- **Naïve CPU approach**: single-threaded scan, sequential accumulation.
- **Parallel GPU approach**:

  - Split coefficients into $p$ segments.
  - Each thread computes local accumulations in its segment.
  - Synchronize and propagate high-order segment results to lower-order segments.
  - Update results in parallel.

This removes sequential bottlenecks and scales efficiently with thread count.

<details><summary>Code</summary>

```Algorithm
Algorithm 1 Scanning-based Long Division
  acc ← 0
  for i = n-1, n-2, ..., 1 do
      acc ← z · acc + a[i]
      b[i-1] ← acc
  end for

Algorithm 2 Parallelized Division
  In total p threads, each handles l = n/p coefficients
  id ∈ {0, ..., p-1}
  acc ← 0
  for i = l-1, l-2, ..., 0 do
      acc ← z · acc + a[id*l + i]
      b[id*l + i] ← acc
  end for
  syncthreads()
  acc ← 0
  for i = p-1, p-2, ..., id+1 do
      acc ← z^l · acc + b[i*l]
  end for
  syncthreads()
  for i = l-1, l-2, ..., 0 do
      acc ← z · acc
      b[id*l + i] ← b[id*l + i] + acc
  end for

```

</details>

#### Custom Gates: Compiler-Generated GPU Kernels

Custom gates often dominate proof generation time (especially witness generation and quotient polynomial computation). Since these gates are **user-defined**, the accelerator must support a wide variety of operations.

**Solution**: A lightweight **Gate Script** language:

- The user writes a simple formula (e.g., $y_0 = c_0 \cdot x_0 + x_1 + x_2$).
- A compiler performs:

  - **Variable analysis** – identify inputs, outputs, intermediates.
  - **Operation translation** – map arithmetic to GPU/CPU code.
  - **Auxiliary code generation** – handle memory operations and constraint reductions.

This allows flexible, hardware-optimized execution of arbitrary gate logic without requiring users to write GPU-specific code.

---

### 1.2 FPGA Acceleration for ZKP – MSM and Hashing

#### Multi-Scalar Multiplication (MSM)

In Zero-Knowledge Proof (ZKP) systems such as **zk-SNARKs**, **Multi-Scalar Multiplication (MSM)** is a major computational bottleneck. The operation is defined as:

$$
MSM = \sum_{i=0}^{N-1} k_i P_i
$$

where $k_i$ is a scalar and $P_i$ is an elliptic curve point.
A naïve MSM implementation computes each $k_i P_i$ individually and then sums the results, leading to a huge number of **Elliptic Curve (EC) additions**.

#### Pippenger’s Algorithm for MSM

Pippenger’s algorithm reduces the number of EC additions by _accumulating points first_ and multiplying scalars later. The main idea is to **group points into “buckets”** according to scalar bits (windowing), sum them, and then combine results—dramatically reducing EC addition count for large-scale MSM.

However, the algorithm’s performance depends heavily on:

- **Window size** $c$ (larger windows → more buckets)
- **Memory usage** (bucket storage on FPGA is expensive)
- **Bucket accumulation strategy**

In FPGA implementations, **decoupling window size from bucket count** allows reusing the same bucket storage across multiple iterations. This trades a small increase in external bandwidth for _massive on-chip memory savings_.

##### Three Core Steps of Pippenger’s Algorithm

- **Bucket Accumulation**

  ```math
  B_{ws} \gets B_{ws} + P_i
  ```

  Partition each scalar into windows; each window value corresponds to a bucket. Points mapped to the same bucket are summed early to reduce later EC additions.

- **Bucket Aggregation**

  ```math
  G_w = \sum_{s=1}^{2^c - 1} s \cdot B_{ws}
  ```

  Combine bucket sums into a partial result per window. On FPGA, this step benefits from _segmentation_, enabling pipelined execution and reducing idle cycles.

- **Final Window Accumulation**

  ```math
  MSM = \sum_{w=0}^{W-1} 2^{c w} G_w
  ```

  Merge all window results with binary-weighted addition, further minimizing EC additions:

  ```math
  \text{Total EC additions} = W \cdot \big(N + 2 \cdot (2^c - 1)\big) + (W-1) \cdot (c + 1)
  ```

#### Algorithmic Optimizations for MSM

Two broad categories of optimizations exist:

- **Complexity-Reducing**

  - **Signed-Digit Representation**: Cuts bucket count per window from $2^c - 1$ to $2^{c-1}$.
  - **BGMW Method**: Reduces the number of bucket sets, sometimes to just one.
  - **Efficiently Computable Endomorphism**: Uses curve endomorphisms to split one large scalar multiplication into two smaller ones.

- **Hardware Latency-Reducing**

  - **Bucket Segmentation**: Avoids pipeline stalls by processing bucket aggregation in $M$ independent segments.
  - **Scheduling Optimization**: Minimizes “collisions” in the bucket accumulation phase to prevent idle FPGA pipelines.

#### Hashing in ZKP Systems

Hashes are critical in ZKPs for **Merkle trees**, **commitments**, and the **Fiat–Shamir transform**.
Traditional hashes (SHA-256, Keccak) are **bit-operation-heavy** and inefficient in finite-field arithmetic, making them costly to represent in ZKP circuits.

**ZK-friendly hashes**—such as **Poseidon**, **Griffin**, **Rescue-Prime**—are designed for finite-field arithmetic and perform well inside proof systems. However, they tend to be slower on CPUs/GPUs due to lack of hardware-level optimization.

#### FPGA Acceleration for ZK-Friendly Hashes

FPGA-based accelerators can exploit **parallel finite-field arithmetic** to run these hashes much faster:

- **Griffin** – multiplication-heavy, efficient in arithmetic-dominated settings.
- **Rescue-Prime** – prioritizes security and algebraic structure.
- **Reinforced Concrete** – hybrid design balancing performance and security.

#### Hardware Design Features

- **Arithmetic Unit Optimization**: Pipelined finite-field units for modular multiplication, addition, and inversion.
- **Parameter Flexibility**: A single FPGA bitstream can support multiple hash parameters (field sizes, round counts).
- **Streaming I/O**: AXI bus streams inputs/outputs to minimize memory bottlenecks.
- **Batching + Pipelining**: Processes multiple hash computations simultaneously for high throughput.

#### Key Techniques

- **Fast Division via Lookup Tables**
  Division is costly in hardware. If divisors $s_i$ are constants, precompute $1/s_i$ and store $(D / s_i)$ (where $D$ is a power of 2) in a LUT.
  Division becomes: $\text{Result} = (x \times D / s_i) \gg 508$

  —just a multiply and shift.

- **Reconfigurable Modular Multiplier**
  A single modular multiplier supports multiple operations:

  - **Quadratic function** (used in Griffin)
  - **Decomposition** (via LUT-based division)
  - **Composition** (parallel smaller multiplications)
    Mode switching avoids hardware duplication.

- **Fast Power Mapping**
  Rescue-Prime and Griffin use mappings $x \mapsto x^d$ and $x \mapsto x^{1/d}$ over large fields (e.g., BN254).
  Implemented via **square-and-multiply**, using 1 modular multiplier (resource-optimized) or 2 in parallel (latency-optimized).

- **Pipelined Sponge Architecture**
  All hash functions implemented as **sponge constructions** with pipelining and batching for maximum throughput.

---

### 1.3 ASIC

Traditional hardware accelerators for Zero-Knowledge Proofs (ZKPs)—including GPUs, FPGAs, and ASICs—were designed mainly for classical protocols that rely heavily on expensive elliptic curve arithmetic. While these accelerators work well for those protocols, a new class of **hash-based ZKP protocols** has emerged. These protocols remove the _trusted setup_ requirement, eliminate costly elliptic curve operations, and significantly reduce algorithmic complexity.

However, they introduce a wider variety of computational kernels, such as:

- **NTT (Number Theoretic Transform)**
- **Merkle tree construction**
- **Poseidon hashing**
- Various **polynomial operations**

If each kernel were implemented with its own dedicated hardware unit, the chip area would be large, resource utilization would be poor, and new kernels would still need to fall back to the CPU—causing costly data transfers and underutilized hardware.

The modern solution is a **“unified hardware + flexible mapping”** strategy:

- Use a **single, general-purpose hardware architecture** capable of efficiently executing many different kernels.
- Maintain high resource utilization without over-specializing for any one kernel.
- Support a wide range of ZKP primitives, such as modular arithmetic and common data-access patterns.

#### Unified Vector-Systolic Architecture

The unified architecture is based on **Vector-Systolic Arrays (VSAs)**—a variant of the classic systolic array architecture widely used for matrix and vector computations.

**Key features include:**

- **Homogeneous VSA design**: No kernel-specific customization, ensuring flexibility for evolving ZKP protocols.
- **Local links** between Processing Elements (PEs) for efficient neighbor-to-neighbor communication.
- **New vector processing modes** that allow each PE column to operate as an independent vector unit.
- **Global SRAM cache** for data buffering and reuse.
- **Double-buffered scratchpad memory** between VSA and off-chip DRAM to hide memory latency.
- **Global transpose buffer** for efficient in-chip data reordering (e.g., NTT or polynomial transforms).
- **Twiddle Factor Generator** for on-the-fly computation of NTT rotation factors using a modular multiplier + buffer.

Each **PE** contains:

- Modular add and multiply units
- Local register files for fast data access
- Systolic links for horizontal/vertical data movement

#### Kernel Mapping Strategies

A flexible mapping layer allows the hardware to support multiple ZKP kernels without architectural changes.

##### NTT Acceleration

Large NTTs are decomposed into **multi-dimensional NTTs**—breaking them into small, fixed-size NTTs that map perfectly to the hardware resources.

- Between small NTTs, **element-wise twiddle factor multiplication** is applied.
- The mapping uses **Multi-path Delay Commutator (MDC) pipelines** with register file buffers and transpose buffers to ensure continuous data flow and support multiple NTT variants.
- Idle PEs are reused for extra twiddle multiplications.

##### Poseidon Hash

Poseidon is used in Merkle tree construction, Fiat–Shamir transforms, and Proof-of-Work computations.

- Its permutation rounds include **Full Rounds**, **Partial Rounds**, and **Pre-Partial Rounds**, with some steps only applied to `state[0]`.
- Matrix multiplications use either **dense MDS matrices** or **sparse MDS matrices**.
- The VSA’s neighbor-linked PEs handle all computations, with pipeline scheduling and register buffering ensuring correct timing.
- Reverse links distribute single-element results efficiently, supporting sparse matrix operations.
- The architecture achieves **one state per cycle throughput**.

##### Merkle Tree

Merkle tree construction benefits from:

- Blocked processing
- Parallel + pipelined execution
- Optimized memory layouts

##### Polynomial Operations

Polynomial computations combine NTT acceleration, vector modes, and block-level buffering.

- Special dependency chains (e.g., partial products) are handled with grouped PE computations and inter-group passing.

#### Compiler-Aware Hardware Utilization

A compiler automatically maps different kernels to the unified architecture, ensuring:

- Maximum PE utilization
- Minimal idle cycles
- Consistent performance across different ZKP protocol workloads

This approach mirrors neural network accelerators—where a single hardware design efficiently supports convolution, matrix multiplication, and even attention mechanisms—ensuring the ASIC remains relevant as ZKP protocols evolve.

<details><summary>Code</summary>

```Algorithm
 Algorithm 1: Poseidon Permutation in Plonky2
  Input/output: state[12]

  1  function FullRound(state, r):
  2      for i ← 0 to 11 do
  3          state[i] ← state[i] + RoundConst[r][i];
  4          state[i] ← state[i]^7;
  5      state ← state × MDSMatrix;
  6      return state;

  7  function PartialRound(state, r):
  8      state[0] ← state[0]^7;
  9      state[0] ← state[0] + PartialRoundConst[r];
  10     state ← state × SparseMDSMatrix;
  11     return state;

  12 function PrePartialRound(state):
  13     state ← state + PrePartialRoundConst;
  14     state ← state × PreMDSMatrix;
  15     return state;

  16 for r ← 0 to 3 do state ← FullRound(state, r);
  17 state ← PrePartialRound(state)
  18 for r ← 0 to 21 do state ← PartialRound(state, r);
  19 for r ← 4 to 7 do state ← FullRound(state, r);
```

</details>

---

### 1.4 ZKP-Specific Acceleration Instructions

ZKP-specific acceleration instructions refer to adding specialized hardware features—such as SIMD instruction extensions or hash acceleration units—into general-purpose CPUs/GPUs. The idea is similar to **AES-NI** for cryptographic operations.

For example:

- **NVIDIA Hopper GPUs** implement low-level optimizations for elliptic curve and hash computations.
- The **RISC-V** community is exploring dedicated ZKP instruction set extensions.

#### SIMD for ZKP Acceleration

**SIMD (Single Instruction Multiple Data)** allows one instruction to operate on multiple pieces of data simultaneously, as opposed to traditional CPU instructions that operate on only one datum per cycle.

- Platforms: Intel (AVX / AVX512), AMD, ARM, RISC-V
- In ZKP’s **Multi-Scalar Multiplication (MSM)**, SIMD can process multiple scalar–point multiplications in parallel, improving throughput.

A SIMD-accelerated MSM implementation can achieve:

- **No write conflicts** (avoiding concurrent updates to the same memory location)
- **Constant memory overhead**
- **Multi-level parallelism**:

  - _Task-level parallelism_: Different MSM sub-tasks run in different threads.
  - _Loop-level parallelism_: Parallelizing loops within a single MSM task.
  - _Data-level parallelism_: SIMD parallelism for point addition inside a task.

- **Three-tier buffering** to maximize SIMD engine utilization and alleviate memory bottlenecks.

For example, **AVX512-IFMA** instructions can implement up to six SIMD elliptic curve arithmetic engines on a single CPU core, supporting multiple coordinate systems (Affine, Projective, Jacobian) and various point addition forms. Integration into ZKP libraries allows further optimizations such as:

- **Point Deduplication**
- **Three-stage memory optimization**

The result: efficient large-scale MSM execution on commodity CPUs without expensive dedicated hardware—ideal for consumer devices and mobile platforms.

#### The MSM Bottleneck: Bucket Accumulation

The most time-consuming part of MSM is **bucket accumulation**:

$$
Q = \sum k_i P_i
$$

In **Pippenger’s algorithm**, each scalar $k_i$ is assigned to a bucket $B_j$, and its corresponding point $P_i$ is added to that bucket.

- **GPU/FPGAs**: Can assign each sub-task to an independent thread and process in parallel.
- **SIMD**: Requires the _same_ instruction for all parallel data lanes, which creates challenges:

  - Conditional branches are difficult—empty bucket ($B_j = O$) vs. non-empty bucket ($B_j \neq O$) require different instructions.
  - SIMD lanes may be underutilized if the number of sub-tasks $\lceil \lambda / s \rceil$ is not a multiple of the SIMD width.
  - Directly parallelizing sub-tasks in SIMD can cause **write conflicts** (multiple lanes writing to the same bucket).

#### The Three-Level Parallel MSM Architecture

To overcome these limitations, a **three-level parallelism architecture** is used:

- **Task-level Parallelism**

  - Each _sub-MSM_ $G_\alpha = \sum_{i=1}^n k_i^\alpha P_i$ runs in its own thread (`tidα`).
  - Each thread maintains its own **bucket buffer** $buf_\alpha^{(0)}$ to hold partial sums for each bucket, along with point addresses for SIMD processing.

- **Loop-level Parallelism**

  - Inside each thread, points are not added to buckets one by one.
  - Instead, sub-threads (`tidαβ`) batch multiple points into a **point buffer** $buf_{\alpha\beta}^{(1)}$ and track the count $cnt_{\alpha\beta}^{(1)}$.
  - Once the buffer reaches the SIMD width $χ$, a SIMD point-add operation is triggered.

- **Data-level Parallelism (SIMD)**

  - SIMD operations (`SimdPADDaff`) perform $χ$ point additions simultaneously
  - Results are first stored in a **temporary buffer T** (size ≤ $2χ$).
  - A **state machine** controls when points are directly accumulated into buckets vs. when they are combined with other buffered points.

#### State Machine for SIMD MSM Accumulation

| State         | Description                                                                                         |
| ------------- | --------------------------------------------------------------------------------------------------- |
| **State-0**   | Buffer empty; receive SIMD output → go to State-1 (distinct scalars) or State-2 (duplicate scalars) |
| **State-1**   | $χ$ points with distinct scalars → directly add to buckets (`SimdPADDprj`)                          |
| **State-2**   | $χ$ points with duplicate scalars → wait for more SIMD output before accumulation                   |
| **State-3/4** | Mix of cached and new outputs → pair points for accumulation or temporary buffering                 |
| **State-5**   | First cache set cleared → slide buffer and return to State-1/2                                      |

This buffering and scheduling avoids write conflicts and maximizes SIMD lane utilization.

#### Algorithm: SIMD-Accelerated MSM

The algorithm consists of three procedures:

- **Main MSM Loop**:

  - Divide the MSM into $\lceil \lambda / s \rceil$ sub-MSM tasks (Task-level parallelism).
  - Combine partial results with sequential point doubling.

- **SubMSM**:

  - Split each sub-task into loop-parallel sub-threads.
  - Maintain independent bucket buffers for each thread.

- **LoopMSM + SimdPADD**:

  - Fill point buffers.
  - Trigger SIMD additions when buffers are full.
  - Use the state machine to handle bucket accumulation without conflicts.

<details><summary>Code</summary>

```Algorithm
Algorithm 1 SIMD-Accelerated MSM Computation Mechanism
  Input: The MSM size n, the λ-bit scalars k̃ := {ki}i∈[n], the points P̃ := {Pi}i∈[n], the
  windows size s and
         the parallelism number χ
  Output: The MSM result Q = ∑ⁿᵢ₌₁ kiPi

  1: #pragma omp parallel
  2: for α = 1 to ⌈λ/s⌉ do
  3:     Gα ← SubMSM(k̃, P̃, n, λ, α)                    // Open new thread tidα
  4: end for
  5: for Q := G⌈λ/s⌉, α = ⌈λ/s⌉ - 1 to 1 do
  6:     Q ← SeqPDBL(Q, s)                               // Sequential s-times point double
  7:     Q ← Q + Gα
  8: end for
  9: return Q

  10: procedure SubMSM(k̃, P̃, n, λ, α)
  11:     buf(0)α ← {ℓ, Bℓ := O, addrPAℓ := null}ℓ∈[1,2ˢ⁻¹]
  12:     #pragma omp parallel
  13:     for β = 1 to τ do
  14:         βl := (β - 1)⌈n/τ⌉ + 1, βr := β⌈n/τ⌉
  15:         LoopMSM(k̃, P̃, buf(0)α, α, βl, βr)        // Open new thread tidαβ
  16:     end for
  17:     {Bℓ}ℓ∈[2ˢ⁻¹] ← TailTask(buf(0)α)
  18:     Gα ← BucketAggregation({Bℓ}ℓ∈[2ˢ⁻¹])
  19:     return Gα
  20: end procedure

  21: procedure LoopMSM(k̃, P̃, buf(0)α, α, βl, βr)
  22:     buf(1)αβ ← ∅, Tαβ ← ∅, ζ := 1
  23:     for i = βl to βr do
  24:         kiα ← BitsExtract(ki, α, s)                // ki = ∑⌈λ/s⌉ᵢ₌₁ 2^(α-1)s kiα
  25:         if addrPAkiα == null then
  26:             addrPAkiα := addrOf(Pi)
  27:         else
  28:             buf(1)αβ[ζ] ← (ℓζ := kiα, addrPBζ := addrPAkiα, addrPCζ := addrOf(Pi))
  29:             addrPAkiα := null, ζ := ζ + 1
  30:         end if
  31:         if ζ > χ then
  32:             SimdPADD(buf(0)α, buf(1)αβ, Tαβ)
  33:             ζ := 1
  34:         end if
  35:     end for
  36: end procedure

  37: procedure SimdPADD(buf(0)α, buf(1)αβ, Tαβ)
  38:     {ℓj, PBj, PCj}j∈[χ] ← buf(1)αβ
  39:     {Tj}j∈[χ] ← SimdPADDaff({PBj, PCj}j∈[χ])    // Tj = PBj + PCj
  40:     Push {ℓj, Tj}j∈[χ] into Tαβ
  41:     if len(Tαβ) == χ then
  42:         if ∃a,b ∈ [χ] s.t. ℓa == ℓb then return    // State-2
  43:         else
  44:             {Bℓj} ← SimdPADDprj({Bℓj, Tj}j∈[χ]) and pop the first χ tuples in Tαβ
  // State-1
  45:         end if
  46:     else                                            // State-3 or State 4
  47:         S ← ∅, t = 1
  48:         for a = 1 to 2χ do
  49:             if ∃b > a & b ∉ S s.t. ℓa == ℓb then
  50:                 buf(2)αβ[t] := (Ta, Tb, Tχ+t), S = S ∪ {a,b}    // Refer to Tχ+t = Ta
  + Tb
  51:             else
  52:                 buf(2)αβ[t] := (Bℓa, Ta, Bℓa), S = S ∪ {a}      // Refer to Bℓa = Bℓa
  + Ta
  53:             end if
  54:             t = t + 1
  55:             if t > χ then
  56:                 SimdPADDprj(buf(2)αβ) and update buf(1)αβ and Tαβ    // State-5
  57:                 Pop the first χ tuples in Tαβ and goto Line 41
  58:             end if
  59:         end for
  60:     end if
  61: end procedure

```

</details>

---

## 2. Trusted Execution Environments (TEE)

A **Trusted Execution Environment (TEE)** is essentially a hardware-based isolated environment (e.g., a CPU or GPU enclave) where computations can run securely, shielded from interference by the operating system. The key purpose of a TEE is to provide a **trusted computing environment** that can produce **hardware attestations**—cryptographic proofs that allow on-chain verifiers to confirm the computation was executed correctly inside the TEE and that the resulting state updates are valid. The primary trust assumption lies in the **security and integrity guarantees provided by the TEE hardware vendor**.

In simple terms, TEEs allow you to perform computations off-chain while still giving the blockchain confidence that the results are trustworthy—without relying on the host OS or the operator’s honesty. **TEE-based rollups** use hardware isolation to produce proofs, enabling high throughput, fast finality, and low-cost off-chain operations. The main trade-off is that TEEs are proprietary hardware, making public auditing difficult, and any discovered vulnerabilities must be patched via vendor firmware updates.

| Rollup Type | Computation Method                 | Finality                | Throughput | Cost | Security Assumption          |
| ----------- | ---------------------------------- | ----------------------- | ---------- | ---- | ---------------------------- |
| Optimistic  | Delayed verification + Fraud Proof | Slow (challenge period) | High       | Low  | At least one honest observer |
| ZK          | zk-SNARK / zk-STARK                | Fast                    | Medium     | High | Correctness of ZK proof      |
| TEE         | Hardware-isolated execution        | Fast                    | High       | Low  | TEE hardware security        |

With a TEE, all critical computation—such as message routing or storage—runs inside the enclave. Even if the host server is compromised or controlled by a malicious operator, the attacker cannot inspect metadata or sensitive data. Unlike traditional **multi-party non-collusion models**, a TEE approach does not require multiple independent servers; it can even be deployed within a single data center while still achieving low latency and high bandwidth. TEEs enable **secure execution of arbitrary programs on potentially hostile remote servers** while preserving:

- **Confidentiality** – Program state and data remain invisible to external entities.
- **Integrity** – The execution behavior is guaranteed to be correct.

Examples of TEEs include **Intel SGX**, **AMD SEV**, and earlier **Intel TXT**. The goal is to ensure that even when programs are deployed on untrusted cloud servers, both data and code remain protected.

### Practical Challenges of TEEs

Real-world TEEs face several issues:

- **Microarchitectural Side Channels**
  These include vulnerabilities such as **speculative execution attacks** or **voltage fault injection**. Such attacks can bypass hardware protections and must be mitigated by the hardware vendor through microcode or firmware patches. They fall within the **hardware threat model**.

- **Software Side Channels**
  These occur when an application unintentionally leaks information through its **memory access patterns** or **control flow behavior**. Unlike microarchitectural channels, these are **outside the TEE’s native threat model** and must be addressed by the application developer.

### Fully Oblivious Algorithms in TEEs

To fully protect sensitive data inside a TEE, developers may use **fully oblivious algorithms**. These algorithms ensure that:

- **Control flow** (branching, loops, etc.) does not depend on private data.
- **Memory access patterns** remain identical regardless of input.

In other words, the **execution path** and **memory read/write behavior** are independent of sensitive data, preventing external observers from inferring secrets.

While this approach incurs **some performance overhead** compared to standard algorithms, it can—when combined with TEE hardware—achieve **highly efficient and privacy-preserving applications**. Several real-world deployments have already demonstrated its practicality.

---

## 3. High-Bandwidth Memory (HBM), Processing-in-Memory (PIM), and SmartNIC Acceleration

Many zero-knowledge (ZK) proof algorithms can be parallelized using SIMD, GPUs, or FPGAs, but they often remain **bottlenecked by memory bandwidth**. Specialized hardware architectures—such as high-bandwidth memory and in-memory computing—help address this limitation, while SmartNICs target performance at the network data transfer level.

### Storage Bandwidth–Oriented Hardware: HBM & PIM

#### High-Bandwidth Memory (HBM)

**HBM (High Bandwidth Memory)** is a 3D-stacked DRAM technology designed to deliver **hundreds of gigabytes per second** of memory bandwidth at **low power consumption**. This is particularly advantageous for ZK provers, where large-scale polynomial or circuit computations require frequent, random memory access. By reducing the memory bottleneck, HBM can significantly speed up proof generation.

**Architecture**:

- HBM stacks consist of multiple **DRAM dies** vertically integrated with a **logic die** via **Through-Silicon Vias (TSVs)**.
- The **logic die** handles I/O, buffering, and interface functions with the processor (CPU, GPU, or FPGA).
- DRAM dies store the actual data, organized in a hierarchy:

  - **Channels** → split into **pseudo-channels (pChannels)** → each with multiple **bank groups (BG)** → each BG containing multiple **banks**.

- Each bank stores data in rows, accessed via a row buffer and column decoder.
- The multi-channel and pseudo-channel design increases parallelism, which is critical for workloads like **Sparse Matrix-Vector Multiplication (SpMV)** that require high random-access throughput.

#### Processing-In-Memory (PIM)

**PIM (Processing-in-Memory)** integrates compute units directly into the memory die, reducing data movement between CPU/GPU and memory. This is especially effective for ZK workloads with heavy data shuffling, such as sparse matrix multiplications.

**Key benefits for ZK systems**:

- Eliminates large volumes of data transfers between compute cores and memory.
- Reduces both **latency** and **energy consumption**.
- Particularly suited for **zk-STARK** or polynomial-commitment-based proof systems, where memory access dominates compute time.

**HBM-PIM Enhancements**:

- Adds **SIMD floating-point units**, register files, and control registers inside each memory die.
- In exchange for compute logic, some DRAM rows are replaced by logic circuits (e.g., **half of each bank’s rows**).
- Supports two operation modes:

  - **Single-Bank (SB)** – behaves like traditional DRAM.
  - **All-Bank (AB)** – broadcasts PIM instructions to all banks for parallel execution.

- Instruction set includes **RISC-style control, arithmetic, and data movement** commands.
- **Limitation**: Does not support **indirect indexing**, making it less efficient for sparse matrices where non-zero elements are irregularly distributed.

#### Sparse Matrix–Vector Multiplication (SpMV) Challenge

SpMV is a core operation in scientific computing, machine learning, graph analytics, and many ZK proof algorithms. Sparse matrices are mostly zeros, with non-zero entries unevenly distributed. Access patterns are **irregular and indirect**, which:

- Causes **bank imbalance** in PIM, overloading some memory banks while others remain idle.
- Leads to wasted bandwidth on traditional CPUs/GPUs.
- Makes it challenging to optimize with architectures originally designed for dense matrices.

### Network Data Transfer–Oriented Hardware: SmartNIC

A **SmartNIC** is a network interface card equipped with programmable processing units—such as **FPGAs**, **ARM cores**, or **network processing units (NPUs)**—capable of executing computations directly on incoming and outgoing data.

**Advantages for ZK Proof Systems**:

- Offloads computation-intensive and protocol-handling tasks from the host CPU.
- Can handle batch hashing, small-scale polynomial/matrix operations, and encryption/decryption directly at the network layer.
- Reduces proof generation/verification latency in distributed ZK systems.

#### FPGA-Based SmartNIC

An **FPGA-based SmartNIC** leverages the reconfigurable logic of an FPGA to execute parallel workloads and specialized algorithms.

**Key Features**:

- **With or without embedded microprocessors**:

  - Some FPGAs include ARM cores for hybrid processing.
  - Others can implement a **soft CPU** in programmable logic for SoC-like flexibility.

- **On-chip memory**: Very low latency, useful for caching and intermediate data storage.
- **Off-chip DRAM**: Large capacity, accessible through parallel memory controllers for high throughput.
- **PCIe Interface**: Typically x8 or x16 lanes, with some modern systems supporting x32, enabling high-speed, low-latency transfers between NIC and host.
- **Data Flow**: Packets arrive via the NIC’s physical interface, are processed on the FPGA, and then transmitted to the host via PCIe.

This architecture allows SmartNICs to **perform compute directly on the data in transit**, effectively merging network and compute layers for performance-critical applications like large-scale ZK proof systems.

[UniZK](https://github.com/tsinghua-ideal/UniZK)
[SIMD](https://github.com/JR-account/SimdMSM)
