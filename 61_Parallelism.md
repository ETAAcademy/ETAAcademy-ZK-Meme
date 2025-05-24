# ETAAcademy-ZKMeme: 61. Parallelism

/ˈparəlɛlɪz(ə)m/

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>61. Parallelism </td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Parallelism</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# The Evolution of Blockchain Execution: From Sequential to Distributed Parallelism

Blockchains evolve beyond traditional sequential execution into a complex, multidimensional scaling paradigm. At the state model level, the UTXO model offers natural parallelism but increases development complexity. In contrast, the account-based model simplifies development yet imposes constraints due to its sequential execution nature. Parallel computation is becoming increasingly refined, moving from coarse-grained, account-level parallelism (e.g., Solana) to fine-grained, instruction-level parallelism (e.g., GatlingX). Techniques such as recursive zero-knowledge proof parallelization, decoupling of storage and execution, direct state access, asynchronous node loading, and execution-read-update pipelines aim to address the growing performance gap, where consensus protocols have scaled for high throughput but execution remains bottlenecked by I/O limitations.

Execution architectures are diverging into two main paths: (1) parallel-augmented blockchains that retain EVM compatibility (e.g., Monad, MegaETH), and (2) chains with natively concurrent architectures that rebuild from the ground up (e.g., Solana, Sui). Solana, for example, has established a significant technological moat through end-to-end system optimization, ranging from client-side enhancements and consensus-layer improvements to networking, hardware acceleration, and Layer-2 expansion. Meanwhile, asynchronous concurrent systems based on the Actor model (e.g., AO, Internet Computer Protocol) offer theoretically unlimited scalability by isolating state and using message-passing for coordination. Together, these approaches are pushing blockchain forward—from a single-machine state machine model to a truly distributed, high-parallel computational platform.

---

In modern computing systems, execution models span a wide spectrum—**sequential, concurrent, parallel, asynchronous, distributed**, and **pipeline execution**. Each serves distinct purposes across various architectures: web servers leverage concurrency and asynchronous processing to handle high volumes of requests; large-scale data systems utilize parallel and distributed processing to boost throughput; and CPUs rely on pipelining and parallelism to maximize instruction-level efficiency. Choosing the appropriate execution model depends on a combination of task characteristics, hardware resources, performance requirements, and maintenance costs—striking a balance to achieve efficient and scalable system performance.

When applied to **blockchain systems**, however, these execution models encounter fundamental limitations. Most mainstream blockchains operate on two state models: the **UTXO model** (e.g., Bitcoin, Cardano) and the **account model** (e.g., Ethereum, Solana). The UTXO model naturally supports parallel transaction processing but increases development complexity. In contrast, the account model—while more developer-friendly—relies on a globally shared state. Since it’s often impossible to statically determine which part of the state a transaction modifies, systems must fall back to **sequential execution** to preserve determinism. This constraint results in limited throughput, delayed block propagation, high transaction fees, and inefficient hardware utilization. Even **Layer 2** scaling solutions struggle to overcome this fundamental bottleneck, as execution within each L2 remains sequential and inter-L2 communication introduces added complexity.

To address these challenges, blockchain scalability has evolved into a **multi-layered technical paradigm**, primarily categorized into five dimensions:

- **Execution-enhanced** (e.g., intra-chain parallelism, GPU acceleration),
- **State-isolated** (e.g., sharding, UTXO-based designs),
- **Off-chain outsourcing** (e.g., Rollups, data availability layers),
- **Structural decoupling** (e.g., modular blockchain architectures),
- **Asynchronous concurrency** (e.g., Actor-based systems).

Among these, **parallel computing** has emerged as a cornerstone of blockchain scalability. Depending on the granularity, parallelism can be applied at various levels:

- **Account-level** (ETH, Solana),
- **Object-level** (Sui),
- **Transaction-level** (Monad, Aptos),
- **Call-level or Micro-VM** (MegaETH),
- **Instruction-level** (GatlingX).

As execution granularity increases, the potential parallelism improves, but so does system complexity.

#### Actor-Based Architectures: A New Execution Model

In contrast to traditional paradigms, **asynchronous concurrent systems**, particularly those based on the **Actor model** (e.g., AO, Internet Computer Protocol \[ICP], Cartesi), offer a fundamentally reimagined execution model. These systems embrace **isolated state, asynchronous message passing, and decoupled processes** to achieve high scalability and fault tolerance. Each smart contract is treated as an independent, asynchronous computation unit, communicating through messages instead of shared memory. This design eliminates global state contention and enables **unbounded horizontal scalability**.

- **AO**, built atop the **Arweave decentralized storage layer**, introduces a _Process-based actor architecture_ ideal for complex computational tasks, such as AI agents or parallel transaction workflows. It can theoretically allocate up to 2²⁵⁶ bytes of isolated storage per process.
- **ICP** employs _Canister-based smart agents_ within a Wasm virtual machine, using a **subnet-based distributed consensus system**. It provides an all-in-one on-chain platform for full-stack Web3 applications, emphasizing stateful hosting capabilities.

These architectures draw from over 50 years of Actor-Oriented Programming research, enabling **non-blocking, asynchronous communication** across computational tasks. With no global resource contention, Actor-based chains offer a promising path toward "on-chain operating systems"—a step beyond traditional scalability models such as Rollups or sharding. However, developer tooling and ecosystem maturity are still evolving.

#### Communication and Execution in Parallel Systems

In parallel and concurrent computing, communication mechanisms such as **Sockets, RPC, event notification systems, memory mapping, and inter-process pipes** form the backbone of inter-process coordination. Two major interaction models dominate:

- **Shared Memory**: Allows multiple threads or contracts to access a common state with mutual exclusion mechanisms (e.g., locks). While straightforward, it inherently limits parallelism due to inevitable queuing and contention.
- **Message Passing**: Avoids shared state entirely, enabling high concurrency and reduced contention—particularly well-suited for blockchain environments seeking scalable, asynchronous execution.

#### Diverging Paths in Blockchain Parallelism

The roadmap for blockchain parallelism is diverging into two main streams:

- **EVM-compatible parallel-enhanced chains** (e.g., Monad, MegaETH): These retain compatibility with the Ethereum ecosystem while reengineering the execution engine for concurrency. Many are in testnet or early mainnet phases.
- **Natively parallel chains** (e.g., Solana, Sui, Aptos): These systems redesign virtual machines and execution logic from the ground up to support parallelism, yielding real-world performance benefits but facing challenges in Ethereum compatibility.

Chains like **Solana**, **Sui**, and **Sei v2** have demonstrated practical success by rearchitecting the execution stack:

- **Solana** uses **account-level locking** and a multithreaded execution engine to schedule transactions in parallel. System-wide optimizations include:

  - **Tile Architect** for modular client-side processing,
  - **Rotor/Votor** replacing Tower BFT for faster consensus,
  - **doublezero** fiber optics and undersea cables for low-latency networking,
  - **FPGA-based hardware acceleration** to speed up transaction processing—rivalling even zk-proving hardware.

- **MoveVM** (used by Sui and Aptos) employs **object ownership and static concurrency analysis** to parallelize execution without conflicts.
- **Sei v2** integrates a multithreaded order matching engine with **CosmWasm**, enabling fine-grained execution control.

Meanwhile, **Fuel** introduces a high-performance execution layer for Ethereum based on an optimized UTXO model and its custom **Sway language**, facilitating deterministic parallelism with Ethereum compatibility.

Blockchain scalability is no longer confined to single-dimensional goals like TPS increases. It now encompasses a **holistic evolution of execution engines, storage architectures, communication protocols, and system structures**. The journey moves from shared memory to message-driven interaction, from serial to fine-grained parallel execution, and ultimately to fully asynchronous, actor-driven computation.

This transformation is not just about performance—it marks a **paradigm shift** from simulating distributed systems to realizing **true distributed computation platforms**. As the blockchain stack evolves into a modular, composable, and efficient ecosystem, the boundaries between smart contracts, services, and operating systems begin to blur—ushering in a new era of decentralized computing.

---

## 1. Ethereum Scalability

### 1.1 zk-Powered Parallelism

Ethereum scalability has evolved along two major technical paths: **Layer 2 solutions** and **sharding**. Layer 2 approaches—such as Rollups (optimistic and zk), state channels (e.g., Lightning Network), and sidechains (e.g., Zendoo)—aim to offload computation off-chain while preserving security guarantees through cryptographic proofs or economic incentives. However, these solutions face challenges in terms of **data availability**, **cross-chain communication**, and **trust assumptions**. Meanwhile, sharding attempts to split the blockchain into multiple concurrent segments to process transactions in parallel but introduces architectural complexity and inter-shard coordination overhead.

Recent innovations, such as **SNARKtor** and **Mina**, leverage recursive zero-knowledge proofs (ZKPs) to dramatically reduce verification costs. Despite these advances, achieving _secure and efficient transaction parallelization_ remains an active research challenge.

#### A zk-Based Parallel Execution Model

To address Ethereum’s fundamental performance constraints, a new class of **zk-parallel execution frameworks** proposes a modular execution and proof structure. The core idea is to decouple transaction execution from block validation by dividing **non-conflicting transactions** into independent **batches**, processed in parallel by multiple **batch provers**. Each batch independently generates a SNARK proof. These batch proofs are then aggregated by a **block producer** into a final block-level proof, ensuring correctness without requiring all nodes to re-execute the transactions. This significantly reduces block propagation time and increases overall throughput, while recursive aggregation minimizes the on-chain verification cost.

In traditional account-based systems like Ethereum, all contract state resides in a global shared state, making parallel execution difficult. To overcome this, zk-parallel systems redesign the ledger model by **partitioning contract state into substates**, each owned and managed by a specific account.

This enables **parallel execution of different users’ interactions** with the same contract. For example, instead of storing token balances inside an ERC-20 contract’s state, each user account maintains its own substate representing their balance. The zkVM ensures that a smart contract can only read/write substates it owns, preserving integrity while enabling parallelism.

In this model:

- **User Accounts** and **Contract Accounts** are both containers of substates.
- Substates include both self-owned data (e.g., balances) and delegated data from contracts.
- The **zkVM enforces ownership constraints**, preventing contracts from modifying states they don’t own.

The system uses **Compact Sparse Merkle Trees (CSMT)** to store the global state. Each node in the tree maps a hashed key—derived from the account address, substate owner, and field index—to a value. This structure enables:

- Efficient **proof of membership or non-membership**,
- Scalable **batch proof generation**, and
- A cryptographically verifiable mapping from account addresses to substates.

This structure replaces the monolithic global state with a modular, proof-friendly state model.

Each transaction is a **cryptographically signed instruction** containing:

- **Read Set**: List of substates the transaction reads
- **Write Set**: Substates it intends to modify
- **Data**: Parameters such as value transfer or contract invocation
- **Gas Parameters**: Minimum gas fee, gas limit, and gas price

Transactions must declare their **read/write sets** upfront. For complex state-dependent logic, a **two-step execution model** can be applied to first reveal dependencies and then execute accordingly. The transaction cost is calculated as `max(min_gas, actual_gas_used) × gas_price`, bounded by the gas limit.

A block in this architecture contains:

- A list of **state modifications**: identifying accounts, substate owners, field identifiers, and new values
- An ordered list of **transaction hashes**
- The **previous block hash**
- The **new global state root** after applying state changes
- A **SNARK proof** verifying the correctness of state transitions

This structure ensures both **verifiability** and **completeness** of state transitions.

#### Recursive Proof Hierarchy

In a zk-parallel blockchain architecture, recursive zero-knowledge proofs (zkSNARKs) are used to efficiently validate the correctness of transaction execution and global state transitions. The system is built on a hierarchy of composable proofs, each responsible for verifying a specific aspect of the computation. Below, we describe each proof component, including its mathematical formulation and purpose in the proof system. The system constructs a **recursive SNARK hierarchy** to validate the entire chain state with a single proof. The process involves:

- **Tx-Base**: Individual transaction proofs
- **Tx-Merge**: Aggregated proofs of multiple transactions
- **Mem-Proof**: State membership validation proofs
- **Batch-Proof**: Proof of a set of transactions within a batch
- **Global State Transition Proof**: Verifies state changes across batches
- **Conflict-Freedom Proof**: Ensures inter-batch transactions do not conflict
- **Block Proof**: Final proof validating the full block

**Notation**

- $SV_i^{(b)}$: The i-th state view within batch b
- $Tx_i^{(b)}$: The i-th transaction in batch b
- $S_l$: The l-th global state
- $H()$: Cryptographic hash function
- $π$: Zero-knowledge proof (proof object)

**1) Transaction Proof (Tx-Base)**

This is the most basic level of proof, validating the correct execution of a single transaction.

- **Setup**:
  $(pk_{\text{TxBase}}, vk_{\text{TxBase}}) \leftarrow \text{Setup}(1^\lambda)$

- **Proving**:
  $\pi^i_{TxBase} \leftarrow \text{Prove}(pk_{TxBase}, (SV_i^{(b)}, SV_{i+1}^{(b)}, H(Tx(b)_i)), (Tx(b)_i))$

- **Verification**:
  $\text{Verify}(vk_{\text{TxBase}}, (SV_i^{(b)}, SV_{i+1}^{(b)}, htx_i), \pi^i_{\text{TxBase}}) \in \{\text{true}, \text{false}\}$

**Interpretation:**
This proof ensures that executing transaction $Tx_i^{(b)}$ correctly transitions the state from $SV_i^{(b)}$ to $SV_{i+1}^{(b)}$.

**2) Transaction Merge Proof (Tx-Merge)**

This aggregates multiple transaction proofs into a single proof for efficiency.

- **Setup**:
  $(pk_{\text{TxMerge}}, vk_{\text{TxMerge}}) \leftarrow \text{Setup}(1^\lambda)$

- **Proving**:
  $\pi_{\text{TxMerge}}^{i..j} \leftarrow \text{Prove}(pk_{\text{TxMerge}}, a, w)$
  where:

  - $a = (SV_i^{(b)}, SV_{j+1}^{(b)}, htx_{i..j})$: public inputs
  - $w$: witness (intermediate state transitions and individual proofs)

**Interpretation:**
This proof confirms that a sequence of transactions from $Tx_i^{(b)}$ to $Tx_j^{(b)}$ collectively and correctly transitions the state from $SV_i^{(b)}$ to $SV_{j+1}^{(b)}$.

**3) Membership Proof (Mem-Proof)**

This proof validates that a local state view is consistent with the current global state.

- **Setup**:
  $(pk_{\text{Mem}}, vk_{\text{Mem}}) \leftarrow \text{Setup}(1^{\lambda})$

- **Proving**:
  $\pi_{\text{Mem}}^b \leftarrow \text{Prove}(pk_{\text{Mem}}, (SV_i^{(b)}, S_l), w)$

- **Verification**:
  $\text{Verify}(vk_{\text{Mem}}, (SV_i^{(b)}, S_l), \pi_{\text{Mem}}^b) \in \{\text{true}, \text{false}\}$

**Interpretation:**
Proves that $SV_i^{(b)}$, the local state view for batch $b$, is a valid subset of the global state $S_l$.

**4) Batch Execution Proof (Batch-Proof)**

This proof confirms the correct execution of an entire batch of transactions.

- **Proving**:
  $\pi{\text{Batch}}^b \leftarrow \text{Prove}(pk_{\text{Batch}}, a, w)$
  where:

  - $a = (S_0, SV_{m+1}^{(b)}, htx(b)_{0..m})$: public inputs
  - $w = (SV_0^{(b)}, \pi_{\text{Mem}^b}, \pi_{0..m})^{TxMerge}$: witness

**Interpretation:**
Proves that batch `b` transitions from the initial state $SV_0^{(b)}$ to final state $SV_{m+1}^{(b)}$ correctly, and that this process is consistent with the starting global state $S_0$.

**5) Global State Transition Proof (Global-State-Trans)**

This validates that the global state correctly updates after a batch is applied.

- **Proving**:
  $\pi_{\text{StateTrans}}^b \leftarrow \text{Prove}(pk_{\text{StateTrans}}, a, w)$
  where:

  - $a = (S_l, SV_{m+1}^{(b)}, S_{l+1})$

- **Verification**:
  $\text{Verify}(vk_{\text{StateTrans}}, (S_l, SV_{m+1}^{(b)}, S_{l+1}), \pi_{\text{StateTrans}}^b) \in \{\text{true}, \text{false}\}$

**Interpretation:**
Confirms that applying state changes in $SV_{m+1}^{(b)}$ to global state $S_l$ results in the correct next global state $S_{l+1}$.

**6) Final Batch Proof (Final-Batch-Proof)**

This aggregates the batch execution and global state transition proofs into a unified statement.

- **Proving**:
  $\pi_{\text{FinalBatch}^b} \leftarrow \text{Prove}(pk_{\text{FinalBatch}}, a, w)$
  where:

  - $a = (S_0, S_l, SV_m^{(b)}, S_{l+1}, h)$
  - $w = (\pi_{\text{Batch}}^b, \pi_{\text{StateTrans}}^b)$

**Interpretation:**
Combines the correctness of the batch’s execution and its application to the global state, serving as a compact and final proof of that batch's validity.

**7) No-Conflict Proof (No-Confl)**

This ensures that multiple batches do not interfere with each other.

- **Proving**:
  $\pi_{\text{NoConfl}} \leftarrow \text{Prove}(pk_{\text{NoConfl}}, htx_{0..n}, w)$

**Interpretation:**
Proves that transactions across different batches do not conflict, meaning:

- No two batches modify the same account.
- No batch reads data modified by another batch.

This condition is essential for safe **parallel execution** of multiple batches.

This hierarchical zkSNARK system enables scalable, secure, and verifiable execution in zk-powered blockchains, supporting high throughput and light client validation with succinct proofs.

---

### 1.2 Parallel-Enhanced Chains

Despite multiple rounds of scaling efforts, Ethereum's execution layer continues to face performance bottlenecks. In response, a new wave of _parallel-enhanced chains_—notably **Monad** and **MegaETH**—is emerging to address these challenges. These projects aim to boost throughput and reduce latency while maintaining compatibility with the Ethereum Virtual Machine (EVM), utilizing techniques such as _delayed execution_, _state partitioning_, and _asynchronous pipelines_. Their architectures are purpose-built for high-concurrency, high-throughput scenarios. Additionally, several projects are exploring various layers of parallelism to improve EVM execution:

- **Pharos Network** introduces a multi-tiered parallel architecture with a full-lifecycle asynchronous pipeline, dual virtual machine execution, modular subnets, and modular consensus mechanisms. Its _Pharos Store_ acts as a high-performance storage engine underpinning this architecture.

- **Reddio** combines zkRollup with GPU-based parallelism. It leverages multi-threaded scheduling and a CUDA-compatible EVM to support transaction- and operation-level parallelism.

- **GatlingX** proposes a radical _GPU-EVM_ architecture, dynamically compiling EVM bytecode into CUDA tasks to enable instruction-level parallelism (ILP), thus breaking the inherent sequential nature of the traditional EVM.

- **Artela** introduces the EVM++ model, integrating WebAssembly (WASM) to allow parallelism at the level of function and extension calls. This approach supports runtime injection of custom modules, offering modular composability and extensibility for complex applications.

Together, these initiatives represent a multi-layered stack of EVM parallelization—from instruction-level all the way to architectural modularity.

#### Monad: A High-Throughput Parallel EVM Chain

**Monad** represents one of the most advanced attempts to parallelize EVM execution without sacrificing compatibility. Its architecture is built around three core innovations:

- **Pipelined Execution**
- **Asynchronous Decoupling of Consensus and Execution**
- **Optimistic Parallel Execution**

The key idea is to decompose the traditional blockchain execution pipeline into distinct stages that can operate concurrently. Transactions are executed optimistically in parallel, with dynamic conflict detection to ensure correctness—all while making minimal changes to the EVM semantics.

**Asynchronous Execution**

Asynchronous execution in Monad decouples consensus from execution. Instead of requiring nodes to fully execute transactions before reaching consensus, Monad allows nodes to agree on transaction order first. Execution then proceeds in a separate pipeline, possibly delayed by a few blocks.

This decoupling significantly increases the available execution time per block. Unlike traditional blockchains—where execution time is tightly constrained due to its role in consensus—Monad allows the entire block time to be dedicated to execution, improving efficiency and scalability. Although execution lags behind consensus, final state transitions remain deterministic once consensus on transaction order is reached.

**Block Lifecycle and Merkle Root Synchronization**

Unlike Ethereum, Monad block proposals do not include the Merkle root of the current state. Instead, they reference the Merkle root from _D_ blocks prior, ensuring that all nodes remain synchronized. Each block moves through four phases: **Proposed**, **Voted**, **Finalized**, and **Verified**. Nodes may begin _speculative execution_ as soon as a block is proposed, enabling them to pre-compute state changes and update Merkle pointers early.

New accounts cannot initiate transactions until their balance modifications are verified, but smart contracts can combine balance updates with downstream actions to mitigate this delay.

**Defense Against Denial-of-Service Attacks**

To prevent DoS attacks (e.g., spamming the mempool with zero-balance transactions), Monad nodes verify during consensus whether each account has sufficient balance to cover the maximum possible expenditure for each transaction:

```
max_expenditure = value + gas_limit × max_fee_per_gas
```

Balances are tracked and decremented as transactions are added to blocks, ensuring that proposed transactions do not exceed available funds—even if full state views lag slightly behind.

**Maintaining EVM Compatibility**

Despite introducing parallel execution, Monad retains full compatibility with Ethereum’s sequential EVM semantics. Monad blocks contain linearly ordered transactions, and their execution results match those of Ethereum. This ensures seamless integration with existing tools and smart contracts.

To manage correctness under _optimistic execution_, Monad tracks the inputs and outputs of parallel transactions. If a later transaction reads data that was changed by an earlier transaction, the affected transaction is re-executed. While this incurs some overhead, many operations (e.g., signature recovery) do not depend on state and need not be repeated. Cached state can also reduce redundant computation during re-execution.

#### MegaETH: A Micro-VM Architecture for Maximal Parallelism

**MegaETH** takes a more radical approach to parallelism. Rather than optimistically layering parallel execution onto a sequential engine, MegaETH completely redefines the EVM execution model. Its architecture is centered around three key components:

- **Micro-VMs**: Each account or contract operates within its own isolated virtual machine.
- **State Dependency DAGs**: A directed acyclic graph tracks inter-account state dependencies.
- **Asynchronous Message Passing**: Transactions are coordinated through messages, not direct calls.

This model transforms the traditional monolithic EVM into a "multi-threaded" environment _within a single chain_. Unlike sharding—which scales horizontally by splitting the chain—MegaETH vertically scales the execution layer while preserving chain integrity and global state consistency.

Each Micro-VM operates independently, and the dependency DAG ensures safe execution order by resolving state interactions. This approach enables extremely high concurrency and low latency, especially in scenarios with minimal state overlap between accounts.

---

## 2. Parallel Execution in Blockchain Systems

Blockchain is a type of fault-tolerant distributed system, fundamentally built on **State Machine Replication (SMR)** and **consensus**. Each node functions as a state machine, executing transactions in the same order to maintain consistency. Traditional consensus protocols like **Nakamoto consensus** (used in Bitcoin and Ethereum 1.0) have limited throughput. To address this, modern blockchains are adopting **Byzantine Fault Tolerant (BFT)** consensus protocols—such as **HotStuff** and **PBFT**—which can process hundreds of thousands of transactions per second. However, a new bottleneck has emerged: **smart contract execution**.

Most BFT protocols are **leader-based**, where a single node coordinates the protocol. This centralized coordination limits scalability and becomes a performance bottleneck. To overcome this, **leaderless consensus** protocols are gaining attention. Next-generation blockchain systems such as **Sui**, **Hashgraph**, and **Conflux** adopt **DAG-based** or **asynchronous consensus**, eliminating the need for a single coordinator and theoretically enabling higher throughput. However, these systems still face a critical issue: **the execution engine remains serial**, which continues to limit overall system throughput. For example, Ethereum still only supports around **100 transactions per second (TPS)**, primarily due to the sequential nature of smart contract execution.

**Challenges in Parallel Execution**

The central challenge in parallel execution lies in handling **conflicting transactions**—those that access or modify the same part of the state. When such conflicts occur, careful scheduling is required to maintain correctness. A common strategy is to **execute non-conflicting transactions in parallel** while **serializing conflicting ones**, balancing performance with correctness.

Traditional concurrency models like **Order-Execute** and **Execute-Order-Validate** either:

- Perform serial execution after consensus, leading to redundant computation and missed parallelism, or
- Speculatively execute transactions before consensus, risking wasted computation in the case of conflicts or reordering.

These models are often **incompatible with leaderless consensus**, highlighting the need for **parallel execution frameworks tailored to decentralized, asynchronous environments**.

**Insights from Database Systems**

Techniques from database systems offer valuable insights for blockchain parallelism, including:

- **Reconnaissance Queries**: Probing likely data dependencies in advance.
- **Multi-Version Concurrency Control (MVCC)**: Maintaining multiple versions of state to allow concurrent reads and writes.
- **Optimistic Lock Location Prediction (OLLP)** from Calvin: Predicting data access patterns during execution to optimize scheduling.

These strategies help inform blockchain designs that can execute transactions in parallel and resolve conflicts after the fact.

Current parallel execution frameworks fall into three broad categories:

- **Explicit read/write sets**: Transactions declare which parts of the state they access. This approach is difficult to apply to general-purpose smart contracts.
- **Static analysis tools**: Automatically extract dependencies, but often use coarse granularity, leading to overly conservative scheduling and missed parallelism.
- **Optimistic Concurrency Control (OCC)**: Executes transactions in parallel without prior knowledge of conflicts, then detects and resolves conflicts during post-execution validation. While OCC can significantly improve parallelism, it suffers in high-conflict workloads due to frequent rollbacks.

**Key Techniques for Breaking the Storage Bottlenecks**

Even with effective parallel scheduling, **the storage layer often becomes the bottleneck**. This is because state access is sequential and suffers from **I/O amplification**, where even small state changes trigger multiple disk operations. Research shows that **disk I/O accounts for around 70% of execution overhead**, largely due to:

- The complexity of structures like the **Merkle Patricia Trie (MPT)** used in Ethereum for state storage.
- The need to read multiple trie nodes to access a single value.
- Additional writes required after execution to persist updated state.

To overcome these limitations, three key techniques have emerged to improve performance and scalability:

- **Direct State Access**: Allows high-speed access to state without traversing the MPT.
- **Asynchronous Parallel Node Loading**: Preloads trie nodes concurrently during execution to reduce I/O latency.
- **Pipelined Workflows**: Decouples execution, state reading, and storage updates into overlapping stages, maximizing hardware utilization.

These innovations address both execution and storage bottlenecks, forming the foundation for high-performance, scalable blockchain systems capable of supporting massively parallel transaction processing.

---

### 2.1 Concurrency Control Techniques in Database Systems

Database systems are among the earliest and most thoroughly developed applications of **parallel execution technologies**. In today's era of big data and cloud computing, databases face immense pressure from massive data volumes and increasingly complex queries. To meet performance demands in multi-core and distributed environments, leveraging parallel execution is essential for enhancing scalability, particularly in high-concurrency and large-scale data scenarios. This has become a major focus in database design and optimization research.

#### Scalability Challenges in Database Systems

Modern database systems encounter several challenges when scaling:

- **I/O Bottlenecks**: As data volume grows rapidly, disk I/O becomes a major performance limiter. A single storage device struggles to handle high-concurrency workloads, resulting in slower query response times.
- **Complex Query Overhead**: Operations such as joins, sorting, and aggregation are computationally expensive. Serial execution of such tasks further degrades performance.
- **Transaction Conflicts and Lock Contention**: With multiple users performing concurrent operations, transaction conflicts increase. Locking mechanisms can lead to longer wait times and reduced system throughput.
- **Resource Allocation and Fault Tolerance**: As the system scales, efficiently allocating computing resources and eliminating single points of failure becomes more complex and critical.

To achieve true scalability, database systems must address several key issues, including **data partitioning and distribution**, **load balancing**, **transaction management and consistency**, and **resource scheduling with dynamic scaling**. These challenges span across different system layers and are foundational to building high-performance, adaptable databases.

#### The Power of Parallel Execution

Parallel execution techniques significantly improve database scalability. By decomposing complex queries or transactions into smaller, independently executable units, databases can exploit multi-core processors and distributed architectures to run operations simultaneously—greatly reducing response time.

**Multi-threaded concurrency** also minimizes lock contention and boosts throughput. When paired with smart task allocation and load balancing, systems can maximize resource utilization while minimizing idle capacity. As a result, parallelism not only enhances processing capabilities and response speed but also equips databases to handle massive data volumes and high-concurrency scenarios more effectively.

Database systems use several parallelization strategies:

- **Query Plan Parallelism**: Complex operations—like joins, aggregations, and sorts—are broken down into sub-tasks and dispatched to different computing units for concurrent execution. For example, join operations can be processed in parallel by partitioning data and assigning chunks to multiple nodes, then merging the intermediate results.

- **Query Execution Parallelism**: Even a single query can be divided into smaller actions, such as distributed data scans, that are executed concurrently across processing units. Data partitioning is critical here to ensure even workload distribution and optimal efficiency.

Distributing data across multiple nodes alleviates the pressure on individual machines and improves overall throughput. There are two main forms:

- **Horizontal Partitioning**: Rows are split across nodes, each handling a different subset of the data.
- **Vertical Partitioning**: Columns are split, beneficial for queries that access only specific fields.

A simplified model of distributed execution time can be expressed as:

$T_{\text{total}} = \sum_{i=1}^{n} \frac{D_i}{P_i}$

Where $T_{\text{total}}$ is the total processing time, $D_i$ is the data size of partition $i$, and $P_i$ is the number of processing units handling it. Increasing node count improves parallel capacity and scalability, but also introduces new challenges such as **data synchronization** and **network communication overhead**.

Transaction synchronization is crucial for maintaining **data integrity and system reliability**, especially in multi-threaded contexts. Two major strategies are used:

- **Optimistic (Proactive) Concurrency Control**: Transactions proceed concurrently and are checked for conflicts during commit. If no conflicts are detected, changes are applied; otherwise, the transaction is rolled back and retried. This approach is effective when conflicts are rare.

- **Pessimistic (Reactive) Concurrency Control**: Locking mechanisms prevent concurrent modifications to the same data, reducing conflicts but potentially increasing lock contention under high load.

In distributed systems, the **Two-Phase Commit (2PC)** protocol is often used to ensure consistency, though it can incur significant performance overhead in high-latency environments. Effective parallel transaction processing must carefully balance **consistency and efficiency**.

To fully leverage modern hardware, tasks must be fine-grained and intelligently scheduled across cores or nodes. Key optimization areas include:

- **Task Decomposition**: Breaking workloads into smaller, manageable units that can be executed concurrently.
- **Resource Management**: Efficiently allocating CPU, memory, and I/O bandwidth to minimize contention and idle time.
- **Data Synchronization**: Ensuring consistency across tasks and preventing stale or conflicting reads/writes.

Achieving high scalability also depends on robust hardware—including **multi-core processors**, **high-capacity storage**, and **low-latency networks**—as well as software platforms that support **distributed processing**, **transactional control**, and **dynamic resource allocation**. Only through close integration of hardware and software can database systems fully realize the potential of parallel execution and scalable design.

---

### 2.2 Leaderless Parallel Execution in Blockchain Systems

In blockchain systems, consensus protocols can generally be categorized into two types: **leader-based** and **leaderless**. Leader-based protocols rely on a designated leader to coordinate the consensus process, such as disseminating proposals or collecting votes from other nodes. Well-known examples include **PBFT (Practical Byzantine Fault Tolerance)** and **HotStuff**. In PBFT, a single node serves as the leader until it fails or becomes unreachable, at which point a view change protocol is triggered to elect a new leader. HotStuff improves on this by proactively rotating the leader role, allowing each node to periodically propose blocks.

In these leader-based protocols, the leader typically has knowledge of prior proposals and can speculatively execute transactions before consensus is reached. It then embeds execution results and dependency information into the proposal. Upon receiving the proposal, other nodes re-execute the transactions in the proposed order, ensuring consistent results across the network.

By contrast, **leaderless protocols** eliminate the notion of a fixed leader, enabling any node to propose transactions. These protocols often pre-allocate sequence space to nodes or operate under asynchronous assumptions. The final output is a combination of all committed proposals. Since leaderless protocols allow for concurrent proposals, they can better utilize system resources (CPU, network bandwidth), balancing workloads across the network. However, this concurrency introduces challenges—namely, the lack of complete context for each proposer, which may result in inconsistencies if conflicting proposals are not fully accounted for.

#### Execution Models in Blockchain

Two primary execution paradigms dominate blockchain systems:

- **Order-Execute (OE)**: Transactions are first ordered through consensus, and then all nodes execute them in that order—either serially or with conservative parallelism. While simple and general-purpose, this approach limits performance due to unnecessary serialization of independent transactions.

- **Execute-Order-Validate (EOV)**: A designated node pre-executes and predicts a transaction order before consensus. Other nodes then validate by re-executing the transactions. This model supports greater parallelism but usually relies on a leader to predict dependencies and resolve conflicts. When discrepancies arise, rollbacks and conflict resolution mechanisms must be employed.

To overcome these limitations, blockchain systems can draw inspiration from database concurrency control techniques. Examples include:

- **Reconnaissance Queries**: Used to detect transaction dependencies early by splitting uncertain transactions into predictable read-then-write operations.
- **Multi-Version Concurrency Control (MVCC)**: Helps reduce conflicts by maintaining multiple versions of data.
- **Calvin’s Optimistic Lock Location Prediction (OLLP)**: Predicts locking positions to optimize scheduling.

These methods embody a key principle: **the effectiveness of parallel execution lies not in blind concurrency, but in early and accurate detection of inter-transaction dependencies**, enabling maximum concurrency while preserving determinism.

#### A Leaderless Execution Framework with Speculation

Building on this principle, a new execution paradigm is proposed that integrates speculative execution with leaderless consensus. Each node speculatively processes and analyzes transactions _before_ consensus, predicting dependencies without requiring prior knowledge of read/write sets. The execution proceeds in three main phases:

- **Speculation**: Each node speculatively executes its assigned transactions to gather data about read/write sets, conflicts, and execution traces. This maximizes concurrency and reveals dependencies early.

- **Ordering**: Based on the speculative results, nodes build a partially ordered execution plan (a DAG) that captures transaction dependencies. These plans are included in block proposals and fed into the consensus mechanism.

- **Replay**: Once consensus is achieved on a block and its execution plan, all nodes replay the transactions deterministically, following the dependency graph. Independent transactions can be processed in parallel, while conflicting ones follow the agreed order.

**Speculation Phase: Building the Dependency DAG**

The speculation phase aims to reduce unnecessary ordering constraints by revealing actual dependencies. The key steps include:

- **Parallel Execution**: Transactions are speculatively executed in parallel to collect their read/write sets.
- **Dependency Chain Construction**: Transactions that access the same keys are grouped into dependency chains. A single transaction can belong to multiple chains.
- **Sorting Chains**: Chains are sorted by length, with longer chains prioritized to maximize critical path execution early.
- **Transaction Reordering**: Transactions from longer chains are placed earlier in the final list to improve execution efficiency.
- **DAG Construction**: A directed acyclic graph is built based on the dependency types:

  - **Write-After-Write (WAW)**
  - **Write-After-Read (WAR)**
  - **Read-After-Write (RAW)**
    WAW dependencies take precedence when multiple dependencies exist between transactions.

A simplified version of the speculation algorithm:

<details><summary><b> Algorithm </b></summary>

```jsx
Algorithm 1: Speculation phase

Variables:
    BLOCK: a block containing a list of transactions
    DAG: a two-dimensional array storing dependencies
    access: a map from key to tx_chain

procedure Speculate():
    txs ← BLOCK.transactions
    ParallelExecute(txs)
    tx_chains ← SortDependencyChains(txs)
    txs ← 0
    for chain ∈ tx_chains do
        for tx ∈ chain do
            /* Adjust the order of txs */
            if tx /∈ txs then
                append(txs, tx)
    BLOCK.transactions ← txs
    DAG ← BuildDAG(txs)
    call Consensus(⟨BLOCK,DAG⟩)

function ParallelExecute(txs):
    for tx ∈ txs do
        /* Run in parallel mode */
        (tx.RS,tx.WS) ← execute(tx)

function SortDependencyChains(txs):
    for tx ∈ txs do
        for ∀ key ∈ tx.RS ∪ tx.WS do
            append(access[key], tx)
    for chain ∈ access do
        append(tx_chains, chain)
    Sort tx_chains from the longest to the shortest
    return tx_chains

function BuildDAG(txs):
    for tx ∈ txs do
        for ∀ tx′: tx′.idx < tx.idx do
            if tx′.WS ∩ tx.WS ≠ 0 then
                DAG [tx][tx′] ← WAW
                continue
            if tx′.RS ∩ tx.WS ≠ 0 then
                DAG [tx][tx′] ← WAR
            if tx′.WS ∩ tx.RS ≠ 0 then
                if DAG [tx][tx′] == WAR then
                    DAG [tx][tx′] ← WAW
                else
                    DAG [tx][tx′] ← RAW
    return DAG
```

</details>

#### Replay Phase: Deterministic and Efficient Execution

After consensus, each node replays the transactions deterministically in two subphases:

- **Phase 1**: Transactions are grouped into batches where intra-batch transactions do not conflict. These batches are executed in parallel, but their results are not applied until the whole batch completes. Transactions accessing previously unseen keys must wait.

- **Phase 2**: Transactions whose read/write sets diverged from the speculation phase are re-executed. This captures new dependencies that were not previously detected.

<details><summary><b> Algorithm </b></summary>

```jsx
Algorithm 2: Replay phase

Variables:
    TxsRe: a set of transactions to be re-executed

procedure Replay():
    (BLOCK,DAG) ← Consensus()
    txs ← BLOCK.transactions
    ready ← PopTxsBatch(txs)
    while ready = /0 do
        for tx ∈ ready do
            /* Run in parallel mode */
            (rs,ws) ← execute(tx)
            if rs = tx.RS ∨ ws ≠ tx.WS then
                append(TxsRe, tx)
            ready ← ready \ {tx}
        for tx ∈ ready do
            commit(tx)
        ready ← PopTxsBatch(txs)

procedure ReExecute():
    for tx ∈ TxsRe do
        execute(tx)
        commit(tx)

function PopTxsBatch(txs):
    ready_txs ← 0
    for tx ∈ txs do
        if tx has no WAW-dependencies then
            if no WAR ∨ no RAW then
                append(ready_txs, tx)
                txs ← txs \ {tx}
    return ready_txs

function commit(tx):
    Apply updates in tx.WS to the world state
```

</details>

#### Reducing Redundant Re-executions

Not all changes in read/write sets necessitate re-execution. A refined strategy is introduced to determine whether re-execution is truly needed:

- **New Conflict Keys**: If a transaction accesses a key not seen during speculation and which was also accessed by other transactions, re-execution is likely needed.
- **Non-conflicting Read of New Keys**: If a transaction reads a key unseen by others and that key is not newly written by others, it may proceed without re-execution.
- **Newly Written Keys**: If a transaction writes to a previously unseen key, it must wait until all transactions complete to ensure consistency.

This optimization significantly reduces unnecessary re-executions and improves overall performance.

<details><summary><b> Algorithm </b></summary>

```jsx
Algorithm 3: Improved replay strategy

Variables:
    all_keys: set of all keys accessed by transactions in speculation
    read: set of transactions that read new keys
    new_keys: set of newly written keys
    ready: set of ready transactions
    TxsRe: set of transactions to be re-executed

procedure ImprovedReplay():
    all_keys ← all keys accessed by tx in txs in speculation
    while ready = /0 do
        for tx ∈ ready do
            /* Run in parallel mode */
            (rs,ws) ← execute(tx)
            if tx reads/writes a new key key′ ∈ all_keys then
                append(TxsRe, tx)
                ready ← ready \ {tx}
                continue
            if tx reads a new key key′ /∈ all_keys then
                append(read, tx)
            if tx writes a new key key′ /∈ all_keys then
                new_keys ← new_keys ∪ {key′}
                read ← read \ {tx : tx has not read any key ∈ new_keys}
        commit(tx ∈ ready \ read) based on DAG
        for tx ∈ read do
            execute(tx)
            commit(tx)

```

</details>

In DeFi applications, transaction ordering directly impacts profit. During the speculation phase, proposers or miners may attempt **front-running** or **sandwich attacks** to maximize their own gains instead of overall throughput. To mitigate such adversarial behavior, existing solutions take two approaches:

- **Restrictive ordering rules**: Introduce constraints to enforce fairness, but at the cost of potential performance.
- **Incentive mechanisms**: Use economic incentives to encourage proposers to maximize concurrency and fairness, such as separating the roles of proposer and block builder.

---

### 2.3 MPT, Asynchronous Parallel Execution and Pipeline

**Merkle Patricia Trie (MPT)**

Blockchain systems use state databases to efficiently store and verify ledger state. Ethereum, for instance, adopts the **Merkle Patricia Trie (MPT)** as its core state structure. The MPT organizes all account and contract states into a prefix tree, where each node’s hash serves both as a lookup index and a cryptographic proof. The root hash ensures global state consistency. All node data is stored in an underlying key-value database. Users can verify the integrity and correctness of any state by obtaining a Merkle proof from a full node and recomputing hashes from the bottom up to compare with the block header’s root hash.

MPT combines both authentication and indexing functionalities and contains three types of nodes:

- **Branch Node**: Has up to 17 children—16 for each possible nibble (4-bit value) and one for storing the value when no further nibbles remain.
- **Leaf Node**: Stores a value along with the remaining encoded nibble path.
- **Extension Node**: Represents a shared nibble path segment pointing to a single child node, used to compress paths for efficiency.

Ethereum uses a **layered Merkle Patricia Trie** structure for its state database. At the top level, the **account trie** stores all accounts (addresses are hashed using Keccak-256). Each account includes a balance, nonce, `codeHash`, and `storageRoot`.

- **Externally Owned Accounts (EOAs)** contain only a balance and nonce.
- **Contract Accounts** contain executable bytecode and a separate **storage trie**.

Each contract's storage trie maps **hashed storage slots** (keys) to their corresponding values, ensuring separation and verifiability of contract state.

To query account-related data (e.g., balance or nonce), one can directly access the account trie. To access a contract variable, the system first retrieves the account's `storageRoot` from the account trie and then traverses the corresponding storage trie.

#### Asynchronous Parallel Node Loading in Ethereum

Ethereum currently processes transactions within a block sequentially to ensure deterministic state transitions across all nodes. While this guarantees global consensus and security, it significantly limits throughput. In contrast, traditional databases leverage parallel execution and serialization protocols to enhance performance. However, parallel execution in blockchains must adhere to **deterministic serializability**, meaning the outcome of parallel execution must match the result of a predefined sequential order.

Existing parallel execution strategies fall into two main categories:

- **Declared Access Sets**: Systems like FISCO BCOS require users to explicitly declare each transaction’s read/write sets. This is impractical for Ethereum’s general-purpose smart contracts.
- **Optimistic Concurrency Control (OCC)**: Transactions are executed in parallel against the same snapshot of the state. After execution, a conflict detection phase identifies and aborts transactions that violate serializability, which are then retried.

However, under real Ethereum workloads, these strategies provide limited speedup. This is mainly due to:

- The presence of hotspot variables, leading to frequent conflicts that force serial execution.
- The coarse granularity of most conflict detection methods, which often operate at the transaction level without deeper analysis of intra-contract state access patterns.

To address these limitations, a **batch-parallel execution strategy** combined with fine-grained conflict detection and rollback is employed. This ensures both high concurrency and deterministic execution.

The mechanism involves several key phases:

- Optimistic Execution: Each batch of transactions is executed independently on lightweight replicas of the global state database. Transactions optimistically assume no conflicts to maximize parallelism.

- Conflict Detection: After execution, a coordinator checks for conflicts. A conflict is defined as:

> If transaction $T_i$ writes to a state item, and a later transaction $T_j$ (where $i < j$) reads the same item, a **write-after-read conflict** occurs.

Since $T_j$ reads a stale value (not including $T_i$’s write), this violates serializability, and $T_j$ must be aborted.

- Conflict Resolution: The coordinator commits all non-conflicting transactions and merges their state updates into the global state, aborts and postpones conflicting transactions to the next execution round.

This process repeats iteratively until all transactions are successfully executed.

<details><summary><b> Algorithm </b></summary>

```jsx
Algorithm 1: Transaction batch fetching

Input:
    T: Transaction set in block Bl
    η: thread number
Output:
    Tbatch: A batch of transactions

procedure BatchFetch(T, η):
    Tbatch ← ∅
    for i ← 0; i < η; i ← i + 1 do
        Tpi ← NextTx(T, Tbatch)
        if Tpi ≠ null then
            Tbatch ← Tbatch ∪ {Tpi}
            T ← T \ {Tpi}
        else
            break
    if len(Tbatch) < η then
        Tbatch ← fill_Txs(Tbatch)
    return Tr

procedure NextTx(T, Tbatch):
    Tnext ← null
    for i ← 0; i < len(T); i ← i + 1 do
        Tpi ← T[i]
        r ← true
        foreach Tpj ∈ Tbatch do
            if explicit_conflict(Tpi, Tpj) is true then
                r ← false
        if r is true then
            Tnext ← Tpi
            break
    return Tnext
```

</details>

Transactions are selected for parallel execution based on the absence of explicit conflicts. If a full batch cannot be formed, the remaining slots are filled with the next available transactions, even if they might conflict.

<details><summary><b> Algorithm </b></summary>

```jsx
Algorithm 2: Parallel Transaction Execution

Input:
    Tbatch: Transaction batch
    S: Global state database
    T: Set of remained transactions
    R: Set of reads of executed transactions
    W: Set of writes of executed transactions
    Inext: Index of next transaction to be committed

Output:
    S: Global state after executing transactions in Tbatch
    W: Set of state updates written by successfully committed transactions
    R: Set of reads of executed transactions
    W: Set of writes of executed transactions
    Inext: Index of next transaction to be committed

procedure ParallelExecute(Tbatch, S, T, R, W, Inext):
    for i ← 0; i < len(Tbatch); i ← i + 1 do
        /* Execute each transaction on a separate thread */
        Tpi ← Tbatch[i]
        Spi ← light_Copy(Spi)
        Rpi, Wpi ← parallel_Execute(Tpi, Spi)
        R ← append(R, Rpi)
        W ← append(W, Wpi)

    /* Wait for all threads to terminate */
    wait()
    sort(R)
    sort(W)

    /* Merge states produced by all threads */
    W ← ∅
    for i ← 0; i < len(R); i ← i + 1 do
        if Rpi overlaps with Wp0,...,Wpi−1 then
            abort(Tbatch[i])
            T ← T ∪ {Tbatch[i]}
        else if pi = Inext then
            Smerge ← merge_State(Smerge, Wpi)
            W ← W ∪ Wpi
            R ← R \ {Rpi}
            W ← W \ {Wpi}
            Inext ← Inext + 1

    S ← Smerge
    return S, W, R, W, Inext
```

</details>

**Asynchronous State Database Architecture**

To further boost performance, especially for large-scale deployments, the **asynchronous state database** design decouples execution from storage access. This overcomes I/O bottlenecks caused by millions of accounts and contracts stored in a shared key-value database.

- **Direct State Reading (Ddirect)**
  Traditional Merkle Patricia Trie (MPT) traversal incurs high latency due to disk I/O. To mitigate this, a direct access database `Ddirect` is introduced, mapping accounts and contract storage keys directly to values (e.g., `<A, V>` or `<A||κ, v>`). This enables single-step retrieval without trie traversal, reducing latency significantly.

- **In-Memory State Cache (Cstate) and Async Loading (Qret)**
  All state updates are first written to `Cstate`, a fast in-memory cache. On a cache miss, the system falls back to `Ddirect` and asynchronously queues a node loading task in `Qret`. This allows computation to continue without blocking on I/O.

- **Consistency Challenges**
  While direct reads improve performance, they introduce consistency challenges with the canonical MPT structure. To address this, state commits are scheduled strategically to ensure consistency is eventually maintained.

<details><summary><b> Algorithm </b></summary>

```jsx
Algorithm 3: State database operation

Global:
    S: State database
    Cstate: Memory state cache
    Qret: Task queue

procedure Get(A, κ):
    v ← Cstate.get(A||κ)
    if v = null then
        Ddirect ← direct database in S for direct state reading
        v ← direct_Get(Ddirect, A||κ)
        Cstate.set(A||κ, v)
    return v

procedure Set(A, κ, v):
    Cstate.set(A||κ, v)
    o ← ⟨A, κ, v⟩
    Qret.Push(o)
```

</details>

**Asynchronous Node Retrieval**

To alleviate write-induced I/O stalls, a dedicated asynchronous node retrieval mechanism fetches required MPT nodes on demand.

- Write operations are buffered in `Cstate` and added to `Qret`.
- Background threads process `Qret`, traversing the account or storage trie to load required nodes from the underlying node database (e.g., LevelDB).
- Retrieved nodes are cached in memory (`Cnode`) to accelerate subsequent access.

This design leverages LevelDB’s support for concurrent reads to parallelize node loading and reduce execution wait times.

<details><summary><b> Algorithm </b></summary>

```jsx
Algorithm 4: Asynchronous node retrieval

Global:
    S: State database
    Cnode: Memory node cache
    Qret: Retrieval task queue

procedure AsyncNodeRetrieval():
    /* Run continuously on a separate thread */
    while true do
        o = ⟨A, κ, v⟩ ← Qret.Pop()
        Dnode ← node database in S for nodes
        Trieacc ← account trie in S
        V ← load_Nodes(Trieacc, A, Dnode, Cnode)
        if κ = null then
            return
        Triestorage ← storage_Trie(V, A)
        load_Nodes(Triestorage, κ, Dnode, Cnode)

procedure load_Nodes(Trie, key, Dnode, Cnode):
    next ← the root hash of MPT Trie
    node ← null
    while next ≠ null do
        if node next is not in Cnode then
            node ← node_Retrieve(Dnode, next)
            set_Node(Cnode, next, node)
        else
            node ← get_Node(Cnode, next)
        if node is the leaf then
            return the value in node
        next ← next_Node(node, A)
```

</details>

---

#### **Asynchronous Pipeline Workflow for Parallel State Updates in Blockchains**

Traditional state database updates in blockchains follow a strictly **sequential process** consisting of three stages:

- **Update** – Modify the content of nodes (e.g., account balances or contract storage).
- **Hash** – Recalculate the hash values along the updated path in the Merkle Patricia Trie (MPT).
- **Store** – Persist the updated nodes to the database.

This sequential dependency exists because a node's hash is determined by the hashes of its child nodes. Therefore, storage must wait until hashing is completed. Furthermore, when multiple transactions modify the same node repeatedly, redundant hashing and storage operations waste resources.

To overcome these bottlenecks, we introduce an **asynchronous pipeline mechanism** based on **task queues** that parallelizes the update, hashing, and storage phases. This architecture significantly enhances transaction execution throughput and state persistence efficiency.

The pipeline leverages “**commit points**” to safely determine when nodes can be hashed and persisted. Fine-grained locking and concurrency checks ensure correctness and avoid redundant computation. Separate strategies are designed for both the **account trie** and the **contract storage trie**, enabling decoupled and parallel execution of traditionally serial workflows.

The pipeline consists of **three asynchronous threads** and their respective task queues:

- **`AsyCommitAccount()`**: Identifies account trie nodes that have reached their commit point and pushes them to the `Qhash` queue.
- **`HashThread()`**: Continuously processes nodes from `Qhash`:

  - Computes their hash if all child nodes are ready.
  - Pushes the hashed nodes into the `Qstore` queue.
  - Recursively propagates to parent nodes if they are also ready to be hashed.

- **`StoreThread()`**: Serializes nodes from `Qstore` and writes them to the database.

These threads are coordinated by a **main controller thread**, which orchestrates execution and storage in an asynchronous and parallelized manner.

The execution pipeline operates as follows:

- Initialize necessary asynchronous threads: retrieval threads, commit threads, a hash thread, and a storage thread.
- The main loop fetches batches of transactions and distributes them to multiple worker threads for concurrent execution.
- After executing a batch, it collects the state changes.
- For each modified account:

  - If it will no longer be accessed by future transactions in the current batch, its updates are pushed into the `Qcommit` queue.

- The `AsyCommitAccount()` procedure triggers the next stages of hashing and storage asynchronously.

<details><summary><b> Algorithm </b></summary>

```jsx
Algorithm 5: Pipelined workflow in state database

Global:
    Qhash: Task queue for hash
    Qstore: Task queue for store
    S: Global state database

procedure AsyCommitAccount():
    N ← nodes in the account trie that have reached commit points without dirty children
    for N in N do
        Qhash.Push(N)

procedure HashThread():
    for true do
        N ← Qhash.Pop()
        for can_Hash(N) do
            if d ← Hash(N) is not null then
                set_Hash(N, d)
                Qstore.Push(N)
            else
                break
            N ← parent node of N

procedure StoreThread():
    Dnode ← node database in S for nodes
    for true do
        N ← Qstore.Pop()
        if ctx ← Ser(N) is not null then
            store_Node(Dnode, ctx, N)
```

</details>

By decoupling execution from storage, the system achieves **maximized parallelism** and **minimized I/O bottlenecks**.

<details><summary><b> Algorithm </b></summary>

```jsx
Algorithm 6: Framework of parallel transaction execution based on asynchronous storage

Global:
    T: Transaction set
    Cnode: Memory node cache
    Cstate: Memory state cache
    η: Number of worker threads
    ζr: Number of retrieve threads
    ζc: Number of commit threads

Input:
    S: State database
Output:
    S: New state database

procedure ParallelExecution(S):
    Init(AsyStateRetrieval, ζr)    /* Algorithm 4 */
    Init(AsyCommitStorage, ζc)
    Init(HashThread, 1)            /* Algorithm 5 */
    Init(StoreThread, 1)           /* Algorithm 5 */

    Inext ← 0
    R ← ∅
    W ← ∅
    S* ← S

    while len(T) = 0 do
        /* Algorithm 1 */
        Tbatch ← BatchFetching(T, η)

        /* Algorithm 2 */
        S*, W, R, W, Inext ← BatchParallelExec(Tbatch, S*, T, R, W, Inext)
        W ← update_Merge(W, W)

        Accs ← the set of distinguished accounts in W
        foreach A ∈ Accs do
            if A will not be accessed by transactions in T explicitly then
                UA ← updates to account A in W
                O ← ⟨A, UA⟩
                Qcommit.Push(O)

        AsyCommitAccount()         /* Algorithm 5 */
        wait_Storage_Commit()
        AsyCommitAccount()         /* Algorithm 5 */
        wait_Account_Commit()
        S ← S*

    return S

procedure AysCommitStorage():
    while true do
        A, UA ← Qcommit.Pop()
        TrieA ← the MPT (contract storage) for account A
        write_Updates(TrieA, UA)
        rootA ← TrieA.Commit()
        Trie ← the account trie
        update_Account(Trie, A)
```

</details>

**Correctness: Deterministic Serializability**

To ensure that parallel execution produces the same final state as sequential execution, the system guarantees **deterministic serializability**. This is achieved via:

- **Batch execution**
- **Read-write set detection**
- **Conflict rollback**
- **Safe merging**

These mechanisms ensure that each transaction is only committed if it has read from a consistent state. Every batch is guaranteed to make progress (at least one transaction is committed), ensuring that the system eventually and safely executes all transactions.

**Commit Point Estimation for Account Trie Nodes**

To decide whether a node can be committed (i.e., it will no longer be modified), the system estimates the commit point probabilistically.

Assume:

- The top layers of the account trie (e.g., top 4 layers) are fully populated.
- Each transaction modifies on average `µ` accounts.
- `m` transactions remain unexecuted.
- A node at level `r` has a probability of $16^{-r}$ to be modified in each update.

The probability that a node `N` at level `r` will **not** be modified in the remaining updates is:

$\Pr[X_r = 0 \mid u] \approx e^{-u / 16^r}, \quad \text{where } u = \mu m$

If this probability exceeds a threshold `α` (e.g., 0.9), the node is considered safe to commit. Therefore:

- For `r ≥ 4` and `u ≤ 4000`, the probability is already high — commit point is set at 4000.

- For `r ≤ 3`, solve:

  $m ≤ \frac{-16^r \ln α}{\mu}$

- For root levels `r ≤ 1`, commit point is conservatively set to 0.

**Note:** Even if a node reaches the commit point, it cannot be committed if it's a leaf node and its corresponding account is still accessed in future transactions.

**Crash Recovery Mechanism**

To maintain **consistency during crashes**, every state update not only records the key-value pair ⟨A||κ, v⟩ but also includes the **block height** `l`, forming a triplet ⟨A||κ, v, l⟩.

In the event of a node crash:

- The system replays blocks from persistent storage.
- During recovery, it discards any state updates with a block height greater than the current replayed block, ensuring consistency.

This ensures correctness of recovered states **without requiring a full resync of the entire database**, significantly improving fault recovery efficiency.

This asynchronous pipeline framework successfully decouples execution from storage, improves concurrency, and ensures correctness — enabling scalable and performant state management for next-generation blockchain systems.

[Monad](https://github.com/monad-developers)

[MegaETH](https://github.com/megaeth-labs)
