# ETAAcademy-ZKMeme: 60. Miden & Edge Computing

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>60. Miden & Edge Computing</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Miden & Edge Computing</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Miden: ZK and Edge Computing

Traditional blockchains face key challenges such as **execution bloat**, **state bloat**, and **privacy limitations**. **Edge blockchain technology**, which combines **edge computing** with **zero-knowledge proofs (ZKPs)**, offers a powerful solution. Edge computing reduces latency by placing computation and storage near data sources, while blockchain ensures a trustworthy, tamper-proof infrastructure. A **three-tier Edge–Cloud Collaborative Computing (ECCC)** architecture further enhances performance by blending the scalability of cloud computing with the responsiveness of edge nodes. Meanwhile, **ZKPs reshape blockchain processing** by shifting execution and state to the client, allowing nodes to verify lightweight proofs instead of re-executing transactions. This reduces resource demands and improves throughput. In systems like the **Miden edge blockchain**, efficient hash functions such as **GSponge** and **Sponge2** optimize proof generation, making the approach especially suitable for **resource-constrained environments**. This architecture maintains blockchain’s core values—**decentralization**, **immutability**, and **trustlessness**—while enabling **scalability and privacy**. It’s ideal not only for Miden VM, but also for other **STARK-based zero-knowledge systems**.

---

Traditional blockchains like Bitcoin, Ethereum, and Solana operate on a core principle: all network nodes must execute and verify every transaction. While this ensures strong security and consensus, it leads to three fundamental limitations:

- **Execution Bloat** – Requiring every node to re-execute all transactions is inherently wasteful and restricts throughput to the speed of the slowest participant.
- **State Bloat** – Nodes must store ever-growing amounts of state data, leading to scalability issues.
- **Lack of Privacy** – All transaction details are public by default, making privacy a systemic challenge.

To address these constraints, the convergence of blockchain with **edge computing** offers a compelling solution. Edge computing pushes computation and storage closer to the data source, reducing latency and improving responsiveness for applications running on devices or in distributed environments. When integrated with blockchain’s decentralized, transparent, tamper-proof nature, edge computing gains a trust layer for secure coordination among devices, data sharing, and resource management.

This integration also enables the automation of resource allocation and coordination through smart contracts, while distributed ledgers provide data integrity and transactional transparency. In particular, blockchain can resolve core challenges in edge environments—such as trust, auditability, and coordination—without reverting to centralized control.

**Edge-Cloud Collaborative Computing (ECCC)**

The **Edge-Cloud Collaborative Computing** paradigm leverages a three-tier architecture—cloud, edge, and end devices—to combine the computational power of the cloud with the low-latency responsiveness of edge nodes. This distributed intelligence framework balances resource usage through parallelism across tasks, data, and models. It dynamically manages critical resources, including computational power ($C_k$), communication bandwidth ($B_{ij}$), and storage capacity ($S_k$).

Advanced model optimization techniques are used throughout the system:

- **Compression methods**: pruning, quantization, and knowledge distillation
- **Adaptation methods**: transfer learning, federated learning, and continual learning
- **Architecture search**: using NAS (Neural Architecture Search) for efficient model design

To manage workloads efficiently, the system incorporates delay-aware, energy-efficient, and dependency-aware offloading strategies. Reinforcement learning and cooperative resource scheduling mechanisms ensure dynamic, adaptive performance. Privacy and data protection are embedded through differential privacy, secure multi-party computation (MPC), and data anonymization techniques.

Looking forward, research is expanding toward **next-generation AI** (e.g., AutoML, LLM optimization, neuromorphic computing) and **system-level improvements** (e.g., adaptive optimization, fault tolerance, and dynamic resource orchestration), addressing core challenges around model efficiency, integration, and performance bottlenecks.

**Zero-Knowledge Proofs Redefining the Execution Model**

The advent of **Zero-Knowledge (ZK) technology** presents a paradigm shift for blockchain architecture. With ZK proofs, it's no longer necessary for all network participants to execute and re-execute transactions to verify state validity. Instead, **execution and state storage can move to the edge**, or even the user device.

Here’s how it works:

- Users execute transactions locally and maintain their own state.
- A ZK proof of this execution is generated and sent to the network.
- The network only needs to verify a short, cryptographic proof (often just a 32-byte commitment), rather than replaying the transaction logic.

Because ZK proof verification is orders of magnitude faster than the computation it represents, the complexity of smart contracts becomes constrained by the **user’s hardware**, not the network's slowest validator. Even smartphones can now verify high-throughput, complex transactions in real time. Moreover, block producers can propose new blocks without needing to know or manage global state—tracking only public commitments and proof aggregation.

**Miden: A Case Study in Edge Blockchain Architecture**

Projects like **Miden**, a STARK-based ZK-rollup under the ecosystem, exemplify this edge-first design philosophy. By redefining “who executes” and “who stores” in blockchain systems, Miden aims to solve execution bloat, state bloat, and privacy leakage simultaneously—without sacrificing decentralization or self-custody.

Miden’s architecture leverages innovations like:

- **GSponge and Sponge2**: efficient hash constructions built into the Miden VM
- **Domain separation and unified sponge design**: secure yet performant approaches to hash varied inputs, such as metadata and Merkle trees

These cryptographic enhancements are mathematically proven and highly efficient, especially for zero-knowledge systems running in **resource-constrained environments**, like edge devices or embedded hardware. The innovations not only optimize Miden VM performance but are also generalizable to other STARK-based ZK systems.

---

## 1. Distributed Intelligence and Model Optimization in Edge-Cloud Collaborative Computing

Edge-Cloud Collaborative Computing (ECCC) represents a transformative evolution in distributed computing, seamlessly combining the high computational power of cloud computing with the low-latency responsiveness of edge computing. By leveraging a three-tier architecture—**cloud layer**, **edge layer**, and **end device layer**—ECCC departs from traditional centralized computing paradigms. In this architecture, the **cloud** serves as the backbone for large-scale data processing and model training, offering powerful compute, storage, and analytics capabilities. Meanwhile, the **end device layer** is responsible for data generation and consumption at the network edge. Given the cost implications of transmitting data from cloud to users, **edge computing** helps reduce these overheads by relocating data and processing closer to users via geographically distributed edge servers.

At the core of this paradigm lies a sophisticated resource management framework. Resource allocation across layers can be expressed using a function $D(R_i, L_j)$, and the trade-offs among performance, cost, and latency are modeled using a utility function \$U(P, C, L)\$. This design achieves several key benefits: latency and bandwidth usage are reduced by processing data near its source; scalability and fault tolerance are enhanced via distributed workloads; and collaborative resource sharing is enabled across nodes through federated edge intelligence—making the system ideal for mission-critical applications.

Beyond architectural evolution, ECCC embodies a conceptual shift toward distributed intelligence, offering an infrastructure that supports intelligent applications with **efficiency**, **flexibility**, and **reliability**. Its distributed intelligence framework operates on three core parallelism principles—**task parallelism** ($T = {t\_1, t\_2, ..., t\_n}$), **data parallelism** ($D = {d\_1, d\_2, ..., d\_m}$), and **model parallelism** ($M = {m\_1, m\_2, ..., m\_p}$)—to maximize resource utilization. It merges edge intelligence (for real-time, low-latency tasks) with cloud intelligence (for high-complexity workloads), leveraging smart model partitioning, dynamic resource scheduling, and model compression techniques to build an adaptive and high-performing computational ecosystem.

ECCC systems revolve around three primary resources: **computational power** ($C\_k$), **communication bandwidth** ($B\_{ij}$), and **storage capacity** ($S\_k$). These resources are constrained by physical and operational limitations, which directly impact system performance. To address the challenges of device heterogeneity, workload volatility, and limited energy (particularly for battery-powered edge devices), ECCC employs **dynamic resource allocation strategies** that aim to minimize task execution cost (e.g., latency), formalized as:

$\min \sum_{t \in T} \text{Cost}(M(t), t), \quad \text{subject to} \quad \forall n \in N: \sum_{t: M(t)=n} R(t) \leq C_n$

Energy efficiency is achieved through **strategic task offloading**, with energy consumption modeled as:

$E_{ij}(t) = E_{\text{comm}}(t, B_{ij}) + E_{\text{comp}}(t, C_j)$

Furthermore, **efficient edge storage management**—enabled by data caching and prefetching strategies—integrates Quality of Service (QoS) considerations to support diverse smart device scenarios. AI-powered techniques, such as model compression, adaptation, and resource-aware optimization, allow for intelligent distribution and execution of models across the edge-cloud continuum.

### 1.1 Model Optimization: Compression, Adaptation, Resource-Aware Optimization, and Neural Architecture Search

Model optimization in ECCC encompasses four major strategies: **compression**, **adaptation**, **resource-aware optimization**, and **neural architecture search (NAS)**.

#### Model Compression

Compression addresses the limitations of deploying complex AI models on resource-constrained edge devices. It includes:

- **Pruning**: Eliminating redundant parameters from neural networks.
- **Quantization**: Reducing numerical precision (e.g., from 32-bit floats to 8-bit integers).
- **Knowledge Distillation**: Training a smaller "student" model to replicate the behavior of a larger "teacher" model.

Successful implementations include:

- TSCF’s neuron pruning framework,
- Hybrid SD’s structured optimization of Stable Diffusion,
- QuAsyncFL’s integration of quantization into federated learning,
- GKT’s use of large language models to guide lightweight models.

These techniques align with the goal of optimizing the utility function \$U(P, C, L)\$, balancing performance, cost, and latency. Most approaches use both cloud and edge resources collaboratively, enabling efficient AI deployment across diverse domains—from healthcare to manufacturing.

#### Model Adaptation

Model adaptation is achieved via three learning paradigms:

- **Transfer Learning (TL)**: Uses pre-trained models to accelerate learning on new tasks, as seen in surface defect detection and edge-cloud cooperative environments.
- **Federated Learning (FL)**: Enables decentralized training on local data with global model aggregation, maintaining data privacy. Notable examples include FedCAE and QuAsyncFL.
- **Continual Learning (CL)**: Allows models to learn from sequential data without catastrophic forgetting, ideal for dynamic contexts such as digital twins and intelligent traffic systems.

#### Resource-Aware Optimization

This strategy is critical for maintaining performance amid fluctuating workloads and resource availability. Three complementary approaches include:

- **Dynamic Resource Allocation**: Adapts to workload and system changes in real time.
- **Collaborative Resource Management**: Coordinates resource use across edge nodes and cloud via hybrid and federated learning techniques.
- **Reinforcement Learning (RL)**: Learns optimal policies based on environmental feedback.

Examples include:

- **Dynamic Model Selection (DMS)**: Chooses the best-suited model from a candidate pool based on current system resources.
- **Model Partitioning and Offloading**: Splits models into subcomponents for strategic deployment, such as offloading compute-intensive parts to the cloud.
- **RL Techniques**: Approaches like Deep Q-Learning and APPO adapt model configurations and resource allocations to balance accuracy and computational cost.

#### Neural Architecture Search (NAS)

NAS automates the design of neural network architectures tailored for edge-cloud systems, using:

- **RL-based NAS**: Agents interact with an environment representing the architecture search space, receiving rewards based on model performance. These are especially effective for low-resource edge devices.
- **Evolutionary NAS**: Uses mechanisms such as selection, mutation, and crossover to evolve architectures. For example, GA-DPSO optimizes scientific workflows by minimizing data transmission in edge-cloud systems.
- **Gradient-based NAS**: Frames architecture search as a differentiable optimization problem, enabling efficient search via gradient descent (e.g., DARTS).

All three NAS approaches incorporate **resource constraints**—including compute, memory, and energy budgets—into their optimization objectives, enabling the design of adaptive models suited to dynamic workloads and environments.

### 1.2 Resource Management: Task Offloading, Allocation, and Energy Optimization

Edge-cloud collaborative computing has emerged as a pivotal paradigm for enabling intelligent services in latency-sensitive, resource-constrained environments. By leveraging AI technologies, this system architecture enhances resource efficiency, energy savings, and data privacy.

#### Task Offloading Strategies

In AI-driven edge-cloud systems, intelligent task offloading mechanisms optimize resources and improve overall performance through three primary strategies:

- **Latency-Aware Offloading**: Techniques such as Lyapunov optimization and joint service placement with request routing are used to minimize response times for time-sensitive applications. For instance, algorithms like iGATS and DRLIS have demonstrated up to a **33.42× reduction in task processing latency** by effectively reducing delays for sensitive workloads.

- **Energy-Efficient Offloading**: These strategies focus on balancing computation and communication energy costs, particularly for battery-powered devices. Approaches such as PDCO and ETCRA consider task complexity, network conditions, and hardware characteristics to achieve substantial energy savings without compromising performance.

- **Dependency-Aware Offloading**: For complex applications composed of interdependent subtasks, algorithms like PRAISE restructure the task call graph into sequential layers and model task dependencies using Directed Acyclic Graphs (DAGs) to optimize offloading decisions and resource allocation.

#### Resource Allocation Strategies

To manage dynamic workloads in distributed systems, resource allocation is optimized through:

- **Dynamic Resource Allocation**: This includes NUMA node selection, core consolidation, and service placement adaptation. It improves performance and energy efficiency in diverse scenarios, from drone communication networks to serverless distributed edge-cloud systems.

- **Collaborative Resource Management**: Resources are coordinated across edge nodes and the cloud, often leveraging multi-timescale optimization and federated learning principles. For example, OTFAC uses edge servers as local aggregators to reduce latency and employs hybrid scaling strategies to dynamically provision cloud resources.

- **Reinforcement Learning-Based Allocation**: By interacting with the environment and learning from performance rewards, RL-based methods optimize metrics such as latency, energy usage, and task completion time. Advanced techniques integrate Graph Neural Networks and Deep Q-Networks to model complex dependencies, while multi-agent RL supports decentralized learning and coordinated resource control.

#### Energy Optimization Dimensions

Energy efficiency is achieved across three key dimensions:

- **Power Consumption Modeling**: Accurate models incorporate device-level parameters (e.g., Arm Cortex-M0 frequency and idle states) and system-level demands (compute, communication, storage), enabling informed energy-aware decisions.

- **Energy-Aware Scheduling**: Dynamic Voltage and Frequency Scaling (DVFS) allows processors to adjust power settings based on real-time workloads. Algorithms like DVFO jointly optimize DVFS and offloading decisions for minimal inference energy consumption, while techniques such as Co-HDRL integrate sleep-wake cycles and green computing strategies.

- **Joint Optimization Techniques**: Multi-objective optimization (MOO) frameworks balance conflicting goals like minimizing delay and energy use. Solutions include differential evolution algorithms (e.g., Alopex) that account for time, cost, and load balancing, and hybrid workload collaboration models that reduce carbon emissions while improving system profitability.

### 1.3 Privacy and Security Enhancement Strategies

Ensuring data privacy and system security in edge-cloud AI systems is critical, especially given the heterogeneity of devices and data processing modes.

#### Privacy-Preserving Learning

A comprehensive privacy framework integrates:

- **Federated Learning with Differential Privacy**: Combines secure aggregation with noise injection (e.g., zCDP, CaPC) to protect individual contributions while supporting collaborative AI development.

- **Secure Multi-Party Computation (SMPC)**: Enables privacy-preserving joint computation among multiple entities (e.g., hospitals) without sharing raw data. Frameworks like FlexSMC manage dynamic environments and use coding-based SMPC (e.g., CMPC) for enhanced efficiency.

- **Differential Privacy Techniques**: Add calibrated noise to protect intermediate outputs and local model updates in federated learning, ensuring meaningful insights while preserving user anonymity.

- **Data Anonymization**: Tools such as StyleID decouple identity from visual data, while adaptive anonymization techniques dynamically tailor privacy protection based on environmental and dataset characteristics.

#### AI Model Reliability

AI model integrity is ensured through multiple mechanisms:

- **Model Integrity Protection**: Utilizes cryptographic hashes, code signing, and Trusted Execution Environments (TEEs) to authenticate model origins and secure execution.

- **Attack Detection and Defense**: Defenses against adversarial inputs include anomaly detection and adversarial training. Federated learning aids in distributed defense against model poisoning via secure aggregation and contribution evaluation.

- **Secure Model Updates**: Digital signatures, encrypted channels, and rollback mechanisms safeguard model updates during transmission and execution.

- **Trust Management Systems**: Reputation frameworks track entity behavior using both quantitative metrics (e.g., latency, accuracy) and qualitative assessments (e.g., user feedback), with custom trust standards tailored to specific application domains (e.g., high reliability for autonomous driving, high privacy for personalized healthcare).

#### Secure Communication

Secure communication is achieved through:

- **Encrypted Data Transmission**: Combines symmetric (high-speed) and asymmetric (high-security) encryption, supported by protocols like TLS/SSL, VPNs, and blockchain-based secure channels.

- **Authentication and Access Control**: Multi-Factor Authentication (MFA) and Role-Based Access Control (RBAC) ensure robust identity verification and flexible permission management. Systems like DOACIR and P-MOD enable controlled sharing and tiered access.

- **Attribute-Based Encryption**: Techniques such as Ciphertext-Policy Attribute-Based Encryption (CP-ABE) enforce fine-grained access control based on user attributes.

- **Blockchain-Based Security**: Decentralized architecture eliminates single points of failure, guarantees data integrity, provides transparent audit trails, and supports smart contract-based enforcement of security policies.

**Current Challenges and Future Research Directions**

Despite advancements, several technical challenges remain:

- **Model Optimization**: Adapting large models (e.g., LLMs) to resource-constrained edge devices requires techniques like pruning, quantization, and knowledge distillation to maintain accuracy while reducing complexity.

- **System Integration**: Managing hardware and software heterogeneity demands seamless interoperability. Technologies such as containerization and WebAssembly enhance cross-platform compatibility.

- **Performance Bottlenecks**: Latency and bandwidth constraints, limited edge computing and memory capacity, and real-time requirements pose challenges. Solutions include task offloading, data compression, distributed storage (e.g., EdgeKV), and adaptive resource allocation.

Future research is expected to advance in two major areas:

- **Advanced AI Techniques**: These include Neural Architecture Search (NAS), AutoML, optimized LLMs for edge inference (e.g., AgileQuant, EdgeMoE), autonomous AI agents, embodied AI for constrained devices, neuromorphic computing for energy efficiency, quantum edge computing, and 6G integration for ultra-low latency and high bandwidth.

- **System-Level Improvements**: Innovations will center on dynamic resource management (e.g., ETCRA algorithm), adaptive optimization (e.g., Shoggoth’s minibatch training, TIGO for demand adaptation), and fault tolerance (e.g., DeepFT for failure prediction, CAROL for resilient node management).

---

## 2. Miden: A New Architecture for Edge-Optimized, Privacy-Preserving Blockchains

At the heart of the Miden blockchain lies the Miden Virtual Machine (Miden VM), a zero-knowledge proof-optimized virtual machine built to support any language that compiles to WebAssembly, with Rust being the primary language for writing smart contracts. Miden introduces a fundamentally novel architecture by marrying the Actor model—a well-established framework for building concurrent and distributed systems—with zero-knowledge proofs (ZKPs). In this design, each actor operates as a lightweight state machine that updates its internal state and communicates by sending and receiving messages. Miden adapts this model to the blockchain environment by treating each account as an autonomous state machine and defining a hybrid state model that combines the account-based structure of Ethereum with the note-based (UTXO-like) paradigm of Bitcoin and Zcash.

### 2.1 Accounts and Notes

In Miden's system, the two fundamental units are **accounts** and **notes**. Notes serve as the medium for asset transfer and interaction between accounts. An account’s state changes by consuming and producing notes. The global state of the blockchain is distributed across three specialized cryptographic databases:

- **Account Database**: A tiered sparse Merkle tree that stores the latest state of every account.
- **Note Database**: An append-only Merkle Mountain Range that holds all notes ever created.
- **Nullifier Database**: A tiered sparse Merkle tree that tracks which notes have been spent.

Communication between accounts is entirely mediated through notes, which are not merely containers of value but programmable entities with Turing-complete **spend scripts**. These scripts define the conditions under which a note can be spent, effectively acting as on-chain expressions of intent.

A key innovation in Miden is how it decomposes asset transfers into two independent, local transactions. The first transaction, executed by the sender, creates a note and moves assets into it without modifying the recipient’s account. The second transaction, executed by the receiver, consumes the note and transfers the assets into their own account. This decoupling ensures that neither party’s transaction requires immediate synchronization with the other, enabling **truly concurrent and local execution** of transactions—a foundational requirement for scalable edge execution.

Miden supports two account models:

- **Public Accounts**, where the full state is stored on-chain and openly visible.
- **Private Accounts**, where only a 32-byte cryptographic commitment is stored on-chain, significantly reducing state bloat even if large amounts of data are associated with the account off-chain.

Users interact with public network accounts—such as an AMM (Automated Market Maker) smart contract—via private accounts by locally generating notes that encode their intent. These intent notes are later processed by the network, which consumes them in a network-level transaction to complete the interaction. The network then issues new output notes (e.g., with returned or swapped assets) back to the user. This model enables efficient, privacy-preserving, and decentralized interaction without imposing state-update costs on the smart contract or requiring real-time coordination between parties.

**Miden’s Revolutionary Approach to State Growth and Privacy in Blockchain Systems**

Miden introduces a groundbreaking strategy for managing state growth by shifting the burden of data storage away from network operators and onto users. Instead of maintaining full account states on-chain, the system requires only cryptographic commitments to be stored. This allows block producers and verifiers to operate without access to complete state data. For example, a private account in Miden occupies just 40 bytes of on-chain storage—meaning even with one billion accounts, the total storage requirement would be only around 40 GB.

This lean model is made viable through zero-knowledge proofs, enabling efficient verification of state transitions without exposing underlying data. To manage database growth, Miden introduces an **epoch-based nullifier system**, which prunes spent notes in a controlled manner, and leverages a **concurrent state model** that simplifies block production. The system’s split-database architecture supports **client-side proving**—notes' proof paths remain valid even as the global state evolves. Furthermore, Miden enhances privacy by breaking the linkability between note creation and consumption. Only parties who possess complete note details can determine if a note has been spent. This feature creates a positive feedback loop between **maximum privacy** and **minimal state growth**, reinforcing Miden’s unique Actor-based off-chain concurrent state model as a scalable and privacy-preserving blockchain solution.

#### Redefining Blockchain Privacy

In traditional blockchains, transaction validation requires full re-execution, necessitating transparency and leaving little room for privacy. This requirement also acts as a bottleneck for scalability. Miden departs from this paradigm by offering a **layered framework for privacy**, classifying privacy into four tiers:

- **No Privacy** – Fully transparent; all data accessible to everyone.
- **Web2-like Privacy** – Visibility limited to transaction participants and operators.
- **Strong Privacy** – Only transaction participants have complete visibility.
- **Absolute Privacy** – No party has full access to transaction data.

Miden explicitly targets the **strong privacy** level, where only senders and receivers can view transaction contents. To support this, Miden offers three data storage strategies for accounts and notes:

- **Public Storage**: All data is on-chain and visible to all—zero privacy but full transparency.
- **Encrypted Storage**: Data is stored on-chain but encrypted; visible only to those with decryption keys. While it preserves privacy, it offers limited scalability benefits.
- **Off-chain Storage**: Only cryptographic commitments are stored on-chain, while actual data resides off-chain. This mode combines high privacy and state efficiency.

When private data storage (whether encrypted or off-chain) is paired with **local transaction execution**, transaction details remain visible only to direct participants. Advanced privacy tools, such as **relay accounts**, can even obscure the sender’s identity from the receiver.

#### Privacy Challenges and Miden’s Progressive Model

While Miden offers robust privacy guarantees, it still faces challenges—especially with public notes. Observers can analyze the **nullifier database**, which contains unique identifiers for spent notes, to infer who spent a given note, including sender, receiver, and asset details. In contrast, notes stored off-chain expose only a hash and (optionally masked) sender address, making it practically impossible for outside parties to determine nullifier details or spending status.

Another concern arises from **operator visibility**. Since operators may see user requests, a potential solution is for clients to request more notes than needed and apply local filtering to obscure intentions. Managing off-chain state securely and efficiently is also a non-trivial problem. Currently, Miden operates in what it calls the “**privacy training wheels**” phase. Clients must submit all transaction data and proofs, offering **Web2-level privacy** for now. However, the project roadmap includes steadily increasing privacy guarantees, paving the way toward stronger protections and scalable, decentralized computing.

---

### 2.2 GSponge & Sponge2 in Miden VM

Cryptographic hash functions are fundamental to modern cryptography, providing fixed-size outputs from variable-length inputs. Their roles span from ensuring data integrity and verifying digital signatures to authenticating transactions. In recent years, their utility has expanded into blockchain scaling solutions, particularly Layer 2 protocols like **Miden**. At their core, hash functions emulate random oracles by combining idealized cryptographic primitives—random functions, permutations, and block ciphers—to produce outputs that are collision-resistant, pre-image resistant, and pseudorandom.

Hash function strategies fall into two primary categories:

- **Tree-based constructions** (e.g., Merkle Trees), which allow parallel processing but require large internal states.
- **Cascade-based constructions**, like the Merkle–Damgård and **Sponge constructions**, which process inputs sequentially and maintain compact state representations.

The **Sponge construction** has become a standard in modern cryptography. It divides internal state into two parts: a **capacity** $c$ ensuring security, and a **rate** $r$ determining throughput. Input data is absorbed through the rate portion, with various transformations applied to produce a secure digest. The sponge model has since evolved into several variants—such as **Overwrite Sponge**—and has been extended beyond binary domains into **finite fields**, supporting algebraic-friendly hash functions like **Rescue**, **Poseidon**, and **XHash**.

However, one of the major challenges in sponge-based hashing is handling inputs of arbitrary lengths efficiently. Traditional padding methods, such as the commonly used **pad10**, are simple but inefficient. They add extra permutation rounds even when inputs are perfectly aligned, which leads to unnecessary overhead—especially problematic for small messages. This inefficiency is precisely what the **GSponge** family of hash functions aims to resolve.

In Miden’s zero-knowledge virtual machine (zkVM), efficient and secure hashing is essential for transaction execution, proof generation, and state updates. The **GSponge** and **Sponge2** constructions were introduced to optimize performance and simplify security proofs within the context of finite field arithmetic.

Unlike traditional sponge models that rely on general-purpose padding, **GSponge** is designed with **domain-specific padding** strategies that avoid unnecessary rounds. It achieves optimal performance for short messages while maintaining strong cryptographic properties. Moreover, **GSponge** offers an important advancement in the realm of **indifferentiability proofs**, a modern security criterion for hash function soundness.

The security of hash constructions is typically established using a **two-step approach**:

- **Generic Security Proof** – Demonstrate that the hash structure is secure when built from ideal primitives.
- **Instantiation** – Replace the ideal primitives with concrete, secure algorithms.

However, generic security is not enough. Even if a hash function uses a perfect permutation, **structural vulnerabilities** can be exploited through what's known as **generic attacks**. Classic examples include:

- **Joux’s multicollision attack**
- **Kelsey–Schneier's second pre-image attack**
- **Kelsey–Kohno's herding attack**

While these attacks originally targeted Merkle–Damgård constructions, they also apply to other structured hash functions. To defend against these threats, cryptographers introduced the concept of **indifferentiability**, developed by **Maurer et al.** and later formalized by **Coron et al.** This notion assesses how well a hash function emulates a **random oracle** when viewed by any adversary with oracle access to both the function and the ideal primitive.

Indifferentiability proofs require the construction of a **simulator** that bridges the ideal primitive and the random oracle. This simulator must produce indistinguishable outputs in both directions, a task that is highly non-trivial. Even minor changes to the hashing structure can demand an entirely new proof, making this an intricate and error-prone process.

**GSponge**, however, is designed with **proof simplicity in mind**. It simplifies simulator design and reduces the burden of demonstrating indifferentiability. This is a crucial advantage in real-world applications like **Miden**, where new hashing structures must frequently adapt to performance constraints without compromising security guarantees.

#### GSponge: A Unified Framework for Sponge-Based Hash Constructions

**GSponge** is a powerful and general framework for sponge-based constructions, designed to unify and generalize a wide range of existing sponge variants. It is built upon a permutation function π and an injective padding function, and is parametrized by two configurable parameters: $u \in \{0,1\}$ and $r_0$. The formal definition of the construction is:

$\text{GSponge}_{(u, r_0)}[\pi] : \mathbb{F}_p^* \rightarrow \mathbb{F}_p^r$

where $\pi : \mathbb{F}_p^b \rightarrow \mathbb{F}_p^b$ is a permutation over a state of size $b = r + c$, with $r$ and $c$ denoting the _rate_ and _capacity_, respectively.

The input message $M$ of arbitrary length is first processed by the padding function $\text{pad}_{(r, r_0)}$ to produce a structured vector $P$. This padded input is of the form $[x, M, y]$, uniquely defined for each message and configured by $r_0 < c$, where $x$ and $y$ are fixed paddings added to ensure well-defined input segmentation.

The parameter $u$ controls whether the rate portion of the permutation’s output is fed forward into the next permutation input (chaining). This single switch determines whether the mode operates as a standard sponge or as a “rewrite” sponge.

Thanks to the flexibility of its parameters, **GSponge** can precisely replicate numerous known constructions:

- When $u = 1$, $r_0 = 0$, $p = 2^n$, and a specific padding rule is used:

  $\text{pad}_{(r,0)}(M) = [\langle 0 \rangle, M, \langle 1 \rangle, \langle 0 \rangle^{r - 1 - (|M| \mod r)}]$

  GSponge behaves identically to the _original sponge construction_.

- Changing $u$ to 0 under the same conditions gives the _original rewrite mode_.

- Removing the restriction $p = 2^n$ allows modeling of _algebraic sponge_ and _algebraic rewrite_ constructions.

This unified formulation not only simplifies the interpretation and comparison of various sponge-based designs, but also provides a shared theoretical foundation for their security proofs—bridging gaps where some sponge variants previously lacked formal analysis.

The primary security objective of GSponge is _indifferentiability from a random oracle_, which—when combined with a sufficiently large output size—naturally implies key hash properties such as preimage resistance and collision resistance.

One of the key insights of the GSponge framework is that its security analysis can be substantially simplified by abstracting away two seemingly significant parameters:

- **The Chaining Parameter $u$**
  It turns out that $u$ has no impact on the statistical distribution of outputs; although different values of $u$ lead to differently _ordered_ outputs, their distributions are identical. Thus, from a security standpoint, they are indistinguishable.

- **The Padding Function**
  All injective padding functions over the same message space are pairwise bijective, which ensures that any two GSponge instances with different injective paddings (but same $u$ and $r_0$) are statistically equivalent. That is, their outputs have a statistical distance of zero.

These two findings allow the reduction of the overall security proof to just one configuration:

> **It suffices to prove that $\text{GSponge}_{(0, r_0)}$ with an arbitrary injective padding is indifferentiable from a random oracle**.

This major simplification reduces the complexity of security proofs and ensures that **all configurations** of GSponge inherit this robustness, laying a solid theoretical foundation for its practical instantiation—**Sponge2**.

#### Sponge2: An Efficient Instantiation of the GSponge Framework

**Sponge2** is an optimized instantiation of the GSponge framework, designed for performance-critical environments. Built over a permutation function $\pi: \mathbb{F}_p^b \rightarrow \mathbb{F}_p^b$, it introduces two key innovations through a novel domain separation mechanism and padding rule:

- **Enhanced Input Capacity**: It can absorb an additional $r_0 < c$ field elements beyond the standard rate $r$, effectively extending the capacity of the first permutation call.
- **Reduced Permutation Calls**: For rate-aligned messages, Sponge2 eliminates the need for a final permutation call, reducing computational overhead.

These optimizations are made possible by a **dynamic domain separator** $i$, which is defined as follows:

- For short messages: $i = r + r_0 - |M|$
- Otherwise: $i = (r - ((|M| - r_0) \mod r)) \mod r$

This adaptive separator guarantees unique and unambiguous processing of messages of varying lengths, preventing collisions and ambiguity in interpretation.

Sponge2 maintains strong theoretical guarantees by being _indifferentiable from a random oracle_, with a provable upper bound on the adversary’s advantage:

$\text{Adv}^{\text{ro-indiff}}_{\text{Sponge2}[\pi]}(A) \leq \frac{q_P(q_P-1)}{2p^b} + \frac{q_P}{p^{c - r_0}}\left(1 + \frac{1}{p-1}\right) + \frac{q_P^2}{p^c} \left(1 + \frac{p^{-c + r_0 + 1}}{p - 1 - p^{-c + r_0 + 1}}\right)$

Here, $\pi$ is chosen uniformly at random from the set of permutations over $\mathbb{F}_p^b$, $r_0 < c$ is a fixed constant, and $q_P$ is the number of queries made by the adversary.

In fields of **odd characteristic** $(p > 2)$, the bound simplifies significantly when $r_0 = c/2$:

$\text{Adv}^{\text{ro-indiff}}_{\text{Sponge2}[\pi]}(A) \leq \frac{3q_P}{p^{c/2}}$

This means Sponge2 achieves at least $(c \cdot \log_2 p - 4)/2$ bits of security. For example, when $p \approx 2^{64}$ and $c = 4$, this yields approximately **126 bits** of security. Such performance-to-security trade-offs make Sponge2 particularly well-suited for environments with strict resource constraints and small message sizes.

**Performance in Miden VM**

Sponge2 has demonstrated **significant performance gains** when implemented in the Miden virtual machine by replacing the currently used **Rescue-Prime Optimized (RPO)** hash function. RPO, itself a permutation-based sponge construction over $\mathbb{F}_p^b$ with $p$ a 64-bit prime and parameters $r = 2c = 8$, is widely employed across various Miden VM hashing tasks.

Sponge2 offers improvements across **three primary use cases**:

- **2-to-1 Hashing with Metadata**
  Traditional sponges require two permutation calls to hash 9 elements (8 data + 1 metadata). Sponge2, with $r_0 = r + c/2 = 10$, handles all 9 in a **single permutation**, yielding a **50% performance boost**.

- **Leaf Hashing in Merkle Trees**
  For a leaf consisting of 64 elements, traditional sponges require $\lceil 64/8 \rceil = 9$ calls. Sponge2 reduces this to **8 calls**, achieving a **12.5% efficiency gain**.

- **Variable Input Length (VIL) Hashing**
  For messages of the form $r \cdot \alpha_M + i_M$ (where $0 \leq i_M \leq r-1$), Sponge2 saves one permutation call whenever the message satisfies $0 \leq i_M \leq r_0 - r$ and $\alpha_M > 0$. This yields a **speedup of $100 / \alpha_M$%** in such cases. For uniformly sampled messages, this condition occurs with **probability $100 \cdot (r_0 - r + 1)/r \% = 37.5\%$** (given $r_0 = 10, r = 8$).

These mathematically grounded improvements make Sponge2 not only ideal for **Miden VM** but also **broadly applicable across STARK-based zero-knowledge proof systems** and **resource-constrained cryptographic environments**. By preserving the robust security guarantees of GSponge while significantly improving performance, Sponge2 serves as a compelling drop-in replacement for existing sponge constructions in performance-sensitive applications.

<details><summary><b> Code </b></summary>

<details><summary><b> miden-objects/src/account </b></summary>

```rust
// crates/miden-objects/src/account/mod.rs
pub struct Account {
    id: AccountId,
    vault: AssetVault,
    storage: AccountStorage,
    code: AccountCode,
    nonce: Felt,
}

impl Account {
    pub fn from_parts(
        id: AccountId,
        vault: AssetVault,
        storage: AccountStorage,
        code: AccountCode,
        nonce: Felt,
    ) -> Self {
        Self { id, vault, storage, code, nonce }
    }

    pub fn commitment(&self) -> Digest {
        hash_account(
            self.id,
            self.nonce,
            self.vault.root(),
            self.storage.commitment(),
            self.code.commitment(),
        )
    }

    pub fn apply_delta(&mut self, delta: &AccountDelta) -> Result<(), AccountError> {
        self.vault
            .apply_delta(delta.vault())
            .map_err(AccountError::AssetVaultUpdateError)?;

        self.storage.apply_delta(delta.storage())?;

        if let Some(nonce) = delta.nonce() {
            self.set_nonce(nonce)?;
        }

        Ok(())
    }
}
```

```rust
// crates/miden-objects/src/account/delta/mod.rs
pub struct AccountDelta {
    storage: AccountStorageDelta,
    vault: AccountVaultDelta,
    nonce: Option<Felt>,
}

impl AccountDelta {
    pub fn new(
        storage: AccountStorageDelta,
        vault: AccountVaultDelta,
        nonce: Option<Felt>,
    ) -> Result<Self, AccountDeltaError> {
        validate_nonce(nonce, &storage, &vault)?;
        Ok(Self { storage, vault, nonce })
    }

    pub fn merge(&mut self, other: Self) -> Result<(), AccountDeltaError> {
        match (&mut self.nonce, other.nonce) {
            (Some(old), Some(new)) if new.as_int() <= old.as_int() => {
                return Err(AccountDeltaError::InconsistentNonceUpdate(format!(
                    "new nonce {new} is not larger than the old nonce {old}"
                )));
            },
            (old, new) => *old = new.or(*old),
        };
        self.storage.merge(other.storage)?;
        self.vault.merge(other.vault)
    }
}
```

```rust
// crates/miden-objects/src/account/storage/mod.rs
pub struct AccountStorage {
    slots: Vec<StorageSlot>,
}

impl AccountStorage {
    pub const MAX_NUM_STORAGE_SLOTS: usize = 255;

    pub fn new(slots: Vec<StorageSlot>) -> Result<AccountStorage, AccountError> {
        let num_slots = slots.len();

        if num_slots > Self::MAX_NUM_STORAGE_SLOTS {
            return Err(AccountError::StorageTooManySlots(num_slots as u64));
        }

        Ok(Self { slots })
    }

    pub fn commitment(&self) -> Digest {
        build_slots_commitment(&self.slots)
    }

    pub fn get_item(&self, index: u8) -> Result<Digest, AccountError> {
        self.slots
            .get(index as usize)
            .ok_or(AccountError::StorageIndexOutOfBounds {
                slots_len: self.slots.len() as u8,
                index,
            })
            .map(|slot| slot.value().into())
    }
}
```

```rust
// crates/miden-objects/src/account/mod.rs
pub fn hash_account(
    id: AccountId,
    nonce: Felt,
    vault_root: Digest,
    storage_commitment: Digest,
    code_commitment: Digest,
) -> Digest {
    let mut elements = [ZERO; 16];
    elements[0] = id.suffix();
    elements[1] = id.prefix().as_felt();
    elements[3] = nonce;
    elements[4..8].copy_from_slice(&*vault_root);
    elements[8..12].copy_from_slice(&*storage_commitment);
    elements[12..].copy_from_slice(&*code_commitment);
    Hasher::hash_elements(&elements)
}
```

</details>

<details><summary><b> miden-objects/src/asset </b></summary>

```rust
// crates/miden-objects/src/asset/mod.rs
pub enum Asset {
    Fungible(FungibleAsset),
    NonFungible(NonFungibleAsset),
}

// crates/miden-objects/src/asset/fungible.rs
pub struct FungibleAsset {
    value: Word,
}

impl FungibleAsset {
    pub fn new(faucet_id: AccountIdPrefix, amount: u64) -> Result<Self, AssetError> {
        if amount > MAX_FUNGIBLE_ASSET_AMOUNT {
            return Err(AssetError::FungibleAssetAmountTooLarge(amount));
        }

        let value = [
            Felt::new(amount),
            ZERO,
            faucet_id.suffix(),
            faucet_id.as_felt(),
        ];

        Ok(Self { value })
    }

    pub fn amount(&self) -> u64 {
        self.value[0].as_int()
    }
}

// crates/miden-objects/src/asset/vault.rs
pub struct AssetVault {
    assets: BTreeMap<Word, Word>,
    root: Digest,
}

impl AssetVault {
    pub fn add_asset(&mut self, asset: Asset) -> Result<(), AssetError> {
        let key = asset.vault_key();

        match asset {
            Asset::Fungible(asset) => self.add_fungible_asset(asset)?,
            Asset::NonFungible(asset) => self.add_non_fungible_asset(asset)?,
        }

        self.update_root();
        Ok(())
    }
}

```

</details>

<details><summary><b> miden-objects/src/note </b></summary>

```rust
// crates/miden-objects/src/note/mod.rs
pub struct Note {
    header: NoteHeader,
    details: NoteDetails,
    nullifier: Nullifier,
}

impl Note {
    pub fn new(assets: NoteAssets, metadata: NoteMetadata, recipient: NoteRecipient) -> Self {
        let details = NoteDetails::new(assets, recipient);
        let header = NoteHeader::new(details.id(), metadata);
        let nullifier = details.nullifier();

        Self { header, details, nullifier }
    }

    pub fn commitment(&self) -> Digest {
        self.header.commitment()
    }
}

// crates/miden-objects/src/note/metadata.rs
pub struct NoteMetadata {
    sender: AccountId,
    tag: NoteTag,
    created_at: BlockNumber,
    note_type: NoteType,
}

// crates/miden-objects/src/note/recipient.rs
pub struct NoteRecipient {
    serial_num: Word,
    script: NoteScript,
    inputs: NoteInputs,
}

// crates/miden-objects/src/note/script.rs
pub struct NoteScript {
    mast_root: Digest,
}

```

</details>

<details><summary><b> miden-objects/src/transaction </b></summary>

```rust
// crates/miden-objects/src/transaction/executed_tx.rs
pub struct ExecutedTransaction {
    id: OnceCell<TransactionId>,
    tx_inputs: TransactionInputs,
    tx_outputs: TransactionOutputs,
    account_delta: AccountDelta,
    tx_args: TransactionArgs,
    advice_witness: AdviceInputs,
    tx_measurements: TransactionMeasurements,
}

impl ExecutedTransaction {
    pub fn new(
        tx_inputs: TransactionInputs,
        tx_outputs: TransactionOutputs,
        account_delta: AccountDelta,
        tx_args: TransactionArgs,
        advice_witness: AdviceInputs,
        tx_measurements: TransactionMeasurements,
    ) -> Self {
        assert_eq!(tx_inputs.account().id(), tx_outputs.account.id());

        Self {
            id: OnceCell::new(),
            tx_inputs,
            tx_outputs,
            account_delta,
            tx_args,
            advice_witness,
            tx_measurements,
        }
    }
}

// crates/miden-objects/src/transaction/proven_tx.rs
pub struct ProvenTransaction {
    tx_hash: TransactionId,
    account_update: TxAccountUpdate,
    output_notes: OutputNotes,
    input_notes: Vec<InputNoteCommitment>,
    expiration_block_num: BlockNumber,
    proof: Proof,
}

// crates/miden-objects/src/transaction/inputs.rs
pub struct TransactionInputs {
    account: Account,
    seed: Word,
    block_header: BlockHeader,
    mmr: Mmr,
    input_notes: InputNotes<InputNote>,
}

```

</details>

<details><summary><b> miden-objects/src/block </b></summary>

```rust
// crates/miden-objects/src/block/header.rs
pub struct BlockHeader {
    block_num: BlockNumber,
    prev_hash: Digest,
    account_root: Digest,
    nullifier_root: Digest,
    note_root: Digest,
    batch_root: Digest,
    proof_hash: Digest,
}

// crates/miden-objects/src/block/proposed_block.rs
pub struct ProposedBlock {
    header: BlockHeader,
    account_updates: Vec<AccountUpdate>,
    note_outputs: Vec<OutputNote>,
    nullifiers: Vec<Nullifier>,
    batch_root: Option<Digest>,
}

// crates/miden-objects/src/block/proven_block.rs
pub struct ProvenBlock {
    header: BlockHeader,
    proof: Proof,
}

// crates/miden-objects/src/block/account_tree.rs
pub struct AccountTree {
    root: Digest,
    accounts: BTreeMap<AccountId, Digest>,
}

// crates/miden-objects/src/block/nullifier_tree.rs
pub struct NullifierTree {
    root: Digest,
    nullifiers: BTreeSet<Nullifier>,
}

```

</details>

<details><summary><b> miden-objects/src/executor </b></summary>

```rust
// crates/miden-tx/src/executor/mod.rs
pub struct TransactionExecutor {
    data_store: Arc<dyn DataStore>,
    authenticator: Option<Arc<dyn TransactionAuthenticator>>,
    exec_options: ExecutionOptions,
}

impl TransactionExecutor {
    pub fn new(
        data_store: Arc<dyn DataStore>,
        authenticator: Option<Arc<dyn TransactionAuthenticator>>,
    ) -> Self {
        Self {
            data_store,
            authenticator,
            exec_options: ExecutionOptions::new(
                Some(MAX_TX_EXECUTION_CYCLES),
                MIN_TX_EXECUTION_CYCLES,
                false,
                false,
            ).expect("Must not fail while max cycles is more than min trace length"),
        }
    }

    #[maybe_async]
    pub fn execute_transaction(
        &self,
        account_id: AccountId,
        block_ref: BlockNumber,
        notes: InputNotes<InputNote>,
        tx_args: TransactionArgs,
    ) -> Result<ExecutedTransaction, TransactionExecutorError> {
        let mut ref_blocks = validate_input_notes(&notes, block_ref)?;
        ref_blocks.insert(block_ref);

        let (account, seed, ref_block, mmr) =
            maybe_await!(self.data_store.get_transaction_inputs(account_id, ref_blocks))
                .map_err(TransactionExecutorError::FetchTransactionInputsFailed)?;

        validate_account_inputs(&tx_args, &ref_block)?;

        let tx_inputs = TransactionInputs::new(account, seed, ref_block, mmr, notes)
            .map_err(TransactionExecutorError::InvalidTransactionInputs)?;

        let (stack_inputs, advice_inputs) =
            TransactionKernel::prepare_inputs(&tx_inputs, &tx_args, None)
                .map_err(TransactionExecutorError::InvalidTransactionInputs)?;

        let advice_recorder: RecAdviceProvider = advice_inputs.into();
        let mut host = TransactionHost::new(
            tx_inputs.account().into(),
            advice_recorder,
            self.data_store.clone(),
            self.authenticator.clone(),
            tx_args.foreign_account_code_commitments(),
        ).map_err(TransactionExecutorError::TransactionHostCreationFailed)?;

        let result = vm_processor::execute(
            &TransactionKernel::main(),
            stack_inputs,
            &mut host,
            self.exec_options,
        ).map_err(TransactionExecutorError::TransactionProgramExecutionFailed)?;

        build_executed_transaction(tx_args, tx_inputs, result.stack_outputs().clone(), host)
    }
}

```

</details>

<details><summary><b> miden-objects/src/prover </b></summary>

```rust
// crates/miden-tx/src/prover/local.rs
pub struct LocalTransactionProver {
    options: ProvingOptions,
}

impl LocalTransactionProver {
    pub fn new(options: ProvingOptions) -> Self {
        Self { options }
    }

    pub fn prove(
        &self,
        executed_tx: ExecutedTransaction,
    ) -> Result<ProvenTransaction, TransactionProvingError> {
        let (account_delta, tx_outputs, tx_witness, _) = executed_tx.into_parts();

        let (pub_inputs, priv_inputs) = build_proof_inputs(
            &tx_witness.tx_inputs,
            &tx_outputs,
            &account_delta,
            &tx_witness.tx_args,
            &tx_witness.advice_witness,
        )?;

        let proof = generate_proof(
            &TransactionKernel::main(),
            pub_inputs,
            priv_inputs,
            &self.options,
        )?;

        let proven_tx = ProvenTransaction::new(
            tx_witness.tx_inputs.account().id(),
            account_delta,
            tx_outputs,
            proof,
        )?;

        Ok(proven_tx)
    }
}

```

</details>

<details><summary><b> miden-lib/asm </b></summary>

```assembly
// crates/miden-lib/asm/note_scripts/P2ID.masm
# P2ID (Pay-to-ID) Script Example
use.miden::account
use.miden::note
use.miden::contracts::wallets::basic->wallet

const.ERR_P2ID_WRONG_NUMBER_OF_INPUTS=0x0002c000
const.ERR_P2ID_TARGET_ACCT_MISMATCH=0x0002c001

proc.add_note_assets_to_account
    push.0 exec.note::get_assets
    mul.4 dup.1 add
    padw movup.5
    dup dup.6 neq
    while.true
        dup movdn.5
        mem_loadw
        padw swapw padw padw swapdw
        call.wallet::receive_asset
        dropw dropw dropw
        movup.4 add.4 dup dup.6 neq
    end
    drop dropw drop
end

begin
    push.0 exec.note::get_inputs
    eq.2 assert.err=ERR_P2ID_WRONG_NUMBER_OF_INPUTS
    padw movup.4 mem_loadw drop drop
    exec.account::get_id
    exec.account::is_id_equal assert.err=ERR_P2ID_TARGET_ACCT_MISMATCH
    exec.add_note_assets_to_account
end

```

</details>

<details><summary><b> miden-tx/src/auth </b></summary>

```rust
// crates/miden-tx/src/auth/mod.rs
pub trait TransactionAuthenticator: Send + Sync {
    fn authenticate(
        &self,
        account_id: AccountId,
        message: Word,
        signature: &[u8],
    ) -> Result<(), AuthenticationError>;
}

// crates/miden-tx/src/auth/rpo_falcon512.rs
pub struct RpoFalcon512Authenticator {
    pub_key_provider: Arc<dyn PublicKeyProvider>,
}

impl TransactionAuthenticator for RpoFalcon512Authenticator {
    fn authenticate(
        &self,
        account_id: AccountId,
        message: Word,
        signature: &[u8],
    ) -> Result<(), AuthenticationError> {
        let pub_key = self.pub_key_provider.get_public_key(account_id)?;

        let signature = Signature::from_bytes(signature)
            .map_err(|_| AuthenticationError::InvalidSignature)?;

        pub_key.verify(message.as_bytes(), &signature)
            .map_err(|_| AuthenticationError::SignatureVerificationFailed)
    }
}

```

</details>

<details><summary><b> miden-objects/src/utils/mmr.rs </b></summary>

```rust
// crates/miden-objects/src/utils/mmr.rs
pub struct Mmr {
    peaks: Vec<Digest>,
    num_leaves: u64,
}

impl Mmr {
    pub fn new(peaks: Vec<Digest>, num_leaves: u64) -> Self {
        Self { peaks, num_leaves }
    }

    pub fn root(&self) -> Digest {
        if self.peaks.is_empty() {
            return EMPTY_WORD;
        }

        let mut result = self.peaks[0];
        for peak in &self.peaks[1..] {
            result = Hasher::merge(&result, peak);
        }

        result
    }

    pub fn verify_proof(
        &self,
        leaf: Digest,
        leaf_index: u64,
        proof: &[Digest],
    ) -> Result<bool, MmrError> {
        if leaf_index >= self.num_leaves {
            return Err(MmrError::LeafIndexOutOfBounds);
        }

        let mut current = leaf;
        let mut pos = leaf_index;

        for &sibling in proof {
            if pos % 2 == 0 {
                current = Hasher::merge(&current, &sibling);
            } else {
                current = Hasher::merge(&sibling, &current);
            }
            pos /= 2;
        }

        Ok(self.peaks.contains(&current))
    }
}

```

</details>

</details>

[Miden](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/miden-base-next)
