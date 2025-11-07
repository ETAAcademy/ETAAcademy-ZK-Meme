# ETAAcademy-ZKMeme: 73. ZK & Transformer

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>73. ZK & Transformer</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZK_Transformer</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Privacy-Preserving and IP-Protective Transformer Inference via FHE, MPC, and ZKP

Transformer models deliver state-of-the-art performance across AI tasks, yet deploying them as cloud services introduces significant privacy and intellectual property risks. To address these, practitioners combine homomorphic encryption (HE), secure multi-party computation (MPC), zero-knowledge proofs (ZKP), and watermarking so that inputs and parameters remain encrypted throughout inference, protecting both user data and model IP.

On the MPC side, malicious-secure three-party computation (3PC) with VDPF and redundant computation achieves constant-round comparisons; together with field/domain reduction and dynamic modulus switching, this balances efficiency and accuracy for Softmax, GELU, and LayerNorm. On the FHE side, the multiplicative-depth bottleneck is tackled via polynomial/power approximations, SIMD-style packing, and rotation-avoiding engineering strategies, improving matrix multiplications and nonlinear layers.

For ZKP, linear relations are handled via circuitization and GKR, while nonlinear relations use hints with range proofs or Lasso lookup; CRPC’s polynomial position encoding compresses multiplication constraints and PSQ reduces variable counts, substantially cutting proving costs.

In parallel, the core Transformer components and the evolution of attention (multi-head, low-rank/sparse/approximate, high-dimensional/cross-modal, memory-augmented) form a unified roadmap that is privacy-preserving, verifiable, and scalable.

## 1. Secure Transformer Inference with MPC and HE: Constant-Round Comparison, Efficient Nonlinear Layers, and Dynamic Modulus Conversion

Transformer-based models achieve state-of-the-art performance across a wide range of AI tasks, but deploying them as cloud inference services raises serious privacy and intellectual-property concerns. To protect user inputs as well as model parameters, systems often rely on homomorphic encryption, secure multi-party computation (MPC), zero-knowledge proofs, or watermarking techniques. These tools enable inference without revealing sensitive user data or leaking model weights.

### 1.1 Secure Transformer Inference with MPC

Secure two-party computation (2PC) protocols—such as **Iron, MPCformer, SIGMA, BOLT, Nimbus, BumbleBee, and SHAFT**—support private inference for large BERT/GPT-style models with hundreds of millions or even billions of parameters.
However, 2PC-based solutions suffer from High communication cost, Frequent back-and-forth interaction, Poor scalability with long sequences or very large models.

To overcome these limitations, several systems adopt a **(2+1)-party** model, where a **trusted third party (TTP)** provides preprocessing material offline.

Fully maliciously secure **3PC frameworks**, such as **PUMA, Ditto, Privformer, Mosformer**, and systems like **ABY3, SWIFT, Pika, CryptFlow, Falcon**, have attracted increasing attention.
Yet, even in 3PC, one major bottleneck remains:

> **The comparison operation (e.g., checking x > y) is expensive under malicious security.**

Existing malicious-secure comparison protocols typically reduce comparison to **MSB (most significant bit) extraction**, requiring

- **O(ℓ)** or **O(log ℓ)** rounds of communication (ℓ = bit length),
  which dominates end-to-end latency—particularly harmful in transformer inference where comparisons appear in GELU, Softmax, LayerNorm, Division and reciprocal-square-root operations

These nonlinear functions are fundamental to Transformer architectures, but are notoriously incompatible with efficient MPC.

#### **Constant-Round Malicious-Secure Comparison using VDPF + Redundant Evaluation**

To address these challenges, the protocol replaces MSB-based comparison with a combination of:

- **VDPF (Verifiable Distributed Point Function)**
- **Random masking**
  **Redundant evaluation for malicious security**

This achieves **constant-round comparison** even under fully malicious adversaries.

#### **Key Ideas**

- **Convert signed comparison to unsigned comparison**

  A random mask _r_ is used to hide the input:

  $$
  \hat{x} = (x + r) \bmod 2^\ell
  $$

  The original MSB can be reconstructed via masked components:

  $$
  \text{MSB}(x) = \text{MSB}(\hat{x}) \oplus \text{MSB}(2^\ell - r) \oplus 1_{\hat{y}_0 > 2^{\ell-1} - \hat{y}_1 - 1}
  $$

  The final term requires only an **unsigned comparison**, which the protocol can securely compute.

- **Reduce unsigned comparison to equality tests**

  The unsigned comparison is decomposed into several **equality-matching subproblems**, each solvable using VDPFs.

  VDPFs provide Very low communication, Single-round evaluation, Efficient malicious-secure verification during preprocessing

  Thus, the comparison can be completed with **one round of communication**.

- **Malicious security via redundant execution**

  VDPF alone ensures key correctness in the offline phase, but not correctness during online evaluation.
  To close this gap, the parties **independently evaluate multiple redundant executions** and cross-check results.

#### Efficient Secure Evaluation of Transformer Nonlinearities

Transformer nonlinear layers such as **Softmax, GELU, LayerNorm** involve functions like:

- Exponential $e^x$
- Division $1/x$
- Square root or reciprocal square root $1/\sqrt{x}$

Traditional MPC implementations rely on:

- **High-degree polynomial approximations** → expensive communication
- **Large lookup tables (LUTs)** → infeasible for 32-bit domains (e.g., $2^{32}$) entries)

#### Domain Reduction for Softmax and Inverse Functions

Softmax inputs are always positive, enabling **domain reduction**:

Choose radix (b), find integer (k) such that:

$$
b^k \le x < b^{k+1}
$$

Normalize:

$$
x' = x / b^{k+1} \in [1/b,, 1)
$$

This drastically shrinks the LUT size.
For example,

- Original 32-bit inverse table = $2^{32}$ entries
- After domain reduction (b = 2, 12-bit precision), only:

$$
(1 - 1/2) \cdot 2^{12} = 2048
$$

entries are required.

Unlike prior methods, this approach **preserves full model accuracy** and does **not** require retraining or fine-tuning.

#### Dynamic Modulus Conversion across Transformer Layers

Most MPC frameworks use the 64-bit ring $\mathbb{Z}_{2^{64}}$ to avoid overflow and leakage.
However 64-bit arithmetic is expensive. It increases communication cost, and nonlinear layers don't require the full 64-bit domain.

**Solution: Layer-wise modulus adaptation**

The protocol uses **dynamic modulus conversion**, switching between:

- **64-bit modulus** for linear layers
- **16- or 32-bit modulus** for nonlinear layers (Softmax, GELU, LayerNorm)

<details><summary>Code</summary>

```Algorithm
Algorithm 1 Unsigned Integer Comparison (ΠUCMP)
Input: 𝑃𝑏 and 𝑃𝑏+1 input (2/2)-shared bitwise representation [[ 𝑥]] of
a private value 𝑥 ∈ U2𝑛 , where [[ 𝑥 [𝑖]]] ∈ Z2𝑛 , for 𝑖 ∈ [0, 𝑛 − 1],
along with a public integer 𝑦 ∈ U2𝑛 .
Output: 𝑃𝑏 and 𝑃𝑏+1 output (2/2)-shared [[𝑟𝑒𝑠]], where 𝑟𝑒𝑠 = 1{𝑥 <𝑦}.
[Setup] Upon initialization, the party 𝑃𝑏+2 does:
1: for i = {0, 1, 2, ⋯, n−1} do
2:     Randomly sample ã[i] ←$− Z₂ⁿ and ⟦ã[i]⟧₀ ←$− Z₂ⁿ
3:     ⟦ã[i]⟧₁ = ã[i] − ⟦ã[i]⟧₀
4:     (k₀•[i], k₁•[i]) ← VDPF.Gen(1^λ, ã[i], 1)
5: end for
6: P_{b+2} sends (⟦ã⟧₀, k₀•) to P_b and (⟦ã⟧₁, k₁•) to P_{b+1}.
7: P_b and P_{b+1} jointly run the DPF key verification protocol in [11]
       to check the well-formedness of the VDPF keys. If the check fails, abort.
8: for i = {0, 1, 2, ⋯, n−1} do
9:     P_b and P_{b+1} expand its DPF keys on domain Z₂ⁿ² to produce
           (2/2)-shared vectors ⟦V[i]⟧ by VDPF.BVEval,
           where V is a two-dimensional array and ⟦V[i]⟧ is produced by
           k₀•[i] and k₁•[i], which represents the share of a one-hot vector
           at point ã[i].
10:    P_b and P_{b+1} jointly compute:
11:        ⟦t⟧ = Σ_{j=0}^{2ⁿ} ⟦V[i][j]⟧
12:        ⟦s⟧ = ⟦ã[i]⟧ − Σ_{j=0}^{2ⁿ} (j ⋅ ⟦V[i][j]⟧)
13:    P_b and P_{b+1} open ⟦t⟧, ⟦s⟧ then check if t = 1 and s = 0.
           If the check fails, abort.
14: end for
15: for i = {n−1, n−2, ⋯, 0} do
16:     ⟦u[i]⟧ = ⟦x[i]⟧ − y[i]
17:     ⟦w[i]⟧ = ⟦x[i]⟧ ⊕ y[i]
18:     ⟦c[i]⟧ = ⟦u[i]⟧ + 1 + Σ_{k=i+1}^{n} ⟦w[k]⟧
19: end for
20: for i = {0, 1, 2, ⋯, n−1} do
21:     ⟦δ[i]⟧ = ⟦c[i]⟧ + ⟦ã[i]⟧
22: end for
23: P_b and P_{b+1} open δ⃗ over ring Z₂ⁿ.
24: return ⟦res⟧ = Σ_{i=0}^{n−1} ⟦V[i][δ[i]]⟧.

```

</details>

---

### 1.2 Private LLM Inference with Homomorphic Encryption

Privacy-preserving inference for large language models (LLMs) using cryptography can be broadly categorized into two families:

- **Interactive protocols based on multi-party computation (MPC)**
- **Non-interactive protocols based on fully homomorphic encryption (FHE)**

FHE-based methods face a fundamental challenge: **multiplicative depth**.
Higher multiplicative depth requires:

- larger cryptographic parameters,
- greater noise budget,
- more frequent and expensive bootstrapping operations.

Since matrix multiplication is a core ingredient of many workloads—including data analytics and machine learning—this depth constraint makes Transformer inference particularly challenging, as Transformers rely heavily on large linear projections and attention matrices.

#### FHE-Friendly Transformer Adaptations

One research direction modifies the Transformer architecture itself to better suit FHE.
Examples include replacing non-polynomial functions (e.g., Softmax) with **polynomial approximations**.

- Replace non-polynomial operations with polynomials.
- **Power-Softmax** substitutes Softmax’s exponential with power functions and uses conventional fine-tuning.
- Other works approximate specific Transformer components—Softmax, LayerNorm, or both—to reduce multiplicative depth.

The state-of-the-art in this direction is **Powerformer (ACL ’25)**, which replaces Softmax and LayerNorm in a BERT-base model using power functions and linear functions.

However, the major limitation is that these approaches **require retraining or knowledge distillation**, which restricts generalization to other Transformer architectures.

#### Pure FHE Solutions: NEXUS, THOR, and MoAI

A second line of research focuses on **fully homomorphic, non-interactive** protocols:

- **NEXUS (NDSS ’25)** proposes FHE-friendly matrix multiplication strategies and activation approximations.
  However, NEXUS does not provide a full end-to-end Transformer inference pipeline.

- **THOR (CCS ’25)** implements the first end-to-end secure Transformer inference system entirely under FHE.
  It supports ciphertext–ciphertext matrix multiplication via **format conversion** and, unlike approximation-based systems, **does not modify any Transformer components** (e.g., Softmax or LayerNorm).
  Thus, no fine-tuning or retraining is required.

#### Efficient Matrix Computation in FHE

FHE does not evaluate each element of a matrix independently; this would be prohibitively slow.
Instead, modern FHE uses **SIMD packing**, placing multiple values into a single ciphertext, each in different **slots**. A single homomorphic operation acts on all slots simultaneously.

Different packing strategies correspond to different data layouts:

- **Column Packing**: Each ciphertext encodes a column vector, ideal for **matrix–vector multiplication** in linear layers, and reduces rotation requirements.

- **Diagonal Packing**: Elements are arranged along diagonals within the ciphertext, which enables efficient **matrix–matrix multiplication** or operations requiring position-wise alignment, such as Softmax.

Packing is central to performance: the correct layout minimizes the number of expensive operations, especially rotations.

#### **Softmax and LayerNorm Without Rotations**

In FHE, **rotation** means cyclically shifting ciphertext slots:

$$
[a_1, a_2, \dots, a_n] \rightarrow [a_2, a_3, \dots, a_1]
$$

Rotations are necessary for summation (e.g., computing the denominator of Softmax), but they are extremely expensive because each rotation requires a **Galois key** operation.

A major engineering goal is therefore **rotation-free Softmax and LayerNorm**.

#### Ciphertext–Ciphertext Matrix Multiplication in FHE

Transformer attention involves ciphertext–ciphertext (CT–CT) multiplications, such as:

- $QK^T$
- $\text{Softmax}(QK^T)V$

CT–CT multiplication typically requires rotations to aggregate intermediate results.
To reduce this overhead, THOR and similar systems:

- Use **interleaved data layouts**: Distribute matrix blocks across slots in a way that reduces the number of required rotations.

- Exploit SIMD parallelism: Perform multiple partial products in parallel within a single ciphertext.

- Apply plaintext–ciphertext multiplications where possible

Linear layers often only require plaintext weights, which avoids rotations entirely.

This results in significantly faster homomorphic implementation of Transformer computations.

---

## 2. Verifiable LLM Inference with Zero-Knowledge Proofs: Challenges and Recent Advances

Today’s large language models (LLMs) such as GPT, Claude, and Gemini all follow a cloud-based paradigm: _AI companies deploy the model on their own servers, and users access it via APIs_.
This architecture has a fundamental drawback—**the cost of inference is extremely high** (GPUs, electricity, maintenance). Because of this, a provider could _cheat_: instead of running an expensive model, it might silently substitute a cheaper model.

**Zero-knowledge proofs (ZKPs)** provide a natural solution.
They enable a service provider (**Prover**) to convince a user or regulator (**Verifier**) that $y = f(x, w)$ was computed correctly, where:

- (f) is the model’s computation (e.g., transformer inference),
- (x) is the user’s input (public),
- (w) is the model’s weights (private),
- (y) is the output,
- ($p_i$) is a zero-knowledge proof ensuring correctness **without revealing** the weights.

However, building ZK proofs for LLMs is extremely challenging. Transformer models contain many architectural components:

- **Linear layers** (matrix multiplications): ~90% of compute
- **Nonlinear layers** (GELU, Softmax, LayerNorm): difficult to represent in ZK
- **Large depth and huge parameter counts** (hundreds of layers, billions of parameters)

#### (1) Efficient Proofs for Linear Layers

All ZKP systems ultimately operate on **arithmetic relations** composed of additions and multiplications. These relations are represented as **arithmetic circuits**, and for efficient verification we often use the **GKR protocol** or **matrix multiplication verification protocols** such as Thaler’s classic scheme.

Thaler’s protocol provides the best-known practical method for verifying linear layers, but it requires building **multilinear extension tables** (“bookkeeping tables”) for every matrix column. This results in approximately (4nm) field multiplications—expensive at LLM scale.

Two improvements are commonly applied:

- Grouping Algorithm: A technique that reduces about **half the multiplications** by grouping terms into structured batches.

- Exploiting Matrix Structure: Linear layers in Transformers often include **padding elements** (known zeros) and **quantized values** with small ranges.

These allow the prover to **replace many multiplications with additions**, which are far cheaper. At scale, this yields roughly a **10× speedup**.

#### (2) Nonlinear Layers: Division, Sqrt, Exponential, and More

Nonlinear functions such as division, square root, exponentiation, GELU, Softmax, and LayerNorm are problematic because they are **not polynomial**, so they do not map cleanly into arithmetic circuits.

Early systems used floating-point circuits or iterative approximation, which led to enormous proving costs.

Two recent ideas dramatically reduce overhead:

- Simulation of Nonlinear Operations: Instead of re-computing the operation inside the circuit, the prover supplies the true output as an **advice value**, and the circuit checks its validity.

- Range Relations & Lookup Arguments: A small constraint system checks whether the advice value lies within the mathematically correct range.

Examples:

- **Division / sqrt**: verify the result lies in the correct interval.
- **Exponential / Softmax**: use _lookup tables_ via the highly efficient **Lasso** protocol for table-based verification.

These methods anchor correctness while avoiding costly intermediate arithmetic.

### QAP Complexity and Why Matrix Multiplication Is Expensive

In zk-SNARKs, an arithmetic circuit is expressed as a **Quadratic Arithmetic Program (QAP)**.
The efficiency of a QAP depends on:

- **Number of constraints**, proportional to the number of multiplication gates.
- **Number of variables**, equal to the size of the full assignment.

For a matrix multiplication (Y = XW), every term $x_{ik} w_{kj}$ requires its own multiplication constraint.
Thus, a naïve zk-SNARK implementation costs: $O(a \times b \times n)$

constraints for an $a \times n$ matrix times an $n \times b$ matrix—far too expensive for transformer workloads.

Earlier work reduced constraints by adding _dummy variables_ (virtual terms), but applying this directly to matrix multiplication explodes the number of variables, negating the benefit.

#### CRPC: Polynomial Encoding to Compress $O((n^3))$ Constraints to O((n))

The **Compressed Row-Permutation Check (CRPC)** approach avoids element-wise multiplication constraints entirely by converting matrices into **polynomial encodings**.

Key idea:
For fixed (k), each product $x_{ik} w_{kj}$ belongs to the outer product of X’s (k)-th column and W’s (k)-th row:

$$
y_{ij} = \sum_{k=0}^{n-1} x_{ik}w_{kj}.
$$

Instead of proving these individually, CRPC encodes the relevant matrix slices as polynomials.

Example:

- Encode column 0 of (X):

$$
X_0(Z) = Z^0 x_{00} + Z^2 x_{10} + Z^4 x_{20}.
$$

- Encode row 0 of (W):

$$
W_0(Z) = Z^0 w_{00} + Z^1 w_{01}.
$$

Their product naturally contains all terms $x_{i0} w_{0j}$, with **different powers of (Z)** labeling their positions.

The full CRPC identity is:

$$
\sum_{j=0}^{b-1} \sum_{i=0}^{a-1} Z^{ib + j} y_{ij}
\sum_{k=0}^{n-1}
\left(
\sum_{i=0}^{a-1} Z^{ib} x_{ik}
\right)
\cdot
\left(
\sum_{j=0}^{b-1} Z^j w_{kj}
\right).
$$

- **Left-hand side:** encodes all output elements $y_{ij}$
- **Right-hand side:** sums all column–row polynomial products
- **Exponents:** act as location tags preventing term collisions

**Result:**
CRPC transforms an $O(n^3)$ constraint system into **O(n)** polynomial multiplications.

The **Prefix-Sum Query (PSQ)** technique further optimizes the _accumulation_ logic in matrix multiplication, reducing the number of intermediate variables in the QAP. This lowers prover time and memory usage significantly.

---

## 3. Transformer-Based Large Language Models: Architecture, Components, and Variants

Modern large language models (LLMs) are built from a stack of (L) Transformer blocks.
Each Transformer block contains several key components, including:

- matrix-multiplication layers,
- LayerNorm layers,
- multi-head self-attention layers,
- and a GeLU-based feed-forward network (FFN).

The input to an LLM is an $(s \times d)$ embedding matrix (x), where (s) is the sequence length and (d) is the hidden dimension. After passing through the first Transformer block, the output is again an $(s \times d)$ matrix (f), which is then fed to the next block. This process repeats across the full depth of the model.

### Transformer Architecture Overview

The Transformer is a neural architecture built entirely around the **attention mechanism**, designed for processing sequential data such as natural language, code, or time-series signals. Unlike RNNs or LSTMs—which process tokens sequentially and suffer from vanishing gradients—Transformers dispense with recurrence altogether. They operate in parallel and rely entirely on attention and position encodings.

A standard Transformer model consists of two primary components:

- **Encoder stack**: Reads the input sequence and produces context-aware representations.
- **Decoder stack**: Generates output tokens (e.g., translations or predictions), using both prior outputs and encoder information.

Because Transformers have no inherent notion of sequence order, **positional encoding**, typically defined with sinusoidal functions, is added to the input embeddings to provide each token with its relative position in the sequence. These sinusoidal encodings also allow the model to generalize to longer sequences than those seen during training.

Each **encoder block** contains:

- **Multi-head self-attention layer**
- **Feed-forward neural network (FFN)**
- **Two LayerNorm operations**
- **Residual connections** inserted between major sublayers to stabilize gradients and improve training efficiency.

After attention, each token’s representation is independently processed through a feed-forward network, enabling the model to learn nonlinear transformations and enrich the representational capacity. LayerNorm is essential for stabilizing the intermediate activations, mitigating exploding or vanishing gradients, and supporting deeper architectures.

#### Token Embedding and Input Representation

Text inputs must be mapped to continuous vectors known as **token embeddings**.
Given a vocabulary of size $N_v$ and model dimension $d_{\text{model}}$, the embedding matrix is:

$$
W_e \in \mathbb{R}^{d_{\text{model}} \times N_v}.
$$

For each token (t), the embedding is simply:

$$
\text{embedding}(t) = W_e[:, t],
$$

which corresponds to a lookup operation retrieving the (d)-dimensional token vector.

#### Self-Attention: The Core Mechanism

Self-attention is the central innovation of the Transformer architecture.
It allows each token to attend to all other tokens in the sequence, capturing long-range dependencies without sequential computation.

Given input matrix (X), the attention mechanism forms:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V.
$$

The scaled dot-product attention is:

$$
\text{Attention}(Q, K, V)
\text{Softmax}!\left(
\frac{QK^T}{\sqrt{d_{\text{model}}}}
\right)V.
$$

- $QK^T$ computes pairwise similarity between tokens.
- The denominator $\sqrt{d_{\text{model}}}$ stabilizes gradients.
- Softmax converts similarities into a probability distribution.
- Multiplication with V yields a weighted combination of token features.

This mechanism allows the model to **dynamically aggregate information** across the entire sequence, without the bottlenecks of recurrence or convolution.

#### Multi-Head Attention

Instead of using a single attention distribution, multi-head attention enables the model to capture different relational patterns simultaneously:

$$
\text{head}_i
\text{Attention}(QW_Q^{(i)}, KW_K^{(i)}, VW_V^{(i)}).
$$

Outputs from all heads are concatenated and projected:

$$
\text{MultiHead}(Q,K,V)
\text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O.
$$

#### Feed-Forward Network (FFN)

Each Transformer block contains a two-layer feed-forward network:

$$
\text{FFN}(X) = \text{GELU}(XW_1 + b_1) W_2 + b_2.
$$

The **GELU** activation provides a smoother nonlinearity compared to ReLU and significantly improves model expressiveness.

#### Layer Normalization

LayerNorm stabilizes training and accelerates convergence.
Given an input vector $x \in \mathbb{R}^k$:

$$
m = \frac{1}{k} \sum_i x_i,
\qquad
v = \frac{1}{k} \sum_i (x_i - m)^2,
$$

$$
y_i = \gamma \cdot \frac{x_i - m}{\sqrt{v}} + \beta.
$$

The learnable parameters $\gamma$ and $\beta$ allow the model to adjust normalized activations adaptively.

### Variants and Extensions of Attention Mechanisms

Transformers owe their effectiveness to attention’s ability to model **global dependencies** in a single computational step. Earlier architectures such as RNNs and LSTMs struggled with long-term dependencies due to gradient decay over long sequences. Attention sidesteps these issues completely by enabling direct token-to-token interaction regardless of distance.

Research over the past years has produced a large ecosystem of attention mechanisms, each addressing different computational or structural limitations.

#### Classic Self-Attention

Based on Vaswani et al. (2017), encompassing:

- Scaled dot-product attention
- Multi-head attention
- Relative position encoding

#### Efficient Attention (Reducing $O(n^2)$ Complexity)

To handle very long sequences, many approximate or compressed attention mechanisms were introduced:

- **Linear Attention**
- **Linformer** (low-rank compression of (K) and (V))
- **Sparse Transformer / Longformer** (local + selective global attention)
- **BigBird** (local + random + global hybrid)
- **Performer** (random feature maps approximating softmax)
- **Reformer** (LSH-based attention + reversible layers)

#### Multi-Dimensional and Structured Attention

Designed for images, videos, audio, and other high-dimensional modalities:

- **Axial Attention** (attend along rows and columns separately)
- **Dual Attention** (spatial + channel dependencies)
- **Swin Transformer** (shifted window attention for vision)
- **Conformer** (Transformer + CNN hybrid for speech)

#### Memory-Augmented and Recurrent Attention

To extend context beyond the fixed window:

- **Transformer-XL** (segment-level memory)
- **Reformer** (reversible layers + hashing)
- **Infini-Attention** (infinite context via compressed memory)

#### Domain-Specific Transformer Architectures

Examples include:

- NLP: GPT, BERT, T5
- Vision: ViT, Swin, SegFormer
- Multi-modal: AdaViT, CLIP-style cross-attention models
- Speech: Conformer

#### Cross-Modal and Graph Attention

Used for multimodal and structured data:

- **MulT** (video + text)
- **IMRAM** (image–text retrieval)
- **Set Transformer** (unordered set input)
- **Graph Attention Networks (GAT)** and extensions such as GAAN and hypergraph attention

---

[TF-Encrypted](https://github.com/tf-encrypted/tf-encrypted)
[MP-SPDZ](https://github.com/data61/MP-SPDZ)
[Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
