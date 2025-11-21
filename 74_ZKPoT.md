# ETAAcademy-ZKMeme: 74. ZKPoT

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>74. ZKPoT</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZKPoT</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Zero-Knowledge Proofs of Training, Verifiable Optimization, Unlearning and Randomness

Zero-Knowledge Proofs of Training (zkPoT) certify that a model was trained under a specified algorithm without revealing either the model or the data. For RNNs, zkPoT represents all linear operations as matrix multiplications verified via sumcheck, handles nonlinear components such as ReLU and softmax using lookup tables (TaSSLE), and employs Incremental Verifiable Computation (IVC) to fold transcripts and recursively aggregate commitments and evaluations, ensuring that verification time and proof size grow only logarithmically with the number of iterations.

In blockchain settings, zkPoT-based consensus together with IPFS’s content-addressed DHT storage enables decentralized, efficient, and auditable Federated Learning aggregation, while avoiding the high cost of PoW and the centralization risks of PoS.

For verifiable machine unlearning, bit masking enforces feature-, sample-, and class-level forgetting with invariant dataset commitments and circuits. A state-preserving AND operator guarantees irrecoverability, Verifiable Random Functions (VRFs) fix minibatch order to shrink the forgery search space, and per-update ZK proofs require that the gradient of each remaining sample differs from that of the forgotten sample by a minimum threshold to prevent gradient replication attacks.

To eliminate security weaknesses arising from training randomness, zkPoT proves the final solution rather than the training trajectory. Strong convexity yield computable convexity bounds (e.g., Quadratic Growth, Polyak-Łojasiewicz (PL) inequality, and strong monotonicity), implemented using fixed-point interval arithmetic and tight piecewise-linear upper and lower bounds for real-valued and nonlinear operations. The total approximation error remains small while preserving soundness, supported by classical optimization theory that guarantees both convergence and verifiability.

---

## 1. Zero-Knowledge Training Proofs (zkPoT) for Recurrent Neural Networks

Zero-knowledge proofs of training (zkPoT) allow a prover to demonstrate that a machine-learning model was trained _exactly_ according to a specified training algorithm—without revealing the model weights, the training data, or any sensitive intermediate information. zkPoT has gained increasing attention because it enables proofs of properties such as:

- that specific data **was not** used in training,
- that the model was trained under regulatory constraints such as **fairness** [SWF22] or **differential privacy** [STC24],
- while keeping both the model parameters and the dataset **fully private**.

Existing zkPoT protocols (e.g., [APKP24, GGJ23]) typically require the prover to commit to the training data and the initial model, and then prove that _there exists_ an input (training data + randomness) such that executing the reference training algorithm (e.g., SGD) yields the committed final model.

However, proving the training of **Recurrent Neural Networks (RNNs)** introduces additional challenges. RNNs contain:

- **hidden-state recurrence** across time steps,
- **weight sharing** across all unrolled layers,
- and nontrivial nonlinear activations.

These must all be faithfully enforced within the zero-knowledge circuit, requiring the proof to verify the entire forward pass, backward pass, and gradient updates across time.

To tackle these difficulties, modern zkPoT frameworks certify all RNN forward activations, backward gradients, and parameter updates, and produce a _succinct_ proof covering the full training trajectory using **incrementally verifiable computation (IVC)**.

### RNN Computation and Training

An RNN processes sequential data (text, audio, time series, etc.) by maintaining a hidden state $h^{(t)}$ that summarizes past information. At each time step:

$$
s^{(t)} = F(s^{(t-1)}, x^{(t)}), \qquad
y^{(t)} = G(s^{(t)}).
$$

Assume the input sequence is embedded into vectors $x^{(t)} \in \mathbb{R}^d$ and stacked:

$$
X = \begin{bmatrix}
x^{(1)} \\
x^{(2)} \\
\vdots \\
x^{(T)}
\end{bmatrix} \in \mathbb{R}^{T \times d}.
$$

#### 1) Forward Pass

Hidden-state update $h^{(t)} = \mathrm{Act}(W_h h^{(t-1)} + W_e x^{(t)} + b_1)$,

where:

- $W_h$: hidden-to-hidden matrix,
- $W_e$: input-to-hidden matrix,
- $b_1$: bias,
- $\mathrm{Act}$: activation (ReLU, sigmoid, tanh).

Output prediction $\hat{y}^{(t)} = \mathrm{softmax}(W_y h^{(t)} + b_2)$.

#### 2) Loss Function

Using cross-entropy

$$
L^{(t)} = -\sum_{k=1}^K y_k^{(t)} \log \hat{y}_k^{(t)}, \qquad
L = \frac{1}{T}\sum_{t=1}^T L^{(t)}.
$$

#### 3) Backpropagation Through Time (BPTT)

Training a recurrent neural network (RNN) is more complex than training a feed-forward network because RNNs contain **recurrence**: the hidden state at each time step depends on the previous one. As a result, gradients must flow not only backward through layers, but also **backward through time**. This process is known as **BackPropagation Through Time (BPTT)**.

Because each hidden state $h^{(t)}$ contributes to the next, the error at time (t) is influenced by both the current prediction and all future time steps. Unrolling an RNN in time gives a long chain:

$$
x^{(1)} \rightarrow h^{(1)} \rightarrow h^{(2)} \rightarrow \cdots \rightarrow h^{(T)},
$$

so the gradient must propagate all the way backward along this chain.

**Output Layer Gradients**

At each time step, we compute the output error

$$
\delta^{(t)} = \hat{y}^{(t)} - y^{(t)}.
$$

The gradient of the loss with respect to the output weights is then

$$
\frac{\partial L}{\partial W_y}
= \delta^{(t)} {h^{(t)}}^\top.
$$

Intuitively, this states that the output layer’s mistake $\delta^{(t)}$ must be attributed to the hidden state $h^{(t)}$ that produced the prediction, so the gradient is their outer product.

**Gradients Flow into the Hidden State**

The output error also flows back into the hidden state:

$$
\frac{\partial L}{\partial h^{(t)}} = W_y^\top \delta^{(t)}.
$$

Thus, the error at the output layer becomes part of the error signal used to update the hidden-state dynamics.

**Backpropagation Into the Pre-Activation**

The hidden state is produced from a pre-activation value

$$
a^{(t)} = W_h h^{(t-1)} + W_e x^{(t)} + b_1.
$$

To pass the gradient backward into this linear combination, we multiply by the derivative of the activation function:

$$
\frac{\partial L}{\partial a^{(t)}}
= \frac{\partial L}{\partial h^{(t)}} \circ \text{Act}'(a^{(t)}),
$$

where $\circ$ denotes elementwise multiplication.

**Gradients for Recurrent and Input Weights**

Once we have $\frac{\partial L}{\partial a^{(t)}}$, we can compute gradients for the two key weight matrices:

$$
\frac{\partial L}{\partial W_h}
= \left( \frac{\partial L}{\partial a^{(t)}} \right)
{h^{(t-1)}}^\top,
$$

$$
\frac{\partial L}{\partial W_e}
= \left( \frac{\partial L}{\partial a^{(t)}} \right)
{x^{(t)}}^\top.
$$

These expressions reflect the sources of the error: part comes from the _past_ hidden state $h^{(t-1)}$, and part from the _current_ input $x^{(t)}$.

**Backpropagation Through Time**

Because $h^{(t)}$ itself depends on $h^{(t-1)}$, the gradient must continue flowing backward:

$$
\frac{\partial L}{\partial h^{(t-1)}}
= W_h^\top \left(\frac{\partial L}{\partial a^{(t)}}\right).
$$

This recursive flow of gradients through time steps is what gives BPTT its name: each time step influences all earlier ones.

### Parameter Update

Finally, all parameters are updated using gradient descent:

$$
\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}
$$

where the parameters are $(W_y, b_2, W_h, W_e, b_1)$.

In practical training, more sophisticated optimizers (Adam, RMSProp, Adagrad) are typically used instead of basic gradient descent. RNN training also commonly applies **gradient clipping**, a crucial technique to prevent exploding gradients during backpropagation through long sequences.

### Zero-Knowledge Proof of RNN Training

In zkPoT, the prover must convince the verifier that **all forward and backward computations were executed correctly**, and that the resulting parameters match the committed model.

The computation decomposes cleanly as:

- **linear operations**
  (matrix multiplication, addition, Hadamard product)
  → cheap in ZK using _matrix-sumcheck_
- **nonlinear operations**
  (ReLU, sigmoid, exp, softmax normalization)
  → handled by _lookup arguments_ or TaSSLE-style tables.

#### Linear Operations → Unified via Matrix Sumcheck

Every linear operation is reduced to matrix multiplication. Example:

$$
\sum_{t=1}^{T} M^{(t)} x^{(t)} =
\big[M^{(1)} M^{(2)} \cdots M^{(T)}\big]
\begin{bmatrix}
x^{(1)}\\
\vdots \\
x^{(T)}
\end{bmatrix}.
$$

Elementwise products become:

$$
x \odot y = \mathrm{diag}(x) * y.
$$

Thus the entire forward–backward process collapses into **a sequence of matrix multiplications**, all certifiable with one unified sumcheck framework.

#### Nonlinear Activations → Lookup Tables

The system precomputes quantized activation tables, e.g.:

$$
T_\text{ReLU} = {(x,y) \mid y = \max(x,0)}.
$$

To prove correctness, the prover only needs to show:

- the pair $(x_i, y_i)$ appears in the committed lookup table
- without proving the logic of the activation itself.

Softmax involves tables for:

- $\exp(x)$
- normalization coefficients
- rounding / quantization.

These are efficiently handled via TaSSLE-style batched lookup arguments.

### Folding All Computation into a Single Transcript

The forward and backward passes generate:

- matrix commitments,
- lookup commitments,
- sumcheck transcripts,
- Fiat–Shamir challenges,
- evaluation claims.

Everything is folded into a single transcript, allowing the verifier to check only **a constant number of polynomial openings**.

### Incrementally Verifiable Computation for Full Training

An RNN unfolds across (T) steps. Each step produces:

- its own PBPTT (proved BPTT) proof,
- evaluation claims $(\sigma_j, r_j, v_j)$,
- lookup and matrix commitments.

Naively verifying all steps is too costly.
Instead, zkPoT employs **IVC** (Incrementally Verifiable Computation):

#### Commitment Aggregation

For all matrices (M_i) produced at iteration (i):

$$
M^* = \sum_i \alpha_i M_i,
$$

where $\alpha_i$ are random scalars.

The verifier checks only a _single_ sampled entry:

$$
M^*[r,c] = \sum_i \alpha_i M_i[r,c].
$$

A single sumcheck certifies the linear relation across all iterations.

#### Evaluation Claim Aggregation

Each iteration emits multiple evaluation claims. zkPoT compresses them via random linear combination into:

$$
(\sigma_{\text{agg}}, v_{\text{agg}}).
$$

Thus each iteration contributes only **one** aggregated evaluation.

#### Recursive Proof Tree

A (k)-ary recursion tree is constructed:

- Leaves: per-iteration PBPTT proofs
- Internal nodes: verifier circuits that check and aggregate child proofs
- Root: a single succinct proof covering the entire training process

The recursion depth:

$$
\mathrm{depth} = \lceil \log_k T \rceil.
$$

Verifier runtime and proof size grow only **logarithmically in the number of time steps**.

<details><summary>Code</summary>

```Algorithm
Algorithm 1: Backpropagation Through Time (BPTT)
Input: Weights Wy,i−1, Wh,i−1, We,i−1; biases b1,i−1, b2,i−1; initial hidden state h_i^(0); inputs {x_i^(t)}_{t=1..T} and labels {y_i^(t)}_{t=1..T}
Output: Updated weights Wy,i, Wh,i, We,i and b1,i, b2,i

INITIALIZE:
1: ∇Wy,i ← 0, ∇Wh,i ← 0, ∇We,i ← 0, ∇b1,i ← 0, ∇b2,i ← 0
2: L ← 0

FORWARD PASS:
3: for t = 1 to T do
4: a_i^(t) ← Wh,i−1 h_i^(t−1) + We,i−1 x_i^(t) + b1,i−1     (a_i^(t) ∈ ℝ^h)
5: h_i^(t) ← Act(a_i^(t))
6: z_i^(t) ← Wy,i−1 h_i^(t) + b2,i−1
7: ŷ_i^(t) ← softmax(z_i^(t))
8: L^(t) ← −∑_k y_i,k^(t) log(ŷ_i,k^(t))
9: L ← L + L^(t)
10: end for

BACKWARD PASS:
11: for t = T down to 1 do
12: δ_i^(t) ← ŷ_i^(t) − y_i^(t)     (δ_i^(t) ∈ ℝ^K)
13: ∇Wy,i ← ∇Wy,i + δ_i^(t) (h_i^(t))^⊤
14: ∇b2,i ← ∇b2,i + δ_i^(t)
15: if t == T then
16: h_i'^(t) ← Wy,i−1^⊤ δ_i^(t)
17: else
18: h_i'^(t) ← Wy,i−1^⊤ δ_i^(t) + Wh,i−1^⊤ a_i'^(t+1)
19: end if
20: a_i'^(t) ← h_i'^(t) ∘ Act'(a_i^(t))     (a_i'^(t) ∈ ℝ^h)
21: ∇Wh,i ← ∇Wh,i + a_i'^(t) (h_i^(t−1))^⊤
22: ∇We,i ← ∇We,i + a_i'^(t) (x_i^(t))^⊤
23: ∇b1,i ← ∇b1,i + a_i'^(t)
24: end for

UPDATE WEIGHTS:
25: Wy,i ← Wy,i−1 − η · (1/T) ∇Wy,i
26: Wh,i ← Wh,i−1 − η · (1/T) ∇Wh,i
27: We,i ← We,i−1 − η · (1/T) ∇We,i
28: b1,i ← b1,i−1 − η · (1/T) ∇b1,i
29: b2,i ← b2,i−1 − η · (1/T) ∇b2,i
```

</details>

---

### Zero-Knowledge Proofs of Training for Decentralized Federated Learning

Zero-Knowledge Proofs of Training (ZKPoT) can be embedded into blockchain frameworks to provide decentralized consensus over model updates. In such systems, each iteration selects a node to aggregate model updates from participants. Federated Learning (FL) enables multiple parties to collaboratively train models while keeping data private and secure; blockchain augments FL with stronger security guarantees, transparent audit trails, and protection against data and model tampering.

**Limitations of Traditional FL Consensus**
Many blockchain-secured FL systems rely on conventional consensus mechanisms. Proof of Work (PoW) is computationally expensive; Proof of Stake (PoS) improves energy efficiency but introduces centralization risks by favoring large stakeholders. Learning-based consensus approaches where miners aim to achieve a target accuracy to earn the right to commit a model are also inefficient. Due to the statistical nature of ZKPs, verifying the highest-accuracy model may require hundreds to thousands of interactions between miners and validators. Random selection of validators among all nodes further risks choosing an adversarial validator who can disrupt protocol progress. Moreover, when a central server generates ZKPs, users can locally verify aggregated results, but sending model updates from local clients to an untrusted central server may still leak information—an issue these frameworks do not fully resolve.

**ZKPoT Consensus with zk-SNARKs**
A ZKPoT-driven consensus uses succinct, non-interactive zero-knowledge proofs (zk-SNARKs) to validate a participant’s contribution based on model performance, removing the inefficiencies of traditional consensus and reducing privacy risks inherent to learning-based voting. By tailoring block and transaction formats for ZKPoT and integrating content-addressed storage, the system streamlines consensus flow and significantly reduces communication and storage overhead.

**IPFS Integration**
InterPlanetary File System (IPFS) is a decentralized, peer-to-peer system designed for efficient file sharing and secure storage. Files are addressed by their cryptographic content hash rather than by location, and these hashes are managed via a distributed hash table (DHT). Any change to file content changes its hash, preventing tampering and eliminating redundancy. Files are replicated across nodes and shared among network users, accelerating communication. In the ZKPoT framework, IPFS stores global aggregation proofs of the training algorithm and anchors them within the blockchain structure, lowering storage costs while ensuring integrity.

---

## 2. Verifiable Machine Unlearning with zkPoT

Recent work shows that **zkPoT** (Zero-Knowledge Proof of Training) can significantly strengthen the emerging field of **verifiable machine unlearning**.
Machine unlearning requires a model to **remove the influence of specific training data**, as if the data had never been used during training.
**Verifiable** unlearning extends this by ensuring that a third party can **cryptographically verify** that the model has indeed forgotten the designated data.

The main motivations for machine unlearning include:

- **Access revocation** — when sensitive or copyrighted data was accidentally included in training.
- **Model correction** — when training data contains outdated knowledge, toxic content, unethical material, or misinformation. Unlearning provides a principled correction mechanism.
- **Transparency and auditability** — unlearning supports regulatory compliance and builds trust by enabling trainers to provably modify or sanitize models.

### Limitations of Early Verifiable Unlearning: Membership-Proof Frameworks

Early research on verifiable unlearning relied on **Prediction Comparison**:
a backdoor is implanted into data that should be forgotten, and the model is checked to see if the backdoor behavior disappears after unlearning.
However, this approach is fundamentally flawed due to randomness in SGD.

Shumailov et al. (NeurIPS 2021) proved that merely changing **the order of minibatches in SGD** can produce a model that appears indistinguishable from one trained on a different dataset.
Thus, an attacker can train on data _including_ the supposedly forgotten samples, yet forge a model that appears to be trained on the cleaned dataset.

### The Hash-List Framework

Most existing verifiable-unlearning designs use **hash-based dataset management**.
Two lists are maintained:

- **D** — training set (hashes only, not plaintext)
- **U** — list of data requested for unlearning

Two commitment roots are maintained:
`root(D)` and `root(U)`.

When a user requests unlearning:

- The item’s hash is removed from **D** and appended to **U**, producing a new dataset $(D' = D \setminus U)$.
- The trainer retrains the model on (D') and produces a new zkPoT proof.

To prove a data point belongs in **U**, the trainer provides a **hash chain** from the data hash to `root(U)`.

### Why the Hash-List Design Fails

Despite appearing systematic, this design suffers from severe limitations:

**No Feature-Level or Class-Level Unlearning**

Membership proofs can only verify **set membership**.
They cannot prove whether:

- a feature has been modified
- a label has been corrected

Thus, systems resort to adding **placeholders** (e.g., zero-padded features), which can be exploited in forging attacks because placeholders themselves are indistinguishable from legitimate data.

**Dynamic Datasets Are Expensive and Insecure**

Every unlearning request changes the dataset:

- Requires recomputing `root(D)` and `root(U)`
- Requires regenerating the entire ZK circuit
- Proofs scale as (O(|U|)) due to linear hash chains
- Easily breaks under frequent dataset updates

Modern ZK systems aim for **O(log n)** (Merkle trees) or **O(1)** (vector commitments), so this is unacceptable.

**Cannot Resist SGD Forgery Attacks**

Because SGD uses random minibatches, an attacker can choose minibatches that:

- do not contain unlearned data
- but yield **nearly identical gradient updates**

Thus producing a forged model nearly identical to the “correct” unlearned model.

**Gradient Replica Problem**

Even worse: if another sample in the dataset has a **similar gradient** to the forgotten sample (a _gradient replica_), then deleting the original data has no real effect.

### Bit-Masking: A ZK-Friendly Approach to Verifiable Unlearning

Bit-masking offers decisive advantages over membership-proof frameworks:

- Supports **feature-level**, **sample-level**, and **class-level** unlearning
- Dataset commitments remain **unchanged**
- Training circuits remain **invariant** (no recompilation)
- Unlearning is enforced directly inside the ZK proof

The core idea is a **private bit-mask matrix** $b\in{0,1}$, revealed only inside zkPoT:

- (b = 1): the data participates normally
- (b = 0): the data is forcibly zeroed out during gradient computation

**Feature-Level Unlearning**

Given an input matrix $x \in \mathbb{R}^{N \times J}$,
provide a mask $b\in{0,1}^{N \times J}$,
and compute gradients using:

$$
x' = x \circ b
$$

Any feature with $b_{i,j}=0$ contributes **no effect** to the gradient.

For linear regression:

$$
\nabla w_j = \frac{1}{\hat N}\sum_i
(x_{i,j}b_{i,j})
\Big(\sum_k x_{i,k}b_{i,k}w_k + \delta - \hat y_i\Big)
$$

This is mathematically equivalent to **deleting** those features entirely.

**Irrecoverable Unlearning**

To prevent a malicious trainer from “restoring” deleted data in later rounds,
a **state-preserving mask** is used:

```math
b_t^{\text{new}} \leftarrow b_{t-1} \& b_t
```

Thus once a feature is forgotten at any round, it **can never be reintroduced**.

**Sample-Level Unlearning**

Use an $N\times 1$ mask $b_i$.
The per-sample gradient term becomes:

$$
(y_i - \hat y_i) b_i
$$

If $b_i = 0$, the entire sample disappears from training.

**Class-Level Unlearning (Label Correction)**

Labels typically are not “deleted” but **corrected**.
Use XOR for bitwise label flipping:

$$
y' = y \oplus b
$$

Again, done entirely inside the ZK circuit.

### Preventing Gradient-Forgery Attacks

Even after deleting a sample, an attacker might find another minibatch $\tilde d\subseteq D\setminus U$ whose gradient approximates that of a minibatch (d) that _does_ contain forgotten data:

$$
\big|\nabla L(w,d) - \nabla L(w,\tilde d)\big| < \epsilon
$$

**Search Space Explosion**

Without constraints, the search space is huge:

- Forgery minibatch:
  $|S_f| = \binom{|D|-|U|}{|\tilde d|}$
- Target minibatch:
  $|S_t| = 2^{|D|-|U|}$

Attackers can simply brute-force combinations.

**Using VRFs to Eliminate Search Control**

With a **Verifiable Random Function (VRF)** determining the minibatch order:

- the attacker **cannot choose minibatches**
- the attacker **cannot retry combinations**

The search space collapses to:

$$
|S_f| = \left\lceil\frac{|D|}{|\tilde d|}\right\rceil
$$

The probability gap:

$$
1 - (1-p)^{\binom{|D|-|U|}{|\tilde d|}}
\gg
1 - (1-p)^{\lceil |D|/|\tilde d|\rceil}
$$

Hence successful forgery becomes astronomically harder.

**Replacement-by-Classwise-Neighbor Attacks**

A more sophisticated attack replaces the forgotten sample with its nearest **same-class** neighbor:

$$
N(x_u, y_u) =
\arg\min_{(x,y)\in D\setminus U,y=y_u} \|x - x_u\|_2
$$

If the neighbor’s gradient is similar enough, it acts as a **gradient replica**.

To prevent this, the trainer must produce a ZK proof that for every sample used in training:

$$
\big|\nabla L(x_m,y_m) - \nabla L(x_u,y_u)\big| > \xi
$$

Any sample within threshold $\xi$ is considered a forbidden replica.

---

## 3. Rethinking Randomness in ZKPoT: From Verifying the Training Trace to Proving Optimality Vicinity

Zero-Knowledge Proofs of Training (ZKPoT) aim to verify that a model was trained according to a prescribed training algorithm—typically SGD—over a specific dataset. A fundamental challenge arises from the fact that modern training algorithms rely heavily on randomness: **shuffling orders, random minibatch selection, random initialization, dropout masks**, and more. Where this randomness comes from directly affects the _security_, _trustworthiness_, and _efficiency_ of a ZKPoT system.

### The Randomness Problem in ZKPoT

**Prover-generated randomness**

In many works such as GGJ+23 and XZL+23, randomness simply comes from the prover. This matches real-world training pipelines: the model trainer generates its own random seed.

However, this also introduces a major vulnerability:
**Rejection sampling attacks.**
The prover can repeatedly sample seeds—seed₁, seed₂, seed₃, …—until one produces a model with the desired behavior. The ZK proof merely attests that _given that seed_, the training followed the prescribed algorithm; it does _not_ ensure the seed itself was chosen honestly or uniformly.

Thus, prover-generated randomness is fundamentally manipulable.

**Verifier-generated randomness**

If the verifier generates the randomness, then randomness is guaranteed to be _fair_ and non-manipulable. Unfortunately, this makes ZKPoT **interactive and impractical**, since real training requires massive amounts of randomness (per iteration, per batch, per dropout mask, per initialization).
With multiple verifiers or decentralized verification, this approach becomes infeasible.

**Random Oracle / Fiat–Shamir (FS) randomness**

Recent works (e.g., LLLX23, CSD24, SAB24) rely on the Random Oracle model. Typically, the prover hashes a commitment to derive randomness:

$$
r = H(C).
$$

But a subtle problem remains:
The prover can manipulate the commitment $C$ until the derived $r$ is favorable. It is a more hidden form of rejection sampling.

A seemingly safe approach is to derive $r$ from the training data itself:

$$
r = H(\text{training data}).
$$

But then the prover must **prove—in ZK—that the data used for randomness generation is exactly the committed training data**, which is extremely expensive for large datasets (GB-scale).

### A Conceptual Breakthrough: Stop Verifying the Trajectory

Randomness is the source of almost every security weakness in trajectory-based ZKPoT. This motivates a paradigm shift:

**Instead of proving that “the training process was executed correctly,”
prove that “the final model is close to the optimal solution.”**

This eliminates randomness entirely.

**Optimization-based soundness**

The prover no longer needs to prove:

- which minibatches were sampled,
- in what order,
- using which random seeds,
- across how many iterations.

Instead, the prover outputs a trained model $w$, and proves:

$$
||w - w^*||_2 \le \varepsilon,
$$

i.e., the model lies in an **ϵ-vicinity** of the true optimum $w^*$ induced by the data.

If $w$ is close to $w^*$, then predictions coincide:

```math
\| f(x, w) - f(x, w^*) \|_2 \le K \, \|w - w^*\|_2,
```

due to the Lipschitz continuity of the prediction function.
Thus:

- malicious training trajectories cannot deviate significantly,
- all randomness becomes irrelevant,
- the hardware and optimization algorithm become irrelevant.

This is purely a statement about the **mathematics of the optimization problem**, not its execution.

### The Mathematics: Strong Convexity as the Foundation

The core obstacle:
$w^*$ is unknown and cannot be computed inside the ZK proof.

**Strong convexity connects distance to gradient norm**

If the loss $L$ is $m$-strongly convex:

$$
L(y) \ge L(x) + \langle \nabla L(x), y-x \rangle + \frac{m}{2}\|y-x\|_2^2,
$$

then the distance to the optimum satisfies the bound:

$$
||w - w^*||_2 \le \frac{||\nabla L(w)||_2}{m}.
$$

This bound is extremely valuable—but $m$ is generally unknown or data-dependent.

**L2 regularization to the rescue**

Consider

$$
L(w) = L_{\text{emp}}(w) + \frac{\lambda}{2}\|w\|_2^2.
$$

The regularizer guarantees $L$ is **λ-strongly convex**, meaning:

$$
m \ge \lambda.
$$

Thus we obtain a fully computable, model-dependent upper bound:

$$
||w - w^*||_2 \le \frac{||\nabla L(w)||_2}{\lambda}.
$$

This can be checked entirely inside a ZK circuit without ever computing $w^*$.

### ZK Implementation Challenges: Reals vs. Discrete Fields

ZK proofs operate over discrete fields, but gradients involve real arithmetic and nonlinear functions (sigmoid, softmax). The solution is:

**Fixed-point interval arithmetic**

Every real value v is represented as an interval [L, U], guaranteeing:

$$
v \in [L, U].
$$

All operations propagate upper/lower bounds conservatively (over-approximation), ensuring soundness.

**Piecewise-linear approximations of sigmoid**

Existing approximations are too coarse, asymmetric (only upper or lower bound but not both), and expensive for ZK.

A correct ZK-friendly approach must provide simultaneously:

- a piecewise-linear lower bound,
- a piecewise-linear upper bound,
- small approximation error,
- efficient constraint representation.

These are essential to compute tight gradient bounds for logistic regression and related models.

### Error Decomposition: Why the Bound Remains Sound

The final epsilon-certified radius $\varepsilon_{\text{fp}}$ contains three sources of overestimation:

- **Strong convexity gap** $\Delta_{\text{sc}} = |\varepsilon_{\text{sc}} - \varepsilon_{\text{real}}|$:
  Using the convexity inequality adds slack.

- **Regularization gap** $\Delta_{\text{rg}} = |\varepsilon_{\text{reg}} - \varepsilon_{\text{sc}}|$:
  Using $\lambda$ instead of the true strong convexity constant $m$ introduces further slack.

- **Fixed-point gap** $\Delta_{\text{fp}} = |\varepsilon_{\text{fp}} - \varepsilon_{\text{reg}}|$:
  Interval arithmetic and piecewise approximations introduce rounding error.

Overall:

$$
\varepsilon_{\text{fp}} \ge \varepsilon_{\text{reg}} \ge \varepsilon_{\text{sc}} \ge \varepsilon_{\text{real}}.
$$

This maintains full soundness while keeping the bound practical.
Empirical results show:

- $\Delta_{\text{sc}} + \Delta_{\text{rg}} \approx 10^{-4}$,
- $\Delta_{\text{fp}} \approx 10^{-3}$.

These are negligible for most ML models.

### Strong Convexity Bounds Used in the Framework

If $f$ is $\mu$-strongly convex, i.e.:

$$
f(y) \ge f(x) + \langle \nabla f(x), y-x \rangle + \frac{\mu}{2}\|y-x\|_2^2,
$$

then the classical bounds follow:

**Quadratic Growth**

```math
f(x) - f(x^*) \ge \frac{\mu}{2}\|x - x^*\|_2^2.
```

**Polyak–Łojasiewicz (PL) inequality**

$$
\|\nabla f(x)\|_2^2 \ge 2\mu \big(f(x) - f(x^*)\big).
$$

**Strong Monotonicity**

$$
\langle \nabla f(x) - \nabla f(y), x-y \rangle \ge \mu \|x-y\|_2^2.
$$

**Lipschitz-like gradient lower bound**

$$
\|\nabla f(x) - \nabla f(y)\|_2 \ge \mu \|x-y\|_2.
$$

These inequalities lie at the heart of optimization theory (Nesterov, Bubeck) and play a crucial role in ZKML soundness proofs, FL convergence proofs, and ZK proofs of optimum vicinity.

---

[Kaizen](https://github.com/zkPoTs/kaizen)
[ZKpot](https://github.com/guruvamsi-policharla/zkpot)
[zkDL](https://github.com/SafeAILab/zkDL)
