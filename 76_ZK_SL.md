# ETAAcademy-ZKMeme: 76. ZK MPHE for Split Learning

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>76. ZKSL</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZKSL</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Beyond Partitioning: Enhancing Split Learning Privacy with Homomorphic Encryption and Zero-Knowledge Proofs

While Split Learning (SL) mitigates raw data exposure by partitioning models, it remains susceptible to critical vulnerabilities such as visual invertibility, feature space hijacking, and backdoor poisoning. To address these threats,comprehensive, multi-layered defense architectures synthesize cryptographic rigor with dynamic monitoring.

At the foundational layer, Homomorphic Encryption (HE) guarantees confidentiality, utilizing the CKKS scheme for approximate floating-point computations via rescaling and the BFV scheme for precise integer arithmetic through modulus embedding.

The intermediate layer addresses scalability and malicious threats by integrating Multi-Party Homomorphic Encryption (MPHE) with Zero-Knowledge Arguments of Knowledge (ZKAoK) within a PIOP framework. Through techniques like modulus switching, vector representation, and NTT optimization, this layer rigorously verifies key generation, ciphertext integrity, and flooding noise throughout the protocol lifecycle, effectively neutralizing IND-CPA-D attacks.

Finally, the top layer implements dynamic monitoring via Discrete Cosine Transform (DCT) and rotational distance metrics to detect concealed backdoors by identifying frequency domain anomalies and abrupt trajectory shifts. Together, these layers establish a robust shield that ensures both data privacy and computational integrity.

---

## 1. Privacy and Security Challenges in Split Learning and Cryptographic Defenses

Split Learning (SL) is a distributed learning paradigm designed to enable collaborative model training without sharing raw data. In SL, a neural network is partitioned into two segments: the client-side subnetwork and the server-side subnetwork, separated at a cut layer (also known as the shear layer). During training, the client computes the forward pass on its local data up to the cut layer and transmits only the intermediate activations—commonly referred to as _smashed data_—to the server. The server then completes the remaining forward and backward computations and returns gradients to the client. This architecture reduces computational burden on resource-constrained clients and prevents direct exposure of raw data to the server.

While SL offers inherent privacy advantages compared to centralized training, _data non-sharing does not automatically imply security_. Clients and servers are independent entities and may not be fully trusted. Consequently, SL systems are vulnerable to a variety of privacy leakage and integrity attacks, motivating the integration of cryptographic techniques such as Homomorphic Encryption (HE) and Zero-Knowledge Proofs (ZKP).

#### Threat Model and Reverse Attacks in Split Learning

A major class of threats in SL arises from **reverse attacks**, which can be understood as client-driven or server-driven manipulations of the training process. These attacks are closely related to backdoor attacks, poisoning attacks, and model manipulation attacks.

The term _“reverse”_ reflects two intertwined dimensions:

- **Reversal of training objectives**:
  Although a participant appears to follow the protocol and contribute to improving overall accuracy, its true objective is to induce malicious behaviors in specific input subspaces. As a result, the global model performs well on standard validation data while behaving abnormally or adversarially on attacker-chosen inputs.

- **Reversal of trust assumptions**:
  In SL, the server implicitly trusts the client to produce honest intermediate representations. However, the client controls the local data distribution and the feature extractor. Malicious clients can exploit this asymmetry by subtly steering the global model in undetectable ways.

#### Privacy Leakage via Smashed Data

Even when raw data are not shared, smashed data may still leak sensitive information. A variety of **inference and reconstruction attacks** have been proposed to exploit this vulnerability:

(1) Visual Invertibility Attacks

If the cut layer is placed too early (e.g., after only one convolutional layer), the smashed data may remain visually similar to the original inputs. In such cases, sensitive information is effectively exposed without requiring sophisticated attacks.

(2) Feature-Space Hijacking Attacks (FSHA)

FSHA is an active, server-side attack in which the server intentionally deviates from the honest training objective. By manipulating gradients—often using discriminator-like components inspired by GANs—the server coerces the client’s encoder into producing representations that are easily invertible. This attack typically requires auxiliary data and is difficult for the client to detect.

(3) Model Inversion Attacks

In model inversion, the attacker seeks an input $x$ such that the output of the client-side model matches the observed smashed data. Prior work has shown that even without auxiliary data, knowing the client-side architecture may suffice to reconstruct sensitive inputs, particularly when the client subnetwork is shallow.

(4) Pseudo-Client Attacks (PCAT)

PCAT is a passive and highly stealthy attack. The server follows the SL protocol honestly while secretly training a decoder using auxiliary data. Over time, it learns to map smashed data back to raw inputs without affecting the main task’s performance.

(5) SDAR (Simulator Decoding with Adversarial Regularization)

SDAR generalizes prior attacks by reducing assumptions about data distributions and model structures. The attacker trains a simulator that mimics the client’s behavior while enforcing decodability through adversarial regularization. This approach has proven effective even against U-shaped SL architectures.

#### Defense Strategies in Split Learning

Existing defenses against privacy leakage and integrity violations in SL can be broadly categorized into four classes:

(1) Data Encryption

Cryptographic approaches, including Homomorphic Encryption (HE) and Secure Multi-Party Computation (SMPC), prevent the server from accessing plaintext smashed data. HE schemes such as CKKS (supporting approximate floating-point arithmetic) and BFV (supporting exact integer arithmetic) enable computation directly on encrypted activations. While providing strong theoretical privacy guarantees, these methods incur significant computational and communication overhead.

Hybrid approaches (e.g., SL + HE) encrypt smashed data before transmission, while U-shaped SL combined with HE has been explored for specific domains such as time-series data. However, scaling these methods to multi-client scenarios remains challenging.

(2) Data Decorrelation

These methods aim to reduce statistical dependence between raw data and smashed data. Techniques include adding hidden layers on the client side, incorporating distance correlation (DC) loss into the training objective, pruning information-rich channels, and applying quantization or binarization. While effective to varying degrees, many of these methods involve trade-offs between privacy and model accuracy.

(3) Noise-Based Mechanisms

Inspired by differential privacy, noise is injected into smashed data to obscure sensitive information. Adaptive approaches such as _Shredder_ treat noise as a learnable parameter, dynamically balancing privacy and accuracy. Although computationally efficient, excessive noise degrades model performance, especially under non-IID data distributions.

(4) Attack-Specific Defenses

These include techniques tailored to particular attack vectors, such as non-local parameter sharing to mitigate model inversion, adversarial pretraining of feature extractors (e.g., ResSFL), patch-based training to disrupt spatial structure, and gradient consistency checks to detect FSHA.

#### Limitations of Federated Learning Defenses in SL

Defensive mechanisms commonly used in Federated Learning (FL) do not directly transfer to SL. Secure aggregation relies on parallel updates and aggregation, which is incompatible with SL’s sequential training process. Robust aggregation methods such as KRUM and Multi-KRUM require simultaneous access to multiple client updates, a condition not satisfied in standard SL settings. Differential privacy, while useful, often leads to significant accuracy degradation, particularly in non-IID scenarios.

#### Cryptographic Foundations: HE and ZKP in Split Learning

Homomorphic Encryption addresses the **privacy problem**, enabling servers to compute on encrypted smashed data without ever observing plaintext. Zero-Knowledge Proofs address the **trust problem**, ensuring that both clients and servers follow the prescribed protocol honestly.

In SL, ZKP can be used for **computation integrity verification**:

- The client proves that it has correctly executed the forward computation without embedding backdoors:

  $\pi_i = \text{ZKP.Prove}(f(X_i; \theta_i), pk_s)$

- The server verifies the proof without accessing raw data:

  $\text{ZKP.Verify}(f(X_i; \theta_i), \pi_i, pk_s) = \text{True}$

Similarly, the server may prove the correctness of backward computations to the client. Together, HE and ZKP enable SL systems to achieve both _data confidentiality_ and _computation integrity_, preventing information leakage and malicious manipulation while preserving collaborative learning.

---

### A Multi-Layer Defense Framework Against Client-Side Backdoor Attacks in Split Learning

To address the unique security challenges of Split Learning (SL), we consider a defense framework specifically designed to detect and mitigate **client-side backdoor attacks**. Unlike parallel distributed learning paradigms, SL operates sequentially and relies heavily on the trustworthiness of individual clients, making it particularly vulnerable to stealthy malicious behavior. The proposed framework integrates cryptographic protection with model-behavior monitoring, forming a multi-layer defense architecture.

#### Overview of the Defense Architecture

The defense system is composed of three tightly coupled layers:

- **Data Confidentiality Layer**
  Homomorphic Encryption (HE) is employed to encrypt intermediate representations (smashed data), ensuring that the server cannot access or infer raw client data during training.

- **Computation Integrity Layer**
  Zero-Knowledge Proofs (ZKP) are used to guarantee that both the client and the server execute their respective computations honestly and according to the protocol, without revealing sensitive information.

- **Model Behavior Monitoring Layer**
  A dedicated detection mechanism analyzes model updates to identify anomalous patterns indicative of backdoor insertion. Upon detection, a rollback mechanism restores the model to a previously verified safe state.

Together, these layers provide end-to-end protection covering data privacy, protocol compliance, and model integrity.

#### Circular Backward Analysis and Rollback Mechanism

The core of the defense framework is a **Circular Backward Analysis** performed after each client’s training phase. Rather than relying solely on forward verification, the system analyzes the _effect_ of client updates on the global model by examining changes across successive training iterations.

If suspicious behavior is detected, the system activates a **rollback mechanism**, reverting the server-side model to the most recent verified checkpoint. The server maintains a history of validated model states, allowing it to discard malicious updates before they propagate further into the training process.

This rollback strategy is particularly effective in SL, where clients participate sequentially and where a single malicious client could otherwise permanently corrupt the model.

#### Dual Analysis: Static and Dynamic Detection

To improve robustness and reduce false positives, the framework employs a **dual-analysis approach**, combining static and dynamic indicators of malicious behavior.

**Frequency-Domain Analysis Using Discrete Cosine Transform**

The static analysis component relies on the **Discrete Cosine Transform (DCT)** to examine model parameter updates in the frequency domain. Rather than inspecting raw parameter values directly (spatial domain), the method analyzes how the _frequency distribution_ of model updates evolves over time.

Let $B_t$ and $B_{t-1}$ denote consecutive server-side model parameters or update vectors. The difference between these updates is transformed into the frequency domain as:

$$
S_t = \text{DCT}_{\text{low}}(B_t - B_{t-1})
$$

Here, $\text{DCT}_{\text{low}}(\cdot)$ extracts the low-frequency components of the transformed update. In normal training, optimization primarily refines global, smooth features of the model, leading to stable and gradual changes in low-frequency energy. In contrast, backdoor attacks often introduce abrupt or irregular modifications that disrupt this stability, causing detectable deviations in the frequency spectrum.

This approach leverages a well-known observation: while malicious updates may appear subtle or noisy in the parameter space, they often reveal distinctive patterns when analyzed in the frequency domain.

#### Mathematical Foundation of the DCT-Based Analysis

The Discrete Cosine Transform decomposes a matrix into a weighted sum of cosine basis functions at different frequencies. Given an input matrix $A \in \mathbb{R}^{M \times N}$, the two-dimensional DCT produces a frequency-domain representation $F$, where:

$$
F_{uv} = \alpha_u \alpha_v
\sum_{x=0}^{M-1} \sum_{y=0}^{N-1}
A_{xy}
\cos\left(\frac{\pi(2x+1)u}{2M}\right)
\cos\left(\frac{\pi(2y+1)v}{2N}\right)
$$

Each coefficient $F_{uv}$ measures the similarity between the input matrix and a cosine pattern of frequency $(u, v)$. Low-frequency components capture smooth, global structures in the model parameters, while high-frequency components correspond to localized or noisy variations. Backdoor attacks often exploit high-frequency or irregular patterns that are difficult to detect in the spatial domain but become evident after transformation.

**Dynamic Analysis via Rotational Distance**

Complementing the static frequency analysis, the dynamic component measures **directional consistency** between consecutive model updates using the **rotational distance** metric. Instead of focusing on the magnitude of parameter changes, this method examines the _direction_ of updates in parameter space.

The rotational distance between updates $B_t$ and $B_{t-1}$ is defined as:

$$
\theta(t) =
\arccos\left(
\frac{B_t \cdot B_{t-1}}{|B_t| , |B_{t-1}|}
\right)
$$

This expression computes the angle between two update vectors using cosine similarity. Under normal gradient-based optimization, consecutive updates tend to follow similar directions as the model converges toward an optimum, resulting in small and stable angles. In contrast, a backdoor attack introduces abrupt directional shifts as the attacker attempts to steer the model toward encoding malicious behaviors. Such shifts manifest as sudden increases in $\theta(t)$, providing strong evidence of anomalous training dynamics.

**Integrated Detection and Mitigation**

By jointly analyzing frequency-domain deviations and directional instability, the framework achieves high sensitivity to client-side backdoor attacks while maintaining low false-positive rates. When either analysis indicates suspicious behavior beyond predefined thresholds, the system triggers the rollback mechanism, restoring the model to a previously verified state.

This integrated approach ensures that malicious updates are detected early, isolated, and neutralized, all while preserving model utility and training efficiency.

---

## 2. Homomorphic Encryption for Privacy-Preserving Neural Network Training

Homomorphic Encryption (HE) is a fundamental cryptographic technique for protecting **data confidentiality** during both training and inference in collaborative or distributed machine learning systems. In such settings, clients wish to leverage powerful server-side models without ever revealing their raw data. Among existing HE schemes, **CKKS** (designed for approximate real-number arithmetic) and **BFV** (designed for exact integer arithmetic) are the most widely adopted.

This section explains how HE—particularly CKKS—enables secure neural network computation, including forward and backward propagation, while highlighting its mathematical foundations and practical limitations.

#### Encrypted Data Flow in Neural Network Training

Each client $C_i$ encrypts its local dataset $X_i$ using its public key $pk_i$:

$$
X_{\text{enc}_i} = \text{HE.Enc}(X_i, pk_i)
$$

This ensures that data are already encrypted **before leaving the client**, guaranteeing confidentiality against the server.

The server receives encrypted activations $Z_{enc_i}$ and treats them as inputs to its neural network layers (from $Layer_{cut+1}$ to $Layer_{end}$). Crucially, the server **never decrypts** these values.

#### Homomorphic Computation on Encrypted Data

Homomorphic encryption supports arithmetic operations directly on ciphertexts:

$$
\text{Enc}(A) + \text{Enc}(B) = \text{Enc}(A + B), \quad
\text{Enc}(A) \times \text{Enc}(B) = \text{Enc}(A \times B)
$$

Using these properties, the server evaluates its neural network on encrypted inputs:

$$
y_{\text{enc}_S} = \text{HE.Eval}(Z_{\text{enc}_i}; \theta_S)
$$

The output $y_{\text{enc}_S}$ remains encrypted and is sent back to the client, who alone possesses the private key required for decryption.

#### Linear Layers and Matrix Multiplication

Neural networks are dominated by **matrix multiplications**, which are naturally compatible with HE.

Consider a simple example:

$$
\mathbf{x} = [x_1, x_2], \quad
W =
\begin{bmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22}
\end{bmatrix}
$$

The output is

$$
\begin{aligned}
y_1 &= x_1 w_{11} + x_2 w_{21} \\
y_2 &= x_1 w_{12} + x_2 w_{22}
\end{aligned}
$$

After encryption, the server receives $[\text{Enc}(x_1), \text{Enc}(x_2)]$. It multiplies these ciphertexts by plaintext weights $w_{ij}$, sums the results homomorphically, and obtains encrypted outputs $[\text{Enc}(y_1), \text{Enc}(y_2)]$. No plaintext data are ever revealed.

#### The Challenge of Nonlinear Activations

While HE excels at addition and multiplication, it struggles with nonlinear functions such as **ReLU**, **Sigmoid**, and **Softmax**. Two common strategies are used:

- **Client-side nonlinearity**:
  The server computes encrypted linear layers and sends results back to the client. The client decrypts, applies nonlinear functions, re-encrypts, and returns the ciphertexts.

- **Polynomial approximation**:
  Nonlinear functions are approximated using low-degree polynomials (e.g., replacing ReLU with $x^2$) that are HE-friendly.

Both approaches introduce trade-offs between efficiency, accuracy, and communication cost.

#### Encrypted Backpropagation and Gradient Flow

Because the server cannot observe prediction errors directly, the **client must act as a training oracle**:

- The client decrypts the model’s prediction.
- It computes the loss and the initial gradient.
- This gradient (encrypted) is sent back to the server.

Using homomorphic operations, the server backpropagates this encrypted error through its layers and updates model parameters—without ever learning the true labels or intermediate activations.

#### Why CKKS Is the Scheme of Choice

Traditional schemes like BFV perform **exact integer arithmetic**, which is ill-suited for neural networks that rely on floating-point values and tolerate small numerical errors.

CKKS (Cheon–Kim–Kim–Song) introduces **approximate arithmetic**:

$$
\text{Dec}(\text{Enc}(m_1) + \text{Enc}(m_2)) \approx m_1 + m_2
$$

This approximation is acceptable—and often negligible—in deep learning, dramatically improving performance and scalability.

#### Mathematical Foundations of CKKS

**Message, Polynomial, and Ciphertext Spaces**

CKKS computation involves three mathematical domains:

- **Message space**:
  $\mathbb{C}^{N/2}$, a vector of complex (floating-point) numbers.
- **Plaintext space**:
  A polynomial ring $R = \mathbb{Z}[X]/(X^N + 1)$.
- **Ciphertext space**:
  A pair of polynomials $(c_0, c_1) \in R_q^2$.

**Encoding via Canonical Embedding**

Encoding maps a floating-point vector into a polynomial:

- **Scaling**: Multiply the vector by a large factor $\Delta$ (e.g., $2^{40}$).
- **Inverse canonical embedding**: Map the scaled vector into a polynomial:

  $m(X) = \left\lfloor \Delta \cdot \sigma^{-1}(\mathbf{z}) \right\rceil$

- **Discretization**: Polynomial coefficients are rounded to integers.

The canonical embedding $\sigma$ evaluates a polynomial at complex roots of unity, and its inverse allows multiple real values to be packed into a single polynomial—enabling **SIMD batching**.

#### Encryption Based on RLWE

CKKS security relies on the **Ring Learning With Errors (RLWE)** assumption.

- **Secret key**: A short polynomial $s$
- **Public key**:

  $pk = (b, a), \quad b = -a s + e \pmod q$

To encrypt a message polynomial $m$, fresh randomness $u, e_0, e_1$ is used:

$$
\begin{aligned}
c_0 &= b u + e_0 + m \pmod q \
c_1 &= a u + e_1 \pmod q
\end{aligned}
$$

The message is hidden by noise and can only be recovered using the secret key.

#### SIMD Batching and Efficient Evaluation

Thanks to canonical embedding, polynomial multiplication corresponds to **component-wise multiplication** in the message space:

$$
\sigma(m_1 \cdot m_2) = \sigma(m_1) \odot \sigma(m_2)
$$

Thus, a single ciphertext multiplication performs $N/2$ independent floating-point multiplications—providing massive parallelism.

#### Rescaling: Managing Numerical Growth

Each ciphertext multiplication doubles the scale:

$$
(\Delta m_1) \times (\Delta m_2) = \Delta^2 m_1 m_2
$$

To prevent overflow, CKKS applies **rescaling**, which divides the ciphertext by $\Delta$ while reducing the modulus:

$$
ct' = \left\lfloor \frac{q_{L-1}}{q_L} \cdot ct_{\text{mult}} \right\rceil \bmod q_{L-1}
$$

Each rescaling consumes one modulus level, limiting the maximum depth of the neural network. This is analogous to multi-stage rocket boosters: deeper computation requires a larger initial modulus chain.

#### Decryption and Approximate Decoding

Decryption computes:

$$
m' = c_0 + c_1 s \pmod q
$$

Substituting the ciphertext definitions yields:

$$
m' = m + (e u + e_0 + e_1 s)
$$

As long as the total noise remains below $q/2$, the message can be recovered.

Finally, decoding applies the canonical embedding and rescales by $\Delta$:

$$
\mathbf{z}_{\text{final}} = \sigma(m') / \Delta
$$

The result is an **approximation** of the original data—precise enough for deep learning, yet secure against adversaries.

---

### The BFV Homomorphic Encryption Scheme: Exact Arithmetic by Design

**BFV (Brakerski–Fan–Vercauteren)** is another foundational pillar of homomorphic encryption, but its philosophy is fundamentally different from CKKS. If CKKS can be viewed as an _engineering-oriented_ scheme—where small approximation errors are tolerated for efficiency—BFV is more like a **meticulous accountant**: every computation must be _exact_, with zero numerical error.

BFV supports **exact integer arithmetic** only. All plaintexts are integers, and all homomorphic computations are guaranteed to be correct modulo a specified plaintext modulus. For example, encrypting (2) and (3), multiplying them homomorphically, and decrypting will yield exactly (6)—never (5.9999).

All BFV computations take place modulo a plaintext modulus $t$. For instance, if $t = 10$, then

$7 + 5 = 12 \equiv 2 \pmod{10}$.

#### Why Use BFV in Machine Learning?

At first glance, BFV appears ill-suited for deep learning, which predominantly relies on floating-point arithmetic. However, BFV remains highly relevant in several important scenarios:

**Quantized Neural Networks**

Many production systems deploy **quantized models**, where parameters and activations are converted from 32-bit floating point to 8-bit or 16-bit integers (e.g., int8) for speed and memory efficiency. In such models:

- All computations are integer-only
- Numerical exactness is required

BFV is a natural fit here: the encrypted computation produces **bitwise-identical results** to plaintext int8 inference, with zero approximation error.

**Boolean Logic and Comparisons**

Certain operations—such as comparisons ((x > 0)), thresholding, or Boolean logic—are difficult to express accurately under CKKS due to its approximate nature. BFV, while slower, is more suitable for such discrete operations because it preserves exactness throughout computation.

#### Core Design Principle of BFV

The central idea of BFV is to **separate signal from noise by scale**.

- **Plaintext modulus $t$**: small (e.g., (2), (1024))
- **Ciphertext modulus $q$**: very large (e.g., $2^{60}$)
- **Scaling factor**

  $\Delta = \left\lfloor \frac{q}{t} \right\rfloor$

Plaintext messages are lifted into the _high bits_ of the ciphertext space by multiplying them with $\Delta$, while all noise terms remain confined to the _low bits_. As long as noise does not overflow upward, it can be cleanly removed during decryption.

All operations are performed in polynomial rings:

- $R_t = \mathbb{Z}_t[X]/(X^N + 1)$
- $R_q = \mathbb{Z}_q[X]/(X^N + 1)$

This scale separation is the mathematical foundation of BFV’s exactness.

#### Encoding: Coefficient Mapping

Unlike CKKS, BFV uses a **direct and exact encoding** strategy.

Given an integer vector

$$
\mathbf{m} = [m_0, m_1, \dots, m_{N-1}], \quad m_i \in [0, t-1],
$$

the encoding simply maps coefficients to a polynomial:

$$
m(X) = m_0 + m_1 X + m_2 X^2 + \dots + m_{N-1} X^{N-1} \in R_t.
$$

No scaling, no approximation, and no floating-point conversion are required.
For example, the vector ([1, 5, 9]) is encoded as:

$$
m(X) = 1 + 5X + 9X^2.
$$

#### Encryption and Noise Management

BFV encryption is based on the **Ring Learning With Errors (RLWE)** assumption, similar to CKKS, but the handling of noise is fundamentally different.

**Key Generation**

- **Secret key**: a small polynomial $s$
- **Public key**:

  $pk = (b, a), \quad b = -(a \cdot s + e) \pmod q,$

  where $e$ is a small Gaussian noise polynomial.

**Encryption**

To encrypt a plaintext polynomial $m$, fresh randomness $u, e_0, e_1$ is sampled:

$$
\begin{aligned}
c_0 &= b \cdot u + e_0 + \Delta \cdot m \pmod q, \
c_1 &= a \cdot u + e_1 \pmod q.
\end{aligned}
$$

The crucial term is $\Delta \cdot m$:

- Because $\Delta \approx q/t$ is very large,
- The plaintext is embedded into the **most significant bits (MSB)** of the ciphertext,
- While all noise terms remain in the **least significant bits (LSB)**.

This structural separation enables exact recovery.

#### Decryption: Noise Removal by Rounding

Decryption explicitly removes noise rather than tolerating it.

#### Inner Product

Compute:

$$
\begin{aligned}
v &= c_0 + c_1 \cdot s \
&= \Delta \cdot m + \underbrace{(e_0 + e_1 s - e u)}_{E} \pmod q.
\end{aligned}
$$

Thus,

$$
v \approx \frac{q}{t} \cdot m + E.
$$

#### Scaling and Rounding

To recover $m$, BFV applies:

$$
m' = \left\lfloor \frac{t}{q} \cdot v \right\rceil \pmod t.
$$

Substituting $v$:

$$
\frac{t}{q} \cdot v = m + \frac{t}{q} \cdot E.
$$

As long as the accumulated noise satisfies:

$$
|E| < \frac{q}{2t},
$$

the term $\frac{t}{q} \cdot E$ remains strictly smaller than (0.5), and rounding **perfectly eliminates the noise**, yielding the exact plaintext $m$.

This is the defining mechanism behind BFV’s correctness guarantee.

#### Homomorphic Computation and Noise Growth

Homomorphic addition and multiplication increase the noise term $E$. BFV correctness depends on ensuring that noise growth never exceeds the rounding threshold. This constraint limits circuit depth and motivates techniques such as:

- Parameter tuning
- Modulus switching
- Noise flooding (in advanced protocols)

#### Share Aggregation and Threshold Decryption

In distributed or multi-party settings, decryption is often performed collaboratively.

Each participant computes a partial decryption share:

$$
h_i = c_0 \cdot s_i + e_{\text{flood}} \pmod q.
$$

The server aggregates all shares:

$$
m \approx \text{Merge}(ct_{\text{res}}, h_1, \dots, h_N),
$$

canceling masking terms and recovering the plaintext.

#### Security Against Malicious Participants

In malicious threat models, participants may attempt to cheat by:

- Submitting malformed public key shares during DKG
- Providing incorrect partial decryptions

To enforce honest behavior, systems employing BFV-based distributed decryption must incorporate **ZKAoK (Zero-Knowledge Arguments of Knowledge)**. These proofs ensure that each participant:

- Knows the secret corresponding to their public contribution
- Computes decryption shares correctly

without revealing any secret information.

---

## 3. From Standard Homomorphic Encryption to Maliciously Secure Multi-Party HE

#### Standard Homomorphic Encryption

Traditional **Homomorphic Encryption (HE)** enables computation directly on encrypted data without decryption. Its canonical deployment model is **server–client**:

- The client encrypts its data and sends ciphertexts to the server.
- The server evaluates a function homomorphically on ciphertexts.
- The encrypted result is returned and decrypted by the client.

A key advantage of standard HE is its **optimal round complexity**. Unlike secure multi-party computation (MPC), which often requires many rounds of interaction, HE-based protocols typically complete in only a few rounds. Communication cost is also relatively low and depends mainly on **input and output size**, not on the complexity of the computation itself. This makes HE particularly attractive for applications such as:

- Private Information Retrieval (PIR)
- Private Set Intersection (PSI)
- Secure inference for machine learning models

#### The Scalability Bottleneck of Standard HE

Despite its efficiency, standard HE has a fundamental limitation:
**ciphertexts encrypted under different public keys cannot be combined**.

If three users (A, B, C) encrypt their data using distinct public keys, the server cannot add or multiply those ciphertexts together. This severely limits scalability in collaborative settings.

A natural workaround is to have all participants agree on a **shared public key** and encrypt their inputs under that key. This idea leads directly to **Multi-Party Homomorphic Encryption (MPHE)**.

#### Multi-Party Homomorphic Encryption (MPHE)

MPHE extends HE to multi-party settings by enabling multiple participants to jointly generate cryptographic keys and perform homomorphic computation on collectively encrypted data. Importantly:

- MPHE can be used to construct **general MPC protocols**
- It inherits HE’s key efficiency advantage: communication cost depends only on input/output size, not circuit depth

In its basic form, MPHE protocols assume a **semi-honest adversary model**.

#### Semi-Honest vs. Malicious Security

**Semi-Honest Model**

In the semi-honest (honest-but-curious) model, participants are assumed to follow the protocol exactly, while possibly trying to infer additional information from observed messages. Under this assumption, MPHE-based MPC protocols are relatively straightforward to design.

**Malicious Model**

In the malicious model, participants may deviate arbitrarily from the protocol. They may:

- Submit malformed ciphertexts
- Manipulate public keys
- Provide incorrect decryption shares

To defend against such behavior, **every message** (ciphertext, public key, decryption share) must be accompanied by a **Zero-Knowledge Argument of Knowledge (ZKAoK)** proving:

> “This object is well-formed, and I know the secret values underlying it.”

This requirement poses a major challenge, because MPHE objects—especially keys and ciphertexts—have highly structured algebraic forms. Naively generating ZKAoKs for them results in proofs that are **prohibitively large and slow to verify**.

#### IND-CPA-D Attacks and the Need for Compilation

MPHE schemes are typically secure only against semi-honest adversaries. In the presence of malicious adversaries, they become vulnerable to powerful attacks, most notably **IND-CPA-D attacks**.

In such attacks, an adversary submits malformed ciphertexts and exploits information leaked during decryption (or error handling) to recover secret keys. Recent work has shown that this attack vector is extremely dangerous in practice.

The standard countermeasure is **compilation**:

$$
\text{Passively Secure MPHE} + \text{ZKAoK} \Rightarrow \text{Maliciously Secure MPHE}
$$

However, the bottleneck remains the efficiency of ZKAoKs.

PELTA, previously the state of the art, suffers from three major limitations:

- **Poor scalability**
  PELTA is built on the LANES framework, which targets small prime moduli. In contrast, HE uses large composite moduli represented via RNS (product of many small primes). PELTA must generate proofs separately for each prime, leading to dozens of repetitions and extremely high overhead.

- **Lack of automorphism key verification**
  Automorphism (Galois) keys are essential for ciphertext rotations, which are required for matrix multiplication and advanced HE protocols. Without them, MPHE supports only basic arithmetic.

- **Incomplete evaluation**
  PELTA evaluates only ciphertext encryption/decryption proofs, omitting full public key validation. This raises serious concerns about practical deployability.

#### Polynomial IOPs (PIOPs) as a Solution

**Polynomial Interactive Oracle Proofs (PIOPs)**—the foundation of modern systems like Plonk—provide a powerful alternative. PIOPs reduce verification to polynomial consistency checks and can produce **sublinear-size proofs**, which is critical when MPHE public keys are hundreds of megabytes in size.

Key MPHE objects that must be verified include:

- Encryption keys
- Relinearization keys
- Automorphism (rotation) keys

PIOPs are instantiated using **lattice-based Polynomial Commitment Schemes (PCS)**, making them compatible with HE security assumptions.

#### Bridging the Structural Gap: Rings vs. Vectors

A fundamental challenge arises from a mismatch in algebraic structure:

- MPHE operates over polynomial rings

  $R_q = \mathbb{Z}_q[X]/(X^N + 1)$

- PIOPs are designed for vector spaces

  $\mathbb{Z}_p^N$

**Ring–Vector Isomorphism**

When the condition $2N \mid (p - 1)$ holds, the two structures are **isomorphic**. This allows polynomial constraints in MPHE to be translated into vector constraints suitable for PIOPs.

#### Two Core Techniques

**Modulus Switching: From (q) to (p)**

HE moduli (q) are typically composite (RNS), while PIOPs require a large prime field (p).

Instead of proving correctness separately modulo each $q_i$, we modify the protocol:

- Generate keys, ciphertexts, and decryption shares directly in $R_p$
- Generate ZKAoKs efficiently in the prime field
- Switch values back to $R_q$ only for homomorphic computation

Crucially:

$$
\text{Well-formed in } R_p \iff \text{Well-formed in } R_q
$$

This avoids the massive overhead of proofs over RNS moduli.

**Vector Representations of Polynomials**

We use two complementary views of the same polynomial:

- **Coefficient form**

  $\text{Coeff}(a) = (a_0, a_1, \dots, a_{N-1})$

  Used for **norm and noise bound checks**

- **NTT form**

  $\text{NTT}(a) = (a(\xi), a(\xi^3), \dots, a(\xi^{2N-1}))$

  Used for **arithmetic relations**

Under the ring–vector isomorphism, polynomial multiplication becomes **component-wise multiplication**, which is exactly what **Row Check PIOPs** handle efficiently.

#### Optimizing Representation Switching

Switching between coefficient and NTT forms corresponds to a large linear transformation.

- Naive verification costs $O(N^2)$, which is infeasible for $N = 65{,}536$
- We exploit the identity:

  $T^\top \vec{v} = N \cdot \text{iNTT}(J \cdot \vec{v})$

  reducing complexity to $O(N \log N)$

This optimization transforms an impractical verification into an efficient one.

#### Reducing Inequalities to Equalities

PIOPs excel at checking polynomial equalities, but not inequalities. To prove coefficient bounds $a_i \in [-B, B]$, we use **ternary decomposition**:

- Decompose (a) into a weighted sum of vectors with entries in ({-1, 0, 1})
- Enforce ternary constraints via:

  $x(x - 1)(x + 1) = 0$

This converts complex range checks into simple polynomial equations, perfectly suited to Row Check PIOPs.

#### Securing the MPHE Lifecycle

To achieve malicious security, ZKAoKs are required at three critical stages:

- **Setup Phase**
  Verify public key shares, encryption keys, relinearization keys, and automorphism keys

- **Encryption Phase**
  Prove that input ciphertexts are well-formed and noise-bounded

- **Decryption Phase**
  Prove correctness of partial decryption shares and noise distribution

Together, these checks compile a semi-honest MPHE protocol into a **fully maliciously secure MPHE system**.

---

### Zero-Knowledge Arguments of Knowledge for MPHE via PIOPs

In multi-party homomorphic encryption (MPHE), ensuring correctness and security in the presence of potentially malicious participants requires more than passive assumptions. In particular, all cryptographic objects—public keys, ciphertexts, evaluation keys, and decryption shares—must be **well-formed**. This work adopts **Polynomial Interactive Oracle Proofs (PIOPs)** to construct efficient zero-knowledge arguments of knowledge (ZKAoKs) that enforce such guarantees throughout the MPHE lifecycle.

A central design principle is the strategy of **“prove over $R_p$, use over $R_q$”**, i.e., generating proofs modulo a smaller ring $R_p$ to reduce proof cost, while ensuring that correctness and security are preserved when lifting the objects back to the larger modulus ring $R_q$.

#### PIOPs for the Setup Phase

During the setup phase, the protocol must verify that all participants generate **valid public key shares**. Only if public keys are correctly formed can subsequent homomorphic operations—encryption, multiplication, and rotations—be guaranteed to function securely and correctly.

**Encryption Key (ek)**

The encryption key is used to encrypt plaintext data. In RLWE-based schemes, it has the form

$$
pk = (b, a), \quad \text{where } b = -a \cdot s + e.
$$

The prover must show that the public key was generated from a secret key $s$ and a small noise polynomial $e$.

- **Algebraic correctness.**
  The verifier checks the relation

  $p = -u_{ek} \cdot s + e_{ek} \pmod p,$

  using a **Row Check PIOP**.

- **Noise bound.**
  The coefficients of $s$ and $e_{ek}$ must be small:

  $s_∞ ≤ B,  e_{ek_∞} ≤ B.$

  This is enforced using a **Norm Check PIOP**.

**Relinearization Key (rlk)**

Relinearization keys are required to compress ciphertexts after homomorphic multiplication.

- **Structural correctness.**
  The three components $(r_0, r_1, r_2)$ must satisfy the prescribed linear construction equations. This is verified via a **Row Check PIOP**.

- **Noise bound.**
  The secret key $s$, auxiliary polynomial $f_{rlk}$, and noise $e_{rlk}$ must all have small coefficients. A **Norm Check PIOP** enforces this constraint.

Importantly, once the noise bound is verified in $R_p$, the key remains valid after lifting to $R_q$.

**Automorphism Key (atk)**

Automorphism keys enable ciphertext rotations (e.g., cyclic shifts in packed plaintext vectors) corresponding to the automorphism $X \mapsto X^k$.

Verifying automorphism keys is challenging in the coefficient domain, since the relation between $s(X)$ and $s(X^k)$ is non-local. However, in the **NTT domain**, this automorphism corresponds to a simple permutation:

$$
\mathrm{NTT}(s(X^k)) = P \cdot \mathrm{NTT}(s(X)),
$$

where $P$ is a permutation matrix.

Because $P$ is sparse, this relation can be efficiently verified using a **Linear Check PIOP**, achieving near-linear verification complexity.

#### PIOPs for the Encryption Phase

During the encryption phase, each participant encrypts their private data into a ciphertext

$ct = (c_0, c_1)$

and broadcasts it to the other parties. To prevent maliciously crafted ciphertexts—such as those with excessive noise or adversarial structure—the protocol must verify **ciphertext well-formedness**.

Under the BFV/CKKS encryption formula, the ciphertext satisfies

$$
\begin{aligned}
c_0 &= p \cdot f + \lfloor p/t \rceil \cdot m + e_0 \pmod p, \
c_1 &= u_{ek} \cdot f + e_1 \pmod p,
\end{aligned}
$$

where:

- $m$ is the plaintext message,
- $f, e_0, e_1$ are freshly sampled small-noise polynomials,
- $p, u_{ek}$ are public parameters.

The proof verifies:

- The algebraic consistency of the encryption equations.
- That $f, e_0, e_1$ have bounded norms $(\le B)$.

Once these properties hold in $R_p$, the ciphertext remains decryptable with controlled noise after modulus switching to $R_q$.

#### PIOPs for the Decryption Phase

In the final phase, participants cooperatively decrypt the result. Each participant computes a **partial decryption share**

$$
d = c_1 \cdot s + e_{\text{flood}} \pmod p,
$$

where $s$ is the participant’s secret key and $e_{\text{flood}}$ is a **flooding (smudging) noise** added to hide information about $s$.

**Challenge: Large Noise Verification**

The flooding noise $e_{\text{flood}}$ is intentionally very large—often hundreds of bits—far exceeding the usual noise bound $B$. Directly applying a Norm Check would be prohibitively expensive, as its cost grows with $\log B_{\text{flood}}$.

**Noise Decomposition Technique**

To avoid this overhead, the protocol constructs the flooding noise as

$$
e_{\text{flood}} = [u_{DD} \cdot f_{DD} + e_{DD}]_{B_{DD}},
$$

where:

- $f_{DD}, e_{DD}$ are small-noise polynomials,
- $u_{DD}$ is a public random polynomial,
- $B_{DD}$ denotes reduction modulo a large bound $B_{DD}$.

There exists an auxiliary polynomial $k$ such that

$$
B_{DD} \cdot k = u_{DD} \cdot f_{DD} + e_{DD} - e_{\text{flood}} \pmod p,
$$

which is equivalent to proving

$$
e_{\text{flood}} \equiv u_{DD} \cdot f_{DD} + e_{DD} \pmod{B_{DD}}.
$$

This transformation reduces the problem of verifying a **single large noise polynomial** to verifying:

- several **small-norm polynomials**, and
- a simple modular consistency equation.

As a result, the protocol avoids expensive large-range proofs while still guaranteeing correctness and zero-knowledge security.
