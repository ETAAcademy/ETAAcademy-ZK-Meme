# ETAAcademy-ZKMeme: 75. ZK Cross-chain Interoperability

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>75. ZKCCI</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZKCCI</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Zero-Knowledge Verification and Public Randomness for Cross-Chain Interoperability

Blockchain cross-chain interoperability can be improved by using public, tamper-proof randomness together with verifiable cross-chain data, enabling a new interoperability model that does not rely on centralized relays or multisignature bridges. In this design, a VRF seed is derived from a public randomness beacon (e.g., NIST Beacon 2.0) to fairly and unpredictably select a committee. Committee members fetch external data and validate signatures, membership, ordering rules, and median calculations inside a circuit (e.g., Halo2), allowing any destination chain to verify the result independently. From a security perspective, adversarial influence on committee selection is kept extremely low, and the combination of fresh randomness across epochs provides strong forward security.

In fully asynchronous environments, a post-quantum Distributed Randomness Beacon (DRB) can be built using SIS/Module-SIS–based batched polynomial commitments. Together with RBC broadcast and Merkle-based share-reconstruction proofs, this approach yields high-throughput, bias-resistant randomness. The split-and-fold technique keeps commitment and opening proofs logarithmic in size while preserving a transparent and quantum-resistant setup.

Finally, Quantum Random Number Generators (QRNGs) derive entropy from inherently unpredictable quantum physical processes. Through trusted or semi-trusted measurements, parameter estimation, and extraction using Toeplitz or Trevisan extractors, they produce certified random bits that are statistically indistinguishable from ideal uniform randomness.

---

## 1. **ZK-Oracle for Cross-Chain Interoperability: Architecture, Randomness, and Halo2 Circuits**

Cross-chain interoperability (CCI) requires blockchains to exchange data in a trust-minimized, verifiable manner. Two components are essential:

- **Unbiased, unpredictable randomness** for committee selection.
- **Cross-chain data packets** whose correctness can be verified by any chain.

Traditional randomness approaches suffer from trust assumptions:

- **VRF single-node randomness** introduces latency and allows the node to see results early.
- **Commit–reveal schemes** suffer from selective-reveal manipulation and front-running attacks.
- **Distributed Randomness Beacons (DRBs)** are more robust but often assume synchrony and have high communication complexity.

Modern cross-chain oracles and bridges remain partially centralized or rely on weak trust assumptions. A fully decentralized oracle must provide **data + proof**, rather than requiring any chain to trust "who" produced the data.

Sober et al. propose a **zk-SNARK–based oracle** where each oracle packet contains a **Halo2 proof** certifying that:

- the data was correctly collected from a threshold of reporters,
- the reporters were selected from a verifiable committee based on unbiased randomness,
- the value (e.g., median) was computed correctly.

Every chain performs **local verification** of the ZK proof without trusting any aggregator or third-party relay.

### **(1) Lightweight Cross-Chain Verifier Contracts**

Each destination chain deploys only:

- a **Halo2 verifier contract**, and
- an **oracle-packet parser**.

The system operates in discrete epochs. In each epoch the system performs:

- randomness generation,
- VRF-based reporter committee selection,
- data collection,
- Halo2 proof generation,
- cross-chain verification.

### **(2) NIST Beacon 2.0 as a Practical QRNG Substitute**

Quantum RNG devices are expensive and difficult to operate at scale. Instead, the system uses **NIST Randomness Beacon 2.0** as a _software-based quantum randomness proxy_:

- produces 512-bit randomness every 60 seconds,
- values are hash-chained (tamper-evident),
- publicly available and verifiable,
- unpredictable until published.

At the start of epoch _t_, the beacon output provides:

$$
\rho_t := H(\text{beacon}[r])
$$

This serves as the **epoch seed**. Every validator computes a VRF output:

$$
y_i,; \pi_i := \mathrm{VRF}_{sk_i}(\rho_t || t)
$$

Committee selection sorts validators by $y_i$; the first _n_ become the **reporter committee** $R_t$.

Security properties:

- VRF outputs are unpredictable → **no one can anticipate committee membership**.
- VRF outputs are verifiable → **Sybil resistance**.
- The beacon seed is unmanipulable → **bias resistance**.

Probability an adversary controlling _b_ out of _n_ reporters can predict membership:

$$
\Pr[\text{Predict } R_t] \le \frac{b}{n} + 2^{-\kappa}
$$

with (\kappa = 128), the cryptographic bias term is negligible.

The beacon provides strong **forward secrecy**:

Even if an adversary compromises historical randomness, as long as _one_ recent $\rho_t$ was unpredictable at the time of publication, the VRF seed remains unbiased.

### **(3) Reporter Protocol per Epoch**

For each reporter (r_i):

- Compute VRF selection result

  $y_{r_i} = VRF_{sk_{r_i}}(\rho_t || t)$

- Fetch external data (e.g., price feed) and return a value $v_i$.

- Sign the value bound to the epoch:

  $\sigma_i = \mathrm{Sign}_{sk{r_i}}(v_i || t)$

The system must compute:

$$
P_t = \mathrm{Median}(v_1,\dots,v_n)
$$

and produce a **Halo2 SNARK proof** that this result is correct.

### **(4) SNARK Statement and Halo2 Circuit Structure**

The core NP statement:

$$
\exists {v_i, \sigma_i} \text{ s.t.}
\begin{cases}
\mathrm{VerifySig}(pk_{r_i}, v_i || t, \sigma_i) = 1,\quad \forall i,[3pt]
|{\sigma_i}| \ge f_{\min}, [3pt]
P_t = \mathrm{Median}(v_1,\dots,v_n)
\end{cases}
$$

Halo2 implements this with the following circuit components:

**Signature Verification Chips**

Implements BLS or ECDSA verification inside the SNARK. For ECDSA:

- non-native field arithmetic,
- reconstruction of $kG$ and signature equation

  $k^{-1}(H(m)+rd) \bmod n$

- enforce

  $x(s^{-1}H(m)G + s^{-1}r,pk) = r$

**Committee Membership Enforcement**

Uses a **lookup table** of valid committee public keys $C_t$.
Witness $pk_{r_i}$ must appear exactly once in the table. Also enforces the threshold:

$$
\sum_i m_i \ge f_{\min}, \qquad m_i \in {0,1}
$$

**Sorting and Permutation Constraints**

To compute the median, the circuit must enforce that:

- values are sorted,
- sorted list is a **permutation** of the original.

Halo2 uses **Permutation Arguments** (Plonk-style copy constraints).
Equivalent polynomial identity:

$$
\prod_i (X - v_i) = \prod_i (X - s_i)
$$

Also constraints:

- monotonicity $s_i \le s_{i+1}$
- sum equalities $\sum v_i = \sum s_i$, $\sum v_i^2 = \sum s_i^2$

**Median Extraction**

Once sorted, median is:

$$
P_t = s_{\lfloor n/2 \rfloor}
$$

The proof binds public output $P_t$ to this index.

**Epoch Binding**

Ensures values come from the _current_ epoch:

$$
H(v_i || t) = H_i^{(\text{sig})}
$$

Prevents cross-epoch replay attacks.

**Range Checks**

Ensures values are within reasonable limits:

- avoids pathological extremes,
- prevents overflow,
- prevents long-tail manipulation (e.g., injecting absurdly large garbage values).

### **(5) Final Oracle Packet**

Each epoch produces a packet:

$$
O_t = \langle P_t,, \pi_t,, t \rangle
$$

where:

- $P_t$: the median price,
- $\pi_t$: the Halo2 proof,
- $t$: epoch metadata.

Any blockchain $C_k$ can independently verify:

$$
\mathrm{Verify}_{C_k}(\pi_t) = \text{true}
$$

No committee, aggregator, or relay needs to be trusted.

### **(6) Security Guarantees**

**Unpredictability & Sybil Resistance**

$$
\Pr[\text{predict committee}] \le \frac{b}{n} + 2^{-\kappa}
$$

**Forward Secrecy**

If _any one_ beacon output from the past _k_ rounds was fresh and unbiased at publication:

$$
H_{\text{joint}} \ge k \cdot H_{\min}
$$

Thus, compromising old randomness does not help.

**Multi-Epoch Attack Cost**

$$
\Pr[\text{bias m epochs}] \le m \cdot \left(e^{-\lambda S} + 2^{-\kappa}\right)
$$

where:

- $S$ = honest stake,
- $\lambda$ = minimal adversarial corruption cost per epoch.

The cost of long-term corruption grows **exponentially**.

<details><summary>Code</summary>

```Algorithm
Algorithm 1 Committee Selection via VRF Quantum Entropy
Require: Epoch index t, Quantum entropy pulse ρt, Reporter set R = {ri} with keys (pki, ski), Desired committee size n
Ensure: Selected committee Rt and VRF outputs
1: for all ri ∈ R do
2: yi ← VRFski(ρt∥t)
3: πi ← VRF proof of yi
4: end for
5: Rt ← Select n reporters with smallest yi
6: return Rt and {(ri, yi, πi)} for all ri ∈ Rt

```

</details>

---

## 2. **Post-Quantum Distributed Randomness Beacons (DRB): A High-Throughput, Asynchronous, and Lattice-Based Design**

Distributed Randomness Beacons (DRBs) provide **public, unpredictable, and unbiasable randomness** to decentralized applications, blockchains, multi-party computation (MPC) protocols, and cross-chain interoperability layers. Unlike single-party randomness sources such as VRFs or commit–reveal schemes, DRBs aggregate partial randomness from **multiple mutually distrustful participants**, ensuring that no single node can manipulate the output.

Modern DRB systems include **threshold BLS-based designs** such as drand (Cloudflare, NIST), **Dfinity Threshold Relay**, **RandHound/RandShare**, **Algorand BA**, and recent schemes such as **Rondo (Crypto 2023)** and **Cornucopia (Eurocrypt 2024)**. These constructions rely on **threshold signatures, DKG, and VSS** protocols.

However, nearly all widely deployed DRB schemes suffer from one fatal weakness:

> **They are not post-quantum secure.**
> Threshold BLS, DLog-based DKG, Pedersen VSS, and pairing-based commitments all break in the quantum era.

To build a DRB that is **fully asynchronous, scalable, batched, and post-quantum secure**, we must replace every DLog-based component with lattice-based primitives, and solve the consensus–randomness circular dependency in asynchronous networks.

This article presents the architecture of such a system: a **Post-Quantum Batch Asynchronous Verifiable Secret Sharing (bAVSS-PQ)**–based DRB using **SIS/Module-SIS polynomial commitments**, Merkle-verified shares, and pipelined beacon epochs.

### (1） Why Asynchronous Post-Quantum DRB Is Hard

In an asynchronous setting—where message delays are unbounded and clocks are not synchronized—randomness generation becomes a core challenge. Classical DRBs often assume synchrony or weak synchrony; asynchronous beacons must operate without timing assumptions.

#### Challenges

- **Consensus–randomness circular dependency**

  - Consensus needs randomness (for leader election, view change).
  - Randomness needs consensus (to agree which shares/secrets are valid).
    This feedback loop is a core difficulty.

- **VSS and DKG do not batch**
  Classical AVSS schemes require a full sharing protocol per secret.
  Sharing λ² secrets costs λ² independent VSS executions → catastrophic throughput.

- **Most DRBs depend on DLog hardness**

  - Threshold BLS
  - Feldman/Pedersen VSS
  - DKG (Gennaro et al.)
  - Pairings (KZG commitments)
    All break with Shor’s algorithm → not post-quantum.

- **VDF-based randomness is not asynchronous**
  VDFs require synchronized delays, incompatible with asynchronous protocols.

A future-proof DRB must be:

- **fully asynchronous**
- **quantum-resistant**
- **batched for high throughput**
- **consensus-compatible**

bAVSS-PQ is the enabling primitive.

### (2）Architecture Overview

The system pipeline consists of:

- **Setup:** post-quantum commitments, reliable broadcast, PQ digital signatures
- **Sharing (Dealer):** lattice-based polynomial commitments; batched VSS
- **Verification (Reply):** PQ opening proofs + Merkle proofs
- **Confirmation:** ≥2t+1 approvals → global certificate of correct sharing
- **Reconstruction:** Merkle-verified share exchange to reconstruct secrets
- **Randomness Generation:** aggregate previous secrets into beacon output

This pipeline decouples liveness from synchrony and ensures that randomness is unbiasable unless attackers control ≥ t participants.

#### Setup Phase

Each node initializes:

- a **post-quantum digital signature key pair**
  (e.g. Dilithium, Falcon, SPHINCS+, GeMSS)
- **SIS/Module-SIS polynomial commitment parameters**
  (transparent setup)
- **Reliable Broadcast (RBC)** primitives
- local state for future polynomial commitments and share verification

No trusted setup is required.

#### Sharing Phase — Batched PQ Verifiable Secret Sharing

A dealer wants to share **θ secrets at once**. For each secret $s_{d,j}$, define a degree-t polynomial:

$$
s_j(x) = s_{d,j} + a_{j,1}x + \cdots + a_{j,t}x^t.
$$

Batch all θ polynomials into a matrix (F) and flatten it into a vector:

$$
\mathbf{f} = \mathrm{vec}(F).
$$

**SIS-Based Polynomial Commitment**

A lattice-based commitment is computed:

$$
t = A\mathbf{f} + G\mathbf{r} \pmod q,
$$

where:

- **A** is a public random matrix,
- **G** is a gadget matrix,
- **r** is short lattice randomness,
- binding security reduces to **Module-SIS hardness**.

**Share Distribution**

For each node i, the dealer sends:

- the evaluations $u_{i,j} = s_j(i)$
- opening proofs $\pi_{i,j}$
- a **Merkle proof** linking the share to a broadcasted Merkle root

Merkle trees ensure **horizontal consistency** across the batch.

#### Reply Phase — Detecting Dishonest Dealers

Node i performs:

- **Polynomial commitment opening verification**

$$
\mathrm{PC. Verify}(t_j, i, u_{i,j}, \pi_{i,j})
$$

- **Merkle proof verification**

$$
\mathrm{Merkle.Verify}(root_{L,j}, u_{i,j}, proof_{i,j})
$$

These checks guarantee:

- shares are consistent with the committed polynomials,
- shares match the global broadcast,
- the dealer cannot equivocate.

Node i then signs a message confirming that share j is valid.

#### Confirm Phase — Dealer Proves Honesty

The dealer must collect **2t + 1 replies**.

Why 2t + 1?

- At most t replies can be Byzantine.
- Therefore, at least **t+1 honest approvals** exist.
- This guarantees that the committed batch is **globally valid**.

The dealer packages these responses into a **certificate $C_e$** and broadcasts it.

#### Reconstruction Phase — Merkle-Verified Share Exchange

All nodes:

- Use certificate $C_e$ to authenticate the dealer.
- Broadcast their shares along with Merkle proofs.
- Verify all received shares with Merkle + PC proofs.
- Recover the underlying polynomials and secrets.

**Critical improvement:**
Unlike earlier protocols (e.g., Breeze), **every share must undergo Merkle verification even in reconstruction**, preventing malicious nodes from injecting fake shares during reconstruction.

### Post-Quantum Polynomial Commitments with Batching

SIS-based PC supports:

- **batch commitments** over θ polynomials
- **proof recursion (folding)** giving logarithmic-size proofs
- **Module-SIS security level** (quantum-resistant)

A vector of size $N = \theta(d+1)$ is recursively compressed:

$$
N \rightarrow N/r \rightarrow N/r^2 \rightarrow \cdots
$$

until only a short witness remains.

Thus, proof size grows as:

$$
O(\log(\theta(d+1))),
$$

yielding effective batching where:

- committing 100× more polynomials increases log-size proofs only slightly,
- verification remains fast.

This batching is what allows high-throughput DRB generation.

### Beacon Generation

Once secrets of epoch (e-1) are reconstructed:

$$
R_e = H(s_{e-1,1} \parallel s_{e-1,2} \parallel \cdots )
$$

This output:

- is unbiased unless ≥ t participants are malicious,
- can be verified publicly,
- feeds into the next epoch in a pipelined manner.

<details><summary>Code</summary>

```Algorithm

Algorithm 2 bAVSS-PQ: Post-quantum Batched AVSS Protocol
1: SETUP PHASE
2: All nodes: Run Setup(1λ) → (PC, RBC, (ski, pki)i∈[n]) ▷ Initialize
PC, RBC, and key pairs
3: SHARING PHASE
4: Dealer d: Sample θ t-degree polynomials {sj (·)}j∈[θ] with sj (0) =
sd,j
5: t ← PC.Commit(sj (·)) for j ∈ [θ] ▷ Commit to polynomials
6: for i ∈ [n] do
7:      (ui, πi) ← PC.Open(i) ▷ Generate shares and proofs
8:      Build Merkle tree over {uk,j}k∈[n]
9:      Get rootL,j , proofi,j for j ∈ [θ]
10:     Send ⟨SHARE, t, ui, πi, {proofi,j}j∈[θ]⟩ to node i
11: end for
12: RBC.Broadcast( {rootL,j}j∈[θ]
) ▷ Ensure share consistency
13: REPLY PHASE
14: Node i:
15: On ⟨SHARE, t, ui, πi, {proofi,j}j∈[θ]⟩, RBC.Deliver(rootL,j ):
16: if PC.Verify(t, i, ui, πi) = 1
17: and Merkle.Verify(rootL,j , ui,j , proofi,j ) = 1, ∀j ∈ [θ] then
18:     Send ⟨REPLY,sign(ski, t)⟩ to dealer d
19: end if
20: CONFIRM PHASE
21: Dealer d: Collect 2t + 1 valid signatures into C, output C
22: RECONSTRUCTION PHASE
23: Node i: On ⟨RECON, C, j⟩
24: Select shares {ui,j} from dealer d of C ▷ Use shares from certified
dealer
25: Broadcast ⟨RECON, d, j, ui,j , proofi,j ⟩
26: Upon receiving ⟨RECON, d, j, uk,j , proofk,j ⟩ from node k
27:     if Merkle.Verify(rootd,j , uk,j , proofk,j ) = 1 then
28:         Collect uk,j ▷ Verify and store valid share
29:     end if
30: Interpolate sd,j from t + 1 valid shares uk,j

```

</details>

---

## **3. Quantum Random Number Generators (QRNGs)**

Quantum Random Number Generators (QRNGs) are designed to produce randomness whose unpredictability is rooted in fundamental quantum mechanics. A typical QRNG pipeline consists of the following components:

**Quantum source → Quantum measurement → Raw bit generation → Entropy extraction → Final random numbers.**

Among these stages, the **quantum source** is the most critical. Several physical mechanisms serve as viable quantum entropy sources:

| Source Type                                       | Physical Principle                          | Engineering Difficulty | Commercial Deployment      |
| ------------------------------------------------- | ------------------------------------------- | ---------------------- | -------------------------- |
| **Single-photon polarization measurement**        | 45° polarization measured in X/Z bases      | Medium                 | Widely deployed            |
| **Optical phase noise**                           | Intrinsic phase fluctuations of a laser     | Low                    | Mass-produced, inexpensive |
| **SPDC (Spontaneous Parametric Down-Conversion)** | Random photon-pair generation               | High                   | Mostly research            |
| **Quantum tunneling noise**                       | Random tunneling current through a junction | Medium                 | Chip-level integration     |
| **Electron spin measurements**                    | NV centers, quantum dots                    | High                   | Early research stage       |

The two dominant families today are:

- **Laser phase noise (high-speed QRNGs)**
- **Single-photon polarization measurement (high-security, certifiable QRNGs)**

Because raw quantum measurements are biased (e.g., 60% → “1”), a **strong randomness extractor** is required. Typical extractors include **Trevisan extractors** and **Toeplitz hashing**, sometimes combined with SHA-256/3 as a final compression step. Extractors ensure that:

$$
\text{uniform bits} = \mathrm{Extractor}(\text{raw bits}, \text{public seed}),
$$

and are secure as long as the raw bits contain sufficient min-entropy.

### **(1) Three Classes of QRNGs**

**Trusted-Device QRNGs (most common)**

Both the source and measurement device are assumed to be honest. A typical architecture uses:

- split laser pulses
- interference for phase-noise extraction
- APDs for detection
- extractor for final randomness

These devices are **fast, cheap, and suitable for integration**, but vulnerable to supply-chain attacks.

**Device-Independent QRNGs (DI-QRNG, highest security)**

Security relies solely on **Bell inequality violation**, not on trusting any hardware. A source generates entangled photon pairs, each measured with a randomly chosen basis. If the observed data violate a Bell inequality, then the process _must_ have contained intrinsic quantum randomness.

Bell violation implies a min-entropy lower bound:

$$
H_\infty(\text{raw bits}) \ge N(1-\epsilon),
$$

leading to a guessing probability:

$$
p_{\mathrm{guess}} \le 2^{-H_\infty}.
$$

This is the strongest possible form of randomness certification.
However, DI-QRNGs are slow, expensive, and suitable mainly for national-level cryptographic infrastructure.

**Semi-Trusted / Source-Independent QRNGs (SI-QRNG)**

This category is the **most practical secure architecture** today.
The **source is untrusted** (even adversarial), while **the measurement device is trusted**.

Eve may fully control the optical source—sending arbitrary states, arbitrary photon numbers, high-dimensional states, or deliberately encoding information. Despite this, SI-QRNGs remain secure through the following mechanisms:

### **(2) Architecture of a Modern Silicon-Integrated SI-QRNG**

A state-of-the-art SI-QRNG typically consists of two modules:

#### (A) **Untrusted Optical Source**

- 1550 nm pulsed laser (50 MHz, 200 ps)
- FPGA controlling pulse emission
- Variable Optical Attenuator (VOA)
- Eve can control all properties (photon number, polarization, multi-photon content)

The light is coupled into the **trusted silicon-photonic decoder chip**.

#### (B) **Trusted Measurement Module**

This chip performs all the operations needed for secure randomness extraction:

**Polarization-to-Path Conversion (PSR)**

Silicon photonics cannot reliably maintain polarization.
The **Polarization Splitter-Rotator** converts:

- $|H\rangle \rightarrow$ path 1
- $|V\rangle \rightarrow$ path 2

Thus any incoming qubit $|\psi\rangle = \alpha|H\rangle + \beta|V\rangle$ becomes a **path-encoded qubit**, which silicon photonics can process accurately.

**Passive Basis Choice**

Two **multimode interferometers (MMIs)** function as 50:50 beam splitters. They **passively** select the measurement basis:

- one arm → **Z basis**
- the other → **X basis**

Because the choice is physical and passive (not controlled by electronics), neither Eve nor the device can bias or predict it—a source of genuine quantum unpredictability.

**Polarization Controllers for SU(2) Rotations**

Each measurement arm uses a **polarization controller (PC)** composed of:

- thermo-optic phase shifters (PS1, PS2)
- Mach–Zehnder interferometers (MZIs)

PC2 implements X-basis measurement, mapping:

- $|D\rangle = (|H\rangle + |V\rangle)/\sqrt{2}$ → “correct” detector
- $|A\rangle = (|H\rangle - |V\rangle)/\sqrt{2}$ → “error” detector

PC1 performs Z-basis measurement (H / V).

**Photon Detection (4 SPDs)**

Four single-photon detectors record outcomes:

- Z basis: H and V
- X basis: D (“correct”) and A (“error”)

Z-basis outcomes produce **raw random bits**.
X-basis outcomes are used for **error estimation** (Eve information estimation).

### **(3) Security Mechanics**

**Squashing Model**

Even if Eve sends arbitrary high-dimensional light, the measurement device is theoretically equivalent to a “squashing channel”:

- multi-photon → qubit or vacuum
- high-dimensional → qubit or vacuum

This maintains information-theoretic security.

**Random Sampling + Parameter Estimation**

Let:

- $N_Z$ be the number of Z-basis events
- $N_X$ be the number of X-basis events
- $N = N_X + N_Z$

X basis error rate:

$$
e_b^X = \frac{k}{N_X},
$$

where (k) counts A clicks and half-weighted double-clicks.

Using random sampling theory (Fung et al.):

$$
e_p^Z \le e_b^X + \theta,
$$

where $\theta$ is the statistical deviation.

This bounds Eve’s information about Z-basis outputs.

**Randomness Extraction (Toeplitz Hashing)**

After phase-error estimation, the QRNG computes the number of extractable secure bits:

$$
R = N_Z - N_Z H(e_b^X + \theta) - t_e,
$$

where

- $H(\cdot)$ is the binary Shannon entropy
- $t_e$ controls the extractor failure probability

Higher X-basis error → more Eve information → less extractable randomness. Final randomness has trace distance ≤ $2^{-100}$ or smaller.

---

[Halo2](https://zcash.github.io/halo2)
[Drand](https://github.com/drand/drand)
[Cirq](https://github.com/quantumlib/Cirq)
