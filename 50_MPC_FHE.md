# ETAAcademy-ZKMeme: 50. MPC & FHE

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>50. MPC and FHE</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>MPC_FHE</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Multi-Party Computation (MPC) and Fully Homomorphic Encryption(FHE)

**Multi-Party Computation (MPC)** is a pivotal cryptographic method that allows multiple parties to jointly compute a function over aggregated data while ensuring the privacy of their individual inputs. Key technologies supporting MPC include **Garbled Circuits**, **Secret Sharing**, **Homomorphic Encryption**, and **Differential Privacy**.

In privacy-preserving computation, **Fully Homomorphic Encryption (FHE)**, **Zero-Knowledge Proofs (ZKP)**, and MPC work in synergy to ensure data security and computational integrity. FHE enables operations on encrypted data by lattice-based cryptography which are not easily solvable by quantum computers. This is why FHE is considered quantum-resistant. ZKP verifies the correctness of results without re-executing the computations, and MPC ensures trustworthy distributed processing. Together, these technologies form the backbone of secure data sharing and decentralized applications across industries such as finance, healthcare, and beyond.

---

## 1. Multi-Party Computation

### Centralized vs. Decentralized Multi-Party MPC

**Multi-Party Computation (MPC)** is a cryptographic protocol that allows multiple participants to collaboratively compute a function over their private data without revealing the data itself. The mathematical foundation of MPC involves computing a function $f(x_1, x_2, \dots, x_n)$ over private inputs $x_1, x_2, \dots, x_n$ from different participants to produce an aggregate output $y$, ensuring that each participant’s private data remains confidential and local.

By employing techniques such as **additive secret sharing**, MPC splits private data into multiple shares. These shares are distributed among participants, ensuring no single share contains meaningful information. Reconstruction of the original data requires cooperation from all participants. To verify the correctness of each step in the computation process, **Zero-Knowledge Proofs (ZKP)** are often used. If any participant attempts to cheat, the protocol can detect it, potentially imposing penalties on the malicious actor.

MPC can be broadly classified into two categories:

1. **Centralized MPC**: A central server aggregates data for computation, with safeguards like **Differential Privacy (DP)** to protect individual data.
2. **Decentralized MPC**: Computations are carried out directly among participants, without relying on a central server.

#### Centralized MPC: Leveraging Differential Privacy

In **Centralized MPC**, a trusted central server securely aggregates data from multiple participants. To protect privacy, participants first obfuscate their data locally before sharing it with the server. A key technique in this domain is **Differential Privacy (DP)**, which ensures that individual contributions remain indistinguishable in the aggregated data.

**Differential Privacy (DP)** works by adding random noise to the shared data, making it statistically difficult for adversaries to infer whether specific data points are present in the dataset:

For two adjacent datasets $D$ and $D'$ differing by only one record, a random algorithm $M$ satisfies $\epsilon$-differential privacy if, for all subsets $S$ of the range of $M$:

$\Pr[M(D) \in S] \leq e^\epsilon \Pr[M(D') \in S] + \delta$

Here, $\epsilon > 0$ represents the privacy budget, with smaller values of $\epsilon$ providing stronger privacy guarantees.

An example is **Central Differential Privacy (CDP)**, where noise is added to aggregated parameters before transmission to prevent identification of individual contributions.

#### Decentralized MPC: Secure Computation Without Central Servers

**Decentralized MPC** bypasses the need for a central server, enabling participants to aggregate results through direct communication. This approach dates back to Andrew Yao’s seminal **“Millionaire’s Problem”** in 1982, which addressed how two individuals could determine who is wealthier without disclosing their exact wealth.

Unlike centralized methods, decentralized MPC is inherently more resilient to centralized risks. However, it typically incurs higher communication costs and requires efficient coordination among participants. Despite these challenges, decentralized MPC is particularly valuable in scenarios where centralization poses trust or reliability concerns.

**Key Techniques in Decentralized MPC**

1. **Garbled Circuits (GC)**:  
   Introduced by Andrew Yao in 1986, Garbled Circuits encode computation in Boolean circuits to securely evaluate functions between two participants. This method balances **performance** (efficient computation) and **security** (resistance to semi-honest and malicious adversaries). Modern implementations of GC focus on optimizing execution while maintaining robust security, making it suitable for secure two-party and multi-party computations.

2. **Secret Sharing (SS)**:  
   Secret Sharing splits a secret into multiple shares distributed among participants. No individual share reveals any information about the original secret. Reconstruction of the secret requires a predefined threshold of shares, ensuring data privacy even in distributed settings. This technique is especially useful in privacy-preserving machine learning, where sensitive data remains undisclosed during model training.

3. **Homomorphic Encryption (HE)**:  
   Homomorphic Encryption enables computations on encrypted data without the need for decryption. This ensures data remains secure throughout the computation process. For example, participants can compute encrypted results collaboratively and share the final encrypted output, preserving data privacy while enabling collaborative analysis.

### Comparing Centralized and Decentralized MPC

| **Aspect**            | **Centralized MPC**                             | **Decentralized MPC**                            |
| --------------------- | ----------------------------------------------- | ------------------------------------------------ |
| **Architecture**      | Relies on a central server for data aggregation | Peer-to-peer computation among participants      |
| **Privacy Approach**  | Utilizes techniques like Differential Privacy   | Employs Garbled Circuits, Secret Sharing, and HE |
| **Efficiency**        | Higher computational efficiency                 | Higher communication overhead                    |
| **Trust Assumptions** | Requires trust in the central server            | Eliminates single points of trust                |

---

### Multi-Party Computation (MPC) Protocols: Specialized and General Approaches

In Multi-Party Computation (MPC), **specialized protocols** are optimized for specific functions, such as Private Set Intersection (PSI) or MPC wallets. These protocols offer superior efficiency but are limited in scope. Conversely, **general protocols**, such as Yao’s Garbled Circuits (GC), can compute any discrete function that can be represented as a fixed-size circuit, making them versatile and widely applicable. This article focuses on general protocols and their key attributes.

#### Security in MPC Protocols

The security of an MPC protocol ensures **privacy** and **correctness**:

- **Privacy:** Parties learn only the computed result without revealing their private inputs.
- **Correctness:** All parties obtain accurate outputs.

Additionally, protocols must satisfy:

1. **Input Independence:** Prevent adversaries from influencing honest parties’ inputs.
2. **Output Guarantee:** Ensure honest parties receive their results.
3. **Fairness:** Adversaries can only access their outputs if honest parties receive theirs.

The security of MPC is defined by comparing protocol execution in a **real-world** setting with an **ideal-world** scenario. In the ideal-world model, a trusted third party computes results while maintaining privacy and correctness. Real-world protocols must simulate this behavior, ensuring security even in the presence of adversaries.

#### Adversarial Models in MPC

MPC protocols address various adversarial behaviors:

- **Semi-honest adversaries** follow the protocol but attempt to infer additional information.
- **Malicious adversaries** actively deviate to manipulate results or extract data.
- **Covert adversaries** are malicious but risk detection, with detection probabilities adjustable to the desired level.

Corruption strategies in MPC include:

1. **Static Corruption:** Adversaries corrupt specific parties at the protocol's outset.
2. **Adaptive Corruption:** Adversaries dynamically choose parties to corrupt during execution.
3. **Proactive Security:** Ensures security even when honest parties are temporarily corrupted.

#### Efficient MPC Protocols

Numerous protocols address the trade-offs between communication, computation, and security. Examples include:

- **Yao’s GC**
- **GMW**
- **BGW**
- **IT GC**
- **PSI**
- **SPDZ, PFK, ADMM, etc.**

These protocols often employ techniques like encryption and secret sharing to protect data during computation. Notably, encryption can be viewed as a specialized form of secret sharing. For example, encrypting a message $m$ using a key $k$ ($\text{Enc}_k(m)$) divides $m$ into two components: $k$ and $Enc_k(m)$.

#### **Yao’s Garbled Circuits (GC) Protocol**

Yao’s GC protocol is among the most prominent and efficient general MPC protocols. It transforms function evaluation into an encrypted lookup problem, maintaining a constant number of communication rounds regardless of circuit depth. While its communication complexity is not optimal, its stable round count makes it highly practical.

1. **Function Representation:**  
   The function $F(x, y)$ is expressed as a lookup table $T$, where each entry corresponds to an output value:

   $T_{x,y} = F(x,y)$

2. **Key Generation and Encryption:**

   - Party $P_1$ (holding $x$) generates random keys $k_x$ and $k_y$ for each input.
   - Each table entry is encrypted using these keys:
     $Enc_{k_x, k_y}(T_{x, y})$

3. **Key Distribution:**

   - $P_1$ directly sends $k_x$ to $P_2$.
   - $P_1$ uses an **Oblivious Transfer (OT)** protocol to transmit $k_y$ to $P_2$.

4. **Decryption and Output:**  
   $P_2$ decrypts the corresponding table entry to obtain the output:

   $F(x, y) = Dec_{k_x, k_y}(Enc_{k_x, k_y}(T_{x, y}))$

This ensures $P_2$ only decrypts the relevant output while other information remains hidden.

**Enhancements: Point-and-Permute Technique**

A crucial challenge in GC is how $P_2$ determines the correct table row to decrypt. The **Point-and-Permute** technique addresses this efficiently by embedding pointers within the encryption keys. This eliminates the need for error detection methods like appended zero strings.

1. **Pointer Construction:**  
   The last bits of $k_x$ and $k_y$ serve as pointers to specific table rows:

   $\text{Pointer}_x = k_x[\lceil \log|X| \rceil], \quad \text{Pointer}_y = k_y[\lceil \log|Y| \rceil]$

2. **Pointer Combination:**  
   These pointers are concatenated to form the final row identifier:

   $\text{Pointer} = \text{Pointer}_x \parallel \text{Pointer}_y$

3. **Key Preservation:**  
   Pointers are appended to keys, maintaining their original length for security:
   $k'_x = k_x \parallel \text{Pointer}_x, \quad k'_y = k_y \parallel \text{Pointer}_y$

**Practical Implementation in Circuit Evaluation**

Each gate in a circuit corresponds to a small lookup table (LUT) with four entries. For a gate $G$ with inputs $w_i, w_j$ and output $w_t$, the table $T_G$ contains:

$T_G =
\left[
Enc_{k^i_0, k^j_0}(k^t_0), Enc_{k_0^i, k_1^j}(k_0^t),
Enc_{k_1^i, k_0^j}(k_0^t), Enc_{k_1^i, k_1^j}(k_1^t)
\right]$

This ensures that $P_2$ decrypts only the active path of the circuit without learning other information.

---

#### **GMW Protocol**

The **GMW (Goldreich-Micali-Wigderson) protocol** is a foundational method in secure multiparty computation (MPC). Unlike Yao's Garbled Circuits (GC), which typically handle values using encrypted wire labels, the GMW protocol employs additive secret sharing to manage the circuit's values. While Yao's GC is inherently a two-party protocol (though it can be extended to multiparty settings with additional techniques), the GMW protocol is naturally multiparty-friendly and offers superior scalability for such computations.

Participants in the GMW protocol hold additive shares of wire values rather than encrypted labels as in Yao's GC. The key steps of the GMW protocol include generating random wire labels, constructing encrypted circuits, transmitting input labels via Oblivious Transfer (OT), evaluating the circuit using encrypted truth tables and pointer bits, and decoding the final output through decryption tables. By leveraging hashing and XOR operations, the protocol ensures secure computation while effectively concealing the plaintext values of inputs and outputs.

1. **Wire Label Generation**  
   For each wire $w_i$, two random labels are generated:

   $w_0^i = (k_0^i, p_0^i), \quad w_1^i = (k_1^i, p_1^i)$

   Here, $p_0^i = 1 - p_1^i$ represents the pointer bit.

2. **Encrypted Circuit Generation**  
   Each gate $G_i$ in the circuit, assuming a 2-input Boolean gate, has input labels:

   $w_0^a = (k_0^a, p_0^a), \; w_1^a = (k_1^a, p_1^a) \quad \text{and} \quad w_0^b = (k_0^b, p_0^b), \; w_1^b = (k_1^b, p_1^b)$

   The output labels are $w_0^c = (k_0^c, p_0^c)$ and $w_1^c = (k_1^c, p_1^c)$.  
   For each input combination $v_a, v_b \in \{0, 1\}$, the protocol computes an entry in the encrypted truth table:

   $e_{v_a, v_b} = H(k_{v_a}^a || k_{v_b}^b || i) \oplus w_{g(v_a, v_b)}^c$

   where $H$ is a cryptographic hash function, and $g(v_a, v_b)$ is the gate's output function.

3. **Output Decoding Table**  
   For each output wire $w_i$, the decoding table maps the output value $v \in \{0, 1\}$:
   $e_v = H(k_v^i || \text{"out"} || j) \oplus v$

---

#### **BGW Protocol**

The **BGW (Ben-Or, Goldwasser, Wigderson) protocol** is another essential MPC protocol designed for secure multiparty computation. It is well-suited for arithmetic circuits and uses **Shamir’s secret sharing** as its foundation. The protocol ensures security as long as $2t + 1 \leq n$, where $t$ is the threshold for tolerated adversarial participants, and $n$ is the total number of participants.

In the BGW protocol, each participant holds shares of secret values represented as points on a polynomial. The protocol supports both addition and multiplication gates, with addition being computed locally and multiplication requiring an additional degree reduction step. The protocol guarantees correctness and privacy even with malicious participants.

1. **Shamir’s Secret Sharing**  
   A secret $v$ is represented using a polynomial $p(x)$ such that $p(0) = v$. Each participant receives a share $p(i)$. With a threshold $t$, any $t$ shares reveal no information about the secret.

2. **Addition Gates**  
   For input wires $\alpha$ and $\beta$ with shared values $[v_\alpha]$ and $[v_\beta]$, the output is $v_\alpha + v_\beta$. Each participant computes locally:

   $p_\gamma(i) = p_\alpha(i) + p_\beta(i)$

   resulting in a new polynomial $p_\gamma(x) = p_\alpha(x) + p_\beta(x)$ that represents the sum.

3. **Multiplication Gates**  
   For input wires $\alpha$ and $\beta$ with shared values $[v_\alpha]$ and $[v_\beta]$, the product $v_\alpha \times v_\beta$ is computed. After locally calculating the product, the resulting polynomial $q(x) = p_\alpha(x) \cdot p_\beta(x)$ has degree $2t$. A degree reduction step is performed to adjust $q(x)$ to degree $t$:
   $q(0) = \sum_{i=1}^{2t+1} \lambda_i q(i)$
   where $\lambda_i$ are Lagrange coefficients, and $q(i)$ are the shares held by participants. This ensures the correct sharing of the product.

---

#### Constant-Round Multiparty Computation: The BMR Protocol

In the field of secure multiparty computation (MPC), early protocols such as Yao’s Garbled Circuits (GC), Goldreich-Micali-Wigderson (GMW), Ben-Or-Goldwasser-Wigderson (BGW), and Chaum-Crepeau-Damgård (CCD) all have computation rounds proportional to the circuit depth $C$. However, the **Beaver-Micali-Rogaway (BMR) protocol** broke new ground by enabling constant-round MPC that is independent of circuit depth. BMR achieves this while tolerating corruption of any number of parties, provided fewer than $n$ participants are malicious.

The BMR protocol extends Yao’s GC to multiparty settings by generating encrypted circuits in a distributed manner. This ensures the computation can be performed in constant rounds regardless of circuit depth. Each participant generates wire labels and partial encrypted gate entries, avoiding any single participant or subset knowing the complete circuit structure. To maintain security, **“flip bits”** are introduced to prevent evaluators from inferring plaintext values based on their partial labels.

For a gate $G_i$, the encrypted table entry for inputs $w_a$ and $w_b$ that produces output $w_c$ is calculated as:

$e_{v_a, v_b} = w_c \oplus f_c \oplus \bigoplus_{j=1}^n \left( F(i, w_a \oplus f_a) \oplus F(i, w_b \oplus f_b) \right)$

Here, $F$ is a pseudorandom function (PRF), $\oplus$ represents XOR, and $f_a, f_b, f_c$ are flip bits derived by XORing the shared flip bit values from all parties. The encrypted rows for each gate are organized using XOR operations to produce the final encrypted circuit. By leveraging these techniques, BMR achieves both efficiency and security in multiparty computation, regardless of circuit depth.

---

#### Information-Theoretic Garbled Circuits (IT GC)

The **Information-Theoretic Garbled Circuit (IT GC)** is a secure computation paradigm implemented in the OT-hybrid model. Unlike traditional approaches such as Yao’s GC or the GMW protocol, IT GC encrypts the circuit wires rather than directly encrypting participants' data. This reduces communication costs and replaces cryptographic primitives with lightweight XOR operations and bit-mixing techniques. IT GC encrypts gates incrementally, eliminating the need for conventional encrypted tables.

The **GESS (Gate-Enabling Secret Sharing)** scheme, which underpins IT GC, uses shared wire labels to encode secrets. It allows valid label combinations to reconstruct a Boolean gate’s output while preserving input confidentiality. IT GC's gate-by-gate encryption makes it particularly efficient for wide circuits, such as those used in machine learning.

1. **Secret Sharing and Reconstruction**: Gate outputs are shared using input wire labels, enabling valid label combinations to reconstruct output values without revealing additional information.
2. **No Encryption Tables**: Unlike Yao’s GC, GESS does not rely on precomputed encryption tables, simplifying the circuit’s construction.
3. **Incremental Gate Encryption**: Gates are encrypted step by step without requiring plaintext reconstruction during computation.
4. **Workflow**: For a Boolean gate $G$, output labels $s_0, s_1$ are shared secrets. Input labels $(sh_{1,0}, sh_{1,1})$ and $(sh_{2,0}, sh_{2,1})$ are designed such that any valid combination of input labels uniquely reconstructs $G(i, j)$ while keeping all other information private.

---

#### Private Set Intersection (PSI)

Customized protocols are designed for specific use cases, significantly enhancing efficiency in scenarios requiring high performance, such as **Private Set Intersection (PSI)**. These specialized protocols trade generality for performance, often requiring additional security proofs and implementation complexity. PSI protocols enable parties to compute the intersection of their input sets without exposing any other information.

Modern PSI schemes, such as the **PSSZ protocol**, leverage techniques like Oblivious Pseudo-Random Functions (OPRF) and hashing. In PSSZ, sets are mapped into fixed-size buckets using a 3-way Cuckoo hashing scheme:

1. **Hashing Input Elements**: Party $P_2$ hashes its set into buckets and computes OPRF outputs for multiple hash iterations.
2. **Candidate Computation**: Party $P_1$ computes a candidate PRF output set, shuffles it, and sends it to $P_2$.
3. **Intersection Identification**: $P_2$ identifies intersecting elements by comparing PRF outputs.

The PSSZ protocol relies on the pseudorandomness of PRFs and carefully chosen hash parameters to minimize collisions, ensuring correctness and security under a semi-honest model. Despite requiring multiple communication rounds, it significantly improves efficiency compared to general-purpose MPC frameworks.

---

#### Advances

Recent advancements in MPC incorporate techniques like OT extension and visual cryptography to improve efficiency. Customized PSI protocols often outperform general-purpose MPC in specific applications. For example, **Google’s secure aggregation protocol** uses secret sharing to aggregate gradients without revealing individual contributions, enabling efficient and privacy-preserving machine learning.

Other innovations include decentralized MPC frameworks like SPDZ, which use Shamir’s $k-out-of-n$ secret sharing under a semi-honest model. Privacy-preserving approaches such as **PFK-Means** transmit shared gradients instead of raw encrypted data, while methods like **ADMM (Alternating Direction Method of Multipliers)** optimize communication patterns among data owners.

While domain-specific protocols continue to push boundaries, cases where customized schemes vastly outperform optimized general-purpose MPC remain rare. This reflects the maturity and robustness of modern MPC frameworks in diverse applications.

<details><summary><b> Code</b></summary>

<details><summary><b> Python </b></summary>

```Python
# This module defines multiplication on SharedScalars, using the SPDZ
# algorithm for multiplication [1].
#
# [1] https://bristolcrypto.blogspot.com/2016/10/what-is-spdz-part-2-circuit-evaluation.html

# Small hack:
#
# We can't import the SharedScalar class in this module as that would
# create a circular dependency.
#
# However, we'd obviously still like to be able to construct new
# SharedScalars here when doing arithmetic. To be able to do so,
# we can use `type(sh)` to get access to the SharedScalar class &
# constructor.

from .finite_ring import mod, rand_element
from .secret_sharing import n_to_shares
from random import choice

def mult_2sh(sh1, sh2):
    '''Implements multiplication on two SharedScalars.'''
    # Make sure that these two SharedScalars are compatible
    sh1._assert_can_operate(sh2)

    # Generate a random multiplication triple (public)
    a, b = rand_element(sh1.Q), rand_element(sh1.Q)
    c = mod(a * b, sh1.Q)

    # Share the triple across all machines
    # (It'd be nicer to use the higher-level PrivateScalar.share() here,
    # but we don't have access to PrivateScalar in this module.)
    machines = list(sh1.owners)
    shared_a = type(sh1)(n_to_shares(a, machines, sh1.Q), sh1.Q)
    shared_b = type(sh1)(n_to_shares(b, machines, sh1.Q), sh1.Q)
    shared_c = type(sh1)(n_to_shares(c, machines, sh1.Q), sh1.Q)

    # Compute sh1 - a, sh2 - b (shared)
    shared_sh1_m_a = sh1 - shared_a
    shared_sh2_m_b = sh2 - shared_b

    # Reconstruct sh1 - a, sh2 - b (public)
    rand_machine = choice(machines)
    sh1_m_a = shared_sh1_m_a.reconstruct(rand_machine).value
    sh2_m_b = shared_sh2_m_b.reconstruct(rand_machine).value

    # Magic! Compute each machine's share of the product
    shared_prod = shared_c + (sh1_m_a * shared_b) + (sh2_m_b * shared_a) + (sh1_m_a * sh2_m_b)
    return shared_prod

def mult_sh_pub(sh, pub):
    '''Implements multiplication on a SharedScalar and a public integer.'''
    # To do the multiplication, we multiply the integer with all shares
    prod_shares = [share * pub for share in sh.shares]
    return type(sh)(prod_shares, Q=sh.Q)

```

</details>

<details><summary><b> C++ </b></summary>

```C++

#include "OT/BaseOT.h"
#include "Tools/random.h"
#include "Tools/benchmarking.h"
#include "Tools/Bundle.h"
#include "Processor/OnlineOptions.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <pthread.h>

#if defined(__linux__) and defined(__x86_64__)
#include <cpuid.h>
#endif

extern "C" {
#ifndef NO_AVX_OT
#include "SimpleOT/ot_sender.h"
#include "SimpleOT/ot_receiver.h"
#endif
#include "SimplestOT_C/ref10/ot_sender.h"
#include "SimplestOT_C/ref10/ot_receiver.h"
}

using namespace std;

const char* role_to_str(OT_ROLE role)
{
    if (role == RECEIVER)
        return "RECEIVER";
    if (role == SENDER)
        return "SENDER";
    return "BOTH";
}

OT_ROLE INV_ROLE(OT_ROLE role)
{
    if (role == RECEIVER)
        return SENDER;
    if (role == SENDER)
        return RECEIVER;
    else
        return BOTH;
}

void send_if_ot_sender(TwoPartyPlayer* P, vector<octetStream>& os, OT_ROLE role)
{
    if (role == SENDER)
    {
        P->send(os[0]);
    }
    else if (role == RECEIVER)
    {
        P->receive(os[1]);
    }
    else
    {
        // both sender + receiver
        P->send_receive_player(os);
    }
}

void send_if_ot_receiver(TwoPartyPlayer* P, vector<octetStream>& os, OT_ROLE role)
{
    if (role == RECEIVER)
    {
        P->send(os[0]);
    }
    else if (role == SENDER)
    {
        P->receive(os[1]);
    }
    else
    {
        // both
        P->send_receive_player(os);
    }
}

// type-dependent redirection

void sender_genS(ref10_SENDER* s, unsigned char* S_pack)
{
    ref10_sender_genS(s, S_pack);
}

void sender_keygen(ref10_SENDER* s, unsigned char* Rs_pack,
        unsigned char (*keys)[4][HASHBYTES])
{
    ref10_sender_keygen(s, Rs_pack, keys);
}

void receiver_maketable(ref10_RECEIVER* r)
{
    ref10_receiver_maketable(r);
}

void receiver_procS(ref10_RECEIVER* r)
{
    ref10_receiver_procS(r);
}

void receiver_rsgen(ref10_RECEIVER* r, unsigned char* Rs_pack,
        unsigned char* cs)
{
    ref10_receiver_rsgen(r, Rs_pack, cs);
}

void receiver_keygen(ref10_RECEIVER* r, unsigned char (*keys)[HASHBYTES])
{
    ref10_receiver_keygen(r, keys);
}

void BaseOT::allocate()
{
    for (int i = 0; i < nOT; i++)
    {
        sender_inputs[i][0] = BitVector(8 * AES_BLK_SIZE);
        sender_inputs[i][1] = BitVector(8 * AES_BLK_SIZE);
        receiver_outputs[i] = BitVector(8 * AES_BLK_SIZE);
    }
}

int BaseOT::avx = -1;

bool BaseOT::use_avx()
{
    if (avx == -1)
    {
        avx = cpu_has_avx(true);
#if defined(__linux__) and defined(__x86_64__)
        int info[4];
        __cpuid(0x80000003, info[0], info[1], info[2], info[3]);
        string str((char*) info, 16);
        if (OnlineOptions::singleton.has_option("debug_cpu"))
            cerr << "CPU: " << str << endl;
        if (str.find("Gold 63") != string::npos)
            avx = 0;
#endif
    }

    return avx;
}

void BaseOT::exec_base(bool new_receiver_inputs)
{
#ifndef NO_AVX_OT
    if (use_avx())
        exec_base<SIMPLEOT_SENDER, SIMPLEOT_RECEIVER>(new_receiver_inputs);
    else
#endif
        exec_base<ref10_SENDER, ref10_RECEIVER>(new_receiver_inputs);
}

// See https://eprint.iacr.org/2015/267.pdf
template<class T, class U>
void BaseOT::exec_base(bool new_receiver_inputs)
{
    int i, j, k;
    size_t len;
    PRNG G;
    G.ReSeed();
    vector<octetStream> os(2);
    T sender;
    U receiver;

    unsigned char S_pack[ PACKBYTES ];
    unsigned char Rs_pack[ 2 ][ 4 * PACKBYTES ];
    unsigned char sender_keys[ 2 ][ 4 ][ HASHBYTES ];
    unsigned char receiver_keys[ 4 ][ HASHBYTES ];
    unsigned char cs[ 4 ];

    if (ot_role & SENDER)
    {
        // Sample a and compute A=g^a
        sender_genS(&sender, S_pack);
        // Send A
        os[0].store_bytes(S_pack, sizeof(S_pack));
    }
    send_if_ot_sender(P, os, ot_role);

    if (ot_role & RECEIVER)
    {
        // Receive A
        len = sizeof(receiver.S_pack);
        os[1].get_bytes((octet*) receiver.S_pack, len);
        if (len != HASHBYTES)
        {
            cerr << "Received invalid length in base OT\n";
            exit(1);
        }

        // Process A
        receiver_procS(&receiver);
        receiver_maketable(&receiver);
    }

    os[0].reset_write_head();
    allocate();

    for (i = 0; i < nOT; i += 4)
    {
        if (ot_role & RECEIVER)
        {
            for (j = 0; j < 4 and (i + j) < nOT; j++)
            {
                // Process choice bits
                if (new_receiver_inputs)
                    receiver_inputs[i + j] = G.get_uchar()&1;
                cs[j] = receiver_inputs[i + j].get();
            }
            // Compute B
            receiver_rsgen(&receiver, Rs_pack[0], cs);
            // Send B
            os[0].store_bytes(Rs_pack[0], sizeof(Rs_pack[0]));
            // Compute k_R
            receiver_keygen(&receiver, receiver_keys);

            // Copy keys to receiver_outputs
            for (j = 0; j < 4 and (i + j) < nOT; j++)
            {
                for (k = 0; k < AES_BLK_SIZE; k++)
                {
                    receiver_outputs[i + j].set_byte(k, receiver_keys[j][k]);
                }
            }

#ifdef BASE_OT_DEBUG
            for (j = 0; j < 4; j++)
                for (k = 0; k < AES_BLK_SIZE; k++)
                {
                    printf("%4d-th receiver key:", i+j);
                    for (k = 0; k < HASHBYTES; k++) printf("%.2X", receiver_keys[j][k]);
                    printf("\n");
                }

            printf("\n");
#endif
        }
    }

    send_if_ot_receiver(P, os, ot_role);

    for (i = 0; i < nOT; i += 4)
    {
        if (ot_role & SENDER)
        {
            // Receive B
            len = sizeof(Rs_pack[1]);
            os[1].get_bytes((octet*) Rs_pack[1], len);
            if (len != sizeof(Rs_pack[1]))
            {
                cerr << "Received invalid length in base OT\n";
                exit(1);
            }
            // Compute k_0 and k_1
            sender_keygen(&sender, Rs_pack[1], sender_keys);

            // Copy 128 bits of keys to sender_inputs
            for (j = 0; j < 4 and (i + j) < nOT; j++)
            {
                for (k = 0; k < AES_BLK_SIZE; k++)
                {
                    sender_inputs[i + j][0].set_byte(k, sender_keys[0][j][k]);
                    sender_inputs[i + j][1].set_byte(k, sender_keys[1][j][k]);
                }
            }
        }
        #ifdef BASE_OT_DEBUG
        for (j = 0; j < 4; j++)
        {
            if (ot_role & SENDER)
            {
                printf("%4d-th sender keys:", i+j);
                for (k = 0; k < HASHBYTES; k++) printf("%.2X", sender_keys[0][j][k]);
                printf(" ");
                for (k = 0; k < HASHBYTES; k++) printf("%.2X", sender_keys[1][j][k]);
                printf("\n");
            }
        }

        printf("\n");
        #endif
    }

    if (ot_role & SENDER)
        for (int i = 0; i < nOT; i++)
        {
            if(sender_inputs.at(i).at(0) == sender_inputs.at(i).at(1))
            {
                string error = "Sender outputs are the same at " + to_string(i)
                        + ": " + sender_inputs[i][0].str();
#ifdef NO_AVX_OT
                error += "This is a known problem with some Xeon CPUs. ";
                error += "We would appreciate if you report the output of "
                        "'cat /proc/cpuinfo | grep name'. ";
                error += "Try compiling with 'AVX_SIMPLEOT = 0' in CONFIG.mine";
#endif
                throw runtime_error(error);
            }
        }

    // Hash with counter to avoid collisions
    for (int i = 0; i < nOT; i++)
    {
        if (ot_role & RECEIVER)
            hash_with_id(receiver_outputs.at(i), i);
        if (ot_role & SENDER)
            for (int j = 0; j < 2; j++)
                hash_with_id(sender_inputs.at(i).at(j), i);
    }

    if (ot_role & SENDER)
        for (int i = 0; i < nOT; i++)
            assert(sender_inputs.at(i).at(0) != sender_inputs.at(i).at(1));

    // Set PRG seeds
    set_seeds();

    if (ot_role & SENDER)
        for (int i = 0; i < nOT; i++)
            assert(sender_inputs.at(i).at(0) != sender_inputs.at(i).at(1));
}

void BaseOT::hash_with_id(BitVector& bits, long id)
{
    assert(bits.size_bytes() >= AES_BLK_SIZE);
    Hash hash;
    hash.update(bits.get_ptr(), bits.size_bytes());
    hash.update(&id, sizeof(id));
    hash.final(bits.get_ptr(), bits.size_bytes());
}

void BaseOT::set_seeds()
{
    for (int i = 0; i < nOT; i++)
    {
        // Set PRG seeds
        if (ot_role & SENDER)
        {
            G_sender[i][0].SetSeed(sender_inputs[i][0].get_ptr());
            G_sender[i][1].SetSeed(sender_inputs[i][1].get_ptr());
        }
        if (ot_role & RECEIVER)
        {
            G_receiver[i].SetSeed(receiver_outputs[i].get_ptr());
        }
    }
    extend_length();
}

void BaseOT::extend_length()
{
    for (int i = 0; i < nOT; i++)
    {
        if (ot_role & SENDER)
        {
            sender_inputs[i][0].randomize(G_sender[i][0]);
            sender_inputs[i][1].randomize(G_sender[i][1]);
        }
        if (ot_role & RECEIVER)
        {
            receiver_outputs[i].randomize(G_receiver[i]);
        }
    }
}


void BaseOT::check()
{
    vector<octetStream> os(2);
    BitVector tmp_vector(8 * AES_BLK_SIZE);


    for (int i = 0; i < nOT; i++)
    {
        if (ot_role == SENDER)
        {
            // send both inputs over
            sender_inputs[i][0].pack(os[0]);
            sender_inputs[i][1].pack(os[0]);
            P->send(os[0]);
        }
        else if (ot_role == RECEIVER)
        {
            P->receive(os[1]);
        }
        else
        {
            // both sender + receiver
            sender_inputs[i][0].pack(os[0]);
            sender_inputs[i][1].pack(os[0]);
            P->send_receive_player(os);
        }
        if (ot_role & RECEIVER)
        {
            tmp_vector.unpack(os[1]);

            if (receiver_inputs[i] == 1)
            {
                tmp_vector.unpack(os[1]);
            }
            if (!tmp_vector.equals(receiver_outputs[i]))
            {
                cerr << "Incorrect OT\n";
                exit(1);
            }
        }
        os[0].reset_write_head();
        os[1].reset_write_head();
    }
}


void FakeOT::exec_base(bool new_receiver_inputs)
{
    insecure("base OTs");
    PRNG G;
    G.ReSeed();
    vector<octetStream> os(2);
    vector<BitVector> bv(2, 128);

    allocate();

    if ((ot_role & RECEIVER) && new_receiver_inputs)
    {
        for (int i = 0; i < nOT; i++)
            // Generate my receiver inputs
            receiver_inputs[i] = G.get_uchar()&1;
    }

    if (ot_role & SENDER)
        for (int i = 0; i < nOT; i++)
            for (int j = 0; j < 2; j++)
            {
                sender_inputs[i][j].randomize(G);
                sender_inputs[i][j].pack(os[0]);
            }

    send_if_ot_sender(P, os, ot_role);

    if (ot_role & RECEIVER)
        for (int i = 0; i < nOT; i++)
        {
            for (int j = 0; j < 2; j++)
                bv[j].unpack(os[1]);
            receiver_outputs[i] = bv[receiver_inputs[i].get()];
        }

    set_seeds();
}


```

</details>

</details>

---

## 2. Fully Homomorphic Encryption (FHE)

Fully Homomorphic Encryption (FHE) is a powerful cryptographic technology that enables mathematical operations to be performed directly on encrypted data without the need for decryption. The result of these operations remains in encrypted form, and only individuals with the decryption key can access the final plaintext.

Homomorphic encryption relies on a mathematical property called homomorphism, where a function between two algebraic structures preserves their operations, such as those found in groups, rings, or vector spaces. This concept is pivotal in cryptographic schemes, enabling computations on encrypted data while guaranteeing correctness and decryptability of results.

1. **Homomorphism in Groups**

A function $f: \mathbb{G}_1 \to \mathbb{G}_2$ is a homomorphism between groups $(\mathbb{G}_1, \cdot)$ and $(\mathbb{G}_2, \oplus)$ if:
$$f(x \cdot y) = f(x) \oplus f(y), \quad \forall x, y \in \mathbb{G}_1.$$  
Here, the operation in $\mathbb{G}_2$ applies to the image of $f(x)$ and $f(y)$.

2. **Homomorphism in Cryptography**

- **RSA Encryption:** Homomorphism in RSA encryption allows operations on ciphertexts to yield the same result as performing the operation on plaintexts and then encrypting the outcome:
  $$E(m_1) \cdot E(m_2) = (m_1^e \cdot m_2^e) \mod n = E(m_1 \cdot m_2).$$
- **KZG Commitments:** The Kate-Zaverucha-Goldberg (KZG) commitment scheme demonstrates additive homomorphism, enabling operations like:
  $$\text{cm}(p_1(x) + p_2(x)) = \text{cm}(p_1(x)) \oplus \text{cm}(p_2(x)).$$

---

Fully Homomorphic Encryption (FHE) is an advanced cryptographic technique based on lattice-based cryptography that supports direct mathematical operations, such as addition and multiplication, on ciphertexts while maintaining their encrypted state. Unlike traditional encryption methods, FHE enables computations on encrypted data without the need for decryption, ensuring privacy throughout the process. Several FHE implementation libraries are available today, such as OpenFHE, Microsoft SEAL, and Λ∘λ. By combining Partial Homomorphic Encryption (PHE) schemes with "recryption" technology, FHE can handle operations on circuits of unlimited depth and support computations with various data types, ranging from integers to real numbers. The security of FHE is based on NP-hard lattice problems, such as Ring-Learning with Errors (RLWE), which are resistant to quantum computing attacks. This makes FHE a vital tool in the fields of privacy protection and secure encrypted computation, with applications in areas like private queries and secure data analysis.

1. **Lattice-Based Security of FHE**

FHE relies on lattice-based cryptography, particularly the **Short Vector Problem (SVP)** or **Ring-Learning with Errors (RLWE)** problems. These problems are NP-hard and are not easily solvable by quantum computers, which is why FHE is considered quantum-resistant. Lattices are generated through integer linear combinations of basis vectors, and **Ideal Lattices** correspond to ideals within polynomial rings (such as operations on even-numbered sets). These ideals inherit the addition and multiplication operations of the underlying ring, providing the foundation for FHE's security.

2. **Noise in FHE**

FHE ciphertexts typically contain a small amount of noise. As long as the noise remains below a certain threshold, decryption will be successful. If the noise grows too large due to repeated operations, decryption may fail. This is why noise management is crucial in FHE, especially in operations involving multiple rounds of computation.

3. **Homomorphic Encryption Variants**

- **Partial Homomorphic Encryption (PHE):** PHE supports a single operation, either addition or multiplication, making it simple to implement but with limited functionality.
- **Somewhat Homomorphic Encryption (SHE):** SHE supports a limited number of addition and multiplication operations, making it suitable for basic polynomial computations. However, each operation increases the noise in the ciphertext, which limits the number of operations before the noise becomes too large to decrypt properly. The homomorphic properties for addition and multiplication in SHE are expressed as follows:

  $E(a + b) = E(a) \oplus E(b), \quad E(a \times b) = E(a) \cdot E(b)$

- **Fully Homomorphic Encryption (FHE):** FHE, in contrast, supports arbitrarily complex computations, making it the most powerful encryption scheme. However, FHE is also the most difficult to implement and optimize, requiring significant computational resources to achieve.

4. **Recryption Mechanism**

A critical innovation in FHE is the "recryption" mechanism. If a recryption algorithm, denoted as $\text{Recrypt}$, can be introduced to reduce the noise in a ciphertext, it allows the ciphertext $E(m)$ to be transformed into a new ciphertext $E'(m)$ with lower noise. This mechanism enables the transition from SHE to FHE, extending the potential depth of computations that can be performed.

5. **Circuit Depth and Bootstrapping**

SHE can only handle circuits of limited depth, meaning it supports only a finite number of addition or multiplication operations before the noise becomes too large to manage. However, by modifying the decryption circuit’s multiplication depth and introducing a "bootstrapping" technique, it is possible to upgrade SHE to FHE. Bootstrapping enables FHE to handle operations on circuits of unlimited depth, making it suitable for a broader range of computations.

6. **Common FHE Schemes**

- **BFV and BGV:** These schemes are suited for integer computations, providing a foundation for performing homomorphic operations on integer data.
- **CKKS:** The CKKS scheme is optimized for real number computations, enabling homomorphic encryption to be applied to more complex numerical data.
- **DM and CGGI:** These schemes are designed for Boolean circuits and can implement arbitrary functions via look-up tables, broadening the scope of FHE’s applicability.

---

### Techniques and Challenges in Multiplication and Error Management

Homomorphic Encryption (HE) allows operations such as addition and multiplication to be performed on encrypted data without decryption, offering significant privacy-preserving benefits. While addition is relatively simple in HE, multiplication introduces additional complexity. This article explores the techniques used to handle homomorphic encryption's challenges, particularly the exponential growth of errors during multiplication, and the approaches to manage these errors effectively.

#### 1. **Homomorphic Addition**

In homomorphic encryption, addition is straightforward. When adding two ciphertexts, the result is simply the sum of the corresponding noise terms and message values, as shown by the equation:

$$
\langle c_1 + c_2, k \rangle = 2(e_1 + e_2) + m_1 + m_2 \pmod{p}
$$

Here, $c_1$ and $c_2$ are ciphertexts, $k$ is the key, $m_1$ and $m_2$ are the original messages, and $e_1$ and $e_2$ are the noise (error) terms. Since addition is performed modulo $p$, the operation is relatively simple and does not lead to exponential growth of errors. However, homomorphic multiplication is significantly more complicated.

#### 2. **Homomorphic Multiplication**

Multiplication in homomorphic encryption is more complex because vectors do not have a natural multiplication operation. To simulate multiplication, the **outer product** technique is typically used. The outer product involves multiplying each element of one vector by each element of another, resulting in a new vector:

$$
a \otimes b = a_1 b_1 + a_2 b_1 + \dots + a_n b_1 + a_1 b_2 + \dots + a_n b_n
$$

In encryption, the multiplication of ciphertexts is realized using the mathematical formula:

$$
\langle a \otimes b, c \otimes d \rangle = \langle a, c \rangle \times \langle b, d \rangle
$$

This approach expresses the multiplication of ciphertexts in terms of inner products of encrypted components. However, the outer product method introduces a problem: the size of the ciphertext and key grows quadratically, leading to an increase in computational and storage overhead.

#### 3. **Relinearization Technique**

To address the issue of quadratic growth in ciphertext and key size, the **relinearization** technique is used. This technique introduces a "relinearization key," which consists of encrypted fragments of the key. The relinearization key can be viewed as "noisy" encryptions of $k \otimes k$ under the key $k$, where $k$ is the encryption key. The key allows the evaluator to compute inner products in the encrypted domain, such as:

$$
\langle c_1 \otimes c_2, k \otimes k \rangle
$$

The result is a value of the form $k_i \times k_j \times 2^d + 2e$, where $i, j$ are indices of the key components, $d$ is an exponent, and $2^d < p$. This allows the key components to be combined during the multiplication of ciphertexts, thereby reducing the quadratic growth in ciphertext and key sizes. While this technique introduces some additional error, the errors are relatively small and manageable, enabling the computation of $m_1 \times m_2$ in encrypted form. However, this method does not entirely solve the issue of rapidly growing errors, and further techniques are needed to control error expansion.

#### 4. **Modulus Switching for Error Reduction**

**Modulus switching** is a technique used to reduce errors by decreasing the modulus of the ciphertext, thus mitigating the error growth. This method involves dividing the ciphertext by 2 (mod $p$), shrinking the modulus, and performing integer floor operations to reduce the absolute size of the errors. Each multiplication operation causes a constant increase in error, preventing exponential error growth and stabilizing the ciphertext for subsequent computations. This method is particularly important for preserving the correctness and precision of ciphertexts in complex operations such as exponentiation.

1. **Initial Encryption Formula**:

   $\langle c, k \rangle = m + 2e$

   where $m \in \{0, 1\}$, $c$ is the ciphertext, $k$ is the key, and $e$ is the error term.

2. **Dividing by 2 (mod $p$)**:

   $\langle c', k \rangle = m \cdot \frac{p}{2} + e \pmod{p}$

   where $c'$ is the new ciphertext after dividing by 2, and $e$ is the new error term.

3. **Multiplying by $\frac{q}{p}$** (integer floor):

   $\langle c'', k \rangle = m \cdot \frac{q}{2} + e' + e_2 \pmod{q}$

   where $e' = e \cdot \frac{p}{q}$, and $e_2$ is a small rounding error.

4. **Multiplying by 2 (mod $q$)**:

   $\langle c''', k \rangle = m \cdot 1 + 2e' + 2e_2 \pmod{q}$

   where $c'''$ is the final transformed ciphertext with error $2e' + 2e_2$, which is smaller than the original error.

This process reduces the magnitude of the error and prevents it from growing exponentially, ensuring the ciphertext remains stable during complex computations.

#### 5. **Matrix Representation in FHE**

Another approach to homomorphic encryption involves representing ciphertexts as matrices rather than vectors. The key $k$ acts as a "close feature vector," and when multiplied with the matrix, the resulting encrypted result tends to be very close to zero or the key itself. This allows direct addition and multiplication operations on the ciphertext, but multiplication still leads to quadratic error growth. To manage this, techniques such as matrix transformations can be employed to control the growth of errors. Although this method faces efficiency challenges, it holds promise for future applications in privacy-preserving computations.

1. **Addition**:

   $k \times (C_1 + C_2) = (m_1 + m_2) \times k + (e_1 + e_2)$

2. **Multiplication**:

   $k \times C_1 \times C_2 = (m_1 \times k + e_1) \times C_2 = m_1 \times m_2 \times k + m_1 \times e_2 + e_1 \times C_2$

   In this calculation, the first term $m_1 \times m_2 \times k$ represents the expected result, while the second and third terms $m_1 \times e_2$ and $e_1 \times C_2$ represent error terms. Multiplication causes the error terms to increase, especially with squared growth.

---

### BFV, CKKS, and BGV

Fully Homomorphic Encryption (FHE) enables computations on encrypted data without decryption, providing privacy-preserving solutions for various applications. Among the most prominent FHE schemes are **Brakerski/Fan-Vercauteren (BFV)**, **Cheon-Kim-Kim-Song (CKKS)**, and **Brakerski-Gentry-Vaikuntanathan (BGV)**. Each scheme is designed to meet different needs and has distinct features, optimizing for various types of data and computational requirements.

#### 1. **Brakerski/Fan-Vercauteren (BFV) Scheme**

The **BFV scheme** is designed for integer arithmetic in homomorphic encryption. It is particularly suitable for applications that require precise integer calculations, such as statistical analysis, database queries, and counting tasks. The scheme supports both addition and multiplication of encrypted integers while maintaining data accuracy.

- **Security Assumption**: The BFV scheme is based on the **Ring-Learning With Errors (Ring-LWE)** problem, which provides its security foundation.
- **Plaintext Representation**: Plaintext data is represented as polynomials over integer rings, and the ciphertext is formed by encrypting these polynomials.
- **Operations**: The scheme supports arbitrary additions and a fixed number of multiplications. The computational complexity increases as the ciphertext size grows, making the scheme less efficient for large computations.
- **Applications**: BFV is ideal for scenarios that demand exact calculations on encrypted integer data, such as in statistical computations and tasks requiring accurate counts.

#### 2. **Cheon-Kim-Kim-Song (CKKS) Scheme**

The **CKKS scheme** is designed for homomorphic encryption of real and floating-point numbers. It supports **approximate computations**, making it particularly suited for applications that do not require exact precision but can tolerate some error. The CKKS scheme is widely used in fields like machine learning, where real-valued or complex computations are common, and it also provides the **Bootstrapping** functionality to restore the ciphertext's usability after computations.

- **Plaintext Representation**: In CKKS, plaintext data is encoded as vectors of complex numbers, which are then encrypted into polynomials.
- **Approximation**: This scheme allows for efficient computations on real and complex numbers, but the results may suffer from rounding errors, leading to a loss of precision.
- **Bootstrapping**: A key feature of CKKS is its ability to use bootstrapping to refresh the ciphertexts, ensuring the continued usefulness of encrypted data after repeated computations.
- **Applications**: CKKS is particularly suited for large-scale applications such as **machine learning**, **signal processing**, and any task that involves statistical analysis and prediction models on encrypted floating-point data.

#### 3. **Brakerski-Gentry-Vaikuntanathan (BGV) Scheme**

The **BGV scheme** is another integer homomorphic encryption scheme that is tailored for **multi-party computation (MPC)** scenarios. It supports both addition and multiplication operations and allows for parameter adjustments to balance computation efficiency and ciphertext size. BGV is considered more flexible than BFV, as it offers multiple key management strategies to suit varying use cases.

- **Security Assumption**: Like BFV, BGV is based on **Ring-LWE** but also incorporates **modulus switching** techniques for efficiency.
- **Optimizations**: The BGV scheme supports a variety of optimizations, such as **batch encoding**, which allows for efficient vectorized operations and better handling of large datasets.
- **Applications**: BGV is ideal for multi-party computations, **database queries**, and analysis tasks that require secure interactions between multiple parties. It is particularly useful in scenarios that involve complex computations with encrypted data, where several participants are involved in the process.

### **Comparison of BFV, CKKS, and BGV**

Here’s a comparison of the three FHE schemes based on their characteristics, applications, and optimizations:

| Feature                      | **BFV**                        | **CKKS**                                        | **BGV**                                   |
| ---------------------------- | ------------------------------ | ----------------------------------------------- | ----------------------------------------- |
| **Supported Data Types**     | Integers                       | Real numbers and complex numbers                | Integers                                  |
| **Precision of Computation** | Exact computations             | Approximate computations                        | Exact computations                        |
| **Typical Applications**     | Statistical analysis, counting | Machine learning, signal processing             | Multi-party computation, database queries |
| **Technical Optimizations**  | Ring-LWE, modulus switching    | Complex number encoding, approximate encryption | Ring-LWE, batch encoding                  |

<details><summary><b> Code</b></summary>

<details><summary><b> Python </b></summary>

```python

# Initial Settings
from openfhe import *
import os

# import openfhe.PKESchemeFeature as Feature


def main():
    # Sample Program: Step 1: Set CryptoContext
    parameters = CCParamsBGVRNS()
    parameters.SetPlaintextModulus(65537)
    parameters.SetMultiplicativeDepth(2)

    crypto_context = GenCryptoContext(parameters)
    # Enable features that you wish to use
    crypto_context.Enable(PKESchemeFeature.PKE)
    crypto_context.Enable(PKESchemeFeature.KEYSWITCH)
    crypto_context.Enable(PKESchemeFeature.LEVELEDSHE)

    # Sample Program: Step 2: Key Generation

    # Generate a public/private key pair
    key_pair = crypto_context.KeyGen()

    # Generate the relinearization key
    crypto_context.EvalMultKeyGen(key_pair.secretKey)

    # Generate the rotation evaluation keys
    crypto_context.EvalRotateKeyGen(key_pair.secretKey, [1, 2, -1, -2])

    # Sample Program: Step 3: Encryption

    # First plaintext vector is encoded
    vector_of_ints1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    plaintext1 = crypto_context.MakePackedPlaintext(vector_of_ints1)

    # Second plaintext vector is encoded
    vector_of_ints2 = [3, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    plaintext2 = crypto_context.MakePackedPlaintext(vector_of_ints2)

    # Third plaintext vector is encoded
    vector_of_ints3 = [1, 2, 5, 2, 5, 6, 7, 8, 9, 10, 11, 12]
    plaintext3 = crypto_context.MakePackedPlaintext(vector_of_ints3)

    # The encoded vectors are encrypted
    ciphertext1 = crypto_context.Encrypt(key_pair.publicKey, plaintext1)
    ciphertext2 = crypto_context.Encrypt(key_pair.publicKey, plaintext2)
    ciphertext3 = crypto_context.Encrypt(key_pair.publicKey, plaintext3)

    #  Sample Program: Step 4: Evaluation

    # Homomorphic additions
    ciphertext_add12 = crypto_context.EvalAdd(ciphertext1, ciphertext2)
    ciphertext_add_result = crypto_context.EvalAdd(ciphertext_add12, ciphertext3)

    # Homomorphic Multiplication
    ciphertext_mult12 = crypto_context.EvalMult(ciphertext1, ciphertext2)
    ciphertext_mult_result = crypto_context.EvalMult(ciphertext_mult12, ciphertext3)

    # Homomorphic Rotations
    ciphertext_rot1 = crypto_context.EvalRotate(ciphertext1, 1)
    ciphertext_rot2 = crypto_context.EvalRotate(ciphertext1, 2)
    ciphertext_rot3 = crypto_context.EvalRotate(ciphertext1, -1)
    ciphertext_rot4 = crypto_context.EvalRotate(ciphertext1, -2)

    # Sample Program: Step 5: Decryption

    # Decrypt the result of additions
    plaintext_add_result = crypto_context.Decrypt(
        ciphertext_add_result, key_pair.secretKey
    )

    # Decrypt the result of multiplications
    plaintext_mult_result = crypto_context.Decrypt(
        ciphertext_mult_result, key_pair.secretKey
    )

    # Decrypt the result of rotations
    plaintextRot1 = crypto_context.Decrypt(ciphertext_rot1, key_pair.secretKey)
    plaintextRot2 = crypto_context.Decrypt(ciphertext_rot2, key_pair.secretKey)
    plaintextRot3 = crypto_context.Decrypt(ciphertext_rot3, key_pair.secretKey)
    plaintextRot4 = crypto_context.Decrypt(ciphertext_rot4, key_pair.secretKey)

    plaintextRot1.SetLength(len(vector_of_ints1))
    plaintextRot2.SetLength(len(vector_of_ints1))
    plaintextRot3.SetLength(len(vector_of_ints1))
    plaintextRot4.SetLength(len(vector_of_ints1))

    print("Plaintext #1: " + str(plaintext1))
    print("Plaintext #2: " + str(plaintext2))
    print("Plaintext #3: " + str(plaintext3))

    # Output Results
    print("\nResults of homomorphic computations")
    print("#1 + #2 + #3 = " + str(plaintext_add_result))
    print("#1 * #2 * #3 = " + str(plaintext_mult_result))
    print("Left rotation of #1 by 1 = " + str(plaintextRot1))
    print("Left rotation of #1 by 2 = " + str(plaintextRot2))
    print("Right rotation of #1 by 1 = " + str(plaintextRot3))
    print("Right rotation of #1 by 2 = " + str(plaintextRot4))


if __name__ == "__main__":
    main()

```

</details>

<details><summary><b> C++ </b></summary>

```c++

//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

/*
  Example of a computation circuit of depth 3
  BGVrns demo for a homomorphic multiplication of depth 6 and three different approaches for depth-3 multiplications
 */

#define PROFILE

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>

#include "openfhe.h"

using namespace lbcrypto;

int main(int argc, char* argv[]) {
    ////////////////////////////////////////////////////////////
    // Set-up of parameters
    ////////////////////////////////////////////////////////////

    std::cout << "\nThis code demonstrates the use of the BGVrns scheme for "
                 "homomorphic multiplication. "
              << std::endl;
    std::cout << "This code shows how to auto-generate parameters during run-time "
                 "based on desired plaintext moduli and security levels. "
              << std::endl;
    std::cout << "In this demonstration we use three input plaintexts and show "
                 "how to both add them together and multiply them together.\n"
              << std::endl;

    // benchmarking variables
    TimeVar t;
    double processingTime(0.0);

    // Crypto Parameters
    // # of evalMults = 3 (first 3) is used to support the multiplication of 7
    // ciphertexts, i.e., ceiling{log2{7}} Max depth is set to 3 (second 3) to
    // generate homomorphic evaluation multiplication keys for s^2 and s^3
    CCParams<CryptoContextBGVRNS> parameters;
    parameters.SetMultiplicativeDepth(3);
    parameters.SetPlaintextModulus(536903681);
    parameters.SetMaxRelinSkDeg(3);

    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
    // enable features that you wish to use
    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);
    cryptoContext->Enable(ADVANCEDSHE);

    std::cout << "\np = " << cryptoContext->GetCryptoParameters()->GetPlaintextModulus() << std::endl;
    std::cout << "n = " << cryptoContext->GetCryptoParameters()->GetElementParams()->GetCyclotomicOrder() / 2
              << std::endl;
    std::cout << "log2 q = "
              << log2(cryptoContext->GetCryptoParameters()->GetElementParams()->GetModulus().ConvertToDouble())
              << std::endl;

    // Initialize Public Key Containers
    KeyPair<DCRTPoly> keyPair;

    // Perform Key Generation Operation

    std::cout << "\nRunning key generation (used for source data)..." << std::endl;

    TIC(t);

    keyPair = cryptoContext->KeyGen();

    processingTime = TOC(t);
    std::cout << "Key generation time: " << processingTime << "ms" << std::endl;

    if (!keyPair.good()) {
        std::cout << "Key generation failed!" << std::endl;
        exit(1);
    }

    std::cout << "Running key generation for homomorphic multiplication "
                 "evaluation keys..."
              << std::endl;

    TIC(t);

    cryptoContext->EvalMultKeysGen(keyPair.secretKey);

    processingTime = TOC(t);
    std::cout << "Key generation time for homomorphic multiplication evaluation keys: " << processingTime << "ms"
              << std::endl;

    ////////////////////////////////////////////////////////////
    // Encode source data
    ////////////////////////////////////////////////////////////

    std::vector<int64_t> vectorOfInts1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext1               = cryptoContext->MakePackedPlaintext(vectorOfInts1);

    std::vector<int64_t> vectorOfInts2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext2               = cryptoContext->MakePackedPlaintext(vectorOfInts2);

    std::vector<int64_t> vectorOfInts3 = {2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext3               = cryptoContext->MakePackedPlaintext(vectorOfInts3);

    std::vector<int64_t> vectorOfInts4 = {2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext4               = cryptoContext->MakePackedPlaintext(vectorOfInts4);

    std::vector<int64_t> vectorOfInts5 = {3, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext5               = cryptoContext->MakePackedPlaintext(vectorOfInts5);

    std::vector<int64_t> vectorOfInts6 = {3, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext6               = cryptoContext->MakePackedPlaintext(vectorOfInts6);

    std::vector<int64_t> vectorOfInts7 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext7               = cryptoContext->MakePackedPlaintext(vectorOfInts7);

    std::cout << "\nOriginal Plaintext #1: \n";
    std::cout << plaintext1 << std::endl;

    std::cout << "\nOriginal Plaintext #2: \n";
    std::cout << plaintext2 << std::endl;

    std::cout << "\nOriginal Plaintext #3: \n";
    std::cout << plaintext3 << std::endl;

    std::cout << "\nOriginal Plaintext #4: \n";
    std::cout << plaintext4 << std::endl;

    std::cout << "\nOriginal Plaintext #5: \n";
    std::cout << plaintext5 << std::endl;

    std::cout << "\nOriginal Plaintext #6: \n";
    std::cout << plaintext6 << std::endl;

    std::cout << "\nOriginal Plaintext #7: \n";
    std::cout << plaintext7 << std::endl;

    ////////////////////////////////////////////////////////////
    // Encryption
    ////////////////////////////////////////////////////////////

    std::cout << "\nRunning encryption of all plaintexts... ";

    std::vector<Ciphertext<DCRTPoly>> ciphertexts;

    TIC(t);

    ciphertexts.push_back(cryptoContext->Encrypt(keyPair.publicKey, plaintext1));
    ciphertexts.push_back(cryptoContext->Encrypt(keyPair.publicKey, plaintext2));
    ciphertexts.push_back(cryptoContext->Encrypt(keyPair.publicKey, plaintext3));
    ciphertexts.push_back(cryptoContext->Encrypt(keyPair.publicKey, plaintext4));
    ciphertexts.push_back(cryptoContext->Encrypt(keyPair.publicKey, plaintext5));
    ciphertexts.push_back(cryptoContext->Encrypt(keyPair.publicKey, plaintext6));
    ciphertexts.push_back(cryptoContext->Encrypt(keyPair.publicKey, plaintext7));

    processingTime = TOC(t);

    std::cout << "Completed\n";

    std::cout << "\nAverage encryption time: " << processingTime / 7 << "ms" << std::endl;

    ////////////////////////////////////////////////////////////
    // Homomorphic multiplication of 2 ciphertexts
    ////////////////////////////////////////////////////////////

    TIC(t);

    auto ciphertextMult = cryptoContext->EvalMult(ciphertexts[0], ciphertexts[1]);

    processingTime = TOC(t);
    std::cout << "\nTotal time of multiplying 2 ciphertexts using EvalMult w/ "
                 "relinearization: "
              << processingTime << "ms" << std::endl;

    Plaintext plaintextDecMult;

    TIC(t);

    cryptoContext->Decrypt(keyPair.secretKey, ciphertextMult, &plaintextDecMult);

    processingTime = TOC(t);
    std::cout << "\nDecryption time: " << processingTime << "ms" << std::endl;

    plaintextDecMult->SetLength(plaintext1->GetLength());

    std::cout << "\nResult of homomorphic multiplication of ciphertexts #1 and #2: \n";
    std::cout << plaintextDecMult << std::endl;

    ////////////////////////////////////////////////////////////
    // Homomorphic multiplication of 7 ciphertexts
    ////////////////////////////////////////////////////////////

    std::cout << "\nRunning a binary-tree multiplication of 7 ciphertexts...";

    TIC(t);

    auto ciphertextMult7 = cryptoContext->EvalMultMany(ciphertexts);

    processingTime = TOC(t);

    std::cout << "Completed\n";

    std::cout << "\nTotal time of multiplying 7 ciphertexts using EvalMultMany: " << processingTime << "ms"
              << std::endl;

    Plaintext plaintextDecMult7;

    cryptoContext->Decrypt(keyPair.secretKey, ciphertextMult7, &plaintextDecMult7);

    plaintextDecMult7->SetLength(plaintext1->GetLength());

    std::cout << "\nResult of 6 homomorphic multiplications: \n";
    std::cout << plaintextDecMult7 << std::endl;

    ////////////////////////////////////////////////////////////
    // Homomorphic multiplication of 3 ciphertexts where relinearization is done
    // at the end
    ////////////////////////////////////////////////////////////

    std::cout << "\nRunning a depth-3 multiplication w/o relinearization until the "
                 "very end...";

    TIC(t);

    auto ciphertextMult12 = cryptoContext->EvalMultNoRelin(ciphertexts[0], ciphertexts[1]);
    cryptoContext->ModReduceInPlace(ciphertextMult12);

    processingTime = TOC(t);

    std::cout << "Completed\n";

    std::cout << "Time of multiplying 2 ciphertexts w/o relinearization: " << processingTime << "ms" << std::endl;

    auto ciphertexts2 = cryptoContext->ModReduce(ciphertexts[2]);

    auto ciphertextMult123 = cryptoContext->EvalMultAndRelinearize(ciphertextMult12, ciphertexts2);

    Plaintext plaintextDecMult123;

    cryptoContext->Decrypt(keyPair.secretKey, ciphertextMult123, &plaintextDecMult123);

    plaintextDecMult123->SetLength(plaintext1->GetLength());

    std::cout << "\nResult of 2 homomorphic multiplications: \n";
    std::cout << plaintextDecMult123 << std::endl;

    ////////////////////////////////////////////////////////////
    // Homomorphic multiplication of 3 ciphertexts w/o any relinearization
    ////////////////////////////////////////////////////////////

    std::cout << "\nRunning a depth-3 multiplication w/o relinearization...";

    ciphertextMult12 = cryptoContext->EvalMultNoRelin(ciphertexts[0], ciphertexts[1]);
    cryptoContext->ModReduceInPlace(ciphertextMult12);
    ciphertextMult123 = cryptoContext->EvalMultNoRelin(ciphertextMult12, ciphertexts2);
    ciphertextMult123 = cryptoContext->ModReduce(ciphertextMult123);
    std::cout << "Completed\n";

    cryptoContext->Decrypt(keyPair.secretKey, ciphertextMult123, &plaintextDecMult123);

    plaintextDecMult123->SetLength(plaintext1->GetLength());

    std::cout << "\nResult of 3 homomorphic multiplications: \n";
    std::cout << plaintextDecMult123 << std::endl;

    ////////////////////////////////////////////////////////////
    // Homomorphic multiplication of 3 ciphertexts w/ relinearization after each
    // multiplication
    ////////////////////////////////////////////////////////////

    std::cout << "\nRunning a depth-3 multiplication w/ relinearization after each "
                 "multiplication...";

    TIC(t);

    ciphertextMult12 = cryptoContext->EvalMult(ciphertexts[0], ciphertexts[1]);
    cryptoContext->ModReduceInPlace(ciphertextMult12);

    processingTime = TOC(t);
    std::cout << "Completed\n";
    std::cout << "Time for multiplying 3 ciphertexts w/ relinearization: " << processingTime << "ms" << std::endl;

    ciphertextMult123 = cryptoContext->EvalMult(ciphertextMult12, ciphertexts2);

    cryptoContext->Decrypt(keyPair.secretKey, ciphertextMult123, &plaintextDecMult123);

    plaintextDecMult123->SetLength(plaintext1->GetLength());

    std::cout << "\nResult of 3 homomorphic multiplications: \n";
    std::cout << plaintextDecMult123 << std::endl;

    return 0;
}

```

</details>

</details>

[MPC](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/MPC)

[FHE](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/FHE)

<div  align="center"> 
<img src="images/50_MPC_FHE.gif" width="50%" />
</div>
