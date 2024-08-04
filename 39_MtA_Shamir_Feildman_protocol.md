# ETAAcademy-ZKMeme: 39. MtA, Shamir & Feildman protocol

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>39. MtA, Shamir & Feildman protocol</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>MtA_Shamir_Feildman_protocol</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

## Share Conversion and Secret Sharing Protocols

### Share Conversion Protocol

The **Multiplicative to Additive (MtA)** is fundamental to threshold signature schemes, transforming multiplicative shares into additive ones. This facilitates the combination of individual contributions to form a final signature. Except Paillier cryptosystem, it can also be implemented using other transfer protocols, each having its own pros and cons. For example, Paillier introduces RSA, so it isn't purely elliptic curve-based, but its shorter ciphertexts are suitable for share conversion protocols that involve transmitting large amounts of data.

**Specific Steps:**

1. **Paillier Encryption and Range Proof:**

   - The Paillier public key $pk$ is $n$, and the private key $sk$ is $p$, $q$ or $\lambda$, where $n := p \cdot q$ and $\lambda := \text{lcm}(p-1, q-1)$.
   - Participant $P_1$ selects a random number $a \in \mathbb{Z}\_n$ and computes the ciphertext $c_1 := Enc_{pk}(a)$ using Paillier homomorphic encryption. Then, $P_1$ sends the ciphertext $c_1$ along with the range proof zk-RangeProof $\{a | a < q^3, c_1 = Enc_{pk}(a)\}$.

2. **Homomorphic Computation by Second Participant:**

   - Participant $P_2$ receives $c_1$ and the range proof zk-RangeProof $\{a \mid a < q^3, c_1 = \text{Enc}_{pk}(a)\}$. After verifying the range proof, $P_2$ uses $c_1$ for computation, selecting two random numbers $b$ and $\beta \in \mathbb{Z}\_n$.
   - $P_2$ homomorphically computes $c_2 := (b ⨂ c_1）⊕ Enc_{pk} = Enc_{pk}(ab + \beta' \mod n), B ：= g^b$, and $B := g^b$. $P_2$ sets its additive share as $\beta$, with $\beta = -\beta' \mod n$. $P_2$ sends $c_2$ and the range proof zk-RangeProof $\{b, \beta' \mid b < q^3, \beta' < q^7, c_2 = (b ⨂ c_1) \oplus \text{Enc}_{pk}(\beta'), B = g^b\}$.

3. **Decryption and Additive Share by First Participant:**
   - Participant $P_1$ receives $c_2$ and the range proof, verifies the proof, and decrypts $c_2$ to obtain its additive share $\alpha$, where $\alpha = ab + \beta \mod n$.

By the end of the protocol, both participants hold additive shares, $\alpha$ and $\beta$, without revealing their original inputs, $a$ and $b$, to each other. The magic lies in the relationship: $\alpha + \beta = ab$. This transformation from multiplication to addition is crucial, as it enables participants to securely combine their shares additively to produce the final signature.

### Secret Sharing Protocol

Secret sharing protocols evolved from centralized approaches like Shamir's and Feldman's schemes to more distributed models. While the former relied on a single trusted party, the latter introduced mechanisms where participants collaboratively manage the secret, eliminating the need for a central authority.

**Shamir's Secret Sharing**

1. The common secret is $sk \in [1, n-1]$. A trusted third party selects $t-1$ random numbers $a_1, ..., a_{t-1} \in [1, n-1]$ to construct a polynomial $p(x) = sk + a_1 x^1 + ... + a_{t-1} x^{t-1}$.
2. The party computes the polynomial values $p(i) := sk + a_1 i^1 + ... + a_{t-1} i^{t-1}$ for $i$ participants, and securely sends each value to the corresponding participant (encrypted with their public key), e.g., sending $p_1$ to participant 1, $p_2$ to participant 2.
3. In secret reconstruction, $t$ participants broadcast their polynomial values $p(1), ..., p(t)$, allowing the secret $sk$ to be reconstructed using solving linear equations or Lagrange interpolation. These values can be used to directly compute $t$ signature shares, combining them to get the final signature.

**Feldman's Verifiable Secret Sharing (VSS)**

While Shamir's secret sharing provides a foundation for secure secret distribution, it lacks built-in error detection. Feldman's scheme addresses this by incorporating a verification mechanism that helps identify potential errors during the reconstruction process.

1. Compute Feldman verification tuples: $A_0 := sk \cdot G, A_1 := a_1 \cdot G, ..., A_{t-1} := a\_{t-1} \cdot G$, and broadcast these discrete logarithms.
2. Verification: Participant $j$ receives the polynomial value $p(j) := sk + a_1 j^1 + ... + a_{t-1} j^{t-1}$, and performs Feldman verification, ensuring $p(j)\cdot G = (sk + a_1j^1 +...+a_{t-1}j^{t-1}) \cdot G = A_0 + j^1 \cdot A_1 +...+j^t\cdot A_{t-1} = \sum\nolimits_{j=0}^{t-1}i^jA_t$.

**Centralized Secret Refreshing Protocol**

To mitigate the risk of compromised shares, secret refreshing techniques are employed. By periodically updating share values, the overall system's resilience against potential attacks is enhanced.

1. Refreshing Method 1 constructs a new polynomial without a constant term: Choose new $t-1$ random numbers to form a new polynomial $p'(x)$, ensuring the constant term remains the secret $sk$. Construct Lagrange redundancy $p'(i)$, compute Feldman verification tuples $A_i'$, and verify $p’(j)\cdot G = \sum\nolimits_{j=0}^{t-1}i^jA’_t$. Participants then sum the polynomial values $p(j) + p'(j)$.
2. Refreshing Method 2 involves the trusted third party constructing a new polynomial $f(x) = p(j) + p'(j)$, equivalent to Method 1 but the new polynomial is directly sent to participants.

Method 1 preserves the original data by updating it, while Method 2 requires creating new data, effectively discarding the old data.

**Distributed Verifiable Secret Sharing (DVSS) Protocol**

In contrast to centralized approaches, distributed secret sharing involves each participant creating their own polynomial and sharing relevant information with others to collectively manage the secret.

1. Each of the $t$ participants chooses a random number $u_1, u_2, ..., u_t$, computes their public keys $U_1 := u_1 \cdot G, U_2 := u_2 \cdot G, ..., U_t := u_t \cdot G$, and broadcasts commitments $(KGC_1, KGD_1) = \text{Com}(U_1), (KGC_2, KGD_2) = \text{Com}(U_2), ..., (KGC_t, KGD_t) = \text{Com}(U_t)$. After verifying commitments, the common public key is $PK = U_1 + U_2 + ... + U_t$.
2. Each participant constructs their own $t-1$ degree polynomial $p_i(x) = u_i + x \cdot a_1^i + ... + x^{t-1} a_{t-1}^i$. They compute Lagrange redundancy, store their polynomial $p_i(x)$, and securely send values to others, performing Feldman verification and computing partial private keys. For example, participant $P_1$ stores $p_1$, sends $p_1(2), p_1(3)$ to $P_2, P_3$, computes and broadcasts Feldman verification tuples, and performs Feldman verification to calculate partial private key $x_1 := \sum*{i=1}^{t} p_i(1)$. They then broadcast partial public keys $X_1 := PK + (\sum*{i=1}^{t} p_i(a_1^i + ... + a_{t-1}^i)) \cdot G$.

**Distributed Partial Private Key Refreshing**

Similar to the centralized refreshing approach, a new polynomial with a zero constant term, denoted as $p'_i(x)$, is constructed for each participant. The updated partial private key $x_1 ：= \sum\nolimits_{i=1}^{t}p_i(1) + \sum\nolimits_{i=1}^{t}p_i'(1)= \sum\nolimits_{i=1}^{t}p_i(u_i + a_1^i +...+a_{t-1}^i) + \sum\nolimits_{i=1}^{t}p_i'(u_i + a_1^i +...+a_{t-1}^i)= sk +\sum\nolimits_{i=1}^{t}p_i(a_1^i +...+a_{t-1}^i+a_1'^i +...+a_{t-1}'^i)$ and public key $X_1 ：= PK +(\sum\nolimits_{i=1}^{t}p_i(a_1^i +...+a_{t-1}^i+a_1'^i +...+a_{t-1}'^i))\cdot G$, are calculated by summing the evaluations of both the original and new polynomials at specific points, incorporating the contributions from all participants.
