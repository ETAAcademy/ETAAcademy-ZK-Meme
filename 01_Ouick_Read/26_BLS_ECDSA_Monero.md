# ETAAcademy-ZKMeme: 26. BLS, ECDSA, Monero

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>26. BLS_ECDSA_Monero</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>BLS_ECDSA_Monero</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

## Demystifying Digital Signatures: BLS, ECDSA, and Monero

This guide explores three major signature families: BLS, ECDSA, and Monero, each of which caters to specific needs within the cryptographic landscape.

### BLS Signatures: Efficient for Blockchain Consensus

- **Strengths:** Ideal for blockchain consensus voting due to efficient batch verification capabilities.
- **Algorithm:** Relies on bilinear pairings, where $e(a · G_1, b · G_2) = e(G_1, G_2)^{ab}$. Here, $G_1$ and $G_2$ are cyclic groups of prime order p, with a hash function H and a secret random number $x ∈ Z_p$( private key), while the public key is $v = g^x_2 ∈ G_2$.
- **Signature and Verification:**
  - Signature: $σ =  H(M)^x$, where M is the message.
  - Verification: $e(σ, g_2) = e(H(m)^x, g_2) = e(H(m), g_2)^x = e(H(m), g^x_2) = e(H(m), v)$, ensures the signature originated from the private key corresponding to the public key v.
- **Security Enhancements:**
  - Adding random bits (b ∈ {0, 1}) or random numbers (r ∈ $Z_p$) to the message (M') reduces reduction loss (potential vulnerability) to 2 bits and 1 bit respectively.
    - M’ = {M|b} 或 M’ = {M|b}
    - $σ = H(M')^x$
    - $e(σ, g_2) = e(H(m'), v)$

**BLS Batch Verification:**

- **Two methods:**
  1. **Aggregate Signatures:** Combines multiple signatures into one for verification using bilinear maps. Successful verification of this single signature confirms the validity of all individual signatures. The number of bilinear maps needed is n + 1 (where n is the total number of signatures).
  - $e(σ_{1,2,...,n, g_2}) = e(σ_1·σ_2·...·σ_n, g_2) = e(σ_1, g_2)·e(σ_2, g_2)·...·e(σ_n, g_2)$
  2. **Linear Combination:** Generates n signatures for the same message/block. These signatures and public keys are combined with random numbers to obtain a public key (V) and a signature (U), which requires only two bilinear maps for verification.
  - $e(U, g_2) = e(H(M),V)$

**BLS Signature Derivatives:**

- BBRO, ZSS, and BB signatures are all derived from BLS and offer additional features.
- BBRO and ZSS use one private key (a) and two public keys ($g_1$ and h). They differ in the number of signatures generated and verification processes.
  - BBRO: $σ = (σ_1, σ_2) = (h^aHash(m)^r, g^r)$, $e(σ_1, g) = e(h^a · Hash(m)^r, g) = e(h^a, g)e(Hash(m)^r, g) = e(g_1, h)e(Hash(m), σ_2)$
  - ZSS: $σ = h^{1/(a + Hash(m))}$, $e(σ, g_1g^{Hash(m)}) = e(h^{1/(a+Hash(m)), g^ag^{Hash(m)}}) = e(h,g)$
- BB signatures employ two private keys (a and β) and three public keys ($g_1$, $g_2$, and h).
  - $σ = (σ_1, σ_2) = (r, h^{1/(a+βm+r)})$
  - $e(σ_2, g_1g^m_2g^{σ_1}) = e(h^{1/(a+βm+r), g^ag^{mβ}g^r}) = e(h,g)$

### ECDSA Signatures: The Workhorse for Transactions

- **Widely used:** ECDSA (Elliptic Curve Digital Signature Algorithm) and its variants like Schnorr, EdDSA, are prevalent for transaction signatures.
- **Process:** Involves four steps - commitment, challenge, response, and verification.
- **Schnorr Signature Example:**
  - Private key (u ∈ [1, n - 1]), public key (Y = u · G), message (m), random number (k ∈ [1, n - 1]).
  - Commitment (R = k · G).
  - Challenge (e = hash(m, R)).
  - Response (z = k + e · u mod n).
  - Signature (R, z).
  - Verification: z · G = (k + e · u) · G = R + e · Y. (Leakage risk: Using the same k for different messages can reveal the private key).

**EdDSA:**

- Improves security by deriving k from a hash function using message and private key parameters, ensuring a unique k for each message.
  - $k = sha256(hi_{bit}, m) \mod n$

**ECDSA on Elliptic Curve Groups:**

- Similar to Schnorr, but the public key PK is derived from the low bit of private key d:
  - $(low_{bit}, hi_{bit}) := sha512(d)$
  - $y = low_{bit}$, $PK = y·G$,
- Potential private key leakage is mitigated by calculating k using a hash of the private key and message, $k = sha256(hi_{bit}, m) \mod n$

### Monero Signatures: Prioritizing Anonymity

- **Key Distinction:** Utilizes a "key image" ($I = x · H_p(P)$) that hides the public key (P), achieving anonymity.
- **Process:**
  - Message (m) comprises five UTXOs (unspent transaction outputs), including user's UTXO.
  - Five random numbers ($q_i$) and four additional random numbers ($w_i$).
  - Commitment: Creates five points ($L_i$) using $q_i$, G, $P_i$ (public keys), and $w_i$.
    - $\{L1, L2, L3, L4, L5\} = \{q_1G + w_1P_1, q_2G+w_2P_2, q_3G, q_4G+w_4P_4, q_5G +w_5P_5\}$
  - Challenge (c) is derived from a hash function using the message, commitment points, and additional random points, $c = H_s(m, L_1, ...., L_5, R_1,...R_5)$.
  - Response: Calculates five response values ($c_i$) and five new random numbers ($r_i$).
    - $\{c_1, c_2, c_3, c_4, c_5\} = \{w_1, w_2, c-(c_1 + c_2 + c_4 + c_5), w_4, w_5\}$
    - $\{r_1, r_2, r_3, r_4, r_5\} = \{q_1, q_2, q_3 - c_3 · x, q_4, q_5\}$
  - Signature (σ) includes key image, response values, and new random numbers, $σ  = \{I, c_1, ..., c_5, r_1, ..., r_5\}$
  - Verification: Ensures the consistency between the challenge, message, commitment points, and response values, $c_1+...+c_5 = H_s(m, L_1',...,L_5', R_1',...,R_5')$

**In Conclusion:**

BLS excels in blockchain consensus due to its efficient batch verification. ECDSA remains a
