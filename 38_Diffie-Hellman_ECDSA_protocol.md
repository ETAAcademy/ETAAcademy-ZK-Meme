# ETAAcademy-ZKMeme: 38. Diffie-Hellman & ECDSA protocol

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>38. Diffie-Hellman & ECDSA protocol</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Diffie-Hellman_ECDSA_protocol</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

## A Novel Approach to Two-Party Cryptography

A groundbreaking cryptographic protocol has been proposed, which introduces innovative methods for key establishment, signature generation, and security enhancements in two-party interactions. This protocol leverages a combination of well-established cryptographic techniques and novel approaches to create a robust and secure framework.

### Two-Party Key

The Diffie-Hellman key exchange involves scalar multiplication of each party's private key with a base point on an elliptic curve to obtain a shared public key: $Q_{common} = x_1x_2G$. This is analogous to $p$ being a prime number and $g$ its primitive root, where $Q_2^{x_1} = (g^{x_2})^{x_1} = (g^{x_1})^{x_2} = Q_1^{x_2}$.

To verify the authenticity of the shared secret, the protocol incorporates zero-knowledge proofs. Both parties employ [zk-Schnorr proofs](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/37_Zk_Schnoor_Paillier.md) to demonstrate knowledge of their respective private keys without revealing them, i.e., for private key $x_1$, generating $proof_1$, sending $(proof_1, Q_1)$, verifying $(proof_2, Q_2)$, and vice versa.

Additionally, [zk-Paillier-N proofs](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/37_Zk_Schnoor_Paillier.md) are used to certify the validity of the Paillier key pairs involved in potential encryption schemes $(pk, sk)$, meaning the public key $N$ and private key $\phi(N)$ are coprime,i.e., $gcd(N, \phi(N)) = 1$.

To circumvent the complexities of key exchange and proofs, the protocol also considers:

- Trusted Third-Party Involvement: A trusted entity can collect private keys from both parties, compute the shared secret, and securely distribute it.
- Ideal Function (Smart Contract): A smart contract can serve as a trusted intermediary, handling key exchange and secret computation in a verifiable and transparent manner. An ideal function can be considered as a smart contract on Ethereum that strictly enforces preset rules and maintains confidentiality.

An alternative key combination method is proposed where private keys are added instead of multiplied. In this case, the shared public key is the sum of the individual public keys, and the shared private key is the sum of the individual private keys: $Q_{common} = Q_1 + Q_2$, with the common private key $x_{common} = x_1 + x_2$.

### Two-Party Signature

Building upon the established shared secret from the two-party key exchange, the ECDSA protocol extends to include signature generation and verification. Both parties have already used zk-Schnorr to prove they know their random point $k_1$ or $k_2$.

1. Party $P_2$ calculates the Diffie-Hellman common random point $R := k_2 \cdot R_1$, parses the coordinates of $R$ as $(r_x, r_y) := R$, and computes $r := r_x \mod |F_r|$. Then, $P_2$ selects a random number $\rho \in Z_{F_r^2}$ and performs Paillier homomorphic encryption to compute $c_1 := Enc_{pk}(\rho \cdot |F_r| + [k_2^{-1} \cdot m' \mod |F_r|])$, which effectively generates a random value. Next, $P_2$ performs the homomorphic computation $v := k^{-1}_2 \cdot r \cdot x_2 \mod |F_r|$, then $c_2 := v ⨂ c_{key}$, effectively combining the private keys $x_1x_2$, and finally $c_3 := c_1 ⊕ c_2$. Homomorphic addition is used to ensure that the common private key $x_1, x_2$ is not directly exposed and $c_{key} = Enc_{pk}$ is a key from the key generation phase held by $P_2$. Party $P_2$ then sends $c_3$ to $P_1$.

2. Party $P_1$ generates the signature. $P_1$ calculates the Diffie-Hellman common random point $R := k_1 \cdot R_2$, parses $(r_x, r_y) := R$, and computes $r := r_x \mod |F_r|$. $P_1$ then decrypts $c_3$ using Pailliar decryption to obtain $s' = Dec_{sk}(c_3)$, computes $s'' = k^{-1}_1 \cdot s' \mod |F_r|$, and sets $s = \min\{ s'', |F_r| - s''\}$. $P_1$ then sends the signature $(r, s)$ to $P_2$ or broadcasts it directly to the blockchain system for verifiers to check the signature $(m', (r, bool, s), Q_{common})$.

### How to Prevent Private Key Theft and Recover the Common Private Key

1. **Key Derivation:** is based on calculating new private and public keys from old ones without re-running the key generation protocol. The new common public key $Q_{common, new} = f_l \cdot Q_{common} = (f_l \cdot x_1)x_2 \cdot G = x_{1, new}x_2 \cdot G$. Here, $f$ is calculated using the chain code, common public key, and a public counter with HMAC 512 (similar to a hash), $f = HMAC512(cc, Q_{common}, counter)$. The 512-bit output can be split into two 256-bit parts, $f = f_l || f_r$, where $|f_l| = |f_r| = 256$. The chain code $cc$ is the hash value $cc = sha256(Q_{common})$. In public chains like Bitcoin and Monero, it's necessary to have multiple key pairs (as opposed to Ethereum's account model, which requires only one key pair). This precaution is mainly to ensure security on the users ($Q_1$, $x_1$), while the server remains unchanged, with plenty of methods to protect it.

2. **Key Refresh:** To further strengthen the security of the system, a key refreshing mechanism is implemented. This process involves periodically updating private keys while preserving the corresponding public key. For example, $x_{common} = x_1x_2 = (r^{-1}x_1)(rx_2)$ or $x_{common} = x_1 + x_2 = (x_1 + \Delta) + (x_2 - \Delta)$. If a hacker obtains the old private key, it will be useless because it has been refreshed to a new private key while the public key remains unchanged.

3. **Recover Key:**: To address the potential risk of server failure or malicious behavior, a recovery mechanism involving a trusted custodian is implemented. The server splits its private key $x_2$ into $n$ fragments $\{x_{2,1}, ..., x_{2,n}\}$, each small enough for brute force attacks. The server then encrypts each fragment using the custodian's public key $Q_e$ via [ElGamal encryption](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/25_EIGamal_Gramer-Shoup_ECIES.md) $(D_i, E_i) := (x_{2, i} \cdot G + r_i \cdot Q_e, r_i \cdot G)_{i \in \{1,...,n\}}$, and sends the ciphertext to user $P_1$. If the server is compromised, the custodian will reveal its private key $k_e$, enabling users and attackers to obtain $x_{2,i} \cdot G := (D_i - k_e \cdot E_i)_{i \in {1,...,n}}$, and brute-force $x_{2,i}$. The user can then compute the server's private key $x_2 := \sum\nolimits_{i=1}^{n} x_{2,i}$ and recover the full private key $x = x_1x_2$, ensuring only they can access the funds.
