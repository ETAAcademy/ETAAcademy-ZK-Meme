# ETAAcademy-ZKMeme: 25. EIGamal, Gramer-Shoup, ECIES

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>25. EIGamal_Gramer-Shoup_ECIES</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>EIGamal_Gramer-Shoup_ECIES</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

## Securing Your Messages: A Look at ElGamal Encryption on Elliptic Curves

ElGamal encryption, a cornerstone of public-key cryptography, gets a boost when combined with elliptic curves. This article explores different ElGamal encryption variations on elliptic curves, highlighting their functionalities and security improvements.

### Standard ElGamal Encryption:

Imagine a secure communication channel where you want to send a secret message (M) to someone. ElGamal encryption helps achieve this. Here's how it works on elliptic curves:

1. **Setting Up:**

   - The public key includes a special point or generator G on an elliptic curve with the order of p, and the public key, another point, $G_1 = a · G$ derived mathematically from G.
   - The private key is a secret number $a ∈ Z_p$.

2. **Encryption:**

   - The message (M) is converted into a point on the curve (a technical process, M ∈ G).
   - A random number $r ∈ Z_p$ is chosen.
   - The ciphertext consists of two parts:
     - $C_1 = r · G$.
     - $C_2 = M + r·G_1$.

3. **Decryption:**
   - The receiver uses their private key a and the ciphertext $(C_1, C_2)$ to recover the message M.
     - $M = (M+ r·G_1)-ar·G = C_2 - a·C_1$

### Hashed ElGamal Encryption:

Standard ElGamal becomes cumbersome when dealing with lengthy messages. Hashed ElGamal offers a solution:

1. The message m is converted into a digest using a hash function.

- m ∈ $\{0, 1 \}^{256}$,
- Public key: $G_1$,
- Random number: $r ∈ Z_p$
- Ciphertext: $C_1 = r·G, C_2 = m ⊕ hash( r · G_1)$

2. Encryption proceeds similarly to standard ElGamal, but $C_2$ combines the message digest with a hash of (r multiplied by $G_1$).

- Ciphertext: $(C_1, C_2)$,
- Private key: a,
- m = $(m ⊕ hash(r · G_1)⊕ hash(ar · G))=C_2 ⊕ hash( a·G_1)$

3. Decryption involves combining the received hash values with the private key (_a_) to retrieve the original message digest.

### ElGamal Encryption with Enhanced Security:

This variant, inspired by Cramer-Shoup encryption, utilizes multiple private and public key elements for increased security:

1. **Key Setup:**

   - The public key includes three points (C, D, and H), i.e., $C = x_1·G + x_2·G_2, D = y_1·G+y_2G_2, H = z·G_1$, derived from two generators $G_1, G_2$.
   - The private key consists of five secret numbers $(x_1, x_2, y_1, y_2, z) ∈ Z_p$.

2. **Encryption:**

   - The message M remains a point on the curve, M ∈ G.
   - A random number $r ∈ Z_p$ is chosen.
   - The ciphertext includes four parts:
     - $U_1 = r·G_1, U_2 = r·G_2$
     - $E = r·H + M,$ $a=hash(U_1, U_2, E)$,
     - $V= r·C+ ra ·D$.

3. **Decryption:**
   - The receiver uses the private key to verify the ciphertext's integrity through a mathematical check involving V= $rC+raD = r(x_1G_1 + x_2 ·G_2)+ra(y_1·G_1 + y_2 ·G_2) =x_1r·G_1+ y_1ar·G_1+ x_2r·G_2+y_2ar·G_2 = (x_1 + y_1a)r·G_1+ (x_2+y_2a)r·G_2 =(x_1+ y_1a)·U_1 + (x_2 +y_2a)·U_2$.
   - If the verification succeeds, the message M is recovered by subtracting terms from E, $M = (r·H+M)-zr·G_1 = E -z·U_1$.

**ECIES Encryption: A Modern Twist**

Elliptic Curve Integrated Encryption Scheme (ECIES) incorporates additional features for improved security:

- A technique called Key Derivation Function (KDF), i.e, [CTR](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/01_Ouick_Read/16_CTR.md) and [AES](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/01_Ouick_Read/22_AES.md), generates encryption and message authentication keys from a random number. This reduces private key size.
- Message Authentication Code (MAC) replaces the verification step in Cramer-Shoup encryption. Imagine a unique "fingerprint" created for the message using the derived key. The sender calculates this fingerprint and attaches it to the encrypted message. During decryption, the receiver uses the same key to generate a fingerprint of the received message. If both fingerprints match, it ensures the message hasn't been tampered with during transmission.

1. **Key Setup:**

   - Elliptic Curve Group: We have an elliptic curve group G with a specific order n and a generator point G.
   - Private Key: A secret random number d is chosen.
   - Public Key: This is derived by multiplying the generator point G by the private key $d ∈ Z_p$, $Q = d · G$.
   - Constant: A constant value h is used in the encryption process.

2. **Encryption:**

   - A new random number k is chosen within a specific range between 1 and n-1.
   - Two points are calculated based on the chosen random numbers:
     - **R:** This point is obtained by multiplying the generator point G by the random number k: R = k · G.
     - **Z:** This point is derived from the constant h, the public key point Q, and the new random number k: Z = hk · Q.
   - The x-coordinate (abscissa) of point Z is extracted denoted as x.
     - This x-coordinate and the point R are used as inputs to a Key Derivation Function (KDF): $(k_1, k_2)$ = KDF(x, R)
     - The KDF outputs two important keys:
     - **Encryption Key ($k_1$):** Used for encrypting the message $m ∈ {0, 1}$^*$ using the CTR mode of the AES (Advanced Encryption Standard) symmetric cipher, $C = CTR-AES-Enc_{k_1}$(m).
     - **MAC Key ($k_2$):** Used for generating a Message Authentication Code (MAC) to ensure message integrity, $t = MAC_{k_2}(C)$.

3. **Decryption:**

   - The receiver uses the private key to recover the encryption and MAC keys.
     - They decrypt the message using CTR mode and the recovered key. The actual message m, represented as a sequence of bits $\{0, 1\}^*$, is encrypted using the CTR mode of AES with the derived encryption key ($k_1$): $m = CTR-AES-Dec_{k_1}(C)$, where Z = hk·Q=hkd·G = hd·R.
     - The MAC tag is verified to ensure message integrity.
