# ETAAcademy-ZKMeme: 30. Paillier Homomorphic Encryption

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>30. Paillier Homomorphic Encryption</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Paillier_Homomorphic_Encryption</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

### Paillier Homomorphic Encryption

Paillier encryption is a powerful tool for secure computations on encrypted data. Here's a breakdown of its key points:

**Key Generation**:

1. We start with two large prime numbers, p and q, of the same length ensured by gcd(pq, (p-1)(q-1)) = 1.
2. Compute n = pq and the least common multiple $\lambda = \text{lcm}(p-1, q-1)$.
3. Define the division function $L(y) = \frac{y-1}{n}$.
4. Choose a positive integer $g = 1 + n ∈ Z^*_{n^2}$, such that $\mu = (L(g^λ \mod n^2))^{-1} \mod n$.
5. Public Key: The public key is simply n = p\*q.
6. private Keys: The private keys are the original prime numbers p and q and the key generation can be simplified:

- Set g = n + 1, $\lambda = \phi(n)$, and $\mu = \phi(n)^{-1} \mod n$, where $\phi(n) = (p-1)(q-1)$.

**Encryption**:

1. For a message $m \in \mathbb{Z}\_n$, choose a random number $r \in \mathbb{Z}^*_n$.
2. Compute the ciphertext $c := g^m \cdot r^n \mod n^2$.

**Decryption**:

1. Given the ciphertext $c \in \mathbb{Z}_{n^2}$, compute the plaintext as $m := L(c^\lambda \mod n^2) \cdot \mu \mod n$.

The decryption relies on the properties of the Carmichael function and the Taylor expansion $(1 + n)^x \mod n^2 = 1 + nx \mod n^2$.

**Homomorphism**:

Paillier encryption is special because it allows performing calculations directly on encrypted data:

- **Addition ⊕**:
  Given two ciphertexts
  $c_1 ⊕ c_2 = c_1c_2 \mod n^2 = (g^{m_1} · r^n_1 \mod n^2)(g^{m_1} · r^n_1 \mod n^2) = g^{m_1+m_2}·(r_1r_2)^n \mod n^2$, the homomorphic addition is:
  $c_1 ⊕ c_2 = c_1c_2 \mod n^2 = Enc_{pk}(m_1 + m_2 \mod n)$
- **Scalar Multiplication**:
  For $a \in Z_n$ and $c=Enc_{pk}(m)$ :
  $a ⨂ c = c^a \mod n^2 = g^{am}· (r^a)^n \mod n^2 = Enc_{pk}(a·m \mod n)$

**Optimizations**:
There are techniques to improve the efficiency of Paillier encryption:

1. **Jacobi and Legendre Symbols**: Using specific prime numbers and additional calculations can reduce the size of random numbers needed, making computations faster.

   - Choose primes p and q as Blum integers: $p = q = 3 \mod 4$ and $\text{gcd}(p-1, q-1) = 2$.
   - Choose a random number $x \in \mathbb{Z}^*_n$ and compute $h = -x^2 \mod n$ with Jacobi symbol -1 to ensure the existence of quadratic residues.
   - Set $f = h^n \mod n$. The private key remains (p, q), but the public key is extended to (n, f), reducing the length of the random number a. If the factorization problem is hard, the computation of $f^a \mod n^2$, where $a \in Z_{2^{n/2}}$, is indistinguishable from the original $r^n \mod n^2$. Thus, this adjustment maintains the same level of security.

2. **Chinese Remainder Theorem (CRT)**: This mathematical concept can speed up both encryption and decryption by working with smaller numbers.
   - Leverage the isomorphism $\mathbb{Z}\_n \cong \mathbb{Z}\_p \times \mathbb{Z}\_q$ to accelerate encryption and decryption.
   - Encryption: $c = g^m \cdot r^n \mod n^2 = g^m \cdot f^a \mod n^2$.
   - Decryption: Compute $c^\lambda \mod n^2$ using CRT for efficiency.

**Application**:
Paillier encryption is used in a subprotocol of the ECDSA multi-signature protocol called the Share Conversion Protocol (MtA):

- Alice encrypts her share using her Paillier public key and sends it to Bob.
- Bob performs homomorphic computations and sends the result back to Alice.
- Alice decrypts the result using her Paillier private key.

This demonstrates how Paillier encryption enables secure computations on sensitive information while keeping the data itself confidential.
