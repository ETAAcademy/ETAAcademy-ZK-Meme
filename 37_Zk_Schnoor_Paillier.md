# ETAAcademy-ZKMeme: 37. zk-Schnorr & Paillier-N

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>37. zk-Schnorr & Paillier-N</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>zk_Schnorr_Paillier_N</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

## Zero-Knowledge Proofs

Zero-knowledge proofs (ZKPs) enable one party to prove to another that they possess specific knowledge without revealing any information beyond the proof itself. zk-Schnorr and zk-Paillier-N are two examples of ZKPs.

### zk-Schnorr Proof Protocol

zk-Schnorr Proof Protocol is used to prove knowledge of an ECC private key without revealing it. There are two versions of this protocol: **send commitment or challenge and response => compute the challenge or commitment**

**Setup**: Let G be the generator of an elliptic curve with scalar field $F_r$, and base field $F_q$. The prover’s private key is $sk$, and the public key is $PK$, satisfying the discrete logarithm relation $PK = sk \cdot G$.

- 1. Prover selects a random number r and computes $R := r \cdot G$.
- 2. Prover then calculates the challenge $c = \text{hash}(PK, R) \mod |F_r|$.

- **Version A**:

  3. Response is $z = r + c \cdot sk \mod |F_r|$, and send $(R, z)$.
  4. Verifier calculates $c = \text{hash}(PK, R) \mod |F_r|$ and checks that $z \cdot G = R + c \cdot PK$.

- **Version B**:

  3. Response is $z := r + c \cdot sk \mod |F_r|$, and send and send $(c, z)$.
  4. Verifier computes $R = z \cdot G - c \cdot PK$ and checks that $c = \text{hash}(PK, R) \mod |F_r|$.

Both versions ensure that the private key $sk$ remains confidential during the proof process.

### zk-Paillier-N Proof

**zk-Paillier-N Proof** is used to prove knowledge of the Paillier key pair, specifically that the public key $N$ and the private key $\phi(N)$ are coprime, i.e., $\gcd(N, \phi(N)) = 1$. **Here, $\Phi(n)$ remains secret because $\Phi(n) = (p - 1)(q - 1)$. However, $N$ can be public as it is known but does not reveal $p$ and $q$.**

**1. Principles:**

- **Primitive Roots**: For a prime $p$, there exists a primitive root $g$ such that $g^{\phi(p)} \equiv 1 \mod p$ and $g^{\frac{p-1}{\Delta}} \neq 1 \mod p$ where $\Delta$ is a prime factor of $p-1$, because $\frac{p-1}{\Delta} < \phi(p)$, contradicting the definition of primitive roots where
  $\phi(p)$ should be the smallest such exponent.

- **Baby-Step Giant-Step Algorithm (BSGS)**: To solve $g^N \equiv t \mod p$ where $g$ and $p$ are coprime and $g$ is a primitive root, $A, B \in [0, \sqrt{p}]$, and $N = A \sqrt{p} - B$. This transforms the congruence $g^{A \sqrt{p} - B} \equiv t \mod p$ into $g^{A \sqrt{p}} \equiv t g^B \mod p$. By brute force enumeration of $A$ and $B$, we could compute the values of both sides of the equation to determine $N$.

- **Modular Congruence**: If $\gcd(N, (p-1)) = 1$, the congruence $Ng \equiv t \mod (p-1)$ has a unique solution for $g$. Since $\gcd(N, (p-1)) = 1$, we can directly cancel $N$ from the congruence, giving $Ng \mod (p-1) \equiv t \mod (p-1) \equiv Ng' \mod (p-1)$.

- **Nth Residue**: If the congruence $g^N \equiv t \mod p$ has a solution $g$, t is an $Nth$ residue modulo $p$, otherwise an $Nth$ non-residue, where a unique $g$ is the a primitive root modulo $p$. To solve this, construct the congruence and use the BSGS algorithm to find the unique solution, resolving the $Nth$ residue problem.

**2. zk-Paillier-N Proof of Paillier Private Key**

- **Key Generation**: The prover generates a Paillier key pair and sends the public key $N$ to the verifier.

- **Nth Power Proof Protocol**: Using the Paillier public key, the prover calculates $t = g^N \mod N^2$. The prover then uses the $Nth$ power proof protocol (with $s=1$) to demonstrate that they know the Nth root of $t$, which is $g$, such that $t \equiv g^{n^s} \mod n^{s+1}$:

  - 1. Commitment: prover selects a random number $r$ and computes $a = r^{n^s} \mod n^{s+1}$ and sends $a$.
  - 2. Challenge: verifier selects a random k-bit challenge $e$ and sends it to the prover.
  - 3. Response: prover computes $z := r \cdot g^e \mod n^{s+1}$ and sends $z$.
  - 4. Verification: verifier checks if $z^{n^s} == a t^e \mod n^{s+1}$ holds true.

    This protocol can be seen as an instance of the Sigma protocol, which includes commitment, challenge, response, and verification steps.

The prover generates a proof and sends $t$ along with the proof to the verifier. The verifier uses the Paillier private key to extract the $Nth$ root of $t$ and checks if it matches $g' == g$. If the verification succeeds, it confirms that the prover knows the Paillier private key.
