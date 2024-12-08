# ETAAcademy-ZKMeme: 7. PRG

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>07. PRG</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>PRG</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

PRG (Pseudo-random Generator) is the fundamental component of a stream cipher, and the security of the stream cipher hinges entirely on the PRG. The security of PRG entails that its output should be indistinguishable from a sequence of random bits of the same length. Indistinguishability is measured by the difference in probabilities between an attacker correctly distinguishing the output of the PRG from truly random bits in various experiments, Adv := | Pr[W_0] - Pr[W_1]|, that is, the advantage is zero (|Pr[W0] – Pr[W1]|=0), or the advantage (Adv = |Pr[W0] – Pr[W1]|) is negligible.

The unpredictability of a PRG refers to scenarios where the attacker's ability to guess the next bit of the output based on previous bits should be equivalent to random chance. If all computationally feasible attackers cannot predict the output with any significant advantage, then the PRG is considered unpredictable. And the security of PRG and unpredictability are equivalent.

For example, The ciphertext c is represented as c = G(k) ⊕ m. Based on the values of the first i consecutive bits of G(k), denoted as r[0, …, i-1], the attacker can guess the value of r[i], and more until the plaintext information is leaked. A attacker, tossing a coin and leaving it to chance, has a probability of success Pr[A] equal to 1/2. The larger |Pr[A] – 1/2|, the greater the advantage.

Stream cipher's perfect secrecy is defined under ciphertext-only attacks, while its semantic security is defined under chosen plaintext attacks. The former involves an attacker having only ciphertexts, while the latter allows the attacker to choose plaintexts and observe the corresponding ciphertexts. Specifically, there are four types of attacks: 1) ciphertext-only attack: the attacker only has some ciphertexts; 2)known-plaintext attack: the attacker also knows the plaintext corresponding to the ciphertext; 3)chosen-plaintext attack (CPA): the attacker can choose some plaintexts and obtain the corresponding ciphertexts; 4) chosen-ciphertext attack (CCA): the attacker can choose some ciphertexts and obtain the corresponding plaintexts.

Stream cipher' semantic security implies that an attacker should be unable to discern any meaningful information about the plaintext by solely examining the ciphertext. This remains true even if the attacker has access to chosen plaintext-ciphertext pairs. If all feasible attackers fail to gain a significant advantage in distinguishing the ciphertexts of plaintexts, the encryption scheme is considered semantically secure, that is, after encrypting with E, the ciphertexts of m0 and m1 are computationally indistinguishable (denoted as ${E(k, m_0)} ≈_c {E(k, m_1)}$) for all efficiently chosen equal-length plaintexts $m_0, m_1 ∈ M$, where (E, D) is a symmetric encryption scheme defined on (K, M, C).
