# ETAAcademy-ZKMeme: 14. CPA

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>14. CPA</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>cpa</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

### CPA Security:

The concept of CPA security is similar to that of semantic security, both referring to security under chosen plaintext attacks. The difference lies in the fact that the CPA security model allows attackers to make multiple queries to the challenger (since the challenger encrypts a plaintext sent by the attacker with the key each time, simulating the scenario of key reuse). In experiment EXP(0), the challenger always encrypts the plaintext $m_0$ ​ received from the attacker, while in EXP(1), it always encrypts the plaintext $m_1$. Each query must consist of two plaintexts of equal length, but the lengths of plaintexts between different queries can vary.

### ECB (Electronic Code Book)

ECB is an encryption method that divides the message into blocks, encrypts each block using a block cipher sequentially, and concatenates the resulting ciphertext blocks to form the final ciphertext. However, ECB is not suitable for encrypting multiple message blocks because identical plaintext blocks produce identical ciphertext blocks, which can lead to information leakage. Furthermore, ECB mode has vulnerabilities in terms of semantic security, as attackers can infer plaintext information by observing identical and different ciphertext blocks. Attacker A generates two plaintexts $m_0$ ​ and $m_1$ , each consisting of two plaintext blocks. Both blocks of $m_0$ ​ are identical, denoted as m[0], while the blocks of m_1 ​ are different, denoted as m′[0] and m′[1]. Thus, $𝐴𝑑𝑣 = ∣𝑃𝑟[𝑊_0] − 𝑃𝑟 [𝑊_1 ] ∣ = ∣ 0 − 1 ∣ = 1$, which is significant. Therefore, to ensure the overall security and semantic security of the message, it is necessary to use other more secure encryption modes such as CBC and CTR.

### Randomness or Nonce

To achieve CPA security, introducing randomness or nonce can address the issue of information leakage due to key reuse. The introduction of randomness or nonce ensures that the ciphertext depends not only on the key and plaintext but also on the random number or nonce associated with it, thereby increasing the randomness of the ciphertext. Randomized encryption schemes and the introduction of nonce are two methods to achieve this goal, both effectively preventing identical plaintexts from producing identical ciphertexts, thus ensuring the security of the encryption process. It is important to ensure that the space of random numbers and nonce values is sufficiently large to reduce the likelihood of repeated selection.
