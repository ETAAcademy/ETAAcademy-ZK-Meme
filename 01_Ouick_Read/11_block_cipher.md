# ETAAcademy-ZKMeme: 11. Block Cipher

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>11. Block Cipher</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>block_cipher</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

E represents the encryption algorithm, and D represents its corresponding decryption algorithm. Together, (E, D) form a block cipher defined on (K, X), where K is the key space and X is the block space. Both the plaintext and ciphertext spaces of a block cipher are finite sets, denoted as X. This means that for ∀ k ∈ K and ∀ x ∈ X, the encryption E(k, x) produces a result within X. Each key k in the key space corresponds to a distinct encryption function, denoted as $f_k(.):= E(k, .)$, which transforms plaintext blocks into ciphertext blocks. Additionally, the inverse function of encryption, denoted as $D(k, .) := f_k^{-1}(.)$, serves as the decryption function.

In this security model of block ciphers, the security of a block cipher demands that the encryption permutation E(k, .), when given a randomly generated key k, behaves computationally like a random permutation. A random permutation, represented by a function f: X→X, rearranges elements within set X such that each element has a unique correspondence. These permutations collectively form a set Perm[X], from which a random permutation is selected. The security model involves two challengers: one randomly selects a key to generate an encryption permutation, and the other selects a random permutation. The attacker's objective is to distinguish between these scenarios by providing inputs and observing outputs. A secure block cipher should be computationally indistinguishable from a random permutation. The attacker's advantage, denoted as $Adv := |Pr[W_0] – Pr[W_1]|$, measures the discrepancy in the probability of obtaining the same output in different experiments.

The security of block ciphers can be summarized in three main aspects:

Unpredictability model: This model examines the interaction between challengers and attackers. The challenger randomly chooses a key k, and the attacker sends a series of plaintexts $x_1 ,…,x_n$ to the challenger. For each plaintext $x_i$ , the challenger returns the corresponding ciphertext $E(k,x_i )$. The attacker aims to predict the output tuple (x,y), where y = E(k,x), without prior knowledge of x. If no efficient attacker can predict successfully, the block cipher is deemed unpredictable.
Security by Contradiction: Assuming predictability of a block cipher leads to the existence of an efficient attacker who can successfully predict outcomes with a non-negligible probability. By constructing an algorithm B that exploits this predictability, the contradiction arises as B can distinguish between experiments with a non-negligible probability, violating the security requirement.
Key Space Matters: In the unpredictability model, if the key space |K| is not polynomial, the probability 1/|K| becomes non-negligible. In such cases, attackers can generate valid tuples with a non-negligible probability, undermining security. Therefore, a secure block cipher must possess a key space of super-polynomial size to thwart exhaustive key search attacks.
