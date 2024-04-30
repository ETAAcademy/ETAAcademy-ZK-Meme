# ETAAcademy-ZKMeme: 18. Ideal Cipher Model (ICM)

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>18. ICM</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ICM</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

- **Ideal Cipher Model (ICM)**: In the ICM model, the encryption function E is replaced with a series of random permutations $\{\Pi_k: k \in K\}$, where each key k corresponds to a random permutation $\Pi_k$. Attackers can query the challenger for the results of these random permutations without knowing the specific key or permutation. Encryption is represented as $E_k(x) = \Pi_k(x)$, and decryption as $D_{k'}(y) = \Pi_{k'}^{-1}(y)$, where $\Pi_k$ is the random permutation corresponding to key $k$ and $\Pi_{k'}^{-1}$ is the inverse permutation corresponding to key k'.

- **Key Recovery Attack**: The key recovery attack aims to find the key used to encrypt a given plaintext, leading to key leakage. In a simple exhaustive attack, the attacker iterates through the key space, decrypting the ciphertext with each key and comparing the results to the known plaintext. The attacker can query the challenger for three types of information: the result of applying a random permutation to a plaintext, (k', a_i) to get $b_i = \Pi_{k'}(a_i)$, and $(k', b_i)$  to get $a_i = \Pi_{k'}^{-1}(b_i)$. The attacker then outputs their guessed key $\hat{k} \in K$, with success defined as finding the correct key k.

- **Application of the Ideal Cipher Model**: In a key recovery attack using the ICM, the attacker prepares Q pairs of plaintext-ciphertext and exhaustively tries each possible key k', comparing the resulting permutations with those obtained using the known key k. The attacker's advantage Adv satisfies $Adv \geq 1 - \frac{|K|}{(|X| - Q)^Q}$, where $|K|$ is the size of the key space, |X| is the size of the message space, and Q is the number of pairs of plaintext-ciphertext. This attack leverages the ICM's properties, simulating queries to the original encryption function using queries to random permutations.

- **Explanation**: The proof replaces the encryption function E with random permutations $Pi_k$, queries the challenger for the results of applying these permutations, and exhaustively compares these results to those obtained using different keys. If a key k' results in the same permutation as the known key k, the attacker infers that k = k'. The attacker only needs to prepare three pairs of plaintext-ciphertext for the attack, making it practical for real-world scenarios.
