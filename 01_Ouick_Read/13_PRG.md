# ETAAcademy-ZKMeme: 13. PRF

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>13. PRF</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>prf</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

### PRF Pseudo-Random Function

is similar to a block cipher PRP, where a random function resembles a random permutation, and attackers aim to distinguish between a randomly chosen function and a function based on a key. A block cipher can also be referred to as a PRP because it shares similarities with PRF in terms of definition and security models. Compared to PRF, PRP is a pseudo-random permutation with an inverse function. PRF is a deterministic function that takes a key and a data block as input and produces an output, y = F(k,x) ∈ Y. The security requirement for PRF is that, given a randomly generated key k, the function F(k,⋅) should appear "random" when mapped from X to Y. In the security model of PRF, attackers attempt to distinguish between randomly chosen functions and key-based functions.

### Relationship between PRP (Block Cipher) and PRF:

PRP (Pseudo-Random Permutation) and PRF (Pseudo-Random Function) are related, but the security of one does not guarantee the security of the other. PRP and PRF are conceptually related but not entirely the same. PRP is a part of PRF, but a secure PRP does not necessarily imply a secure PRF. The PRF Switching Lemma states that when the number of queries Q is polynomial, a secure PRP can also be considered a secure PRF. If (E,D) is a secure PRP defined on (K,X) with N=∣X∣, and an attacker A queries the challenger at most Q times (where Q is polynomial), then:|Adv_PRP(A) - Adv_PRF(A) | ≤ Q^2/2N. Here, Adv_PRP(A) represents the advantage of attacker A in the PRP security model, and Adv_PRF(A) represents the advantage of attacker A in the PRF security model.

### Construction of PRG (Pseudo-Random Generator) based on PRF (Pseudo-Random Function):

Both block ciphers (PRF) and stream ciphers (PRG) can be constructed based on each other. Utilizing PRF to construct PRG is straightforward. Let F be a PRF defined on (K,X,Y), and let $x_1 ​ ,…,x_n$ ​ be pairwise distinct elements in X. We can construct a PRG, denoted as G, with seed space 𝐾 K and output space $Y^n$ . The construction of G is as follows: $G(k):=(F(k,x_1 ​ ),…,F(k,x_n ​ ))$. It sequentially applies F to process $x_1 ​ ,…,x_n$ ​ with the key k, combining all outputs of F to form a final output containing n components.

### Construction of PRF based on PRG:

Tree-based Construction (previous output is the subsequent input): $F(k,x):=G_{x_n} ​ ​ (…G_{x_2} ​ ​ (G_{x_1} ​ ​ (G_x ​ )))$. Construction based on PRG is a method to build a PRF (Pseudo-Random Function) from a PRG (Pseudo-Random Generator). By iteratively applying the output of PRG, an output related to the input can be generated. This construction can be described using a complete binary tree, where each node corresponds to an output of PRG. This method is secure for fixed-length inputs but not for variable-length inputs. Let G be a PRG defined on $(S,S^2 )$, meaning the output length of G is twice its seed length. For simplicity, let $G(s):=(G_0 ​ (s),G_1 ​ (s))$, where $G_0 ​ (s)$ represents the first half of the output of G(s), and G_1 ​ (s) represents the second half. Based on G, a PRF denoted as F can be constructed, with a key space of S (same as the seed space of G), an input space of {0,1} n , and an output space also of S. For any key k∈S and input $x=(x_1 ​ ,…,x_n ​ )∈{0,1} n$ , F is constructed as $F(k,x):=G{x_n} ​ ​ (…G{x_2} ​ ​ (G{x_1} ​ ​ (G_x ​ )))$.
