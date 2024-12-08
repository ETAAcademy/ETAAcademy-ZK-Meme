# ETAAcademy-ZKMeme: 3. Stream Cipher

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>03. Stream Cipher</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>stream_cipher</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

Stream Cipher utilizes a pseudo-random generator (PRG) to extend a short key into a longer one. Subsequently, this extended key is combined bit by bit with the plaintext through XOR operations to produce the ciphertext. Conversely, it can also be XORed with the ciphertext to get the original plaintext.

Pseudo-random sequences come in various forms, but in cryptography, a pseudo-random sequence is one that cannot be distinguished from a genuinely random sequence. For a sequence to be considered random, it must meet two criteria: (1) Each bit should have a probability of 1/2 of being generated; (2) The generation of each bit should be statistically independent. That means whether a sequence is random or not depends on how it is generated rather than its appearance.

It is important to note that computers cannot generate truly random sequences (except for quantum computers). And the pseudo-random generator(PRG or PRNG) is used to produce pseudo-random sequences, hence its name. Let G be a function with an input length of s and an output length of n, where s is much smaller than n. In this case, $G: {0, 1}^s$ → ${0, 1}^n$, where s << n. Such a function G is termed as a PRG. It's worth noting that PRG must be efficiently computable, deterministic, and have a significantly smaller input length compared to its output length.

Considering G as a PRG, $G: {0, 1}^s$ → ${0, 1}^n$, $K={0,1}^s$, $M=C={0, 1}^n$, a stream cipher can be defined as follows:

<center>(E, D) is a stream cipher defined on (K, M, C):</center>
<center>E(k, m): G(k)⊕m = c;</center>
<center>D(k, c): G(k)⊕c=m;</center>

The stream cipher replaces the original random key with the output G(k) of the PRG, so that k represents the seed used in the PRG, which serves as the key in the stream cipher. The PRG's role is to transform the seed into an extensive output, which is then XORed with the plaintext to generate the ciphertext. The security of the stream cipher is dependent on the ability of the PRG's output to closely resemble true randomness, making it indistinguishable from an equally long random sequence.
