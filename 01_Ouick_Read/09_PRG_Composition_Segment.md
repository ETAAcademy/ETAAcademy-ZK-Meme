# ETAAcademy-ZKMeme: 9. PRG Composition and Segment

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>09. PRG Composition and Segment</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>PRG_Composition_Segment</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

While standard PRGs (Pseudorandom Number Generators) have fixed output lengths, we can achieve longer or shorter sequences for practical applications:

### Parallel Composition (Fast, Seed-Hungry):

Let $G: {0, 1}^s -> {0,1}^n$ be a secure PRG,

then $G':{0,1}^{s×t} -> {0,1}^{n×t}$

Imagine a secure PRG, G, that takes seeds of length s and outputs sequences of length n. We can construct a new PRG, G', that works in parallel on multiple processors for faster generation. G' takes a seed of length t × s (combining t individual seeds of length s). It feeds each of these t seeds into G independently, resulting in t outputs of length n. Finally, G' combines these t outputs into a single sequence of length n × t.The advantage of this method is its fast execution in multi-processor systems. However, the drawback is that G' requires t times the length of seeds compared to G, which can be cumbersome and time-consuming to obtain.

### Blum-Micali Serial Construction (Seed-Efficient, Slower):

Let $G: {0, 1}^s -> {0,1}^n$ be a secure PRG,

then $G':{0,1}^s -> {0,1}^{n×t+s}$

This method, known as the Blum-Micali construction, builds upon a secure PRG G with seed length s and output length n. The new PRG, G', also has a seed length of s but outputs a sequence of length n × t + s.
It works by iteratively calling G on the seed:

- Choose a random seed k.
- Call G(k) and split its output into two parts.
- The first part (length s) becomes the seed for the next call (k_1).
- The second part (length n) becomes the first output segment (r_1).
- Repeat this process (t times) until the last call to G.
  Finally, output the last seed (k_t) and all the collected output segments (r_1 to r_t).

### Extracting Secure Randomness:

The beauty of secure PRGs lies in their flexibility. You can extract any segment of their output and use it as a secure pseudo-random sequence. This extracted sequence is statistically indistinguishable from a truly random sequence of the same length., such as using a segment of the PRG output as a key in AES-128 encryption.
