# ETAAcademy-ZKMeme: 24. SHA256, Keccak, and Poseidon

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>24. SHA256_Keccak_Poseidon</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>SHA256_Keccak_Poseidon</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

## Hashing Showdown: SHA-2 vs Keccak vs Poseidon

The digital world relies on secure ways to verify data integrity. Hash functions play a crucial role in this by creating unique "fingerprints" for any piece of information. This article explores four popular hash functions: SHA-2, BLAKE, Keccak (a variant of SHA-3), and Poseidon, highlighting their differences and suitability for specific tasks.

**SHA-2: The Robust Veteran**

SHA-2 (Secure Hash Algorithm 2) is a widely used and trusted hashing algorithm. It boasts a robust design with several rounds of processing involving bitwise operations and constant values.

- 1. Initial Hash Values:

  - SHA-2 uses 8 constant values, each 256 bits long.

  - The first 32 bits of each constant come from:

    - Taking the square root of a prime number (2, 3, 5, 7, 11, 13, 17, 19).
    - Converting the decimal part of the square root to a hexadecimal number.
    - Taking the first 8 hexadecimal digits (32 bits).
    - For example: $\sqrt{2}$ => 0.414213562373095048;=> 0.414213562373095048≈6∗16−1+a∗16−2+0∗16−3+...; => 0x6a09e667.

  - The remaining 64 constants are the first 64 prime numbers, with their first 32 bits taken from their cube root (similar to the first 8).

- 2. Basic Operations:

  - SHA-2 uses several bitwise operations to manipulate data during processing. Here's a simplified explanation:
  - Ch(x, y, z): Combines bitwise AND and XOR operations based on x, y, and z, Ch(x, y, z) = (x ∧ y) ⊕ (¬x ∧ z).
  - Ma(x, y, z): Similar to Ch, but uses different combinations of AND operations, Ma(x, y, z) = (x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z).
  - Σ(x): Rotates and shifts the bits of x by specific amounts, Σ0(x) = S2(x) ⊕ S13(x) ⊕ S22(x); Σ1(x) = S6(x) ⊕ S11(x) ⊕ S25(x).
  - σ(x): Similar to Σ, but with different shift amounts, σ0(x) = S7
    (x) ⊕ S18(x) ⊕ R3(x); σ1(x) = S17(x) ⊕ S19(x) ⊕ R10(x).

- 3. Data Preprocessing:
  - SHA-2 pads the message with bits before processing.
  - Padding starts with a single '1' bit, followed by zeros until the message length (in bits) plus 64 is a multiple of 512.
  - Essentially, the message is padded to a specific size for efficient processing.
- 4. [Round Function](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/01_Ouick_Read/22_AES.md):
  - The message is divided into 512-bit blocks, msg’= {msg, 1, 0,…0, msg-len-64bit}, and k zero bits calculated by (Len(msg)+1+k+64)/512=0.
  - Each block goes through multiple rounds of processing.
  - In each round:
  - The block is combined with additional constant values.
  - The combined data is processed using the basic operations (Ch, Ma, Σ, σ).
  - The result is used for the next round or becomes the final hash value.

SHA-2 excels in security and is ideal for various applications like digital signatures and file verification. However, its complex processing can be resource-intensive.

**Keccak: The Secure but Resource-Hungry Contender**

In SHA3, BLAKE is the fastest algorithm, while Keccak is the most secure. One of the primary difference from SHA2 lies in the round function. BLAKE involves 8 round functions (G0,...,G7) and 10 rounds. Keccak requires 24 rounds, with W=64 and L=6, resulting in nr=24 rounds. Each round includes θ(theta), ρ(rho), π(pi), χ(chi), and ι(iota) steps. Processing arranges 1600-bit data into a 5x5x64 structure, yielding a final 1088-bit output, with the top 256 bits serving as the hash value.

**Poseidon: The Efficient Challenger in the zk-SNARK Arena**

Zero-knowledge proofs (zk-SNARKs) are a cryptographic technique for proving information validity without revealing the details. Here, Poseidon, a hash function specifically designed for zk-SNARKs.

- It involves three main components:
  - Adding a constant value to the data.
  - Applying an S-box (substitution box) for non-linear transformation. This involves multiplying by $x^5$ (raised to the power of 5).
  - Applying a linear transformation using a full-rank matrix m' := A • m.

This streamlined approach makes Poseidon significantly faster and more resource-friendly compared to SHA-2 and Keccak, making it a preferred choice for zk-SNARK applications.
