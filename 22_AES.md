# ETAAcademy-ZKMeme: 22. AES

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>22. AES</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>AES</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

The Data Encryption Standard (DES) has a key length of 56 bits, which results in a theoretical security strength of $2^{56}$, insufficient for providing adequate security. As a result, the Advanced Encryption Standard (AES) was developed. It involves expanding to n+1 keys and the round function of SubBytes, ShiftRows, MixColumns, and AddRoundKey for ciphertext. Conversely, ciphertext can be decrypted by applying the inverse round function with the key for plaintext. Therefore, the round function and key expansion are the most crucial parts of AES.

The round function consists of four operations: SubBytes, ShiftRows, MixColumns, and AddRoundKey.

- SubBytes, a non-linear transformation, where each byte of input data is replaced by another byte according to a predefined substitution box (S-box) to increase data non-linearity and security. For instance, in the SubBytes operation, the input's upper 4 bits determine the row index (x coordinate), while the lower 4 bits determine the column index (y coordinate). Consequently, the S-box corresponding to this row-column combination is accessed to find the substituted value. This substitution process can also be inverted by finding the inverse of the substituted value. Additionally, this substitution operation can be equivalently expressed using matrix notation, where each byte is XORed with specific bytes from adjacent positions, resulting in a new byte, that is, b'_i = b_i⊕ b_{i + 4 mod8}⊕ b_{i + 5 mod8}⊕ b_{i + 6 mod8}⊕ b_{i + 7 mod8} ⊕ c_i, The inverse operation involves a similar XOR process but with a different set of adjacent bytes, enabling the transformation to be reversed, b'_i = b_i⊕ b_{i + 2 mod8}⊕ b_{i + 5 mod8}⊕ b_{i + 7 mod8} ⊕ d_i.

- ShiftRows is a linear transformation that cyclically shifts each row in the data block to the left by a fixed number of positions. This operation can also be represented as a multiplication operation between the input data and a predefined matrix. In matrix notation, the linear expression for shifting a row is: state'[i][j] = state[i][(j+i)%4], where i and j range from 0 to 3, representing the horizontal and vertical coordinates respectively. The reverse row shifting operation is the inverse of this process, where the linear expression is: state'[i][j] = state[i][(4+j-i)%4], with i and j ranging from 0 to 3.

- Column mixing (MixColumns) involves two main operations: byte addition (XOR operation) and byte multiplication. Byte multiplication is performed in a finite field, modulo an irreducible polynomial denoted as m(x). When a byte is multiplied by 2, it results in shifting the binary representation of the byte to the left by one bit.

- In the Round Key Addition (AddRoundKey) operation, a simple XOR operation is employed. During encryption, the input of each round is XORed with the round key, resulting in ciphertext: c = m ⊕ k. Conversely, during decryption, the round operation is XORed, enabling the recovery of plaintext using the same round key: m = c ⊕ k.

The key expansion algorithm is a sequential process that involves the generation of multiple round keys using the g function. This function incorporates previous operations such as byte substitution, shifting, and the addition of round constants.

The block cipher encrypts data in 128-bit blocks. For data ranging from 128 bits to 300 Gbits, multiple iterations of the block cipher are required. This entails calling the block cipher repeatedly in a loop, utilizing modes of operation such as CBC, CTR.
