# ETAAcademy-ZKMeme: 16. CTR

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>16. CTR</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ctr</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

### CTR

CTR mode is a classic block cipher working mode that utilizes a counter and a pseudo-random function. During encryption, an initialization vector (IV) is selected and used as a counter. A pseudo-random function generates a key stream, which is then XORed with the plaintext to produce the ciphertext. Decryption follows a similar process, where the same IV and pseudo-random function generate the key stream, which is then XORed with the ciphertext to retrieve the plaintext. CTR mode features parallel execution and does not require padding because each encryption operation uses a different IV, ensuring that the same plaintext corresponds to different ciphertexts, thus achieving CPA security. The proper selection of IV is crucial as it directly impacts the security of the system.

### IV selection

IV selection in CTR mode encryption has security issues with CTRC due to potential IV reuse and proposes two methods for secure IV selection: using a full-length IV or a half-length IV with a counter. RFC 3686 recommends a 128-bit IV with a fixed random portion and a counter, ensuring security and synchronicity between sender and receiver.

### CTR mode is superior to CBC mode for the following reasons:

- Parallelism: Encryption and decryption in CTR mode can be performed in parallel, whereas encryption in CBC mode is sequential.
- Ciphertext Length: For short plaintexts, CTR mode produces shorter ciphertexts compared to CBC mode.
- Encryption Algorithm: CTR mode only requires the encryption algorithm E, whereas CBC mode requires both encryption algorithm E and decryption algorithm D.
  Both modes can use random IVs or counter IVs. While CTR mode, e.g. CTRC, and CBC mode, e.g. CBCC, typically result in shorter ciphertexts, CBCC's security requires additional encryption of the counter IV, which reduces efficiency.
