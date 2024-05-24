# ETAAcademy-ZKMeme: 15. CBC

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>15. CBC</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>cbc</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

### CBC(Cipher Block Chaining mode)

CBC is a block cipher working mode based on PRP that achieves CPA security. In CBC mode, each plaintext block is first XORed with the previous ciphertext block, then encrypted using a PRP, forming a chain of ciphertext blocks. During encryption, an initialization vector (IV) is chosen and XORed with the first plaintext block m[1], then the result is encrypted using E to obtain the corresponding ciphertext block c[1], and subsequently, each plaintext block is XORed with the previous ciphertext block. The decryption process is the opposite, where the ciphertext is decrypted using the PRP's inverse function and then XORed with the previous ciphertext block to obtain the corresponding plaintext block. It is important to note that the ciphertext may be longer than the plaintext because the IV typically needs to be sent to the decryptor. Encryption needs to be performed sequentially, while decryption can be performed in parallel.

### Padding

CBC mode requires the use of PRP for security, as both encryption and decryption rely on PRP and its inverse permutation. CBC mode needs to handle cases where the message length is not fixed during encryption, which involves the issue of padding. If the message length is not an integer multiple of the block length, extra information needs to be padded after the last plaintext block to make its length reach a full block. Common padding methods include representing the number of padded bytes using digits or using systematic padding patterns such as PKCS#7. Additionally, an effective padding method called ciphertext stealing is introduced to avoid increasing the length of the ciphertext due to padding. Ciphertext stealing cleverly utilizes the last two ciphertext blocks to ensure that the ciphertext length matches the plaintext length. However, padding is still necessary if the message length does not exceed one block.

### IV

The choice of IV in CBC mode is crucial for security. CBC\$ and CBCC are two common methods, where the former uses random numbers and the latter uses counters. The IV must be unpredictable. In CBC$, the IV is typically generated using a pseudorandom generator but must be unpredictable to avoid security issues. In CBCC, the counter is predictable, which poses security risks if used directly. To increase unpredictability, another key can be used to encrypt the IV before XORing it with the first plaintext block, forming CBC-ESSIV. CPA security assessment based on nonce evaluates the security of encryption schemes, where attackers can manipulate the nonce to affect the encryption process. CBC-ESSIV may have security risks when using the same key, but it is CPA secure when using different keys. Therefore, IV selection and key management need to be handled carefully to ensure system security.
