# ETAAcademy-ZKMeme: 12. ZKEVM

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>12. ZKEVM</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>zkevm</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

More Detail for zkEVM:
[Plonk](https://github.com/zcash/halo2/blob/27c4187673a9c6ade13fbdbd4f20955530c22d7f/src/plonk/lookup/prover.rs)
[EVM](https://github.com/ethereum/go-ethereum/blob/f05419f0fb8c5328dca92ea9fb184d082300344a/core/vm/interpreter.go)

## Efficient VM Verification with Plonk:

Instead of using Merkle trees, which are expensive for each operation, a plonk mapping approach is used to verify a virtual machine (VM) efficiently. This allows for quick access to data (opcodes and values) needed for verification. Halo2 uses Zcash's recursive zero-knowledge proofs, which employ a specific recursive curve (non-Ethereum compatible curve) where the scalar field equals the base field. It replaces KZG or Dan commitments with vector inner product commitments, eliminates bilinear mappings (pure discrete logarithms), and achieves recursive zero-knowledge proofs. This allows the entire algorithm to be verified using a circuit board, eliminating the need to validate certain elements separately. Compared to previous methods that required bilinear mappings on the circuit board, this approach simplifies verification significantly.

## Bytecode Translation:

Translating bytecode (VM instructions) into a verifiable format is difficult, especially for jumps between instructions. To solve this, a method called "plookup" is used. Plookup lets us choose instructions in any order and verify them against a program counter, ensuring they follow the correct execution flow.

Plonk constructs polynomials based on gate constraints and linear constraints. The gate constraint represents a multiplication gate and an addition gate: $𝑄_𝐿 ( 𝑋 ) ⋅ 𝑎 ( 𝑋 ) + 𝑄_𝑅 ( 𝑋 ) ⋅ 𝑏 ( 𝑋 ) + 𝑄_𝑜 ( 𝑋 ) ⋅ 𝑐 ( 𝑋 ) + 𝑄_𝑀 ( 𝑋 ) ⋅ 𝑎 ( 𝑋 ) ⋅ 𝑏 ( 𝑋 ) + 𝑄_𝑐 ( 𝑋 ) = 𝑍 ( 𝑥 ) ⋅ 𝐻 ( 𝑥 )$. The linear constraint represents the correct connection between gates: 𝑓 ( 𝑖 ⋅ 𝐺 ) = 𝑓 ( 𝑖 ⋅ 𝐺 ) + 𝛽 ⋅ 𝑖 + 𝛾 , 𝑔 ( 𝑖 ⋅ 𝐺 ) = 𝑔 ( 𝑖 ⋅ 𝐺 ) + 𝛽 ⋅ 𝜎 ( 𝑖 ) + 𝛾. The aggregation proof linearly combines n proofs to verify a bilinear pairing $𝑒 ( 𝑥 ⋅ 𝐺_1 , 𝑦 ⋅ 𝐺_2 ) = 𝑒 ( 𝑗 ⋅ 𝐺_1 , 𝑘 ⋅ 𝐺_2 )$. Therefore, Plonk has a very limited expressive power which needs and circuit optimization and table look-up optimization.

By using a coordinate accumulator to prove that elements belongs to this XOR table, and the operation results can be used with the table (while ensuring zero-knowledge). Precomputing a publicly available and correct input-output table Table for speed improvement, at the cost of storing the table on the Ethereum layer, increasing storage gas and the complexity of verification, but save gate resources for more transaction units. The accumulator is used to prove that secret data is in the table, so as to prove that the operations on input and output are correct.

## Stack Circuit:

The VM's stack is crucial for calculations and comparisons. A stack commitment is used to track changes to the stack during verification. Additionally, a call_context helps manage memory, storage, and stack operations when the VM makes calls to other functions. This acts as a central storage for all VM data like stack, memory, and other elements. Sub-circuits for Stack and Memory ensure that the VM accesses data in a valid way. By separating memory usage between different types of calls (external vs internal), this approach optimizes memory usage and data access during verification.
