# ETAAcademy-ZKMeme: 20. Plonk

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>20. Plonk</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Plonk</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

The Halo 2 proving system is structured around three key components: Plonkish Arithmetization, which efficiently expresses constraints and satisfying witnesses as low-degree polynomials, accommodating both local and global constraints; the Inner Product Argument, serving as the polynomial commitment scheme for efficient verification of proofs by committing to polynomials and checking against challenges from the verifier; and an Accumulation Scheme, derived from the polynomial commitment scheme, facilitating recursion by transforming the verification process into a constant-size accumulator, ensuring efficiency and scalability in proof verification.

Plonkish arithmetization enables flexible circuit design without multiple trusted setups, using Lagrange polynomials and roots of unity to represent circuit wires. Ultra Plonk introduces the lookup argument for efficiency, aiding in operations like SHA256. Different types of columns in circuits, such as instance, advice, and fixed columns, are used. The optimization goal is to reduce circuit area to minimize computation time and proof size, crucial for Fast Fourier Transforms (FFTs) and proof commitments. API enhancements like dynamic lookup tables and a multi-phase prover are used for circuit developers. The polynomial commitment process involves arithmetizing statements, vanishing arguments, and multi-point openings, leading to the inner product argument for efficient verification.

Commitments are categorized based on queried points, with evaluations interpolated to form polynomials for correctness checks. The inner product argument, detailed later, enables logarithmic-sized opening proofs, akin to a radix-2 FFT. The process then transitions to an accumulation scheme, splitting proof verification into cheap accumulation and expensive decider steps. Incrementally verifiable computation (IVC) allows for iterative accumulator updates and verification, with proofs consisting of the accumulator, snark proof, and witness.
