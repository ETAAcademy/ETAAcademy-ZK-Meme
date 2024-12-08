# ETAAcademy-ZKMeme: 8. zkStark FRI

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>08. zkStark FRI</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>zkStark_FRI</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

More Detail for zkStark:<br>
[STARKs, Part I: Proofs with Polynomials](https://vitalik.eth.limo/general/2017/11/09/starks_part_1.html) <br>
[STARKs, Part II: Thank Goodness It's FRI-day](https://vitalik.eth.limo/general/2017/11/22/starks_part_2.html) <br>
[STARKs, Part 3: Into the Weeds](https://vitalik.eth.limo/general/2018/07/21/starks_part_3.html)

The zkStark proof system operates on a Merkle tree structure, where the leaves of the tree contain hash values derived from the polynomial's coefficients. These hash values are combined pairwise until reaching the root node of the tree.

The prover sends the Merkle root to the verifier, and the verification method involves connecting two points to form a straight line, using fixed points (x-coordinate, y-coordinate); the verification checks if the third point lies on the same line. Generally, the verifier validates the Merkle tree and only needs to use D+1 points to verify that D+1 points all lie on the polynomial, thereby verifying that the degree of the polynomial is less than D.

If the polynomial Q(x) = C(P(x)) = D(x) ⋅ Z(x), where Q(x) equals 0 at all these x-coordinates, then it is a multiple of the minimal polynomial that equals 0 at all these x-coordinates: $Z(x) = (x - x_1)⋅(x - x_2)⋅...⋅(x - x_n).$ Indirect proof is achieved by providing the quotient $D(x)=\frac{Q(x)}{z(x)}$.

Additionally, using the Fast Reed-Solomon Interactive Oracle Proofs (FRI), the computational complexity for the prover decreases exponentially by a factor of 2 with each iteration. This process folds logD times, halving the number of random numbers each time into a linear combination polynomial. This results in only needing $O(log^2n)$ operations to test a polynomial of degree D, where $D=2^n$.

The verification process primarily involves validating the Merkle branches and FRI proofs, both of which require significant computational resources:

```python

for i, pos in enumerate(positions):
    x = f.exp(G2, pos)
    x_to_the_steps = f.exp(x, steps)
    mbranch1 =  verify_branch(m_root, pos, branches[i*3])
    mbranch2 =  verify_branch(m_root, (pos+skips)%precision, branches[i*3+1])
    l_of_x = verify_branch(l_root, pos, branches[i*3 + 2], output_as_int=True)

    p_of_x = int.from_bytes(mbranch1[:32], 'big')
    p_of_g1x = int.from_bytes(mbranch2[:32], 'big')
    d_of_x = int.from_bytes(mbranch1[32:64], 'big')
    b_of_x = int.from_bytes(mbranch1[64:], 'big')

    zvalue = f.div(f.exp(x, steps) - 1,
                   x - last_step_position)
    k_of_x = f.eval_poly_at(constants_mini_polynomial, f.exp(x, skips2))

    # Check transition constraints Q(x) = Z(x) * D(x)
    assert (p_of_g1x - p_of_x ** 3 - k_of_x - zvalue * d_of_x) % modulus == 0

    # Check boundary constraints B(x) * Z2(x) + I(x) = P(x)
    interpolant = f.lagrange_interp_2([1, last_step_position], [inp, output])
    zeropoly2 = f.mul_polys([-1, 1], [-last_step_position, 1])
    assert (p_of_x - b_of_x * f.eval_poly_at(zeropoly2, x) -
            f.eval_poly_at(interpolant, x)) % modulus == 0

    # Check correctness of the linear combination
    assert (l_of_x - d_of_x -
            k1 * p_of_x - k2 * p_of_x * x_to_the_steps -
            k3 * b_of_x - k4 * b_of_x * x_to_the_steps) % modulus == 0



```
