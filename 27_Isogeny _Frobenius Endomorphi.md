# ETAAcademy-ZKMeme: 27. Isogeny and Frobenius Endomorphi

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>27. Isogeny_FrobeniusEndomorphi</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Isogeny_FrobeniusEndomorphi</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

## Isogeny in Cryptography: Preserving Elliptic Curve Structure

When building cryptography resistant to quantum attacks, several approaches are prominent, including lattice-based, multivariate, and isogeny-based cryptography. Among them, homological mappings play a crucial role in preserving the structure of elliptic curves. These mappings act like bridges between elliptic curves, maintaining their key properties.

### Key Concepts:

- **Isogeny:** A special type of mapping between elliptic curves (E(K) and E'(K)) denoted by Φ, i.e., Φ: E(K) → E’(K). It ensures that every point in the output curve (E'(K)) has a corresponding point in the input curve (E(K)). Additionally, the infinity point of the input curve maps to the infinity point of the output curve (Φ(O) = O').

- **Standard Form:** A specific representation of a Isogeny, using radial coordinates, expresses the mapping as $Φ（x, y）= (\frac{u(x)}{v(x)}, \frac{s(x)}{t(x)}y)$, where $v^3(x)|t^2(x), t^2(x)|v^3(x)f(x)$ and $y^2 = x^3 +ax + b$， $f(x)=x^3 +ax + b$, P = (x, y ) ∈ E(K).

### properties of Isogeny:

- **Degree:** The maximum degree of the polynomials u(x) and v(x) determines the degree of the homology $deg(Φ) := max(deg(u(x), deg(v(x))))$.
- **Separability:** A Isogeny is separable if the derivative of $(\frac{u(x)}{v(x)})’$ is not zero, while Non-separable one means $(\frac{u(x)}{v(x)})’ = θ.$

### Special Cases:

- **Endomorphism:** A isogeny mapping an elliptic curve to itself. The most common endomorphism maps a point P to its nth multiple (Φ: [n], P → nP). Here, n = 0 maps all points to infinity (not a full homomorphism), n ≠ 0 is a homology, and n = ±1 is an automorphism.
- **Frobenius Endomorphism:** A specific automorphism denoted by π, which raises coordinates of points in the curve to a power based on the field characteristic $π: E(F_q) → E(F_q)$, $(X: Y: Z) → (X^q: Y^q: Z^q)$. The standard form (Φ(x, y)) shows: $y^2 = x^3 + ax + b, a, b ∈ F_q$ , $(x, y) ∈ F^2_q, π(x, y) = (x^q, (x^3 + ax +b)^{\frac{q-1}{2}}y)$; $u(x) = x^q, v(x) = e; s(x) = (x^3 + ax +b)^{\frac{q-1}{2}}$, t(x) = e;

### Decomposition of Isogeny:

Any Isogeny with a characteristic greater than 0 can be broken down into a combination of separable and inseparable homologies. This decomposition often involves repeated applications of the Frobenius map followed by a final separable homology.

- $Φ(x, y) = Φ(a(x^p), b(x^p)y^p)$;
- ⇒ $a_1(x, y) := (a(x), b(x)y)$, $Φ = a_1   ○π’$;
- ⇒ $a_1(x, y) = a_1(a_1(x^p), b_1(x^p)y^p)$, $a_1(x, y) := (a_1(x), b_1(x)y)$, $Φ = a_1   ○π’ = a_2○π’○π’$
- ⇒ $Φ =  a   ○(π’)^n$, $a := a_n$, $Φ =  a   ○(π’)^n$;

In essence, Isogeny offer a powerful tool for working with elliptic curves in cryptography while ensuring the preservation of their essential structure.
