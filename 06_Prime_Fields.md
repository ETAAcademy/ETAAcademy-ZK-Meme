# ETAAcademy-ZKMeme: 6. Prime Fields

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>06. Prime Fields</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Prime_fields</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

Finite fields are essential in cryptography because they work well with computers. To create one, start with a prime number p and define elements as {0, 1, ..., p-1}, using standard addition and multiplication operations modulo p. The extended Euclidean algorithm finds multiplicative inverses.

For STARKs, we need fields with a subgroup of order $2^k$, achieved with prime fields of the form $p = f \cdot 2^k + 1$. This structure ensures this subgroup of $2^k$ evenly spaced points on the complex unit circle within the field.

For example, a specific finite field with $1 + 407 \cdot 2^{119}$ elements has a sufficiently large subgroup of power-of-two order, which ensures the existence of this subgroup $2^k$. Additionally, it provides the user with generators for both the entire multiplicative group and the power-of-two subgroups. A generator for a subgroup of order n is referred to as a primitive nth root.

```python
    def generator( self ):
        assert(self.p == 1 + 407 * ( 1 << 119 )), "Do not know generator for other fields beyond 1+407*2^119"
        return FieldElement(85408008396924667383611388730472331217, self)

    def primitive_nth_root( self, n ):
        if self.p == 1 + 407 * ( 1 << 119 ):
            assert(n <= 1 << 119 and (n & (n-1)) == 0), "Field does not have nth root of unity where n > 2^119 or not power of two."
            root = FieldElement(85408008396924667383611388730472331217, self)
            order = 1 << 119
            while order != n:
                root = root^2
                order = order/2
            return root
        else:
            assert(False), "Unknown field, can't return root of unity."

```
