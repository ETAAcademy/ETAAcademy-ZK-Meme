# ETAAcademy-ZKMeme: 48. Binius

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>48. Binius</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Binius</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

# Advancements in Polynomial Techniques of Binius for Scalable and Secure Zero-Knowledge Proof Systems

One effective method to reduce computational complexity is to decrease the bit-width of the computation field (bit-width refers to the number of bits used to represent a number or variable). In zero-knowledge proofs, SNARKs typically use 256-bit fields, while STARKs often use smaller fields like Mersenne31, which is 32-bit. Binius takes this even further by reducing the bit-width to just 1 bit.

## 1. Extension Fields and Reducing Computation Complexity in Zero-Knowledge Proofs

In cryptographic systems, working within smaller finite fields can be computationally efficient but introduces potential security risks. This is where **field extension** becomes crucial; it enhances security by expanding the field size without compromising computational efficiency. Systems like Binius rely on binary fields, and these fields depend heavily on extension techniques to maintain practical usability.

Zero-knowledge proof systems like SNARKs and STARKs verify the correctness of computations through **arithmetization**: converting statements about programs into mathematical equations involving polynomials. Valid solutions to these equations correspond to correct program executions.

While technically STARKs are a type of SNARK, in practice, SNARKs usually refer to systems based on elliptic curves, whereas STARKs rely on hash functions. In STARKs, data is viewed as polynomials, and the values of these polynomials are computed at numerous points. The Merkle root of these extended data points serves as a **polynomial commitment** to ensure the authenticity of the data.

#### Verifying with Random Sampling

To verify the correctness of an arithmetic equation in STARKs and similar zero-knowledge proof systems, a common method is **random point sampling**. For instance, by choosing a random point $r$, the following equation might be checked:

$$
H(r) \times Z(r) \stackrel{?}{=} F(r+2) - F(r+1) - F(r)
$$

If an attacker can predict the random point $r$ in advance, they could potentially deceive the proof system. When the modulus (e.g., $2^{64} - 2^{32} + 1$) is small, an attacker might succeed by trying a few billion guesses. However, if the modulus is close to $2^{256}$, the large number of potential random points makes it nearly impossible for an attacker to guess correctly, thus ensuring security.

#### Introduction of Extension Fields

To enhance security, **Plonky2** introduces **extension fields**. For example, one can define an element $y$ such that $y^3 = 5$, then use combinations of 1, $y$, and $y^2$ to construct an extended field. This approach increases the total number of random points to approximately $2^{93}$, making it significantly harder for an attacker to guess the correct point.

In practice, most polynomial calculations still occur in a smaller field (e.g., $2^{31} - 1$). The transition to the extension field only happens during random point checks and Fast Reed-Solomon Interactive (FRI) proofs, ensuring sufficient security without sacrificing efficiency. Hence, even with the added layer of field extension, Plonky2 maintains high performance.

#### The Binius Solution

The Binius protocol tackles large-scale bit-level data computation by leveraging **multivariate polynomials** and a **hypercube structure**. It represents data efficiently and manages polynomial evaluation challenges and error correction code expansion. Binius uses a multivariate polynomial $F(x_1, x_2, \ldots, x_k)$ to represent data, mapping each bit to a point on the hypercube. For instance, if we want to represent the first 16 numbers in the Fibonacci sequence, we can denote them as $F(0,0,0,0)$ to $F(1,1,1,1)$.

In error correction, Binius treats the hypercube as a grid for coding expansion, allowing data redundancy checks even with a limited number of points. One approach in STARK protocols involves starting with an initial set of **n values** and then using **Reed-Solomon encoding** to expand this set to 8n values (typically within a range of 2n to 32n). This redundancy increases the chances of detecting and correcting errors during subsequent verification steps.

STARK then samples **Merkle branches** from these expanded values for verification. Due to the limited "space" in a hypercube for effective Merkle branch sampling, STARK employs a technique to transform the hypercube into a more manageable structure, ensuring accurate proof computations.

## 2. The Binius Protocol: Efficient Polynomial Commitment and Zero-Knowledge Proofs

The **Binius protocol** is designed for polynomial commitment, leveraging advanced mathematical and cryptographic techniques such as Reed-Solomon encoding, polynomial evaluation, and Merkle trees. Here’s a summary of how the Binius protocol operates:

#### **Hypercube to Square Mapping**

At the core of the Binius protocol is the concept of mapping a hypercube onto a square. This process begins by taking a 2D array (or square) as input. Each row is then treated as a polynomial and is expanded using **Reed-Solomon encoding**. For instance, a polynomial derived from an original row might be evaluated at $x = \{0, 1, 2, 3\}$ and then extended to additional points such as $x = \{4, 5, 6, 7\}$.

#### **Merkle Tree Commitment**

The extended data is treated as columns, and a **Merkle tree** is constructed from it. The Merkle root serves as a cryptographic commitment to the data. Merkle trees ensure data integrity and immutability, providing a secure mechanism for later verification without exposing the entire dataset.

#### **Zero-Knowledge Proofs**

The prover’s goal in the Binius protocol is to demonstrate the correctness of polynomial evaluations at specific points without revealing the polynomial itself. The prover sends some column data and partial calculations (like tensor products) to the verifier. The verifier uses the linear properties of Reed-Solomon encoding to check if these calculations align with the original commitment.

### **Tensor Product Calculations**

**Tensor product** is a mathematical operation used in the protocol to perform linear combinations in multi-dimensional data. Here, the prover divides the evaluation points into two parts: columns and rows. The tensor product allows the prover to verify computations without exposing the full data, maintaining privacy.

#### **Verification Process**

The verifier re-evaluates the encoded data and linear combinations to check whether the polynomial evaluations match the committed values. This way, the verifier confirms the correctness of the submitted data without needing direct access to the polynomial details.

In essence, the Binius protocol efficiently decomposes data into rows and columns, utilizing the properties of linear algebra and polynomials to achieve secure commitments and zero-knowledge proofs. One significant advantage of this approach is the reduction in the required field size (only half the field size is needed) while preserving data privacy and integrity.

#### Example: Tensor Product and Linear Combination

To understand the practical use of tensor products and linear combinations in this process, let's break down an example involving partial evaluations and compressed calculations.

1. **Column Tensor Product**

The tensor product for the column part is calculated as follows:

$$
\bigotimes_{i=0}^1 (1 - r_i, r_i)
$$

Here, $r_0$ and $r_1$ are a set of coordinate values. The tensor product generates all possible combinations of these elements:

$$
\text{Column Part} = [(1 - r_0)(1 - r_1), r_0(1 - r_1), (1 - r_0)r_1, r_0r_1]
$$

2. **Row Tensor Product**

Similarly, for the row part:

$$
\bigotimes_{i=2}^3 (1 - r_i, r_i)
$$

Given $r_2$ and $r_3$ as another set of coordinates, the row tensor product becomes:

$$
\text{Row Part} = [(1 - r_2)(1 - r_3), r_2(1 - r_3), (1 - r_2)r_3, r_2r_3]
$$

3. **Concrete Calculation Example**

Assume $r_2 = 3$ and $r_3 = 4$, then:

$$
\text{Row Part} = [(1 - 3)(1 - 4), 3(1 - 4), (1 - 3)4, 3 \times 4] = [6, -9, -8, 12]
$$

This results in a vector of 4 values representing all tensor product results for the row part.

4. **Computing a New "Row"**

Next, we perform a weighted linear combination of given row vectors:

$$
[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 8], [9, 7, 9, 3]
$$

Using the row part tensor product as weights, we compute:

$$
t' = [3, 1, 4, 1] \times 6 + [5, 9, 2, 6] \times (-9) + [5, 3, 5, 8] \times (-8) + [9, 7, 9, 3] \times 12
$$

This yields:

$$
t' = [41, -15, 74, -76]
$$

5. **Partial Evaluation and Compressed Calculation**

This example demonstrates **partial evaluation**, a process where only a subset of tensor product results is computed, optimizing the polynomial evaluation. By performing partial calculations, the protocol reduces the computational burden and improves efficiency. The partial result can then be shared, and the remaining computation is completed by others, leading to the final evaluation without direct access to all data points.

<details><summary><b> Code</b></summary>

<details><summary><b> Python</b></summary>

```python

# Treat `evals` as the evaluations of a multilinear polynomial over {0,1}^k.
# That is, if evals is [a,b,c,d], then a=P(0,0), b=P(1,0), c=P(0,1), d=P(1,1)
def multilinear_poly_eval(data, pt):
    if isinstance(pt, list):
        pt = np.array([
            int_to_bigbin(x) if isinstance(x, int) else x for x in pt
        ])
    assert log2(len(data) * 8) == pt.shape[0]
    evals_array = np.zeros((len(data) * 8, 8), dtype=np.uint16)
    evals_array[:,0] = bytestobits(data)
    for coord in reversed(pt):
        halflen = len(evals_array)//2
        top = evals_array[:halflen]
        bottom = evals_array[halflen:]
        evals_array = big_mul(bottom ^ top, coord) ^ top
    return evals_array[0]

# Returns the 2^k-long list of all possible results of walking through pt
# (an evaluation point) and at each step taking either coord or 1-coord.
# This is a natural companion method to `multilinear_poly_eval`, because
# it gives a list where `output[i]` equals
# `multilinear_poly_eval([0, 0 ... 1 ... 0, 0], pt)`, where the 1 is in
# position i.
def evaluation_tensor_product(pt):
    if isinstance(pt, list):
        pt = np.array([
            int_to_bigbin(x) if isinstance(x, int) else x for x in pt
        ])
    o = np.array([int_to_bigbin(1)])
    for coord in pt:
        o_times_coord = big_mul(o, coord)
        o = np.concatenate((
            o_times_coord ^ o,
            o_times_coord
        ))
    return o

```

</details>

<details><summary><b> Rust</b></summary>

```rust

impl<F, FE> TensorAlgebra<F, FE>
where
	F: Field,
	FE: ExtensionField<F>,
{
	/// Constructs an element from a vector of vertical subring elements.
	///
	/// ## Preconditions
	///
	/// * `elems` must have length `FE::DEGREE`, otherwise this will pad or truncate.
	pub fn new(mut elems: Vec<FE>) -> Self {
		elems.resize(FE::DEGREE, FE::ZERO);
		Self {
			elems,
			_marker: PhantomData,
		}
	}

	/// Returns $\kappa$, the base-2 logarithm of the extension degree.
	pub const fn kappa() -> usize {
		checked_log_2(FE::DEGREE)
	}

	/// Returns the byte size of an element.
	pub fn byte_size() -> usize {
		mem::size_of::<FE>() << Self::kappa()
	}

	/// Returns the multiplicative identity element, one.
	pub fn one() -> Self {
		let mut one = Self::default();
		one.elems[0] = FE::ONE;
		one
	}

	/// Returns a slice of the vertical subfield elements composing the tensor algebra element.
	pub fn vertical_elems(&self) -> &[FE] {
		&self.elems
	}

	/// Tensor product of a vertical subring element and a horizontal subring element.
	pub fn tensor(vertical: FE, horizontal: FE) -> Self {
		let elems = horizontal
			.iter_bases()
			.map(|base| vertical * base)
			.collect();
		Self {
			elems,
			_marker: PhantomData,
		}
	}

	/// Constructs a [`TensorAlgebra`] in the vertical subring.
	pub fn from_vertical(x: FE) -> Self {
		let mut elems = vec![FE::ZERO; FE::DEGREE];
		elems[0] = x;
		Self {
			elems,
			_marker: PhantomData,
		}
	}

	/// If the algebra element lives in the vertical subring, this returns it as a field element.
	pub fn try_extract_vertical(&self) -> Option<FE> {
		self.elems
			.iter()
			.skip(1)
			.all(|&elem| elem == FE::ZERO)
			.then_some(self.elems[0])
	}

	/// Multiply by an element from the vertical subring.
	pub fn scale_vertical(mut self, scalar: FE) -> Self {
		for elem_i in self.elems.iter_mut() {
			*elem_i *= scalar;
		}
		self
	}
}

```

</details>

</details>

## 3. Binary Fields and Tower Extensions

**Binary fields** are a type of finite field built using arithmetic modulo 2. We start with the simplest binary field, $F_2$, which consists of only two elements: 0 and 1. Here’s how addition and multiplication work in this field:

- **Addition (XOR operation):**

$$
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & 0 & 1 \\
1 & 1 & 0 \\
\end{array}
$$

- **Multiplication (AND operation):**

$$
\begin{array}{c|cc}
\times & 0 & 1 \\
\hline
0 & 0 & 0 \\
1 & 0 & 1 \\
\end{array}
$$

#### Extended Binary Fields

Starting from $F_2$, we can construct larger binary fields using extensions. To extend $F_2$, we introduce a new element $x$ that satisfies a specific polynomial relation, such as $x^2 = x + 1$. By adding this new element, we create an extended field with more elements.

In this extended field, the elements can be represented as $0, 1, x,$ and $x + 1$. The operations for addition and multiplication can be described by the following tables:

- **Addition (XOR operation):**

$$
\begin{array}{c|cccc}
 & 0 & 1 & x & x + 1 \\
\hline
0 & 0 & 1 & x & x + 1 \\
1 & 1 & 0 & x + 1 & x \\
x & x & x + 1 & 0 & 1 \\
x + 1 & x + 1 & x & 1 & 0 \\
\end{array}
$$

- **Multiplication:**

$$
\begin{array}{c|cccc}
\times & 0 & 1 & x & x + 1 \\
\hline
0 & 0 & 0 & 0 & 0 \\
1 & 0 & 1 & x & x + 1 \\
x & 0 & x & 1 & x + 1 \\
x + 1 & 0 & x + 1 & x + 1 & 1 \\
\end{array}
$$

#### Tower Construction

We can extend binary fields further using a method known as **tower construction**. Each extension layer acts like a new "floor" in a tower, where each new element satisfies a relation similar to $x^k = x^{k-1} + 1$. This recursive definition allows us to build increasingly large binary fields.

For example, consider a binary number like $1100101010001111$. We can interpret this as follows:

$$
1100101010001111 = 11001010 + 10001111 \cdot x^3
$$

Breaking this down further:

$$
11001010 = 1100 + 1010 \cdot x^2 + 1000 \cdot x^3 + 1111 \cdot x^2 x^3
$$

This representation allows for easy manipulation of elements and is particularly efficient for computer processing.

#### Operations in Binary Fields

Within binary fields, the basic arithmetic operations include:

1. **Addition (XOR operation):**

In binary fields, addition is performed using the XOR operation, where $x + x = 0$. This property makes addition very efficient, as it simply involves bitwise XOR between binary numbers.

2. **Multiplication:**

Multiplication in binary fields can be implemented using recursive algorithms. The process involves splitting the numbers into halves, performing multiplication on each part, and then combining the results. The multiplication also requires a **reduction** step, where if the result exceeds the field’s size, we subtract a specific polynomial to keep the result within the field.

For example, given $x$ and a specific irreducible polynomial $f(x)$, any product exceeding the degree of $f(x)$ is reduced by $f(x)$.

3. **Division:**

Division is performed using multiplication by the inverse. The inverse of an element $x$ in a binary field can be found using a method like the extended Euclidean algorithm or through **Fermat’s Little Theorem**. In binary fields, the inverse of $x$ is given by:

$$
x^{-1} = x^{2^{2k} - 2}
$$

Here, $2^{2k}$ represents the field size, and this formula leverages the properties of powers in finite fields to compute the multiplicative inverse.

#### Benefits of Binary Fields and Tower Extensions

The use of binary fields and their tower extensions provides several advantages:

- **Efficiency:** Operations like addition and multiplication are highly efficient because they can be reduced to bitwise operations. This makes them particularly well-suited for implementation in hardware and cryptographic algorithms.
- **Scalability:** The tower construction method allows for the creation of fields of any desired size. This flexibility is useful in applications requiring specific field sizes, such as cryptographic protocols and error-correcting codes.
- **Simplicity:** Binary fields simplify arithmetic operations since addition is just XOR and multiplication is relatively straightforward with the recursive structure. These properties are advantageous in areas like coding theory and cryptography, where efficient and reliable arithmetic is crucial.

<details><summary><b> Code</b></summary>

<details><summary><b> Python</b></summary>

```python

def additive_ntt(vals, start=0):
    vals = [B(val) for val in vals]
    if len(vals) == 1:
        return vals
    halflen = len(vals)//2
    L, R = vals[:halflen], vals[halflen:]
    coeff1 = get_Wi_eval(log2(halflen), start)
    sub_input1 = [i+j*coeff1 for i,j in zip(L, R)]
    sub_input2 = [i+j for i,j in zip(sub_input1, R)]
    o = (
        additive_ntt(sub_input1, start) +
        additive_ntt(sub_input2, start + halflen)
    )
    # print('for {} at {} used coeffs {}, {}; returning {}'.format(vals, start, coeff1, coeff2, o))
    return o

# Converts evaluations into coefficients (in the above basis) of a polynomial
def inv_additive_ntt(vals, start=0):
    vals = [B(val) for val in vals]
    if len(vals) == 1:
        return vals
    halflen = len(vals)//2
    L = inv_additive_ntt(vals[:halflen], start)
    R = inv_additive_ntt(vals[halflen:], start + halflen)
    coeff1 = get_Wi_eval(log2(halflen), start)
    coeff2 = coeff1 + 1
    o = (
        [i*coeff2+j*coeff1 for i,j in zip(L, R)] +
        [i+j for i,j in zip(L, R)]
    )
    # print('for {} at {} used coeffs {}, {}; returning {}'.format(vals, start, coeff1, coeff2, o))
    return o

# Reed-Solomon extension, using the efficient algorithms above
def extend(data, expansion_factor=2):
    data = [B(val) for val in data]
    return additive_ntt(
        inv_additive_ntt(data) +
        [B(0)] * len(data) * (expansion_factor - 1)
    )

```

</details>

<details><summary><b> Rust</b></summary>

```rust

pub trait TowerField: BinaryField
where
	Self: From<Self::Canonical>,
	Self::Canonical: From<Self>,
{
	/// The level $\iota$ in the tower, where this field is isomorphic to $T_{\iota}$.
	const TOWER_LEVEL: usize = Self::N_BITS.ilog2() as usize;

	/// The canonical field isomorphic to this tower field.
	/// Currently for every tower field, the canonical field is Fan-Paar's binary field of the same degree.
	type Canonical: TowerField + SerializeBytes + DeserializeBytes;

	fn basis(iota: usize, i: usize) -> Result<Self, Error> {
		if iota > Self::TOWER_LEVEL {
			return Err(Error::ExtensionDegreeTooHigh);
		}
		let n_basis_elts = 1 << (Self::TOWER_LEVEL - iota);
		if i >= n_basis_elts {
			return Err(Error::IndexOutOfRange {
				index: i,
				max: n_basis_elts,
			});
		}
		<Self as ExtensionField<BinaryField1b>>::basis(i << iota)
	}

	/// Multiplies a field element by the canonical primitive element of the extension $T_{\iota + 1} / T_{iota}$.
	///
	/// We represent the tower field $T_{\iota + 1}$ as a vector space over $T_{\iota}$ with the basis $\{1, \beta^{(\iota)}_1\}$.
	/// This operation multiplies the element by $\beta^{(\iota)}_1$.
	///
	/// ## Throws
	///
	/// * `Error::ExtensionDegreeTooHigh` if `iota >= Self::TOWER_LEVEL`
	fn mul_primitive(self, iota: usize) -> Result<Self, Error> {
		Ok(self * <Self as ExtensionField<BinaryField1b>>::basis(1 << iota)?)
	}

	binary_tower!(
	BinaryField1b(U1)
	< BinaryField2b(U2)
	< BinaryField4b(U4)
	< BinaryField8b(u8)
	< BinaryField16b(u16)
	< BinaryField32b(u32)
	< BinaryField64b(u64)
	< BinaryField128b(u128)
);

macro_rules! serialize_deserialize {
	($bin_type:ty, SmallU<$U:literal>) => {
		impl SerializeBytes for $bin_type {
			fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
				if write_buf.remaining_mut() < 1 {
					return Err(SerializationError::WriteBufferFull);
				}
				let b = self.0.val();
				write_buf.put_u8(b);
				Ok(())
			}
		}

		impl DeserializeBytes for $bin_type {
			fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
				if read_buf.remaining() < 1 {
					return Err(SerializationError::NotEnoughBytes);
				}
				let b: u8 = read_buf.get_u8();
				Ok(Self(SmallU::<$U>::new(b)))
			}
		}
	};
	($bin_type:ty, $inner_type:ty) => {
		impl SerializeBytes for $bin_type {
			fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
				if write_buf.remaining_mut() < (<$inner_type>::BITS / 8) as usize {
					return Err(SerializationError::WriteBufferFull);
				}
				write_buf.put_slice(&self.0.to_le_bytes());
				Ok(())
			}
		}

		impl DeserializeBytes for $bin_type {
			fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
				let mut inner = <$inner_type>::default().to_le_bytes();
				if read_buf.remaining() < inner.len() {
					return Err(SerializationError::NotEnoughBytes);
				}
				read_buf.copy_to_slice(&mut inner);
				Ok(Self(<$inner_type>::from_le_bytes(inner)))
			}
		}
	};
}

```

</details>

</details>

## 4. Small Domain Polynomial Commitment Scheme

The **small domain polynomial commitment scheme** is a cryptographic method for committing to a polynomial and later proving statements about it without revealing the polynomial itself. This scheme leverages **Lagrange basis polynomials** and the **Schwartz-Zippel theorem** to verify properties of polynomials.

#### Lagrange Basis Polynomials and Schwartz-Zippel Theorem

The **Lagrange basis polynomials** are used to represent a polynomial in a standard form, making it easier to analyze its structure and properties. In this scheme, we work with polynomials in the ring $T_{\tau}[R_{\ell1}, \ldots, R_{\ell-2}]$, where the polynomials are expressed using Lagrange basis. At least one of these combinations will have a non-zero coefficient vector.

The **Schwartz-Zippel theorem** helps estimate the number of zeros a polynomial might have, providing a way to check certain properties of the polynomial. Applying the theorem to all $2^{\kappa}$ possible positions allows us to limit the maximum value that the polynomial can achieve at any given point. Thus, it helps in proving that at least one specific position satisfies a predefined constraint.

By analyzing the polynomial across these positions, we can argue that $\mu(R_{b,j}) \le \ell_0^{-1} \cdot |T_{\tau}|$, which is a key part of ensuring the polynomial commitment's correctness.

#### Construction Steps

The construction of the polynomial commitment scheme involves several steps, including setup, commit, open, proof, and verification.

1. **Setup (Key Generation)**

During this phase, parameters for the polynomial commitment are generated. These parameters include polynomial dimensions, matrix sizes, encoding methods, and other variables needed for the cryptographic encoding. The setup process can be described as follows:

- Parameters are chosen as $params \leftarrow \Pi.Setup(1^{\lambda}, \ell, \iota)$.
- Given the inputs $1^{\lambda}, \ell, \iota$, integers $\ell_0$ and $\ell_1$ are selected such that $\ell_0 + \ell_1 = \ell$.
- The sizes $m_0 := 2^{\ell_0}$ and $m_1 := 2^{\ell_1}$ are defined.
- Additional parameters such as $\kappa$, tower height $\tau$, and a matrix code $C \subset T^{n\iota + \kappa}$ are chosen with $n = 2^{O(\ell)}$ and $d = \Omega(n)$.
- A repetition parameter $\gamma = \Theta(\lambda)$ is also set.

2. **Commit**

In the commitment phase, the polynomial is converted into a matrix form using the Lagrange basis, followed by encoding each row of the matrix to produce a Merkle commitment (similar to a hash tree). The steps are as follows:

- The input polynomial $t(X_0, \ldots, X_{\ell-1})$ is represented as $(t_0, \ldots, t_{2^{\ell}-1})$.
- Using the Lagrange basis, the polynomial is arranged into a $m_0 \times m_1$ matrix.
- Column indices are grouped into blocks of size $2^{\kappa}$, resulting in a $m_0 \times \frac{m_1}{2^{\kappa}}$ matrix.
- An encoding function is applied to each row, producing an $m_0 \times n$ matrix.
- The Merkle commitment $c$ and the opening hint $u$ (the matrix itself) are generated: $(c, u) \leftarrow \Pi.Commit(params, t)$.

3. **Open**

The opening phase checks the integrity of the Merkle commitment, verifying if it corresponds to a specific polynomial. The verification process involves:

- The prover sends the committed polynomial and the Merkle proof paths to the verifier.
- The verifier checks the Merkle paths and verifies that the commitment corresponds to the polynomial.

The verification step can be formalized as $b \leftarrow \Pi.Open(params, c; t, u)$.

4. **Proof and Verification Protocol**

An interactive proof protocol is defined to provide the verifier with confidence about the committed polynomial's properties. Using the **Fiat–Shamir transform**, this interactive protocol is made non-interactive, suitable for a public environment.

In this protocol:

- The **prover** computes matrix-vector products and generates Merkle proof paths.
- The **verifier** samples random positions to check the data's integrity.

The detailed steps are:

- The prover computes $t'$ as a matrix-vector product.
- The verifier selects random indices $j_i$ and sends the index set $J$ to the prover.
- The prover provides the corresponding columns and Merkle paths for the verifier to check.
- The verifier performs a consistency check, ensuring the integrity of the encoded vectors.

Finally, the verifier ensures that:

1. $s = t' \cdot N_{\ell_1-1}^{\ell_1} (1 - r_i, r_i)$ is satisfied, where $s$ is the expected value.
2. The verifier checks the Merkle proofs and ensures that $N_{\ell-1}^{\ell_1} (1 - r_i, r_i) \cdot (u_{i,j})_{i=0}^{m_0-1} = u'_j$.

#### Key features

This **small domain polynomial commitment scheme** is a robust cryptographic method with the following key features:

- **Efficiency:** The scheme is designed for efficient verification, leveraging properties of Lagrange basis polynomials and Merkle trees to minimize computational overhead.
- **Security:** Using the Schwartz-Zippel theorem and Fiat–Shamir transformation, the scheme ensures strong guarantees about the committed polynomial's integrity without revealing its content.
- **Scalability:** The construction is flexible and can be scaled to accommodate different polynomial sizes and complexities, making it suitable for various cryptographic applications like zero-knowledge proofs and verifiable computation.

Overall, this scheme provides a secure, scalable, and efficient way to commit to polynomials and later prove properties about them, making it an essential tool in modern cryptographic protocols.

<details><summary><b> Code</b></summary>

<details><summary><b> Python</b></summary>

```python

# "Block-level encoding-based polynomial commitment scheme", section 3.11 of
# https://eprint.iacr.org/2023/1784.pdf
#
# An optimized implementation that tries to go as fast as you can in python,
# using numpy

def optimized_binius_proof(evaluations, evaluation_point):
    # Rearrange evaluations into a row_length * row_count grid
    assert len(evaluation_point) == log2(len(evaluations) * 8)
    log_row_length, log_row_count, row_length, row_count = \
        choose_row_length_and_count(log2(len(evaluations) * 8))
    # Directly treat the rows as a list of uint16's
    rows = (
        np.frombuffer(evaluations, dtype='<u2')
        .reshape((row_count, row_length // PACKING_FACTOR))
    )
    # Fast-Fourier extend the rows
    extended_rows = extend(rows, EXPANSION_FACTOR)
    extended_row_length = row_length * EXPANSION_FACTOR // PACKING_FACTOR

    # Compute t_prime, a linear combination of the rows
    # The linear combination is carefully chosen so that the evaluation of the
    # multilinear polynomial at the `evaluation_point` is itself either on
    # t_prime, or a linear combination of elements of t_prime
    row_combination = \
        evaluation_tensor_product(evaluation_point[log_row_length:])
    assert len(row_combination) == len(rows) == row_count
    rows_as_bits_transpose = uint16s_to_bits(rows).transpose()
    t_prime = np.zeros(
        (rows_as_bits_transpose.shape[0], row_combination.shape[1]),
        dtype=np.uint16
    )
    for j in range(row_combination.shape[1]):
        t_prime[:,j:j+1] ^= xor_along_axis(
            (
                rows_as_bits_transpose[:,:,np.newaxis] *
                row_combination[np.newaxis,:,j:j+1]
            ), 1
        )
    #t_prime = multisubset(row_combination, rows_as_bits_transpose)

    # Pack columns into a Merkle tree, to commit to them
    columns = np.transpose(extended_rows)
    bytes_per_element = PACKING_FACTOR//8
    packed_columns = [col.tobytes('C') for col in columns]
    merkle_tree = merkelize(packed_columns)
    root = get_root(merkle_tree)

    # Challenge in a few positions, to get branches
    challenges = get_challenges(root, extended_row_length)

    # Compute evaluation. Note that this is much faster than computing it
    # "directly"
    col_combination = \
        evaluation_tensor_product(evaluation_point[:log_row_length])
    computed_eval = xor_along_axis(
        big_mul(t_prime, col_combination),
        0
    )
    return {
        'root': root,
        'evaluation_point': evaluation_point,
        'eval': computed_eval,
        't_prime': t_prime,
        'columns': columns[challenges],
        'branches': [get_branch(merkle_tree, c) for c in challenges]
    }

def verify_optimized_binius_proof(proof):
    columns, evaluation_point, value, t_prime, root, branches = (
        proof['columns'],
        proof['evaluation_point'],
        proof['eval'],
        proof['t_prime'],
        proof['root'],
        proof['branches'],
    )

    # Compute the row length and row count of the grid. Should output same
    # numbers as what prover gave
    log_row_length, log_row_count, row_length, row_count = \
        choose_row_length_and_count(len(evaluation_point))
    extended_row_length = row_length * EXPANSION_FACTOR // PACKING_FACTOR

    # Compute challenges. Should output the same as what prover computed
    challenges = get_challenges(root, extended_row_length)

    # Verify the correctness of the Merkle branches
    bytes_per_element = PACKING_FACTOR//8
    for challenge, branch, col in zip(challenges + 0, branches, columns):
        packed_column = col.tobytes('C')
        print(f"Verifying Merkle branch for column {challenge}")
        assert verify_branch(root, challenge, packed_column, branch)


    # Use the same Reed-Solomon code that the prover used to extend the rows,
    # but to extend t_prime. We do this separately for each bit of t_prime
    t_prime_bit_length = 128
    # Each t_prime, as a 128-bit bitarray
    t_prime_bits = uint16s_to_bits(t_prime)
    # Treat this as 128 bit-rows, and re-pack those
    rows = bits_to_uint16s(np.transpose(t_prime_bits))
    # And FFT-extend that re-packing
    extended_slices = extend(rows, EXPANSION_FACTOR)

    # Here, we take advantage of the linearity of the code. A linear combination
    # of the Reed-Solomon extension gives the same result as an extension of the
    # linear combination.
    row_combination = \
        evaluation_tensor_product(evaluation_point[log_row_length:])
    # Each column is a vector of row_count uint16's. Convert each uint16 into
    # bits
    column_bits = uint16s_to_bits(columns[..., np.newaxis])
    # Take the same linear combination the prover used to compute t_prime, and
    # apply it to the columns of bits. The transpose ensures that we get a 32*16
    # matrix of bit-columns
    computed_tprimes = multisubset(
        row_combination,
        np.transpose(column_bits, (0,2,1))
    )
    # Convert our FFT-extended t_prime rows (or rather, the 32 uint16s at the
    # column positions) into bits

    extended_slices_bits = uint16s_to_bits(
        extended_slices[:, challenges, np.newaxis]
    )
    # The bits of the t_prime extension should equal the bits of the row linear
    # combination of the column bits
    assert np.array_equal(
        computed_tprimes,
        bits_to_uint16s(np.transpose(extended_slices_bits, (1, 2, 0)))
    )
    print("T_prime matches linear combinations of Merkle branches")

    # Take the right linear combination of elements *within* t_prime to
    # extract the evaluation of the original multilinear polynomial at
    # the desired point
    col_combination = \
        evaluation_tensor_product(evaluation_point[:log_row_length])
    computed_eval = xor_along_axis(
        big_mul(t_prime, col_combination),
        0
    )
    print(f"Testing evaluation: expected {value} computed {computed_eval}")
    assert np.array_equal(computed_eval, value)
    return True


```

</details>

<details><summary><b> Rust</b></summary>

```rust

impl<U, F, FA, FI, FE, LC, H, VCS> PolyCommitScheme<PackedType<U, F>, FE>
	for TensorPCS<U, F, FA, FI, FE, LC, H, VCS>
where
	U: PackScalar<F>
		+ PackScalar<FA>
		+ PackScalar<FI, Packed: PackedFieldIndexable>
		+ PackScalar<FE, Packed: PackedFieldIndexable>,
	F: Field,
	FA: Field,
	FI: ExtensionField<F> + ExtensionField<FA>,
	FE: ExtensionField<F> + ExtensionField<FI> + TowerField,
	LC: LinearCode<P = PackedType<U, FA>> + Sync,
	H: HashDigest<PackedType<U, FI>> + Sync,
	H::Digest: Copy + Default + Send,
	VCS: VectorCommitScheme<H::Digest> + Sync,
{
	type Commitment = VCS::Commitment;
	type Committed = (Vec<RowMajorMatrix<PackedType<U, FI>>>, VCS::Committed);
	type Proof = Proof<U, FI, FE, VCS::Proof>;
	type Error = Error;

	fn n_vars(&self) -> usize {
		self.log_rows() + self.log_cols()
	}

	#[instrument(skip_all, name = "tensor_pcs::commit", level = "debug")]
	fn commit<Data>(
		&self,
		polys: &[MultilinearExtension<PackedType<U, F>, Data>],
	) -> Result<(Self::Commitment, Self::Committed), Error>
	where
		Data: Deref<Target = [PackedType<U, F>]> + Send + Sync,
	{
		for poly in polys {
			if poly.n_vars() != self.n_vars() {
				bail!(Error::IncorrectPolynomialSize {
					expected: self.n_vars(),
				});
			}
		}

		// These conditions are checked by the constructor, so are safe to assert defensively
		let pi_width = PackedType::<U, FI>::WIDTH;
		debug_assert_eq!(self.code.dim() % pi_width, 0);

		// Dimensions as an intermediate field matrix.
		let n_rows = 1 << self.log_rows;
		let n_cols_enc = self.code.len();

		let results = polys
			.par_iter()
			.map(|poly| -> Result<_, Error> {
				let mut encoded =
					vec![PackedType::<U, FI>::default(); n_rows * n_cols_enc / pi_width];
				let poly_vals_packed =
					<PackedType<U, FI> as PackedExtension<F>>::cast_exts(poly.evals());

				transpose::transpose(
					PackedType::<U, FI>::unpack_scalars(poly_vals_packed),
					PackedType::<U, FI>::unpack_scalars_mut(
						&mut encoded[..n_rows * self.code.dim() / pi_width],
					),
					1 << self.code.dim_bits(),
					1 << self.log_rows,
				);

				self.code
					.encode_batch_inplace(
						<PackedType<U, FI> as PackedExtension<FA>>::cast_bases_mut(&mut encoded),
						self.log_rows + log2_strict_usize(<FI as ExtensionField<FA>>::DEGREE),
					)
					.map_err(|err| Error::EncodeError(Box::new(err)))?;

				let mut digests = vec![H::Digest::default(); n_cols_enc];
				encoded
					.par_chunks_exact(n_rows / pi_width)
					.map(H::hash)
					.collect_into_vec(&mut digests);

				let encoded_mat = RowMajorMatrix::new(encoded, n_rows / pi_width);

				Ok((digests, encoded_mat))
			})
			.collect::<Vec<_>>();

		let mut encoded_mats = Vec::with_capacity(polys.len());
		let mut all_digests = Vec::with_capacity(polys.len());
		for result in results {
			let (digests, encoded_mat) = result?;
			all_digests.push(digests);
			encoded_mats.push(encoded_mat);
		}

		let (commitment, vcs_committed) = self
			.vcs
			.commit_batch(&all_digests)
			.map_err(|err| Error::VectorCommit(Box::new(err)))?;
		Ok((commitment, (encoded_mats, vcs_committed)))
	}

	#[instrument(skip_all, name = "tensor_pcs::prove_evaluation", level = "debug")]
	fn prove_evaluation<Data, CH, Backend>(
		&self,
		challenger: &mut CH,
		committed: &Self::Committed,
		polys: &[MultilinearExtension<PackedType<U, F>, Data>],
		query: &[FE],
		backend: &Backend,
	) -> Result<Self::Proof, Error>
	where
		Data: Deref<Target = [PackedType<U, F>]> + Send + Sync,
		CH: CanObserve<FE> + CanSample<FE> + CanSampleBits<usize>,
		Backend: ComputationBackend,
	{
		let n_polys = polys.len();
		let n_challenges = log2_ceil_usize(n_polys);
		let mixing_challenges = challenger.sample_vec(n_challenges);
		let mixing_coefficients =
			&backend.tensor_product_full_query(&mixing_challenges)?[..n_polys];

		let (col_major_mats, ref vcs_committed) = committed;
		if col_major_mats.len() != n_polys {
			bail!(Error::NumBatchedMismatchError {
				err_str: format!("In prove_evaluation: number of polynomials {} must match number of committed matrices {}", n_polys, col_major_mats.len()),
			});
		}

		if query.len() != self.n_vars() {
			bail!(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars(),
			});
		}

		let code_len_bits = log2_strict_usize(self.code.len());
		let log_block_size = log2_strict_usize(<FI as ExtensionField<F>>::DEGREE);
		let log_n_cols = self.code.dim_bits() + log_block_size;

		let partial_query = backend.multilinear_query(&query[log_n_cols..])?;
		let ts = polys;
		let t_primes = ts
			.iter()
			.map(|t| t.evaluate_partial_high(&partial_query))
			.collect::<Result<Vec<_>, _>>()?;
		let t_prime = mix_t_primes(log_n_cols, &t_primes, mixing_coefficients)?;

		challenger.observe_slice(PackedType::<U, FE>::unpack_scalars(t_prime.evals()));
		let merkle_proofs = repeat_with(|| challenger.sample_bits(code_len_bits))
			.take(self.n_test_queries)
			.map(|index| {
				let vcs_proof = self
					.vcs
					.prove_batch_opening(vcs_committed, index)
					.map_err(|err| Error::VectorCommit(Box::new(err)))?;

				let cols: Vec<_> = col_major_mats
					.iter()
					.map(|col_major_mat| col_major_mat.row_slice(index).to_vec())
					.collect();

				Ok((cols, vcs_proof))
			})
			.collect::<Result<_, Error>>()?;

		Ok(Proof {
			n_polys,
			mixed_t_prime: t_prime,
			vcs_proofs: merkle_proofs,
		})
	}

	#[instrument(skip_all, name = "tensor_pcs::verify_evaluation", level = "debug")]
	fn verify_evaluation<CH, Backend>(
		&self,
		challenger: &mut CH,
		commitment: &Self::Commitment,
		query: &[FE],
		proof: Self::Proof,
		values: &[FE],
		backend: &Backend,
	) -> Result<(), Error>
	where
		CH: CanObserve<FE> + CanSample<FE> + CanSampleBits<usize>,
		Backend: ComputationBackend,
	{
		// These are all checked during construction, so it is safe to assert as a defensive
		// measure.
		let p_width = PackedType::<U, F>::WIDTH;
		let pi_width = PackedType::<U, FI>::WIDTH;
		let pe_width = PackedType::<U, FE>::WIDTH;
		debug_assert_eq!(self.code.dim() % pi_width, 0);
		debug_assert_eq!((1 << self.log_rows) % pi_width, 0);
		debug_assert_eq!(self.code.dim() % pi_width, 0);
		debug_assert_eq!(self.code.dim() % pe_width, 0);

		if values.len() != proof.n_polys {
			bail!(Error::NumBatchedMismatchError {
				err_str:
					format!("In verify_evaluation: proof number of polynomials {} must match number of opened values {}", proof.n_polys, values.len()),
			});
		}

		let n_challenges = log2_ceil_usize(proof.n_polys);
		let mixing_challenges = challenger.sample_vec(n_challenges);
		let mixing_coefficients = &backend
			.tensor_product_full_query::<PackedType<U, FE>>(&mixing_challenges)?[..proof.n_polys];
		let value =
			inner_product_unchecked(values.iter().copied(), iter_packed_slice(mixing_coefficients));

		if query.len() != self.n_vars() {
			bail!(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars(),
			});
		}

		self.check_proof_shape(&proof)?;

		// Code length is checked to be a power of two in the constructor
		let code_len_bits = log2_strict_usize(self.code.len());
		let block_size = <FI as ExtensionField<F>>::DEGREE;
		let log_block_size = log2_strict_usize(block_size);
		let log_n_cols = self.code.dim_bits() + log_block_size;

		let n_rows = 1 << self.log_rows;

		challenger.observe_slice(<PackedType<U, FE>>::unpack_scalars(proof.mixed_t_prime.evals()));

		// Check evaluation of t' matches the claimed value
		let multilin_query =
			backend.multilinear_query::<PackedType<U, FE>>(&query[..log_n_cols])?;
		let computed_value = proof
			.mixed_t_prime
			.evaluate(&multilin_query)
			.expect("query is the correct size by check_proof_shape checks");
		if computed_value != value {
			return Err(VerificationError::IncorrectEvaluation.into());
		}

		// Encode t' into u'
		let mut u_prime = vec![
			PackedType::<U, FE>::default();
			(1 << (code_len_bits + log_block_size)) / pe_width
		];
		self.encode_ext(proof.mixed_t_prime.evals(), &mut u_prime)?;

		// Check vector commitment openings.
		let columns = proof
			.vcs_proofs
			.into_iter()
			.map(|(cols, vcs_proof)| {
				let index = challenger.sample_bits(code_len_bits);

				let leaf_digests = cols.iter().map(H::hash);

				self.vcs
					.verify_batch_opening(commitment, index, vcs_proof, leaf_digests)
					.map_err(|err| Error::VectorCommit(Box::new(err)))?;

				Ok((index, cols))
			})
			.collect::<Result<Vec<_>, Error>>()?;

		// Get the sequence of column tests.
		let column_tests = columns
			.into_iter()
			.flat_map(|(index, cols)| {
				let mut batched_column_test = (0..block_size)
					.map(|j| {
						let u_prime_i = get_packed_slice(&u_prime, index << log_block_size | j);
						let base_cols = Vec::with_capacity(proof.n_polys);
						(u_prime_i, base_cols)
					})
					.collect::<Vec<_>>();

				for mut col in cols {
					// Checked by check_proof_shape
					debug_assert_eq!(col.len(), n_rows / pi_width);

					// Pad column with empty elements to accommodate the following scalar transpose.
					if n_rows < p_width {
						col.resize(p_width, Default::default());
					}

					// The columns are committed to and provided by the prover as packed vectors of
					// intermediate field elements. We need to transpose them into packed base field
					// elements to perform the consistency checks. Allocate col_transposed as packed
					// intermediate field elements to guarantee alignment.
					let mut col_transposed = col.clone();
					let base_cols = <PackedType<U, FI> as PackedExtension<F>>::cast_bases_mut(
						&mut col_transposed,
					);
					transpose_scalars(&col, base_cols).expect(
						"guaranteed safe because of parameter checks in constructor; \
							alignment is guaranteed the cast from a PI slice",
					);

					for (j, col) in base_cols
						.chunks_exact(base_cols.len() / block_size)
						.enumerate()
					{
						// Trim off padding rows by converting from packed vec to scalar vec.
						let scalars_col = iter_packed_slice(col).take(n_rows).collect::<Vec<_>>();
						batched_column_test[j].1.push(scalars_col);
					}
				}
				batched_column_test
			})
			.collect::<Vec<_>>();

		// Batch evaluate all opened columns
		let multilin_query =
			backend.multilinear_query::<PackedType<U, FE>>(&query[log_n_cols..])?;
		let incorrect_evaluation = column_tests
			.par_iter()
			.map(|(expected, leaves)| {
				let actual_evals =
					leaves
						.par_iter()
						.map(|leaf| {
							MultilinearExtension::from_values_slice(leaf)
						.expect("leaf is guaranteed power of two length due to check_proof_shape")
						.evaluate(&multilin_query)
						.expect("failed to evaluate")
						})
						.collect::<Vec<_>>();
				(expected, actual_evals)
			})
			.any(|(expected_result, unmixed_actual_results)| {
				// Check that opened column evaluations match u'
				let actual_result = inner_product_unchecked(
					unmixed_actual_results.into_iter(),
					iter_packed_slice(mixing_coefficients),
				);
				actual_result != *expected_result
			});

		if incorrect_evaluation {
			return Err(VerificationError::IncorrectPartialEvaluation.into());
		} else {
			Ok(())
		}
	}

```

</details>

</details>

## 5. Polynomial IOP and Plonkish Protocols: Enhancements and Adaptations in Binary Tower Fields

Several interactive protocols and polynomial IOPs (Interactive Oracle Proofs) could be specialized for a binary tower field setting, which introduces a practical SNARK (Succinct Non-interactive Argument of Knowledge) and its arithmetic scheme, extending the traditional PLONK system to support multivariate constraints, particularly multivariable gate constraints and global copy constraints. While some parts of these protocols adapt known techniques, the tower structure provides additional performance improvements.

#### Polynomial Interactive Oracle Proof (IOP)

A **Polynomial IOP** $Π = (I, P, V)$ is an interactive protocol that enables participants to use a polynomial oracle. This oracle can handle submission and query requests, returning corresponding results. When it receives a submission request, the oracle generates a unique handle to represent the polynomial. For query requests, the oracle evaluates the polynomial and returns the result. The security of a Polynomial IOP requires an expected PPT (Probabilistic Polynomial Time) simulator $E$ and a negligible function $negl$, ensuring that the success probability of the protocol is negligibly close to the success probability of relation $R$.

**Polynomial predicates** include queries, summation, zero-checking, product, multiset, permutation, and lookup, which are used for specific computations and validations within the protocol. Each predicate $Query(r, s)_{ι, ℓ}$ can be directly evaluated by the verifier through a single query to the polynomial oracle.

**Polynomial Oracle**

Consider a fixed tower height $\tau := \tau(λ)$ and a binary tower $T_0 \subset T_1 \subset \cdots \subset T_\tau$.

- When $I$ or $P$ inputs $(\text{submit}, ι, ℓ, t)$, where $ι \in \{0, \ldots, \tau\}$, $ℓ \in \mathbb{N}$, and $t \in T_ι[X_0, \ldots, X_{ℓ-1}]_{\leq 1}$, it outputs $(\text{receipt}, ι, ℓ, [t])$ to $I$, $P$, and $V$, where $[t]$ is a unique handle for polynomial $t$.
- When $V$ inputs $(\text{query}, [t], r)$, where $r \in T_\tau^\ell$, it responds with $(\text{evaluation}, t(r_0, \ldots, r_{ℓ-1}))$.

#### Virtual polynomials

**Virtual polynomials** are constructed from multiple polynomial handles and an arithmetic circuit, where leaf nodes can be variables or constants. These virtual polynomials can be evaluated at any input, although this may not always be efficient. A **virtual polynomial protocol** handles inputs of multiple virtual polynomials and ensures their security. **Composed virtual polynomials** are generated by replacing certain handles with other virtual polynomials, forming new virtual polynomials.

**Common Virtual Polynomial Protocols**

1. **Sumcheck Protocol**: This protocol verifies whether a polynomial $T$ sums to a given value $e$ over a set $B^ℓ$. It internally invokes the implicit query protocol of $T$ to ensure security. The soundness error is bounded by $ℓ \cdot d / |T_\tau|$, where $d$ is the maximum univariate degree of $T$.
2. **Zerocheck Protocol**: This verifies if the polynomial $T$ evaluates to zero over the set $B^ℓ$. The protocol employs random sampling and the Sumcheck protocol for security. The soundness error depends on the combined use of Zerocheck and Sumcheck.

3. **Product Check Protocol**: This protocol checks the product relationship between two polynomials $T$ and $U$ over $B^ℓ$. It constructs an auxiliary function $f$ and merges operations to form a virtual polynomial $[T']$. Its security is ensured by the results of the Zerocheck protocol.

4. **Multiset Check Protocol**: This verifies the equality of multisets of several polynomials. It reduces the multiset check to the product check, relying on the previously defined product check for security.

5. **Permutation Check Protocol**: This ensures equality of a list of polynomials under a specific permutation. The protocol constructs virtual polynomials and performs a multiset check, ensuring consistent values across all polynomials under the given permutation.

**New Virtual Polynomial Constructions**

New constructions like packing, shifting, and saturation provide efficient tools for handling complex polynomial computations across various dimensions and structures.

1. **Packing Construction**: The packing operation `packκ` reduces a polynomial $t$ from dimension $ℓ$ to a lower dimension $ℓ-κ$. It processes adjacent $2κ$ elements, combining them into a new element.
   **Formula**: $\text{pack}_κ(t) := \sum t(v \parallel u) \cdot β_v$, where $u$ is a variable in the lower dimension $ℓ-κ$, and $v \in B_κ$. The resulting polynomial is multilinear and can be evaluated efficiently.

3. **Shifting Construction**: The shifting operation `shiftb,o` divides the index set of polynomial $t$ into $b$-dimensional sub-cubes and cyclically shifts each sub-array.
   **Formula**: $shift_{b,o}(t) := (t(s_{b,o}(v)))$, where $s_{b,o}$ defines the shift, $v \in B_ℓ$. The shifting indicator function `s-indb,o` checks for the shift relation between inputs. The resulting polynomial is multilinear and supports efficient evaluation.
   

5. **Saturation Construction**: The saturation operation `satb,o` divides the index set of polynomial $t$ into $b$-dimensional sub-cubes and "saturates" each block with a specific value.
   **Formula**: $sat_{b,o}(t)(v) = t(o_0, \ldots, o_{b-1}, v_b, \ldots, v_{ℓ-1})$. The resulting polynomial is multilinear and allows for efficient evaluation.

These new virtual polynomial constructions provide robust tools for handling complex polynomial computations efficiently, offering foundational support for subsequent polynomial algorithms and enhancing their performance across different dimensions and structures.

#### Lasso and Lookup Protocols in ZK-SNARKs

The **Lasso protocol** is a lookup technique designed to optimize set membership checks in ZK-SNARKs by utilizing virtual polynomials and multiset checks. This protocol simulates large table operations efficiently, adapting them to polynomial interactive proofs (IOP) frameworks.

**Components of Lasso Protocol**

1. **Virtual Polynomials for Large Table Operations**:
   The core of Lasso relies on virtual polynomials to simulate operations on large datasets. These polynomials serve as efficient representations of large tables in the protocol, enabling scalable lookups without explicitly handling massive datasets.

2. **Lookup Process**:
   The lookup protocol is used to determine whether specific values exist within a given set. Instead of performing traditional search operations, Lasso compresses the lookup process into a multiset assertion, drastically improving efficiency.

3. **Multiset Check**:
   The backbone of Lasso's security is the multiset check, which ensures that two virtual polynomials are equivalent within the polynomial protocol. This ensures the correctness of the lookup without requiring direct enumeration of elements.

**Adapting Lasso to Binary Fields**

The original Lasso protocol was designed for fields with large characteristic. In binary fields, where the characteristic is small (often 2), arithmetic operations behave differently. For example, addition becomes equivalent to XOR, making it necessary to use **multiplicative generators** for accurate memory indexing. The protocol adapts by using multiplication instead of addition to increment memory counters, ensuring correctness in binary fields.

Lasso Lookup Protocol Steps:

1. **Parameter Setup**:

   - Define polynomial variable count $\ell$ and virtual polynomial domain $T$.
   - Set integers $ζ$ and a generator $α$. $ζ$ is chosen such that $|T_ζ| - 1 > 2\ell$, ensuring enough elements for lookup purposes.
   - $α$ is a generator of the multiplicative group $T^* ζ$.

2. **Array Initialization**:

   - The prover $P$ initializes arrays $R$ and $F$ within $T B^ℓ_ζ$.
   - $F$ is initialized as a vector filled with ones.

3. **Lookup Process**:

   - For each $v \in B^\ell$, $P$ randomly selects $v' \in B^\ell$ such that $U(v) = T(v')$.
   - $R[v]$ is assigned the value $F[v']$, and then $F[v']$ is multiplied by the generator $α$.
   - This process encodes the lookup by updating $R$ and $F$ arrays.

4. **Reciprocal Calculation and Zero Check**:

   - $P$ computes the reciprocal array $R'$ such that $R' = 1/R(v)$ for each $v$.
   - Both $P$ and the verifier $V$ perform a zero check on the polynomial $R \cdot R' - 1$, confirming the correctness of $R$ and $R'$.

5. **Multiset Check**:
   - The prover $P$ and verifier $V$ perform a multiset pairing check between merged sets: $(merge(T, U), merge(O, W))$ and $(merge(T, U), merge(F, R))$.
   - This step validates the relationship between $T$ and $U$, ensuring the lookup operation's accuracy and security.

#### PLONKish Arithmeticization

Arithmeticization in PLONKish systems transforms general computation into algebraic form, capturing the logic through polynomial constraints. The prover's data is organized into a **trace matrix**, where each element is a field element representing part of the computation trace.

In PLONKish, the indexed relation is represented as a tuple $(i, x; w)$:

- **i**: Public parameters describing the constraint system, such as field characteristics, trace matrix size, fixed and witness columns count, and copy constraints.
- **x**: Public input to the circuit.
- **w**: Private input (witness data) of the circuit.

PLONKish uses several parameters for its constraint system:

- **τ**: Height of the field tower.
- **ℓ**: Logarithm of the trace matrix row count.
- **ξ**: Logarithm of the statement length.
- **nφ** and **nψ**: Number of fixed and witness columns, respectively.
- **ι**: A mapping from witness columns to field tower indices.
- **a**: Array of fixed columns.
- **g**: List of virtual polynomials for gate constraints.
- **σ**: A global copy constraint map, ensuring consistency across columns.

**Protocol Execution**

The protocol execution involves the following steps:

1. **Indexer Operation**:

   - The indexer $I$ submits polynomial extensions of fixed columns and sets up parameters for permutation checks.
   - For each fixed column $i$, $I$ requests submission to the polynomial oracle and receives a handle.
   - $I$ also configures permutation checks, obtaining additional handles for permutation indices.

2. **Prover and Verifier Interaction**:
   - The prover $P$ and verifier $V$ interactively submit polynomials, perform zero checks, and validate permutation constraints.
   - $P$ pads and extends the witness polynomials and submits them to the oracle.
   - $V$ verifies the correctness of the indices and performs zero checks on gate constraints.
   - Finally, $P$ and $V$ conduct permutation checks to ensure the consistency of the data across different trace matrix columns.

The protocol's security relies on constructing a simulator $E$, which emulates the verifier internally to detect incorrect polynomial submissions. The existence of $E$ ensures that even when facing adversaries, the protocol maintains security by validating that the witness polynomials meet the expected relations.

By enhancing the PLONK framework with tower fields and new virtual polynomial constructions, this approach achieves improved efficiency in proof verification. The integration of complex constraints and multiset checks provides a robust mechanism for validating arbitrary computations while ensuring soundness and security. The proposed protocols extend the applicability of PLONKish systems to binary fields, optimizing their performance and enabling new use cases in practical SNARK applications.

<details><summary><b> Code</b></summary>

<details><summary><b> Python</b></summary>

```python


# Computes the polynomial that returns 0 on 0.....(1<<i)-1 and 1 on (1<<i)
# Relies on the identity W{i+1}(X) = Wi(X) * (Wi(X) + Wi(1<<i))
def get_Wi(i):
    if i == 0:
        return [B(0), B(1)]
    else:
        prev = get_Wi(i - 1)
        o = mul_polys(prev, add_polys(prev, [B(1)]))
        inv_quot = eval_poly_at(o, B(1<<i)).inv()
        return [x*inv_quot for x in o]

# Multiplying an element in the i'th level subfield by X_i can be done in
# an optimized way. See sec III of https://ieeexplore.ieee.org/document/612935
def mul_by_Xi(x, N):
    assert x.shape[-1] == N
    if N == 1:
        return mul(x, 256)
    L, R = x[..., :N//2], x[..., N//2:]
    outR = mul_by_Xi(R, N//2) ^ L
    return np.concatenate((R, outR), axis=-1)

# Build a 65536*65536 multiplication table of uint16's (takes 8 GB RAM)
def build_mul_table():
    table = np.zeros((65536, 65536), dtype=np.uint16)

    for i in [2**x for x in range(16)]:
        top_p_of_2 = 0
        for j in range(1, 65536):
            if (j & (j-1)) == 0:
                table[i, j] = binmul(i, j)
                top_p_of_2 = j
            else:
                table[i][j] = table[i][top_p_of_2] ^ table[i][j - top_p_of_2]

    for i in range(1, 65536):
        if (i & (i-1)) == 0:
            top_p_of_2 = i
        else:
            table[i] = table[top_p_of_2] ^ table[i - top_p_of_2]

    return table




```

</details>

<details><summary><b> Rust</b></summary>

```rust

pub struct LassoBatches {
	pub counts_batch_ids: Vec<BatchId>,
	pub final_counts_batch_id: BatchId,

	pub counts: Vec<OracleId>,
	pub final_counts: OracleId,
}

#[derive(Debug, Getters)]
pub struct LassoClaim<F: Field> {
	/// T polynomial - the table being "looked up"
	#[get = "pub"]
	t_oracle: MultilinearPolyOracle<F>,
	/// U polynomials - each element of U must equal some element of T
	#[get = "pub"]
	u_oracles: Vec<MultilinearPolyOracle<F>>,
}

#[derive(Debug, Getters)]
pub struct LassoWitness<'a, PW: PackedField, L: AsRef<[usize]>> {
	#[get = "pub"]
	t_polynomial: MultilinearWitness<'a, PW>,
	#[get = "pub"]
	u_polynomials: Vec<MultilinearWitness<'a, PW>>,
	#[get = "pub"]
	u_to_t_mappings: Vec<L>,
}

#[derive(Debug, Default)]
pub struct LassoProof<F: Field> {
	pub left_grand_products: Vec<F>,
	pub right_grand_products: Vec<F>,
	pub counts_grand_products: Vec<F>,
}

pub fn reduce_lasso_claim<C: TowerField, F: TowerField + ExtensionField<C> + From<C>>(
	oracles: &mut MultilinearOracleSet<F>,
	lasso_claim: &LassoClaim<F>,
	lasso_batches: &LassoBatches,
	gamma: F,
	alpha: F,
) -> Result<(GkrClaimOracleIds, LassoReducedClaimOracleIds), Error> {
	let t_n_vars = lasso_claim.t_oracle.n_vars();

	let final_counts_oracle = oracles.oracle(lasso_batches.final_counts);

	if final_counts_oracle.n_vars() != t_n_vars {
		bail!(Error::CountsNumVariablesMismatch);
	}

	let alpha_gen = alpha * C::MULTIPLICATIVE_GENERATOR;

	let mut mixed_u_counts_oracle_ids = Vec::new();
	let mut mixed_u_counts_plus_one_oracle_ids = Vec::new();

	let mut gkr_claim_oracle_ids = GkrClaimOracleIds::default();

	for (counts_oracle_id, u_oracle) in izip!(&lasso_batches.counts, &lasso_claim.u_oracles) {
		let u_n_vars = u_oracle.n_vars();

		let counts_oracle = oracles.oracle(*counts_oracle_id);

		if counts_oracle.n_vars() != u_n_vars {
			bail!(Error::CountsNumVariablesMismatch);
		}

		let mixed_u_counts_oracle_id = oracles.add_linear_combination_with_offset(
			u_n_vars,
			gamma,
			[(u_oracle.id(), F::ONE), (*counts_oracle_id, alpha)],
		)?;

		mixed_u_counts_oracle_ids.push(mixed_u_counts_oracle_id);

		let mixed_u_counts_plus_one_oracle_id = oracles.add_linear_combination_with_offset(
			u_n_vars,
			gamma,
			[(u_oracle.id(), F::ONE), (*counts_oracle_id, alpha_gen)],
		)?;

		mixed_u_counts_plus_one_oracle_ids.push(mixed_u_counts_plus_one_oracle_id);

		gkr_claim_oracle_ids.left.push(mixed_u_counts_oracle_id);
		gkr_claim_oracle_ids
			.right
			.push(mixed_u_counts_plus_one_oracle_id);
		gkr_claim_oracle_ids.counts.push(*counts_oracle_id);
	}

	let ones_oracle_id = oracles.add_transparent(Constant {
		n_vars: t_n_vars,
		value: F::ONE,
	})?;

	let mixed_t_final_counts_oracle_id = oracles.add_linear_combination_with_offset(
		t_n_vars,
		gamma,
		[
			(lasso_claim.t_oracle.id(), F::ONE),
			(lasso_batches.final_counts, alpha),
		],
	)?;

	let mixed_t_one_oracle_id = oracles.add_linear_combination_with_offset(
		t_n_vars,
		gamma,
		[(lasso_claim.t_oracle.id(), F::ONE), (ones_oracle_id, alpha)],
	)?;

	let lasso_claim_oracles = LassoReducedClaimOracleIds {
		ones_oracle_id,
		mixed_t_final_counts_oracle_id,
		mixed_t_one_oracle_id,
		mixed_u_counts_oracle_ids,
		mixed_u_counts_plus_one_oracle_ids,
	};

	gkr_claim_oracle_ids
		.left
		.push(mixed_t_final_counts_oracle_id);
	gkr_claim_oracle_ids.right.push(mixed_t_one_oracle_id);
	gkr_claim_oracle_ids.counts.push(lasso_batches.final_counts);

	Ok((gkr_claim_oracle_ids, lasso_claim_oracles))
}


```

</details>

</details>

## Conclusion

Modern zero-knowledge proof systems like STARKs, Binius, and PLONKish leverage advanced techniques such as field extensions, virtual polynomials, and efficient multivariate polynomial representations to address computational and security challenges. These methods enable scalable and efficient computations, particularly in binary fields and tower extensions, optimizing performance while ensuring robust security.

The Binius protocol utilizes partial evaluations and tensor products to decompose data into manageable parts, enhancing verification efficiency. Meanwhile, small domain polynomial commitment schemes leverage Lagrange basis polynomials, Merkle trees, and the Fiat–Shamir transformation to achieve efficient, secure, and scalable polynomial commitments. The Lasso protocol, integrated with PLONKish's arithmeticization framework, further enhances the efficiency and security of lookups in polynomial IOPs, using reciprocal checks and multiset checks for robust verification.

Together, these innovations provide a versatile foundation for cryptographic protocols, supporting complex polynomial computations and enabling high-performance zero-knowledge proofs.

[Binius](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/binius)

[Binius-Main](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/binius-main)

<div  align="center"> 
<img src="images/48_binius.gif" width="50%" />
</div>
