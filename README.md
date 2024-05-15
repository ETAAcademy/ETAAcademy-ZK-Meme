# ETAAcademy-ZK-Meme: Learn ZK in One Sentence

🐬Recipe: ETA Academy ZKmeme<br>
💓Ingredients: Each ZK terms is explained in a short, easy-to-understand article, 75 ZK terms covered, with 25 articles already written. How many left to code💓<br>
🥰Tips: More terms and articles will be added. Dive in with us!🚀

ETA Academy is a research community dedicated to zero-knowledge proofs (zk). Fueled by a daily Twitter check-in summarizing one zk learning point, the ETAcademy-ZK-Meme series was born. With the initial 50 entries, we aim for this meme collection to become everyone's go-to pocket guide for understanding zk quickly. Join the movement and help us make zk accessible! [twitter](https://twitter.com/pwhattie/status/1744001346314134003)

Here are some ways to participate in the ETAAcademy-ZK-Meme series:

- Create your own memes and share them on Twitter with the hashtag #ETAcademyZKMeme.
- Translate the memes into other languages.
- Use the memes to teach others about zk.

By participating in the series, you can help to make zk more accessible to everyone.

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

## A

Account Abstract: EIP4337 contracts with EOAs: UserOperation(Calls)→Bundler(Miner, Relayer)→EntryPoint→Wallet(Validate, Paymaster, execute)
EIP3074 EOAs with contracts:AUTH&AUTHCALL for(v, r, s), commit & call
EIP2938:rlp([nonce, target, data]), PAYGAS, NONCE.

Arithmetic circuits, using ➕ and ✖️ gates, asserts inputs instance(x) and witness(w) from a finite field F, C(x, w) = y. R1CS constraint system, a circuit-based interactive proof, states multiplication or gates for left input ⊗ right input = output, L × C ⊗ R×C = O ×C.🦕

## B

Bilinear pairing e:G1×G2→Gt for mult
if elliptic curves $G_1, G_2,G_t$ are isomorphic, cyclic group, G=G1=G2 , generator g, prime p, a,b∈Fp finite field, then $e(g^a,g^b)=e(g,g)^{ab}, e(g,g)≠e_{Gt},$ addition $e(ag,bg)=e(g,g)^{ab},$ and key exchange $e(g^b,g^c)^a=g^{abc}.$

Block ciphers merge the plaintext and ciphertext spaces into a block space. They serve not only for encryption but also as fundamental components in constructing various cryptographic tools like stream ciphers, hash functions, message authentication codes (MACs), and others.

## C

[CBC](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/01_Ouick_Read/15_CBC.md) mode utilizes a PRP to achieve CPA security by XORing each plaintext block with the previous ciphertext block before encryption, employing an initialization vector (IV) to ensure security, and allowing parallel decryption.

Chinese Remainder states that coprime integers $n_1, ..., n_m ∈ N,$ with their product $n = Π_{i=1}^m n_i,$ and any integers $a_1, ...,a_m ∈ Z,$ there exists a unique residue x of a modulo n, a and (a mod $n_1,$ ..., a mod $n_m)$ break larger problem into smaller one.

[CPA](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/01_Ouick_Read/14_CPA.md) security, akin to semantic security, addresses chosen plaintext attacks, but in CPA, attackers can query the challenger many times, where each query involves encrypting either m_0 or m_1, maintaining equal-length plaintexts but allowing variation across queries.

[CTR](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/01_Ouick_Read/16_CTR.md) employs a pseudo-random function, achieves CPA security by XOR the key stream generated from an IV with plaintext during encryption and decryption.

Cyclic Group or Monogenous Group is a group generated by a single element🐬， $<🐬> = { 🐬^k | k ∈ Z },$ and the more such 🐬, the more powerful properties <🐬>.

## D

Direct Product, 🍬×🍤 <=> 🍱, is an operation that takes two groups G 🍬 and H 🍤and constructs a new group 🍱, usually denoted G × H, in turn, which looks like a group 🍱 has many 🍬sub.groups 🍤.

## E

[Elliptic Curve](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/01_Ouick_Read/01_Elliptic_Curves.md) : $y^2 = x^3 + ax + b, a,b ∈ F,$ $char(F) ≠ 2,3,$ and $△ = 4a^3 + 27b^2 ≠ 0.$ => The points (x, y) in the curve known as P, Q, or R, and the infinity point O, forms a set E = {(x, y) | $y^2 = x^3$ + ax + b} ∪ {O} => E & add operation yields a group (E, +).

Elliptic curve addition: For $P=(x_1, y_1),$ $Q=(x_2, y 2),$ P≠Q，P=Q, Both have intersection points, O or R: 1. Geometric ➕: P + Q = -O or R: P+Q+O=O⇒P+Q=O, P+Q+R=O⇒P+Q=−R. 2. Coordinate ➕：O= 0； $R (x3,−y3)=(λ2−x1−x2,−λ3+(x1+x2)λ−c),$ if the line y=λx+c.

## F

Fflonk & Dan: Fflonk changes single-point polynomials into a polynomial multi-point calculation, where k= 1, Dan verifies pairings on polynomials, not points, by three double-point operations.

Fiat-Shamir: Non-interactive Fiat, compared to interactive zk proof, hash c to for randomization, i.e. A: $v ∈ Z^*_p,$ $t = g^v$ mod p ⇒ B: c = H(g, y, t) ⇒ A: r = v - cx mod {φ(p)}, ⇒ B: $t ≡ g^rg^c,$ as $g^{r}y^{c} ≡ g^{v-cx}g^{xc} ≡ g^{v} ≡ t$ and ${ g^{φ (p)} ≡ 1}🛸$

Fields(F) extend the concepts of Groups (add, sub) and Rings (add, sub, mult) by introducing division，i.e., non-zero element has a mult inverse $a^{-1}.$ The characteristic of a field is the order of the additive identity, usually 0 or a prime number.

Finite Fields & Point Compress (PC): 1) Elliptic curves $(F_p),$ scattered points, have coordinate from F_p and ➕, f(x), and △ modulo p, $|E(F_p)| ≤ 2p + 1;$ 2) Pc trades off space for time for expansion issue of elliptic cryptography, stores (x, y) as (x, y mod 2).🌹

F/K, like F and F[x], means F is the field extension of K, if K is a subfield of F, with f(x) whose coefficients from K, if a ∈ F. All such f(x) forms an ideal J generated by the minimal polynomial f(x), or the monic, irreducible g(x) with its root "a".

Finite Field
The finite field GF(p), or $F_p,$ is characterized by having a prime number of elements and a characteristic of p. Fields of the same order are isomorphic, and its extension, denoted as $F_p^n,$ represents a field of order $p^n.🎡🐦$

## G

Goldwasser-Micali(GM): Legendre $x/p = x/q = -1,$ Jacobi $x/N = (x/p)(x/q) = 1,$ public key (x, N), private key(p, q) => quadratic non-residue encryption $c_i = r^2x$ mod N => quadratic residue decryption $c_p^{(p-1)/2}$ = 1 mod p, $c_q^{(q-1)/2}$ = 1 mod q.

Groth16: upgrades Pinocchio, also verifies bilinearpairing e( \[A\]1 , \[B\]2 ) = e(α $G_1$, β $G_2$ ) ⋅ e ( $∑^l_{i=0}$ $\frac{βu \* i(x)+αv_i(x)+w_i(x)}{γ} G_1, γG_2$ ) ⋅ e( $[C]_1,$ $δG_2$ ) by QAP, $∑^m \* {i=0}a \* iu_i(x) ∑^m \* {i=0} a \* iv_i(x) = ∑^m \* {i=0}a_iω_i(x) + h(x)z(x),$ but less constraints.

## H

Halo2 like UltralPlonk, creates final-poly by Plookup, vanishing, multipoint opening argument, p(X) = q'(X) + \[x_4\] $\Sigma^{n \* q-1}$ \* ${i=0}[x^i_4]q \* i(X)$ , but verified by IPAs.

Halo2 Fibonacci API utilizes a constraint system with columns for advice, instance, fixed, and selector. It's optimized by regions to implement the Fibonacci trait through configuration, chip, and circuit, e.g. f(n) = f(n-1) + f(n-2).

Homo and Iso: Similar to group homomorphisms (addition) but with multiplication, ring homomorphisms has a kernel, the inverse image of 0. The Isomorphism asserts that two rings share identical structures with different element names, e.g. the quotient ring R/Ker(f) ≅ Im(f).

## I

Ideals (I) and Quotient Rings (R/I) are like normal subgroups with Absorption Law and quotient groups. 🐣(I) include {0}, R itself, and principal ideals (a) = {ab | b ∈ R}, while 🐥R/I has add (a+I)+(b+I)=a+b+I, mult (a+I)(b+I)=ab+I, with a congruence relation a≡b(mod I).

Infinite & Singular: 1. Elliptic curve is an affine equation, $(x, y) ∈ A^2(F)$ =>projective $(X, Y, Z) ∈ P^2(F).$ If Z = 0, they are infinite point O, & line $P^1(F).$ 2. If $△ = 4a^3 + 27b^2 = 0,$ <=> not smooth <=> repeated roots, singular points lie on the x-axis.

IPAs, using elliptic curve add and mult to verify and not to reveal the polynomial $P(x) = Σ_ic_ix^is, C(P) = Σ_ic_ig_i,$ with generator 'g', coefficients 'c', random a, aggregate each cg into a larger one until $c'_0 g'_0 = C', i.e. C' = C + a^2L + {R}/{a^2}.🛸$

Modified IPAs have generators on elliptic curves by hashing system parameters(SP) , $G_i= hash(G,i, SP), i=1, ...,n ,$ and vector inner product $z= <a^→, b^→>=z+x^{-2}l_z + x^2f_z,$ rather than multi-party secure computation and quotient polynomial commitment of KZG.

Isogeny, a group surjective homomorphism of elliptic curves, Φ: E(K) → E'(K), maps the infinite point of E(K) to that of E'(K), Φ(O) = O'. Endomorphism is an isogeny of an elliptic curve to itself, while automorphism is isogeny, endomorphism and isomorphism.

## K

$K^{(n)},$ the n-th cyclotomic field over field K, contains K and the roots of polynomials $Q_n(x)=x^n-e=(x-x_1)...(x -x_i)...(x-x_n)=\Pi^n_{s=1}(x - a^s),$ i.e., the splitting field over K. The set E(n) formed by these roots forms a cyclic group for finite fields.

KZG obtains f(x): $[f(S)] = [a_0S^0 + a_1S^1 + a_2S^2] = a_0[S^0] + a_1[S^1] + a_2[S^2] = a_i[S^i]$ by Lagrange interpolation, constructs and transfroms $h(x) = {f(x)-f(z)}/{x-z}$ into bilinear pairing to verify S instead of the f(x), $e([S-z], [h(S)]) = e([f(S)]-[f(z)], [1]).$

## M

Merkle commitment: By computing the Merkle root' := Merkle(c, path(c)) based on the node and its verification path (c, path(c)), where root' = root, it ensures that c exists in the Merkle tree.

[Modular blockchains](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/edit/main/README.md#:~:text=05_Modular_Blockchains) organize tasks into separate layers or modules, like consensus, execution, data availability, and settlement, offering enhanced flexibility and scalability compared to monolithic blockchains.

## P

Pinocchio: 1) create elliptic curve generator for all operand to simplify constraints, $g_l  =g^{ρl};$ 2) let verifier & prover compute together $L(x)=L_v(x)+L_p(x);$ 3) obfuscate &random for zk proof $(L(s) + δ_l) ·(R(s) + δ_r) - (O(s) + δ_o) =t(s)·(h(s) + △).$

PLONK: It creates polynomials by Gate & Linear Constraints(➕, ✖️, &), not R1CS, $Q_L(X)⋅a(X)+Q_R(X)⋅b(X)+Q_o(X)⋅c(X)+Q_M(X)⋅a(X)⋅b(X)+Qc(X)=Z(x)⋅H(x), f(i⋅G)=f(i⋅G)+ β⋅i+γ,g(i⋅G)=g(i⋅G)+β⋅σ(i)+γ,$ Aggregation proofs to a Bilinear $e(x⋅G1, y⋅G2)=e(j⋅ G1, k⋅G2).$

Plookup coordinate accumulators prove polynomial ⊂ and/or table for gate constraints; i.e., given $f∈F^n,t∈F^d, s∈F^{n+d}, F(β, γ)≔(1+β)^n⋅∏_{i∈[n]}(γ+fi)∏_{i∈[d−1]}(γ(1 + β) + t_i + β ⋅ t_{i+1}), G(β, γ);$ then $Z(g^n+1) = 1 => F ≡ G => f ⊂ t, s = (f, t).$ ❤️🛸

[PRF](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/01_Ouick_Read/13_PRG.md) resembles a block cipher PRP, where distinguishing between randomly chosen and key-based functions is central, with PRP serving as a pseudo-random permutation akin to PRF's deterministic function aiming for randomness with a randomly generated key.

Primitive f(x) over F_q is the minimal polynomial of a primitive element in the field $F_q[X]$ that f(x) with non-zero a constant term must be monic, $ord(f(x)) = q^m - 1,$ where the polynomial order ord(f(x)) is the smallest positive integer n for $f(x)|(x^n-e).$

Primitive Root & Discrete Logarithm: An integer g 🐳 is a primitive root (mod n) that generates every element for a group $Z_n^*,$ So g🐳 may serve as a base $🧜‍♀️g^x$ ≡ a (mod n) for a discrete logarithm x = $log_g^a$ and as a generator 🐬for a cyclic group <🐬> = { $🐬^k$ | k ∈ Z }.

Polynomial P(x) = $Σ_{j=0}^{n}{a_jx^j}$ is a sum of terms with the highest power of deg(P), its coefficient Lc(P), common add and mult, while factoring used to find roots (x for P(x)=0) construct Li(x) for Lagrange interpolation P(x) = $Σ*{i=0}^n y_i ⋅L_i(x).$

Polynomial Constraints, to increase randomness and prevent proving forgery, use v, α，β, γ to constrain p(x)=L(x) ⋅ R(x) - O(x) in terms of value (0, 1), transforming e.g. O ⋅R = L, consistency $v_{L,i} = v_{R,i} = v_{o,i} = v_{ β,i}， i∈{1,…,n}$ and constants.

Polynomial ring R[x] has a and x from its base ring R, i.e., if R is a commutative or integral ring, the R[x] inherits these properties. But a field F[x] is different, e.g. $Q(x) = a_n^{-1} P(x)$ is a monic polynomial if P(x) ∈ F[x] with $Lc(P) = a_n.$

[Pseudo-random](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/01_Ouick_Read/07_PRG.md) sequences come in various forms, but in cryptography, a pseudo-random sequence is one that cannot be distinguished from a genuinely random sequence, Adv := | Pr[W_0] - Pr[W_1]|.

## Q

QAP, the polynomial form of R1CS, relies on li (x) and ci, public instance and private witness to produces: $L(x)=Σ^n_{i=1}c_i×l_i(x),$ similar R(x), O(x),and then $p(x)=L(x) ⋅ R(x)-O(x)=(Σ^n_{i=1}c_i ⋅ l_i(x))×(Σ^n_{i=1}c_i ⋅ r_i(x))-(Σ^n_{i=1}c_i ⋅ o_i(x))=0 .🌍🥤🏇$

Quadratic Residues:
Euler's Criterion asserts that for an odd prime p and gcd(a, p) = 1, it holds that $a^{(p-1)/2} ≡a/p (mod p).$ When Legendre symbol $a/p = 1$ a is a quadratic residue modulo n,. $x^2 ≡ a \pmod{n},$ having $(p−1)/2$ quadratic residues p.🐣

## R

Ring simply seen as a group has two binary operations, ➕ and ✖️, e.g. the rings of integers Z and integers modulo n Zn are also groups. Now for a ring's interesting feature: based on Distributive Law, (-🍬)🐶 = -(🍬🐶) = 🍬(-🐶) = -(-🍬)(-🐶), if 🍬, 🐶 ∈ R 🪐.

## S

SHA256 table generates 256-bit random zk proof by 8+64 constants, padding Delta+1+k+64 mod 512 = 0, expansion & compression $MapH_i:= Map(H_{i-1}, M_i),$ which constrains boolean to spread lookup + arithmetic circuit (Plonk), (x, x') ∈ Table, x' = spread (x).

SHA256 constraints have expansion of $Wi=σ_1(W_{i−2})田W_{i−7}田σ_0(W_{i−15})田W_{i−16},$ spread of modular $a田b =c,$ functions of $Ma(A,B,C)=(A∧B)⊕(A∧C)⊕(B∧C), Ch(E,F,G), ∑_0(A)=(A>>>2)⊕(A>>>13)⊕(A>>>22), ∑_1(E)$ constraints to algebraic operation.

Simple extension K(a) is the smallest extension field of subfield K and “a”, the homomorphic image to be an isomorphism with the quotient ring ofits irreducible polynomial g(x), $Im(t) ≅ K[X]/Ker(t) ⇒ Im(t) ≅ K[X]/ (g(x)) ⇒ K(a) ≅ K[X]/ (g(x)).$

Splitting field L，as simple extensions added by elements, is the minimal extension of field K and the roots of polynomial factors $f(x) = b(x-a_1)(x-a_2)...(x-a_m).$ The finite field of $p^n$ is isomorphic to its unique splitting field of $x^{p^n}−x$ over $Z_p.$

[Stream Ciphers](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/01_Ouick_Read/03_Stream_Cipher.md) use a PRG to generate a longer key from a shorter one. This extended key is then XORed with the plaintext to produce ciphertext, or with ciphertext to retrieve the original plaintext, $G: {0, 1}^s$ → ${0, 1}^n$, $K={0,1}^s$, $M=C={0, 1}^n$, E(k, m): G(k)⊕m = c, D(k, c): G(k)⊕c=m, (E, D) is a stream cipher defined on (K, M, C).

## T

Trace (Tr) and Norm (N) are ➕ and ✖️ mappings from extension field to its base，a ∈ F = $F_q^m,$ K= $F_q,$ $Tr_{F/K}(a)$ = $a + a^q + ... + a^{q^{m-1}};$ Compute: minimal polynomial => characteristic polynomial $g(x)^{ m/d}$ => $Tr_{F/K}(a)$ = $-a_{m-1},$ $N_{F/K}(a)=(-1)^ma_0.$

Tweedledum & Amortization take parallel computation on polynomial commitments and value, add random and secret a for Sigma zk proof, C':=A' +z'U+r'H = U+r'H = $A+x^{-2}L_a+x^2R_a$ + $z+x^{-2}l_z+x^2r_zU$ + $(r+x^{-2}r_L+x^2r_H)H$ = $C+x^{-2}L+x^{-2}R;$ $c·C +R = z_1G+z_1b·U+z_2·H.$

## U

UltraPlonk: PK of Plonk KZG (or Dan + Fflonk) PK, Plookup table T*{1,i}, T*{2,i},T*{3,i},i=1,..,n, circuit to create quotient polynomial, verify bilinear pairing by VK on ETH, e(Wη(x)1 + u· W{ωη}(x) · 1, χ2) = e(η· W * η(x)1+uηω·W{ωη}x1 + F1-E1,l2).

## Z

ZK-EVMs scale ETH by improving verification or EVM compatibility from ETH-equiv(PSE, Taiko), EVM-equiv(Scroll, Polygon), Almost EVM-equiv(Gas adjust), to language-equiv(Starkware, zkSync), e.g. Geth→Trace→Roller(zkEVM Aggr. circuit →Aggr. proof) → L1 contract.

zk Homomorphism in projective coordinates: For Φ: E(K) → E'(K), E(k) be $y^2 = x^3 + ax + b,$ $f(x) = x^3 + ax + b,$ P = (x, y) ∈ E(K), the standard form is $Φ(x, y) = ({u(x)}/{v(x)}, {s(x)}/{t(x)}y),$ with divisor relations: $v^3(x)|t^2(x)$ and $t^2(x)|v^3(x)f(x).$

ZK Proof: For m = 0 <=> z = r, m = 1 <=> z = rx, validators verifie the quadratic residue directly by $c · x^{2b} ≡ z^2$ mod $N,$ compared to the GM algorithm. The zk proof ensures validators succesfully verify z when m = 0 even if they doesn't know x.

[zkStark](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/README.md#:~:text=07_PRG.md-,08_zkStark,-.md): RS Codes improves 2^n Trace poly on AIR, not circuit PK & VK, immune to for loops. Unchanged order blowup root $2^kN$ poly, hash Poseidon2, combined by quotient poly, yields $DEEP(x) = a_0{s_0(x) - s_0(z)}/{x-z} + a_1{s_1(x) - s_1(z)}/{x-z} + a_2{CP(x)-CP(z)}/{x-z}.$ For FRI, along with folding n and Merkle commitments of verified by grinding factors like Pow.

zkStark AIR & ALI convert arithmetic & boundary constraints into divisibility over a finite field that AIR use quotient polynomials verify trace $P={P_1(X^→, Y^→),...,P_s(X^→,Y^→)}$ and ALI diminish the time and space complexity to reduce polynomials into one as FRI.

zkStark Fibonacci: F(X,Y)=Z, like zkSnark's Sigma H=wG, trace T to satisfy transition and boundary constraints by quotient polynomials $C_0(x) = {P_1(x+1) - (P_0(x)+P_1(x))} / {∑^i_{0,...,n-1}(x-i)}, C_1(x), C_2(x)={P_0(x)-X}/{x-0}, C_3(x)$ combined into one f(x) for FRI.

zkStark FRI reduces a polynomial of degree d to two merged into one by random weights of Fait-Shamir for Merkle commitment, after log d steps, to create a constant $f_{log(d)}(x)=g_{log(d)}(x)+a_{log(d)}·h_{log(d)}(x)$ verified by $f(z_1)=f(w), d/2^{log(d)} < 1.$
