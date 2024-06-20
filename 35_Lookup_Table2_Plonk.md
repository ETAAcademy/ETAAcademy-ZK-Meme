# ETAAcademy-ZKMeme: 35. Lookup Table 2: Halo2 V.S. Plonk

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>35. Lookup Table 2</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Lookup_Table2_Plonk</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

## Lookup Arguments: Halo2 V.S. Plonk

### What is a Lookup Gate?

- Zero-Knowledge Proofs (ZKPs) rely on various types of gates to perform computations. Custom Gates are specialized tools that enforce specific polynomial relationships between their inputs and outputs. Lookup Gates, on the other hand, can manage any type of relationship between inputs and outputs. Both Halo2-lookup and Plookup are powerful tools for implementing lookup arguments in ZKPs.

### Halo2 Lookup

- The Halo2-lookup scheme demonstrates the implementation of lookup gates. It focuses on proving that a query vector's elements belong to a table vector, represented as $\vec{f} \subseteq \vec{t}$. The process involves:

  1.  **Sorting (Polynomial Proof):**
      To ensure that the query vector $\vec{f}$ is sorted in the same order as the table vector $\vec{t},$ the Halo2-lookup scheme employs auxiliary vectors $\vec{f}'$ and $\vec{f}'.$ The rule is that each unmarked element in $\vec{f}'$ equals its left neighbor, and each marked element equals the corresponding element in $\vec{t}'$, $f'_i=f'\_{i-1}$ or $f'_i=t'_i$. To prevent cyclic rollovers, $\vec{f}'$ and $\vec{t}'$ must start with the same element, $f'_0=t'_0$. Using Lagrange Basis for polynomial encoding, we get:
      
$$
(f'(X)-f'(\omega^{-1}\cdot X))\cdot (f'(X)-t'(X)) = 0, \quad \forall x\in H
$$   
      
      
      
$$
L_0(X)\cdot(f'(X)-t'(X)) = 0, \quad \forall x\in H
$$
      
  2.  **Permutation Proof:** Ensures $(\vec{f}, \vec{f}')$ and $(\vec{t}, \vec{t}')$ satisfy certain permutation relations:

$$
\frac{z(\omega\cdot X)}{z(X)}=\frac{(f(X)+\gamma_1)(t(X)+\gamma_2)}{(f'(X)+\gamma_1)(t'(X)+\gamma_2)}
$$

$$
L_0(X)\cdot (z(X) - 1) = 0, \quad \forall x\in H
$$

### Plookup

- **Step 1:** Define an auxiliary vector $\vec{s}$ as a permutation of elements from $\{f_i\} \cup \{t_i\}$.

  - **Step 1.2:** Ensure $\vec{s}$ is sorted according to $\vec{t}$ by treating each element and its neighbor as a multiset:

    $$
    \begin{array}{ccccc}
    S &  & T  & & F \\
    \{(s_i, s\_{i+1})\} & =_{multiset} & \{(t_i, t\_{i+1})\} &\cup&\{(f_i,f_i)\}\\
    \end{array}
    $$

    For example:

    $$
    \begin{array}{ccccc}
    \{(1,1), (1,2),(2,2),(2,2),(2,3),(3,3),(3,4)\} & =\_{multiset} & \{(1,2),(2,3),(3,4)\} &\cup&\{(3,3),(2,2),(2,2),(1,1)\}\\
    \end{array}
    $$

- **Step 1.3:** Using verifier-provided challenges \(\beta\) and \(\gamma\), fold these pairs into single values for permutation argument:

  $$
  \{s_i + \beta s_{i+1}\}=\{t_i + \beta t_{i+1}\}\cup\{(1+\beta)f_i\}
  $$

  Transform the multiset equality argument into a grand product argument:

  $$
  \begin{split}
  &\prod_i{((1+\beta)f_i+\gamma)(t_i+\beta\cdot t_{i+1}+\gamma)} \\
  =&\prod_i
  {(s_i+\beta\cdot s_{i+1}+\gamma)}
  \end{split}
  $$

  In the Plookup scheme, this proof transformation is not used. Instead, the order of $\beta$ and $\gamma$ is swapped: first, $\gamma$ is used for the product permutation, and then $\beta$ is used for folding:

  $$
  \{(s_i+\gamma) + \beta (s_{i+1}+\gamma)\}=\{(t_i + \gamma) + \beta (t_{i+1}+\gamma)\}\cup\{(f_i+\gamma)+ \beta(f_i+\gamma)\}
  $$

The relevant Grand Product constraint equation is:

$$
\begin{split}
&\prod_i{(1+\beta)(f_i+\gamma)(t_i+\beta\cdot t_{i+1}+(1+\beta)\gamma)} \\
=&\prod_i
{(s_i+\beta\cdot s_{i+1}+(1+\beta)\gamma)}
\end{split}
$$

- **Step 1.4:** However, this introduces a new problem. The degree of the $\vec{s}$ polynomial exceeds the degree of $\vec{f}$ or $\vec{t}$. Plookup solves this by splitting $\vec{s}$ into two halves, $\vec{s}^{lo}$ and $\vec{s}^{hi}$, but the last element of $\vec{s}^{lo}$ must equal the first element of $\vec{s}^{hi}$:

$$
\vec{s}^{lo}_{N-1} = \vec{s}^{hi}_0
$$

Next, the Prover introduces an accumulator auxiliary vector $\vec{z}$ to prove the Grand Product:

$$
z_0=1, \quad z_{i+1}=z_i\cdot \frac{(1+\beta)(f_i+\gamma)(t_i+\beta\cdot t_{i+1}+\gamma(1+\beta))}
{(s^{lo}_i+\beta\cdot s^{lo}\_{i+1}+\gamma(1+\beta))(s^{hi}_i+\beta\cdot s^{hi}\_{i+1}+\gamma(1+\beta))},\quad z\_{N-1}=1
$$

By encoding $\vec{z}$, we obtain the polynomial $z(X)$, which should satisfy the following constraints:

$$
\begin{split}
L_0(X)\cdot(z(X)-1) & = 0 \\
L_{N-1}(X)\cdot(s^{lo}(X)-s^{hi}(\omega\cdot X)) & = 0 \\
L_{N-1}(X)\cdot(z(X)-1) & = 0\\
\end{split}
$$

Additionally, according to the recurrence relation of $\vec{z}$, $z(X)$ must also satisfy the following constraint:

$$
\small
\begin{split}
(X-\omega^{N-1})\cdot z(X)\cdot\Big((1+\beta)(f(X)+\gamma)\Big)\cdot\Big(t(X)+\beta\cdot t(\omega\cdot X)+\gamma(1+\beta)\Big) \qquad \\
-(X-\omega^{N-1})\cdot z(\omega\cdot X)\cdot\Big(s^{lo}(X)+\beta\cdot s^{lo}(\omega\cdot X)+\gamma(1+\beta)\Big)\cdot\Big(s^{hi}(X)+\beta\cdot s^{hi}(\omega\cdot X)+\gamma(1+\beta)\Big) = 0
\end{split}
$$

**Optimization in Plonkup:**

Split $\vec{s}$ into even and odd indexed parts:

$$
\begin{split}
\vec{s}^{even}&=(s_0,s_2,s_4,\ldots,s_{2n-2})\\
\vec{s}^{odd}&=(s_1,s_3,s_5,\ldots,s_{2n-1})\\
\end{split}
$$

In this approach, there's no need to limit the length of $\vec{f}$ to N-1; it can extend up to N. This allows the length of $\vec{s}$ to reach 2N. This is possible because the relationship between $(\vec{f}, \vec{t}, \vec{s}^{even}, \vec{s}^{odd})$ can wrap around to the starting position within the subgroup H.

The $\vec{z}$ vector can be redefined as:

$$
z_0 = 1,\quad z_{i+1}=z_i\cdot \frac{(1+\beta)(f_i+\gamma)(t_i+\beta\cdot t_{i+1}+\gamma(1+\beta))}
{(s^{even}_i+\beta\cdot s^{odd}\_{i}+\gamma(1+\beta))(s^{odd}_i+\beta\cdot s^{even}\_{i+1}+\gamma(1+\beta))}
$$

**Multi-Column Tables:**
**Collapsing a multi-column table into a single column table using random challenge numbers.** 

Suppose the computation table is $(\vec{t}\_1, \vec{t}\_2, \vec{t}\_3)$， then the corresponding lookup record should also be a three-column table, denoted as $(\vec{f}\_1,\vec{f}\_2,\vec{f}\_3)$. If we want to prove that $(f_{1,i},f_{2,i},f_{3,i})=(t_{1,j},t_{2,j},f_{3,j})$， we can ask the Verifier for a random challenge number $\eta$, and collapse the computation table horizontally as follows:

$$
\vec{t} = \vec{t}_1+\eta\cdot\vec{t}_2+\eta^2\cdot\vec{t}_3
$$

Similarly, the Prover collapses the lookup record horizontally:

$$
\vec{f} = \vec{f}_1+\eta\cdot\vec{f}_2+\eta^2\cdot\vec{f}_3
$$
