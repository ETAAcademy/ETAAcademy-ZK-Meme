# ETAAcademy-ZKMeme: 44. Bit Reverse and Batch

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>44. BitReverse_Batch</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>BitReverse_Batch</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

The STARKs protocol relies on several key cryptographic tools to ensure the proof of large-scale computations:

1. **Polynomial Commitments**: Implemented using Merkle trees.
2. **Fiat-Shamir Heuristic**: Converts interactive protocols into non-interactive ones.
3. **FRI Protocol**: Verifies the low-degree property of polynomials.
4. **Random Sampling and Batch Verification**: Reduces the verifier's computational load.
5. **Proof of Work (Grinding)**: The verifier requires the prover to perform a proof of work (similar to mining). The prover needs to find a random string `nonce` such that the hash function result meets certain criteria (e.g., leading zeros), proving that the prover has expended computational effort.

**Steps:**

1. Write a **Cairo program**.
2. The program is compiled into **Cairo VM bytecode**.
3. The VM executes the code and provides an **execution trace**.
4. The trace is passed to the STARK prover, who creates a proof of correct execution based on Cairo’s **AIR (Algebraic Intermediate Representation)**.
5. The proof is then sent to the verifier, who checks its validity.

In this version, two notable aspects are the **bit-reversal symmetric arrangement** and **batch processing**.

---

### 1. Optimizing Bit-Reversal Symmetric Arrangement

The `interpolate_and_commit` function interpolates a polynomial and obtains the **LDE (low-degree extension) trace evaluation** over the LDE domain. The first optimization, `lde_trace_permuted`, uses a unique **bit-reversal ordering** for the Merkle tree leaves. In the original order, elements are arranged sequentially as $x_1, x_2, x_3, \dots$, but in this case, a symmetric arrangement is required, such that $p_1(x)$ is immediately followed by $p_1(-x)$, convenient for computing:

$$
P\_{i+1}(x^2) = \frac{P_i(x) + P_i(-x)}{2} + \beta \frac{P_i(x) - P_i(-x)}{2x}
$$

In bit-reversal, we perform an XOR operation, which, from a binary perspective, means reversing the digits of a number. For example, $p(hw)$ and $p(-hw)$ (where $-hw$ geometrically represents a 180-degree rotation of $hw$) can be written as $hw^{i+2^{k-1}}$, equivalent to $hw^4$, satisfying the odd-even symmetric arrangement as shown below:

$$
P\_{i+1}(x^2) = \frac{P_i(x) + P_i(-x)}{2} + \beta \frac{P_i(x) - P_i(-x)}{2x}
$$

Bit-reversal arrangement:

$$
[p(h), p(hw), p(hw^2), p(hw^3), p(hw^4), p(hw^5), p(hw^6), p(hw^7)]
$$

$$
[p(h), p(hw^4), p(hw), p(hw^5), p(hw^2), p(hw^6), p(hw^3), p(hw^7)]
$$

- Code:
  ```rust
  let mut lde_trace_permuted = lde_trace_evaluations.clone();
  for col in lde_trace_permuted.iter_mut() {
      in_place_bit_reverse_permute(col);
  }
  ```

### 2. Batch Commit

The key idea behind batch processing is combining multiple tasks (e.g., evaluations of different polynomials or parts of a proof) and verifying them together instead of separately.

- **Polynomial Batch Verification**: The prover submits multiple evaluations, representing different computational steps or intermediate results. The verifier generates random coefficients and combines these values into one. The verifier then checks if this combined value satisfies the polynomial properties.
- **Merkle Tree Batch Processing**: STARKs use Merkle trees to commit polynomial evaluations. Batch processing can be applied to the verification of multiple Merkle tree paths by combining them, reducing the verifier's workload.

For instance, consider a trace represented as a matrix. By transposing it, you can organize the values in columns, and each column represents the evaluations of a point. The original trace (for example, in the Fibonacci STARK example) produces traces such as trace `a`, `b`, `c` by [LDE blowup](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/41_LDE.md). These traces are hashed together in a batch, instead of generating separate hash trees for each constraint,transposing the matrix allows for more efficient batch processing of evaluations by organizing the trace into columns.

For example:

Trace:

$$
\begin{array}{c|c|c|c|} trace_a & trace_b & trace_c\\
y_{a0}& y_{b0} & y_{c0}\\
y_{a1}& y_{b1} & y_{c1}\\
y_{a2}& y_{b2} & y_{c2}\\
\end{array}
$$

Transpose:

$$
\begin{array}{c|c|c|c|} trace_a & y_{a0}& y_{a1} & y_{a2}\\
trace_b & y_{b0}& y_{b1} & y_{b2}\\
trace_c & y_{c0}& y_{c1} & y_{c2}\\
\end{array}
$$

- Code:
  ```rust
  // Compute commitment.
  let lde_trace_permuted_rows = columns2rows(lde_trace_permuted);
  let (lde_trace_merkle_tree, lde_trace_merkle_root) = Self::batch_commit(&lde_trace_permuted_rows);
  ```

---

### Fibonacci Example Illustration

**Prover Side**:

1. Compute the trace polynomial `t` by interpolating the trace column over a set of $2^n$-th roots of unity $\{g^i : 0 \leq i < 2^n\}$.
2. Compute the boundary polynomial `B`.
3. Compute the transition constraint polynomial `C`.
4. Construct the composition polynomial `H` from `B` and `C`.
5. Sample an out-of-domain point `z` and provide the evaluation $H(z)$ and the necessary trace evaluations $t(z)$, $t(zg)$, and $t(zg^2)$.
6. Sample a domain point $x_0$ and provide the evaluation $H(x_0)$ and $t(x_0)$.
7. Construct the deep composition polynomial `Deep(x)` from `H`, `t`, and the evaluations.
8. Perform FRI on `Deep(x)` and provide the resulting FRI commitment to the verifier.
9. Provide the Merkle root of `t` and the Merkle proof of $t(x_0)$.

**Verifier Side**:

1. Reconstruct the evaluations $B(z)$ and $C(z)$ from the trace evaluations. Verify that $H(z) = \beta_1 B(z) + \beta_2 C(z)$.
2. Check the evaluations $H(x_0)$ and $t(x_0)$ and verify $Deep(x_0)$ matches the equation:

$$
Deep(x_0) = \gamma_1 \dfrac{H(x_0) - H(z)}{x_0 - z} + \gamma_2 \dfrac{t(x_0) - t(z)}{x_0 - z} + \gamma_3 \dfrac{t(x_0) - t(zg)}{x_0 - zg} + \gamma_4 \dfrac{t(x_0) - t(zg^2)}{x_0 - zg^2}
$$

3. Verify the FRI commitment.
4. Check that $t(x_0)$ belongs to the trace using the Merkle root and proof.

---

### Round One: Commit main and extended trace

The `round_1_randomized_air_with_preprocessing` function is the first round of the STARK proof protocol, responsible for processing the main trace and generating necessary commitments and challenges. It begins by interpolating the main trace table using the `interpolate_and_commit` method, which returns the trace polynomials, their evaluations, and the corresponding Merkle tree and root. These results are encapsulated in a `Round1CommitmentData` structure for the main trace. The function then generates RAP (Random Auxiliary Proof) challenges using the `air.build_rap_challenges` method. If an auxiliary trace exists, it is also processed similarly through interpolation and commitment, resulting in another `Round1CommitmentData` structure. The evaluations from both the main and auxiliary traces are then combined to create an LDE (Low-Degree Extension) trace table using the `LDETraceTable::from_columns` method. Finally, the function returns a `Round1` structure containing the LDE trace, the main and auxiliary commitments, and the RAP challenges, encapsulating all the necessary data for the next steps in the proof process.

<details><summary><b> Code</b></summary>

```rust
   /// Returns the result of the first round of the STARK Prove protocol.
    fn round_1_randomized_air_with_preprocessing(
        air: &A,
        main_trace: &TraceTable<A::Field>,
        domain: &Domain<A::Field>,
        transcript: &mut impl IsTranscript<A::FieldExtension>,
    ) -> Result<Round1<A>, ProvingError>
    where
        FieldElement<A::Field>: AsBytes + Send + Sync,
        FieldElement<A::FieldExtension>: AsBytes + Send + Sync,
    {
        let (trace_polys, evaluations, main_merkle_tree, main_merkle_root) =
            Self::interpolate_and_commit::<A::Field>(main_trace, domain, transcript);

        let main = Round1CommitmentData::<A::Field> {
            trace_polys,
            lde_trace_merkle_tree: main_merkle_tree,
            lde_trace_merkle_root: main_merkle_root,
        };

        let rap_challenges = air.build_rap_challenges(transcript);

        let aux_trace = air.build_auxiliary_trace(main_trace, &rap_challenges);
        let (aux, aux_evaluations) = if !aux_trace.is_empty() {
            let (aux_trace_polys, aux_trace_polys_evaluations, aux_merkle_tree, aux_merkle_root) =
                Self::interpolate_and_commit(&aux_trace, domain, transcript);
            let aux_evaluations = aux_trace_polys_evaluations;
            let aux = Some(Round1CommitmentData::<A::FieldExtension> {
                trace_polys: aux_trace_polys,
                lde_trace_merkle_tree: aux_merkle_tree,
                lde_trace_merkle_root: aux_merkle_root,
            });
            (aux, aux_evaluations)
        } else {
            (None, Vec::new())
        };

        let lde_trace = LDETraceTable::from_columns(
            evaluations,
            aux_evaluations,
            A::STEP_SIZE,
            domain.blowup_factor,
        );

        Ok(Round1 {
            lde_trace,
            main,
            aux,
            rap_challenges,
        })
    }


```

</details>

### Round Two: Construct of composition polynomial H

The `round_2_compute_composition_polynomial` function computes the evaluations of the composition polynomial on the LDE domain. It first creates a `ConstraintEvaluator` to evaluate the constraints based on the RAP challenges and the LDE trace from the first round. After obtaining the constraint evaluations, it interpolates these to derive the coefficients of the composition polynomial using FFT. The polynomial is then split into parts, and each part is evaluated on the LDE domain. Finally, it calls `commit_composition_polynomial` to generate the Merkle tree and commitment for these evaluations, returning a `Round2` structure that encapsulates the evaluations, polynomial parts, and their corresponding Merkle tree and root.

The `commit_composition_polynomial` function is responsible for creating a Merkle tree and a commitment for the evaluations of the parts of the composition polynomial. It begins by initializing a new vector to store the evaluations, where it iterates through the evaluations of each part, constructing rows of evaluations for the Merkle tree. After collecting these evaluations, it applies a bit-reversal permutation to optimize the data layout for subsequent processing. The evaluations are then merged in pairs to reduce the overall size before calling `batch_commit`, which generates the Merkle tree and commitment for the merged evaluations.

<details><summary><b> Code</b></summary>

```rust

    /// Returns the Merkle tree and the commitment to the evaluations of the parts of the
    /// composition polynomial.
    fn commit_composition_polynomial(
        lde_composition_poly_parts_evaluations: &[Vec<FieldElement<A::FieldExtension>>],
    ) -> (BatchedMerkleTree<A::FieldExtension>, Commitment)
    where
        FieldElement<A::Field>: AsBytes + Sync + Send,
        FieldElement<A::FieldExtension>: AsBytes + Sync + Send,
    {
        // TODO: Remove clones
        let mut lde_composition_poly_evaluations = Vec::new();
        for i in 0..lde_composition_poly_parts_evaluations[0].len() {
            let mut row = Vec::new();
            for evaluation in lde_composition_poly_parts_evaluations.iter() {
                row.push(evaluation[i].clone());
            }
            lde_composition_poly_evaluations.push(row);
        }

        in_place_bit_reverse_permute(&mut lde_composition_poly_evaluations);

        let mut lde_composition_poly_evaluations_merged = Vec::new();
        for chunk in lde_composition_poly_evaluations.chunks(2) {
            let (mut chunk0, chunk1) = (chunk[0].clone(), &chunk[1]);
            chunk0.extend_from_slice(chunk1);
            lde_composition_poly_evaluations_merged.push(chunk0);
        }

        Self::batch_commit(&lde_composition_poly_evaluations_merged)
    }

    /// Returns the result of the second round of the STARK Prove protocol.
    fn round_2_compute_composition_polynomial(
        air: &A,
        domain: &Domain<A::Field>,
        round_1_result: &Round1<A>,
        transition_coefficients: &[FieldElement<A::FieldExtension>],
        boundary_coefficients: &[FieldElement<A::FieldExtension>],
    ) -> Round2<A::FieldExtension>
    where
        A: Send + Sync,
        FieldElement<A::Field>: AsBytes + Send + Sync,
        FieldElement<A::FieldExtension>: AsBytes + Send + Sync,
    {
        // Compute the evaluations of the composition polynomial on the LDE domain.
        let evaluator = ConstraintEvaluator::new(air, &round_1_result.rap_challenges);
        let constraint_evaluations = evaluator.evaluate(
            air,
            &round_1_result.lde_trace,
            domain,
            transition_coefficients,
            boundary_coefficients,
            &round_1_result.rap_challenges,
        );

        // Get coefficients of the composition poly H
        let composition_poly =
            Polynomial::interpolate_offset_fft(&constraint_evaluations, &domain.coset_offset)
                .unwrap();

        let number_of_parts = air.composition_poly_degree_bound() / air.trace_length();
        let composition_poly_parts = composition_poly.break_in_parts(number_of_parts);

        let lde_composition_poly_parts_evaluations: Vec<_> = composition_poly_parts
            .iter()
            .map(|part| {
                evaluate_polynomial_on_lde_domain(
                    part,
                    domain.blowup_factor,
                    domain.interpolation_domain_size,
                    &domain.coset_offset,
                )
                .unwrap()
            })
            .collect();

        let (composition_poly_merkle_tree, composition_poly_root) =
            Self::commit_composition_polynomial(&lde_composition_poly_parts_evaluations);

        Round2 {
            lde_composition_poly_evaluations: lde_composition_poly_parts_evaluations,
            composition_poly_parts,
            composition_poly_merkle_tree,
            composition_poly_root,
        }
    }

```

</details>

### Round Three: Evaluation of polynomial at z

The `round_3_evaluate_polynomials_in_out_of_domain_element` function is the third round of the STARK proof protocol, focusing on evaluating polynomials at an out-of-domain point `z`. It begins by calculating `z_power`, which is `z` raised to the power equal to the number of parts in the composition polynomial. The function then evaluates each part of the composition polynomial at this `z_power`, collecting these evaluations into a vector called `composition_poly_parts_ood_evaluation`. Following this, it retrieves the out-of-domain evaluations for the trace polynomials using the `get_trace_evaluations` method, which evaluates the main trace and any auxiliary trace at the specified point `z`, considering the transition offsets and the primitive root used for interpolation. This process ensures that the evaluations are consistent and can be verified against the composition polynomial. Finally, the function returns a `Round3` structure containing both the trace evaluations and the evaluations of the composition polynomial parts, providing essential data for the subsequent verification steps in the STARK proof process.

<details><summary><b> Code</b></summary>

```rust

    /// Returns the result of the third round of the STARK Prove protocol.
    fn round_3_evaluate_polynomials_in_out_of_domain_element(
        air: &A,
        domain: &Domain<A::Field>,
        round_1_result: &Round1<A>,
        round_2_result: &Round2<A::FieldExtension>,
        z: &FieldElement<A::FieldExtension>,
    ) -> Round3<A::FieldExtension>
    where
        FieldElement<A::Field>: AsBytes + Sync + Send,
        FieldElement<A::FieldExtension>: AsBytes + Sync + Send,
    {
        let z_power = z.pow(round_2_result.composition_poly_parts.len());

        // Evaluate H_i in z^N for all i, where N is the number of parts the composition poly was
        // broken into.
        let composition_poly_parts_ood_evaluation: Vec<_> = round_2_result
            .composition_poly_parts
            .iter()
            .map(|part| part.evaluate(&z_power))
            .collect();

        // Returns the Out of Domain Frame for the given trace polynomials, out of domain evaluation point (called `z` in the literature),
        // frame offsets given by the AIR and primitive root used for interpolating the trace polynomials.
        // An out of domain frame is nothing more than the evaluation of the trace polynomials in the points required by the
        // verifier to check the consistency between the trace and the composition polynomial.
        //
        // In the fibonacci example, the ood frame is simply the evaluations `[t(z), t(z * g), t(z * g^2)]`, where `t` is the trace
        // polynomial and `g` is the primitive root of unity used when interpolating `t`.
        let trace_ood_evaluations =
            crate::trace::get_trace_evaluations::<A::Field, A::FieldExtension>(
                &round_1_result.main.trace_polys,
                round_1_result
                    .aux
                    .as_ref()
                    .map(|aux| &aux.trace_polys)
                    .unwrap_or(&vec![]),
                z,
                &air.context().transition_offsets,
                &domain.trace_primitive_root,
                A::STEP_SIZE,
            );

        Round3 {
            trace_ood_evaluations,
            composition_poly_parts_ood_evaluation,
        }
    }

```

</details>

### Round Four: FRI commit and query

The `round_4_compute_and_run_fri_on_the_deep_composition_polynomial` function represents the fourth round of the STARK proof protocol, focusing on computing the deep composition polynomial and executing the FRI (Fast Reed-Solomon Interactive) protocol. Initially, the function retrieves the coset offset from the AIR context and samples a random element `gamma` from the transcript. It calculates the number of terms in the composition polynomial and the trace polynomials, generating a vector of deep composition coefficients based on `gamma`. The function then computes the deep composition polynomial by calling `compute_deep_composition_poly`, which combines the trace and composition polynomials. Following this, the function performs the FRI commitment phase, generating the last value and layers of the FRI protocol, and appends a nonce to the transcript if security bits are specified. Finally, it samples query indexes, executes the FRI query phase, collects the Merkle roots of the FRI layers, and opens the deep composition polynomial, returning a `Round4` structure that encapsulates all relevant data for the proof.

The `compute_deep_composition_poly` function is responsible for calculating the deep composition polynomial, which is a linear combination of the trace and composition polynomials, with coefficients sampled by the verifier. The function begins by computing `z_power`, which is `z` raised to the power equal to the number of parts in the composition polynomial. It then iterates through each part of the composition polynomial, evaluating it at `z_power` and constructing the polynomial terms based on the differences between the evaluated parts and their corresponding evaluations. The function ensures that the resulting polynomial evaluates to zero at `z_power` and performs Ruffini division to eliminate this root. Additionally, it retrieves the necessary trace evaluations and computes the trace terms for the deep composition polynomial, ultimately returning the computed deep composition polynomial. This process ensures the polynomial's correctness and validity, which is crucial for the subsequent verification steps in the STARK proof.

The `open_deep_composition_poly` function is designed to open the deep composition polynomial at specified indexes and their symmetric elements. For each index, the function first calls `open_trace_polys` to obtain the opening values for the main trace polynomial and then calls `open_composition_poly` to retrieve the opening values for the composition polynomial. If an auxiliary trace polynomial exists, the function also opens it. The results from these openings are encapsulated in a `DeepPolynomialOpening` structure, which is then added to the openings vector. Finally, the function returns all the openings, allowing for the verification of the deep composition polynomial and its associated trace polynomials during the proof verification process.

<details><summary><b> Code</b></summary>

```rust

    /// Returns the result of the fourth round of the STARK Prove protocol.
    fn round_4_compute_and_run_fri_on_the_deep_composition_polynomial(
        air: &A,
        domain: &Domain<A::Field>,
        round_1_result: &Round1<A>,
        round_2_result: &Round2<A::FieldExtension>,
        round_3_result: &Round3<A::FieldExtension>,
        z: &FieldElement<A::FieldExtension>,
        transcript: &mut impl IsTranscript<A::FieldExtension>,
    ) -> Round4<A::Field, A::FieldExtension>
    where
        FieldElement<A::Field>: AsBytes + Send + Sync,
        FieldElement<A::FieldExtension>: AsBytes + Send + Sync,
    {
        let coset_offset_u64 = air.context().proof_options.coset_offset;
        let coset_offset = FieldElement::<A::Field>::from(coset_offset_u64);

        let gamma = transcript.sample_field_element();
        let n_terms_composition_poly = round_2_result.lde_composition_poly_evaluations.len();
        let n_terms_trace = air.context().transition_offsets.len() * air.context().trace_columns;

        // <<<< Receive challenges: 𝛾, 𝛾'
        let mut deep_composition_coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(n_terms_composition_poly + n_terms_trace)
                .collect();

        let trace_poly_coeffients: Vec<_> = deep_composition_coefficients
            .drain(..n_terms_trace)
            .collect();

        // <<<< Receive challenges: 𝛾ⱼ, 𝛾ⱼ'
        let gammas = deep_composition_coefficients;

        // Compute p₀ (deep composition polynomial)
        let deep_composition_poly = Self::compute_deep_composition_poly(
            air,
            &round_1_result.all_trace_polys(),
            round_2_result,
            round_3_result,
            z,
            &domain.trace_primitive_root,
            &gammas,
            &trace_poly_coeffients,
        );

        let domain_size = domain.lde_roots_of_unity_coset.len();

        // FRI commit and query phases
        let (fri_last_value, fri_layers) = fri::commit_phase::<A::Field, A::FieldExtension>(
            domain.root_order as usize,
            deep_composition_poly,
            transcript,
            &coset_offset,
            domain_size,
        );

        // grinding: generate nonce and append it to the transcript
        let security_bits = air.context().proof_options.grinding_factor;
        let mut nonce = None;
        if security_bits > 0 {
            let nonce_value = grinding::generate_nonce(&transcript.state(), security_bits)
                .expect("nonce not found");
            transcript.append_bytes(&nonce_value.to_be_bytes());
            nonce = Some(nonce_value);
        }

        let number_of_queries = air.options().fri_number_of_queries;
        let iotas = Self::sample_query_indexes(number_of_queries, domain, transcript);
        let query_list = fri::query_phase(&fri_layers, &iotas);

        let fri_layers_merkle_roots: Vec<_> = fri_layers
            .iter()
            .map(|layer| layer.merkle_tree.root)
            .collect();

        let deep_poly_openings =
            Self::open_deep_composition_poly(domain, round_1_result, round_2_result, &iotas);

        Round4 {
            fri_last_value,
            fri_layers_merkle_roots,
            deep_poly_openings,
            query_list,
            nonce,
        }
    }

    fn sample_query_indexes(
        number_of_queries: usize,
        domain: &Domain<A::Field>,
        transcript: &mut impl IsTranscript<A::FieldExtension>,
    ) -> Vec<usize> {
        let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
        (0..number_of_queries)
            .map(|_| (transcript.sample_u64(domain_size >> 1)) as usize)
            .collect::<Vec<usize>>()
    }

    /// Returns the DEEP composition polynomial that the prover then commits to using
    /// FRI. This polynomial is a linear combination of the trace polynomial and the
    /// composition polynomial, with coefficients sampled by the verifier (i.e. using Fiat-Shamir).
    #[allow(clippy::too_many_arguments)]
    fn compute_deep_composition_poly(
        air: &A,
        trace_polys: &[Polynomial<FieldElement<A::FieldExtension>>],
        round_2_result: &Round2<A::FieldExtension>,
        round_3_result: &Round3<A::FieldExtension>,
        z: &FieldElement<A::FieldExtension>,
        primitive_root: &FieldElement<A::Field>,
        composition_poly_gammas: &[FieldElement<A::FieldExtension>],
        trace_terms_gammas: &[FieldElement<A::FieldExtension>],
    ) -> Polynomial<FieldElement<A::FieldExtension>>
    where
        FieldElement<A::Field>: AsBytes + Send + Sync,
        FieldElement<A::FieldExtension>: AsBytes + Send + Sync,
    {
        let z_power = z.pow(round_2_result.composition_poly_parts.len());

        // ∑ᵢ 𝛾ᵢ ( Hᵢ − Hᵢ(z^N) ) / ( X − z^N )
        let mut h_terms = Polynomial::zero();
        for (i, part) in round_2_result.composition_poly_parts.iter().enumerate() {
            // h_i_eval is the evaluation of the i-th part of the composition polynomial at z^N,
            // where N is the number of parts of the composition polynomial.
            let h_i_eval = &round_3_result.composition_poly_parts_ood_evaluation[i];
            let h_i_term = &composition_poly_gammas[i] * (part - h_i_eval);
            h_terms = h_terms + h_i_term;
        }
        assert_eq!(h_terms.evaluate(&z_power), FieldElement::zero());
        h_terms.ruffini_division_inplace(&z_power);

        // Get trace evaluations needed for the trace terms of the deep composition polynomial
        let transition_offsets = &air.context().transition_offsets;
        let trace_frame_evaluations = &round_3_result.trace_ood_evaluations;

        // Compute the sum of all the trace terms of the deep composition polynomial.
        // There is one term for every trace polynomial and for every row in the frame.
        // ∑ ⱼₖ [ 𝛾ₖ ( tⱼ − tⱼ(z) ) / ( X − zgᵏ )]

        // @@@ this could be const
        let trace_frame_length = trace_frame_evaluations.height;

        #[cfg(feature = "parallel")]
        let trace_terms = trace_polys
            .par_iter()
            .enumerate()
            .fold(Polynomial::zero, |trace_terms, (i, t_j)| {
                Self::compute_trace_term(
                    &trace_terms,
                    (i, t_j),
                    trace_frame_length,
                    trace_terms_gammas,
                    &trace_frame_evaluations.columns(),
                    transition_offsets,
                    (z, primitive_root),
                )
            })
            .reduce(Polynomial::zero, |a, b| a + b);

        #[cfg(not(feature = "parallel"))]
        let trace_terms =
            trace_polys
                .iter()
                .enumerate()
                .fold(Polynomial::zero(), |trace_terms, (i, t_j)| {
                    Self::compute_trace_term(
                        &trace_terms,
                        (i, t_j),
                        trace_frame_length,
                        trace_terms_gammas,
                        &trace_frame_evaluations.columns(),
                        transition_offsets,
                        (z, primitive_root),
                    )
                });

        h_terms + trace_terms
    }

    /// Adds to `accumulator` the term corresponding to the trace polynomial `t_j` of the Deep
    /// composition polynomial. That is, returns `accumulator + \sum_i \gamma_i \frac{ t_j - t_j(zg^i) }{ X - zg^i }`,
    /// where `i` ranges from `T * j` to `T * j + T - 1`, where `T` is the number of offsets in every frame.
    fn compute_trace_term(
        accumulator: &Polynomial<FieldElement<A::FieldExtension>>,
        (j, t_j): (usize, &Polynomial<FieldElement<A::FieldExtension>>),
        trace_frame_length: usize,
        trace_terms_gammas: &[FieldElement<A::FieldExtension>],
        trace_frame_evaluations: &[Vec<FieldElement<A::FieldExtension>>],
        transition_offsets: &[usize],
        (z, primitive_root): (&FieldElement<A::FieldExtension>, &FieldElement<A::Field>),
    ) -> Polynomial<FieldElement<A::FieldExtension>>
    where
        FieldElement<A::Field>: AsBytes + Send + Sync,
        FieldElement<A::FieldExtension>: AsBytes + Send + Sync,
    {
        let iter_trace_gammas = trace_terms_gammas.iter().skip(j * trace_frame_length);
        let trace_int = trace_frame_evaluations[j]
            .iter()
            .zip(transition_offsets)
            .zip(iter_trace_gammas)
            .fold(
                Polynomial::zero(),
                |trace_agg, ((t_j_z, offset), trace_gamma)| {
                    // @@@ this can be pre-computed
                    let z_shifted = primitive_root.pow(*offset) * z;
                    let mut poly = t_j - t_j_z;
                    poly.ruffini_division_inplace(&z_shifted);
                    trace_agg + poly * trace_gamma
                },
            );

        accumulator + trace_int
    }

    /// Computes values and validity proofs of the evaluations of the composition polynomial parts
    /// at the domain value corresponding to the FRI query challenge `index` and its symmetric
    /// element.
    fn open_composition_poly(
        composition_poly_merkle_tree: &BatchedMerkleTree<A::FieldExtension>,
        lde_composition_poly_evaluations: &[Vec<FieldElement<A::FieldExtension>>],
        index: usize,
    ) -> PolynomialOpenings<A::FieldExtension>
    where
        FieldElement<A::Field>: AsBytes + Sync + Send,
        FieldElement<A::FieldExtension>: AsBytes + Sync + Send,
    {
        let proof = composition_poly_merkle_tree
            .get_proof_by_pos(index)
            .unwrap();

        let lde_composition_poly_parts_evaluation: Vec<_> = lde_composition_poly_evaluations
            .iter()
            .flat_map(|part| {
                vec![
                    part[reverse_index(index * 2, part.len() as u64)].clone(),
                    part[reverse_index(index * 2 + 1, part.len() as u64)].clone(),
                ]
            })
            .collect();

        PolynomialOpenings {
            proof: proof.clone(),
            proof_sym: proof,
            evaluations: lde_composition_poly_parts_evaluation
                .clone()
                .into_iter()
                .step_by(2)
                .collect(),
            evaluations_sym: lde_composition_poly_parts_evaluation
                .into_iter()
                .skip(1)
                .step_by(2)
                .collect(),
        }
    }

    /// Computes values and validity proofs of the evaluations of the trace polynomials
    /// at the domain value corresponding to the FRI query challenge `index` and its symmetric
    /// element.
    fn open_trace_polys<E>(
        domain: &Domain<A::Field>,
        tree: &BatchedMerkleTree<E>,
        lde_trace: &Table<E>,
        challenge: usize,
    ) -> PolynomialOpenings<E>
    where
        FieldElement<A::Field>: AsBytes + Sync + Send,
        FieldElement<E>: AsBytes + Sync + Send,
        A::Field: IsSubFieldOf<E>,
        E: IsField,
    {
        let domain_size = domain.lde_roots_of_unity_coset.len();

        let index = challenge * 2;
        let index_sym = challenge * 2 + 1;
        PolynomialOpenings {
            proof: tree.get_proof_by_pos(index).unwrap(),
            proof_sym: tree.get_proof_by_pos(index_sym).unwrap(),
            evaluations: lde_trace
                .get_row(reverse_index(index, domain_size as u64))
                .to_vec(),
            evaluations_sym: lde_trace
                .get_row(reverse_index(index_sym, domain_size as u64))
                .to_vec(),
        }
    }

    /// Open the deep composition polynomial on a list of indexes and their symmetric elements.
    fn open_deep_composition_poly(
        domain: &Domain<A::Field>,
        round_1_result: &Round1<A>,
        round_2_result: &Round2<A::FieldExtension>,
        indexes_to_open: &[usize],
    ) -> DeepPolynomialOpenings<A::Field, A::FieldExtension>
    where
        FieldElement<A::Field>: AsBytes + Send + Sync,
        FieldElement<A::FieldExtension>: AsBytes + Send + Sync,
    {
        let mut openings = Vec::new();

        for index in indexes_to_open.iter() {
            let main_trace_opening = Self::open_trace_polys::<A::Field>(
                domain,
                &round_1_result.main.lde_trace_merkle_tree,
                &round_1_result.lde_trace.main_table,
                *index,
            );

            let composition_openings = Self::open_composition_poly(
                &round_2_result.composition_poly_merkle_tree,
                &round_2_result.lde_composition_poly_evaluations,
                *index,
            );

            let aux_trace_polys = round_1_result.aux.as_ref().map(|aux| {
                Self::open_trace_polys::<A::FieldExtension>(
                    domain,
                    &aux.lde_trace_merkle_tree,
                    &round_1_result.lde_trace.aux_table,
                    *index,
                )
            });

            openings.push(DeepPolynomialOpening {
                composition_poly: composition_openings,
                main_trace_polys: main_trace_opening,
                aux_trace_polys,
            });
        }

        openings
    }

```

</details>

[STARKS_Recap](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/blob/main/41_LDE.md)

<div style="text-align: center;">
    <img src="../04_WebService/images/crab007.webp" alt="Image 1" width="100%" style="display: inline-block;">
</div>
