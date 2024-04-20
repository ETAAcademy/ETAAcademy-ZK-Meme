# ETAAcademy-ZKMeme: 10. zkStark AIR and Rescue-Prime

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>10. zkStark AIR and Rescue-Prime</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>zkStark_AIR_Rescue-Prime</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

More Detail for zkStark:<br>
[Part I: STARK Overview](https://aszepieniec.github.io/stark-anatomy/overview)<br>
[Part II: Basic Tools](https://aszepieniec.github.io/stark-anatomy/basic-tools)<br>
[Part II: FRI](https://aszepieniec.github.io/stark-anatomy/fri)<br>
[Part IV: The STARK Polynomial IOP](https://aszepieniec.github.io/stark-anatomy/stark)<br>
[Part V: A Rescue-Prime STARK](https://aszepieniec.github.io/stark-anatomy/rescue-prime)<br>
[Part VI: Speeding Things Up](https://aszepieniec.github.io/stark-anatomy/faster)<br>

### AIR: Arithmetic Constraint System

The arithmetic intermediate representation (AIR) is vital in STARKs, framing computations through execution traces and constraints, which are interpolated into polynomials. The trace can be viewed as a series of state transitions of registers. To reduce the time and space complexity for the prover, the polynomials are linearly combined into one polynomial. This verification involves symbolic evaluation and division by a zerofier polynomial to ensure computational integrity. Finally, from the arithmetic constraint system, two types of witnesses are obtained: the execution trace of the entire program to be proven for trace polynomials, the constraints for quotient polynomials, which could be used for polynomial commitments and FRI system as shown in zkStarks Part One.

- Prove

  ```python
  defprove( self, trace, transition_constraints, boundary, proof_stream=None ):
  # create proof stream object if necessary
  if proof_stream== None:
              proof_stream= ProofStream()

  # concatenate randomizers
  for kin range(self.num_randomizers):
              trace= trace+ [[self.field.sample(os.urandom(17))for sin range(self.num_registers)]]

  # interpolate
          trace_domain= [self.omicron^ifor iin range(len(trace))]
          trace_polynomials= []
  for sin range(self.num_registers):
              single_trace= [trace[c][s]for cin range(len(trace))]
              trace_polynomials= trace_polynomials+ [Polynomial.interpolate_domain(trace_domain, single_trace)]

  # subtract boundary interpolants and divide out boundary zerofiers
          boundary_quotients= []
  for sin range(self.num_registers):
              interpolant= self.boundary_interpolants(boundary)[s]
              zerofier= self.boundary_zerofiers(boundary)[s]
              quotient= (trace_polynomials[s]- interpolant)/ zerofier
              boundary_quotients+= [quotient]

  # commit to boundary quotients
          fri_domain= self.fri.eval_domain()
          boundary_quotient_codewords= []
          boundary_quotient_Merkle_roots= []
  for sin range(self.num_registers):
              boundary_quotient_codewords= boundary_quotient_codewords+ [boundary_quotients[s].evaluate_domain(fri_domain)]
              merkle_root= Merkle.commit(boundary_quotient_codewords[s])
              proof_stream.push(merkle_root)

  # symbolically evaluate transition constraints
          point= [Polynomial([self.field.zero(), self.field.one()])]+ trace_polynomials+ [tp.scale(self.omicron)for tpin trace_polynomials]
          transition_polynomials= [a.evaluate_symbolic(point)for ain transition_constraints]

  # divide out zerofier
          transition_quotients= [tp/ self.transition_zerofier()for tpin transition_polynomials]

  # commit to randomizer polynomial
          randomizer_polynomial= Polynomial([self.field.sample(os.urandom(17))for iin range(self.max_degree(transition_constraints)+1)])
          randomizer_codeword= randomizer_polynomial.evaluate_domain(fri_domain)
          randomizer_root= Merkle.commit(randomizer_codeword)
          proof_stream.push(randomizer_root)

  # get weights for nonlinear combination
  #  - 1 randomizer
  #  - 2 for every transition quotient
  #  - 2 for every boundary quotient
          weights= self.sample_weights(1+ 2*len(transition_quotients)+ 2*len(boundary_quotients), proof_stream.prover_fiat_shamir())

  assert([tq.degree()for tqin transition_quotients]== self.transition_quotient_degree_bounds(transition_constraints)), "transition quotient degrees do not match with expectation"

  # compute terms of nonlinear combination polynomial
          x= Polynomial([self.field.zero(), self.field.one()])
          terms= []
          terms+= [randomizer_polynomial]
  for iin range(len(transition_quotients)):
              terms+= [transition_quotients[i]]
              shift= self.max_degree(transition_constraints)- self.transition_quotient_degree_bounds(transition_constraints)[i]
              terms+= [(x^shift)* transition_quotients[i]]
  for iin range(self.num_registers):
              terms+= [boundary_quotients[i]]
              shift= self.max_degree(transition_constraints)- self.boundary_quotient_degree_bounds(len(trace), boundary)[i]
              terms+= [(x^shift)* boundary_quotients[i]]

  # take weighted sum
  # combination = sum(weights[i] * terms[i] for all i)
          combination= reduce(lambda a, b : a+b, [Polynomial([weights[i]])* terms[i]for iin range(len(terms))], Polynomial([]))

  # compute matching codeword
          combined_codeword= combination.evaluate_domain(fri_domain)

  # prove low degree of combination polynomial
          indices= self.fri.prove(combined_codeword, proof_stream)
          indices.sort()
          duplicated_indices= [ifor iin indices]+ [(i+ self.expansion_factor)% self.fri.domain_lengthfor iin indices]

  # open indicated positions in the boundary quotient codewords
  for bqcin boundary_quotient_codewords:
  for iin duplicated_indices:
                  proof_stream.push(bqc[i])
                  path= Merkle.open(i, bqc)
                  proof_stream.push(path)

  # ... as well as in the randomizer
  for iin indices:
              proof_stream.push(randomizer_codeword[i])
              path= Merkle.open(i, randomizer_codeword)
              proof_stream.push(path)

  # the final proof is just the serialized stream
  return proof_stream.serialize()
  ```

- Verify

  ```python
  def verify( self, proof, transition_constraints, boundary, proof_stream=None ):
          H = blake2b

          # infer trace length from boundary conditions
          original_trace_length = 1 + max(c for c, r, v in boundary)
          randomized_trace_length = original_trace_length + self.num_randomizers

          # deserialize with right proof stream
          if proof_stream == None:
              proof_stream = ProofStream()
          proof_stream = proof_stream.deserialize(proof)

          # get Merkle roots of boundary quotient codewords
          boundary_quotient_roots = []
          for s in range(self.num_registers):
              boundary_quotient_roots = boundary_quotient_roots + [proof_stream.pull()]

          # get Merkle root of randomizer polynomial
          randomizer_root = proof_stream.pull()

          # get weights for nonlinear combination
          weights = self.sample_weights(1 + 2*len(transition_constraints) + 2*len(self.boundary_interpolants(boundary)), proof_stream.verifier_fiat_shamir())

          # verify low degree of combination polynomial
          polynomial_values = []
          verifier_accepts = self.fri.verify(proof_stream, polynomial_values)
          polynomial_values.sort(key=lambda iv : iv[0])
          if not verifier_accepts:
              return False

          indices = [i for i,v in polynomial_values]
          values = [v for i,v in polynomial_values]

          # read and verify leafs, which are elements of boundary quotient codewords
          duplicated_indices = [i for i in indices] + [(i + self.expansion_factor) % self.fri.domain_length for i in indices]
          leafs = []
          for r in range(len(boundary_quotient_roots)):
              leafs = leafs + [dict()]
              for i in duplicated_indices:
                  leafs[r][i] = proof_stream.pull()
                  path = proof_stream.pull()
                  verifier_accepts = verifier_accepts and Merkle.verify(boundary_quotient_roots[r], i, path, leafs[r][i])
                  if not verifier_accepts:
                      return False

          # read and verify randomizer leafs
          randomizer = dict()
          for i in indices:
              randomizer[i] = proof_stream.pull()
              path = proof_stream.pull()
              verifier_accepts = verifier_accepts and Merkle.verify(randomizer_root, i, path, randomizer[i])

          # verify leafs of combination polynomial
          for i in range(len(indices)):
              current_index = indices[i] # do need i

              # get trace values by applying a correction to the boundary quotient values (which are the leafs)
              domain_current_index = self.generator * (self.omega^current_index)
              next_index = (current_index + self.expansion_factor) % self.fri.domain_length
              domain_next_index = self.generator * (self.omega^next_index)
              current_trace = [self.field.zero() for s in range(self.num_registers)]
              next_trace = [self.field.zero() for s in range(self.num_registers)]
              for s in range(self.num_registers):
                  zerofier = self.boundary_zerofiers(boundary)[s]
                  interpolant = self.boundary_interpolants(boundary)[s]

                  current_trace[s] = leafs[s][current_index] * zerofier.evaluate(domain_current_index) + interpolant.evaluate(domain_current_index)
                  next_trace[s] = leafs[s][next_index] * zerofier.evaluate(domain_next_index) + interpolant.evaluate(domain_next_index)

              point = [domain_current_index] + current_trace + next_trace
              transition_constraints_values = [transition_constraints[s].evaluate(point) for s in range(len(transition_constraints))]

              # compute nonlinear combination
              counter = 0
              terms = []
              terms += [randomizer[current_index]]
              for s in range(len(transition_constraints_values)):
                  tcv = transition_constraints_values[s]
                  quotient = tcv / self.transition_zerofier().evaluate(domain_current_index)
                  terms += [quotient]
                  shift = self.max_degree(transition_constraints) - self.transition_quotient_degree_bounds(transition_constraints)[s]
                  terms += [quotient * (domain_current_index^shift)]
              for s in range(self.num_registers):
                  bqv = leafs[s][current_index] # boundary quotient value
                  terms += [bqv]
                  shift = self.max_degree(transition_constraints) - self.boundary_quotient_degree_bounds(randomized_trace_length, boundary)[s]
                  terms += [bqv * (domain_current_index^shift)]
              combination = reduce(lambda a, b : a+b, [terms[j] * weights[j] for j in range(len(terms))], self.field.zero())

              # verify against combination polynomial value
              verifier_accepts = verifier_accepts and (combination == values[i])
              if not verifier_accepts:
                  return False

          return verifier_accepts
  ```

### Rescue-Prime: a STARK Proof of AIR and Signature

Rescue-Prime STARK, a concretely useful STARK proof system that serves as both a post-quantum signature scheme and a proof of correct evaluation for the Rescue-Prime hash function. Rescue-Prime is described as an arithmetization-oriented hash function, employing a sponge construction with multiple almost-identical rounds. The steps involved in a single round include forward and backward S-box operations, matrix multiplications, and adding round constants. Transition constraints and boundary constraints are detailed for arithmetizing the Rescue-Prime function, and the process of obtaining witness data (the execution trace).

- Rescue-Prime AIR

  ```python
  defround_constants_polynomials( self, omicron ):
          first_step_constants= []
  for iin range(self.m):
              domain= [omicron^rfor rin range(0, self.N)]
              values= [self.round_constants[2*r*self.m+i]for rin range(0, self.N)]
              univariate= Polynomial.interpolate_domain(domain, values)
              multivariate= MPolynomial.lift(univariate, 0)
              first_step_constants+= [multivariate]
          second_step_constants= []
  for iin range(self.m):
              domain= [omicron^rfor rin range(0, self.N)]
              values= [self.field.zero()]* self.N
  #for r in range(self.N):
  #    print("len(round_constants):", len(self.round_constants), " but grabbing index:", 2*r*self.m+self.m+i, "for r=", r, "for m=", self.m, "for i=", i)
  #    values[r] = self.round_constants[2*r*self.m + self.m + i]
              values= [self.round_constants[2*r*self.m+self.m+i]for rin range(self.N)]
              univariate= Polynomial.interpolate_domain(domain, values)
              multivariate= MPolynomial.lift(univariate, 0)
              second_step_constants+= [multivariate]

  return first_step_constants, second_step_constants

  deftransition_constraints( self, omicron ):
  # get polynomials that interpolate through the round constants
          first_step_constants, second_step_constants= self.round_constants_polynomials(omicron)

  # arithmetize one round of Rescue-Prime
          variables= MPolynomial.variables(1+ 2*self.m, self.field)
          cycle_index= variables[0]
          previous_state= variables[1:(1+self.m)]
          next_state= variables[(1+self.m):(1+2*self.m)]
          air= []
  for iin range(self.m):
  # compute left hand side symbolically
  # lhs = sum(MPolynomial.constant(self.MDS[i][k]) * (previous_state[k]^self.alpha) for k in range(self.m)) + first_step_constants[i]
              lhs= MPolynomial.constant(self.field.zero())
  for kin range(self.m):
                  lhs= lhs+ MPolynomial.constant(self.MDS[i][k])* (previous_state[k]^self.alpha)
              lhs= lhs+ first_step_constants[i]

  # compute right hand side symbolically
  # rhs = sum(MPolynomial.constant(self.MDSinv[i][k]) * (next_state[k] - second_step_constants[k]) for k in range(self.m))^self.alpha
              rhs= MPolynomial.constant(self.field.zero())
  for kin range(self.m):
                  rhs= rhs+ MPolynomial.constant(self.MDSinv[i][k])* (next_state[k]- second_step_constants[k])
              rhs= rhs^self.alpha

  # equate left and right hand sides
              air+= [lhs-rhs]

  return air

      def trace( self, input_element ):
          trace = []

          # absorb
          state = [input_element] + [self.field.zero()] * (self.m - 1)

          # explicit copy to record state into trace
          trace += [[s for s in state]]

          # permutation
          for r in range(self.N):

              # forward half-round
              # S-box
              for i in range(self.m):
                  state[i] = state[i]^self.alpha
              # matrix
              temp = [self.field.zero() for i in range(self.m)]
              for i in range(self.m):
                  for j in range(self.m):
                      temp[i] = temp[i] + self.MDS[i][j] * state[j]
              # constants
              state = [temp[i] + self.round_constants[2*r*self.m+i] for i in range(self.m)]

              # backward half-round
              # S-box
              for i in range(self.m):
                  state[i] = state[i]^self.alphainv
              # matrix
              temp = [self.field.zero() for i in range(self.m)]
              for i in range(self.m):
                  for j in range(self.m):
                      temp[i] = temp[i] + self.MDS[i][j] * state[j]
              # constants
              state = [temp[i] + self.round_constants[2*r*self.m+self.m+i] for i in range(self.m)]

              # record state at this point, with explicit copy
              trace += [[s for s in state]]

          return trace
  ```
