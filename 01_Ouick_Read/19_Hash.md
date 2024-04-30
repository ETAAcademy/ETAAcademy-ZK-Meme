# ETAAcademy-ZKMeme: 19. Hash

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>19. Hash</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Hash</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

- **Hash Functions**: Hash functions play a critical role in cryptography, also known as one-way hash functions. They can be categorized into two types: hash functions without keys and hash functions with keys. A hash function accepts a long message as input and generates a fixed-length output. If h is a hash function with input space X and output space Y, it is defined on (X, Y).

- **Security of Hash Functions**: The security of hash functions can be assessed through three properties:

  1. **Preimage Resistance**: Given a hash value $y \in Y$, finding the corresponding input $x \in X$ is computationally difficult.
  2. **Second Preimage Resistance**: Given an input $x \in X$, finding a different input $x' \in X$ such that $h(x) = h(x')$ is computationally difficult.
  3. **Collision Resistance**: Finding any two distinct inputs $x \neq x' \in X$ such that $h(x) = h(x')$ is computationally difficult.

  In practical applications, breaking a hash function often involves attempting to find collisions. If a collision is found, the hash function is considered insecure.

- **Relationship Between Security Properties**: The probability of finding a collision is equal to the probability of finding a second preimage. Therefore, if a hash function \(H\) is collision-resistant, it is also second preimage-resistant. Introducing a partition $\{C_1, \ldots, C_n\}$ of X, where each set $C_i$ contains elements that collide with each other but not with elements from other sets, allows for the construction of an algorithm that finds a pair of collisions with a probability of 1/2.

- **Probability of Finding Collisions**: In the quest to find collisions in hash functions, exhaustive attacks are the most straightforward approach, with a time complexity of $O(2^n)$. However, birthday attacks are more efficient, allowing for the discovery of collisions in $O(2^{n/2})$ time by leveraging the birthday paradox. This probability can be calculated using the formula $p = 1- (1 - 1/N) x (1- 2/N) x … x (1 – (t - 1)/N)=1 - \frac{N!}{N^t (N - t)!}\$, where $N = 2^n$ for an n-bit hash function. By preparing approximately $t \approx 1.177\sqrt{2^n}$ messages, a collision can be found with a probability of at least 1/2.
