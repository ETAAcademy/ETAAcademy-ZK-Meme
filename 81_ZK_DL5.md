# ETAAcademy-ZKMeme: 81. ZK Deep Learning 5

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>81. ZKDL5</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZKDL5</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# ZKDL5: Paradigms, Search, and Game Theory

Artificial Intelligence has evolved through the tension between logical Symbolic AI and connectionist Neural Networks, characterized by the Modeling, Inference, and Learning (MIL) framework. Linear classifiers like perceptrons face fundamental constraints exemplified by the non-linearly separable XOR problem, necessitating multi-layered neural architectures.

Within state-based models, search problems are defined by state spaces, actions, and cost functions, where algorithms like Depth-First Search (DFS), Breadth-First Search (BFS), and Uniform Cost Search (UCS) manage the trade-off between time and memory. Dynamic Programming optimizes these searches by caching results for overlapping subproblems in acyclic graphs, while A\* Search enhances efficiency using consistent and admissible heuristics derived from problem relaxation.

Neural networks automate feature extraction using layered hidden units and backpropagation, which leverages the chain rule to compute gradients across computation graphs. Game theory extends these models to adversarial environments, employing Minimax and Expectiminimax to maximize utility against various opponents.

Alpha-beta pruning and move ordering optimize game tree searches by eliminating irrelevant branches, while Temporal Difference (TD) learning enables agents to refine evaluation functions through experience and bootstrapping. In simultaneous games, mixed strategies and Von Neumann's Minimax Theorem ensure stability, while non-zero-sum scenarios introduce the Nash Equilibrium to describe states where no player benefits from unilateral deviation, despite potential conflicts between individual and social optimality.

---

## 1. Symbolic AI and Connectionism: "Agents vs. Tools" perspective and the XOR problem

Artificial Intelligence has undergone a historical evolution between the logical paradigm and the connectionist paradigm, possessing the dual attributes of **AI as an agent** (replicating human capabilities) and **AI as a tool** (enhancing human productivity). There is a fundamental tension between **Symbolic AI** (logic-driven, rule-based) and **Connectionism** (neural-inspired, data-driven) in the field of AI, particularly regarding key issues such as **adversarial examples**, **algorithmic bias**, and the mathematical incompatibility of different definitions of **fairness**.

The constraints of linear classifiers (perceptrons) are illustrated by the "XOR problem," which is a fundamental mathematical result. Linear classifiers attempt to use a line (2D), a plane (3D), or a hyperplane (higher dimensions) to separate two classes of data. The "XOR problem" ( $0 \oplus 0 = 0, 0 \oplus 1 = 1, 1 \oplus 0 = 1, 1 \oplus 1 = 0$ ) is non-linearly separable. You cannot separate $(0,0)$ and $(1,1)$ from $(0,1)$ and $(1,0)$ with a single straight line. This mathematical limitation stifled neural network research in 1969, as early researchers focused on single-layer models (perceptrons) restricted by linear boundaries. Multiple concepts of fairness (e.g., calibration vs. error rate balance) are mathematically incompatible, such as $P(\hat{Y}=1 | Y=1, A=a) = P(\hat{Y}=1 | Y=1, A=b)$.

$$
y = \text{sign}(w \cdot x + b) = \text{sign}\left(\sum_{i=1}^{d} w_i x_i + b
\right)
$$

- $y$: Output prediction, typically $\in \{-1, 1\}$ or $\{0, 1\}$.
- $w$: Weight vector, representing the learned parameters of the hyperplane.
- $x$: Input vector, features (e.g., $[x_1, x_2]$ for XOR).
- $b$: Bias term, shifting the decision boundary from the origin.

---

## 1.1 Modeling, Inference, and Learning (MIL): Modeling, Inference, and Learning (MIL) framework and the roadmap from Reflex models to Logic

**Modeling** requires "wise" selection of features, discarding irrelevant information to keep the problem tractable. Once a model is defined, **inference** algorithms (e.g., shortest path solvers) are used to extract answers. **Learning** bridges the gap between the model and the real world using data. Data is used to estimate weights or costs rather than manual encoding—marking a shift from "code complexity" to "data complexity." Roadmap classification based on the model's "intelligence level":

- **Reflex Models:** Fixed computation (linear classifiers, deep neural networks).
- **State-based Models:** Agents with foresight (search, Markov Decision Processes, game theory).
- **Variable-based Models:** Assignments under constraints (Constraint Satisfaction Problems, Bayesian networks).
- **Logic-based Models:** Reasoning using heterogeneous, open-ended information.

### Discrete Optimization (Edit Distance): Edit Distance problem and its efficient solution using Dynamic Programming and Memoization

For discrete optimization problems with overlapping subproblems, such as string alignment, recursive formulas can be defined and solved efficiently using dynamic programming with memoization to avoid redundant computation. **Discrete optimization** involves a search space composed of discrete objects (e.g., strings, paths). The primary tool is **Dynamic Programming (DP)**, which relies on two key properties: **optimal substructure** (the optimal solution to a problem contains the optimal solutions to subproblems) and **overlapping subproblems** (the same subproblems are solved repeatedly).

Discrete optimization seeks to find $\min_{P} 	\text{cost}(P)$. **Edit distance** is the minimum number of edits required to transform string $S$ to $T$. The strategy is to "chip away" at the problem from the end of the strings. **Recursion** uses subproblem $(m, n)$ to represent the edit distance between the first $m$ characters of string $S$ and the first $n$ characters of string $T$.

If characters $S[m]$ and $T[n]$ match, the cost is the distance between prefixes $(m-1, n-1)$. If they do not match, we consider three possible edits:

- **Substitution:** Pay 1 and solve $(m-1, n-1)$.
- **Deletion:** Pay 1 and solve $(m-1, n)$.
- **Insertion:** Pay 1 and solve $(m, n-1)$.

The algorithm takes the minimum of these three options. To solve the **exponential search** problem of naive recursion, **memoization** stores the result of each $(m, n)$ pair in a cache, ensuring each subproblem is computed only once.

The recurrence relation for edit distance: the distance between two empty strings is 0. If one string is empty, the distance is the length of the other (all insertions). If the last characters match, ignore them. If they don't, try all three edits and choose the lowest-cost path. This recursive structure allows us to explore a vast (exponential) number of possible edit sequences in polynomial time ( $O(m \cdot n)$ ) by reusing results. This is the core "inference" step of the edit distance "model." Memoization requires $O(m \cdot n)$ space for the cache, which can be problematic for very long sequences (e.g., DNA). "Inserting into set S is equivalent to deleting from set T," so the code considers one set of operations to transform $S$ to $T$.

$$
D(m,n) =
\begin{cases}
n & m=0 \\
m & n=0 \\
\min
\begin{cases}
D(m-1,n) + 1 \\
D(m,n-1) + 1 \\
D(m-1,n-1) + \delta
\end{cases}
& \text{otherwise}
\end{cases}
$$

where

$$
\delta =
\begin{cases}
0 & S[m]=T[n] \
1 & S[m]\ne T[n]
\end{cases}
$$

$$
0 \le m \le |S|,\quad 0 \le n \le |T|
$$

- $m$: Prefix length of string $S$, $0 \le m \le |S|$.
- $n$: Prefix length of string $T$, $0 \le n \le |T|$.
- $D(m, n)$: Edit distance of prefixes, scalar integer.
- $S, T$: Two input strings, $S = \{s_1, \dots, s_{|S|}\}$.

**Algorithm: Memoized Edit Distance**

The `cache` dictionary stores previously computed values for keys `(m, n)`. It handles **base cases** first, followed by the **match case** (cost 0). The **recursive step** handles mismatched characters by taking the minimum of three branches. **Memoization** transforms an $O(3^{m+n})$ algorithm into $O(m \cdot n)$.

<details><summary>Code</summary>

```python
cache = {}
def recurse(m, n):
    if (m, n) in cache: return cache[(m, n)]
    if m == 0: result = n
    elif n == 0: result = m
    elif s[m-1] == t[n-1]:
        result = recurse(m-1, n-1)
    else:
        sub = 1 + recurse(m-1, n-1)
        delete = 1 + recurse(m-1, n)
        insert = 1 + recurse(m, n-1)
        result = min(sub, delete, insert)
    cache[(m, n)] = result
    return result
```

</details>

---

## 1.2 Features and Neural Networks

### Framework and Optimization Problems

Machine learning can be viewed as an optimization problem: adjusting predictor weights to minimize average training loss for regression or classification tasks. Based on **statistical learning theory** and **loss minimization**, the goal is to find a function (predictor) that maps inputs to outputs by minimizing a "loss function." The loss function quantifies the difference between predicted values and true labels. Based on how loss is computed, it is divided into **regression** (predicting continuous values) and **classification** (predicting discrete labels). In the high-level framework of machine learning, a learner receives a dataset and generates a predictor $f$. This is defined as an optimization problem: minimizing training loss, which is the average loss over all training samples. Loss definitions for different tasks:

- **Regression**: Focuses on **residuals** (the difference between predicted and labeled values). Common loss functions include squared loss and absolute deviation.
- **Classification**: Focuses on **margins** (the product of the score and the label value). Common loss functions include zero-one loss (for accuracy), hinge loss, and logistic loss (for easier optimization).

Training loss objective function: By minimizing the average error across all $n$ samples in the training set, the optimal model weights $w$ can be found. This provides a concrete mathematical goal for the "learning" process. By minimizing this average, we hope the model generalizes well to unseen data. This is the foundational objective function for the entire optimization framework we discuss.

$$
\min_{w \in \mathbb{R}^d} \text{TrainLoss}(w) = \frac{1}{n} \sum_{i=1}^n 	\text{Loss}(x_i, y_i, w)
$$

- $w$: Weight vector, a vector in $d$-dimensional space.
- $n$: Number of training samples, a scalar.
- $x_i$: Input of the $i$-th sample, feature vector or raw input.
- $y_i$: Label of the $i$-th sample, target output.
- $\text{Loss}$: Loss function, measuring the error for a single sample.

Prediction score (linear model): The model's prediction is the dot product (weighted sum) of weights and features extracted from the input. This score defines the "score" driving regression and classification. It represents the model's confidence or estimate. This score is the input to the loss functions defined below.

$$
f_w(x) = w \cdot \phi(x)
$$

- $f_w(x)$: Prediction score, a scalar value.
- $w$: Weight vector, model parameters.
- $\phi(x)$: Feature vector, transformation of input $x$.

Residual (regression): A residual is the signed difference between the model's prediction and the actual value. It captures "overshoot" or "undershoot." Loss functions such as squared loss $(f_w(x) - y)^2$ or absolute loss $|f_w(x) - y|$ are symmetric functions of this residual. Specifically used for regression tasks to measure performance.

$$
\text{Residual} = f_w(x) - y
$$

- $f_w(x)$: Model prediction, scalar.
- $y$: True label, target continuous value.

Margin (classification): The margin is the product of the model score and the actual label. A positive margin indicates the score and label have the same sign (correct classification). A larger positive margin indicates higher confidence in a correct prediction. A negative margin indicates an error. Specifically used for binary classification to define surrogate losses like hinge or logistic loss. The distinction between the score $w \cdot \phi(x)$ and input $x$ is highlighted: $x$ can be non-numeric (e.g., a string), while $\phi(x)$ is a numeric feature vector.

$$
\text{Margin} = (w \cdot \phi(x)) y
$$

- $w \cdot \phi(x)$: Score, model output.
- $y$: True label, $\in \{+1, -1\}$.

### Regression Example and Optimization Algorithms

A concrete regression example illustrates the geometry of the loss surface, with Stochastic Gradient Descent (SGD) serving as a scalable alternative to standard gradient descent. Based on **vector calculus** (gradients) and **iterative optimization**. The **loss surface** visualizes the loss function as a surface in parameter space (weights). **Gradient Descent** is an iterative algorithm that moves weights in the direction of steepest loss descent. **Stochastic Gradient Descent (SGD)** significantly reduces computational costs for large datasets by estimating gradients using a single data point (or mini-batch).

Consider linear regression with three points. It shows how individual losses combine to form a collective training loss function in 2D weight space ( $w_1, w_2$ ). The resulting loss surface is a bowl-shaped paraboloid (in 3D), with the minimum located where the average error is minimized. Comparison of optimization algorithms:

- **Gradient Descent**: Updates weights using the gradient of the _entire_ training set. Slower for large $n$.
- **Stochastic Gradient Descent (SGD)**: Updates weights using the gradient of a _single_ sample. Faster and more scalable, though potentially less stable. It is the dominant method in modern machine learning.

Training loss (sum of squares): Total loss is the average of three quadratic "bowls." Each bowl prefers specific weight values ( $w_1=2, w_1=4, w_2=-1$ ). It visualizes the "tension" between different data points. The optimal $w_1$ will be the average (3) to balance conflicting demands between 2 and 4. This is a concrete instance of the $\text{TrainLoss}(w)$ objective defined earlier.

$$
\text{TrainLoss}(w_1, w_2) = \frac{1}{3} [ (w_1 - 2)^2 + (w_1 - 4)^2 + (w_2 + 1)^2 ]
$$

- $w_1, w_2$: Weights for features $x_1, x_2$, parameters to be optimized.
- $(w_1 - 2)^2$: Loss for Example 1, $x = (1, 0), y = 2$.
- $(w_1 - 4)^2$: Loss for Example 2, $x = (1, 0), y = 4$.
- $(w_2 + 1)^2$: Loss for Example 3, $x = (0, 1), y = -1$.

Stochastic Gradient Descent (SGD) Update: In each step, choose one sample, compute the gradient of its loss, and move weights a small distance $\eta$ in the opposite direction. Unlike standard Gradient Descent (GD) which sums over $n$ samples, SGD updates after processing just one sample. This allows it to handle datasets with millions of points. This implements the "optimization algorithm" part of the framework. SGD may be less stable because it uses a single sample instead of the full gradient. The example uses a simple identity feature map $\phi(x) = x$, but features can typically be much more complex.

$$
w \leftarrow w - \eta
\nabla 	\text{Loss}(x_i, y_i, w)
$$

- $w$: Weight vector, model parameters.
- $\eta$: Learning rate (step size), controlling the distance moved.
- $\nabla \text{Loss}$: Gradient of single-sample loss, the direction of steepest increase.
- $(x_i, y_i)$: Randomly selected training sample, a point in the dataset.

### Features and Hypothesis Classes

Feature templates are organizational principles for scaling feature engineering, introducing hypothesis classes—the set of all predictors reachable by varying model weights. Based on **functional analysis** and **representation theory**, the **feature map** $\phi(x)$ transforms raw data into a numeric vector space. The **hypothesis class** $\mathcal{F}$ is the space of functions explored by the learner. **Sparsity** refers to the property where most feature values are zero, common in high-dimensional domains like Natural Language Processing (NLP). Shifting focus from _how to learn_ ( $w$ ) to _how to represent_ ( $\phi(x)$ ). Feature engineering is considered a key bottleneck in practical machine learning.

- **Feature Templates**: Grouping features computed in the same way (e.g., "contains the word [blank]").

- **Representation**: Discrete/NLP data often uses **sparse vectors** (dictionaries/maps) for efficiency, while modern neural networks often use **dense vectors** (arrays).

- **Hypothesis Class**: Defined by the choice of $\phi(x)$. By changing features (e.g., adding $x^2$), we expand the hypothesis class, making the model more **expressive**. However, larger classes require more careful optimization and risk overfitting.

Hypothesis class: The set of all possible functions that can be created by fixing the feature map $\phi$ and varying the weights $w$. It defines the learner's "search space." If the true relationship between $x$ and $y$ is not in $\mathcal{F}$, the model can never reach perfection. Learning is the process of choosing the "best" $f \in \mathcal{F}$ based on training data.

$$
\mathcal{F} = \{ f_w : w \in \mathbb{R}^d \}
$$

- $\mathcal{F}$: Hypothesis class, a set of functions.
- $f_w$: Predictor with weight $w$, $f_w(x) = w \cdot \phi(x)$.
- $w$: Weight vector, element of $d$-dimensional space.

Expressivity comparison (linear vs. quadratic): $\mathcal{F}_2$ contains all functions in $\mathcal{F}_1$ (by setting $w_2=0$), plus more. Thus, $\mathcal{F}_2$ is more expressive. More expressive models can fit complex data (e.g., curves), but are harder to optimize and more likely to "identify" patterns in noise (overfit). This shows how feature engineering (adding $x^2$ as a feature) directly changes the hypothesis class. "Linear in weights" does not mean "linear in inputs." A model can be linear with respect to its parameters $w$ but non-linear with respect to raw input $x$. " $\Phi$ is the bottleneck ": If features cannot capture the signal, no amount of optimization will help.

$$
\mathcal{F}_1 = \{ x \mapsto w_1 x \} \quad \text{vs.} \quad \mathcal{F}_2 = \{ x \mapsto w_1 x + w_2 x^2 \}
$$

- $\mathcal{F}_1$: Linear hypothesis class, lines through the origin.
- $\mathcal{F}_2$: Quadratic hypothesis class, parabolas through the origin.
- $w_1, w_2$: Weights, parameters.

### Linear and Geometric Perspectives

Linear models can generate complex non-linear decision boundaries in input space by "lifting" data to a higher-dimensional feature space where relationships become linear. Based on **linear algebra** and **topology**. A **decision boundary** is the hypersurface where the model prediction score is zero. **Dimension lifting** is a technique for mapping low-dimensional non-linearly separable data into a higher-dimensional space to make it linearly separable (related to the **kernel trick**).

A common misconception is that "linear classifiers" can only learn straight lines. "Linear" here refers to the relationship between the score and weights $w$, or the score and features $\phi(x)$. Its relationship with the raw input $x$ is not necessarily linear. Key point: By choosing non-linear features (e.g., $x_1^2$), we can represent a circle as a linear plane in 3D feature space ( $x_1, x_2, x_1^2 + x_2^2$ ).

- **Discretization**: Transforming continuous values (time) into intervals.
- **Cross-features**: Capturing interactions between inputs (common words).

Non-linear decision boundary (circle): A linear equation in $w$, but it describes a circle in $x_1$ and $x_2$. It demonstrates that the power of "linear" models lies entirely in the richness of their feature map $\phi(x)$. This justifies using linear optimization methods for complex practical tasks.

$$
w \cdot \phi(x) = 0 \implies w_1 x_1 + w_2 x_2 + w_3 (x_1^2 + x_2^2) = C
$$

- $x_1, x_2$: Raw input, 2D coordinates.
- $x_1^2 + x_2^2$: Non-linear feature, squared distance to origin.
- $w_1, w_2, w_3$: Weights, parameters.
- $C$: Constant defining the circle's radius.

Feature engineering (discretization): Transforming a single continuous feature into a sparse binary feature vector. Linear models cannot naturally handle non-monotonic or non-linear trends in single numeric features. Bucketing allows the model to learn different weights for different time scales (e.g., "5 seconds" vs. "5 days" is very different). This is a practical application of the "feature template" concept. Adding too many features leads to **overfitting**, so don't just "make $\Phi$ as large as possible." The distinction between "linear in $w$" and "linear in $x$" is the most critical conceptual bridge.

$$
\phi(x) = [ \mathbb{I}(0 \le \Delta t < 5s), \mathbb{I}(5s \le \Delta t < 60s), \dots ]
$$

- $\Delta t$: Elapsed time, raw numeric input.
- $\mathbb{I}(\cdot)$: Indicator function, 1 if condition is true, 0 otherwise.

### Neural Network Basics

Neural networks are hierarchical structures of linear classifiers that decompose complex problems into learnable subproblems, effectively automating the feature extraction process. Based on **modular decomposition** and **layered architectures**, the core idea is to decompose complex non-linear functions into a set of simpler intermediate linear functions called **hidden units**. This shifts the burden from manual feature engineering to **parameter learning** across multiple layers.

Using car collision detection as an example for neural networks: The goal is to judge if two cars are "safe" (distance $\ge 1$). This is a non-linear problem in raw space $(x_1, x_2)$. Decomposition:

- **Subproblem 1 ( $h_1$ )**: Is Car 1 to the far right of Car 2? ( $x_1 - x_2 \ge 1$ )
- **Subproblem 2 ( $h_2$ )**: Is Car 2 to the far right of Car 1? ( $x_2 - x_1 \ge 1$)
- **Combination**: Safe if $h_1$ OR $h_2$ is true.

The key is that $h_1$ and $h_2$ can be represented as linear classifiers. In a neural network, the "features" used by the final classifier ( w ) are themselves the outputs of intermediate linear classifiers ( V ). Both intermediate weights ( V ) and final weights ( w ) are learned from data.

Subproblems as dot products: The first subproblem is a linear threshold function. In the car example, it checks $1x_1 - 1x_2 - 1 \ge 0$, equivalent to $x_1 - x_2 \ge 1$. It shows how to encode a high-level logical concept ("far right") as a simple dot product. This $h_1$ serves as a "learned feature" for the next layer.

$$
h_1 = \mathbb{I}(v_1 \cdot \phi(x) \ge 0)
$$

- $v_1$: Weight vector for hidden unit 1, e.g., $[-1, 1, -1]$.
- $\phi(x)$: Input features, $[1, x_1, x_2]$ (including bias).
- $\mathbb{I}(\cdot)$: Indicator (step) function, outputting 0 or 1.

Final prediction score: The final output is a weighted sum of subproblem results. This allows the network to implement logical operations like OR, AND (e.g., if $w_1=1, w_2=1$ and threshold is 0.5, it's an OR gate). This completes a two-layer hierarchy: Input Layer $\to$ Hidden Unit Layer $\to$ Output Score. The 1 in $\phi(x) = [1, x_1, x_2]$ is the "bias term," allowing thresholds to be non-zero. **Manual vs. Automated**: In the example, values for $v_1, v_2, w$ are known. In real neural networks, they are "unknowns" to be fitted through training. **"Linear" issue**: The final score is linear in $w$, but highly non-linear in $x$ due to nesting and indicator functions.

$$
\text{Score} = w_1 h_1 + w_2 h_2 + \dots
$$

- $w_i$: Output layer weights, combining subproblem results.
- $h_i$: Hidden unit outputs, subproblem results.

### Gradients and Neural Network Architecture

Transitioning from discrete step functions to continuous activation functions (e.g., Sigmoid) enables gradient-based optimization. Based on **calculus** and **activation functions**. Optimization with SGD requires objective functions with non-zero gradients. The **step function** used in the car example has zero gradients almost everywhere, making it impossible to "learn" via local updates. **Smoothing** (like sandpaper) replaces these step functions with differentiable curves like **Sigmoid** or **ReLU**.

A fatal flaw of the two-layer model: indicator functions have zero gradients. To solve this, apply the "sandpaper" method to smooth step functions into **logistic functions** (Sigmoid). **Symmetry breaking**: why weights must be randomly initialized. **ReLU**: Modern preference for Rectified Linear Units ( $\max(0, z)$ ) over Sigmoid due to "vanishing gradients" and simpler computation. Neural network definition:

- **Input Layer**: $\phi(x)$
- **Hidden Layer**: $h_j = \sigma(v_j \cdot \phi(x))$, where $\sigma$ is an activation function.
- **Output Layer**: $Score = \sum w_j h_j$.

Logistic (Sigmoid) activation function: Maps any real number to $(0, 1)$. Large positive $\to 1$, large negative $\to 0$. It approximates "True/False" behavior while remaining differentiable. It acts like "sandpaper," allowing gradients to flow through hidden units.

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

- $z$: Logit (input to sigmoid), $v_j \cdot \phi(x)$.
- $e$: Euler's number, $\approx 2.718$.

Derivative of the logistic function: Gradient of $\sigma'(z)$ relative to input, the rate of change. Gradients are non-zero everywhere. It is maximal at $z=0$ (steepest) and approaches zero as $z 	o \pm\infty$ (saturation). Since gradients are never exactly zero, SGD can always find a direction to update weights $v_j$. This enables the "learning strategy."

$$
\sigma'(z) = \sigma(z) (1 - \sigma(z))
$$

Single hidden layer neural network output: The final prediction is a weighted sum of $k$ learned non-linear features. This is the standard form of a simple Multi-Layer Perceptron (MLP). With enough hidden units, it can approximate any continuous function (Universal Approximation Theorem). Combines previous "decomposition" and "differentiability." **Vanishing Gradients**: While sigmoid gradients are non-zero, they become very small for large $|z|$, which is why ReLU is often preferred in deep networks. **Symmetry**: If weights $v_j$ are initialized to 0, all hidden units compute the same value and gradients update them identically. Random initialization is necessary to allow units to focus on different subproblems.

$$
f_{V, w}(x) = \sum_{j=1}^k w_j \sigma(v_j \cdot \phi(x))
$$

- $k$: Number of hidden units, hyperparameter (width).
- $V$: Hidden layer weight matrix, rows are $v_j$ vectors.
- $w$: Output weight vector, scalar $w_j$ for each $j$.

### Backpropagation and Computation Graphs

The backpropagation algorithm is a systematic method for computing gradients of any function composition via forward and backward traversals of a computation graph. Based on **computational differentiation** and **graph theory**. A **computation graph** is a Directed Acyclic Graph (DAG) where nodes represent operations and edges represent data flow. **Backpropagation** is an application of the **chain rule** from calculus, specifically for efficiently computing "root" output (loss) gradients with respect to all internal parameters (weights).

Rather than manually "computing" mathematical formulas for every new architecture, we treat functions as interconnected boxes (building blocks)—a modular approach to computing complex gradients.

Building blocks and local gradients:

- **Sum ( $a+b$ )**: Gradient is 1 for both inputs.
- **Product ( $ab$ )**: Gradient of $a$ is $b$; gradient of $b$ is $a$.
- **Max ( $a, b$ )**: Gradient is 1 for the larger input, 0 for the smaller.
- **Sigmoid ( $\sigma$ )**: Gradient is $\sigma(1-\sigma)$.

**Backpropagation Algorithm**: Foundations of modern libraries like PyTorch/TensorFlow, supporting "Autograd."

- **Forward Pass**: Compute the value of each node ( $f_i$ ) from input to output.

- **Backward Pass**: Compute the sensitivity of each node ( $g_i = \partial \text{Loss} / \partial f_i$ ) from output to input using the chain rule.

Chain rule (amplification): $\partial$ is the partial derivative operator, measuring sensitivity. Total sensitivity of output to input is the product of local sensitivities along the path. It allows computing global impact by observing local operations weights pass through. The core engine of the backward pass in backpropagation.

$$
\frac{\partial \text{Output}}{\partial \text{Input}} = \frac{\partial \text{Output}}{\partial \text{Node}} \times \frac{\partial \text{Node}}{\partial \text{Input}}
$$

Gradient of a product node: $a, b$ are multiplication inputs, can be weights or features. If $a$ changes slightly, output changes by that amount times $b$. In dot product $w \cdot x$, the gradient of weight $w_i$ is just the feature value $x_i$. One of five basic building blocks.

$$
\frac{\partial(a \cdot b)}{\partial a} = b, \quad \frac{\partial(a \cdot b)}{\partial b} = a
$$

Backpropagation recursive step: The total gradient for node $i$ is the sum of gradients returned by its children, each multiplied by the local derivative of that child's operation. Enables efficient computation even if a node's output is used by multiple others. Formally defines the backpropagation algorithm. **Modular Design** emphasizes that once you have "local gradients" for basic blocks, you can build any "castle" (complex model) and get gradients automatically. Complexity is $O(	\text{graph size})$, as efficient as the forward pass itself. **Step vs. Max**: Max gradients are non-zero at one of its inputs, while step gradients are zero everywhere. This is why Max functions (ReLU, Hinge loss) are optimizable.

$$
g_i = \sum_{j \in \text{children}(i)} g_j \cdot \frac{\partial f_j}{\partial f_i}
$$

- $g_i$: Gradient of loss at node $i$, "backprop value."
- $f_j$: Output value of node $j$, "forward value."

### Optimization Theory and Nearest Neighbors

Comparing various model types based on tradeoffs between optimization complexity (convexity), predictive power, and computational efficiency. Based on **convex optimization** and **metric spaces**. A **convex function** is one where any line segment between two points on the graph lies above the graph; such functions are "well-behaved" because any local minimum is a global minimum. **Nearest Neighbor (NN)** is a **non-parametric** method, meaning its complexity grows with dataset size rather than being determined by a fixed set of parameters. Optimization characteristics:

- **Linear Models**: Convex when combined with standard loss functions. Gradient descent is guaranteed to find global optima.
- **Neural Networks**: Non-convex. Optimization can get stuck in local optima, though they perform well in practice.
- **Nearest Neighbors**: Can create complex decision boundaries (**Voronoi diagrams**) but are slow. Prediction is expensive as it must search the whole training set.
  - **Training**: Just store the data.
  - **Prediction**: Find the closest point $x_i$ in training set to target $x$ and return its label $y_i$.

Nearest Neighbor decision rule: To classify a new point, find the "closest" person in the training set and copy their answer. An intuitive "lazy" learning algorithm. Assumes points close in feature space should have similar labels. Highly "expressive" as it assumes no specific functional form (unlike linear/NN).

$$
i^* = \arg \min_{i \in \{1 \dots n\}} \text{dist}(x, x_i) \implies f(x) = y_{i^*}
$$

- $\text{dist}$：Distance metric, often Euclidean.
- $x$：Query point, input to be classified.
- $x_i, y_i$：Training samples stored in memory.

**Voronoi Diagram**: 1-NN decision boundaries are piecewise linear, defined by perpendicular bisectors of lines connecting training points. **Parametric vs. Non-parametric**: Linear/NN are parametric (fixed weights $w$). NN is non-parametric (complexity proportional to $n$). **Curse of Dimensionality**: Distance becomes less meaningful in high-dimensional spaces, a common issue for NN.

Learning tradeoffs ("Pick Two" rule): No "perfect" model. Choose based on needs (e.g., avoid NN for real-time mobile prediction; use NN if data is complex). Provides a framework for model selection in engineering.

| Model Type | Fast Prediction | Easy to Learn | Powerful (Expressive) |

| :--- | :---: | :---: | :---: |

| Linear Models | Yes | Yes (Convex) | No |

| Neural Networks | Yes | No (Non-convex) | Yes |

| Nearest Neighbors | No | Yes (No Training) | Yes |

---

## 2. Paradigms and Search

State-based models from the perspective of Modeling, Inference, and Learning, distinguishing search problems from reflex-based models through future consequences of actions. **Sequential decision-making** is where an agent chooses a sequence of actions to reach a goal, not just a single output. Search is essentially finding valid action sequences under constraints. Unlike reflex models (e.g., image classifiers) where inference is simple function evaluation $f(x) 	o y$, search models explore state spaces to find action sequences. Examples include pathfinding, robot motion planning (translation/rotation joints), game solving (Rubik's cube, 15-puzzle), and machine translation (modeled as word concatenation). The core difference: Search problems must consider **future consequences** of current actions.

Reflex model inference: Output $y$ is computed directly from input $x$ via fixed function $f$. Represents the "reflex" paradigm where decisions are instantaneous without explicit future state modeling. Serves as a baseline for search problems where output is an action sequence. Qualitative distinction between reflex and search.

$$
f(x) \to y
$$

- $x$: Input data (e.g., image, feature vector).
- $f$: Model function (e.g., neural network).
- $y$: Output label or action (typically single value/category).

## 2.1 Search Problem Formalization and Tree Search

Search problems are formalized as tuples of states, actions, costs, and transitions, explored via a "hypothetical" search tree from start to end states. **Formalization** includes: **State Space** (all possible configurations), **Successor Function** (mapping from state and action to new state $Succ: S \to S$), **Cost Function** ( $Cost: S \to \mathbb{R}$ ), and **Search Tree** (conceptual tree of states and actions).

Formalizing components for search problems, e.g., **Transportation Problem**: Agent moves along blocks 1 to $N$. Two actions: walk ( $S \to S+1$, cost 1 ) or magic tram ( $S \to 2S$, cost 2 ). Goal: reach block $N$ with minimum cost. Implementing search: separate **Modeling** (problem definition) from **Inference** (search algorithm).

Search problem components: These five components fully define a deterministic search problem. This abstraction allows universal search algorithms (e.g., BFS, General Search) to apply to any problem mapped to this format. For efficiency, `succAndCost` combines `Actions`, `Succ`, and `Cost`. State is an integer, a simple state space. "Don't pass block N" constraint is handled in `succAndCost`.

$$
\{ s_{start}, Actions(s), Cost(s, a), Succ(s, a), IsEnd(s) \}
$$

- $s_{start}$: Initial state, e.g., all items on the same side.
- $Actions(s)$: Function returning executable actions in state $s$.
- $Cost(s, a)$: Function returning cost of action $a$ in state $s$.
- $Succ(s, a)$: Successor function, $s' = Succ(s, a)$.
- $IsEnd(s)$: Boolean function checking if $s$ is the goal.

**Transportation Problem Class**: Models the tram/walk problem. `startState` returns block 1. `isEnd` checks block $N$. `succAndCost` implements rules: walk adds 1 (cost 1); tram doubles state (cost 2) if not exceeding $N$. Encapsulates the "modeling" phase.

<details><summary>Code</summary>

```python
class TransportationProblem(object):
    def __init__(self, N):
        self.N = N
    def startState(self):
        return 1
    def isEnd(self, state):
        return state == self.N
    def succAndCost(self, state):
        # return list of (action, newState, cost)
        results = []
        if state + 1 <= self.N:
            results.append(('walk', state + 1, 1))
        if state * 2 <= self.N:
            results.append(('tram', state * 2, 2))
        return results
```

</details>

### Search Algorithms: Backtracking, DFS, BFS, and Iterative Deepening

Basic tree search algorithms represent tradeoffs between time complexity (often exponential) and space complexity (linear to exponential).

- **Branching Factor ( $b$ )**: Actions per state.
- **Depth ( $D$ )**: Max search tree depth.
- **Solution Depth ( $d$ )**: Depth of shallowest goal ( $d \le D$ ).
- **Time Complexity**: Number of nodes explored.
- **Space Complexity**: Nodes stored in memory simultaneously.

**Backtracking Search** explores the entire tree recursively. Flexible (handles any cost) but exponential $O(b^D)$. **Depth-First Search (DFS)** stops at the first solution (optimal if edge costs are zero). **Breadth-First Search (BFS)** searches layer by layer, optimal for constant costs, but exponential space $O(b^d)$ to store the frontier. **DFS Iterative Deepening (DFS-ID)** is a hybrid using a "leash" to limit DFS depth, increasing it iteratively to find shallow solutions like BFS while maintaining DFS linear space $O(d)$.

Algorithms prioritize different resources. Backtracking and DFS save space but might take more time in deep trees. BFS saves time for shallow goals but consumes much RAM. DFS-ID achieves $O(d)$ space and $O(b^d)$ time, efficient for shallow goals in large trees. Memory is often the bottleneck before time in AI; $O(b^d)$ space exhausts RAM quickly even at moderate depths. This represents the "inference" phase.

**Backtracking Search Implementation**: Recursive "hypothetical" exploration. Maintains `best` to track min cost path. `recurse` explores branches until `isEnd`, updating `best` if `totalCost` is lower. Recursion depth is $D$. Space complexity $O(D)$ as it only stores the current path.

<details><summary>Code</summary>

```python
def backtrackingSearch(problem):
    best = {'cost': float('inf'), 'history': None}
    def recurse(state, history, totalCost):
        if problem.isEnd(state):
            if totalCost < best['cost']:
                best['cost'] = totalCost
                best['history'] = history
            return
        for action, newState, cost in problem.succAndCost(state):
            recurse(newState, history + [(action, newState, cost)], totalCost + cost)
    recurse(problem.startState(), [], 0)
    return best
```

</details>

DFS-ID is compared to a dog on a leash that is gradually lengthened. If $d \ll D$, BFS time $O(b^d)$ is a "huge improvement" over $O(b^D)$. BFS space $O(b^d)$ is its main drawback.

### Dynamic Programming and State Space Reduction

Dynamic Programming (DP) reduces exponential search time to polynomial time by identifying and caching "future costs" of shared subproblems, provided state space is correctly defined and the graph is acyclic. **DP in Search**:

- **Future Cost ( $FutureCost(s)$ )**: Min cost from current state $s$ to the goal.
- **State**: Summary of past actions sufficient for optimal future decisions.
- **Memoization**: Storing results of expensive function calls to avoid recalculation.
- **Acyclic Graph**: No cycles, allowing natural ordering of subproblems.

DP avoids redundant subproblem computation. Core idea: **Future Cost Recursion**—cost from $s$ to goal is the minimum of action cost plus future cost from $s'$. "State" must contain all info for future decisions. Example: pathfinding with constraints (e.g., "no 3 consecutive odd cities") shows how to construct **Minimal Sufficient States**. Expanding state from "current city" to "current city + past path properties" solves constraints while keeping state space polynomial (e.g., $2N, 3N$ instead of $N^2$).

Future Cost Recursion: Best way from $s$ to goal is trying all first steps $a$ and choosing the best path from $s'$. Recursive definition breaks large problems into smaller ones. With memoization, each state's future cost is computed once. Transforms tree search (visiting state multiple times via different paths) into graph traversal.

$$
FutureCost(s) = \begin{cases} 0 & \text{if } IsEnd(s) \\
\min_{a \in Actions(s)} \{ Cost(s, a) + FutureCost(Succ(s, a)) \} & \text{otherwise} \end{cases}
$$

- $FutureCost(s)$: Optimal cost from state $s$ to goal.
- $s$: Current state.
- $a$: Action taken.
- $Cost(s, a)$: Cost of action $a$ in state $s$.
- $Succ(s, a)$: State after action $a$, denoted $s'$.

**DP with Memoization**: Uses `cache` (dictionary) to store computed values. Checks `cache` before recursive `min` operations. Achieves $O(N)$ time, where $N$ is the number of states.

<details><summary>Code</summary>

```python
cache = {} # state -> futureCost
def futureCost(state):
    if problem.isEnd(state): return 0
    if state in cache: return cache[state]

    result = min(cost + futureCost(newState)
                 for action, newState, cost in problem.succAndCost(state))

    cache[state] = result
    return result
```

</details>

"State summarizes past actions sufficient for optimal future choice." **Acyclic Requirement**: DP requires DAGs for subproblem ordering (no circular dependencies). Complexity: minimal states (e.g., `(currentCity, numOddVisited)`) keep problems solvable.

### Uniform Cost Search (UCS)

Uniform Cost Search (Dijkstra's Algorithm) extends search to cyclic graphs by exploring states in increasing order of "past cost." Uses Explored, Frontier, and Unexplored sets to guarantee optimal solutions with non-negative costs.

- **Past Cost ( $PastCost(s)$ )**: Min cost found so far from start to state $s$.
- **Dijkstra's Algorithm**: Foundation of UCS for shortest paths in non-negative edge weight graphs.
- **Explored Set**: States with known optimal paths.
- **Frontier Set**: "Known Unknowns"—reached states without confirmed optimal paths.
- **Unexplored Set**: "Unknown Unknowns"—not yet reached.

UCS solves DP's DAG limitation by focusing on **past cost** instead of future cost. It maintains a frontier and expands the state with the minimum past cost. This ensures that when a state moves to "Explored," its optimal path from start is found. Requires non-negative edge costs so costs only increase (or stay same) with depth. **Non-negative** assumption is explicit. **Cycles**: UCS handles cycles by tracking "Explored." **Optimality**: Finds lowest-cost path, not necessarily fewest edges (BFS). **Efficiency**: Frontier implemented as a priority queue (min-heap).

$$
PastCost(s') = PastCost(s) + Cost(s, a)
$$

- $PastCost(s')$: Cost to reach new state $s'$.
- $PastCost(s)$: Optimal cost to reach $s$, guaranteed when $s$ is removed from frontier.
- $Cost(s, a)$: Cost of action $a$ to reach $s'$, must be $\ge 0$.

UCS State Transition: Cost to $s'$ via $s$ is total cost to $s$ plus last step $a$. UCS updates beliefs about the best path to $s'$ with this value. If new cost is lower, update $s'$ in frontier. Shortest path inference, pushing forward from start rather than backward from goal like DP.

- **Init**: `Unexplored = {A,B,C,D}`, `Frontier = {(A,0)}`, `Explored = {}`.
- **Step 1**: Pop $A$ (min cost 0). Add to `Explored`. Find neighbors $B$ (cost 1), $C$ (cost 100). Add to `Frontier`.
- **Step 2**: Pop $B$ (min cost 1). Add to `Explored`. Find neighbor $C$. Cost via $B$ is $1+1=2$. Better than 100! Update $C$ in `Frontier` to 2.
- **Step 3**: Pop $C$ (min cost 2). Add to `Explored`. Find neighbor $D$. Cost via $C$ is $2+1=3$. Add $D$ to `Frontier`.
- **Step 4**: Pop $D$ (min cost 3). Goal reached!

Walkthrough shows UCS dynamically updating shortest paths (to $C$). Non-negative costs are necessary; negative costs break the greedy assumption that popping min cost guarantees optimality.

---

## 2.2 Search Algorithms, learning, A\*, and heuristic design

Formalizing search problems through state-action-successor definitions and establishing the theoretical foundation of UCS by proving it always expands the shortest path to any state. **Graph search theory** models problem-solving as traversing nodes (states) and edges (actions). Relies on **optimal substructure** and **greedy property** in non-negative cost graphs: exploring minimum cost frontier reaches global optima.

Basic search algorithms: DFS, BFS, DP, and UCS. Constraints: DFS requires zero costs for optimality, BFS requires uniform costs, DP requires DAGs, UCS requires non-negative costs. Search problem formalized as tuple: start state, action set, cost function, successors, and goal check. **UCS Correctness Theorem** proves the found path is optimal once a state moves from frontier to explored.

UCS Exploration Order (Implicit): UCS prioritizes states with minimum cumulative cost from start. Always choosing the "cheapest" node ensures no shorter paths to visited nodes are ignored. Core difference from BFS (depth) or DFS (stack).

$$
\text{Priority}(s) = \text{PastCost}(s)
$$

- $\text{Priority}(s)$: Sorting value for priority queue, lower is better.
- $\text{PastCost}(s)$: Cumulative cost from $s_{start}$ to $s$.

Path costs in UCS proof: Any alternative path to $s$ must leave "Explored" at some state $t$ and enter "Frontier" at state $u$. Path cost is at least cost to $t$ plus edge cost to $u$. Shows that with non-negative costs, no unexplored path can be cheaper than the current one. Inequality is key to proving current priority is the minimum cost for $s$.

$$
\text{Cost}( \text{GreenPath}) \ge \text{Priority}(t) + \text{Cost}(t, u)
$$

- $\text{GreenPath}$: Alternative path from $s_{start}$ to $s$.
- $t$: State in Explored where path exits.
- $u$: State in Frontier, successor of $t$.

**Uniform Cost Search**: Maintains frontier of discovered but not finalized states, sorted by `past_cost`. Iteratively "explores" cheapest state, checks goal, and updates neighbor costs. `PriorityQueue` implements the expansion order. `if s_prime not in explored` handles cycles.

<details><summary>Code</summary>

```python
def UniformCostSearch(s_start):
    frontier = PriorityQueue()
    frontier.update(s_start, 0)
    explored = set()
    while True:
        past_cost, s = frontier.remove_min()
        if IsEnd(s): return past_cost
        explored.add(s)
        for action, s_prime, cost in SuccessorsAndCosts(s):
            if s_prime not in explored:
                frontier.update(s_prime, past_cost + cost)
```

</details>

**Non-negative Constraint**: UCS correctness assumes $Cost(t, u) \ge 0$. With negative costs, "GreenPath" could become cheaper, breaking the greedy assumption.

**State Augmentation**: Cycles (e.g., city $n$ round trip) handled by adding direction (e.g., `(city, direction)`), converting cyclic graphs to larger DAGs for DP or larger state spaces for UCS.

### Learning Costs in Search (Structured Perceptron)

Structured learning in search contexts treats action cost estimation as a weight adjustment process, iteratively penalizing deviations between predicted and observed optimal paths. **Structured Prediction** is an ML subfield where output spaces consist of complex objects (e.g., paths) rather than scalars/categories. **Structured Perceptron** extends the classic perceptron, using generalized inference (search) to identify "best-scoring" (lowest cost) structures, updating weights based on differences between predicted and true structure features.

Learning as "Inverse Search": Search maps costs to actions; learning maps actions (optimal paths) back to costs. Structured Perceptron initializes costs (weights) and refines them. At each step, runs search (e.g., DP) to generate predicted paths. If prediction differs from truth, decrease weights of true path actions and increase weights of predicted path actions. Generalized to feature-based costs: $Cost(s, a) = w \cdot \phi(s, a)$.

Weight Update (Simplified): Lower costs for actions to be taken (true path) and raise costs for wrongly predicted actions. Pushes search towards the true path in subsequent iterations. "Error-driven" perceptron update.

$$
w_a \leftarrow w_a - 1 \quad (\forall a \in y)
$$

$$
w_a \leftarrow w_a + 1 \quad (\forall a \in y')
$$

- $w_a$: Weight (cost) of action $a$, initialized to 0.
- $y$: True optimal action sequence.
- $y'$: Predicted optimal sequence, output of $f(x; w)$.

Feature-based Generalized Cost: Edge cost is the dot product of weights and features. Allows generalization to states/actions with similar features (e.g., all "walk" actions share a feature). Transition from learning specific action costs to general cost functions.

$$
\text{Cost}(s, a) = w \cdot \phi(s, a)
$$

- $w$: Weight vector, learned parameters.
- $\phi(s, a)$: Feature vector for state $s$ and action $a$.

Structured Perceptron Update (Collins Algorithm): Updates global weight vector by subtracting true path features and adding predicted path features. Ensures "score" (total cost) of true path becomes smaller relative to predicted path. Vector space generalization.

$$
w \leftarrow w - \sum_{i=1}^{|y|} \phi(s_{i-1}, a_i) + \sum_{j=1}^{|y'|} \phi(s'_{j-1}, a'_j)
$$

- $\phi(s, a)$: Feature vector.
- $a_i, s_{i-1}$: True path actions/states.
- $a_j', s'_{j-1}$: Predicted path actions/states.

**Structured Perceptron**: Iterates through training samples. Uses current weights to "predict" best path via search. Adjusts weights on error. `predict` represents the inference step. Internal loop implements the update rule. Might not recover _exact_ weights but focuses on _proportions_ for same optimal behavior. **Overfitting**: simply "fits" training data. If training data only contains "walk," "tram" costs won't be learned correctly.

<details><summary>Code</summary>

```python
def StructuredPerceptron(examples):
    weights = { 'walk': 0, 'tram': 0 }
    for iteration in range(T):
        for x, y in examples:
            y_prime = predict(x, weights) # Calls Search (e.g., DP)
            if y != y_prime:
                for action in y:
                    weights[action] -= 1
                for action in y_prime:
                    weights[action] += 1
```

</details>

### A\* Search and Heuristics

A\* Search optimizes pathfinding by adding heuristic estimates of remaining distance to goal, guiding exploration and reducing unnecessary state expansion. **Informed Search** leverages domain knowledge (heuristics) for node prioritization. Relies on **potential functions** from physics/graph theory: heuristic $h(s)$ acts as a "potential," reshaping the cost landscape to favor paths to goal.

Comparing UCS (PastCost expansion) and A*. A* expands by `PastCost(s) + FutureCost(s)`. Since `FutureCost` is unknown, use heuristic $h(s)$. Mathematical trick: A\* is UCS on a modified graph with new edge costs $Cost'(s, a)$. Modification penalizes moving away from goal and rewards moving towards it based on heuristic changes.

A\* Exploration Order: Expands states based on sum of paid cost and estimated remaining cost. Prioritizes states appearing to be on the shortest path, even with slightly higher `PastCost`. "Informed" part of search.

$$
\text{Order} = \text{PastCost}(s) + h(s)
$$

- $\text{PastCost}(s)$: Actual cost from start to $s$.
- $h(s)$：Heuristic, estimated cost from $s$ to goal.

Modified Edge Cost ( $Cost'$ ): New cost is old cost plus change in heuristic. If $h$ decreases (moving towards goal), cost decreases. If $h$ increases, cost increases. Transforms search to UCS on modified graph where paths to goal are "cheaper." Formal mathematical bridge between UCS and A\*.

$$
\text{Cost}'(s, a) = \text{Cost}(s, a) + h( \text{Succ}(s, a)) - h(s)
$$

- $\text{Cost}(s, a)$: Original edge cost.
- $h(s)$: Heuristic at current state.
- $h( \text{Succ}(s, a))$: Heuristic at successor.

**A Search**: Implemented by wrapping original costs with heuristic modifications and passing to standard UCS.

<details><summary>Code</summary>

```python
# A* is essentially UCS with a modified cost function
def AStar(problem, h):
    def modified_cost(s, a):
        s_prime = problem.successor(s, a)
        return problem.cost(s, a) + h(s_prime) - h(s)

    # Run UCS using modified_cost
    return UniformCostSearch(problem_with_modified_costs)
```

</details>

**Heuristic Quality**: A* effectiveness depends on $h(s)$. If $h(s)=0$, A* is UCS. If $h(s)$ is true "Future Cost," finds optimal path without backtracking. **Negative Costs**: Poor $h(s)$ can introduce negative costs, breaking UCS. Leads to requirements for $h(s)$.

### Heuristic Properties (Consistency and Admissibility)

Heuristic consistency defined by triangle inequality guarantees modified A\* costs are non-negative, ensuring optimal paths are found more efficiently than UCS. **Triangle Inequality** is a metric space principle stating the shortest path between two points is a line; in search, heuristic estimates can't decrease more than the single-step cost. **Induction** proves link between consistency and admissibility.

Key properties: **Consistency** and **Admissibility**. Consistent heuristic satisfies: triangle inequality ( $h(s) \le Cost(s, a) + h(s')$ ) and zero at goal. Consistency is "stronger," ensuring $Cost'(s, a) \ge 0$ for UCS. A* correctness under consistency; efficiency: A* only expands states satisfying $PastCost(s) < PastCost(s_{end}) - h(s)$. **Admissibility** ( $h(s) \le FutureCost(s)$ ): all consistent heuristics are admissible, but not vice versa.

**Heuristic Consistency (Triangle Inequality)**: Estimate from $s$ must be $\le$ cost to $s'$ plus estimate from $s'$. Ensures A* modified cost $Cost'(s, a)$ is never negative. Since A* is UCS on modified costs, non-negative costs are necessary for UCS correctness. Core optimality requirement.

$$
\text{Cost}(s, a) + h(s') - h(s) \ge 0
$$

- $\text{Cost}(s, a)$: Edge cost $s \to s'$, non-negative for UCS.
- $h(s')$: Successor heuristic.
- $h(s)$: Current state heuristic.

A* Correctness (Path Cost Invariance): Total cost to goal in modified graph is original cost minus initial heuristic. Since $h(s_{start})$ is constant, minimizing modified cost is mathematically equivalent to minimizing original cost. Proves A* finds same optimal path as UCS if modified costs are valid.

$$
\sum \text{Cost}' = \sum \text{Cost} - h(s_{start})
$$

- $\sum \text{Cost}'$: Total path cost in modified graph.
- $\sum \text{Cost}$: Total path cost in original graph.
- $h(s_{start})$: Constant heuristic at start.

Efficiency Bounds: A* only expands states where total estimated cost ( $PastCost + h$ ) < actual optimal cost to goal. Explains why A* is more efficient than UCS. UCS expands $PastCost(s) < PastCost(s_{end})$. A\* narrows search space by subtracting $h(s)$ from the threshold. Advantages of higher heuristic values.

$$
\text{Explore} s \text{ if: } \text{PastCost}(s) < \text{PastCost}(s_{end}) - h(s)
$$

- $\text{PastCost}(s)$: Cost to $s$ computed during search.
- $h(s)$：Heuristic, informative estimate.

**Admissibility vs. Consistency**: Most natural heuristics (e.g., Manhattan distance) are consistent. Admissible but inconsistent heuristics can be built. For graph search (multiple paths to state), consistency is the standard requirement. **$h(s_{end})=0$**: Necessary for telescoping sum to simplify to a constant subtraction, preserving path ordering for optimal solutions.

### Relaxation-based Heuristic Design

Heuristics are systematically designed via "relaxation"—removing constraints from original problems to create simplified versions where exact solutions provide admissible and consistent estimates of original future costs. **Problem Relaxation Theory**: Problem $P$ relaxes to $P'$ such that every valid solution in $P$ is valid in $P'$ with $\le$ cost. **Subproblem Independence**: Complex state spaces decomposed into smaller components for simplified computation.

Ideally $h(s) \approx FutureCost(s)$, but must be cheap to compute. Solution: remove constraints to "relax" the problem. E.g., navigation with walls: "relaxing" removes walls, yielding Manhattan distance. 8-puzzle: allowing tile overlap decomposes into 8 independent tasks of moving each tile. Relaxed problem solutions are always consistent and admissible. Tradeoff: more relaxation makes heuristics easier to compute but "looser" (farther from true "future cost"), potentially reducing A\* efficiency.

Heuristics as Relaxed Solutions:

$$
h(s) = \text{FutureCost}_{ \text{relaxed}}(s)
$$

- $h(s)$：Heuristic at $s$, estimate for original.
- $\text{FutureCost}_{ \text{relaxed}}(s)$：Actual optimal cost in simplified problem via DP/closed-form.

To find a heuristic, define a simplified search version and use its true shortest path cost. Generates heuristics guaranteed to be admissible and consistent.

"Best" heuristic is the true "future cost," but as hard as original. Relaxation finds a compromise: much faster computation than original, still providing useful guidance. **Systematic Consistency**: Shortest paths in _any_ relaxed problem are automatically consistent. Simplifies research; no need to prove consistency from scratch, just define a valid relaxation.

---

## 3. Game Playing: Expectimax, Minimax, and Expectiminimax, as well as the Logical Breakdown of Alpha-beta Pruning and Its Computational Complexity

Games are two-player, zero-sum, fully observable, turn-based scenarios formalized via state-based models where utility is determined by terminal states.

- **Zero-sum Game**: Mathematical representation where one participant's gains exactly balance another's losses.
- **MDP vs. Games**: MDP involves agent against "nature" (probabilistic), while games involve agent against another "agent" (adversary).
- **Full Observability**: Assumes both players have perfect info (unlike Poker).

Comparing games to search and MDPs via the "bucket example" illustrating how different adversary assumptions (adversarial, random, cooperative) lead to different strategies. Turn-based, zero-sum. "Halving Game" as example: defining players, start state, end conditions, and successors.

**Formal Game Components:**

| Symbol | Meaning | Definition | Note |

| :--- | :--- | :--- | :--- |

| $S_{start}$ | Start state | Current state | Initial config |

| $Actions(s)$ | Possible actions | Current | State function |

| $Succ(s, a)$ | Successor function | Current | State from $(s, a)$ |

| $IsEnd(s)$ | Terminal check | Current | Boolean function |

| $Utility(s)$ | Agent utility | Current | Only for $IsEnd(s)$ |

| $Player(s)$ | Current player | Current | Player in control |

A game is a sequence of states where players alternate actions until reaching a terminal state with utility for the agent (negative utility for the adversary). Formalization treats games as search problems where terminal "utility" replaces "cost," and transitions involve multiple decision-makers. Maps intuitive "bucket example" to a rigorous algorithmic framework.

**Halving Game**: Turn-based, subtract 1 or halve. First to 0 wins. `player` alternates between $+1$ (agent) and $-1$ (adversary). `utility` only at 0. Zero-sum: $+\infty$ for win, $-\infty$ for loss. **Utility Values**: $\pm \infty$ simplifies win/loss. **State Representation** explicitly encodes "player" for current turn determination.

<details><summary>Code</summary>

```python
class HalvingGame:
    def __init__(self, N):
        self.N = N
    def startState(self):
        return (1, self.N) # Player +1, Number N
    def isEnd(self, state):
        player, number = state
        return number == 0
    def utility(self, state):
        player, number = state
        assert number == 0
        return player * float('inf') # +inf if agent wins, -inf if opponent
    def actions(self, state):
        return ['subtract', 'divide']
    def successor(self, state, action):
        player, number = state
        if action == 'subtract':
            return (-player, number - 1)
        elif action == 'divide':
            return (-player, number // 2)
```

</details>

---

## 3.1 Game Evaluation and Search Algorithms

Mathematical recurrences for evaluating fixed strategies, deriving Expectimax and Minimax for optimal decisions under random or adversarial assumptions. **DP in Decision Making**:

- **Policy Evaluation**: Determining values of states under fixed rules (policies).
- **Expectimax Search**: Algorithm for maximizing expected utility against random adversaries.

**Adversarial Robustness** and **Stochastic Advantage**: **Minimax Optimal Policy ( $\pi_{max}$ )** maximizes worst-case utility. **Expectimax Policy ( $\pi_{expectimax}$ )** maximizes average utility against known random adversaries. **Robustness**: capability to perform well even if adversary assumptions (e.g., randomness) are incorrect.

Performance against different adversaries (bucket example):

- Against Minimizer (adversary): $\pi_{max}$ outperforms others.
- Against non-Minimizer (e.g., random $\pi_7$): $\pi_{max}$ still guarantees minimax value, but might not be optimal compared to tailored $\pi_{expectimax}$.
- Knowing adversary's true policy is more advantageous than assuming adversarial behavior.

**Minimax value as Upper Bound against Minimizer**: Against adversarial opponent ( $\pi_{min}$ ), minimax strategy ( $\pi_{max}$ ) is best. Any other policy (e.g., expectimax assuming non-adversarial) results in equal or lower utility. Confirms $\pi_{max}$ as "safest" against adversaries.

$$
V(\pi_{max}, \pi_{min}) \ge V(\pi_{expectimax}, \pi_{min})
$$

- $V(\pi_1, \pi_2)$: Utility results given agent $\pi_1$ and opponent $\pi_2$.
- $\pi_{max}$: Minimax policy, maximizing worst-case.
- $\pi_{min}$: Opponent's minimizing policy.
- $\pi_{expectimax}$: Expectimax policy, maximizing based on random assumptions.

**Minimax value as Lower Bound**: $\pi_7$ is non-minimizing policy (e.g., random). $\pi_{max}$ guarantees at least minimax value ( $V(\pi_{max}, \pi_{min})$ ). If opponent is "simpler" than perfect minimizer, utility only increases. Confirms $\pi_{max}$ as "safest."

$$
V(\pi_{max}, \pi_{min}) \le V(\pi_{max}, \pi_7)
$$

**Optimality of Knowledge (Expectimax vs. Minimax)**: If opponent is truly random $\pi_7$, expectimax ( $\pi_{expectimax}$ ) achieves higher utility than conservative $\pi_{max}$. Highlights tradeoff between safety and exploitation. Minimax is safe, expectimax exploits "suboptimal" or "random" opponents.

$$
V(\pi_{max}, \pi_7) \le V(\pi_{expectimax}, \pi_7)
$$

Agents typically "don't see" true opponent policies; they make assumptions (adversarial or random). **Transitivity**: $V(\pi_{min}, \pi_{max}) \le V(\pi_{max}, \pi_7) \le V(\pi_{expectimax}, \pi_7)$ shows knowing the policy always improves (or maintains) performance.

### Improving Games and Computational Complexity

Extending game search to include "nature" via Expectiminimax, while solving exponential bottleneck via depth-limited search and heuristic evaluation. **Stochastic Search** and **Heuristics**:

- **Expectiminimax**: Search for games with adversarial (min/max) and random (probability) components.
- **Computational Complexity**: Time/space for game tree search ( $b^{2d}$ ).
- **Evaluation Function (Heuristic)**: Domain-specific function estimating non-terminal state values.

"Nature" as third participant with random behavior (e.g., flipping coin to move buckets). "Expectiminimax" tree includes max, min, and probability nodes. Full tree search for complex games (Chess $b \approx 35, d \approx 50$) is physically impossible. Solution: "Finite Depth Search"—recursion stops at fixed depth $d$, calling evaluation function to estimate state value.

Expectiminimax Recurrence: Adds "Expectation" to min/max for nodes with probabilistic outcomes (e.g., Backgammon, "shifting bucket").

$$
V_{expectiminimax}(s) = \begin{cases} Utility(s) & \text{if } IsEnd(s) \\
\max_{a \in Actions(s)} V_{expectiminimax}(Succ(s, a)) & \text{if } Player(s) = agent \\
\min_{a \in Actions(s)} V_{expectiminimax}(Succ(s, a)) & \text{if } Player(s) = against \\
\sum_{a \in Actions(s)} \pi_{coin}(s, a) V_{expectiminimax}(Succ(s, a)) & \text{if } Player(s) = coin \end{cases}
$$

- $V_{expectiminimax}(s)$：Generalized value under max, min, and probability.
- $\pi_{coin}(s, a)$："random" outcomes probability (e.g., coin flip).

Computational Complexity: Time grows exponentially with depth; memory grows linearly (DFS). Why Chess can't be "solved" with simple recursion; state space larger than atoms in universe.

$$
\text{Time Complexity} = O(b^{2d}), \quad \text{Space Complexity} = O(d)
$$

- $b$: Branching factor.
- $d$: Tree depth, steps per player.

Evaluation Function (Linear Model): Estimates state value via weighted sum of features (Chess: 'king count', 'mobility', 'center control'). "Weak estimate" for non-terminal states. "Future cost" heuristic for games.

$$
Eval(s) = w_1 f_1(s) + w_2 f_2(s) + \dots + w_n f_n(s)
$$

- $w_i$: Weights for feature importance.
- $f_i(s)$: Game-specific attributes.

Depth-limited Minimax: recursion stops at fixed `depth`, using `Eval(s)` at leaves. Handles exponential complexity.

<details><summary>Code</summary>

```python
def recurse(state):
    if game.isEnd(state):
        return (game.utility(state), None)

    player = game.player(state)
    choices = []
    for action in game.actions(state):
        v, a = recurse(game.successor(state, action))
        choices.append((v, action))

    if player == +1: # Agent (Maximizer)
        return max(choices)
    else: # Opponent (Minimizer)
        return min(choices)
```

</details>

**Lack of Guarantees**: Unlike A\* search, the evaluation functions in Minimax do not provide optimality guarantees (they are not necessarily "admissible").
**Branching Factor**: Each player's $2d$ actions (turns) result in an effective depth of $2d$.
**Learning**: The weights $w_i$ are typically "learned" rather than manually adjusted.。

### Optimization via Alpha-beta Pruning & Move

Optimization for Minimax, significantly improving search efficiency by maintaining bounds and pruning subtrees that won't affect the optimal result. **Branch and Bound Search**:

- **Pruning**: Eliminating branches.
- **$\alpha$ (Lower Bound)**: Best value Maximizer (agent) can currently guarantee.
- **$\beta$ (Upper Bound)**: Best value Minimizer (adversary) can currently force upon the agent.
- **Move Ordering**: Prioritizing "better" moves to maximize pruning chances.

Key idea: If value intervals of two branches don't overlap, no need to explore the "worse" one. Optimal path nodes must have same minimax value. Maintaining $a_s$ (max node lower bound) and $b_s$ (min node upper bound) allows skipping subtrees where $a_s \ge b_s$. Efficiency depends on "move order," best case $O(b^d)$ (compared to $O(b^{2d})$ ).

Pruning Condition: If current lower bound (best found so far) is $\ge$ opponent's upper bound (best they can force), branch is irrelevant. Pruning whole parts without losing optimality guarantee.

$$
a_s \ge b_s
$$

- $a_s$: Lower bound at $s$, Maximizer's best.
- $b_s$: Upper bound at $s$, Minimizer's best.

Alpha and Beta definitions (implementation specific): "bottleneck" values passed down. $\alpha$ tracks Maximizer's best move; $\beta$ tracks Minimizer's. "Global" $\alpha$ and $\beta$ per branch for efficient pruning check ( $\alpha \ge \beta$ ).

$$
\alpha_s = \max_{s' \le s} \{ a_{s'} \}, \quad \beta_s = \min_{s' \le s} \{ b_{s'} \}
$$

- $\alpha_s$: Current alpha, max of encountered lower bounds.
- $\beta_s$: Current beta, min of encountered upper bounds.

**Alpha-Beta Minimax**: Implements minimax with `if alpha >= beta: break`. `alpha = max(alpha, v)` updates agent's best; `beta = min(beta, v)` updates opponent's best. `break` prunes remaining actions.

<details><summary>Code</summary>

```python
def alphaBeta(s, alpha, beta):
    if IsEnd(s):
        return Utility(s)

    if Player(s) == agent: # Maximizer
        v = -inf
        for a in Actions(s):
            v = max(v, alphaBeta(Succ(s, a), alpha, beta))
            alpha = max(alpha, v)
            if alpha >= beta: # Prune
                break
        return v
    else: # Minimizer
        v = +inf
        for a in Actions(s):
            v = min(v, alphaBeta(Succ(s, a), alpha, beta))
            beta = min(beta, v)
            if alpha >= beta: # Prune
                break
        return v
```

</details>

**Move Ordering** importance: Seeing "best" move first maximizes $\alpha \ge \beta$ probability, pruning more. **Worst Case**: Poor move order (best move last) still requires $O(b^{2d})$. **Random Order**: Interesting result, random order is better than worst case ( $\approx O(b^{2 \cdot \frac{3}{4} d})$ ). **Optimization**: Max nodes order successors by decreasing `Eval(s)`; Min nodes by increasing `Eval(s)`. **Optimal Efficiency**: Alpha-beta searches twice the depth of standard Minimax for same budget.

---

## 3.2 TD Learning and Game Theory

### Evaluation Functions and Learning Foundations

Transitioning from exhaustive minimax to depth-limited search with learned evaluation functions for high-complexity games. **Minimax Search** and **Function Approximation**. Minimax is a decision rule for minimizing worst-case losses in zero-sum games. Evaluation functions (V) estimate state values without searching to the end, often approximated as weighted feature combinations.

Turn-based zero-sum games: high branching/depth makes exhaustive search impractical. Depth-limited recursion (D) calls evaluation at leaves. Manually designed traditionally, but we propose parameterizing via weights (w) and features ( $\phi$ ) for Supervised or Reinforcement Learning.

Minimax Recurrence: Calculating values by looking $D$ steps ahead. If $D=0$, use estimate. Otherwise, agent chooses best successor (max), opponent chooses worst (min). Informed decisions without seeing game end, balancing cost and strategy. Baseline "prediction" for learning.

$$
V(s, D) = \begin{cases} \text{Eval}(s) & \text{if } D=0 \\
\max_{s' \in \text{Succ}(s)} V(s', D-1) & \text{ if agent moves} \\
\min_{s' \in \text{Succ}(s)} V(s', D-1) & \text{ if opponent moves} \end{cases}
$$

- $V(s, D)$：Minimax value at state $s$ with depth $D$.
- $s$：State, board configuration.
- $D$：Depth limit, lookahead steps.
- $\text{Eval}(s)$：Static estimate of state value.
- $\text{Succ}(s)$：Reachable successors.

Linear Evaluation Function: Weighted sum of features. Simplifies complex states to a scalar value. Model architecture for learning: optimize weights $w_i$ using data.

$$
\text{Eval}(s) = V(s; w) = \sum_{i=1}^n w_i \phi_i(s)
$$

- $V(s; w)$：Learned estimate.
- $w_i$：Feature weights, importance.
- $\phi_i(s)$：Hand-designed features (e.g., queen count).

"V is a function of state parameterized by weights Ws." Motivation for TD Learning: transitioning from manual to learned weights.

### Temporal Difference (TD) Learning: Target, Gradient, and Update

TD learning rule as a gradient descent optimization to minimize squared error between current predictions and bootstrapping target values based on next state. **TD Learning** and **SGD**. TD is a bootstrapping method updating estimates based on other estimates rather than waiting for final outcomes. Combines Monte Carlo sampling and DP bootstrapping. SGD iteratively updates parameters (weights) to minimize loss.

"Experience Episodes" formalized as $(s, a, r, s')$. Goal: move current prediction $V(s; w)$ closer to "Target" (immediate reward $r$ plus discounted next state value $\gamma V(s'; w)$ ). TD update rule derived via gradients of squared error loss. In games, $\gamma=1$. Linear value function update details.

TD Target: Current state reflects "truth" of reward plus expected next state value. Core of bootstrapping. Use future prediction to train current state. Replaces final utility (unknown during game).

$$
\text{Target} = r + \gamma V(s'; w)
$$

- $\text{Target}$：Improved value estimate, "label" for regression.
- $r$：Immediate reward, often 0 before game end.
- $\gamma$：Discount factor; often 1 in games.
- $V(s'; w)$：Predicted value of next state.

Squared Error Loss: Minimize squared distance between prediction and target. Positive loss, harsh penalty for large deviations. Standard regression objective.

$$
\text{Loss}(w) = \frac{1}{2} (V(s; w) - \text{Target})^2
$$

- $\text{Loss}(w)$：Objective to minimize, measuring prediction error.
- $V(s; w)$：Current value estimate.

TD Update Rule (Gradient Descent): Updates weights by moving opposite to loss gradient. Step size depends on error $(V - Target)$ and learning rate. Iterative evaluation function improvement. General TD update form.

$$
w \leftarrow w - \eta [V(s; w) - \text{Target}]
\nabla_w V(s; w)
$$

- $w$：Weights/parameters.
- $\eta$：Learning rate (Eta).
- $\nabla_w V(s; w)$：Prediction gradient.

Linear TD Update: Gradient $\nabla_w (w \cdot \phi(s))$ is just feature vector $\phi(s)$. Updates move weights towards features of current state. Efficient and intuitive: underestimates are boosted, overestimates reduced.

$$
w \leftarrow w - \eta [w \cdot \phi(s) - (r + \gamma w \cdot \phi(s'))] \phi(s)
$$

**Bootstrapping Risk**: "Target" depends on $w$ but treated as constant during gradient derivation. Standard simplification in RL (semi-gradient methods) to avoid complex second-order dependencies. **Reward Sparsity**: $r$ often 0 until end; updates driven by value differences between $s$ and $s'$ ("Temporal Difference").

### TD Practice: Backgammon and Q-Learning Comparison

Linear TD update examples, comparing TD (on-policy value function) and Q-learning (off-policy action-value function). **On-policy Learning** application; $V(s)$ vs. $Q(s, a)$. On-policy (TD) evaluates/improves current decision policy. Off-policy (Q-learning) learns optimal policy without considering agent's specific actions (e.g., during random exploration). $V(s)$ estimates goodness of state; $Q(s, a)$ goodness of action $a$ in $s$. In known-rule games, $V(s)$ is often sufficient.

Applying linear evaluation to state sequences. Weights initialized to 0; prediction and target initially 0. Updates occur only when final rewards are received, correcting "under-adjustment." Subsequent rounds with new weights show "over-adjustment" needing further correction. TD evaluates policy $\pi$ via $V(s)$; Q-learning learns optimal $Q^*(s, a)$ without model rules.

Iterative weight update example: step-by-step. E.g., if $w=0$, $p=0, t=0$ until final $r=1, t=1$. Learning from delayed rewards. Updates only on non-zero prediction error (TD error). $\Delta w = -0.5 \times (-1) \times [1, 2]^T = [0.5, 1]^T$ shows single reward propagation and feature weight update.

$$
w \leftarrow w - \eta [p - t] \phi(s)
$$

- $p$：Prediction ( $w \cdot \phi(s)$ ).
- $t$：Target ( $r + \gamma V(s'; w)$ ).
- $\eta$：Learning rate (0.5 in example).
- $\phi(s)$：Feature vector (e.g., `[1, 2]`).

TD approximates current policy value $V^{\pi}$; Q-learning approximates optimal $Q^*$. In Chess/Backgammon with known rules (successors), $V(s)$ is enough for move selection via prediction. Model-free scenarios need $Q(s, a)$.

$$
V^{\pi}(s) \quad \text{vs.} \quad Q^{*}(s, a)
$$

- $V^{\pi}(s)$：Value of $s$ under $\pi$, TD target.
- $Q^{*}(s, a)$：Optimal state-action value, Q-learning target.
- $\pi$：Current exploration policy (on-policy).

<details><summary>Code</summary>

```python
# Pseudo-code
w = [0.0, 0.0]  # Initialize weights
eta = 0.5       # Learning rate

def episode_update(states, reward, w):
    for i in range(len(states) - 1):
        s = states[i]
        s_prime = states[i+1]

        # Calculate Target (t)
        if i == len(states) - 2: # Last step before terminal
            t = reward # r + 0
        else:
            t = 0 + sum(w[j] * s_prime.phi[j] for j in range(len(w)))

        # Calculate Prediction (p)
        p = sum(w[j] * s.phi[j] for j in range(len(w)))

        # Update weights
        for j in range(len(w)):
            w[j] = w[j] - eta * (p - t) * s.phi[j]
    return w
```

</details>

Iterates through episodes. Calculates targets/predictions for each $(s, s')$ transition, updating weights. Implement iterative updates, highlighting target sparsity before final reward. Terminal state update: $r=1, V(s')=0$. Key boundary case. Distinction between Q-learning and TD: on-policy vs off-policy, model-based vs model-free.

### Simultaneous Games: Mixed Strategies and Von Neumann's Theorem

Simultaneous zero-sum games, pure and mixed strategies. Double Finger Morra proof: mixed strategies make action order irrelevant, leading to Von Neumann's Minimax Theorem. **Simultaneous Games**, **Payoff Matrices**, **Mixed Strategies**. Unlike turn-based (Chess), simultaneous (RPS) requires actions without knowing opponent's choice. Payoff matrix shows utilities. **Pure Strategy**: deterministic choice. **Mixed Strategy**: probability distribution over actions, prevents predictability. **Von Neumann's Theorem**: optimal mixed strategy exists in finite zero-sum two-player games where reveal order doesn't change game value.

Shift from turn-based to simultaneous zero-sum games (Double Finger Morra). Payoff matrix $V(a, b)$ from Player A's perspective. Pure (prob 1) vs Mixed (prob 0.5/0.5 etc.). Value $V(\pi_A, \pi_B)$ as expected payoff.

Sequential analysis of simultaneous games: pure strategies favor the second player. However, if Player A reveals a mixed strategy ( $p$ for action 1, $1-p$ for 2 ), Player B's optimal response is always a pure strategy. A's optimal $p$ minimizes B's exploitation capability, found at the intersection of B's pure strategy expected payoffs. Game value is same regardless of who reveals optimal mixed strategy first.

Expected Value of Mixed Strategy: weighted sum of all payoffs by action probabilities. Allows evaluation of non-deterministic policies. Replaces deterministic nodes with expectations over distributions.

$$
V(\pi_A, \pi_B) = \sum_{a \in A, b \in B} \pi_A(a) \pi_B(b) V(a, b)
$$

- $V(\pi_A, \pi_B)$：Expected game value.
- $\pi_A(a)$：Prob A chooses $a$.
- $\pi_B(b)$：Prob B chooses $b$.
- $V(a, b)$：Payoffs from matrix.

Player A Expected Payoff Equations: If A uses mixed strategy $p$, linear equations represent A's expected payoff depending on B's pure choice 1 or 2. B chooses the option minimizing A's payoff ( $\min(5p-3, -7p+4)$ ). A chooses $p$ to maximize this minimum. Setting equations equal ( $5p-3 = -7p+4$ ) finds optimal $p$ ( $7/12$ ), making B indifferent between 1 and 2, ensuring minimax value ( $-1/12$ ).

$$
E[ \text{Payoff} | B=1] = 2p + (-3)(1-p) = 5p - 3
$$

$$
E[ \text{Payoff} | B=2] = (-3)p + 4(1-p) = -7p + 4
$$

- $p$：Prob A chooses 1.
- $1-p$：Prob A chooses 2.
- $B=1, B=2$：B's pure reaction choices.

B's optimal response is a pure strategy: key insight. If A fixes $\pi_A$, B aims for $\min_{\pi_B} V(\pi_A, \pi_B)$. Since linear in $\pi_B$, minimum must be at probability simplex boundaries (pure strategies). Optimization (finding intersections) generalizes to $n$ actions via Linear Programming duality.

### Non-Zero-Sum Games: Nash Equilibrium and Prisoner's Dilemma

Non-zero-sum games: utilities not purely opposing. **Nash Equilibrium**: state where no player has incentive to unilaterally deviate. Prisoner's Dilemma example. **Non-Zero-Sum**, **Nash Equilibrium**. One player's gain isn't necessarily another's loss; participants can mutually benefit (cooperative) or hurt. Nash Equilibrium is a system stability point where knowing others' strategies, no player benefits from only changing their own.

Real-world interactions between pure competition and cooperation. Payoff matrix with independent utilities ( $V_{Ae} -V_B$ ). Von Neumann's Theorem only for zero-sum; need weaker solution: Nash Equilibrium. Prisoner's Dilemma: "Refuse/Refuse, 1yr each" vs "Testify/Testify, 5yrs each." Best collective result (Refuse/Refuse, 1yr each) is unstable. If one Refuses, other is incentivized to Testify (0yrs vs 10yrs). Only stable state—Nash Equilibrium—is both Testify (5yrs each), as neither can improve by unilaterally switching to Refuse. Nash Existence Theorem guarantees at least one (possibly mixed) equilibrium in finite games.

Nash Equilibrium Condition: Strategy pair $(\pi\*_A, \pi\*_B)$ is Nash if A's payoff can't improve by switching to any $\pi_A$ while B stays $\pi^*_B$, and vice versa. Mathematical condition for stability. No incentive to "deviate." Generalizes minimax (only $V_A = -V_B$) to any payoff structure.

$$
V_A(\pi\*_A, \pi\*_B) \ge V_A(\pi_A, \pi\*_B) \quad \text{and} \quad V_B(\pi\*_A, \pi\*_B) \ge V_B(\pi\*_A, \pi_B)
$$

- $\pi\*_A, \pi\*_B$：Stable Nash Equilibrium strategies.
- $V_A, V_B$：Expected utilities/payoffs.
- $\pi_A, \pi_B$：Alternative strategies (unilateral deviation).

Payoff Matrix (Non-Zero-Sum): Each cell contains a tuple $(V_A, V_B)$. Breaks zero-sum assumption for scenarios like Prisoner's Dilemma where cooperation is possible but unstable.

$$
\text{Payoff}(a, b) = (V_A(a, b), V_B(a, b))
$$

- $V_A, V_B$：Payoffs for $(a, b)$, e.g., prison years.

Contrast between "Socially Optimal" (Refuse/Refuse, 1yr) and Nash Equilibrium (Testify/Testify, 5yrs). Nash is strategic stability, not global utility maximization. Nash Existence Theorem guarantees equilibria under mixed strategies. Prisoner's Dilemma has a pure strategy equilibrium, easy to explain but not guaranteed for all games.
