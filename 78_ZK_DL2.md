# ETAAcademy-ZKMeme: 78. ZK Deep Learning 2

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>78. ZKDL2</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZKDL2</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Practical Deep Learning II: Engineering Hacks for Theory, Algorithms, Diagnostics, and Workflows

Modern machine learning is less a collection of disconnected algorithms and more a rigorous, diagnosable engineering discipline that prioritizes systematic strategy over brute-force experimentation. This philosophy emphasizes building simple baselines first and using sophisticated diagnostics—such as learning curves to distinguish between high bias (underfitting) and high variance (overfitting)—to guide every optimization step. 

At the heart of supervised learning lies the unification principle of Generalized Linear Models, which demonstrates that diverse tasks like regression and classification share a common mathematical foundation through the Exponential Family of distributions. 

This mathematical elegance extends to Support Vector Machines, where the kernel trick allows for linear classification in infinite-dimensional spaces without ever explicitly calculating them. 

As models scale into the realm of Deep Learning, they leverage hierarchical compositions to learn progressively abstract features, while modern optimization techniques like ReLU activations and Momentum help bypass classical hurdles like vanishing gradients. 

For unlabeled data, algorithms like Expectation-Maximization and Factor Analysis provide the machinery to uncover latent structures even in high-dimensional settings where standard models fail. 

Finally, the paradigm of Reinforcement Learning shifts the focus toward long-term decision-making through the Bellman Equation, though it demands careful reward specification to prevent agents from technically succeeding at the math while failing the intended real-world task. Ultimately, by integrating these theoretical insights with an iterative debugging workflow, practitioners can transition from blind trial-and-error to a state of principled machine learning engineering.

---

## 1. Generalized Linear Models (GLM) and Discriminative Learning Algorithms

Machine learning can be broadly categorized into several paradigms:

* **Supervised Learning** (regression and classification)
* **Machine Learning Strategy / Learning Theory** (how to systematically improve models)
* **Deep Learning** (neural networks and backpropagation)
* **Unsupervised Learning** (clustering, ICA, representation learning)
* **Reinforcement Learning** (learning behavior through rewards)

The core idea of **supervised learning** is to learn a mapping from inputs $X$ to outputs $Y$, given labeled training data.

**Regression and Classification**

* **Regression** predicts continuous values.
  Example: predicting house prices ($Y$) based on house size ($X$).

* **Classification** predicts discrete labels (e.g., 0/1 or multiple classes).
  Example: predicting whether a tumor is benign or malignant based on features such as tumor size and patient age.

Common supervised learning algorithms include **Logistic Regression** and **Support Vector Machines (SVMs)**.

**Feature Representation**

In real-world problems, inputs are rarely one-dimensional. Instead, the input $X$ is a **feature vector**:

$X = (x_1, x_2, x_3, \dots, x_n)$

For example, in tumor classification:

* $x_1$: tumor size
* $x_2$: patient age
* $x_3$: cell shape uniformity
* $x_4$: cell adhesion

When there are **two features**, data points can be visualized in a 2D plane, where each point represents a sample and different symbols indicate different classes. The goal of a learning algorithm is to find a **decision boundary** that separates positive and negative examples.

* **Logistic Regression** learns a **linear decision boundary** (a line or hyperplane) and outputs probabilities, making it suitable for classification.
* **Support Vector Machines** can operate in extremely high—even infinite—dimensional feature spaces.

#### The Kernel Trick and Infinite-Dimensional Features

A natural question arises: how can a computer handle infinite-dimensional vectors?

SVMs use the **kernel trick**, which avoids explicitly constructing high-dimensional feature vectors. Instead, they compute inner products directly in the original space, achieving the same effect as performing linear classification in a high-dimensional (or infinite-dimensional) space.

#### Machine Learning Strategy and Learning Theory

Machine learning is not just about writing code—it is a **systematic engineering process** that emphasizes reproducibility, diagnosability, and decision-making.

Many teams waste time by:

* Blindly collecting more data
* Randomly tuning hyperparameters

Experienced practitioners focus on **making informed decisions**, such as:

* Should we collect more data or change the model?
* Is the problem bias-dominated or variance-dominated?

Machine learning is fundamentally about **abstract modeling**, not visualization. While 2D or 3D plots are useful for intuition, real-world problems often involve very high-dimensional spaces that can only be understood mathematically.

### Deep Learning

**Deep learning** is a subset of machine learning based on neural networks.

For example, in autonomous driving:

* Humans provide training data in the form of images and steering actions.
* Initially, the neural network produces random outputs.
* Through **backpropagation** and **gradient descent**, the network gradually improves and eventually learns to drive autonomously.

### Unsupervised Learning

In **unsupervised learning**, we are given data $X$ without labels $Y$. The goal is to discover structure or patterns within the data.

Examples include:

* **Clustering**: grouping similar data points together

  * Google News automatically groups articles about the same event.
  * Market segmentation divides customers into distinct groups.
* **Independent Component Analysis (ICA)**:

  * The classic *cocktail party problem*, where multiple microphones record overlapping voices and the algorithm separates individual speakers.

### Reinforcement Learning

**Reinforcement learning** learns through rewards and penalties, similar to training an animal.

For example:

* You cannot explicitly tell a helicopter how to adjust its controls every second.
* Instead, you reward good behavior (stable flight) and penalize bad behavior (crashes).
* The algorithm learns a policy that maximizes long-term reward.

---

## 1.1 Linear Regression & Gradient Descent

Linear Regression and Gradient Descent are the foundation of all algorithms. The **Hypothesis** is the starting point for all supervised learning. Linear regression assumes a linear relationship between input $x$ and output $y$: $h_\theta(x) = \theta_0 + \theta_1 x_1 + \dots$. For easier matrix operations, adding an $x_0 = 1$ (intercept/bias) allows it to be written as a vector inner product $h_\theta(x) = \theta^T x$. All subsequent advanced algorithms (including neural networks) are based on this vectorized form, where $\theta$ are the parameters or weights. Choosing the right $\theta$ makes the predicted value $h(x)$ and the true value $y$ as close as possible. The standard measurement is **Mean Squared Error** $J(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$. Using squared error is equivalent to assuming errors follow a **Gaussian Distribution** (Maximum Likelihood Estimation under Gaussian noise assumption). The $1/2$ is for convenience during differentiation, as the 2 from the derivative cancels the $1/2$. The task is to find a $\theta$ that minimizes $J(\theta)$.

**Method 1: Gradient Descent** is an iterative algorithm. Imagine you are standing on a mountain (the surface of the Cost Function) and want to descend as fast as possible (minimize the Cost). You should take a step in the steepest downward direction: $\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$. $\alpha$ is the **Learning Rate**. If it is too large, you might overshoot (oscillate or diverge); if it is too small, convergence will be too slow.

$\frac{\partial}{\partial \theta_j} J(\theta)$ is the **Gradient**, which determines the direction of descent. Two main gradient descent strategies:
- **Batch Gradient Descent**: $\frac{\partial J}{\partial \theta_j} = \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$. Every parameter update requires traversing all $m$ training samples to calculate the gradient. If you have 100 million data points, every step takes 100 million calculations, which is too slow.
- **Stochastic Gradient Descent (SGD)**: Each parameter update uses only **1 training sample**. For $i=1$ to $m$: $\theta_j := \theta_j - \alpha (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$. Take a step for every data point encountered. It's extremely fast and handles huge datasets.

**Method 2: Normal Equations** is the "shortcut" given by linear algebra. Core idea: Stop iterating and use a mathematical formula to directly calculate the lowest point. For matrix $X$ (containing all samples) and vector $y$, the Cost Function can be written in matrix form: $J(\theta) = \frac{1}{2} (X\theta - y)^T (X\theta - y)$. The **Closed-form Solution** is found by setting $\nabla_\theta J(\theta) = 0$, leading to $X^T X \theta = X^T y$, which gives $\theta = (X^T X)^{-1} X^T y$. This essentially sets the derivative to 0 and solves for $\theta$. The cost is the step $(X^T X)^{-1}$, which is matrix inversion. If the number of features $n$ is large (e.g., 10,000D), matrix inversion will be slow ($O(n^3)$ complexity).
**Engineering Recommendation**:
- If $n < 10,000$: Use **Normal Equation** (one step, no need to tune the learning rate).
- If $n$ is large: Use **Batch Gradient Descent**.
- If $m$ is huge: Use **Stochastic Gradient Descent (SGD)**.

---

## 1.2 Locally Weighted & Logistic Regression

If the data is non-linear, use **Locally Weighted Regression (LWR/LOESS)**. For classification problems (output 0 or 1), use **Logistic Regression**.
LWR is for when linear regression is too rigid (**Underfitting**) and high-order polynomials are too "crazy" (**Overfitting**).
- **Underfitting**: Using a straight line to fit curved data (e.g., house prices vs. area might not be strictly linear). Too few features, the model is too simple.
- **Overfitting**: Using high-order polynomials (e.g., $x^{10}$) to force-fit every point. Too many features, the model is too complex, and generalization is poor.

If it's hard to find a globally universal function, why not perform a "temporary" linear regression using only data near the prediction point? This is a **Non-parametric algorithm**. Parametric learning algorithms (like linear regression) allow you to throw away data once $\theta$ is optimized. Non-parametric algorithms require keeping all training data for every prediction, recalculating weights for all points (higher weight for closer points, lower for distant ones). Computational cost increases linearly with data size. Focus on samples near the prediction point:
- **Cost Function**: $\sum_{i} w^{(i)} (y^{(i)} - \theta^T x^{(i)})^2$, with added weight $w^{(i)}$.
- **Weighting function**: $w^{(i)} = \exp\left(-\frac{(x^{(i)} - x)^2}{2\tau^2}\right)$
    - If sample $x^{(i)}$ is **very close** to prediction point $x$: $w^{(i)} \approx 1$ (high importance).
    - If sample $x^{(i)}$ is **far** from $x$: $w^{(i)} \approx 0$ (ignored).
    - **$\tau$ (Bandwidth parameter)**: Determines the "range" of neighbors. Smaller $\tau$ only looks at extremely close points (prone to overfitting); larger $\tau$ considers points further away (prone to underfitting).

The reason for using **Squared Error (Probabilistic Interpretation)** $\sum (y-h(x))^2$ is that if we assume the errors $\epsilon$ follow a **Gaussian Distribution** and are IID (Independent and Identically Distributed), the **Maximum Likelihood Estimation (MLE)** leads exactly to the Least Squares solution.
**Assumption**: $y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$, where error $\epsilon^{(i)}$ includes factors not modeled (like the homeowner's mood) and random noise.
Assuming $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$ and is IID: $p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)$. We find $\theta$ that maximizes the probability of observing the current data $(y^{(i)})$. The likelihood function $L(\theta)$ is the product of all sample probabilities. Maximizing $L(\theta)$ is equivalent to maximizing $\log L(\theta)$. The derivation shows:
**Maximizing Likelihood Probability <==> Minimizing Squared Error** (Minimize $\frac{1}{2} \sum (y^{(i)} - \theta^T x^{(i)})^2$).

**Logistic Regression** is a **Classification** algorithm despite the name. Why not use linear regression for classification? It performs poorly (extremely sensitive to outliers, and outputs are not between 0-1). To constrain outputs to $(0, 1)$, the **Sigmoid** function is introduced: $g(z) = \frac{1}{1+e^{-z}}$.
The hypothesis becomes $h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$, where $h_\theta(x)$ represents the **probability** that $y=1$ given $x$, i.e., $P(y=1|x; \theta)$.

**Cost Function**: Squared error is no longer suitable (it would be non-convex with many local optima). Instead, use **Log Likelihood**: $\ell(\theta) = \sum_{i=1}^m \left[ y^{(i)} \log h(x^{(i)}) + (1-y^{(i)}) \log (1-h(x^{(i)})) \right]$, derived from the merged probability formula: $P(y|x;\theta) = (h_\theta(x))^y (1-h_\theta(x))^{1-y}$. We maximize this using **Gradient Ascent**, with the update rule: $\theta_j := \theta_j + \alpha \sum_{i=1}^m (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}$. This formula looks identical to linear regression. The only difference is the definition of $h_\theta(x)$ (Sigmoid vs. Linear). This is not a coincidence; both belong to **Generalized Linear Models (GLM)**.

Gradient descent is "feeling for stones to cross the river" (1st derivative). A faster optimization algorithm is **Newton's Method**, which "reads a map and takes a straight road" (2nd derivative), using curvature information to find where the derivative is 0.
One-dimensional update rule: $\theta := \theta - \frac{f(\theta)}{f'(\theta)}$. Here we want to maximize $\ell(\theta)$, i.e., find where $\ell'(\theta)=0$. Applying this to $\ell'(\theta)$: $\theta := \theta - \frac{\ell'(\theta)}{\ell''(\theta)}$.
In the multi-dimensional case, the second derivative becomes the **Hessian Matrix ($H$)**: $\theta := \theta - H^{-1} \nabla_\theta \ell$.
It converges extremely fast (**Quadratic convergence**), usually in a few iterations. The downside is requiring the inverse of the Hessian $H^{-1}$. If there are 10 features, $H$ is $10 \times 10$. If $n$ is huge (e.g., tens of thousands), calculating the inverse matrix is extremely slow ( $O(n^3)$ ).
**Engineering advice**: Use Newton's method for few features ($n < 1000$); use Gradient Descent for large $n$.

---

## 1.3 Perceptron & GLMs

Linear Regression and Logistic Regression are unified under a broader framework: **Generalized Linear Models (GLMs)**.

**The Perceptron**: Although not widely used today, it is historically significant as the predecessor to neural networks. It is similar to logistic regression but performs **Hard Classification** with no probabilistic interpretation. It uses an abrupt 0/1 step function rather than a smooth probability transition. Because it cannot solve simple XOR problems (it only works for linearly separable data), it led to the first "AI Winter." It serves primarily as an example of an algorithm without a probabilistic basis.

- **Activation Function**:
  
  g(z) = 1   if z ≥ 0  
  g(z) = 0   if z < 0  
- **Update Rule**: $\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}$ — this form is almost identical to the gradient descent for Linear/Logistic regression!
- **Geometric Intuition**: If a classification error occurs, the decision boundary (normal vector $\theta$) is "pulled" towards the sample (a form of vector addition).

**Exponential Family**: A class of distributions with excellent mathematical properties, serving as the solid foundation for GLMs. Examples include **Gaussian** (real numbers, $y \in \mathbb{R}$), **Bernoulli** (0/1, binary classification), **Poisson** (counts, e.g., website clicks), **Gamma/Exponential** (positive numbers), etc. Once a distribution is proven to belong to the exponential family, subsequent derivations follow naturally. A distribution belongs to the exponential family if its PDF can be written as:
$p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta))$
- $\eta$: **Natural Parameter**.
- $T(y)$: **Sufficient Statistic** (usually $T(y)=y$).
- $a(\eta)$: **Log Partition Function** (used for normalization; its derivative yields mean and variance).

**Generalized Linear Models (GLM)**: We don't need to reinvent the wheel for every data type:
- Predicting house prices (real numbers)? Assume $y$ is Gaussian → Derives **Linear Regression**.
- Predicting cancer (0/1)? Assume $y$ is Bernoulli → Derives **Logistic Regression**.
- Predicting website clicks (integers)? Assume $y$ is Poisson → Derives **Poisson Regression**.
Regardless of the chosen distribution, as long as it's in the exponential family, the derived gradient ascent update rule is always the same: $\theta_j := \theta_j + \alpha (y^{(i)} - h(x^{(i)})) x_j^{(i)}$. This is the beauty of mathematical unification.
**Three Assumptions of GLMs**:
(1) **Distribution**: Given $x$, $y$ follows an exponential family distribution $ExponentialFamily(\eta)$.
(2) **Linear Predictor**: We aim to predict the expectation of $T(y)$, assuming the natural parameter $\eta$ is a linear combination of inputs: $\eta = \theta^T x$.
(3) **Prediction Goal**: The algorithm output $h(x)$ should equal the expected value $E[y|x]$.

**Softmax Regression**: Solves **Multi-class Classification** (e.g., classifying digits 0-9). It is a natural generalization of Logistic Regression (and a GLM instance for the Multinomial distribution). $P(y=j|x) = \frac{e^{\theta_j^T x}}{\sum_{l=1}^k e^{\theta_l^T x}}$, where the denominator ensures probabilities sum to 1.
- **Loss Function**: **Cross Entropy**, which is essentially Maximum Likelihood Estimation.
- **Hypothesis**: For $k$ classes, we have parameters $\theta_1, \dots, \theta_k$.
- **Cost Function**: Minimize $-\sum_{i=1}^m \sum_{j=1}^k \mathbb{1}\{y^{(i)}=j\} \log p(y^{(i)}=j|x^{(i)})$.

---

## 1.4 Decision Trees

Decision Trees recursively partition the input space. For any node $m$, let $p_{mk}$ be the proportion of training observations in that node belonging to class $k$. The "greedy" search finds the best split by minimizing the impurity measure $Q_m$.
- Although **Misclassification Loss** $1 - \max_k(p_{mk})$ is simple, it's often insensitive to probability changes. Thus, researchers prefer the **Gini Index**: $\sum_{k=1}^K p_{mk}(1 - p_{mk})$ or **Cross Entropy**: $-\sum_{k=1}^K p_{mk} \log(p_{mk})$. These strictly concave functions better reward splits that increase "purity" (driving $p_{mk}$ towards 0 or 1).
- The algorithm chooses a split $S$ that maximizes **Information Gain**: $Q_{\text{parent}} - (\frac{N_L}{N} Q_{\text{left}} + \frac{N_R}{N} Q_{\text{right}})$.
- To handle high variance in deep trees, techniques like **Pruning** are used to remove branches that don't statistically improve validation performance.

**Ensemble Methods**: Averaging $B$ independent variables (models) with variance $\sigma^2$ can reduce total variance to $\frac{\sigma^2}{B}$. However, since models trained on the same data are correlated (coefficient $\rho$), the actual ensemble variance is $\rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$. This shows ensemble methods are most effective when correlation $\rho$ between models is reduced.
- **Bagging (Bootstrap Aggregating)**: Simulates independent datasets using the Bootstrap method—sampling N observations with replacement to create B new training sets $Z^{*b}$. The ensemble prediction is the average:
```math
 \hat{f}_{bag}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}^{*b}(x).
```
- **Random Forests**: Improve Bagging by injecting more randomness. At each split, they only consider a random subset of $m < p$ features, effectively reducing $\rho$ and overall variance.
- **Boosting**: Unlike Bagging, Boosting fits an additive model sequentially: $F_M(x) = \sum_{m=1}^M \alpha_m h_m(x)$. Each new weak learner $h_m(x)$ is trained to correct the previous iteration's errors. In **Gradient Boosting**, this means fitting $h_m$ to the residuals $y_i - F_{m-1}(x_i)$. In **AdaBoost**, the weight $w_i$ of misclassified samples is increased by $e^{\alpha_m}$, forcing subsequent models to prioritize "difficult" cases, primarily reducing bias.

---

## 2. Generative Learning Algorithms

Machine learning isn't just about drawing a line to separate data (Discriminative); it can also try to understand the generation mechanism of each class (Generative). Algorithms include **Gaussian Discriminant Analysis (GDA)** and **Naive Bayes**.

**Discriminative vs. Generative**:
- **Discriminative** (e.g., Logistic Regression) cares only about how to separate two classes. It learns $p(y|x)$.
- **Generative** (e.g., GDA, Naive Bayes) studies what each class "looks like" first (modeling separately). It learns $p(x|y)$ and $p(y)$, then uses Bayes' Theorem to infer the posterior. If data is scarce, generative models often learn faster by utilizing more assumptions (like Gaussianity). If assumptions are wrong, performance suffers. Logistic Regression is more **Robust**.
- **Prediction**: Using Bayes' Theorem: $p(y=1|x) = \frac{p(x|y=1)p(y=1)}{p(x)}$, where $p(x) = p(x|y=1)p(y=1) + p(x|y=0)p(y=0)$.

**Gaussian Discriminant Analysis (GDA)**: A generative model for continuous data. Assumes $p(x|y)$ follows a Gaussian distribution.
- **Key Insight**: GDA implies Logistic Regression. If $p(x|y)$ is Gaussian, $p(y|x)$ must take the Sigmoid form. However, the reverse isn't true (e.g., if $x|y$ is Poisson, $p(y|x)$ is also Sigmoid). This shows GDA makes stronger assumptions.
- **If assumptions hold**, GDA is more efficient (needs less data). **Logistic Regression** is more flexible and usually performs better when distributions are unknown or non-Gaussian.
- **Assumptions**: $y \sim Bernoulli(\phi)$, $p(x|y=0) \sim \mathcal{N}(\mu_0, \Sigma)$, $p(x|y=1) \sim \mathcal{N}(\mu_1, \Sigma)$. Note: Usually, both classes share the same covariance matrix $\Sigma$ to ensure a linear decision boundary.
- **Training (MLE)**: Estimates are intuitive: $\phi$ is the proportion of $y=1$; $\mu_0, \mu_1$ are class means; $\Sigma$ is the empirical covariance.

**Naive Bayes**: A generative model for discrete/text data (e.g., Spam Classification).
- **Naive Assumption**: Assume features $x_i$ are conditionally independent given the class $y$. Although almost certainly false (e.g., "Neural" and "Networks" are highly correlated), it's surprisingly effective in practice.
- **Independence Assumption**: $p(x_1, \dots, x_n | y) = \prod_{i=1}^n p(x_i|y)$.
- **Laplace Smoothing**: If a word never appeared in training for a class, $p(x_j|y)=0$, which makes the entire posterior 0. Laplace smoothing avoids this by adding 1 to the numerator and $k$ (number of possible values) to the denominator.

---

### 2.1 Naive Bayes and Support Vector Machines

**Naive Bayes** is a classic probabilistic classifier that is particularly effective for text classification tasks such as spam detection and sentiment analysis.

Two common variants are used in practice:

#### Multivariate Bernoulli Model (Early Model)

In the **multivariate Bernoulli model**, each document is represented as a fixed-length binary vector:

* Each dimension corresponds to a word in a predefined dictionary.
* A value of 1 indicates the presence of a word; 0 indicates absence.
* Word frequency is ignored.

While simple and computationally efficient, this model loses important information—such as whether a word appears once or one hundred times.

#### Multinomial Event Model (Modern Model)

The **multinomial event model** better captures the statistical structure of text:

* The feature vector length equals the document length $n$.
* Each entry $x_j$ represents the index of the word at position $j$ in the dictionary.
* Word frequency is explicitly modeled.

As a result, the multinomial model typically outperforms the Bernoulli model in text classification tasks, especially when word repetition carries semantic meaning (e.g., repeated mentions of “drugs” in spam emails).

### Practical Machine Learning Advice

Effective machine learning is as much about strategy as it is about algorithms.

* **Do not over-optimize early**
  Avoid designing an overly complex system at the outset (e.g., handling spelling errors, advanced linguistic features, or elaborate pipelines).

* **Start quick and dirty**
  Build a strong **baseline** first using simple models such as Naive Bayes or Logistic Regression.

* **Perform error analysis**
  Manually inspect misclassified examples to identify systematic errors. This analysis often reveals the most impactful direction for improvement.

### Support Vector Machines (SVM)

**Support Vector Machines** are widely regarded as one of the most effective “turn-key” algorithms in machine learning. Compared to neural networks, they typically require fewer hyperparameters and less manual tuning.

#### Geometric Margin

Logistic regression only requires that predictions satisfy $h_\theta(x) > 0.5$ for correct classification. In contrast, SVMs explicitly aim for **confidence** in their predictions.

The **geometric margin** of a training example is defined as:

$\gamma^{(i)} = \frac{y^{(i)} (w^T x^{(i)} + b)}{|w|}$

It represents the Euclidean distance from a data point to the separating hyperplane.

#### Optimal Margin Classifier

An SVM seeks the hyperplane that **maximizes the minimum margin** across all training examples. Intuitively, this places the decision boundary in the “middle of the road,” maximizing robustness to noise and small perturbations.

By fixing the functional margin to 1, the optimization problem becomes:

$\min_{w,b} \frac{1}{2} \|w\|^2$

$s.t. \quad y^{(i)}(w^T x^{(i)} + b) \ge 1$

This formulation is a **convex quadratic constrained quadratic programming (QCQP)** problem. Convexity guarantees the existence of a unique global optimum.

---

## 2.2 Kernel Trick and Soft-Margin SVM

### The Kernel Trick

One of the most powerful aspects of SVMs is the **kernel trick**, which enables linear classification in extremely high-dimensional—or even infinite-dimensional—feature spaces.

#### Dual Formulation

In the dual formulation of the SVM optimization problem, the data appear only through **inner products**:

$\langle x^{(i)}, x^{(j)} \rangle$

This observation allows us to replace inner products with a **kernel function**.

#### Kernel Functions

A kernel function $K(x, z)$ implicitly defines a mapping $\phi(x)$ into a higher-dimensional space such that:

$K(x, z) = \langle \phi(x), \phi(z) \rangle$

Crucially, we never need to explicitly compute $\phi(x)$. This allows SVMs to perform linear classification in very high-dimensional spaces with the same computational cost as in the original input space.

Common kernels include:

* **Polynomial Kernel**

  $K(x, z) = (x^T z + c)^d$

* **Gaussian (RBF) Kernel**
  
  $K(x, z) = \exp\left(-\frac{\|x - z\|^2}{2\sigma^2}\right)$

The Gaussian kernel corresponds to an **infinite-dimensional feature space** and measures similarity between inputs.

#### Mercer’s Theorem

A function $K(x, z)$ is a valid kernel if its kernel matrix is **positive semi-definite**. This condition is formalized by **Mercer’s Theorem**.

### Soft-Margin Support Vector Machines

Real-world data are rarely perfectly linearly separable. To handle noise and overlapping classes, SVMs introduce **slack variables** $\xi_i$.

The soft-margin optimization problem is:

$\min \frac{1}{2} \|w\|^2 + C \sum \xi_i$

subject to:

$y^{(i)}(w^T x^{(i)} + b) \ge 1 - \xi_i, \xi_i \ge 0$

#### The Role of $C$

The parameter $C$ controls the trade-off between margin size and classification errors:

* **Large $C$**
  Strongly penalizes violations → tighter fit → higher risk of overfitting.

* **Small $C$**
  Allows more violations → smoother decision boundary → higher risk of underfitting.

---

# 3. Model Selection, Bias vs. Variance, and Practical Theory

## 3.1 Bias, Variance, and Regularization

Bias and variance describe the "Goldilocks" problem of model fitting. Through the decomposition of **Expected Prediction Error (EPE)**, for a test point $x_0$, the expected squared error can be decomposed as:
$\text{EPE}(x_0) = \text{Bias}^2(\hat{f}(x_0)) + \text{Var}(\hat{f}(x_0)) + \sigma^2$
- **High Bias (Underfitting)**: The model is too simple. The bias term $(E[\hat{f}(x_0)] - f(x_0))^2$ is large because the expected prediction is far from the true function $f(x_0)$ (e.g., fitting a linear $y=mx+b$ to complex quadratic data).
- **High Variance (Overfitting)**: The model is too complex. The variance term $E[(\hat{f}(x_0) - E[\hat{f}(x_0)])^2]$ is large because the model $\hat{f}$ fluctuates wildly depending on the specific training set.
- **Goal**: Minimize total error by balancing these two terms. Note that $\sigma^2$ is the **Irreducible Error** (noise variance) inherent in the data that no model can eliminate.

**Regularization** prevents overfitting by modifying the learning objective. Instead of just minimizing the cost function $J(\theta)$, we minimize the regularized cost: $J_{\text{reg}}(\theta) = J(\theta) + \lambda \Omega(\theta)$, where $\lambda$ is the regularization parameter and $\Omega(\theta)$ is the penalty term.
- **L2 Regularization (Ridge Regression)**: Penalty $\Omega(\theta) = \|\theta\|_2^2 = \sum \theta_j^2$. Minimizing this forces coefficients $\theta_j$ towards zero, resulting in a smoother curve.
- **$\lambda$ Parameter**: If $\lambda \to \infty$, the penalty dominates, forcing $\theta \approx 0$ and leading to a flat curve (High Bias). If $\lambda \to 0$, it reverts to standard minimization $\min J(\theta)$ with a risk of High Variance.
- **Bayesian Interpretation**: L2 regularization is mathematically equivalent to **Maximum A Posteriori (MAP)** estimation. If we assume a Gaussian prior for parameters $P(\theta) \sim \mathcal{N}(0, \tau^2)$, maximizing the log-posterior is equivalent to minimizing squared error plus a term proportional to $\|\theta\|^2$.

**Data Partitioning (Train/Dev/Test)**:
- **Training Set**: Used to optimize parameters $\hat{\theta}$.
- **Development (Dev/Validation) Set**: Used to select hyperparameters (e.g., polynomial degree $d$ or regularization $\lambda$). We choose the model that minimizes error on this set.
- **Test Set**: Used for the final unbiased performance estimate.
- **Sizes**: Traditional 60/20/20 splits rely on the law of large numbers. In the era of Big Data, if $N=1,000,000$, a 1% test set ($10,000$ samples) can provide a sufficiently tight confidence interval, allowing 98% for training.

**Cross-Validation (for small datasets)**:
- **k-fold CV**: Partitions data into $k$ subsets. In each of $k$ iterations, use one subset for testing and the rest for training. The CV error is the average performance across all rotations.
- **Leave-One-Out CV (LOOCV)**: When $k=N$. Unbiased but computationally expensive ($O(N)$ training cycles).

**Feature Selection**: When input dimension $d$ is huge, we search for a subset $S \subset \{1, \dots, d\}$ to maximize performance. Since checking all $2^d$ subsets is impossible, we use greedy heuristics like **Forward Search**:
(1) Start with an empty set $S_0 = \emptyset$.
(2) In each step, add the feature $j$ that most improves validation accuracy.
(3) Repeat until the marginal gain falls below a threshold $\epsilon$.

## 3.2 Learning Theory

Learning algorithms rely on two fundamental assumptions:
1. **Data Distribution**: There exists a fixed but unknown probability distribution $\mathcal{D}$ on the input-output space $\mathcal{X} \times \mathcal{Y}$.
2. **I.I.D. Assumption**: The training set $S$ consists of $m$ samples drawn independently and identically distributed (i.i.d.) from $\mathcal{D}$. This allows us to link training performance to future performance.

**Error Decomposition**: Total error of our learned hypothesis $\hat{h}$ can be decomposed relative to the optimal hypothesis $h^*$ in class $\mathcal{H}$ and the global optimal function $g$ (Bayes Optimal Predictor):

```math
\varepsilon(\hat{h}) = \underbrace{\varepsilon(g)}_{\text{Irreducible Error}} + \underbrace{(\varepsilon(h^*) - \varepsilon(g))}_{\text{Approximation Error}} + \underbrace{(\varepsilon(\hat{h}) - \varepsilon(h^*))}_{\text{Estimation Error}}
```
- **Approximation Error (Bias)**: Penalty for choosing a limited hypothesis class $\mathcal{H}$ (e.g., linear models) that doesn't contain the true function $g$.
- **Estimation Error (Variance)**: Penalty for having only finite data $S$ to choose $\hat{h}$ instead of infinite data to choose $h^*$.
- **Trade-off**: Increasing complexity of $\mathcal{H}$ lowers approximation error but increases estimation error.

**Experience Risk Minimization (ERM)**: Since we cannot calculate true risk $\varepsilon(h)$, we choose the hypothesis that minimizes **Empirical Risk** (average loss on training set $S$): 

```math
\hat{h}_{\text{ERM}} = \arg\min_{h \in \mathcal{H}} \hat{\varepsilon}_S(h).
```

**Uniform Convergence**:
- **Hoeffding's Inequality**: For a single fixed $h$, the probability that empirical error deviates from true error by more than $\gamma$ decays exponentially with $m$: $P(|\hat{\varepsilon}(h) - \varepsilon(h)| > \gamma) \leq 2 \exp(-2\gamma^2 m)$.
- **Union Bound**: Extending this to the entire class $\mathcal{H}$ of size $k$: $P(\exists h \in \mathcal{H} : |\hat{\varepsilon}(h) - \varepsilon(h)| > \gamma) \leq 2k \exp(-2\gamma^2 m)$.
- This shows that if $m$ is large relative to $\log(k)$, empirical error uniformly converges to true error.

**Sample Complexity & VC Dimension**:
- **Finite Class**: Sample complexity $m \ge \frac{1}{2\gamma^2} (\log(k) + \log(2/\delta))$.
- **Infinite Class & VC Dimension**: For infinite classes (like linear classifiers), we replace $\log(k)$ with **Vapnik-Chervonenkis (VC) Dimension** $d_{VC}$, measuring the "shattering" ability of $\mathcal{H}$. $m = O\left(\frac{1}{\gamma^2} (d_{VC} + \log(1/\delta))\right)$.
- Rule of thumb: Required training samples are roughly linear with the number of parameters ($d_{VC} = d+1$ for linear classifiers).

## 3.3 Applying and Debugging ML Algorithms

**Diagnosis over Intuition**: When a model performs poorly, don't blindly collect data or tune parameters. Use systematic diagnostics.
- **Bias vs. Variance Diagnosis**: Plot **Learning Curves** ($J(\theta)$ vs. $m$).
    - **High Variance (Overfitting)**: Large gap between $J_{\text{train}}$ (low) and $J_{\text{cv}}$ (high). Fix: More data, smaller feature set, or higher $\lambda$.
    - **High Bias (Underfitting)**: Both $J_{\text{train}}$ and $J_{\text{cv}}$ are high and close to each other. Fix: More features, polynomial features, or lower $\lambda$. (Adding data doesn't help).
- **Optimization vs. Objective**: Is the solver failing to minimize $J$, or is $J$ the wrong goal?
    - If $J(\theta_{\text{human}}) < J(\theta_{\text{algo}})$, the **Optimization solver** is the problem.
    - If $J(\theta_{\text{algo}}) \le J(\theta_{\text{human}})$ but the human performs better, the **Objective function $J$** is wrong.

**Component Diagnostics**: Use a "binary search" strategy for complex pipelines.
- Example: Autonomous helicopter. Check simulation ($J_{\text{sim}}$ vs. $J_{\text{real}}$), check human vs. algo.
- **Error Analysis (Pipeline Debugging)**: Identify the bottleneck by replacing components with "Ground Truth" (perfect) outputs one by one. If perfect output from B fixes 20% while perfect A only fixes 5%, B is the bottleneck.
- **Ablation Analysis**: Start with the full complex model and remove components one by one to see what truly contributes to performance (Occam's Razor).

---

## 4 Deep Learning and Neural Networks

Logistic regression can be viewed as the simplest form of a “neural network,” consisting of a single neuron and serving as the most basic building block (the perceptron). The input is a vector $x \in \mathbb{R}^{n_x}$ (for example, a flattened image). The neuron computes a linear combination followed by a nonlinear activation:

$z = w^T x + b, \quad a = \sigma(z) = \frac{1}{1 + e^{-z}}$,

where $w \in \mathbb{R}^{n_x}$ is the weight vector, $b \in \mathbb{R}$ is the bias, and $\sigma(z)$ is the sigmoid activation function. The model hypothesis is $h_{w,b}(x) = a$. The goal is to learn parameters $(w, b)$ that minimize a loss function.

### Multiclass Classification

For problems with $C$ classes (e.g., cat, lion, iguana), the output layer changes depending on the assumptions:

* **Independent logistic regressions (multi-label classification)**:
  If classes are not mutually exclusive, we use $C$ independent sigmoid units. The output is a vector $a \in [0,1]^C$,
  
  $a_i = \sigma(z_i) = \sigma(w_i^T x + b_i), \quad i = 1, \dots, C.$
  
  The loss is the sum of binary cross-entropies:
  
  $L = - \sum_{i=1}^C \left[ y_i \log(a_i) + (1 - y_i)\log(1 - a_i) \right].$

* **Softmax regression (multiclass classification)**:
  If classes are mutually exclusive, raw scores (logits) $z$ are normalized so they sum to 1.

* **Cross-entropy loss**:
  Since $\hat{y}$ is a probability distribution, we use categorical cross-entropy:
  
  $L(\hat{y}, y) = - \sum_{i=1}^C y_i \log(\hat{y}_i),$
  
  where $y$ is a one-hot encoded vector.

### Deep Networks

A deep network is a composition of multiple functions. For an $L$-layer network, the structure is:

$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \quad
a^{[l]} = g^{[l]}(z^{[l]}),$

where $l = 1, \dots, L$, $a^{[0]} = x$ (the input), and $g^{[l]}$ is the activation function of layer $l$ (e.g., ReLU for hidden layers, g(z) = max(0, z)).

From a representation-learning perspective, each layer transforms the feature space. The first layer may learn edges,

$a^{[1]} \approx \text{EdgeDetect}(x),$

the second layer may combine edges into shapes,

$a^{[2]} \approx \text{ShapeDetect}(a^{[1]}),$

and so on. This hierarchical composition allows networks to approximate highly complex nonlinear functions.

### Vectorization and Broadcasting

Efficient computation relies on linear algebra operations on matrices rather than explicit for-loops. For batch processing, we stack $m$ training examples into a matrix

$X \in \mathbb{R}^{n_x \times m}.$

Forward propagation for a layer becomes:

$Z = WX + b,$

where $W \in \mathbb{R}^{n_{\text{neurons}} \times n_x}$, so $Z \in \mathbb{R}^{n_{\text{neurons}} \times m}$.

**Broadcasting**: In the equation $Z = WX + b$, $WX$ is a matrix and $b \in \mathbb{R}^{n_{\text{neurons}} \times 1}$ is a column vector. Python/NumPy automatically broadcasts $b$ horizontally (copies it $m$ times) so element-wise addition is valid:

$Z_{i,j} = (WX)_{i,j} + b_i.$

### Forward and Backward Propagation

Forward propagation computes the loss:

$X \xrightarrow{W^{[1]}, b^{[1]}} A^{[1]} \xrightarrow{\dots} A^{[L-1]}
\xrightarrow{W^{[L]}, b^{[L]}} \hat{Y} \xrightarrow{\mathcal{L}} J.$

Backward propagation (via the chain rule) computes gradients $\frac{\partial J}{\partial W^{[l]}}$ and $\frac{\partial J}{\partial b^{[l]}}$, which are used to update parameters using gradient descent:

$W := W - \alpha , dW.$

Starting from the output layer:

(1) $dZ^{[L]} = A^{[L]} - Y$
   (the derivative of Softmax or Sigmoid combined with cross-entropy loss).

(2) For $l = L, L-1, \dots, 1$:

   * Weight gradient: $dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T}$

   * Bias gradient: $db^{[l]} = \frac{1}{m} \sum_{\text{columns}} dZ^{[l]}$
    
   * Gradient w.r.t. previous activations: $dA^{[l-1]} = W^{[l]T} dZ^{[l]}$
     
   * Gradient w.r.t. previous logits: $dZ^{[l-1]} = dA^{[l-1]} * g'^{[l-1]}(Z^{[l-1]}),$

     where $*$ denotes element-wise multiplication.

---

### 4.1 Backpropagation and Improvements to Neural Networks

### Derivation of Backpropagation

Backpropagation derives the gradients required to update network parameters $W$ and $b$ using the chain rule, computing derivatives layer by layer starting from the output. We aim to minimize the loss $L(\hat{y}, y)$.

Define the error term for layer $l$ as:

$\delta^{[l]} = \frac{\partial L}{\partial z^{[l]}}.$

* **Output layer ($L$)**:
  With Softmax and cross-entropy,
  
  $\delta^{[L]} = \hat{y} - y = a^{[L]} - y.$
  
* **Hidden layers ($l < L$)**:
  Using the chain rule,
  
  $\delta^{[l]} = \left(W^{[l+1]T} \delta^{[l+1]}\right) * g'^{[l]}(z^{[l]}).$

Once $\delta^{[l]}$ is known, the parameter gradients are:

$\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} a^{[l-1]T}, \quad
\frac{\partial L}{\partial b^{[l]}} = \sum \delta^{[l]} \quad (\text{sum over batch}).$

### Shape Analysis

Matrix dimensions are critical. If

$W^{[l]} \in \mathbb{R}^{(n^{[l]}, n^{[l-1]})},$

then

$\delta^{[l]} \in \mathbb{R}^{(n^{[l]}, m)}, \quad
a^{[l-1]} \in \mathbb{R}^{(n^{[l-1]}, m)}.$
Thus,

$\delta^{[l]} a^{[l-1]T} \in \mathbb{R}^{(n^{[l]}, n^{[l-1]})},$

which exactly matches the shape of $W^{[l]}$.

### Caching

Values computed during forward propagation ($z^{[l]}, a^{[l]}$) are cached, since they are required during backpropagation (e.g., for computing $g'(z^{[l]})$ and $a^{[l-1]}$).

### Activation Functions

The choice of nonlinear activation function determines how the network learns:

* **Sigmoid**:
  
  $\sigma(z) = \frac{1}{1 + e^{-z}}, \quad
  \sigma'(z) = \sigma(z)(1 - \sigma(z)).$
  
  When $|z|$ is large, $\sigma'(z) \approx 0$, causing the vanishing gradient problem.

* **tanh**:
  
  $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}},$
  
  which is zero-centered and often easier to optimize than sigmoid, but still suffers from saturation.

* **ReLU (Rectified Linear Unit)**:
  
  ```math
  g(z) = max(0, z),  
  g'(z) = 1 if z > 0, else 0
  ```
  
  ReLU avoids vanishing gradients for positive inputs and typically converges faster.

**Importance of nonlinearity**:
If $g(z) = z$ (linear), a deep network collapses into a single linear transformation:

$W^{[2]}(W^{[1]}x) = (W^{[2]}W^{[1]})x = W'x.$

Nonlinearity enables networks to approximate complex functions.

### Data Preprocessing (Normalization)

We standardize inputs:

$\mu = \frac{1}{m} \sum_{i=1}^m x^{(i)}, \quad
\sigma^2 = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu)^2,$

$x_{\text{norm}} = \frac{x - \mu}{\sigma}.$

Without normalization, if $x_1 \in [0,1]$ and $x_2 \in [0,1000]$, the cost contours become elongated ellipses, causing gradient descent to oscillate. Normalization makes contours more spherical, allowing gradients to point directly toward the minimum.

**Important**: The test set must be normalized using the training set’s $\mu$ and $\sigma$.

### Weight Initialization

If weights are too large ($W \gg 1$), activations explode; if too small ($W \ll 1$), activations vanish. In deep networks, variance can grow or shrink exponentially with depth:

$\text{Var}(a^{[L]}) \approx \text{Var}(a^{[0]}) \prod_{l=1}^L \left(n^{[l-1]} \text{Var}(W^{[l]})\right).$

To keep variance stable across layers ( $Var(a^{[l]}) \approx Var(a^{[l-1]})$ ):

* **Xavier/Glorot initialization** (for tanh):
  
  $W \sim \mathcal{N}\left(0, \frac{1}{n^{[l-1]}}\right),$
  
  or a uniform distribution with limits $\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$.

* **He initialization** (for ReLU):
  
  $W \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]}}\right),$
  
  where the factor 2 accounts for ReLU zeroing out half the activations.

### Optimization Algorithms

* **Mini-batch gradient descent**:
  The training set is split into batches of size $B$, denoted $X^{{t}}$:
  
  $W := W - \alpha \nabla J(W; X^{{t}}, Y^{{t}}).$
  
  Compared to full-batch gradient descent, this yields noisier but faster updates.

* **Momentum**:
  Instead of using the current gradient directly, we compute an exponentially weighted average:
  
  $v_{dW} = \beta v_{dW} + (1 - \beta)dW, \quad
  W := W - \alpha v_{dW}.$
  
  By analogy to physics, $v_{dW}$ is velocity and $dW$ is acceleration. The hyperparameter $\beta$ (e.g., 0.9) acts like friction, damping oscillations in steep directions while accelerating convergence in consistent directions.

---

## 5 Unsupervised Learning

Unsupervised learning handles unlabeled data $\{x^{(i)}\}_{i=1}^m$. Its goal is to discover hidden structures, such as clusters.

**K-Means Clustering**: Finds $k$ cluster centers $\mu_1, \dots, \mu_k$ to minimize the distortion function:
$J(c, \mu) = \sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2$
1. **Initialize**: Randomly choose $k$ cluster centers $\mu_1, \dots, \mu_k \in \mathbb{R}^n$.
2. **Cluster Assignment**: Assign each $x^{(i)}$ to the nearest centroid $c^{(i)} := \arg\min_j ||x^{(i)} - \mu_j||^2$.
3. **Move Centroid**: Update $\mu_j$ to be the mean of points assigned to it: $\mu_j := \frac{\sum_{i=1}^m \mathbb{1}\{c^{(i)}=j\} x^{(i)}}{\sum_{i=1}^m \mathbb{1}\{c^{(i)}=j\}}$.
4. **Repeat** until convergence.
- **Convergence**: Guaranteed to reach a local optimum. Run multiple times with different initializations to find a better global solution.
- **Choosing k**: Usually manual (e.g., "Elbow Method" or business constraints).

**Density Estimation & Anomaly Detection**:
- Goal: Model the probability distribution $p(x)$ of the data.
- **Anomaly Detection**: If $p(x_{\text{new}}) < \epsilon$, flag it as an anomaly.
- **Gaussian Mixture Model (MoG)**: For complex shapes (multimodal data). MoG assumes data is generated by a latent variable $z^{(i)} \sim \text{Multinomial}(\phi)$, followed by a Gaussian emission: $x^{(i)} | z^{(i)}=j \sim \mathcal{N}(\mu_j, \Sigma_j)$.
- Total probability: $p(x) = \sum_{j=1}^k \phi_j \mathcal{N}(x; \mu_j, \Sigma_j)$.

**Expectation-Maximization (EM) Algorithm**:
Since $z^{(i)}$ is hidden, the log-likelihood is hard to solve. EM solves this iteratively:
- **E-step (Expectation)**: Calculate "soft" assignment weights (posterior) $w_j^{(i)} = P(z^{(i)}=j | x^{(i)}; \theta)$.
- **M-step (Maximization)**: Update parameters $\mu_j, \phi_j, \Sigma_j$ using weighted averages based on $w_j^{(i)}$.
- **Theoretical Basis**: Maximize the **Evidence Lower Bound (ELBO)** using Jensen's Inequality:
$\ell(\theta) = \sum_i \log \sum_{z^{(i)}} Q_i(z^{(i)}) \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \ge \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} = \mathcal{L}(\theta, Q)$
- E-step maximizes $\mathcal{L}$ w.r.t. $Q$ by setting $Q$ to the posterior.
- M-step maximizes $\mathcal{L}$ w.r.t. $\theta$.
- Convergence is guaranteed as the likelihood $\ell(\theta)$ increases monotonically.

---

## 5.1 EM & Factor Analysis

**Problem of High-Dimensional Data ($m \ll n$)**:
If features $n=100$ but samples $m=30$, the empirical covariance matrix $\Sigma$ is singular (rank $\le m < n$), making standard Gaussian density uncomputable. Naive solutions like diagonal or spherical matrices fail to capture correlations.

**Factor Analysis**: Models data as being generated from a low-dimensional "latent" space.
- Latent variable $z \in \mathbb{R}^d$ ($d \ll n$) follows $z \sim \mathcal{N}(0, I)$.
- Observed data $x = \mu + \Lambda z + \epsilon$, where $\Lambda$ is the **Factor Loading Matrix** and $\epsilon \sim \mathcal{N}(0, \Psi)$ is diagonal noise.
- Marginal distribution: $x \sim \mathcal{N}(\mu, \Lambda\Lambda^T + \Psi)$.

**EM for Factor Analysis**:
- **E-Step**: Compute $Q_i(z^{(i)}) = p(z^{(i)} | x^{(i)}; \theta) \sim \mathcal{N}(\mu_{z|x}^{(i)}, \Sigma_{z|x})$.
- **M-Step**: Update $\Lambda, \mu, \Psi$ to maximize expected log-likelihood. For example, the update for $\Lambda$:
$\Lambda_{new} = \left( \sum_{i=1}^m (x^{(i)} - \mu) E[z^{(i)}]^T \right) \left( \sum_{i=1}^m E[z^{(i)} (z^{(i)})^T] \right)^{-1}$.

---

## 6 ICA, MDP, and Reinforcement Learning (RL)

**Independent Component Analysis (ICA)**: Separates independent sources from mixed signals (Cocktail Party Problem). $x = As \implies s = Wx$, where $W = A^{-1}$.
- **Requirement**: Sources must be **Non-Gaussian**. Gaussian distributions are rotationally symmetric, making the mixing matrix $A$ unidentifiable.
- **Modeling**: Model the CDF of sources as a Sigmoid $g(s)$, whose derivative $p(s)$ has "heavy tails," fitting real-world signals like speech.
- **Optimization**: Use Stochastic Gradient Ascent to maximize log-likelihood:
  
```math
W := W + \alpha \left( \begin{bmatrix} 1 - 2g(w_1^T x) \\ \vdots \\ 1 - 2g(w_n^T x) \end{bmatrix} x^T + (W^T)^{-1} \right).
```

**Reinforcement Learning (RL)**: Learning from rewards/penalties.
- **Credit Assignment Problem**: Determining which action led to a delayed reward.
- **Markov Decision Process (MDP)**: $(S, A, P_{sa}, \gamma, R)$.
    - $S$: State space.
    - $A$: Action space.
    - $P_{sa}$: Transition probability.
    - $\gamma$: Discount factor.
    - $R$: Reward function.
- **Value Function**: $V^\pi(s) = E \left[ \sum_{t=0}^\infty \gamma^t R(s_t) \mid s_0 = s, \pi \right]$.
- Goal: Find optimal policy $\pi^*$ maximizing $V^\pi(s)$.

---

## 6.1 Solving MDPs

- **Bellman Equation**: $V^\pi(s) = R(s) + \gamma \sum_{s' \in S} P_{s\pi(s)}(s') V^\pi(s')$.
- **Bellman Optimality**:
  
  $V(s) = R(s) + \gamma \cdot \max(a \in A) \sum_{s' \in S} P_{sa}(s') V(s').$
  
- **Value Iteration**: $V_{k+1}(s) := R(s) + \max_{a \in A} \gamma \sum P_{sa}(s') V_k(s')$. Guaranteed geometric convergence to $V^*$.
- **Policy Iteration**: Alternates between **Policy Evaluation** (solving Bellman equations for $V^\pi$) and **Policy Improvement** (making $\pi$ greedy w.r.t. $V^\pi$).
- **Unknown Transitions (Model-based RL)**: Estimate $\hat{P}_{sa}$ using MLE from observed transitions, then run Value Iteration.
- **Exploration vs. Exploitation**: **$\epsilon$-greedy** strategy: with probability $1-\epsilon$, exploit (best action); with probability $\epsilon$, explore (random action).

---

## 6.2 RL in Continuous State Spaces

- **Continuous States**: $S \subseteq \mathbb{R}^n$ (e.g., position, velocity).
- **Discretization Failures**: **Curse of Dimensionality** ($k^n$ states). Only works for $n < 4$.
- **Fitted Value Iteration**: Use a function approximator $V_\theta(s) = \theta^T \phi(s)$.
    (1) Sample states $\{s^{(1)}, \dots, s^{(m)}\}$.
    (2) Estimate targets: $y^{(i)} = \max_a (R(s^{(i)}) + \gamma E [V_\theta(s')])$.
    (3) Solve as a supervised regression: $\min_\theta \sum (\theta^T \phi(s^{(i)}) - y^{(i)})^2$.
- **Simulators**: Often used to model dynamics $s_{t+1} = f(s_t, a_t)$. Adding noise to the simulator makes the policy more robust to real-world model inaccuracies.
- **Control**: In execution, chose $\pi(s) = \arg\max_a (R(s) + \gamma E [V_\theta(s')])$ in real-time.

---

## 6.3 LQR and Generalizations of MDPs

**Generalizations of the MDP Framework**:
- **State-Action Rewards $R(s, a)$**: Rewards depend on both the state and the action taken (e.g., modeling energy consumption of a move). 
- **Finite Horizon MDP**: The process runs for a fixed number of steps $T$. Usually $\gamma=1$. The optimal policy and value function become time-dependent: $V_t(s)$ and $\pi_t(s)$. Solve using dynamic programming from $t=T$ back to $0$.

**Linear Quadratic Regulation (LQR)**:
Exact solution for continuous state/action MDPs under two assumptions:
1. **Linear Dynamics**: $s_{t+1} = A s_t + B a_t + w_t$, with Gaussian noise $w_t$.
2. **Quadratic Cost**: Reward $R(s, a) = -(s_t^T U s_t + a_t^T V a_t)$, where $U$ and $V$ are semi-definite matrices. We want to keep state and control small.

**Key results of LQR**:
- **Quadratic Value Function**: $V_t(s) = s_t^T \Phi_t s_t + \Psi_t$.
- **Riccati Equation**: Recursive equation to update the matrix $\Phi_t$ backwards from $T$.
- **Linear Policy**: The optimal control is a linear function of the state: $a_t^* = - L_t s_t$.
- **Certainty Equivalence**: The optimal policy $L_t$ is independent of the noise covariance $\Sigma_w$. The same control strategy works regardless of wind/noise levels.

---

## 6.4 RL Debugging and Policy Search

Debugging a reinforcement learning agent is notoriously difficult because failure can stem from multiple sources.

**RL Debugging Workflow**:
When a robot fails, identify if the issue is the simulator, the algorithm, or the reward function.
- **Compare $\pi_{RL}$ and $\pi_{human}$**:
    - If $J(\pi_{RL}) < J(\pi_{human})$: The **RL algorithm** failed to find the optimal solution. It might be stuck in a local optimum.
    - If $J(\pi_{RL}) \ge J(\pi_{human})$ but behavior is "bad": The **Reward Function** is wrong. It doesn't capture the desired behavior (e.g., oscillating too much).

**Direct Policy Search**:
Directly learn a parameterized policy $\pi_\theta$ instead of a value function.
- **Random Policy**: e.g., for binary actions, use a Sigmoid policy $\pi_\theta(s, a=1) = \sigma(\theta^T \phi(s))$.
- **REINFORCE Algorithm (Likelihood Ratio Method)**: Stochastic gradient ascent to maximize expected reward $J(\theta)$.
    - Update: $\theta := \theta + \alpha G \sum \nabla_\theta \log \pi_\theta(s_t, a_t)$, where $G$ is the total return.
    - If $G$ is high, increase the probability of those actions; if $G$ is low, decrease it.

**When to use Policy Search vs. Value Functions?**:
- **Use Policy Search (REINFORCE)**:
    - **POMDPs**: When the state is partially observable, $V(s)$ is ill-defined.
    - **Stochastic Policies**: When the optimal solution is a random strategy.
- **Use Value Functions**:
    - **Efficiency**: Usually converges faster with lower variance.
    - **Planning**: Better for deep foresight (e.g., Chess).

**Real-world Applications**:
- **Robotics**: Autonomous helicopters, legged robots learning to walk/climb.
- **Healthcare**: Adaptive treatment strategies (MDP: state=patient health, action=treatment, reward=recovery).
- **Finance**: Optimal execution (state=market conditions/inventory, action=trade volume, reward=profit minus slippage).
