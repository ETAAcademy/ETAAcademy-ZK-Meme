# ETAAcademy-ZKMeme: 80. ZK Deep Learning 4

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>80. ZKDL4</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZKDL4</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Deep Learning for Computer Vision: Self-Supervised Learning (SSL), Generative Models, 3D Vision, Foundation Models and Robot Learning

Self-supervised learning (SSL) trains encoders using pretext tasks like rotation prediction and jigsaw puzzles to learn useful representations without human annotation, evolving into reconstruction-based methods like inpainting and colorization, and scalable frameworks like Masked Autoencoders (MAE) and contrastive learning approaches such as SimCLR and MoCo.

Generative models extend this by learning data distributions through Maximum Likelihood Estimation (MLE), autoregressive models, and Variational Autoencoders (VAEs), while Generative Adversarial Networks (GANs) and Diffusion Models (including Rectified Flow and Latent Diffusion Models) enable high-fidelity image synthesis.

3D vision addresses the transition from 2D pixels to irregular 3D representations like point clouds and meshes, utilizing implicit representations such as Signed Distance Functions (SDFs) and Neural Radiance Fields (NeRF), alongside faster explicit methods like Gaussian Splatting, to model geometry and appearance. Foundation models like CLIP leverage contrastive learning on massive image-text pairs for zero-shot classification, paving the way for modern Vision-Language Models (VLMs) like LLaVA, Flamingo, and Molmo, as well as promptable segmentation models like SAM and chain-of-thought reasoning systems.

Robot learning redefines machine learning as a sequential decision-making process involving embodied perception and Markov Decision Processes (MDPs), employing techniques ranging from model-free Reinforcement Learning (RL) and model-based planning to imitation learning and emerging robot foundation models. Finally, the development of AI vision traces an arc from biological inspiration and early geometric attempts to the deep learning revolution driven by ImageNet, now aiming to build systems that surpass human classification capabilities, address cognitive limits and privacy through hardware-software co-design, and augment human labor in healthcare and robotics while respecting human values and preferences.

## 1. Self-Supervised Learning

Self-supervised learning (SSL) leverages "pretext tasks" that automatically generate labels from raw data to train encoders, capturing general visual features for use in downstream annotated applications. The supervisory signal in self-supervised learning originates from the data itself. A **Pretext Task** is a proxy objective designed to force the model to learn useful representations (e.g., spatial layout, object consistency) without human annotation. Subsequently, these learned representations are evaluated on **downstream tasks**, which are the actual target applications (e.g., classification, detection), usually involving limited annotated data.

### Early Pretext Tasks: Detailed breakdown of Image Rotation and Jigsaw Puzzles

Learning representations (embeddings/latent space) is crucial in Convolutional Neural Networks (CNN) and Transformer models, where large-scale human annotation presents a "bottleneck." To address this, the SSL framework consists of an **encoder** (feature extractor) and a **decoder/head**, both of which are trained on pretext tasks. Once training is complete, the encoder is migrated to downstream tasks. Two specific early pretext tasks include:

1. **Rotation Prediction**: The image is rotated by 0°, 90°, 180°, or 270°, and the model is trained to predict the rotation angle. The intuition is that the model must understand the "visual common sense" (standard orientation) of objects. The rotation task is modeled as a four-class classification problem $y \in \{0, 1, 2, 3\}$, mapping to $\{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$. The model outputs a discrete label representing one of the four possible rotations. By simplifying the continuous rotation problem into four discrete categories, a standard cross-entropy loss function can be used for training. This forces the model to recognize the upright features of objects (e.g., head up, wheels down), implementing the concept of a "pretext task" by creating labels from the data (rotation operations).

2. **Jigsaw Puzzle**: The image is divided into a 3×3 grid, and the order of the grid patches is shuffled. The model predicts the correct permutation. To make the task manageable, the number of permutations is restricted from $9!$ to a subset (e.g., 64 possible permutations). This jigsaw permutation subset task uses a restricted set of permutations to simplify the output space, $P \subset S_9, \quad |P| = 64$, where $P$ is the set of allowed permutations containing a lookup table of 64 categories, and $S_9$ is the symmetric group of degree 9, encompassing all $9! = 362,880$ permutations. Instead of predicting all $362,880$ permutations of the 9 patches, the model chooses only from the 64 predefined permutations. Predicting $9!$ categories is computationally difficult, and many permutations are visually similar. Selecting 64 permutations with "maximum distance" (measured by Hamming distance) simplifies the task into a controllable classification problem while still requiring the model to understand spatial relationships. This is a "lookup table" approach for pretext task annotation.

**Rotation Ambiguity**: Some objects (e.g., circles or textures) are rotationally invariant, which can confuse the model.
**Shortcut Learning**: In jigsaw puzzles, the model may learn to "cheat" by observing low-level edge alignments (chromatic aberration or boundary artifacts) rather than high-level semantics, as noted in (Noroozi & Favaro, 2016). Representation quality is typically evaluated by "freezing" layers and training a shallow linear classifier, which is a standard SSL evaluation metric.

### Reconstruction Tasks: Inpainting, LAB Colorization, and Video Tracking

Reconstruction-based pretext tasks (e.g., image inpainting and colorization) enable models to learn structural and semantic features of images from available data by predicting missing pixels or color channels. In **Reconstruction-based Semantic Learning (SSL)**, the model learns by restoring degraded versions of input images. **Inpainting** (or context encoders) involves using the surrounding environment of an image to predict a missing central region. **Colorization** utilizes different **Color Spaces** (e.g., CIE LAB) to split an image into a luminance channel ($L$) and color channels ($A, B$). The **Split-Brain Autoencoder** further extends this process by training two sub-networks to predict each other's missing channels, effectively cross-referencing information.

- **Image Inpainting**: The masked reconstruction loss is $L_{rec}(x) = || M \odot (x - F((1 - M) \odot x)) ||_2^2$, calculating the square of the difference between the original pixel x and the predicted pixel, but limited only to the masked area. It forces the network to learn the "context" of the unmasked area ($(1-M) \odot x$) to fill in the missing part ($M \odot x$). This is the core objective of image inpainting and auto-encoding tasks. A mask M is applied to the image, and an F encoder-decoder architecture (context encoder, neural network model) is trained to fill the missing region. The loss function includes standard reconstruction loss ($|| \cdot ||_2^2$, Mean Squared Error, MSE) and adversarial loss (Generative Adversarial Network, GAN) to ensure the generated image patches look realistic and not blurry. Many objects can have multiple valid colors (e.g., a shirt can be red or blue). Standard MSE loss functions often produce "average" (grayish) results. Therefore, an adversarial loss function or a classification-based colorization method (mapping to color bins) is required.

- **Colorization**: The attention mechanism in video colorization is $y_i = \sum_j A(i, j) c_j$, where the color of pixel $i$ in a new frame is a weighted average of the color $c_j$ in a reference frame $j$, with weights determined by feature similarity (attention A(i, j)). This explains how SSL can be used for motion and tracking tasks without explicit tracking labels. If the model can correctly "point" to the same object in a previous frame and retrieve its color, it has effectively learned to track that object. Standard RGB images are converted to the LAB color space. The luminance (L) channel is used as input to predict the A and B color channels. This forces the model to understand object categories (e.g., "this shape is a tree, so it should be green").

- **Split-Brain Autoencoder**: This is a generalization method where data is split into two subsets (e.g., $X_1 = L, X_2 = AB$). One network predicts $X_2$ from $X_1$, and the other predicts $X_1$ from $X_2$. In split-brain autoencoders, the final representation is typically the concatenation of the features of the two sub-networks, which achieves higher accuracy than using either sub-network alone.

- **Video Colorization**: This task is used for learning temporal tracking. By colorizing target frames using reference frames, the model implicitly learns the changes of pixels and objects over time through an attention-based mechanism. Video colorization emphasizes temporal consistency; without constraints, colors may flicker. The attention mechanism provides a "copy-paste" constraint that improves consistency.

### Masked Autoencoder (MAE): Examination of the ViT-based asymmetric architecture and high masking ratios

Masked Autoencoders (MAE) achieve scalable self-supervised pre-training by using a high masking rate (75%) on image patches and an asymmetric ViT architecture (where the encoder only processes visible image patches). **Masked Autoencoder (MAE)** is an advanced self-supervised framework based on **Vision Transformers (ViT)**. It features an **asymmetric architecture** (a large encoder for processing visible image patches and a small decoder for image reconstruction), a **high masking rate** (unlike NLP BERT, which uses about 15%, MAE uses about 75%), and evaluation via **linear probing** and **fine-tuning**. MAE is a breakthrough in reconstruction-based SSL; the reconstruction objective function (Mean Squared Error, MSE) only penalizes errors in the parts of the image the model cannot see (the areas covered by the mask). By ignoring visible areas in the loss function, the model is forced to focus entirely on learning the underlying structure and semantics needed to "hallucinate" the missing data. This is a specific implementation of the "reconstruction task" mentioned in the general introduction to SSL.

$$
\mathcal{L} = \frac{1}{\Omega} \sum_{i \in \Omega} (x_i - \hat{x}_i)^2
$$

- $\mathcal{L}$: Total loss, Mean Squared Error (MSE) loss.
- $\Omega$: Masked patch index set, the "target."
- $x_i$: Ground truth pixel values, normalized pixel values in patch $i$.
- $\hat{x}_i$: Predicted pixel values, decoder output.

**Asymmetry**: The decoder is only used during the pre-training phase. For downstream tasks, the decoder is discarded, and only the encoder is used (the encoder only sees real image patches). **Redundancy**: A high masking rate (75%) is crucial in the visual domain because images have high spatial redundancy (neighboring pixels are very similar), unlike words in a sentence. **Mask Tokens**: The encoder _does not_ see mask tokens; they are only introduced at the decoder stage. This is a key design choice that saves memory and computational resources. **Evaluation Mode**: In MAE, fine-tuning often significantly outperforms linear probing, indicating that the representations are highly adaptable but might not be fully "linear" out of the box.

- **Architecture**: The image is divided into multiple image patches. Most patches $x_i$ (e.g., 75%) are randomly removed.
- **Encoder**: Only the remaining 25% visible patches are fed into a large ViT encoder, significantly reducing computational costs during pre-training.
- **Decoder**: Combines the encoded features of visible patches with "mask tokens" (learnable placeholders for missing patches). A lightweight ViT decoder then reconstructs the full original image.
- **Objective**: The model is trained only on the masked patches to minimize the MSE between reconstructed and original pixels.
- **Evaluation**:
  - **Linear Probing**: The encoder is frozen, and only a linear classifier is trained. This evaluates the raw quality of learned features.
  - **Fine-tuning**: The entire model (or parts of the encoder) is updated to adapt to downstream tasks, improving performance.

### Contrastive Learning: Comprehensive breakdown of the InfoNCE loss, SimCLR, and MoCo frameworks

Contrastive learning learns representations by maximizing the similarity between different augmented views of the same image (positive samples) while minimizing the similarity with all other images in a batch or queue (negative samples), using the InfoNCE loss function. This is a similarity-based learning paradigm. Unlike reconstruction-based tasks, contrastive learning does not predict pixels; it learns to map images to a feature space where "similar" things are close to each other and "different" things are far apart. Key concepts include **positive pairs** (two versions of the same image), **negative pairs** (different versions of different images), and **Mutual Information (MI)**, which the loss function attempts to maximize.

Shifting from predicting pixels to predicting image relationships:

1. **Core Idea**: Given an image $x$, apply two different data augmentation methods to obtain $x_i$ and $x_j$ (a positive pair). Treat all other images in the batch as negative samples.

2. **InfoNCE Loss**: A categorical cross-entropy loss that treats similarity matching as a classification problem over the entire batch. The InfoNCE Loss is essentially a softmax classification where the "correct class" is another augmented version of the same image. Minimizing this loss maximizes the similarity between positive samples ($z_i, z_j$) and minimizes similarity with all other negative samples ($z_k$). Mathematically, this corresponds to a lower bound on the **Mutual Information** between the two views. This is the mathematical implementation of the "attract/repel" concept.

$$
\mathcal{L}_{i,j} = -\log \frac{\exp( 	\text{sim}(z_i, z_j) / 	\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp( 	\text{sim}(z_i, z_k) / 	\tau)}
$$

- $\mathcal{L}_{i,j}$: Loss for the positive pair $(i, j)$, InfoNCE / NT-Xent loss.
- $z_i, z_j$: Projected features (embeddings), normalized vectors.
- $\text{sim}(u, v)$: Cosine similarity, $\frac{u^T v}{||u|| ||v||}$.
- $\tau$: Temperature parameter, controlling the "sharpness" of the distribution.
- $N$: Batch size, total samples after augmentation = $2N$.
- $\mathbb{1}$: Indicator function, equals 1 if $k \neq i$, otherwise 0.

3. **SimCLR**:

- Uses powerful data augmentation techniques (cropping, color jittering, etc.).
- Uses a **projection head** (a non-linear MLP) to map features into the space where the loss is calculated.
- Requires very large **batch sizes** (e.g., 4096 or 8192) to provide sufficient negative samples.

4. **MoCo (Momentum Contrast)**: Momentum Update (MoCo) updates the weights of the key encoder as an Exponential Moving Average (EMA) of the query encoder's weights. Because negative samples in the MoCo queue come from previous iterations, the encoder must change slowly. A slow update speed ensures the stability of the key dictionary, facilitating comparisons. If it changed as rapidly as the query encoder, the features in the queue would become "stale" and inconsistent. This allows MoCo to decouple the number of negative samples from the batch size, echoing the "exotic" non-gradient update mechanism that enables large-scale contrastive learning without massive batch sizes.

$$ \theta_k \leftarrow m \theta_k + (1 - m) \theta_q $$

- $\theta_k$: Parameters of the key (momentum) encoder, moving slowly via EMA rather than SGD.
- $\theta_q$: Parameters of the query encoder, updated via backpropagation.
- $m$: Momentum coefficient, typically 0.999.

- Solves batch size limitations by using a **memory queue** to store past negative samples.
- Uses a **momentum encoder** (a slow moving average of the main encoder) to maintain consistency of queue features.
- Conceptualizes the problem as a "dictionary lookup" where the query must match its key.

5. **DINO**: A student-teacher framework utilizing self-distillation techniques without explicit negative samples.

**Projection Head**: The loss is calculated based on $z$ (output of the projection head), but features for downstream tasks are taken from $h$ (output of the encoder). This is because the contrastive task might discard "too much" information to satisfy the objective. **Mutual Information**: A larger number of negative samples provides a "tighter" bound on mutual information, explaining the need for large batches (SimCLR) or large queues (MoCo). **Temperature ($\tau$)** is critical for performance; it scales the dot product to ensure the softmax function does not saturate prematurely.

---

## 2. Generative Models

### 2.1 Generative Models: SSL, MLE, Autoregression and Variational Autoencoders (VAEs)

### SSL Review & Intro: Comparison of discriminative vs. generative models and the role of Bayes' Rule

While **discriminative models** in self-supervised learning learn the conditional probability $p(y|x)$ (predicting label $y$ given feature $x$), **generative models** learn the joint distribution $p(x, y)$ or marginal distribution $p(x)$ (modeling the data itself). **Bayes' Rule** is the fundamental bridge connecting these modeling paradigms.

Generative model theory defines "Probability Mass Competition": since the sum/integral of probabilities must be 1, increasing the likelihood of one image necessarily decreases the likelihood of others. Generative models are viewed as tools for handling **ambiguity** (one input, multiple valid outputs) and **outlier detection** (assigning low probabilities to unreasonable inputs).

The Bayesian model relationship rule shows that conditional generative models can be composed of discriminative models, unconditional generative models, and label priors, demonstrating that these three modeling paradigms are mathematically interrelated. However, in practice, conditional models are usually trained end-to-end rather than combined this way. This explains why "swapping $x$ and $y$" in a model leads to such different (and harder) challenges in visual reasoning.

$$
p(x|y) = \frac{p(y|x) p(x)}{p(y)}
$$

- p(x|y): Conditional generative model, the likelihood of image $x$ given label $y$.
- p(y|x): Discriminative model, the probability of label $y$ given image $x$.
- p(x): Unconditional generative model, the density of data $x$.
- p(y): Label prior, the frequency of label $y$.

The probability normalization constraint, $\int p(x) dx = 1$, defines the probability density function, assigning mass to each $x$. The total probability mass for all possible data points must equal 1. This forms a "zero-sum game" where every image competes for mass. A good generative model must assign mass to realistic images (e.g., a four-legged dog) while avoiding assignment to unrealistic ones (e.g., a three-armed monkey). This is crucial for understanding why density estimation is a harder global optimization problem than simple classification.

**Unconditional vs. Conditional**: Unconditional generative models (p(x)) are mostly "useless" in practice because they cannot control sampling. Most useful applications (e.g., ChatGPT and DALL-E) are conditional generative models (p(x|y)). Contrastive learning (SSL) is more effective for **feature learning** than explicit density estimation, although both are unsupervised. **Scalability**: The success of DINOv2 is largely due to its scale (142 million images) rather than major algorithmic improvements.

### Maximum Likelihood Estimation and Autoregressive Models

Maximum Likelihood Estimation (MLE) provides a principled framework for fitting probability models, and autoregressive models implement this by decomposing the joint data distribution into a series of tractable conditional probabilities. **Maximum Likelihood Estimation (MLE)** is the standard objective function for training explicit density models. **Autoregressive Models** use the **probability chain rule** to transform high-dimensional density estimation into a series of one-dimensional prediction tasks, including the **Independent and Identically Distributed (IID)** assumption and the **log trick** for numerical stability and optimization efficiency.

Training neural networks to simulate density $p(x)$ assumes that the dataset is IID, so the joint likelihood of all samples is the product of individual sample likelihoods. **Log-likelihood** is used to convert this product into a sum, making it easier to optimize via gradient descent.

To handle high-dimensional data like images or text, **autoregressive models** decompose the probability as: $p(x) = \prod p(x_t | x_{<t})$.

- In **language**, this is natural (predicting the next word).
- In **images**, it requires a "raster scan" order (pixel-by-pixel).

While pixel-level autoregression is mathematically precise, its computational cost is high (sequence length is $O(N^2)$ for an $N 	imes N$ image), and it struggled with high-resolution data before the advent of tokenization.

The maximum likelihood objective function finds the neural network weights $\theta$ that make the observed data samples as "likely" as possible. It is the cornerstone of probabilistic learning. By maximizing the likelihood of training data, we are essentially trying to match the model's distribution with the true underlying data distribution. The distinction between "likelihood and probability": probability varies with data $x$ given fixed $\theta$; likelihood varies with $\theta$ given fixed data $x$.

$$
\theta_{ML} = \arg\max \theta \sum_{i=1}^N \log p(x^{(i)}; \theta)
$$

The probability chain rule (autoregression) states that the probability of an entire sequence equals the product of the probability of each element given all preceding elements. This decomposition is an exact identity in probability theory. it allows training models (like RNNs or Transformers) that can handle a series of tasks similar to supervised learning: "predict the next part based on history." By decomposing the joint distribution $V^T$ (vocabulary size $V$, length $T$) into $T$ independent distributions of size $V$, it avoids exponential complexity growth.

$p(x) = p(x_1, x_2, \dots, x_T) = p(x_1) \prod_{t=2}^T p(x_t | x_1, \dots, x_{t-1})$

- $x_t$: Sub-part at time $t$ (token/pixel), a portion of a single sample $x$.
- $x_{<t}$: Prefix sequence, the prediction context.

**Sequence Complexity**: Image autoregressive models (PixelRNN/PixelCNN) encounter issues with long sequences. A $1024 \times 1024$ image produces a sequence of about 3 million sub-pixels.
**Discretization**: Autoregressive models perform best on discrete data (text tokens or 8-bit pixel values), where softmax/cross-entropy loss can be used.
**Likelihood Evaluation**: To obtain the exact probability $p(x)$ of an image from a Transformer, the predicted probabilities for each pixel in the sequence must be multiplied.

### Variational Autoencoders (VAEs): The ELBO derivation, the reparameterization trick, and the regularization role of the KL divergence

Variational Autoencoders (VAE) combine deep learning and variational inference to learn structured latent representations by maximizing the lower bound of the data log-likelihood (ELBO). **Variational Autoencoder (VAE)** is an explicit but approximate density model that introduces a **latent variable** $z$ to explain the structure of data $x$. Key concepts include **variational inference** (using an approximate posterior q(z|x)), **KL divergence** (measuring the difference between distributions), and the **Evidence Lower Bound (ELBO)**, which is the training objective.

Transitioning from traditional **autoencoders** (learning a deterministic bottleneck for reconstruction) to VAEs, which are probabilistic. Calculating the true likelihood function $p(x) = \int p(x|z)p(z)dz$ is extremely difficult because the integral involves neural networks. The **solution** is to introduce an **encoder** (inference network) $q_\phi(z|x)$ to approximate the intractable posterior $p(z|x)$. Deriving the **ELBO** through Bayes' rule and the log trick decomposes the log-likelihood into a **reconstruction term** (how well z restores x) and a **prior term** (how close q(z|x) is to the Gaussian prior p(z)).

The Evidence Lower Bound (ELBO) states that the log-likelihood of data is at least equal to the reconstruction accuracy minus the complexity of the latent code. This is the VAE loss function, balancing compression (KL term) and fidelity (reconstruction term). The KL term acts as a regularizer, preventing the encoder from simply assigning a unique "ID" to each image, which would lead to overfitting.

$$
\log p(x) \geq \mathbb{E}_{z \sim q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) || p(z))
$$

- $\log p(x)$: Data log-likelihood, the "evidence" we want to maximize.
- $q(z|x)$: Approximate posterior distribution, the output of the encoder network.
- $p(x|z)$: Likelihood / Decoder, the output of the decoder network.
- $p(z)$: Latent prior distribution, typically $\mathcal{N}(0, I)$.
- $D_{KL}$: KL divergence, penalty for distribution mismatch.

The reparameterization trick allows gradients to flow through stochastic sampling steps by shifting randomness to an auxiliary noise variable. Instead of sampling directly from a distribution (which is non-differentiable), the sample is expressed as a deterministic transformation of a standard noise variable. This allows standard backpropagation to update encoder weights. Gradients pass through $\mu$ and $\sigma$, while "randomness" is isolated in $\epsilon$. This is crucial for training VAEs end-to-end.

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

- $z$: Latent sample, input to the decoder.
- $\mu$: Predicted mean, encoder output 1.
- $\sigma$: Predicted standard deviation, encoder output 2.
- $\epsilon$: Auxiliary noise, the random element (no gradient).

The "conflict" in the VAE loss function (the flow includes: input x $\to$ encoder $\to \mu, \sigma \to$ reparameterization $\to z \to$ decoder $\to \hat{x}$): The **reconstruction loss** requires unique $\mu$ and $\sigma \to 0$ for each $x$, while the **prior loss (KL loss)** requires $\mu \to 0$ and $\sigma \to 1$. These two parts of the ELBO objective directly conflict, creating a continuous, smooth latent space. Without the KL term, the latent space is "empty" and cannot be sampled; with it, the model is forced to map similar images to similar regions in the latent space. This explains why VAEs can generate new data by sampling $z \sim \mathcal{N}(0, I)$ and passing it to the decoder.

**Approximate Density**: VAEs do not maximize the true likelihood function but rather a lower bound. The "gap" between ELBO and the true likelihood is the KL divergence between the approximate and true posterior distributions.
**Diagonal Gaussian Assumption**: For computational efficiency, $q(z|x)$ is usually assumed to have diagonal covariance, meaning the model does not directly simulate correlations between latent dimensions. VAEs are common because their latent space is "interpretable"—changing one dimension of $z$ might smoothly change facial tilt or finger thickness.

---

### 2.2 Generative Models: Generative Adversarial Networks (GANs), Diffusion Models (Rectified Flow), Latent Diffusion and Conditional Generation

### Generative Adversarial Networks (GANs): The minimax objective, non-saturating loss, and training dynamics

Generative Adversarial Networks (GAN) are an implicit density modeling method where a generator attempts to fool a discriminator in a minimax game, avoiding explicit density estimation by achieving sample generation. **Implicit Density Models** do not output probability values $p(x)$ but provide a mechanism for sampling from the distribution, like **GANs**. Key concepts include the **minimax game** (a game between two networks), **Nash Equilibrium** (the theoretical optimal state where the generator matches the data distribution), and the **non-stationarity** of GAN training (the discriminator's "dataset" changes as the generator learns).

Contrasting GANs with previous explicit methods (VAEs, autoregressive models):

1. **Setup**: The generator ($G$) maps noise $z \sim p(z)$ to the data space $x_{fake}$. The discriminator ($D$) attempts to classify input as real ($x \sim p_{data}$) or fake (G(z)).

2. **Objective**: The discriminator ($D$) maximizes classification accuracy. The generator ($G$) minimizes $D$'s accuracy (trying to fool $D$).

3. **Training Dynamics**: Because there is no static loss curve related to generation quality, the training process is inherently unstable. The discriminator provides gradients for the generator.

4. **Challenges**:

- **Vanishing Gradients**: Early in training, $D$'s performance is too strong, causing $G$'s gradients to flatten. The "non-saturating" loss trick ( log D(G(z))) addresses this.
- **Mode Collapse**: The generator might memorize a few samples to fool $D$ without learning the full distribution.
- **Evaluation**: No objective metrics (like likelihood) to track progress; requires visual inspection or heuristic metrics.

The minimax objective function of GANs: $E_{x}[\log D(x)]$ indicates the discriminator aims to maximize the log probability of correctly identifying real data (output 1); $E_{z}[\log(1 - D(G(z)))]$ indicates the discriminator's goal is to maximize the probability of correctly identifying fake data (output 0, so $1-D(\dots)$ is near 1). The minimax generator tries to minimize the entire quantity (especially the second term) so that $D(G(z)) \approx 1$ (fooling the discriminator). This establishes adversarial dynamics. If trained to optimality, $p_g = p_{data}$. This avoids intractable integrals in VAE/MLE but introduces optimization instability.

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

- $G$: Generator network, maps $z$ to $x$.
- $D$: Discriminator network, maps $x$ to $[0, 1]$ probability.
- $V(D, G)$: Value function, the value $D$ maximizes and $G$ minimizes.
- $p_{data}$: Real data distribution we want to learn.
- $p_z$: Prior noise distribution, typically $\mathcal{N}(0, I)$.

Non-saturating generator loss (heuristic) does not minimize $\log(1 - D(G(z)))$ but maximizes $\log D(G(z))$. When $D$ is very confident (output $\approx 0$ for fake data), the gradient of $\log(1-x)$ is small (flat slope), whereas the gradient of $-\log(x)$ is significantly stronger when $x \approx 0$. This "trick" solves vanishing gradients early in training. The algorithm involves alternating gradient descent (updating $D$, then $G$). $L_G$ is the improved generator loss used in practice as a replacement for the minimax loss.

$$
L_G = - \mathbb{E}_{z \sim p_z(z)}[\log D(G(z))]
$$

**No Loss Curve**: $V$ as a progress indicator is meaningless. A stable $V$ might mean equilibrium or that both models have stalled. **Latent Smoothness**: Despite issues, GANs still learn a smooth latent space where interpolation in $z$ can cause smooth semantic changes in $x$ (not as blurry as VAEs, nor jumping as in simple memory).

### Diffusion Models (Rectified Flow): The noise schedule, velocity matching objective, and Euler integration for inference

Diffusion models (expressed in rectified form) learn to map simple noise to complex data by predicting velocity vectors in a flow field and integrating them over time during inference. **Diffusion Models** use the **Rectified Flow** framework (a simplified diffusion). This includes **noise scheduling** (interpolating data $x$ and noise $z$), **velocity prediction** (learning the denoising direction), and **Ordinary Differential Equation (ODE) solving** (inferring via Euler integration).

Diffusion is an iterative denoising process starting from noise $z \sim \mathcal{N}(0, I)$ and gradually transforming it into data $x \sim p_{data}$.

1. **Rectified Flow Setup**:

- Perform linear interpolation between data $x_0$ and noise $x_1$ (where $x_1 = z$).
- Define path $x_t = (1-t)x_0 + t z$ (or similar).
- Train neural network $v_\theta(x_t, t)$ to predict "velocity" vector $v = z - x_0$ (pointing from data to noise, by convention).

2. **Training**: Sample $t \sim U[0, 1]$, sample $z$, interpolate to get $x_t$, and minimize MSE between predicted and true velocity.
3. **Inference**: Starting from $z$ (time $t=1$), predict velocity and take a small step towards $t=0$ (Euler method), repeating the process.

Rectified Flow interpolation (noise state) at time $t$ is a simple weighted average of data and noise, defining a straight-line trajectory in high-dimensional space connecting specific data and noise points. This simplifies complex stochastic differential equations/score matching formulas in standard diffusion into a geometric "straight path" learning problem.

$$
x_t = (1 - t)x_0 + t z
$$

- $x_t$: Noise sample at time $t$, interpolation.
- $x_0$: Clean data sample from the dataset.
- $z$: Gaussian noise sample, the prior.
- $t$: Time / noise level, $t \in [0, 1]$.

Velocity matching loss trains the network to identify noise-mixed signals ($x_t$) and find the direction connecting the two endpoints ($z-x_0$). If the model perfectly learns this vector field, we can start from any random noise $z$ and follow the arrows to find a valid data point $x_0$. This is a supervised regression problem (unlike GANs), making the training process stable and loss curves meaningful.

$$
L( 	\theta) = \mathbb{E}_{t, x_0, z} [ || v_\theta(x_t, t) - (z - x_0) ||^2 ]
$$

- $v_\theta$: Neural network output, predicted velocity.
- $z - x_0$: Target velocity, vector from data to noise.
- $||\cdot||^2$: MSE loss, L2 norm.

The Euler integration step (inference) moves a small distance in the direction predicted by the model to reduce noise. This is the numerical solution of the ODE defined by the flow field, generating results from noise through iterative refinement. (Similar to a Python training loop: iterate data $\to$ sample $z, \to$ interpolate $\to$ predict $v \to$ MSE loss).

$$
x_{t - \Delta t} = x_t - v_\theta(x_t, t) \cdot \Delta t
$$

- $x_{t - \Delta t}$: Next denoising state, step closer to data.
- $\Delta t$: Step size, e.g., $1/50$ for 50 steps.

**Single-step vs. Multi-step**: Rectified flow models often require many steps (e.g., 50) in inference, making them slower than GANs (single-step), though distillation techniques can address this.
**Stability**: Loss is MSE; if the model learns, MSE consistently decreases, a huge user experience improvement over GAN training.
**Coupling Problem**: The model doesn't know which $x_0$ corresponds to a given noise $x_t$ (many data points might mix into the same noise state); it learns an expected/average vector, which correctly transfers probability mass mathematically.

### Conditional Generation and Latent Diffusion: Examination of Classifier-Free Guidance (CFG) and the two-stage LDM architecture (VAE - Diffusion Transformer)

Latent Diffusion Models (LDM) achieve scalable text-to-image generation by applying a diffusion process in a compressed latent space and leveraging Classifier-Free Guidance (CFG) to strictly follow conditional prompts. **Conditional Diffusion Models** include **Classifier-Free Guidance (CFG)** (a technique for amplifying conditional signals) and **Latent Diffusion Models (LDM)** (a workflow combining VAE compression and diffusion generation). Other aspects include **Diffusion Transformers (DiT)** and video generation.

Methods to control diffusion generation:

1. **Conditioning**: Add extra input $y$ (e.g., text, class label) to the diffusion model $v_\theta(x_t, t, y)$.

2. **Classifier-Free Guidance (CFG)**: Train the model to handle both conditional input ($y \neq \text{null}$) and unconditional input ($y = 	\text{null}$). During inference, extrapolate from the unconditional prediction towards the conditional one.

3. **Latent Diffusion Model (LDM)**: Pixel-space diffusion is computationally massive ($1024 \times 1024$ pixels). The **solution** is using a Variational Autoencoder (VAE) to compress images into a smaller latent space ($z$). Train the diffusion model to generate latent variables $z_t \to z_0$, then use the VAE decoder to map $z_0 \to x$.

4. **Scalability**: Modern models (Sora, Veo) use **Diffusion Transformers (DiT)** to efficiently handle tokenized inputs (text + image/video patches).

Classifier-Free Guidance (CFG) prediction takes the vector pointing to the conditional distribution and subtracts the vector pointing to the general data distribution. Scale the difference and add it back, pushing the sample _away_ from general images _towards_ images highly matching the prompt $y$. This is the key trick for high-fidelity text-to-image generation (e.g., Stable Diffusion, Midjourney).

$$
\tilde{\epsilon}_\theta(x_t, y) = (1 + w) \epsilon_\theta(x_t, y) - w \epsilon_\theta(x_t, \emptyset)
$$

- $\tilde{\epsilon}_\theta$: Guided noise prediction for the sampling step.
- $\epsilon_\theta(x_t, y)$: Conditional prediction given prompt $y$.
- $\epsilon_\theta(x_t, \emptyset)$: Unconditional prediction, prompt discarded.
- $w$: Guidance scale, typically $> 0$ (e.g., 7.5).

The latent diffusion training loop trains a diffusion model to denoise latent representations ($z$) instead of raw pixels ($x$), drastically reducing computational costs (e.g., $64 \times 64$ latent vs. $512 \times 512$ pixel). The diffusion model focuses on semantic structure while the VAE decoder handles high-frequency details. This is the architecture behind Stable Diffusion. Flow includes: input text $\to$ CLIP encoder $\to$ U-Net/DiT (latent space) $\to$ VAE decoder $\to$ image.

$$
L_{LDM} = \mathbb{E}_{z \sim E(x), \epsilon \sim \mathcal{N}(0, I), t} [ || \epsilon - \epsilon_\theta(z_t, t, y) ||^2 ]
$$

- $E(x)$: VAE encoder, fixed during diffusion training.
- $z_t$: Noisy latent value, compressed + noise.
- $\epsilon_\theta$: Diffusion model, predicts added noise.
- $y$: Condition, e.g., text embedding.

**CFG Trade-off**: Higher guidance scale $w$ improves prompt adherence but reduces sample diversity and introduces artifacts ("over-saturation").
**Two-stage Training**: LDM requires first training a VAE (often using GAN loss for sharpness), then freezing it to train the diffusion model.
**Video Complexity**: Scaling LDM to video adds a temporal dimension ($T \times H \times W$) to the latent space, significantly increasing sequence length and memory overhead (e.g., video needs 76k tokens vs. 1k for an image).

---

## 3. 3D Vision and Vision-Language Models

### 3.1 3D Vision

3D vision transitions from regular 2D pixel grids to irregular geometric representations like point clouds and meshes, requiring specialized data structures to capture shape, connectivity, and surface properties. Unlike 2D images structured as regular pixel matrices, 3D data is often irregular. **Explicit Representations** define geometry directly via coordinates or connectivity. **Point Clouds** are collections of 3D coordinates ($x, y, z$), while **Polygonal Meshes** add topological information by defining how these points connect into faces (usually triangles or quads). The concept of **Surface Normals** is introduced—vectors perpendicular to the surface at a given point, crucial for realistic lighting and rendering.

### Explicit (Points, Meshes) vs. Implicit (SDFs, Level Sets)

The challenge of 3D vision is "inverting" the process of turning the 3D world into 2D pixels. It divides 3D representation into **explicit** (points, meshes) and **implicit** (functions). Trade-offs in storage, editing, and rendering: point clouds are flexible but undersampled, while meshes can capture trillions of triangles to build high-fidelity models (e.g., Google Earth).

- **Point Clouds**: The simplest form, often raw output from sensors like LiDAR or depth cameras. They lack connectivity, making it hard to distinguish topologies (e.g., circle vs. torus).
- **Polygonal Meshes**: The standard for computer graphics and gaming, consisting of vertices and faces. While expressive, they pose challenges for early deep learning models due to variable dimensions and irregular structure.

A point cloud is a matrix or set where each row is a coordinate in space. It is the "raw" format of 3D vision, capturing surface locations but lacking a surface "skin." This is the input format for models like PointNet, which must handle the unordered nature of this set.

$$
P = \{p_i\}_{i=1}^N, \quad p_i \in \mathbb{R}^d
$$

- $P$: Point cloud, a set containing $N$ points.
- $p_i$: A single point, a vector in d-dimensional space.
- $N$: Number of points, can be a variable.
- $d$: Dimension, typically 3 ($x,y,z$) or 6 (including normals $n_x, n_y, n_z$).

Surface normals and lighting intuition: the principle of surface normals in rendering $I = L \cdot n$ is that a point's brightness depends on the angle between the light source and the surface "facing" direction. Here, $I$ is perceived intensity; $L$ is the light direction vector; $n$ is the surface normal. Point clouds without normals look "flat." Adding normals (creating "surface elements") enables shading for realistic 3D visualization. Surfaces with normals provide more information about local geometry to deep networks.

**Topological Blind Spots**: Point clouds cannot represent connectivity. A set of points might be a solid or a thin shell; without connectivity (mesh), models face a "partial information" problem.
**Sampling Bias**: "Non-uniform sampling" (e.g., dense at the head, sparse at the tail) is a critical issue in 3D data processing, requiring algorithmic regularization.

### Parameterization and Implicit Representation: parametric circles and implicit spheres

Implicit representations define 3D geometry as the solution to a function $f(x,y,z)=0$, providing an efficient way to judge whether a point is inside or outside an object and allowing for smooth shape blending. Contrast **Parameterized Representations** (mapping directly from low-dimensional parameter space to 3D space) with **Implicit Representations** (defining geometry via constraints)—a mathematical shift from "sampling points" to "querying space."

- **Parameterization**: Uses functions like Bézier curves/surfaces to generate points on a surface by varying parameters $(u, v)$. The advantage is ease of sampling points on the surface by plugging in random $(u, v)$ values. However, it is hard to judge if a specific point in space is inside or outside the object.
- **Implicit**: Defines surfaces as "level sets" where $f(x,y,z) = 0$. The advantage is ease of testing containment ("inside/outside" test) by calculating $f(query)$. If the result is negative, the point is inside. However, finding points exactly on the surface is hard, as it requires solving $f(x,y,z)=0$.
- **Signed Distance Function (SDF)**: A special implicit representation where $f(x,y,z)$ indicates the distance to the nearest surface point, with the sign indicating if the point is inside (negative) or outside (positive).
- **Composition**: Implicit functions support powerful "Boolean" operations. Taking the minimum or maximum of function values finds the union or intersection of shapes, applied in CAD models and **Computer-Aided Design**.

A parameterized circle "paints" the circle's shape by varying $t$ from $0$ to $2\pi$. This is the **explicit/parameterized** method, providing a "surface formula." Extending this to 3D with parameters $(u, v)$ gives parameterized surfaces like Bézier surfaces.

$$
f(t) = (\cos t, \sin t)
$$

- $t$: Parameter, representing angle/degree.
- $f(t)$: 2D point, output coordinates $(x, y)$.

An implicit sphere defines that if the sum of squares of coordinates is exactly 1, the point lies on the unit sphere. This is the **implicit** method: if $f(x,y,z) < 0$, the point is **inside**; if $f(x,y,z) > 0$, it is **outside**. Modern deep learning models (e.g., **DeepSDF** or **NeRF**) use neural networks to approximate $f$ to handle complex shapes like humans or furniture.

$$
f(x, y, z) = x^2 + y^2 + z^2 - 1 = 0
$$

- $x, y, z$: Query point, 3D space coordinates.
- $f$: Implicit function, defining the sphere constraint.

A fundamental trade-off: explicit methods are good for **generating** points; implicit methods are good for **classifying** points.
**Blending**: Implicit functions support "smooth blending." Adding two distance functions creates a smooth transition between objects, hard to achieve with meshes. **CAD Relation**: Linking implicit functions with CAD models, where objects are built via logical operations (union, intersection) on simple primitives. "Metaballs" in Maya/Blender are a common example.

### Voxels and 3D Datasets (3D grids and ShapeNet)

Voxels provide a regular 3D grid representation, allowing direct application of 3D CNNs. Large-scale datasets like ShapeNet drive the shift from traditional graphics to 3D deep learning. **Voxels** (volumetric pixels) and 3D vision data infrastructure:

- **Voxels**: 3D grids of "volume pixels," a binarized or density-based representation of an implicit function sampled on a regular grid.
- **Octrees**: A hierarchical data structure for optimizing voxel storage. Instead of a uniform grid, it recursively divides space only where surfaces exist, saving memory in empty areas.
- **ShapeNet**: A benchmark dataset (55 categories, >51,000 models) providing necessary scale for training 3D neural networks.
- **Objaverse**: A newer, larger dataset (millions of models) including textures and high-quality scans.

When deep learning first entered 3D, researchers were naturally drawn to **Voxels** as direct extensions of 2D pixels, enabling **Volumetric CNNs** (3D CNNs). **Pros**: easy analogy to 2D CNN; supports 3D convolution. **Cons**: massive memory consumption ($O(N^3)$ complexity). A $100^3$ grid has 1 million points but low geometric resolution (jagged edges).

Before 2014, dataset scales were too small (e.g., Princeton Shape Benchmark had only 10 models per category). **ShapeNet** changed this by providing thousands of models for categories like chairs and cars, enabling researchers to train robust generative and discriminative models. Recent research (e.g., Meta's MVImgNet) uses iPhone scans of real objects to bridge the "synthetic-to-real" gap.

Voxel grid complexity: doubling resolution increases memory/compute by 8x ($2^3$), with $Memory \propto N^3$ where $N$ is resolution per dimension (e.g., 32, 64, 128). This "curse of dimensionality" means voxels are rarely used above $64^3$ or $128^3$ without optimization like octrees. Octrees reduce this curse by only dividing space where "data" exists, allowing resolutions up to $256^3$ or $512^3$.

**Graphics vs. Vision Conflict**: "Graphics experts" dislike voxels for inefficiency and poor visual quality, while "vision experts" like them for CNN compatibility. This drove development of more efficient representations like PointNet. **"iPhone Revolution"**: People scanning objects with iPhones for $1 rewards (e.g., **Common Objects in 3D (CO3D)** dataset) highlights the shift from synthetic CAD models to noisy real-world data. **Resolution Bottleneck**: Due to memory, early voxel-based generative models (e.g., early 3D GANs) could only handle "blurry" shapes.

### 3D Deep Learning (PointNet and 3D GAN)

PointNet introduced a breakthrough architecture handling unordered point clouds via symmetric functions for permutation invariance; 3D GAN extends adversarial training to volumetric data for object synthesis. These were among the first "native 3D" deep learning architectures.

- **Multi-View CNN**: A 2D-to-3D approach rendering 3D objects into multiple 2D images processed by standard CNNs.
- **PointNet**: A network directly processing unordered point clouds by satisfying **permutation invariance** (output independent of point order) and **sampling invariance** (robust to point count).
- **3D GAN**: GANs applied to voxel grids.
- **Chamfer Distance (CD)** and **Earth Mover's Distance (EMD)**: Differentiable loss functions for comparing point sets.

Evolution of 3D learning tasks (classification, generation, reconstruction):

1. **Multi-View**: The simplest start. Render a 3D chair from 12 angles, use 2D CNNs, then pool results. Leverages ImageNet pre-training but ignores underlying 3D structure.
2. **PointNet**: Directly processes points instead of voxels. Uses "symmetric functions" (like max pooling) on high-dimensional point embeddings, ensuring global features don't change if point 1 and 2 are swapped.
3. **3D Generation**: Early models used 3D GANs on voxels. Later models (e.g., **AtlasNet**) used MLPs to learn "folding" functions transforming 2D patches into 3D surfaces, yielding smoother results than points or voxels.
4. **Point Comparison**: Training generative models on points needs a loss function. **Chamfer Distance** finds nearest neighbors for each point, while **Earth Mover's Distance** uses optimal one-to-one matching (costlier but more accurate).

PointNet Invariance: To handle point clouds, first transform each point into a feature vector, then use an order-agnostic function (e.g., Max) to "compress" these vectors into one. Finally, use that "summary" for decisions. This is the core of **PointNet**, solving the "unordered set" problem in 3D vision. It's sampling-invariant; adding points doesn't change results as long as the maximum doesn't increase.

$$
g(x_1, \dots, x_n) \approx \gamma \left( \text{MAX}_{i=1 \dots n} \{h(x_i)\}
\right)
$$

- $x_i$: Input point, usually $(x, y, z)$.
- $h$: Point-wise MLP, mapping points to high-dimensional embeddings.
- $\text{MAX}$: Max pooling, a **symmetric function**.
- $\gamma$: Global MLP, processing aggregated features.
- $g$: Final prediction, class label or feature.

Chamfer Distance: For each point in the first set, find the nearest in the second and measure distance. Repeat inversely and sum. This differentiable method compares shapes with misaligned point orders, primarily used as a loss for 3D reconstruction and generation.

$$
d_{CD}(S_1, S_2) = \sum_{x \in S_1} \min_{y \in S_2} ||x - y||_2^2 + \sum_{y \in S_2} \min_{x \in S_1} ||y - x||_2^2
$$

- $S_1, S_2$: Point sets, predicted and ground truth.
- $||\cdot||_2^2$: Squared L2 norm, the distance metric.

**PointNet++** and Graph Neural Networks (GNN) improved by capturing local neighborhood structures, which original PointNet (a global aggregator) largely ignored.
**CycleGAN for 3D**: An interesting application uses **CycleGAN** to convert 3D depth maps to color images, allowing simultaneous adversarial loss on 3D geometry and 2D appearance.
**AtlasNet Innovation**: AtlasNet's "origami" intuition bridges discrete points and smooth parameterized surfaces. Maya/Blender's "UV unwrapping" is the inverse concept.

### Neural Rendering (NeRF and Gaussian Splatting): The NeRF 5D mapping and the Differentiable Volume Rendering integral

Neural Radiance Fields (NeRF) revolutionized 3D vision by representing scenes as continuous 5D functions of density and color, optimized via differentiable volume rendering from 2D images. Gaussian Splatting provides a high-speed explicit alternative. This marks a shift from discrete 3D geometry to continuous **neural fields**.

- **Deep Implicit Functions**: Using MLPs to represent $f(x,y,z) = 	\text{inside/outside}$.
- **Neural Radiance Fields (NeRF)**: A function mapping 3D coordinates $(x,y,z)$ and view angles $(	\theta, \phi)$ to volume density $\sigma$ and RGB color $c$.
- **Differentiable Volume Rendering**: A process integrating density and color along camera rays to compute 2D pixel colors, allowing backpropagation from image loss to 3D parameters.
- **3D Gaussian Splatting**: A recent method using 3D Gaussian "splats" to represent scenes for real-time rendering speeds.

Transition from **DeepSDF** (learning geometry) to **NeRF** (learning appearance). NeRF uses a neural network as "storage" instead of voxels. Querying the network with coordinates tells you what's at that location. **NeRF Mechanism**: For each pixel in a 2D image, fire a ray into the 3D scene. Sample points along the ray, query an MLP for each point's color and density. Use the **volume rendering equation** to "blend" these into one pixel color. Compare with real photos, then update the MLP. **Breakthrough**: NeRF allows 3D reconstruction from 2D photos _without_ explicit 3D supervision, needing only camera pose info. **Gaussian Splatting**: NeRF is slow due to hundreds of MLP queries per ray. Gaussian Splatting replaces MLPs with explicit Gaussian blocks, enabling rasterization (like traditional graphics) over 1000x faster than NeRF (150 FPS vs. 0.05 FPS).

NeRF Mapping: The scene is a large function. Tell me where you are and where you're looking, and I'll tell you the color and transparency. View direction $\mathbf{d}$ is crucial for **view-dependent effects** like specular highlights on shiny cars. Moving your head changes color, but density $\sigma$ stays constant. This is the ultimate "implicit representation."

$$
F_\theta: (\mathbf{x}, \mathbf{d}) 	\to (\mathbf{c}, \sigma)
$$

- $\mathbf{x}$: 3D position, $(x, y, z)$.
- $\mathbf{d}$: Viewing direction, $( \theta, \phi)$ or $(d_x, d_y, d_z)$.
- $\mathbf{c}$: Radiance (color), RGB values.
- $\sigma$: Volume density, opacity / "is something here?".
- $\theta$: MLP parameters, network weights.

Differentiable Volume Rendering: To compute a pixel's color, sum color values of all points on the ray, weighted by their "density" ($\sigma$) and whether they are blocked by objects in front ($T$). This is the "bridge" between 3D and 2D. Since every part of this sum is differentiable, errors in 2D images can train the 3D MLP. This is the core of "neural rendering."

$$
C(\mathbf{r}) = \sum_{i=1}^N T_i (1 - \exp(-\sigma_i \delta_i)) \mathbf{c}_i, \quad T_i = \exp(-\sum_{j=1}^{i-1} \sigma_j \delta_j)
$$

- $C(\mathbf{r})$: Predicted pixel color, final output for ray $\mathbf{r}$.
- $T_i$: Transmissivity, light reaching point $i$ (unblocked).
- $\sigma_i$: Density at point $i$ from NeRF MLP.
- $\mathbf{c}_i$: Color at point $i$ from NeRF MLP.
- $\delta_i$: Distance between samples, the integration step.

**"Empty Space" Problem**: NeRF's biggest weakness is querying the network even in empty space. Gaussian Splatting solves this by only placing "splats" where actual geometry exists.
**Implicit Function Legacy**: NeRF didn't appear from nowhere; it's a direct evolution of **DeepSDF** and **Level Set** methods from 2019.
**Photometry vs. Color**: Compared to old 3D models, NeRF uses "photometry" (view-dependent) rather than just "color" for realistic images.

### Structure and Hierarchical Representation: Modeling hierarchies, symmetries, and the shift toward LLM-driven program synthesis for 3D objects

Modern 3D vision goes beyond geometric details, using GNNs and program synthesis to model structural regularities, hierarchies, and relations for smarter scene understanding. High-level structure of 3D objects and scenes:

- **Hierarchical Graphs**: Representing objects as trees of parts (e.g., chair $\to$ back, seat, base $\to$ legs).
- **Symmetry and Repetition**: Modeling the fact that chairs usually have four identical legs in symmetric arrangement.
- **Mobility**: Capturing how parts move (e.g., a laptop lid can rotate).
- **Program Synthesis**: Using code-like structures (loops, symmetric ops) to define 3D shapes.
- **Scene Graphs**: Modeling relations between objects (e.g., "chair is next to table").

Pixels, points, and voxels only capture "low-level" details.

- **Challenge**: A point cloud of a chair doesn't know it has four legs or that it's symmetric. Changing one leg should change others.
- **Hierarchical Models**: Datasets like **PartNet** provide semantic labels for parts and hierarchies. Neural networks (e.g., recursive or graph neural networks) encode these trees.
- **Structural Regularity**: Models can be trained to enforce symmetry constraints, ensuring legs align perfectly.
- **Future Directions**: Current trends use **Large Language Models (LLM)** to generate **programs** (scripts) building 3D shapes. LLMs understand the "concept" of a chair (needs seat and legs), while implicit functions/NeRF handle fine geometry.

The shift to hierarchical graphs marks a transition from "geometry" to "semantics." The model no longer just predicts points but "parts."
**Language Model Integration**: Using GPT to output 3D building programs is a 2024-2025 frontier trend, leveraging world knowledge to solve 3D ambiguity.
**Robot Linking**: **Mobility** (articulated parts) is vital for robotics, as robots need to know how to interact with drawers or doors, not just see them as static meshes.

---

### 3.2 Foundation Models and CLIP

CLIP is a vision-language foundation model trained on massive image-text pairs using symmetric contrastive loss, enabling robust zero-shot classification without task-specific training. **Foundation Models** are large-scale models pre-trained on diverse data, adaptable to many downstream tasks (e.g., GPT, CLIP). The core mechanism is **contrastive learning**, specifically a **SimCLR**-style objective adapted for multi-modal data. The key innovation is **Zero-shot Classification**, where the model uses natural language descriptions to classify images into categories it never explicitly saw during training.

### Foundation Models & CLIP: The contrastive loss and zero-shot classification mechanism

Contrasting traditional task-specific models (collect data $\to$ train $\to$ test) with foundation models (pre-train once $\to$ adapt everywhere). **Contrastive Language-Image Pre-training** (CLIP) extends SimCLR. Instead of contrasting different augmented views of the same image, CLIP contrasts an image with its corresponding text description.

1. **Training**: Consists of an **Image Encoder** (ResNet or ViT) and a **Text Encoder**. Uses a contrastive loss maximizing cosine similarity between $N$ matching image-text pairs while minimizing similarity with $N^2 - N$ mismatched pairs in a batch.
2. **Zero-shot Inference**: To classify an image without fine-tuning, the model embeds the image and a set of candidate text labels (e.g., "a photo of a dog", "a photo of a cat"). The predicted class is the text embedding with the highest similarity to the image embedding.
3. **Prompt Engineering**: Using prompts like "a photo of a [CLASS]" instead of just [CLASS] significantly improves performance (e.g., +1.3% on ImageNet) by bridging the gap between pre-training data (captions) and test labels.

Contrastive loss (InfoNCE adapted for CLIP): for a given image $I_i$, the model tries to maximize the probability of selecting the correct text $T_i$ from all $N$ possible texts in the batch. This aligns visual and textual representation spaces. If an image contains a "dog," its vector should be close to the "dog" text vector and far from "car." This is a symmetric version of SimCLR loss for all modalities.

$$
\mathcal{L}_{i,j} = - \log \frac{\exp( 	\text{sim}(I_i, T_i) / 	\tau)}{\sum_{k=1}^N \exp( 	\text{sim}(I_i, T_k) / 	\tau)}
$$

- $I_i$: Image embedding, output of image encoder.
- $T_i$: Text embedding, output of text encoder.
- $\text{sim}(u, v)$: Cosine similarity, $\frac{u \cdot v}{||u|| ||v||}$.
- $\tau$: Temperature parameter, controlling distribution softness.
- $N$: Batch size, number of pairs in a mini-batch.

Zero-shot classification via nearest neighbor: to classify image $x$, first embed it into a vector. Then, embed all possible class names (wrapped in prompts) into vectors. Select the class vector closest to the image vector, turning classification into a **retrieval** problem. The model doesn't need re-training for new classes; just change set $C$. This makes foundation models "adaptable to any task."

$$
y^* = \arg\max_{c \in C} \text{sim}(E_{image}(x), E_{text}( \text{"a photo of a } c 	\text{"}))
$$

- $y^*$: Predicted class, the winner.
- $x$: Input image, the query.
- $C$: Class set, e.g., {dog, cat, airplane}.
- $E_{image}$: Image encoder, ResNet or ViT.
- $E_{text}$: Text encoder, Transformer.

**Batch Size Dependency**: CLIP performance heavily depends on large batch sizes (e.g., 32k). Smaller batches lack enough "hard negatives" (e.g., distinguishing Corgis from other dogs needs both in the batch). **Compositional Failure**: CLIP struggles with compositional reasoning (e.g., "cup on grass" vs. "grass in cup") as it learns "bag-of-words" associations rather than structured relations. **Prompt Ensembling**: Averaging embeddings of multiple prompts ("a photo of a dog", "a sketch of a dog") improves performance and robustness of text representation.

#### Vision-Language Models: LLaVA (linear projection), Flamingo (gated cross-attention), and Molmo (data-centric optimization)

Modern Vision-Language Models (VLM) like LLaVA, Flamingo, and Molmo integrate visual features into LLMs via adapters and cross-attention mechanisms, enabling pixel-based multi-modal reasoning. **VLM** development involves integrating vision encoders (e.g., CLIP) with Large Language Models (e.g., LLaMA), using **cross-attention mechanisms**, **adapters** (linear layers or Perceiver samplers), and **grounding** (linking output to specific image regions). Architectural choices vary between LLaVA's simple projection and Flamingo's interleaved cross-attention.

VLM Evolution:

1. **LLaVA**: Uses a frozen CLIP image encoder and a frozen LLM, connected via a simple trainable linear projection. It treats image tokens as prefixes to text prompts.
2. **Flamingo**: Introduced the **Perceiver Resampler** to compress variable-sized image features into fixed tokens. It injects these visual tokens into the LLM via **Gated Cross-Attention** layers interleaved within Transformer blocks, allowing the model to handle interleaved image-text sequences (few-shot learning).
3. **Molmo**: A fully open-source VLM (weights, data, code). It emphasizes **data quality** over quantity, using dense human-annotated captions instead of noisy web alt-text. It features pixel localization (pointing to objects) to reduce hallucinations.

LLaVA Projection (Linear Adapter): extracts high-dimensional visual features from CLIP and multiplies them by a matrix to match the LLM's text embedding dimension. This aligns visual and textual modalities, allowing the LLM to treat image patches as "visual words." This is the simplest "early fusion" method.

$$
h_v = W \cdot Z_{v} + b
$$

- $h_v$: Projected visual tokens, input to LLM.
- $Z_{v}$: CLIP visual features from the penultimate layer.
- $W$: Weight matrix, learnable projection.
- $b$: Bias vector, learnable bias.

Flamingo Gated Cross-Attention: the model attends to image features based on current text context. Results are scaled by a gate parameter $tanh(\alpha)$ and added to original input (residual connection). Initializing $\alpha=0$ means the model starts trained as a standard language model (ignoring images) and gradually learns to integrate visual info, stabilizing training. This allows Flamingo to process multi-modal inputs without destroying pre-trained language capabilities. (Flamingo cross-attention architecture: Dense $\to$ Tanh $\to$ CrossAttn $\to$ Dense $\to$ Tanh $\to$ Add.)

$$
y = tanh(\alpha) \cdot 	\text{CrossAttn}(LN(x), 	\text{image}) + x
$$

- $y$: Layer output, updated hidden state.
- $x$: Input hidden state from previous LLM layer.
- $LN(x)$: Flamingo's gated cross-attention is an extra module inserted into LLM blocks, usually Pre-LN style, applying LayerNorm before Cross-Attn.
- $\text{CrossAttn}$: Cross-attention mechanism (Query = text, Key/Value = image).
- $\alpha$: Learnable gate parameter, initialized to 0.
- $tanh$: Hyperbolic tangent, the activation.

**Data Quality vs. Quantity**: Molmo proved 700k high-quality, dense descriptions can outperform models trained on billions of noisy web pairs. **Grounding**: Molmo can output point coordinates (e.g., for counting or robotics), a key difference from text-only models. **Open vs. Closed**: Bridging the gap between open models (LLaVA) and proprietary ones (GPT-5) by open-sourcing all resources is Molmo's goal.

### Promptable Segmentation (SAM) and Chain-of-Thought Programming

Segment Anything Model (SAM) introduced a promptable foundation model for image segmentation, while multi-modal chain-of-thought programming (e.g., VisProg) achieves complex reasoning by composing foundation models into executable programs. **Promptable Segmentation** (specifically SAM) and **Multi-modal Chain-of-Thought Programming** (VisProg). SAM accepts various prompts (points, boxes, text) to output segmentation masks. Chain-of-thought involves code generation (LLM) linking specialized models (e.g., detection, counting, or reasoning) to solve complex tasks beyond single-model capabilities.

Two key advances:

1. **Segment Anything Model (SAM)**: Trained on a massive dataset (SA-1B: 11M images, 1.1B masks), consisting of an **Image Encoder** (ViT), **Prompt Encoder**, and **Mask Decoder**. It solves ambiguity (e.g., pointing to a shirt vs. a person) by outputting three masks at different granularities.
2. **Chain-of-Thought (VisProg)**: Instead of training one giant model for all tasks, an LLM (GPT) generates Python code calling specialized visual APIs (e.g., `Detect(image, "person")`, `Count(detections)`). This solves multi-step problems (e.g., "which image is more crowded?") by decomposing them into logical steps.

Mask Decoding (Ambiguity Resolution): the model outputs three possible masks for a single prompt. During training, only the mask matching ground truth best is penalized. This solves inherent ambiguity in segmentation prompts (e.g., "pointing to person" vs. "pointing to shirt"). The model learns to predict valid masks at multiple scales, vital for interactive segmentation.

$$
M^* = \arg\max_{m \in \{m_1, m_2, m_3\}} 	\text{IoU}(m, M_{GT})
$$

- $M^*$: Selected mask prediction, the best of 3.
- $m_k$: Predicted mask $k$ (whole, part, subpart).
- $M_{GT}$: Ground truth mask from dataset.
- $\text{IoU}$: Intersection over Union, the overlap metric.

Chain-of-Thought Logic (VisProg example): the LLM translates natural language questions into a sequence of functions calling specialized vision models. Complex reasoning (e.g., "is the person to the left of the car?") is possible without end-to-end task-specific training, leveraging LLM logic and specialized vision perception. (VisProg generates Python-like pseudo-code: `boxes = Detector(img, 'person'); count = len(boxes)`.)

$$
\text{Result} = 	\text{LLM}( 	\text{Question}) 	\to 	\text{Program} 	\to 	\text{Execution}( 	\text{VModels})
$$

- $\text{LLM}$: Large Language Model, e.g., GPT-5.
- $\text{Program}$: Generated Python code, e.g., `img.detect('dog')`.
- $\text{VModels}$: Vision models for detection, segmentation, etc.

**Data Scale**: SAM's success is due to its "data engine" loop (model prediction $\to$ human correction $\to$ re-training), scaling the dataset from thousands to 1 billion masks.
**Zero-shot Transfer**: Chain-of-thought models enable zero-shot transfer to tasks never seen in training (e.g., replacing "desert with grass" via chained segmentation and inpainting).
**Latency Cost**: Chaining multiple large models (GPT + DINO + SAM + Stable Diffusion) introduces significant computational cost and latency compared to a single end-to-end model.

---

## 4. Building and Perceiving in Robot Learning

Robot learning redefines machine learning as a sequential decision-making process involving states, actions, and rewards, requiring embodied perception systems that actively interact with unstructured physical environments. **Robot Learning** differs from standard supervised learning. The core concept is **Sequential Decision Making**, usually modeled as a **Markov Decision Process (MDP)**. Key components include **State ($S$)**, **Action ($A$)**, **Reward ($R$)**, and environmental **Dynamics** (how states evolve). **Embodied Perception** involves an agent's sensors (vision, touch) forming a feedback loop with its actions, contrasting with passive computer vision.

### The shift from IID supervised learning to sequential MDPs and the role of active, embodied perception

Contrasting **Supervised Learning** (mapping static input $x$ $\to$ label $y$) with **Robot Learning** (interacting with a changing world).

1. **Problem Description**: Defined as an agent executing action $a_t$ in state $s_t$ to maximize cumulative reward $r_t$. The environment updates to state $s_{t+1}$ based on the action.
2. **Examples**:

- **Handstand**: State = angle/velocity; Action = force; Reward = staying upright.
- **Go**: State = board position; Action = placing a stone; Reward = win/loss.
- **Language Model**: State = context; Action = next word; Reward = coherence/feedback.

3. **Perception**: Robot perception is **embodied** (physically located), **active** (movable for better views), and **contextualized** (real-time). It handles "unstructured" real-world data (occlusion, deformation) and needs multi-modal perception (touch, depth).

Sequential Interaction Loop: future states depend on current state and chosen actions. This dependency creates the "non-stationarity" challenge in robot learning. Data distribution changes based on the agent's policy. Unlike IID supervised learning, robot data is highly correlated and action-dependent.

$$
s_{t+1} = \mathcal{T}(s_t, a_t)
$$

- $s_t$: State at time $t$, physical configuration.
- $a_t$: Action at time $t$, motor commands/torques.
- $\mathcal{T}$: Transition dynamics, the world's physics.
- $s_{t+1}$: Next state, result of the action.

Objective Function (Implicit): we want to find a policy maximizing total reward over time, defining the optimization goal. The difficulty is "credit assignment"—determining which early action led to final success or failure. This is the fundamental goal of reinforcement learning algorithms discussed later.

$$
\pi^* = \arg\max_\pi \mathbb{E} \left[ \sum_{t=0}^T \gamma^t r(s_t, a_t) \right]
$$

- $\pi$: Policy, mapping $s$ to $a$.
- $r(s, a)$: Reward function, the feedback signal (e.g., +1 on success).
- $\gamma$: Discount factor, weighting future rewards.

**Active Perception**: Robots can _create_ their own data (e.g., moving objects to see behind them), a capability static datasets like ImageNet lack.
**Reward Engineering**: Defining $r_t$ (e.g., "cloth is folded") is non-trivial, often needing careful human evaluation or proxy models.
**Result Assignment**: A key difficulty is the delay between action and result (e.g., 100 moves in Go before winning), making policy training hard.

### Reinforcement Learning and Model-Based Planning: Q-learning, Model-Based Trajectory Optimization, and the trade-offs between model-free and model-based approaches

Contrasting Model-Free Reinforcement Learning (learning policies via trial and error) and Model-Based Learning (learning environment dynamics to plan actions) regarding sample efficiency, sim-to-real transfer, and generalization. **Reinforcement Learning (RL)** algorithms (specifically **Q-learning**) and **Model-Based Planning**.

- **Model-Free RL**: The learner directly maps states to actions or values without understanding physics. Key algorithms: **DQN**, **PPO**, **SAC**.
- **Model-Based RL**: The learner approximates transition dynamics $s_{t+1} = f(s_t, a_t)$ and uses it to simulate the future and optimize trajectories.
- **Sim-to-Real Transfer**: The challenge of transferring policies trained in simulation (low cost, high speed) to the real world (high cost, complex), often using **Domain Randomization**.

Two main approaches:

1. **RL (Model-Free)**: Demonstrated via Atari games (DQN) and AlphaGo. **Q-learning** learns a function $Q(s, a)$ predicting future rewards. Actions are chosen to maximize $Q$.

- **Success**: Locomotion (walking robots) is largely "solved" via RL due to robust sim-to-real transfer via randomization.
- **Failure**: Dexterous manipulation is hard due to complex contact physics and the "sim-to-real gap."

2. **Model-Based Learning**: Learning a "world model" predicting the next state without repeated trial and error. More efficient and interpretable than pure RL. Uses the model to "imagine" outcomes and optimize actions (e.g., gradient descent on action sequences). Examples: "Visual Prediction" (pixel dynamics), keypoint dynamics, and particle dynamics (for deformable objects like dough).

Q-function (Bellman concept): the value of an action is the immediate reward plus the value of the best next action. This recursive definition allows learning delayed rewards via time backpropagation. Used in DQN to master Atari games.

$$
Q(s, a) = \mathbb{E} [r_t + \gamma \max_{a'} Q(s_{t+1}, a')]
$$

- $Q(s, a)$: Quality value, expected return.
- $r_t$: Immediate reward, feedback at step $t$.
- $\max_{a'}$: Greedy policy, best future action.

Model-Based Trajectory Optimization: finding the action sequence minimizing distance between predicted and target states, allowing "zero-shot" planning. If you know the physics ($f_\theta$), you can solve new tasks ($s_{target}$) without re-training policies (e.g., dough manipulation in the "dumpling robot" example). (MPC (Model Predictive Control) logic: plan $T$ steps $\to$ execute 1 step $\to$ observe $\to$ re-plan.)

$$
a^*_{0:T} = \arg\min_{a_{0:T}} \sum_{t=0}^T || \hat{s}_{t} - s_{target} ||^2, \quad \text{s.t. } \hat{s}_{t+1} = f_\theta(\hat{s}_t, a_t)
$$

- $a_{0:T}$: Action sequence, trajectory to optimize.
- $s_t$: Predicted state from learned model $f_\theta$.
- $s_{target}$: Target state, desired configuration.
- $f_\theta$: Neural dynamics model, the learned simulator.

**Sim-to-Real Paradox**: For locomotion, crude simulation with massive randomization works well. For manipulation (fingers grasping objects), precise physics models are vital, and bridging the physical gap is harder.
**The Bitter Lesson**: Rich Sutton's "The Bitter Lesson"—simple methods scaling with compute (e.g., AlphaZero) often outperform complex hand-designed systems.
**Particle Models**: Representing deformable objects (dough, cloth) as particles can build highly accurate learned dynamics models, outperforming traditional physics simulators (MPM) in specific tasks.

### Imitation Learning and Robot Foundation Models: Behavior Cloning (and its cascading error problem), Diffusion Policies for multimodal action generation, and the emerging paradigm of VLA (Vision-Language-Action) foundation models like Pi-Zero

Imitation learning efficiently acquires policies from expert demonstrations but suffers from error accumulation; emerging robot foundation models leverage large-scale pre-training for broad cross-task generalization. Includes **Imitation Learning (IL)**, specifically **Behavioral Cloning (BC)** and **Inverse Reinforcement Learning (IRL)**. Also, **Robot Foundation Models** (or Vision-Language-Action models, VLA) combine LLM semantic reasoning with low-level robot control. Key concepts include **compounding errors** (small mistakes leading to drift) and **Diffusion Policies** (modeling action distribution as a denoising process).

Data-driven methods:

1. **Imitation Learning**:

- **Behavioral Cloning**: Treats policy learning as supervised classification/regression on human data (goal-to-goal).
- **Challenge**: "Cascading Errors." If the robot drifts from the expert's path, it enters unseen states, leading to failure. Solution: **DAgger** (Iterative Data Collection) (e.g., multiple choice).
- **Implicit Boundary Conditions**: Use energy-based models or **Diffusion Models** to represent multi-modal action distributions (e.g., you can go left or right around an obstacle, but the average hits the obstacle).

2. **Robot Foundation Models**: Universal robots (e.g., **RT-1**, **Pi-Zero**).

- **Methods**: Pre-train Vision-Language Models (VLM) on internet data; fine-tune on large-scale robot interaction datasets.
- **Workflow**: Pre-training (semantic knowledge); Post-training (task adaptation).
- **Evaluation Crisis**: Real-world evaluation is slow, noisy, and costly. Simulation is fast but faces the real-world gap.

Behavioral Cloning Objective: minimize the difference between robot predicted actions and actual human actions in the dataset. This is the simplest, most stable robot training method, turning robotics into a "Big Data" supervised learning problem. Used for ALOHA and many modern manipulation baselines.

$$
\theta^* = \arg\min_\theta \sum_{(s, a) \in \mathcal{D}_{expert}} \mathcal{L}(\pi_\theta(s), a)
$$

- $\pi_\theta$: Policy network, the robot's "brain."
- $\mathcal{D}_{expert}$: Demonstration dataset, human teleoperation data.
- $\mathcal{L}$: Loss function, MSE (continuous) or Cross-Entropy (discrete).

Diffusion Policy (Implicit): instead of predicting the mean of a single action, it learns the _distribution_ of valid actions. Samples noise and optimizes it into an action sequence based on state. Handles "multi-modality" (multiple valid solutions) better than MSE regression, which often outputs an average (often invalid). Acclaimed as an advanced method for dexterous tasks. (Pi-Zero workflow: VLM-based co-fine-tuning, combining VQA and action objectives.)

$$
a_t \sim p_\theta(a_t | s_t) \iff a_t = \text{Denoise}(\mathcal{N}(0, I), s_t)
$$

- $p_\theta$: Action distribution, multi-modal.
- $\text{Denoise}$: Diffusion process, iterative optimization.

**Evaluation Bottleneck**: While training is fast, evaluating universal robots is extremely costly (needing days of real testing).
**Human vs. Robot Speed**: Robots trained on teleoperation data are often slower than humans because teleoperation itself is clunky and slow.
**Semantic Generalization**: Foundation models (like Pi-Zero) generalize well semantically (e.g., knowing what a "cup" is) due to their VLM backbones but still struggle with low-level physical generalization (e.g., friction, weight).

---

## 5. Building AI for Human Vision

The evolutionary journey of vision from biological origins to computer vision's exploration of object recognition, specifically how cognitive science and large-scale data (ImageNet) eventually unlocked human-level visual understanding. Based on **Evolutionary Biology** (Cambrian Explosion theory of vision), **Neurophysiology** (EEG, neural correlates like face/place areas), and **Cognitive Science** (object parts, Biederman's conjecture). Transition from **hand-tuned models** (geometric parts) to **statistical machine learning**, and finally **Deep Learning** driven by high-dimensional data (CNNs, Transformers).

### The "Biederman Number" and the evolutionary arc of vision leading to the ImageNet/Deep Learning convergence

AI vision development spans three historical phases:

1. **Evolutionary Insight**: Vision originated ~540 million years ago, sparking an evolutionary arms race. Human vision is exceptionally fast (animal classification in 150ms), suggesting specialized "wetware" for object recognition.
2. **Early Symbolic/Geometric Vision**: 1960s-90s attempts used pre-set geometric primitives to compose objects. This failed to generalize.
3. **Statistical and Data-Driven Vision**: "Biederman's Number" (conjecture that humans recognize 30k-100k categories) inspired the **ImageNet** project. By scaling to 22k categories and 15M images, researchers reached the "Deep Learning Moment" in 2012.
4. **Beyond Objects**: **Scene Graphs** (modeling relations like "person riding horse") and **Image Captioning** (CNN + LSTM/LLM) as the next steps for storytelling and enhancing visual intelligence.

Biederman's Number (Heuristic): the number of visual categories humans can recognize, estimated for a 6-7 year old child. Humans recognize tens of thousands of nouns and visual concepts, not just a few. This number was the direct motivation for creating ImageNet, acting as a "North Star" for the scale needed for generalizable vision. It pushed AI from "small datasets" to "Big Data," a prerequisite for modern AI.

$$
N_{human} \approx [30,000, 100,000]
$$

Convergence of AI (2012 Key Moment): intelligence is an emergent property of these three elements combined at scale. Explains why algorithms existing since the 80s (e.g., neural networks) didn't work until the 2010s. These three form the foundation of the "Deep Learning Revolution" discussed throughout the course.

$$
\text{AI Breakthrough} = \text{Data} ( \text{ImageNet}) + \text{Compute} ( \text{GPU}) + 	\text{Algorithms} ( \text{CNN})
$$

- Data: Large-scale annotated datasets, e.g., 15M images.
- Compute: Parallel processing hardware, specifically NVIDIA GPUs.
- Algorithms: Deep neural network architectures like AlexNet.

**Cognitive Debt**: Computer vision often rediscovers concepts psychologists proposed decades ago (e.g., scene graphs and relation encoding).
**Visual Speed**: The 150ms benchmark for human classification highlights biological efficiency compared to current GPU-reliant processing.
**Dynamic Scenes**: Multi-object activity understanding in dynamic scenes is an unsolved critical problem vital for the future of robotics.

### Building AI to See What Humans Can't

Pushing AI beyond human capabilities (super-human classification) while addressing human cognitive limits (attention blind spots, bias) and privacy challenges in vision systems. Includes **Fine-grained Object Classification** (distinguishing 4000+ bird species or thousands of car models), **Change Blindness** (human attention limits), and **AI Bias** (algorithmic unfairness due to data skew). Also covers **Privacy-Preserving Machine Learning**, specifically **hardware-software hybrid** (custom lenses) and **Federated Learning**.

Shifting from mimicking humans to surpassing and assisting them:

1. **Super-human Performance**: AI can perform fine-grained classification (e.g., identifying car make/model/year in 100 cities) to study socioeconomic patterns (education, voting) human's can't handle manually.
2. **Human Limits**: Examples like the **Stroop Test** and **Change Blindness** show human attention is a bottleneck. In high-risk settings like surgery, this leads to medical errors (e.g., leaving gauze in patients). AI can "see" what tired or distracted humans miss.
3. **Physics of Bias**: Visual illusions (e.g., checkerboard shadow) show our brains have hard-wired physical biases. AI inherits these from data, leading to unfairness in face recognition and medical diagnosis.
4. **Privacy**: A unique "invisible" challenge. A "privacy-preserving lens" physically blocks faces and identifiable features while allowing systems to recognize actions (e.g., a patient falling).

Action Intensity Filtering (Privacy Concept): the lens physically disturbs light, losing high-frequency identity info but preserving low-frequency motion/action info. Unlike software blurring (which can be hacked if raw data exists), hardware filtering never captures private data in the first place. This solves "contextualization" of ambient intelligence in sensitive places like homes or hospitals. (Smart camera workflow for OR counting: detection $\to$ counting $\to$ alert.)

$$
I_{out} = \mathcal{F}_{opt}(x, y, \theta)
$$

- $I_{out}$: Filtered image output, blocking face recognition.
- $\mathcal{F}_{opt}$: Optical-software hybrid function, hardware lens + deblurring.
- $\theta$: Privacy constraints, level of identifiable detail removed.

**Societal Perspective**: Using car make/model detection as a proxy for voting and income patterns is a powerful example of "Computational Social Science."
**Evolutionary Bias**: Fairness in AI is an active technical commitment, not a passive byproduct, as data naturally reflects human evolutionary and societal biases.
**HW/SW Co-design**: The privacy lens shows 3D vision and perception often need more than just better networks; they need better sensors designed for specific human values.

### Building AI for Human Needs

Envisioning AI as an assistant to human labor (healthcare, elder care) and building embodied AI/robotics that respect human preferences and social values. Includes **Ambient Intelligence** (using sensors to monitor health metrics), **Embodied AI** (robotics), and **Human-Computer Interaction**. Key concepts include **Action-Perception Loops**, **Real-world Generalization** (using LLMs/VLMs for robot instructions), and **BEHAVIOR Benchmarking** (measuring robot performance on complex, preference-ranked household tasks).

"Augmentation, Not Replacement":

1. **Labor Shortage**: AI can address severe shortages in elder care and nursing. **Ambient Intelligence** projects use depth sensors to monitor hand hygiene and ICU patient activities (e.g., mobilization) with precision far exceeding human auditing.
2. **Robots in the Real World**: Beyond simple "pick and place." A system using LLMs (to generate code) and VLMs (to identify objects) allows robots to follow complex instructions like "open the top drawer but avoid the vase" without prior training.
3. **Human Values in Tasks**: **BEHAVIOR** benchmarking uses human surveys to rank 1000 tasks (e.g., people want help cleaning toilets but not help buying wedding rings). This ensures AI research goals truly meet human _values_.
4. **Sci-Fi Convergence**: Closing with demonstrations of non-invasive EEG-controlled robotic arms cooking food, highlighting the future of clinical assistance for paralyzed patients.

VLM/LLM-based Robot Planning: natural language commands are translated by an LLM into Python code, which calls a VLM to find objects in the camera feed. This info is mapped to spatial "heatmaps" guiding the robot's motion planner. It breaks the "closed-world" limits of robotics—the robot can handle any object a VLM (like CLIP/LLaVA) recognizes, not just pre-programmed ones. This links **Vision-Language** foundation models with **Robot Learning** tasks. (Robot "visual programming" logic: instruction $\to$ Python logic $\to$ vision API call $\to$ motion planner.)

$$
\text{Instruction} 	\to \text{LLM}( \text{Code}) \to \text{VLM}( \text{Heatmap}) \to \text{Trajectory}
$$

- Heatmap: Probability of relevant object locations, e.g., drawer handle position.
- Trajectory: Motion path of the robotic arm, $q_{1:T}$ configurations.

**Evaluation Gap**: Current state-of-the-art robot algorithms score **zero** on top BEHAVIOR benchmark tasks without privileged info, showing a massive research gap.
**Embodied Loop**: Analogizing biological evolution (eyes guiding movement) with AI evolution (vision models now enabling autonomous robot action).
**Human Preferences**: The discovery that humans _don't_ want robots doing "meaningful" tasks (e.g., wedding rings, baby cereal) shows AI safety and alignment must include cultural and emotional considerations.
