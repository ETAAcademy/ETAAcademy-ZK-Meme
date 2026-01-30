# ETAAcademy-ZKMeme: 79. ZK Deep Learning 3

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>79. ZKDL3</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZKDL3</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Deep Learning for Computer Vision: Foundations, Architectures, and Scalable Systems

Computer vision traces its origins from the biological evolution of sight and early hierarchical theories of the visual cortex to a modern, data-driven discipline where deep learning models learn features directly from massive datasets. 

This progression relies on foundational concepts like linear classifiers, loss functions, and optimization algorithms such as Stochastic Gradient Descent and Adam, which enable neural networks to minimize error through backpropagation. 

Building on these basics, Convolutional Neural Networks (CNNs) utilize learnable filters and pooling layers to capture spatial structures efficiently, evolving into deep architectures like ResNet that solve optimization challenges via skip connections. 

For sequential data, Recurrent Neural Networks (RNNs) and LSTMs introduce memory mechanisms, though they have largely been superseded by Transformers and Vision Transformers (ViT), which leverage self-attention to model long-range dependencies and global context parallelly. 

These advanced architectures power complex visual tasks including object detection, semantic segmentation, and video understanding, which extends analysis into the temporal dimension using techniques like optical flow and 3D convolutions. Ultimately, the computational demands of these massive models require large-scale distributed training, employing intricate strategies like data, tensor, and pipeline parallelism to scale effective learning across thousands of GPUs.

## 1. Comprehensive Review of Neural Networks, Optimization, Transformers, and Advanced Visual Intelligence Tasks

### 1.1 The Evolution of Computer Vision: History, Milestones, and Modern Applications

Computer Vision (CV) is an indispensable part of Artificial Intelligence and serves as its "cornerstone." Deep Learning (DL) is a subset of machine learning built around neural networks. The "core intersection" of computer vision and deep learning is interdisciplinary (intersecting with neuroscience, psychology, robotics, law, etc.). The "evolution of vision" dates back to the Cambrian explosion 540 million years ago, triggered by the appearance of light-sensitive cells (eyes).

A milestone in neuroscience was the research by Hubel and Wiesel in 1959, which discovered that the visual cortex has a hierarchical structure (from simple edges to complex objects) and that neurons have specific "receptive fields." In 1963, Larry Roberts published his PhD thesis on shape. In 1966, MIT launched the "Summer Vision Project" (attempting to solve vision in one summer). In the 1970s, David Marr's systematic approach evolved from "primal sketches" to 3D representations.

The road to deep learning went through "AI winters" (periods of scarce funding and stalled progress) before converging into several parallel paths. The Neocognitron (1980) was a biologically inspired, hand-designed hierarchical network. Backpropagation (1986) was a milestone learning rule that enabled networks to learn from errors without manual adjustment. LeNet (1990s) achieved early success in digit recognition (postal codes/banks). The data revolution (early 21st century) realized that high-capacity models require massive data, leading to the birth of ImageNet (containing 15 million images).

The 2012 breakthrough (AlexNet) was the turning point of the modern "deep learning revolution" at the ImageNet challenge. AlexNet (a Convolutional Neural Network) reduced the error rate by nearly half. This was achieved by combining three elements:

- **Algorithms:** Backpropagation and Convolutional Neural Network (CNN) architectures.
- **Data:** Large-scale datasets like ImageNet.
- **Compute:** The explosion of GPU (NVIDIA) performance.

Modern vision tasks and applications include formalizing computer vision into specific tasks, developing and training vision models from scratch, and understanding the current state and future directions of the field:
- **Recognition:** Image classification, object detection, semantic/instance segmentation.
- **Temporal:** Video classification and activity recognition.
- **Generative/Interactive:** Style transfer, DALL-E (text-to-image), diffusion models, and 3D reconstruction.
- **Embodied AI:** Robots and agents acting in the physical world.

Human-centered AI and ethics: AI is not purely an engineering problem. Engineering issues include bias, social impact, and complexity. Data reflects human history and bias, which can be encoded into algorithms. Positive applications exist in medicine and healthcare, while high-risk applications exist in surveillance or automated financial decisions. It is acknowledged that human vision remains more nuanced and emotional than current computer vision.

---

### 1.2 Data-Driven Classification: Linear Models, Loss Functions, and Geometric Interpretations

The core task of image classification is to assign a label to an image from a fixed set of categories, transitioning from high-level concepts to the mathematical and algorithmic foundations of computer vision. The core challenge is the semantic gap—the difference between how humans perceive images and how computers understand them:

- **Data Structure:** To a computer, an image is just a giant tensor of numbers (e.g., 800x600x3 integers between 0 and 255).
- **Variations:** Algorithms must handle changes that do not affect an object's identity, such as:
  - **Viewpoint & Illumination:** Moving the camera or changing light alters every pixel value.
  - **Deformation & Occlusion:** Cats are flexible; objects hide behind others.
  - **Intraclass Variation:** Different breeds of dogs look very different but are still "dogs."

Data-driven methods: Modern computer vision no longer uses hard-coded rules (e.g., "if edges look like this, it's a cat"). Instead, it employs a data-driven pipeline: collect a dataset of images and labels, train a machine learning classifier, and predict labels for new images. For example, the simplest classifier is the Nearest Neighbor (1-NN):

- **Logic:** To classify a test image, find the most similar single image in the training set and copy its label. For a test sample, calculate its distance to every training sample and assign the label of the closest one.
- **Distance Metrics:**
  - **L1 (Manhattan distance):** The sum of absolute differences between pixels. Sensitive to coordinate rotation. Given two images as vectors $I_1, I_2$, the L1 distance is: $d_1(I_1,I_2)=\sum_p|I^p_1−I^p_2|$
  - **L2 (Euclidean distance):** Calculates the straight-line distance between two vectors. Rotation invariant. $d_2(I_1,I_2)=\sqrt{\sum_p(I^p_1−I^p_2)^2}$. In numpy: `distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))`. In practice, the square root can be omitted as it preserves order.
- **k-Nearest Neighbors (k-NN):** Instead of one neighbor, look at the nearest $k$ neighbors and vote. This smoothes decision boundaries and ignores noise.
- **Performance:**
  - **Training:** $O(1)$ (Fast - just remembers data).
  - **Prediction:** $O(N)$ (Slow - must compare with every training sample). This is the opposite of what real-time applications need.
- **Hyperparameters:** Choices like $k$ and the distance metric are set manually using a Validation Set (never the Test Set) or Cross-Validation.

kNN has drawbacks: the classifier must store all training data, and prediction is expensive. We need a more powerful method that generalizes to neural networks. This includes two components: a score function mapping raw data to class scores, and a loss function quantifying the agreement between predicted scores and ground truth. We minimize the loss relative to parameters.

Linear Classifiers are the foundation of deep learning. To solve speed issues and build stronger models, we use Parametric Classifiers $f(x_i,W,b)=Wx_i+b$. Image classification involves a single matrix multiplication, which is much faster than comparing with all training images.

- **x:** Input image (flattened into a vector).
- **W (Weights):** Learnable parameters.
- **b (Bias):** Allows for data-independent adjustments (e.g., correcting if one class is more frequent).

Three perspectives on parametric classification: Algebraic (matrix multiplication), Visual (each row of $W$ acts as a template or "ghost"), and Geometric (linear hyperplanes separating classes in high-dimensional space).

Loss functions and the Softmax classifier are used to find the right $W$. We measure "dissatisfaction" with current scores. If we input a cat but the cat score is low while others are high, the loss is high.

Multiclass Support Vector Machine (SVM) loss "wants" the correct class score to be higher than incorrect ones by a fixed margin $\Delta$: $L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$.
- **$L_i$:** Loss for the $i$-th sample.
- **$s_{y_i}$:** Score of the correct class.
- **$s_j$:** Score of the incorrect class.
- **$\\Delta$:** Fixed margin (hyperparameter, usually 1.0).
- **Hinge Loss:** The $\max(0, -)$ term is called hinge loss due to its shape.

Softmax Classifier (Cross-Entropy Loss) interprets scores as unnormalized log probabilities: $L_i = -\log\left(\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}\right)$. The Softmax function "squashes" scores into a probability distribution (0 to 1, summing to 1). It minimizes the negative log-likelihood of the correct class. Linear classifiers are fast ($O(1)$ prediction) and serve as "Lego bricks" for neural networks, though they struggle with non-linearly separable data.

---

### 1.3 Training Dynamics: Regularization, Optimization Algorithms, and Loss Landscapes

How do we bridge the gap between model definition and training, preventing "cheating" on training data (regularization) and finding optimal parameters (optimization)?

Data loss measures fit on training data, while regularization ensures generalization to unseen data by intentionally reducing training performance to prevent overfitting. L2 Regularization (Weight Decay) penalizes squared weights ($W^2$), preferring spread-out weights. L1 Regularization penalizes absolute weights ($|W|$), creating sparse vectors. Hyperparameter $\lambda$ controls strength.

Regularization Loss (Full Objective Function): To prevent overfitting and remove weight scaling ambiguity, we add a penalty term: $L = \frac{1}{N} \sum_i L_i + \lambda R(W)$. Regularization encourages smaller, more uniform weights. For SVM with Hinge Loss: 

```math
L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + 1) \right] + \alpha R(W)
```
, where $R(W) = \sum_k \sum_l W_{k,l}^2$. 

Optimization: Exploring the "Loss Landscape" to find minimum error is like a blindfolded hiker feeling the slope to find a valley. Gradient Descent (GD) calculates the gradient (slope) and steps in the opposite direction (downhill). Stochastic Gradient Descent (SGD) uses a small random "mini-batch" (e.g., 256 images) to estimate the gradient, making it faster but noisier.

Problems with standard SGD include Jitter, local minima, and saddle points. In high-dimensional space, gradients can become zero at saddle points that aren't the true bottom. Improvements include:

- **SGD + Momentum:** Simulates a ball rolling down a hill, accumulating velocity to bypass local minima and smooth noise.
- **RMSProp:** Adjusts learning rate element-wise, slowing down in steep directions and speeding up in flat ones.
- **Adam (Adaptive Moment Estimation):** The current industry standard. It combines Momentum and RMSProp. AdamW decouples weight decay from gradient updates for better regularization.

Practical training tips: (1) Learning Rate Schedulers (Step Decay, Cosine Decay, Linear Warmup). (2) Linear Scaling Rule (increase LR proportionally with batch size). (3) Second-Order Optimization (Hessian methods like Newton's Method) are powerful but memory-intensive for deep learning. Adam with Cosine decay is a safe default starting point.

Gradients for $W$: We need the slope of the loss function. The analytical derivative definition is $\frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$. Numerical gradients approximate this with a small $h$ ($0.00001$); they are slow but easy to implement. Analytical gradients use calculus for exact formulas. For SVM loss $L_i$:
- **Correct class row ($w_{y_i}$):** $\nabla_{w_{y_i}} L_i = - \left( \sum_{j\neq y_i} \mathbb{1}(w_j^T x_i - w_{y_i}^T x_i + \Delta > 0) \right) x_i$. Subtracting this negative gradient adds $x_i$ to $w_{y_i}$, moving it closer to the image data.
- **Incorrect class row ($w_j$):** $\nabla_{w_j} L_i = \mathbb{1}(w_j^T x_i - w_{y_i}^T x_i + \Delta > 0) x_i$. Subtracting this pushes the weight $w_j$ away from $x_i$, lowering the incorrect score.
- **$\\mathbb{1}(\dots)$:** Indicator function, equals 1 if the condition is true.

---

### 1.4 Neural Architectures: Multi-Layer Perceptrons and the Backpropagation Mechanism

Neural networks combine linear and non-linear operations. Backpropagation efficiently calculates how to adjust every weight to minimize error. A two-layer network looks like: $f = W_2 \cdot \max(0, W_1x)$:

- **Input Layer:** Raw data features ($x$).
- **Hidden Layer:** Intermediate layer learning "templates."
- **Non-linearity (Activation Function):** Like ReLU ( max(0, x)). Without it, multiple layers collapse into a single linear matrix $W_2(W_1x) = W_3x$.
- **Common Activations:** ReLU (default, fast), Sigmoid/Tanh (historical, prone to vanishing gradients), Leaky ReLU/GELU (modern variants).

Backpropagation (The Learning Engine): How do we calculate gradients in a giant network? We use a Computational Graph.
- **Addition Gate:** Gradient distributor. Passes the upstream gradient equally to all inputs.
- **Multiplication Gate:** Input swapper. Gradient of $x$ depends on $y$.
- **Max Gate:** Gradient router. Sends the full gradient to the "winner" (larger input) and zero to others.

The core is the recursive application of the Chain Rule: $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \frac{\partial q}{\partial x}$.
- **Upstream Gradient:** From the next node.
- **Local Gradient:** Gradient of the current gate's output relative to its input.
- **Downstream Gradient:** Upstream $\times$ Local.

Sigmoid gate example: $\sigma(x) = \frac{1}{1 + e^{-x}}$ has a simplified derivative $\frac{d\sigma(x)}{dx} = (1 - \sigma(x))\sigma(x)$. Knowing the output makes gradient calculation efficient.

Vectorized Backpropagation: Real networks use matrices (tensors). Theoretically, this involves Jacobian matrices, but in practice, we use efficient matrix operations (swaps and transposes). For $D = W \cdot X$:
- $dW = dD \cdot X^T$
- $dX = W^T \cdot dD$

---

## 2. Convolutional, Recurrent, and Attention-Based Architectures

### 2.1 Convolutional Networks: Learning Spatial Hierarchies and Invariant Features

Feature Representations vs. End-to-End Learning: Traditional CV used hand-crafted features (Color histograms, HOG), while modern DL learns features directly from data. The key is the Convolutional Layer. Convolution uses learnable filters (templates) sliding over an image to preserve spatial structure, unlike flattening in fully connected layers. Hyperparameters include kernel size, stride, and padding. Pooling Layers (especially Max Pooling) downsample feature maps to reduce computation and increase the receptive field. CNNs are designed for translation equivariance.

"Spatial structure is paramount": Flattening a $32 \times 32 \times 3$ image into a $3072 \times 1$ vector destroys its 2D grid relationship. CNNs treat input as a 3D tensor (Width $\times$ Height $\times$ Channels) and maintain this layout throughout.

Output Dimensions: Calculated based on input size and hyperparameters: $\text{Output Size} = \frac{W - K + 2P}{S} + 1$.
- **W (Input size):** Width/Height.
- **K (Kernel size):** Filter size (e.g., 3 or 5).
- **P (Padding):** Adding zero pixels to edges to keep dimensions from shrinking.
- **S (Stride):** Pixels moved per step. Stride 2 halves the output size.

"Same" padding keeps output resolution equal to input: $P = (K - 1) / 2$ (assuming $S=1$). If $K=3$, $P=1$.

Convolution Operation ("Sliding Window"): CNNs learn filters as templates. At each position, a dot product is calculated. High output means the region matches the filter (e.g., finding edges). Depth extension: Multiple filters (e.g., 6) create a stack of 6 "feature maps."

Parameter Count: Total weights = $(F \cdot F \cdot D_{in}) \cdot K$. Total parameters = $((F \cdot F \cdot D_{in}) + 1) \cdot K$. This is much smaller than fully connected layers due to parameter sharing across spatial locations.

Receptive Fields: Deeper neurons "see" larger areas of the original image, allowing them to recognize complex objects like "faces" or "wheels" rather than just "lines."

Stacking Small Filters: Stacking three $3 \times 3$ layers has the same receptive field as one $7 \times 7$ layer but with fewer parameters ($27C^2$ vs $49C^2$) and more non-linearity (ReLU), making it more powerful and efficient.

Pooling Layer: An independent downsampling operation. Max pooling selects the maximum value in a grid (e.g., 2x2), providing invariance to small shifts and reducing data by 75%.

Practical applications: Visualizing filters (e.g., AlexNet) shows low layers learning edges/colors, middle layers learning corners/textures, and high layers learning complex object parts.

---

### 2.2 Modern CNN Design: Residual Connections, Normalization, and Training Strategies

Advanced CNN Layers: Besides Conv and Pooling, there are Normalization layers (LayerNorm, BatchNorm) to stabilize training, and Dropout for regularization. BatchNorm calculates statistics over a mini-batch per channel; LayerNorm calculates them over a sample across channels (common in Transformers). Dropout randomly "drops" (zeros) activations during training, forcing the network to learn redundant representations.

Activation Functions: Sigmoid is no longer recommended due to vanishing gradients (derivative near zero for large inputs). ReLU ( max(0, x) ) is the standard but has a "dead zone" for negative inputs. Modern alternatives include GELU and SELU (used in Transformers) which provide non-zero gradients for negative values.

Architecture Evolution: VGG proved stacking $3 \times 3$ filters is efficient. ResNet (Residual Network) solved the "deeper is not always better" paradox (optimization issues, not overfitting) by using "skip connections" ($F(x) + x$), allowing the training of extremely deep networks (e.g., 152 layers).

Successful Training "Tricks":
- **Weight Initialization:** Essential to prevent signals from vanishing or exploding. Kaiming Initialization ($\\sqrt{2/n}$) is designed for ReLU.
- **Data Strategies:** Preprocessing (normalization), Data Augmentation (crop, flip, color jitter), and Transfer Learning (using pre-trained models like ImageNet and fine-tuning only final layers).
- **Debugging:** Try to overfit a single data point (loss should go to zero). Use Random Search instead of Grid Search for hyperparameters.

---

### 2.3 Sequence Modeling: Recurrent Networks, LSTMs, and Temporal Dependencies

RNNs track information over sequences. A simple RNN can detect patterns (e.g., consecutive 1s) by using a hidden state as a "memory buffer." Weights ($W_{hh}, W_{xh}$) learn how to read and write to this memory.

Traditional RNNs (Vanilla RNNs) fail on long sequences due to vanishing gradients. Sequence modeling covers: One-to-many (Image captioning), Many-to-one (Video classification), and Many-to-many (Translation). The core is the hidden state update: $h_t = f_W(h_{t-1}, x_t)$.
- **Math:** $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$. Output generation: $y_t = W_{hy} h_t$.
- **Bottleneck:** Repeated multiplication of gradients $\le 1$ ($tanh$ derivative) causes the model to "forget" distant inputs.

Long Short-Term Memory (LSTM): Designed to solve vanishing gradients by splitting the state into Cell State ($c_t$) and Hidden State ($h_t$). It uses four gates calculated from current input $x_t$ and previous hidden state $h_{t-1}$:
- **Input gate ($i$):** $i = \sigma(W_{xi}x_t + W_{hi}h_{t-1})$. Determines how much to write to the cell state.
- **Forget gate ($f$):** $f = \sigma(W_{xf}x_t + W_{hf}h_{t-1})$. Determines how much to erase from the previous cell state.
- **Output gate ($o$):** $o = \sigma(W_{xo}x_t + W_{ho}h_{t-1})$. Determines how much of the cell state to output.
- **Gate candidate ($g$):** $g = \tanh(W_{xg}x_t + W_{hg}h_{t-1})$. New candidate information to be written.

**Update Rules:**
- **Cell State Update:** $c_t = f \odot c_{t-1} + i \odot g$ (Element-wise multiplication). The old state is filtered by the forget gate and added to the new information.
- **Hidden State Update:** $h_t = o \odot \tanh(c_t)$. The cell state is squashed by tanh and filtered by the output gate.
The Cell State acts as an "information highway" with linear interactions, allowing gradients to flow uninterrupted, similar to skip connections in ResNet.

Character-level Language Models: Predicting the next character. These models (pre-LLM era) could imitate Shakespeare or generate Linux kernel code. Specific neurons spontaneously learn concepts like "Quote Detection" or "If-Statement indentation."

RNN vs. Transformer: Transformers have $N^2$ complexity but are parallelizable. RNNs have linear complexity ($N$) and theoretical infinite context but are sequential. New research into State Space Models (Mamba) aims to combine the benefits.

---

### 2.4 The Transformer Paradigm: Attention Mechanisms, Vision Transformers (ViT), and Modern Variants

Transformers replaced RNNs. In Seq2Seq tasks (like translation), RNNs struggled with a single fixed-length context vector. Attention mechanisms allow the decoder to "look back" at all encoder states, creating dynamic context.

Generalized Attention (Queries, Keys, Values): Based on a search engine analogy.
- **Query (Q):** What you are looking for.
- **Key (K):** Labels/Indices to match against.
- **Value (V):** The actual information.
- **Formula:** $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- **Scaling:** Dividing by $\sqrt{d_k}$ prevents gradients from vanishing in high dimensions.

Variants:
- **Self-Attention:** Q, K, V all come from the same source. It's permutation equivariant (order-blind), requiring Positional Embeddings.
- **Masked Self-Attention:** Used in decoders (like GPT) to prevent "cheating" by looking at future tokens.
- **Multi-Head Attention:** Running several attention layers in parallel to capture different relationships (e.g., syntax vs. semantics).

The Transformer block includes Multi-Head Self-Attention, LayerNorm, Residual connections, and a Feed-Forward Network (MLP). Unlike RNNs, Transformers can process entire sequences in parallel during training.

Vision Transformers (ViT): Images are treated as sequences of words.
- **Patching:** Split $224 \times 224$ image into $16 \times 16$ patches.
- **Linear Projection:** Flatten and project patches into embedding space.
- **Transformer Encoder:** Process patches with self-attention.
- **Output:** Global average pooling for final classification.

Modern Transformer Improvements: Pre-Norm (for stability), RMSNorm (faster, magnitude-based), SwiGLU MLP (gated non-linearity), and Mixture of Experts (MoE - used in Llama/Gemini to scale parameters without increasing per-token compute).
- **RMSNorm:** Replaces standard LayerNorm. It rescales input based on magnitude (RMS) without centering. Formula: $RMS(x) = \sqrt{\epsilon + \frac{1}{N}\sum_{i=1}^{N} x_i^2}$, $y_i = \frac{x_i}{RMS(x)} * \gamma_i$.
- **SwiGLU MLP:** Replaces standard ReLU MLP. Uses a gating mechanism for better control of information flow. Formula: $Y = \text{Swish}(XW_1) \odot (XW_2)W_3$.

---

## 3. Advanced Applications and Large-Scale Training

### 3.1 Advanced Visual Tasks: Segmentation, Object Detection, and Model Interpretability

Although Convolutional Neural Networks (CNNs) were once the standard, Transformers have become mainstream due to better sequence modeling capabilities, despite higher computational costs. Specifically, **Vision Transformers (ViT)** are applied to core computer vision tasks, revisiting self-attention in CV, including image patches, positional embeddings, class tokens, and architectural optimizations like LayerNorm placement, SwiGLU MLP, and Mixture of Experts (MoE).

**Semantic Segmentation (Pixel-level Labeling)**
Assigns a label to every pixel (e.g., "sky", "road", "car").
- **The Patch Problem:** Classifying every pixel individually using a sliding window is computationally impossible.
- **Fully Convolutional Networks (FCN):** Designed to output spatial maps (images) instead of a single vector.
- **Encoder-Decoder Structure:**
  - **Downsampling (Encoder):** Uses standard pooling or strided convolution to reduce image size and capture context.
  - **Upsampling (Decoder):** Must recover the original resolution.
- **Upsampling Techniques:**
  - **Nearest Neighbor / Bed of Nails:** Simple, non-learnable scaling methods.
  - **Max Unpooling:** Remembers the positions of max values from the pooling step and places values back there, filling the rest with zeros.
  - **Transposed Convolution:** A "learnable" upsampling layer (often misnamed deconvolution) used to expand feature maps.
- **U-Net:** A key architecture (state-of-the-art in medical imaging) that uses **skip connections** to copy high-resolution features directly from the encoder to the decoder, preserving fine spatial details (boundaries) often lost during downsampling.

**Object Detection**
Identifies objects using bounding boxes and class labels. The goal is to predict which objects are present and where.
- **R-CNN Series (Region-based):** R-CNN proposed Region -> Crop -> Classify, which was slow. **Fast/Faster R-CNN** runs the CNN once on the whole image, then crops features. It introduced the **Region Proposal Network (RPN)** to learn object locations.
- **YOLO (You Only Look Once):** A single-stage detector. It splits the image into a grid. Each grid cell predicts bounding boxes and probabilities simultaneously. It is faster and suitable for real-time video.
- **DETR (Detection Transformer):** A Transformer-based approach. It uses object queries (learnable slots) to "ask" the encoder for objects. It removes manual components like anchor boxes or Non-Maximum Suppression (NMS).

**Instance Segmentation**
Combines detection and segmentation. Unlike semantic segmentation, it distinguishes different instances of the same class (e.g., "Dog A" vs "Dog B").
- **Mask R-CNN:** Extends Faster R-CNN by adding a third "head" (branch). While predicting class and bounding box, it also predicts a binary mask for the object within that box.

**Interpretability (Visualizing and Understanding)**
Techniques to visualize and understand what information these neural networks are actually processing, moving beyond "black box" predictions. This is vital for safety-critical fields like medicine.
- **Saliency Maps:** Calculate the gradient of the class score with respect to input pixels. This accurately highlights which pixels (e.g., edges of a dog's face) caused the model to predict "dog".
- **Class Activation Mapping (CAM):** Visualizes the weighted sum of final feature maps, but applies only to specific architectures (those ending with Global Average Pooling).
  - **Global Average Pooling (GAP):** $F_k = \frac{1}{HW}\sum_{h,w} f_{h,w,k}$
  - **Class Score:** $S_c = \sum_k w_{k,c} F_k$
  - **CAM Formula:** $M_{c,h,w} = \sum_k w_{k,c} f_{h,w,k}$. The result represents the contribution of pixel region $(h, w)$ to identifying class $c$.
- **Grad-CAM:** A generalization using gradients to weight feature maps. It can be applied to any CNN layer to generate heatmaps showing the network's "focus".
  - **Neuron Importance Weights ($\alpha_k$):** $\alpha_k = \frac{1}{HW}\sum_{h,w} \frac{\partial S_c}{\partial A_{h,w,k}}$. Averages gradients of the score relative to feature maps.
  - **Grad-CAM Heatmap:** $M_{h,w}^c = \text{ReLU}\left(\sum_k \alpha_k A_{h,w,k}\right)$. Weights feature maps by importance and applies ReLU to focus only on positive influences.
- **Attention Maps:** In Transformers (ViT), self-attention weights explicitly show how much the model focuses on different parts of the image when processing a token.

---

### 3.2 Video Understanding: Spatio-Temporal Learning, Optical Flow, and 3D Convolutions

In image classification, the core challenge is spatial features. In video classification, we add a fourth dimension: **Time (T)**.
- **Data Problem:** One minute of 1080p video can be 10GB of raw data. This is too large to feed directly into a GPU.
- **"Clip" Solution:** The model doesn't process the whole movie at once but trains on short clips (e.g., 3-5 seconds).
- **Inference Strategy:** To classify a long video, the model samples multiple clips, predicts for each, and averages the results for the final label.

**Optical Flow (Motion Measurement)**
Measures the movement of pixels between consecutive frames.
- **Definition:** A displacement field $F(x, y) = (dx, dy)$ representing pixel motion from frame $t$ to $t+1$.
- **Assumption:** **Brightness Constancy**—pixel appearance doesn't change significantly, it just moves to a new location: $I_{t+1}(x+dx, y+dy) = I_t(x, y)$.

**Recurrent Convolutional Network (Long-term Structure)**
Used to process frame sequences while maintaining spatial information.
- **Update Rule:** $h_t = \tanh(W_h * h_{t-1} + W_x * x_t)$.
- This modifies the standard RNN update rule by replacing matrix multiplication with **2D convolution ($*$)**. Unlike standard RNNs that flatten images into vectors, this preserves the $(H, W)$ structure, allowing the model to learn how spatial features evolve over time.

**Self-Attention Alignment (Spatio-temporal Relations)**
Used in Transformers and "Non-local Blocks" to relate different parts of the video regardless of spatial or temporal distance.
- **Mechanism:** Linear projections create $q, k, v$ from the input tensor ($T \times C \times H \times W$). This allows the model to "look at" a person in the first frame to help classify an action in the tenth frame, effectively capturing long-range temporal dependencies.
- **Formulas:**
  - **Alignment Score:** $e_{i,j} = \frac{q_j \cdot k_i}{\sqrt{D}}$.
  - **Attention Probabilities:** $a = \text{softmax}(e)$.
  - **Output:** $y_j = \sum_i a_{i,j} v_i$.

**Fusion Strategies (Combining Information Across Time)**
The core engineering problem is how to combine (fuse) information.
- **Single Frame (Strong Baseline):** Treat every frame as an independent image, run a standard 2D CNN, and average the results. Performance is surprisingly high (77.7% on Sports-1M) because scene context often remains consistent.
- **Late Fusion:** Run two independent 2D CNNs on frames far apart, then merge their features at the very end (fully connected layer). **Drawback:** It ignores local motion (e.g., foot moving up and down) until late in the network, losing fine temporal details.
- **Early Fusion:** Stack frames together immediately at the input layer. **Drawback:** It immediately destroys temporal structure. It cannot learn that an action at second 1 is the same as one at second 5.
- **Slow Fusion (3D CNN):** The "Goldilocks" approach. Uses 3D convolutions (Time x Height x Width). Filters slide over space and time. **Key Advantage: Temporal Translation Invariance.** Just like a 2D CNN recognizes a cat in the top-left or bottom-right, a 3D CNN recognizes a "jump" whether it happens at the start or end of the clip.

**3D Convolutions**
While 2D CNNs apply filters to H and W, 3D CNNs apply filters to **T, H, and W**.
- **Filter Shape:** $C_{out} \times C_{in} \times K_t \times K_h \times K_w$. $K_t$ is the temporal kernel size (usually 3).
- **Complexity:** Data flows through a 3D CNN as a $3 \times T \times H \times W$ tensor. Computational cost (GFLOPs) scales with $T \times K_t$, making 3D CNNs much more expensive than 2D CNNs (e.g., C3D is 2.9x heavier than VGG-16).

**Two-Stream Networks (Appearance vs. Motion)**
Based on biological insight that humans can recognize actions just by seeing moving dots without color/texture.
- **Stream 1 (Spatial):** Standard RGB frames. Answers "What objects are in the scene?"
- **Stream 2 (Temporal):** Uses Optical Flow. Answers "How are things moving?"
- The motion stream often outperforms the spatial stream because it's invariant to background clutter.

**I3D (Inflated 3D Conv)**
Provides a formula to convert pre-trained 2D models (ImageNet) to 3D without losing learned spatial knowledge.
- **Inflation Rule:** Initialize 3D kernels from 2D kernels: $W_{3D}(t, h, w) = \frac{W_{2D}(h, w)}{K_t}$.
- Replicate the 2D weights $K_t$ times along the time dimension and divide by $K_t$. This ensures that if the input video is a "static image", the 3D output matches the 2D output, giving the model a head start.

**Long-term Modeling & Multimodality**
- **Long-term:** 3D CNNs handle short actions. To understand a whole movie (long-term context), we use RNNs/LSTMs on top of extracted clip features, or Transformers (Non-local blocks).
- **Multimodality:**
  - **Audio-Visual:** Using video to separate specific sounds (e.g., separating a violin track by watching the bow move).
  - **Egocentric Vision (AR/VR):** Analyzing video from smart glasses to understand social interactions.

---

### 3.3 Distributed Deep Learning: Parallelism Strategies and Hardware Optimization at Scale

Transitioning to training on tens of thousands of GPUs.
- **Hardware:** NVIDIA H100 is the workhorse. **Tensor Cores** are specialized units for matrix multiplication, offering 1000x throughput increase since 2013.
- **Hierarchy:** Bandwidth drops with distance. Chip (3TB/s) -> Server (NVLink 900GB/s) -> Rack (50GB/s) -> Cluster (<50GB/s). Software must mimic this hierarchy.

**The "Five Degrees" of Parallelism**
Since a single GPU cannot hold the model or data, we split the workload along four dimensions of the Transformer tensor: Batch, Time (Sequence), Layer, and Model Width (Channel).

1.  **Data Parallel (DP) & FSDP:**
    -   **DP:** Replicates the model on all GPUs, splitting data. **Problem:** Model weights + optimizer states for 100B+ params exceed H100 memory (80GB).
    -   **FSDP (Fully Sharded Data Parallel):** Instead of replicating, it **shards** (slices) parameters, gradients, and optimizer states across all GPUs. **Dynamic Communication:** When a GPU needs a specific shard for computation, it requests it, uses it, and deletes it. Trades bandwidth for memory.
    -   **Hybrid Sharded Data Parallel (HSDP):** "Best of both worlds". Uses FSDP within a node (fast NVLink) and standard DP between nodes (slower Ethernet/Infiniband), reducing heavy communication across racks.
2.  **Tensor Parallel (TP):**
    -   Splits a specific matrix multiplication $X \times W = Y$. $W$ is split into columns. GPU 1 computes half the output vector, GPU 2 the other half. Requires extremely high bandwidth (NVLink) because synchronization is needed after every layer.
3.  **Pipeline Parallel (PP):**
    -   Layer 1 on GPU 1, Layer 2 on GPU 2. **Bubble Problem:** GPU 2 sits idle while GPU 1 works. **Solution:** Micro-batches. Send 4 small batches quickly so GPU 2 starts working on micro-batch 1 while GPU 1 prepares micro-batch 2.
4.  **Context Parallel (CP):**
    -   Necessary for long contexts (100k+ tokens). The sequence itself is too long for one GPU's memory. The sequence is split. **Challenge:** Attention is $N^2$ (every token looks at every other). Techniques like "Ring Attention" pass blocks of Key/Value pairs around so every GPU eventually sees every token.

**Memory Optimization: Activation Checkpointing**
A trick for models larger than memory.
-   Instead of storing all intermediate activations for backprop (which consumes TBs), delete them after the forward pass and **recompute** them from checkpoints during the backward pass.
-   Increases compute by ~33% but reduces memory cost from $O(N)$ to $O(\sqrt{N})$.

**MFU (Model FLOPs Utilization)**
The "North Star" metric. Real throughput vs Peak Hardware throughput.
-   **< 30%:** Something is wrong (bottlenecks).
-   **~40%:** Excellent (Llama 3 achieved ~38-40%).
-   **> 50%:** Extremely rare on modern hardware due to the "Memory Wall" (compute grows faster than bandwidth).
