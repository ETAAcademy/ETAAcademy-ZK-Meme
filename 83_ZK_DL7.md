# ETAAcademy-ZKMeme: 83. ZK Deep Learning 7

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>83. ZKDL7</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZKDL7</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# The Multimodal Leap: From Textual Understanding to Synthesizing Human Speech

Modern AI is not a single technique but a layered stack of ideas. Some models are optimized to understand language in context, others are trained to behave like assistants, others retrieve external knowledge, and still others convert speech into text or generate speech from text. Taken together, these systems explain why modern AI feels far more capable than the earlier generations of NLP systems that relied on static word vectors, handcrafted rules, or narrow pipelines.

What makes current systems powerful is not just scale, but composition. A modern language system often relies on one set of mechanisms to build contextual representations of meaning, another to shape behavior around instructions and human preferences, another to compensate for the limits of internal memory by retrieving outside evidence, and another to interface with the real world through speech. In other words, modern AI works because different components solve different failure modes: shallow understanding, poor task following, factual unreliability, and difficulty operating across modalities.

Seen this way, these technologies form a single progression rather than a disconnected survey. Bidirectional encoders and masked language models explain how systems learn richer representations of words and sentences by looking at context from both directions. Post-training explains how raw generative models are turned into assistants through instruction tuning, preference learning, alignment methods such as DPO, and inference strategies that increase test-time reasoning capacity. Retrieval models explain how search evolved from sparse lexical matching to dense semantic retrieval and finally to RAG, where language generation is grounded in retrieved documents instead of depending only on what the model memorized during training.

The speech sections extend the same logic into another modality. Phonetics and acoustic feature extraction show how continuous sound is converted into structured signals a model can learn from. ASR shows how those signals become text through alignment, sequence modeling, and self-supervised representation learning. TTS reverses the pipeline, showing how text can be transformed back into natural-sounding speech through audio tokenization, vector quantization, and conditional generation. Across all of these technologies, the common pattern is the same: turn raw input into useful internal structure, optimize that structure for the task at hand, and connect it to outputs that are reliable, grounded, and usable by humans.

This is full stack from bidirectional language understanding to post-trained large language models, retrieval systems, and speech technologies. The goal is to show how each stage solves a specific bottleneck in language intelligence and how those stages fit together into a broader architecture of modern AI.

---

## 1. Bidirectional Language Understanding and the Rise of Masked Language Models

Early neural language models usually learned in a left-to-right fashion: predict the next token from the previous ones. That design works well for generation, but it is not ideal for understanding. Many interpretation tasks, such as classification, question answering, or named entity recognition, depend on seeing the full sentence at once. The word "bank" means something different in "river bank" than in "bank loan," and in many cases the right-hand context matters as much as the left.

This is the key idea behind bidirectional transformer encoders such as BERT. A causal language model uses a mask inside self-attention so that each token can only attend to earlier tokens:

$$
\text{head} = \text{softmax}\left(\text{mask}\left(\frac{QK^T}{\sqrt{d_k}}\right)\right)V
$$

Bidirectional encoders remove that causal mask. Every token can attend to every other token in the sequence:

$$
\text{head} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

That simple architectural change has a large consequence: token representations become contextualized. A word is no longer assigned one fixed embedding for all uses; instead, the model produces a meaning representation shaped by the entire sentence.

### Why masked language modeling works

Once a model can see the whole sentence, ordinary next-token prediction becomes too easy. The workaround is masked language modeling (MLM). During pretraining, a subset of tokens is hidden, randomized, or left unchanged, and the encoder must reconstruct the original words from context.

If $h_i^L$ is the final hidden state for position $i$ and $E$ is the shared embedding matrix, the model projects the contextual vector back into vocabulary space:

$$
u_i = h_i^L E^T
$$

$$
y_i = \text{softmax}(u_i)
$$

The MLM objective is then the average negative log-likelihood over the masked positions $M$:

$$
L_{MLM} = -\frac{1}{|M|} \sum_{i \in M} \log P(x_i \mid h_i^L)
$$

This training setup is less token-efficient than causal training because only a fraction of positions contribute direct supervision, but it forces the network to learn syntax, semantics, and long-range dependencies in a deep way. It also established the now-standard pretrain-then-finetune workflow.

Multilingual training adds another complication. If data sampling follows raw corpus size, high-resource languages dominate learning. A common fix is smoothed language sampling:

$$
q_i = \frac{p_i^\alpha}{\sum_{j=1}^N p_j^\alpha},
\qquad
p_i = \frac{n_i}{\sum_{k=1}^N n_k}
$$

With a smoothing exponent such as $\alpha = 0.3$, low-resource languages are upweighted enough to remain learnable.

### Contextual embeddings changed what a word vector means

Static embeddings such as Word2Vec assign one vector to each word type. Contextual encoders instead assign a vector to each word occurrence. That makes them naturally suited to polysemy and word sense disambiguation.

For a sense $s$, a prototype vector can be defined by averaging the contextual embeddings associated with that sense:

$$
v_s = \frac{1}{n} \sum_i v_i
$$

A new token embedding $t$ can then be matched to the closest sense by cosine similarity:

$$
\text{sense}(t) = \arg\max_{s \in \text{senses}(t)} \text{cosine}(t, v_s)
$$

In practice, contextual embeddings are often anisotropic: they cluster in a few dominant directions, which can distort similarity scores. A common normalization step is z-scoring across the corpus:

$$
\mu = \frac{1}{|C|} \sum_{x \in C} x
$$

$$
\sigma = \sqrt{\frac{1}{|C|} \sum_{x \in C} (x - \mu)^2}
$$

$$
z = \frac{x - \mu}{\sigma}
$$

This makes distances more semantically meaningful.

### Fine-tuning made encoders broadly useful

The reason BERT-style encoders became so influential is that they adapt cleanly to many downstream tasks. For sequence classification, the special `[CLS]` token acts as a sentence-level summary:

$$
y = \text{softmax}(h_{CLS}^L W_C)
$$

For token labeling tasks such as named entity recognition, each token representation gets its own classifier:

$$
y_i = \text{softmax}(h_i^L W_K),
\qquad
t_i = \arg\max_k y_i
$$

Subword tokenization introduces a practical wrinkle. A word may split into several pieces, but the label usually belongs to the original word. A common solution is to apply the gold label to the first subword, mask loss on later subwords, and use the first subword prediction as the word-level output.

A practical challenge in NER is subword misalignment. Modern tokenizers often split single words into multiple subwords, complicating the one-to-one word-to-tag mapping. Here is a common logic flow for handling this during training and decoding:

<details><summary>Code</summary>

```text
Input word: "Sanitas"
Gold Tag: I-LOC
Subwords: ["San", "##itas"]

Training Mapping:
  "San"    -> I-LOC
  "##itas" -> I-LOC

Decoding:
  Tag("San") -> Use as the official word tag
  Tag("##itas") -> Ignore
```

</details>

By predicting the label on the first subword and masking out the loss for subsequent subwords, the model effectively learns to perform entity-level segmentation and categorization.

In short, masked language models gave NLP a strong language-understanding backbone. But they were not designed to behave like conversational assistants. That required a second phase of work.

## 2. From Base Models to Assistants: The Logic of Post-Training

A pretrained causal language model is good at continuation, not necessarily at compliance. Prompt it with a question and it may continue the text rather than answer it. Prompt it with something unsafe and it may produce harmful output. Modern chat models solve this with post-training: a sequence of steps that teach the model to follow instructions, prefer better answers, and spend inference compute more effectively.

### Instruction tuning teaches the model to respond

The first post-training stage is supervised fine-tuning, often called instruction tuning. The data now consists of instruction-response pairs, but the optimization target remains the same next-token objective:

$L(\theta) = -\sum_{i=1}^n \log P(x_i \mid x_{<i}; \theta)$

The difference is not in the math but in the data distribution. The model is no longer trained to continue arbitrary web text; it is trained to produce the kinds of responses humans want when they ask for translations, summaries, code, or explanations.

This step acts like meta-learning. Rather than memorizing individual tasks, the model learns the general pattern of instruction following.

### Preference learning turns comparisons into a training signal

Supervised fine-tuning improves compliance, but not necessarily judgment. Human annotators often find it easier to say which of two answers is better than to write the ideal answer themselves. That insight motivates preference learning.

The Bradley-Terry model converts pairwise preferences into a probability:

$$
P(o_w \succ o_l \mid x) = \sigma(z_w - z_l) = \frac{1}{1 + e^{-(z_w - z_l)}}
$$

Here, $o_w$ is the preferred response, $o_l$ is the rejected response, and $z$ is a latent score. A reward model $r(x,o)$ can be trained with binary cross-entropy:

```math
L_{CE}(x, o_w, o_l)
= -E_{(x, o_w, o_l) \sim D}
\left[\log \sigma\left(r(x, o_w) - r(x, o_l)\right)\right]
```

The reward model learns to assign higher scores to outputs people judge as more helpful, safer, or more accurate.

### DPO simplified alignment

Historically, reward models were paired with reinforcement learning methods such as PPO. The objective combined reward maximization with a KL penalty so the model would not drift too far from a reference policy:

$$
\pi^* = \arg\max_{\pi_\theta}
\mathbb{E}_{x \sim D, o \sim \pi_\theta(o \mid x)}
\left[
r(x,o) - \beta D_{KL}\left(\pi_\theta(o \mid x)\, \| \,\pi_{ref}(o \mid x)\right)
\right]
$$

This works, but it is difficult to optimize and expensive to run. Direct Preference Optimization (DPO) showed that the same alignment target can be expressed directly in terms of policy probabilities:

$$
r(x,o) = \beta \log \frac{\pi^*(o \mid x)}{\pi_{ref}(o \mid x)} + \beta \log Z(x)
$$

That leads to a simpler loss:

```math
\mathcal{L}_{DPO}(\pi_\theta)
= -\mathbb{E}_{(x, o_w, o_l) \sim D}
\left[
\log \sigma
\left(
\beta \log \frac{\pi_\theta(o_w \mid x)}{\pi_{ref}(o_w \mid x)}
- \beta \log \frac{\pi_\theta(o_l \mid x)}{\pi_{ref}(o_l \mid x)}
\right)
\right]
```

The result is a much more practical alignment recipe: no separate RL rollout loop is required, and the optimization looks much closer to supervised learning.

### Test-time compute improves reasoning

Even after alignment, hard reasoning tasks remain challenging if the model is expected to map question to answer in one short burst. Test-time compute addresses that limit by letting the model generate intermediate steps before committing to a final answer.

Chain-of-thought prompting is the most familiar example. The core intuition is simple: more useful intermediate tokens mean more computation devoted to the problem. In effect, the model outsources parts of its reasoning process into the context window itself.

Post-training, then, is what turns a raw language model into something closer to an assistant: it learns to follow instructions, reflect human preferences, and use inference-time computation more effectively.

## 3. Retrieval: From Sparse Matching to Retrieval-Augmented Generation

Language models are powerful, but they are still closed-book systems unless we connect them to external knowledge. That is the role of information retrieval. Search systems evolved from sparse keyword matching to dense semantic retrieval, and modern AI assistants increasingly combine retrieval with generation in one pipeline.

### Sparse retrieval began with tf-idf and cosine similarity

The classical vector space model represents each document as a high-dimensional sparse vector over vocabulary terms. The simplest improvement over raw word counts is tf-idf. Term frequency captures local importance, while inverse document frequency downweights globally common words.

Using a log-scaled term frequency:

$$
tf_{t,d} =
\begin{cases}
1 + \log_{10}\text{count}(t,d) & \text{if } \text{count}(t,d) > 0 \\
0 & \text{otherwise}
\end{cases}
$$

and inverse document frequency:

$$
idf_t = \log_{10}\frac{N}{df_t}
$$

we obtain the familiar product:

$$
\text{tf-idf}(t,d) = tf_{t,d} \cdot idf_t
$$

Similarity between query vector $q$ and document vector $d$ is often measured with cosine similarity:

$$
\text{score}(q,d) = \cos(q,d) = \frac{q \cdot d}{|q||d|}
$$

Cosine normalization matters because it prevents long documents from dominating purely by length.

BM25 improved on tf-idf by handling document length more explicitly and saturating the effect of repeated term matches:

$$
\text{score}(q,d) =
\sum_{t \in q}
\text{IDF} \cdot
\frac{tf_{t,d}}
{k\left(1 - b + b\frac{|d|}{|d_{avg}|}\right) + tf_{t,d}}
$$

These methods became practical at scale because of the inverted index, which maps each term to a postings list of documents that contain it.

### Dense retrieval fixed vocabulary mismatch

Sparse retrieval struggles when the query and the document use different words for the same idea. A search for "automobile" may miss a document about "cars." Dense embeddings address that mismatch by mapping semantically related text into nearby vectors.

Three broad architectures dominate modern neural retrieval:

- Cross-encoders encode query and document together, yielding the best accuracy but making large-scale retrieval too slow.
- Bi-encoders encode queries and documents independently, enabling vector search over precomputed document embeddings.
- Late-interaction models such as ColBERT preserve token-level detail while remaining much cheaper than cross-encoders.

For ColBERT-style scoring, each query token is matched against its most similar document token:

$$
\text{score}(q,d) = \sum_{i=1}^{N} \max_{j=1}^{m} E_{q_i} \cdot E_{d_j}
$$

This retains much of the precision of token interaction without requiring full joint encoding at query time.

### Retrieval-augmented generation grounded LLMs

Retrieval-Augmented Generation (RAG) combines search with a language model. Instead of asking the model to answer from internal memory alone, the system retrieves supporting passages and injects them into the prompt:

$p(x_1, \ldots, x_n) = \prod_{i=1}^{n} p(x_i \mid R(q); \text{instruction}; q; x_{<i})$

Now the next-token distribution depends explicitly on retrieved evidence $R(q)$. This is the key reason RAG reduces hallucination risk and overcomes knowledge cutoffs: the model is no longer forced to rely solely on what it stored during pretraining.

Before the era of open-domain text retrieval, early question-answering systems operated on highly structured, closed-domain databases. For instance, the **BASEBALL** system (1961) used parsed natural language to query a structured attribute-value matrix:

<details><summary>Code</summary>

```python
Month = July
Place = Boston
Day = 7
Game Serial No. = 96
(Team = Red Sox, Score = 5)
(Team = Yankees, Score = 3)
```

</details>

By the 1970s, systems like **LUNAR** employed semantic parsing to translate natural language questions into formal logical queries (similar to Prolog or LISP) to search geological databases:

<details><summary>Code</summary>

```lisp
(TEST (FOR SOME X16 / (SEQ SAMPLES) : T ; (CONTAIN' X16
(NPR* X17 / (QUOTE AL203)) (GREATERTHAN 13 PCT))))
```

</details>

The roots of question answering go back much earlier. Systems such as BASEBALL and LUNAR answered questions by mapping language into symbolic queries over tightly structured databases. They were precise, but brittle. Modern retrieval systems trade exact symbolic control for flexibility over huge bodies of unstructured text. RAG is the latest step in that trajectory: retrieve first, then synthesize.

## 4. The Evolution of Conversational AI: Bridging Speech Recognition and Synthesis

Driven by advanced neural architectures, modern speech artificial intelligence has revolutionized bidirectional human-machine communication through critical breakthroughs in both Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) synthesis. While ASR models increasingly rely on mechanisms like Convolutional Neural Networks, Attention-based architectures, and self-supervised frameworks (such as HuBERT) to efficiently translate complex, continuous acoustic phonetics into text without relying on massive labeled datasets, modern TTS systems perform the exact inverse. By utilizing neural audio codecs and zero-shot modeling techniques, cutting-edge TTS architectures (like VALL-E) can synthesize highly realistic, dynamic human voices from extremely brief audio prompts, treating audio segments like discrete language tokens to achieve seamless, real-time conversational AI.

## 4.1 Speech as Signal: From Phonetics to Feature Extraction

If NLP taught machines to read text, speech technology taught them to hear and speak. The challenge is different because speech is continuous, time-varying, and physically grounded in human articulation.

### Phonetics explains the units of speech

The smallest practical sound segments are phones. To represent them consistently, linguistics uses transcription systems such as the International Phonetic Alphabet (IPA), while speech engineering often uses the ASCII-based ARPAbet for English.

Speech is produced by the interaction of the lungs, larynx, tongue, lips, and vocal tract. Consonants are categorized by place and manner of articulation. Vowels are characterized by tongue height, frontness or backness, and lip rounding. These units are organized into syllables, and languages constrain valid sound patterns through phonotactics.

Above the phone level lies prosody: stress, rhythm, pitch accent, and intonation. These suprasegmental features carry emphasis, emotion, discourse structure, and sentence type. Frameworks such as ToBI convert continuous pitch movement into symbolic annotations, which are especially important for natural text-to-speech generation.

### Acoustic phonetics turns sound into measurable structure

A speech signal is a pressure wave, and any complex wave can be decomposed into sinusoidal components:

$$
y = A \sin(2\pi f t)
$$

To process sound digitally, we sample it into discrete values. The sampling rate must be at least twice the highest frequency of interest, the Nyquist criterion. Once sampled, the signal can be analyzed in the frequency domain with the Discrete Fourier Transform.

Spectrograms visualize how frequency energy changes over time. They expose the source-filter structure of speech: the vibrating vocal folds provide the source, while the vocal tract shapes the resonant frequencies called formants. Vowel identity, in particular, depends heavily on the pattern of formants such as $F_1$ and $F_2$.

Because human hearing is more logarithmic than linear, speech systems often use perceptually motivated frequency scales. The mel scale is a standard example:

$$
m = 1127 \ln\left(1 + \frac{f}{700}\right)
$$

### Feature extraction compresses speech into learnable representations

Raw waveforms are too detailed and unstable for most classical pipelines, so speech is split into short overlapping frames, typically around 25 ms with a 10 ms stride. Each frame is windowed, often with a Hamming window, to reduce edge artifacts. The Fast Fourier Transform is then applied to estimate frame-level frequency content.

The next step is to aggregate energy through a mel filter bank and take the logarithm, yielding the log-mel spectrum. This representation is compact, robust to amplitude scaling, and much closer to how modern neural ASR systems consume audio.

Before log-mel features became dominant in neural pipelines, the standard representation was MFCCs, or Mel Frequency Cepstral Coefficients. MFCCs are derived from the cepstrum:

$$
c[n] = \text{IDFT}\{\log |\text{DFT}\{x[n]\}|\}
$$

The cepstral view helps separate the slowly varying vocal tract envelope from the rapidly oscillating excitation source. In practice, a small number of cepstral coefficients, often supplemented with first- and second-order derivatives, became the classic feature vector for speech recognition.

## 4.2 Automatic Speech Recognition: Turning Audio into Text

Automatic Speech Recognition (ASR) maps a time-varying acoustic signal into a discrete token sequence. That sounds straightforward, but it involves several intertwined challenges: speech is long, noisy, variable across speakers, and only loosely aligned with the words it expresses.

### ASR must solve variability before it can solve language

ASR difficulty depends on vocabulary size, acoustic environment, speaking style, microphone quality, accent, dialect, and spontaneity. Read speech is easier than conversational speech. Close-talk microphones are easier than far-field audio. Standard datasets such as LibriSpeech, Switchboard, and Common Voice exist precisely because these factors matter so much.

### CNN front ends compress acoustic sequences

Many ASR systems begin with convolutional layers that transform raw audio or spectral features into more compact latent representations. A width-1 convolution acts as a frame-wise linear transform:

$$
z_j = x_j w_0
$$

A wider 1D convolution with padding $p$ aggregates local context:

$$
z_j = \sum_{i=-p}^{p} x_{j+i} w_{i+p}
$$

With multiple input channels, the outputs are summed across channels:

$$
z = \sum_{c=1}^{C_i} x^c * w^c
$$

These front ends reduce sequence length and expose local acoustic patterns before the model hands control to a transformer or recurrent backbone.

### Encoder-decoder ASR bridges the gap between frames and words

The central mismatch in ASR is temporal scale: acoustic frames arrive every few milliseconds, while words unfold much more slowly. Attention-based encoder-decoder (AED) models handle this by compressing the acoustic sequence and letting a text decoder attend to it.

If the encoder outputs $H^{enc}$ and the decoder hidden states are $H^{dec[\ell-1]}$, cross-attention takes the form:

$$
Q = H^{dec[\ell-1]}W^Q,
\qquad
K = H^{enc}W^K,
\qquad
V = H^{enc}W^V
$$

$$
\text{CrossAttention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Training usually uses the standard sequence cross-entropy objective:

$$
L_{CE} = - \sum_{i=1}^{m} \log p(y_i \mid y_1, \dots, y_{i-1}, X)
$$

At decoding time, beam search can be improved by rescoring with an external language model:

$$
\text{score}(Y \mid X) =
\frac{1}{|Y|_c}\log P(Y \mid X) + \lambda \log P_{LM}(Y)
$$

### Self-supervised learning reduced the dependence on labeled audio

Large labeled speech corpora are expensive. Self-supervised learning methods such as HuBERT address this by learning from unlabeled audio first. HuBERT clusters acoustic frames with k-means:

$$
\text{cluster}(i) = \arg\min_{1 \leq j \leq k} ||v^{(i)} - \mu_j||^2
$$

$$
\mu_i = \frac{1}{|S_i|} \sum_{v \in S_i} v
$$

The model then masks portions of the audio and predicts their cluster assignments:

$$
p(c \mid X,t) =
\frac{\exp(\text{sim}(Ah_t, e_c)/\tau)}
{\sum_{c'=1}^{C} \exp(\text{sim}(Ah_t, e_{c'})/\tau)}
$$

After pretraining, the resulting representations can be fine-tuned for downstream ASR with far less labeled data.

### CTC remains essential for alignment and streaming

Connectionist Temporal Classification (CTC) offers another way to model speech without explicit frame-level alignments. It predicts a label distribution at every frame:

$$
P_{CTC}(A \mid X) = \prod_{t=1}^{T} p(a_t \mid X)
$$

Because many frame-level paths collapse to the same text sequence, CTC sums over all valid alignments:

$$
P_{CTC}(Y \mid X)
= \sum_{A \in B^{-1}(Y)} P(A \mid X)
= \sum_{A \in B^{-1}(Y)} \prod_{t=1}^{T} p(a_t \mid h_t)
$$

Its conditional independence assumption is limiting, but CTC is especially useful for alignment learning and streaming recognition. Many modern systems combine CTC and encoder-decoder objectives:

```math
L = -\lambda \log P_{encdec}(Y \mid X)
- (1-\lambda)\log P_{CTC}(Y \mid X)
```

### Evaluation is dominated by word error rate

ASR is usually measured with Word Error Rate (WER), derived from edit distance:

$$
\text{WER} =
100 \times
\frac{\text{Insertions} + \text{Substitutions} + \text{Deletions}}
{\text{Total Words in Correct Transcript}}
$$

WER is simple but unforgiving. Insertions can push it beyond 100%, and fair evaluation depends heavily on normalization choices such as punctuation, casing, numerals, and abbreviations. When researchers compare systems, they often use matched-pair tests such as MAPSSWE rather than assuming word errors are independent.

## 4.3 Text-to-Speech: Turning Text Back into Voice

If ASR is about transcription, text-to-speech (TTS) is about synthesis. Classical TTS systems often required large single-speaker corpora and careful engineering. Modern neural TTS systems have moved toward zero-shot generation: given a short prompt from a new speaker, they can synthesize novel speech in that voice.

### Audio tokenizers make language modeling over sound possible

Transformers are poorly suited to raw high-rate waveforms because attention cost grows quadratically with sequence length. Neural audio codecs solve this by compressing speech into a much slower latent sequence.

Models such as ENCODEC use convolutional encoders to downsample the waveform into a compact representation and decoders to reconstruct it. This turns audio into something more like a discrete sequence modeling problem.

### Vector quantization discretizes the acoustic space

Given a continuous vector $v$, vector quantization maps it to the nearest codeword:

$$
\arg\min_i ||v - c_i||^2
$$

Because a single codebook may be too coarse, residual vector quantization (RVQ) quantizes the leftover error after the first assignment:

$$
\text{residual} = z_t - z_{q_t}^{(1)}
$$

Repeating that process across multiple codebooks yields a hierarchical acoustic code that is both compact and expressive.

### Codec training balances reconstruction and perceptual quality

Audio tokenizers are trained as autoencoders with a combination of losses. A basic reconstruction loss is:

$$
L_{\text{reconstruction}}(x, \hat{x}) = \sum_{t=1}^{T} ||x_t - \hat{x}_t||^2
$$

The VQ component encourages the codebooks to track the encoder output:

$$
L_{\text{VQ}}(x, \hat{x}) =
\sum_{t=1}^{T}\sum_{c=1}^{N_c}
||z_t^{(c)} - z_{q_t}^{(c)}||
$$

Together with an adversarial term for perceptual realism, the full objective becomes:

```math
L(x, \hat{x}) =
\lambda_1 L_{\text{reconstruction}}(x, \hat{x})
+ \lambda_2 L_{\text{GAN}}
+ \lambda_3 L_{\text{VQ}}(x, \hat{x})
```

### VALL-E showed how speech generation could be framed as language modeling

VALL-E treats discrete acoustic codes as the target sequence of a conditional language model. If ENCODEC produces a code matrix $C_{T \times 8}$, generation can be split into a two-stage process: an autoregressive model predicts the first code level, and non-autoregressive modules fill in the remaining residual levels.

The decoding objective mirrors standard language modeling:

$C_T = \arg\max_{C_T} \prod_{t=T_0+1}^{T} p(c_{t,:} \mid c_{<t,:}, x)$

Conditioning on text, a short voice prompt, and previously generated acoustic codes allows the system to synthesize speech in an unseen voice with remarkable fidelity.

### Evaluation remains partly subjective

Unlike ASR, TTS cannot be judged by one metric alone. Mean Opinion Score (MOS) and comparison-based protocols such as CMOS remain standard because naturalness and speaker similarity are fundamentally perceptual qualities. Proxy metrics such as using ASR to transcribe generated speech can help, but they do not replace listening tests.

The same modeling tools used for TTS also support adjacent tasks such as speaker verification, diarization, speaker identification, and wake-word detection. In other words, modern speech synthesis is part of a larger machine listening ecosystem rather than a standalone application.

---

Modern AI systems became powerful by stacking solutions to different problems rather than by solving everything with one model. Bidirectional encoders gave NLP deep contextual understanding. Post-training turned raw generators into instruction-following assistants. Retrieval systems grounded answers in external evidence. Speech pipelines translated between sound and symbols in both directions.

Seen together, these technologies form a coherent progression. First, machines learned to represent language well. Then they learned to answer usefully. Then they learned to look things up. Finally, they learned to listen and speak. That layered progression is what defines the current era of language AI.
