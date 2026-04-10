# ETAAcademy-ZKMeme: 84. ZK Deep Learning 8

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>84. ZKDL8</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZKDL8</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# From Sequence Labeling to Dialogue Acts: How Machines Learn Structure, Meaning, and Context

Natural language processing becomes much easier to navigate when its core tasks are arranged by the kind of structure they recover. At the word level, sequence labeling assigns POS tags and named-entity labels, forcing models to deal with lexical ambiguity, BIO span encoding, and the contrast between generative Hidden Markov Models and discriminative Conditional Random Fields. At the sentence level, syntactic parsing splits into two major traditions: constituency parsing, built on context-free grammars, treebanks, CKY dynamic programming, and neural span scorers; and dependency parsing, built on head-dependent graphs, shift-reduce transition systems, maximum spanning tree decoding, and biaffine edge scoring.

Once syntax is in place, it moves into meaning. Information extraction turns text into structured records through relation extraction, event detection, temporal reasoning, and template filling. Semantic role labeling then asks who did what to whom, connecting predicate-argument structure to resources such as PropBank and FrameNet, while affective analysis expands interpretation beyond factual content using sentiment lexicons, VAD-style emotion representations, semi-supervised lexicon induction, and connotation frames. From there, the focus widens to document-level understanding: coreference resolution and entity linking connect mentions to discourse entities and knowledge bases; discourse coherence models use RST, PDTB, centering, and entity-grid reasoning to explain how sentences hang together; and dialogue-act modeling shows how conversations are organized through turn-taking, grounding, adjacency pairs, and annotation schemes such as DAMSL.

Taken together, these sections map a progression from token classification to full discourse and interaction modeling. The goal throughout is the same: to show how NLP systems convert raw language into structured objects that support reasoning, retrieval, interpretation, and downstream decision-making.

---

## 1. Sequence Labeling in NLP: From Part-of-Speech Tagging to Conditional Random Fields

When we read a sentence, we seamlessly parse both the grammatical structure and the real-world semantic entities within it. We intuitively know which words act as nouns or verbs and automatically recognize when a phrase signifies a specific person, location, or organization. In Natural Language Processing (NLP), teaching a machine to do this falls largely under the umbrella of **sequence labeling**.

Two foundational tasks in sequence labeling are **Part-of-Speech (POS) tagging** and **Named Entity Recognition (NER)**. While they focus on different aspects of language—one grammatical, the other semantic—both can be elegantly modeled by assigning a sequence of labels to a sequence of input tokens. Formally, given an input sequence of words $X$, we want to predict a corresponding output sequence of labels $Y$ of the same length:

$$
X = (x_1, x_2, \ldots, x_n), \qquad
Y = (y_1, y_2, \ldots, y_n)
$$

This explores the linguistic background of these tasks and breaks down the classic models used to solve them: Hidden Markov Models (HMMs) and Conditional Random Fields (CRFs).

### The Linguistic Foundation: Word Classes and Entities

Before building mathematical models, we must understand the linguistic abstractions they aim to capture. Parts of speech are not merely dictionary definitions; they are distributional categories defined by how words behave in syntax and morphology. A POS class dictates whether a word can take plural marking, occur with a determiner, or modify a verb.

Generally, word classes are grouped into **open classes** (like nouns, verbs, and adjectives), which continually grow through coinage and borrowing, and **closed classes** (like prepositions and conjunctions), which are relatively fixed structural markers. Cross-linguistically, tagsets often balance broad universal applicability with language-specific quirks. For instance, the Universal Dependencies (UD) tagset uses 17 coarse tags applicable across many languages, whereas the English-oriented Penn Treebank tagset provides much finer granularity, distinguishing between different tenses of verbs (e.g., `VBD` vs. `VBN`).

While POS tagging addresses grammatical syntax, **Named Entity Recognition (NER)** is concerned with semantic typing. It identifies and categorizes proper names such as persons, locations, and organizations. NER is fundamentally harder than token-level POS tagging because it introduces a **segmentation** problem. Real-world named entities often span multiple words (e.g., "New York City"), meaning an NER system must accurately predict both the start and end boundaries of an entity mention, as well as its correct semantic type.

To bridge this gap and treat span predictions as a token-level problem, researchers utilize the **BIO tagging scheme**, which assigns distinct tags to the Beginning (B), Inside (I), and Outside (O) of an entity span. This clever reduction allows standard sequence models to process both POS tags and NER seamlessly.

### The Challenge of Ambiguity

Sequence labeling would be trivial if each word had only one potential label. In reality, text is riddled with **lexical ambiguity**. While most word types in a vocabulary might have only one tag, the most frequent words in a language are highly ambiguous. Consider the word "back," which can function as an adjective, noun, finite verb, infinitival verb, particle, or adverb depending entirely on its surrounding context.

Because of this, POS tagging is framed as a disambiguation task. The baseline for handling this ambiguity is simply predicting the most frequent tag for each word based on training data:

$$
\hat{t}(w) = \arg\max_{t \in \mathcal{T}} C(w, t)
$$

Surprisingly, this naive approach yields about 92% accuracy in English, indicating that while ambiguity is pervasive, it is heavily skewed. However, bridging the gap to human-level performance (~97%) requires models that can leverage sequential context.

### The Generative Approach: Hidden Markov Models (HMMs)

Historically, sequence labeling was dominated by generative models, most notably the **Hidden Markov Model (HMM)**. By imagining that sentences are generated by an underlying probabilistic process, HMMs offer a mathematically elegant way to decode linguistic structure.

An HMM extends a standard Markov chain by treating the sequence of tags as **hidden states** and the sequence of words as **observations**. The architecture is built on two core assumptions:

1. **The Markov Assumption:** The current tag depends only on the immediate preceding tag ( $P(y_i \mid y_{i-1})$ ).
2. **Output Independence:** The current observed word depends solely on the current tag ( $P(x_i \mid y_i)$ ).

Training an HMM involves estimating these transition and emission probabilities directly from a labeled corpus utilizing maximum likelihood.

When presented with a new, unlabeled sentence, identifying the most probable tag sequence means finding the hidden path that maximizes the joint product of these probabilities. The HMM decoding objective becomes:

$$
\hat{Y} \approx \arg\max_Y \prod_{i=1}^{n} P(x_i \mid y_i) P(y_i \mid y_{i-1})
$$

Instead of exhaustively testing an exponentially large number of tag combinations, HMMs employ **Viterbi decoding**. This dynamic programming algorithm builds a time-state lattice. For each step $t$ and state $j$, it computes the best path probability $v_t(j)$:

$$
v_t(j) = \max_{i=1}^{N} v_{t-1}(i)\,a_{ij}\,b_j(o_t)
$$

By tracking the `argmax` at each step and following backpointers from the end of the sentence, Viterbi efficiently extracts the optimal global sequence.

### The Discriminative Shift: Conditional Random Fields (CRFs)

While HMMs are computationally elegant, they enforce strict independence assumptions. What if a sequence tagger needs to consider whether a word is capitalized, whether it possesses a specific prefix or suffix, or whether it appears in a known gazetteer list of city names? Integrating arbitrary, overlapping features into a generative HMM is incredibly awkward.

Enter the **Conditional Random Field (CRF)**, specifically the **linear-chain CRF**. Instead of modeling a joint generative distribution, a CRF is a discriminative, log-linear model that directly computes the conditional probability of a whole tag sequence $Y$ given the entire input sentence $X$:

$$
P(Y \mid X) = \frac{1}{Z(X)} \exp\left(\sum_{k=1}^{K} w_k F_k(X,Y)\right)
$$

Here, $Z(X)$ is the global partition function that normalizes the sum over all possible sequences, $w_k$ represents learned feature weights, and $F_k(X,Y)$ represents the global feature functions.

CRFs construct a globally normalized score by summing local **feature functions** $f_k(y_{i-1}, y_i, X, i)$ across every position in a sentence. Because these features are arbitrarily engineered, a CRF can flexibly query anything about the input—from the current token to distant surrounding words, their capitalization shapes (like `"X'Xxxxxxxx"` for `"L'Occitane"`), and even external dictionary lookups.

Despite allowing arbitrarily powerful input features, the _linear-chain_ limitation—where output labels only conditionally depend on neighboring labels—ensures that the CRF's decoding can still be solved using an efficient Viterbi-style search. Because $Z(X)$ and the exponential function are monotonic and constant for fixed $X$, decoding simply maximizes the linear sum of local weighted features:

$$
\hat{Y} = \arg\max_Y \sum_{i=1}^{n}\sum_{k=1}^{K} w_k f_k(y_{i-1}, y_i, X, i)
$$

### Evaluation and Practical Realities

Beyond algorithms, turning sequence labeling into highly accurate applied systems requires careful evaluation metrics and data architecture.

Evaluating POS taggers usually relies on simple token accuracy, but **NER requires precision, recall, and F1 scores**. F1 is the harmonic mean of precision ( $P$ ) and recall ( $R$ ):

$$
F_1 = \frac{2PR}{P+R}
$$

Because NER predicts structured spans rather than single independent labels, any error in recognizing an entity's boundaries penalizes both recall (for failing to capture the whole entity) and precision (for predicting an incorrect span). A partially correct prediction essentially counts strictly as wrong.

In commercial applications, pure machine learning endpoints are usually supplemented by **hybrid rule-based pipelines**. A production system might first apply high-precision deterministic rules and substring matching, utilize domain-specific dictionaries, and only then fall back on a CRF or neural tagger for the remaining edge cases.

Finally, while English provides a relatively constrained morphological landscape, **morphologically rich languages** such as Turkish or Hungarian flip sequence labeling upside down. In these languages, words encode dense grammatical information like case, gender, possession, and number in one surface string. A single tagging decision involves selecting a structured morphological parse out of many candidates, inflating the required tagset by orders of magnitude and making sequence labeling a vastly more complex, combinatoric challenge.

---

## 2. Syntactic Parsing: Constituency and Dependency

## 2.1 Understanding Constituency Parsing: From Context-Free Grammars to Neural Span-Based Models

Natural language is not just a sequence of words; it possesses a hierarchical structure. Groups of contiguous words, known as _constituents_, act as single syntactic units—like noun phrases or verb phrases. The computational task of assigning these structures to sentences is called syntactic parsing, and it bridges the gap between mere strings of words and meaningful, structured linguistic data.

It explores the foundations of constituency parsing, the classic algorithms used to construct parse trees, and how modern neural networks handle the structural ambiguity inherent in human language.

### The Foundation: Context-Free Grammars and Treebanks

To systematically capture the hierarchical structures of language, linguists use **Context-Free Grammars (CFGs)**. Formally, a CFG is defined as a 4-tuple:

$$
G = (N, \Sigma, R, S)
$$

where $N$ is the set of abstract non-terminal symbols (like `NP` for Noun Phrase or `VP` for Verb Phrase), $\Sigma$ is the set of terminal symbols (actual words), $R$ represents the allowed rewrite rules (productions), and $S$ is the designated start symbol (the root sentence node). For example, $R$ might contain rules like:

$$
S \to NP\ VP, \quad NP \to Det\ Nominal, \quad Noun \to \text{flight}
$$

These rules specify how smaller units combine compositionally into larger ones. Complete syntactic derivations trace from $S$ down to the terminal symbols and form explicit _parse trees_, mapping out the sentence's grammatical backbone.

However, writing a robust grammar by hand is incredibly difficult due to the sheer complexity of natural language. Enter **treebanks**—large corpora of sentences manually annotated with complete parse trees, such as the widely used Penn Treebank. A treebank serves a dual purpose: it acts as a gold-standard evaluation set and implicitly defines an empirical grammar. By extracting every local parent-child configuration across thousands of trees, researchers can induce a probabilistic CFG that reflects actual corpus usage rather than theoretical intuitions alone.

### The Challenge of Structural Ambiguity

One might assume that parsing simply involves checking a sentence against known grammar rules. But language is riddled with **structural ambiguity**, where a single grammatical sentence admits multiple valid parse trees. Consider Groucho Marx's famous joke: _"I shot an elephant in my pajamas."_ Did the speaker shoot an elephant while wearing pajamas, or did the elephant happen to be wearing the pajamas?

This is a classic _PP-attachment ambiguity_. The grammar licenses trees where the Prepositional Phrase ("in my pajamas") attached to either the Verb Phrase or the Noun Phrase. Coordination phrases (e.g., "old men and women") introduce similar branching uncertainty. A parser cannot just find _a_ valid tree; it must represent all possible grammatical combinations and seamlessly identify the most semantically plausible interpretation.

### Dynamic Programming and CKY Parsing

Given the exponential number of possible parse trees for an ambiguous sentence, exhaustively trying all structures is computationally disastrous. The **Cocke-Kasami-Younger (CKY)** algorithm solves this by utilizing dynamic programming.

To use CKY, the grammar must first be converted into **Chomsky Normal Form (CNF)**, meaning every right-hand rule is binarized to branch into exactly two non-terminals ( $A \to B\ C$ ) or yields a single word ( $A \to w$ ). CKY operates bottom-up across word boundaries, known as "fenceposts". It builds an upper-triangular chart where each cell $Chart[i,j]$ keeps track of any non-terminal that can derive the substring encompassing words from index $i$ to $j$:

$$
Chart[i,j] = \{ A \mid A \Rightarrow^* \text{words}[i{:}j] \}, \qquad 0 \le i < j \le n
$$

The core dynamic programming recurrence checks partitions at a split point $k$:

$$
A \in Chart[i,j] \iff \exists k \text{ with } i < k < j,\ \exists (A \to B\ C) \text{ where } B \in Chart[i,k] \text{ and } C \in Chart[k,j]
$$

By treating overlapping spans as reusable subproblems and maintaining backpointers instead of redundantly duplicating subtrees completely, CKY compactly isolates a "shared packed forest" representing all possible parses efficiently. A sentence is then successfully recognized and grammatically valid if the final full-span cell contains the start symbol:

$$
S \in Chart[0,n]
$$

### The Deep Learning Era: Neural Span-Based Parsing

While CKY efficiently finds all grammatical parses, we still need an arbiter to select the _best_ parse. Historically, this involved assigning probabilities to CFG rules, but modern systems rely on neural networks to dynamically score substructures.

In a **Span-Based Neural Parser**, each possible span in the sentence is mapped to an internal vector representation. This is commonly constructed by running a powerful encoder like BERT over the sentence to acquire hidden token embeddings ( $y$ ), and taking the difference between directional boundary encodings at fenceposts $i$ and $j$:

$$
v(i, j) = [\overrightarrow{y_j} - \overrightarrow{y_i}\ ;\ \overleftarrow{y_{j+1}} - \overleftarrow{y_{i+1}}]
$$

A Multi-Layer Perceptron (MLP) then evaluates this learned span difference vector to produce a score $s(i, j, l)$ for each potential label $l$ (e.g., `NP`, `VP`). The score of an entire tree $T$ cleanly formalizes as the summed score of its component constituents:

$$
s(T) = \sum_{(i,j,l)\in T} s(i,j,l)
$$

Discovering the best tree transitions into an optimization problem: finding $\hat{T} = \arg\max_T s(T)$. Strikingly, neural parsers employ a dynamic programming recurrence reminiscent of CKY to achieve globally exact structural search. For any base case single-word span, the optimal subtree score is simply the best label assignment:

$$
s_{\mathrm{best}}(i, i+1) = \max_l s(i, i+1, l)
$$

For larger spans, the algorithm merges optimal left and right subtree scores at split point $k$, adding them to the best candidate parent label score:

$$
s_{\mathrm{best}}(i, j) = \max_l s(i, j, l) + \max_k \big[s_{\mathrm{best}}(i, k) + s_{\mathrm{best}}(k, j)\big]
$$

This allows the parser to compute the exact maximum likelihood tree, bypassing the need to test against hand-crafted symbolic grammar compatibilities.

### Evaluating Parses and Finding Lexical Heads

How do we structurally measure parsing success? Parsers are universally benchmarked against human-annotated gold datasets utilizing the **PARSEVAL** metric framework. A predicted constituent is deemed correct only if it matches the reference constituent's start boundary, end boundary, and semantic non-terminal label flawlessly.

From these strict intersection criteria, researchers compute **labeled precision** ( $P$ ), **labeled recall** ( $R$ ), and importantly, the harmonic mean defining the **F1 score**:

$$
F_1 = \frac{2PR}{P+R}
$$

To aggressively push constituency parsing closer to deeper semantic downstream systems, parser architectures frequently integrate **Head Percolation**. Theoretically, every phrase possesses a lexical _head_—the distinct core word commanding the constituent's overriding grammatical functionality (like the primary verb anchoring a VP). By executing designated priority search models (head rules) over a parse, parsers map internal non-terminals back to origin token heads. Running these rules over a completed CFG tree converts isolated, abstract constituent topologies directly into a connected dependency structure, bridging the abstract phrase tree smoothly into applied semantic meaning.

---

## 2.2 Dependency Parsing: From Transition Systems to Graph-Based Models

While constituency parsing divides sentences into hierarchical, nested phrases, **dependency parsing** takes a more direct approach. It bypasses explicit structural nodes (like Noun Phrases or Verb Phrases) and instead focuses purely on the directed grammatical relations between words. In a dependency grammar, every relation links a **head** (the structurally central word) to a **dependent** (the word that modifies or complements the head).

This approach has become exceptionally popular in modern Natural Language Processing (NLP). Dependency syntax is often much closer to conceptual predicate-argument structure—directly linking verbs to their subjects and objects—and naturally accommodates languages with free word order where logical constituents might be widely separated in the text. It explores the graph-theoretic foundations of dependency syntax and the two major algorithmic paradigms used to parse it: transition-based and graph-based models.

### Representing Dependency Syntax: Graphs and Projectivity

A dependency structure is formally represented as a directed graph:

$$
G = (V, A)
$$

where $V$ represents the vertices (the words in the sentence) and $A$ is the set of ordered arcs corresponding to labeled head-dependent relations (e.g., `NSUBJ` for nominal subject, or `OBJ` for object).

To make parsing computationally tractable and structurally consistent, NLP models generally constrain this graph to be a **rooted tree**, enforcing three strict conditions:

1. There is exactly one designated root word $r$ with no incoming arcs:

$$
\exists !\, r \in V \text{ such that } indeg(r) = 0
$$

2. Every other word has exactly one incoming arc (precisely one head):

$$
\forall v \in V \setminus \{r\}, indeg(v) = 1
$$

3. There is a unique directed path from the root to every word:

$$
\forall v \in V, \text{ there is a unique directed path } r \leadsto v
$$

A critical concept linking this abstract graph to the physical surface order of the sentence is **projectivity**. A dependency tree is projective if every arc only crosses over words that are dominated by the head of that arc. Formally, for a head $h$, a dependent $d$, and any intervening word $w$:

$$
\forall w,\ \min(h,d) < w < \max(h,d) \implies h \leadsto w
$$

Graphically, a projective tree can be drawn above the sentence without any crossing arcs. While non-projective trees are common cross-linguistically, projectivity is an important limitation constraint for certain fast parsing algorithms.

### Transition-Based Parsing: Shift-Reduce for Dependencies

**Transition-based parsing** reconstructs dependency trees incrementally through a left-to-right processing pipeline. It works distinctly like a shift-reduce parser in compiler theory.

The parser's state at any moment is defined by a **configuration**:

$$
c = (\sigma, \beta, R)
$$

where $\sigma$ is a stack holding processed tokens, $\beta$ is a buffer holding remaining unread tokens, and $R$ is the set of dependency relations built so far. In the classic **arc-standard** system, the parser iteratively queries an Oracle to choose one of three actions: `SHIFT` (move a word from buffer to stack), `LEFTARC` (make the second stack item a dependent of the top stack item, popping it), or `RIGHTARC` (make the top stack item a dependent of the second stack item, popping it only once all its own future dependents are already collected).

To move beyond deterministic, greedy decoding, modern systems implement **beam search**, which evaluates multiple competing partial parses simultaneously. The configuration score evaluates path histories additively based on local transition scores:

$$
ConfigScore(c_i) = ConfigScore(c_{i-1}) + Score(t_i, c_{i-1})
$$

#### The Neural Oracle

In early systems, the classifier scoring a transition relied on highly engineered, sparse feature templates. Modern **neural transition parsers** supplant this by encoding the sentence into dense vector representations and feeding the contextual embeddings of the key structural tokens (e.g., the top two stack items $s_1, s_2$ and the first buffer item $b_1$) through a feedforward network (FFN):

$$
\hat{t} = softmax\!\left( FFN\!\left([e(s_2); e(s_1); e(b_1)]\right)\right)
$$

This approach executes parsing extremely efficiently—scaling linearly with the sentence length—but purely greedy models can suffer from error propagation if an early mistake structurally blocks the single globally correct parse.

### Graph-Based Parsing: Global Optimization

Instead of building a parse sequentially, **graph-based parsing** embraces a global search strategy. The parser evaluates the space of all possible valid trees $G_S$ for a sentence $S$, aiming to find the parse that maximizes a global scoring function:

$$
\hat{T}(S) = \arg\max_{t \in G_S} Score(t, S)
$$

To make aggressively searching this exponentially large space tractable, the model relies on the **edge-factored assumption**: the score of an entire tree is strictly decomposed as the sum of the independent scores of its individual edges:

$$
Score(t, S) = \sum_{e \in t} Score(e)
$$

This brilliant reduction reframes dependency parsing to the well-known **Maximum Spanning Tree (MST)** problem for directed graphs. Algorithms like the Chu-Liu-Edmonds decoder can accurately and efficiently find the optimal global tree using this local combination formulation.

#### Biaffine Edge Scoring

While early systems computed edge scores $Score(e)$ with linear dot products over sparse templates, modern implementations integrate the highly expressive **biaffine neural scorer**.

The model encodes the entire sequence contextually. Each token $r_i$ is then mapped through dedicated feedforward networks to construct structurally specific vector representations for when it acts as a **head** versus when it acts as a **dependent**:

$$
h_i^{\text{head}} = FFN_{\text{head}}(r_i), \qquad h_i^{\text{dep}} = FFN_{\text{dep}}(r_i)
$$

The scalar score of creating a directed arc from head word $i$ to dependent word $j$ is then computed using a biaffine operator that cleanly captures deep multiplicative relationships alongside linear concatenation:

$$
Score(i \to j) = Biaff\!\left(h_i^{\text{head}}, h_j^{\text{dep}}\right) = (h_i^{\text{head}})^{\top} U (h_j^{\text{dep}}) + W\big(h_i^{\text{head}} \oplus h_j^{\text{dep}}\big) + b
$$

These pairwise scores are normalized via softmax and passed down to the MST decoder to structurally assemble the global parse. Additionally, because graph-based methods analyze all raw attachments holistically, they naturally encode and construct complicated non-projective trees, drastically increasing their computational robustness across diverse linguistic grammars.

### Evaluation: Measuring Attachment Accuracy

Because dependency parsing pivots entirely on word-to-word graph relations, classical sentence-level Exact Match (where a single errant edge invalidates an entire paragraph) is considered far too pessimistic for incremental developments. Instead, the field measures token-level attachment metrics across gold-standard benchmark datasets.

The two main metrics monitor differing operational success patterns:

- **Unlabeled Attachment Score (UAS):** Measures the percentage of words structurally assigned the computationally correct graph head, ignoring categorical relation labels completely:

$$
UAS = \frac{\{\text{tokens with correct head}\}}{N}
$$

- **Labeled Attachment Score (LAS):** Measures the stricter percentage of words securely assigned both the correct specific head and the correct linguistic dependency label:

$$
LAS = \frac{\{\text{tokens with correct head and correct dependency label}\}}{N}
$$

Comparing the variance gap between a model's UAS and LAS frequently acts as a diagnostic lens to distinguish whether the underlying neural architecture suffers primarily from boundary attachment limits or finer semantic labeling granularity limits.

---

## 3. Semantic Structure and Information Extraction

## 3.1 Unlocking Meaning with Information Extraction: Relations, Events, Time, and Templates

Reading a news article about a corporate acquisition is intuitive for a human but overwhelmingly unstructured for a computer. To make prose queryable, aggregate-able, and mathematically computable, systems rely on **Information Extraction (IE)**—the pipeline that transforms narrative text into strictly structured semantic records.

It explore how IE maps text into precise knowledge graphs: starting from pairwise relation extraction, moving through complex event and temporal modeling, and culminating in full scenario template filling.

### Relation Extraction: Identifying Typed Links

The first step beyond simply recognizing named entities is determining how those entities interact. **Relation Extraction** recovers structured, typed semantic links between entities to build out canonical knowledge bases (like Wikidata or DBpedia). For example, a system reads _"United, a unit of UAL Corp."_ and extracts:

$$
\text{part-of}(\text{United}, \text{UAL Corp.})
$$

Approaches to relation extraction lie on a spectrum of supervision:

- **Handwritten Patterns**: Early systems relied on rigid lexico-syntactic rules. Most famously, Hearst patterns extract taxonomies. If a sentence follows the pattern $NP_0 \text{ such as } NP_1$, we infer membership:

$$
\forall NP_i,\ i \ge 1,\ hyponym(NP_i, NP_0)
$$

- **Supervised Learning**: With enough annotated data, relation extraction is modeled as multiclass classification given contextual features of the subject and object:

$$
p(\text{relation} \mid \text{SUBJ}, \text{OBJ})
$$

- **Semi-Supervised Bootstrapping**: Systems can originate from just a few seed tuples and learn patterns iteratively. To regulate "semantic drift" (bad tuples breaking the feedback loop), metrics evaluate a pattern's confidence $p$ based on its precision against trusted tuples versus total productivity:

$$
Conf_{R\log F}(p) = \frac{|hits(p)|}{|finds(p)|}\log(|finds(p)|)
$$

A tuple's validity is then aggregated via a noisy-or model over all bounding patterns $P'$:

$$
Conf(t) = 1 - \prod_{p \in P'} \left(1 - Conf(p)\right)
$$

- **Open Information Extraction (Open IE)**: Without a fixed ontology, systems (like ReVerb) extract relations directly as raw strings $w$ bounded by noun phrases $x$ and $y$: $r = (x, w, y)$. They enforce rigid syntactic constraints to ensure validity (e.g., $V \mid VP \mid VW^{*}P$).

Because recalling every fact across infinite corpora is impossible, weakly supervised relation extractors are typically evaluated via sampled precision estimations rather than exact sentence-level matching:

$$
\hat{P} = \frac{\text{ correctly extracted relation tuples in the sample}}{\text{ extracted relation tuples in the sample}}
$$

### Extracting Events

Not all facts are static relations—texts consistently describe dynamic happenings. **Event extraction** identifies text spans corresponding to bounded occurrences or states.

Crucially, events are not strictly limited to verbs. Annotations recognize events natively across heterogeneous sequences:

$$
[EVENT\ x]
$$

A supervised sequence labeler reading nominal spans like _"the increase"_ must output an enriched feature classification detailing structural properties:

$$
(class = \text{REPORTING},\ tense = \text{PAST},\ aspect = \text{PERFECTIVE})
$$

Unlike simple NLP tagging, event detection heavily weighs token morphology against lexical context. A fundamental classic feature representation $\phi(w)$ for a given candidate word $w$ is defined as:

$$
\phi(w) =
\{
\text{affixes},
\text{nominalization suffix},
\text{POS},
\text{light-verb indicator},
\text{subject category},
\text{stem},
\text{verb root},
\text{WordNet hypernyms}
\}
$$

### Modeling Time and Aspect

Once events are identified, establishing their placement into timelines requires two distinct mathematical perspectives: **Temporal Location** (Tense) and **Internal Temporal Contour** (Aspect).

Event and timestamp configurations can be formalized qualitatively by 13 interval relations defined in **Allen's interval algebra**:

$$
\mathcal{R}_{Allen} =
\{
\text{before}, \text{after}, \text{overlaps}, \text{overlapped-by}, \text{meets}, \text{met-by},
\text{equals}, \text{starts}, \text{started-by}, \text{finishes}, \text{finished-by},
\text{during}, \text{contains}
\}
$$

#### Reichenbach’s Tense Framework

At the coarsest operational level, event placement is categorized relative to the present:

$$
time(E) \in \{\text{past}, \text{present}, \text{future}\}
$$

But language structurally shifts perspectives using grammatical tense. Reichenbach solved this complex conceptual mapping by analyzing tense configurations as a function over three temporal anchors:

$$ \text{Tense} = f(E, R, U) $$

where **E** refers to _Event time_, **U** is _Utterance time_, and **R** acts as an intermediate _Reference time_. For instance:

- $\text{Simple Past}: E = R < U$
- $\text{Past Perfect}: E < R < U$
- $\text{Present Perfect}: E < R = U$
- $\text{Present Simple}: E = R = U$
- $\text{Future}: U < R = E$
- $\text{Future Perfect}: U < E < R$

#### Vendler's Aspectual Classes

If tense focuses on _when_ an occurrence happens, aspect defines _how_ the occurrence structurally unfolds. Any situation description $e$ is grouped into one of four Vendler aspectual classes:

$$
AspectClass(e) \in \{\text{state}, \text{activity}, \text{accomplishment}, \text{achievement}\}
$$

These classes define the event's telicity (natural boundaries) and durativity:

- **States:** Unchanging property: $State(e) \Rightarrow \neg Change(e)$
- **Activities:** Processes with no built-in endpoint: $Activity(e) \Rightarrow Durative(e) \land \neg Telic(e)$
- **Accomplishments:** Goal-driven sequences concluding with a resulting state: $Accomplishment(e) \Rightarrow Durative(e) \land Telic(e) \land ResultState(e)$
- **Achievements:** Point-like momentary state transitions: $Achievement(e) \Rightarrow Punctual(e) \land Telic(e) \land ResultState(e)$

Aspectual classification is compositional. Modifying "I flew" ( $Activity$ ) into "I flew to New York" naturally converts the expression into an $Accomplishment$.

#### Formal Annotation and Processing (TimeML)

Data resources like TimeBank formally annotate texts using TimeML markup standards targeting core objects:

$$
\mathcal{O}_{TimeML} = \{\text{EVENT}, \text{TIME/TIMEX3}, \text{LINK}\}
$$

Where explicit connections separate temporal, structural aspect, and causal modality:

$$
\mathcal{L}_{TimeML} = \{\text{TLINK}, \text{ALINK}, \text{SLINK}\}
$$

Allowing documents to act natively as connected temporal graphs:

$$
G = (V, E),\quad
V = \{\text{events}, \text{times}\},\quad
E = \{\text{TLINKs}, \text{ALINKs}, \text{SLINKs}\}
$$

Converting textual temporal clues into automated standard expressions is mathematically formulated as an ordered processing sequence:

$$
\mathcal{P}_{time} =
\langle
\text{extract temporal expressions},
\text{normalize temporal expressions},
\text{link events and times}
\rangle
$$

Each temporal expression $x$ is categorized contextually:

$$ TempType(x) \in \{\text{absolute}, \text{relative}, \text{duration}\} $$

Expressions are bounded iteratively via IOB sequencing (e.g., $IOB(\text{last week}) = (B, I)$ ) and mapped down to ISO resolutions centered strictly around the document creation $anchor$:

$$
value(\text{tomorrow}) = anchor + 1,\quad value(\text{yesterday}) = anchor - 1
$$

Finally, pairwise classification bounds the text into ordered temporal logic:

$$
TLINK(x, y) \in \{\text{event-event}, \text{event-time}, \text{time-time}, \text{DCT-event}, \text{DCT-time}\}
$$

### Template Filling: Reconstructing Scenarios

The capstone of IE is **Template Filling**. Instead of extracting individual relations and events in isolation, systems map an overarching document down onto a completely structured narrative scenario (or "script").

A news brief about matching ticket changes canonically compiles into an overall event record:

$$
\text{FARE-RAISE ATTEMPT} =
\{
\text{LEAD AIRLINE}: \text{United Airlines},
\text{AMOUNT}: \$6,
\text{EFFECTIVE DATE}: 2006\text{-}10\text{-}26,
\\ \text{FOLLOWER}: \text{American Airlines}
\}
$$

Modern software architectures divide this dynamically by formally splitting the process between document-level template recognition and granular downstream slot extraction over predefined role parameters:

$$
FillTemplate(d) =
\big(
RecognizeTemplate(d),
\{FillRole_r(d)\}_{r \in Roles}
\big)
$$

Where each slot acts as an independent entity extraction classification:

$$
RoleFiller(r) \in \{\text{binary NP classifier}, \text{sequence model}\}
$$

Historically, iconic template-filler systems like FASTUS orchestrated these extraction mappings by assembling strict multi-stage rule pipelines using finite state transducers:

$$
\mathcal{C}_{FASTUS} =
\langle
\text{Tokens},
\text{Complex Words},
\text{Basic Phrases},
\text{Complex Phrases},
\text{Semantic Patterns},
\text{Merging}
\rangle
$$

Such hierarchies permitted parsing architectures capable of filling highly complex, recursively nested template scripts:

$$
\text{Tie-up-1}[\text{ACTIVITY}] = \text{Activity-1}
$$

---

## 3.2 Sentence Meaning: Semantic Roles, Lexical Resources, and Computational Modeling

Understanding "who did what to whom" is one of the most fundamental challenges in natural language processing (NLP). The same underlying event can be expressed in vastly different syntactic forms—active clauses, passive constructions, nominalizations, or prepositional phrases. Consider the following symbolic generalization for a purchase event across varied syntactic constraints:

$$
\exists e, x, y \; \text{Purchase}(e) \land \text{Buyer}(e, x) \land \text{Theme}(e, y) \land x = \text{XYZ Corporation} \land y = \text{stock}
$$

This text explores how computational linguistics abstracts away from surface variations to capture the core semantic architecture of events using semantic roles, rich lexical databases, and predictive machine learning pipelines.

### The Concept of Semantic Roles

At the core of language, predicates (typically verbs or nouns) express events or states, while their arguments represent participants. **Thematic roles** Historically labeled these participants across semantic boundaries. For instance, the subjects in "Sasha broke the window" and "Pat opened the door" serve the same driving function. We conceptually abstract this relationship as:

$$
\exists e, x, y \; P(e) \land \text{Agent}(e, x) \land \text{Theme}(e, y)
$$

Here, the `AGENT` encompasses the volitional causer, whereas the `THEME` is the object directly affected by the action. However, the same role set surfaces in multiple grammatical ways, a phenomenon called **diathesis alternations**. For example, the verb `break` assumes a thematic grid consisting of `AGENT`, `THEME`, and `INSTRUMENT`:

$$
\text{break}: \{\text{AGENT}, \text{THEME}, \text{INSTRUMENT}\}
$$

The mapping between semantic roles and syntax exhibits extreme flexibility. `break` allows realizing the `AGENT` as the subject and `THEME` as the object, but also permits `INSTRUMENT` or even `THEME` to take the grammatical subject position:

$$
\text{break} \Rightarrow
\{
\text{AGENT}/\text{Subject},\ \text{THEME}/\text{Object};\ \text{THEME}/\text{Subject} \dots
\}
$$

### The Shift to Proto-Roles and Granular Models

Traditional rigid thematic roles falter under computational scrutiny. Instruments like a "knife" can occupy subject positions, whereas a "fork" generally cannot (_"The fork ate the banana"_). Under strict boundaries, attempting to construct a formal logic definition yields poor empirical generalizability.

Consequently, linguistic models introduced:

1. **Proto-roles:** Instead of treating `AGENT` or `PATIENT` as universal constants, they are modeled as soft feature clusters. For example, `ProtoAgentFeatures = \{ \text{volitional}, \text{causal}, \text{sentient}, \text{intentional}, \text{moving} \}`.
2. **Verb-Specific Inventories:** Foregoing the attempt to define a universal set entirely, argument constraints apply purely to a designated semantic frame or verb sense.

### Encoding Semantics: PropBank and FrameNet

This verb-specific adaptation led to two monumental NLP architectures: **PropBank** and **FrameNet**.

**PropBank** assigns numbered arguments relative directly to a predicate-sense. Roles become locally bounded, mapping verbs specifically:

$$
\text{PropBankRoles}(v,s) = \{\text{Arg}0,\text{Arg}1,\text{Arg}2,\ldots\}
$$

For instance, the semantic evaluation for the verb sense `agree.01` allocates strictly defined numbered arguments:

$$
\text{agree}.01 =
\{
\text{Arg}0=\text{Agreer},\
\text{Arg}1=\text{Proposition},\
\text{Arg}2=\text{OtherEntityAgreeing}
\}
$$

While PropBank restricts itself to localized verb senses, **FrameNet** models semantics broadly around cognitive **frames**. A frame clusters shared structural assumptions independent of exact word selection. The `Change position on a scale` frame natively unifies sentences featuring verbs like `"increase"` or `"fall"` alongside nouns like `"rise"` using shared structural labels (`ITEM`, `DIFFERENCE`, `INITIAL VALUE`).

### Automating the Process: Semantic Role Labeling (SRL)

Transforming lexical databases into automated labeling tasks generates **Semantic Role Labeling (SRL)**. SRL is a supervised learning structure tasked with accurately isolating and pairing semantic roles.

Early paradigms employed **feature-based classification**. Systems produced syntactic parses and applied localized predictions mathematically seeking the optimal role ( $\hat{r}$ ):

$$
\hat{r} = \arg\max_{r \in R \cup \{\text{NONE}\}} P(r \mid \phi(n,p,\pi))
$$

Where $\phi(n,p,\pi)$ computes abstract syntactic parameters spanning path, phrase identity, and grammatical context extracted from a tree evaluation:

<details><summary>Code</summary>

```text
function SEMANTICROLELABEL(words) returns labeled tree
  parse ← PARSE(words)
  for each predicate in parse do
    for each node in parse do
      featurevector ← EXTRACTFEATURES(node, predicate, parse)
      CLASSIFYNODE(node, featurevector, parse)
```

</details>

By contrast, modern iterations model SRL using **neural sequence labeling**, classifying entire output arrays over sequence combinations directly mathematically utilizing deep contextual decoders operating atop BIO formats:

$$
\hat{y} = \arg\max_{y \in T} P(y \mid w)
$$

### Selectional Restrictions and Preferences

A parser identifying accurate grammatical linkages might still parse mathematically illogical sentences ("I ate a nearby location"). To correct this, algorithms enforce **selectional restrictions** natively preventing logical collapse:

$$
\exists e, x, y \; \text{Eating}(e) \land \text{Agent}(e,x) \land \text{Theme}(e,y) \land \text{EdibleThing}(y)
$$

Due to the extreme rigidity of strict logic flags (`EdibleThing`), computational heuristics evolved to handle probabilistic **selectional preferences** using entropy constraints—most notably Kullback-Leibler (KL) divergence which detects disparities across distributions:

$$
D(P \parallel Q) = \sum_x P(x)\log \frac{P(x)}{Q(x)}
$$

Leveraging KL parameters directly across lexical databases evaluates a predicate's **Selectional Preference Strength** ( $SR(v)$ ), marking how structurally informative the word becomes over its distribution baseline:

$$
SR(v) = D(P(c \mid v) \parallel P(c)) = \sum_c P(c \mid v)\log \frac{P(c \mid v)}{P(c)}
$$

Additionally measuring an explicitly designated class's likelihood of fulfilling that condition yields a normalized **Selectional Association Score** ( $AR(v,c)$ ):

$$
AR(v,c) = \frac{1}{SR(v)} P(c \mid v)\log \frac{P(c \mid v)}{P(c)}
$$

Ultimately, corpus-based models process compatibility lacking explicit class constraints utilizing conditional probabilities extrapolated over gigantic datasets like $P(n \mid v,r) = \frac{C(n,v,r)}{C(v,r)}$.

### Primitive Decomposition of Predicates

Is it mathematically possible to distill semantic predicates into narrower granular atoms? **Componential analysis** aggressively dismantles verbs. The premise assumes words like `kill` are not theoretically atomic but compose overlapping operations. Semantically, `kill(x,y)` parses mathematically equivalently alongside state transitions:

$$
\text{KILL}(x,y) \Leftrightarrow \text{CAUSE}(x,\text{BECOME}(\text{NOT}(\text{ALIVE}(y))))
$$

Evaluating overlapping states structurally synchronizes inchoative expressions natively. To decompose the event mechanics across alternations of the word `open`:

$$
\text{John opened the door} \Rightarrow \text{CAUSE}(\text{John},\text{BECOME}(\text{OPEN}(\text{door})))
$$

$$
\text{The door opened} \Rightarrow \text{BECOME}(\text{OPEN}(\text{door}))
$$

$$
\text{The door is open} \Rightarrow \text{OPEN}(\text{door})
$$

Approaches like **Conceptual Dependency (CD)** attempt to project all linguistic mechanics entirely onto foundational baseline primitives. For "The waiter brought Mary the check", CD allocates layered transfers involving `ATRANS` (possession manipulation) overlapping synchronized `PTRANS` (physical relocation), trading everyday accessibility for pure representational uniformity.

---

## 3.3 Computational Approaches to Affective Meaning and Sentiment Analysis

Natural Language Processing traditionally focuses on deciphering factual references, but human language is heavily colored by internal evaluations, opinions, and emotional states. Expanding analysis to capture this subjective depth requires models of **affective meaning**. This expansion treats sentiment analysis as just one manifestation of a broader affective spectrum encompassing emotions, interpersonal stances, attitudes, and personality traits.

Instead of treating all words as equally important tokens in a classifier, NLP frequently utilizes **affective lexicons**—directories of emotionally significant words. By establishing a semantic assumption that words reliably carry connotations, we can map textual inputs to nuanced emotional states.

### Defining Emotion: Categorical vs. Dimensional

To analyze emotion computationally, systems typically map affect using one of two dominant conceptual frameworks.

**1. Basic Emotion Theory:** This perspective treats emotions as discrete, atomic categories. A prominent example is Ekman’s six universal basic emotions, framing emotion recognition as a finite multiclass classification problem:

$$
\mathcal{E}_{\text{Ekman}} = \{\text{surprise}, \text{happiness}, \text{anger}, \text{fear}, \text{disgust}, \text{sadness}\}
$$

**2. Dimensional Emotion Theory:** Rather than discrete classes, this theory maps emotional states continuously along psychological axes—most frequently **Valence** (pleasantness), **Arousal** (activation or energy), and **Dominance** (control or power). In this space, any given item $x$ is defined as a coordinate vector:

$$
\mathbf{z}(x) = (v(x), a(x), d(x)) \in \mathbb{R}^3
$$

Standard sentiment analysis is essentially a simplified subcase of this dimensional framework mapping directly to the valence axis: $\text{sentiment}(x) \approx v(x)$.

### The Spectrum of Affect Lexicons

The implementation of these theoretical models produces diverse sentiment and affect lexicons.

- **Binary Sentiment Lexicons:** Encode simple, discrete mappings $L_{\text{bin}}(w) \in \{+1,-1\}$ indicating positive or negative polarities.
- **VAD Lexicons:** Output bounded real-valued vectors representing dimensional scores: $L_{\text{VAD}}(w) \in [0,1]^3$.
- **Multilabel / Intensity Lexicons:** Dictionaries like EmoLex assign multiple binary categories simultaneously, whereas resources like the NRC Emotion Intensity Lexicon output graded continuous association strengths $L_{\text{intensity}}(w,e) \in [0,1]$ across distinct emotions.

Constructing foundational lexicons involves human labeling methodologies ranging from simple crowdsourced ordinal ratings (subsequently aggregated out via modes and thresholds) to comparative systems. For VAD lexicons intrinsically relying on graded vectors, annotations utilize **Best-Worst Scaling**, evaluating a target property $p$ continuously via:

$$
s_p(w) = P(\text{best} \mid w,p) - P(\text{worst} \mid w,p)
$$

### Semi-Supervised Lexicon Induction

Manually labeling every word in existence is generally impossible. NLP navigates this using semi-supervised algorithms to geometrically extrapolate vast lexicons from small "seed" sets of trustworthy affective words.

One core procedure defines a **Semantic Axis**. By plotting the embeddings $E(w)$ of positive and negative seeds, the system isolates average pole centroids:

$$
V^{+} = \frac{1}{n} \sum_{i=1}^{n} E\!\left(w_i^{+}\right), \qquad
V^{-} = \frac{1}{m} \sum_{i=1}^{m} E\!\left(w_i^{-}\right)
$$

Subtracting the negative pole from the positive formulates an evaluative semantic axis ( $V_{\text{axis}} = V^{+} - V^{-}$ ). Any unclassified word is then instantly scored utilizing pure geometric cosine similarity:

$$
score(w) = \cos\!\big(E(w), V_{\text{axis}}\big) = \frac{E(w) \cdot V_{\text{axis}}}{\|E(w)\| \, \|V_{\text{axis}}\|}
$$

An alternative, topological approach utilizes **Label Propagation** (e.g., SentProp). This transforms the entire embedding space into a connected graph based on angular distance weightings:

$$
E_{i,j} = \arccos\!\left(- \frac{w_i^{\top} w_j}{\|w_i\| \, \|w_j\|}\right)
$$

<details><summary>Code</summary>

```text
Algorithm SentProp:
1. Build a lexical graph from word embeddings.
2. Formulate positive and negative seed nodes.
3. Perform random graph walks propagating outward from seeds.
4. Tally normalized positive visitations vs total visitations.
5. Bootstrap variance using seed subsets to ensure stability.
```

</details>

### Supervised Learning of Word Sentiment

When NLP algorithms operate on labeled documents—such as online reviews mapped to star-ratings—they leverage massive datasets to statistically uncover exactly which words belong to which affective buckets.

Raw frequency correlations falter wildly on rare lexicon tokens. To counteract this, models utilize the **Log Odds Ratio with Informative Dirichlet Priors** (Monroe et al.). It anchors isolated empirical frequencies towards a stable global background prior ( $\alpha_w$ ), mapping variations reliably:

$$
\delta_w^{(i-j)} = \log \frac{f_i^w + \alpha_w}{n_i + \alpha_0 - (f_i^w + \alpha_w)} - \log \frac{f_j^w + \alpha_w}{n_j + \alpha_0 - (f_j^w + \alpha_w)}
$$

Dividing this prior-adjusted log odds ratio by an estimation of standard deviation guarantees statistically verified representations: $z_w = \hat{\delta}_w^{(i-j)} / \sqrt{\sigma^2}$.

### Implementing Lexicons in Sentiment Recognition

With lexicons mathematically verified, mapping text documents simply requires counting and cross-referencing matched lexicon tokens algorithmically.

Using a foundational predictive model, a document parses positive matching tokens and negative matching tokens mathematically accumulating respective weighted totals ( $f^+$, $f^-$ ). The deterministic predictor applies a threshold filter $\lambda$ producing a non-neutral output classification only when dominance guarantees clarity:

```math
sentiment =
\begin{cases}
+ & if  \frac{f^{+}}{f^{-}} > \lambda \\
- & if  \frac{f^{-}}{f^{+}} > \lambda \\
0 & \text{otherwise}
\end{cases}
```

More sophisticated supervised classifiers utilize these count calculations as dense parameters layered alongside advanced phrase-extraction methodologies like Pointwise Mutual Information (PMI).

### Entity-Centric Affect and Connotation Frames

Documents rarely evaluate the world through entirely uniformly painted dimensions. Instead, texts hold multifaceted, varying stances dependent on the specific **Entity** mentioned (e.g., "The teenager survived the bombing").

Entity-centric approaches replace raw text counts with contextual embeddings extracting distinct predictions localized solely to character mentions using underlying trained regressors resolving multiple mentions back to document aggregates.

The conceptual peak of this logic formulates **Connotation Frames**. A predicate like "violate" implies entirely overlapping emotional dependencies across the subject doing the violating, the affected theme, and the observing writer's embedded sympathy. Frames enforce relationships spanning not just straightforward positive/negative Sentiment ( $S$ ), but underlying grammatical statuses covering **Agency** and **Power**:

$$
S(\text{writer} \to \text{Role1}), \qquad S(\text{writer} \to \text{Role2}), \qquad S(\text{Role1} \to \text{Role2})
$$

$$
power(\text{AG} < \text{TH}), \qquad power(\text{AG} > \text{TH})
$$

Instead of merely parsing if a chunk of text represents "sadness" or "anger", the system explicitly understands the relational impact connecting language to deeply structured emotional realities.

---

## 4. Discourse and Conversation

## 4.1 Coreference Resolution and Entity Linking

While natural language processing traditionally extracts facts from single sentences, true language comprehension requires tracking entities across an entire document. When a text mentions "Victoria Chen," later refers to "she," and finally identifies "the 38-year-old CEO," the system must recognize that all these phrases map to the same real-world or discourse entity. We solve this via a pipeline consisting of tracking discourse identities (**Coreference Resolution**) and mapping them to universal identifiers (**Entity Linking**).

### Discourse Models and Coreference Phenomena

Every time a text introduces an entity, it creates a new concept in our internal **discourse model**.

- A **mention** (or referring expression) is any textual span that introduces or accesses an entity.
- A **referent** is the underlying discourse entity.
- Two mentions are coreferent if they share the same referent:

$$
m_a \sim m_b \iff ref(m_a) = ref(m_b)
$$

The length and nature of referring expressions act on a spectrum of **salience**. Brand-new entities require descriptive noun phrases, while highly accessible entities are often pruned into pronouns:

$$
Complexity(m_t) \propto \frac{1}{Salience(e_t)}, \quad e_t = ref(m_t)
$$

However, identifying true references is complicated by non-referential text. Mention detection requires broad initial pooling from which invalid spans are filtered:

$$
\mathcal{M}_0(T)=\{s \subseteq T \mid NP(s) \lor PossPron(s) \lor NE(s)\}
$$

$$
\mathcal{M}_{\text{ref}} = \mathcal{M}_0 \setminus \left( \mathcal{E}_{\text{pleonastic}} \cup \mathcal{E}_{\text{appositive}} \cup \mathcal{E}_{\text{predicative}} \cup \mathcal{E}_{\text{numeric}} \cup \mathcal{E}_{\text{embedded}} \right)
$$

For instance, rule-based systems explicitly filter pleonastic/expletive `it` using grammatical patterns:

<details><summary>Code</summary>

```text
# Rule-based filter patterns for pleonastic 'it'
It is Modaladjective that S
It is Modaladjective (for NP) to VP
It is Cogv-ed that S
It seems/appears/means/follows (that) S
```

</details>

### Architectures for Coreference Algorithms

Coreference transforms a text into a set of mentions $\mathcal{M}$ and organizes them into disjoint entity clusters $\mathcal{C}$. Modern NLP classifies architectures into three historical phases.

#### Mention-Pair Architecture

A binary classifier processes every candidate pair of mentions and answers yes/no for coreference.

$$
p_{ij} = p(coref=1 \mid m_j, m_i), \quad j < i
$$

To train this without exploding negative samples, systems conventionally use the **closest-antecedent heuristic**:

<details><summary>Code</summary>

```text
for each mention m_i that has an antecedent:
  find the closest earlier gold antecedent m_j
  add (m_i, m_j) as one positive training pair
  for each mention m_k between m_j and m_i:
    add (m_i, m_k) as a negative training pair
```

</details>

At inference time, decisions must be combined into actual clusters using either a closest-first or best-first clustering strategy followed by transitive closure:

```math
\hat{y}_i^{\text{closest}} = \max \left\{ j < i \mid p_{ij} > \tau \right\} \quad \text{or} \quad \hat{y}_i^{\text{best}} = \arg\max_{1 \le j < i} p_{ij}
```

#### Mention-Ranking Architecture

Rather than independent binary pairs, mention-ranking pits all prior candidate antecedents against one another, combined with a **dummy antecedent $\epsilon$** to account for non-anaphoric mentions.

$$
Y(i) = \{1, \ldots, i-1, \epsilon\}, \quad \hat{y}_i = \arg\max_{y \in Y(i)} s(i,y)
$$

#### Entity-Based Architecture

Entity models bypass pairwise mentions to assign a mention directly to an established cluster, encoding properties like **Cluster Shape**:

$$
Shape(C) = [\tau(m_{c_1}), \tau(m_{c_2}), \ldots, \tau(m_{c_n})]
$$

When processing with classical explicit hand-built features across an anaphor and antecedent cluster, these signals are typically conjoined to learn context-specific combinations over vectors:

$$
\phi_{\text{ent}}(m_i, C_j) = \left[ \phi_{\text{ana}}(m_i); \phi_{\text{cluster}}(C_j); \phi_{\text{entity-pair}}(m_i, C_j) \right]
$$

### End-To-End Neural Mention-Ranking

Modern SOTA coreference employs end-to-end models (like `e2e-coref` with BERT or SpanBERT) where mention detection and antecedent linking are optimized simultaneously without brittle early pruning.

The pipeline enumerates all possible unigrams, bigrams, and spans up to a maximum width ( $\sim \frac{T(T+1)}{2}$ spans ).
To capture a span's semantic identity, its representation vector $g_i$ consists of its boundary tokens combined with $h_{ATT(i)}$, an attention-weighted internal summary that acts as a learned soft head-word:

$$
g_i = \left[ h_{START(i)}, h_{END(i)}, h_{ATT(i)} \right]
$$

$$
a_{i,t} = \frac{\exp(\alpha_t)} {\sum_{k=START(i)}^{END(i)} \exp(\alpha_k)}, \quad h_{ATT(i)} = \sum_{t=START(i)}^{END(i)} a_{i,t} \cdot e_t
$$

The final linking score is seamlessly decomposed into independent mention strengths and dense pair-wise compatibility:

$$
s(i, j) = m(i) + m(j) + c(i, j)
$$

(With $s(i, \epsilon) = 0$).

Because human annotations provide holistic clusters—not specific designated antecedents—the model trains by marginalizing the probability over **all legal gold antecedents** within the cluster:

$$
L = \sum_{i=2}^{N} -\log \sum_{\hat{y} \in Y(i) \cap GOLD(i)} P(\hat{y})
$$

<details><summary>Code</summary>

```text
# Neural Inference Pipeline:
1. Enumerate all spans in document D up to maximum width.
2. Build span representations g_i.
3. Compute mention score m(i) and prune weak candidate spans.
4. For each remaining span i and antecedent candidate j:
     Compute pair score c(i, j).
     Combine into total logit s(i, j) = m(i) + m(j) + c(i, j).
5. Apply softmax to predict best antecedent or epsilon dummy.
6. Apply transitive closure.
```

</details>

### Entity Linking (Wikification)

Entity linking ties a coreferent textual span to an objective knowledge base entry (like Wikipedia).

#### TAGME: Frequency and Coherence Strategy

Older heuristics like TAGME leverage historical hyperlink frequencies to determine a **Prior probability**, and measure **Relatedness/Coherence** by measuring how frequently Wikipedia tracks in-link overlap between two nodes:

$$
p(e \mid a) = \frac{count(a \to e)}{link(a)}
$$

$$
rel(A,B)= \frac{\log\!\big(\max(|in(A)|,|in(B)|)\big)-\log\!\big(|in(A)\cap in(B)|\big)}{\log(|W|)-\log\!\big(\min(|in(A)|,|in(B)|)\big)}
$$

By aggregating the relatedness scores (votes) of other detected anchors in the sentence, TAGME chooses globally coherent real-world entities.

#### ELQ: Neural Bi-Encoders

Modern linking uses Bi-Encoders. A contextual model computes start, end, and internal mention scores to detect mention boundaries, while independently retrieving deeply cached Wikipedia embeddings for comparisons:

$$
x_{e_i} = BERT_{[\mathrm{CLS}]}\!\left( [\mathrm{CLS}]\,t(e_i)[\mathrm{ENT}]\,d(e_i)[\mathrm{SEP}] \right)
$$

Span arrays from the question $y_{i,j}$ are evaluated against entity database items via inner dot product $x_e \cdot y_{i,j}$.

### Structured Coreference Evaluation

Evaluating clusters differs essentially from evaluating isolated labels.

**Link-Based Scoring (MUC):** Tests precision/recall on pairwise connections between instances. However, MUC rewards massive over-aggregating clusters and utterly ignores singletons.

$$
P_{\mathrm{MUC}}= \frac{|Links(H)\cap Links(R)|} {|Links(H)|}, \quad R_{\mathrm{MUC}}= \frac{|Links(H)\cap Links(R)|} {|Links(R)|}
$$

**Mention-Based Scoring (B³):** Tests the structural integrity of clusters localized around the context of each individual mention $i$.

$$
P_i^{B^3}=\frac{|H_i\cap R_i|}{|H_i|}, \quad R_i^{B^3}=\frac{|H_i\cap R_i|}{|R_i|}
$$

To compensate for biases, SOTA benchmarks (like CoNLL) operate explicitly across combined metrics:

$$
CoNLL= \frac{F1_{\mathrm{MUC}}+F1_{\mathrm{CEAF-e}}+F1_{B^3}}{3}
$$

### Real-World Limitations

#### Winograd Schema Problems

Shallow statistics routinely fail against semantic **Winograd Schemas**—minimal pairs where syntax is constant but a single lexical substitution flips the valid human referent.

$$
Ans(x, w_1) = a_1, \quad Ans(x, w_2) = a_2, \quad w_1 \neq w_2
$$

> "The trophy didn't fit into the suitcase because it was too large. What was too large? (The trophy)."

Solving such puzzles fundamentally demands hybrid intersections of raw language-model statistical reasoning ( $f_{\text{LM}}$ ) coupled with physical and logical knowledge encoding ( $f_{\text{KB}}$ ).

#### Gender Bias in NLP

Data imbalance often causes statistical coreference models to rely on stereotypical gender-occupation expectations, exposing dangerous performance gaps ( $BiasGap$ ).

$$
BiasGap(M) = Perf(M,\mathcal{D}_{\text{pro}}) - Perf(M,\mathcal{D}_{\text{anti}}) \gg 0
$$

To fix this, datasets are structurally augmented with gender-swapped training replications:

$$
\mathcal{D}_{\text{aug}} = \mathcal{D}_{\text{OntoNotes}} \cup SwapGender(\mathcal{D}_{\text{OntoNotes}})
$$

This counterfactual re-weighting largely restores parity on anti-stereotypical diagnostic benchmarks (like WinoBias or GAP) without penalizing underlying predictive resolution capabilities.

---

## 4.2 Discourse Coherence and Structure

A fully comprehensible natural language text is not a random bag of sentences; it is a structured, purposeful object. The property that makes a sequence of sentences feel meaningfully connected and interpretable as a unified body of thought is called **Discourse Coherence**.

Coherence operates at two main levels:

- **Local Coherence**: The connectedness between adjacent or nearby sentences.
- **Global Coherence**: The overarching, genre-specific organizational structure of the entire text (such as a narrative arc or a scientific paper's layout).

It explores the representational theories and computational models that allow machines to parse, learn, and evaluate both local and global discourse coherence.

### Local Coherence via Rhetorical Relations

One of the strongest signals of coherence is the presence of an interpretable semantic link between two spans of text.

#### Rhetorical Structure Theory (RST)

RST organizes local coherence relations into a single, global hierarchical tree spanning the entire text. It relies on minimal units called **Elementary Discourse Units (EDUs)**. Crucially, RST relations are asymmetric: they connect a **Nucleus** (the central informational span) to a **Satellite** (a supporting span, often uninterpretable without the nucleus).

$$
RSTRel_{\ell}(N,S), \quad Centrality(N) > Centrality(S)
$$

RST analysis forms a tree $T_{\text{RST}} = (V,E)$ whose leaves are EDUs and whose internal nodes define larger discourse spans combined under labels like `Reason`, `Evidence`, or `Elaboration`.

#### Penn Discourse TreeBank (PDTB)

Instead of a whole-text tree, PDTB focuses on lexically-grounded, shallow pairwise annotation. A relation links two arguments (`Arg1`, `Arg2`) often triggered by a specific **discourse connective** (e.g., _because_, _although_).

$$
r_{\text{PDTB}} =
\begin{cases}
(Arg1, c, \sigma, Arg2) & \text{explicit connective } c \\
(Arg1, \hat{c}, \sigma, Arg2) & \text{implicit/inferred connective } \hat{c}
\end{cases}
$$

These connections belong to a hierarchical ontology consisting of `Temporal`, `Contingency`, `Comparison`, and `Expansion`.

### Discourse Structure Parsing

Discourse parsing translates the previous theories into supervised prediction problems.

#### Transition-Based RST Parsing

Building a full RST tree requires first splitting the text into EDUs via sequence labeling. Then, a neural shift-reduce parser processes the EDUs using a stack ( $\sigma_t$ ) and a queue ( $q_t$ ).

<details><summary>Code</summary>

```text
# Simplified Shift-Reduce Parsing for RST
state = (stack = [], queue = [e1, e2, ..., en])
while not final(state):
  action = parser(state)
  if action == SHIFT:
    move first EDU from queue to stack
  elif action == REDUCE(label, direction):
    merge top two stack subtrees (e.g., using NN, NS, SN directions)
  else:
    POP_ROOT
return discourse_tree
```

</details>

To decide the next action, the system uses hierarchical biLSTMs to build robust span representations. Word embeddings inside an EDU span are average-pooled to build an EDU vector $x^e$. Subtrees are similarly pooled. An encoder-decoder maps the stack and queue configurations into action logits $o_a$, trained via softmax cross-entropy loss:

$$
o = W([h^t_{s_0}; h^t_{s_1}; h^t_{s_2}; h^e_{q_0}]), \quad p_a = \frac{\exp(o_a)}{\sum \exp(o_{a'})}
$$

#### Shallow PDTB Parsing

PDTB parsing avoids global trees, instead solving four subtasks:

1. Detect connectives and disambiguate their senses (e.g., is "and" a conjunction or a discourse connective?).
2. Find Arg1 and Arg2 spans.
3. Classify explicit senses.
4. Predict implicit relations between adjacent sentences using paired sentence embeddings:

$$
\hat{\sigma} = softmax\!\left(W_{\sigma}\,\tanh([h^{(1)}_{\text{CLS}}; h^{(2)}_{\text{CLS}}]) + b_{\sigma}\right)
$$

### Centering and Entity-Based Coherence

Rhetorical relations are not the only way to establish local connectedness. A discourse is also coherent if it reliably tracks **salient entities** rather than chaotically jumping between subjects.

#### Centering Theory

Centering formalizes salience through utterance-level entity transitions. Each utterance $U_n$ evokes a ranked list of **Forward-looking centers** $Cf(U_n)$ based on syntactic prominence (subjects > objects). The most salient is the **Preferred center** $Cp(U_n)$. The entity actually in focus from the _previous_ utterance is the **Backward-looking center** $Cb(U_n)$.

The transition from $U_n$ to $U_{n+1}$ characterizes the continuity:

$$
Trans(U_n,U_{n+1}) =
\begin{cases}
\text{Continue} & Cb(U_{n+1}) = Cb(U_n) \land Cb(U_{n+1}) = Cp(U_{n+1}) \\
\text{Retain} & Cb(U_{n+1}) = Cb(U_n) \land Cb(U_{n+1}) \ne Cp(U_{n+1}) \\
\text{Smooth-Shift} & Cb(U_{n+1}) \ne Cb(U_n) \land Cb(U_{n+1}) = Cp(U_{n+1}) \\
\text{Rough-Shift} & Cb(U_{n+1}) \ne Cb(U_n) \land Cb(U_{n+1}) \ne Cp(U_{n+1})
\end{cases}
$$

Highly coherent text maximizes Continues and avoids Rough-Shifts.

#### The Entity Grid Model

A machine learning translation of centering assumes that text coherence correlates with smooth coreference topologies. An entity grid $G$ is an $N \times M$ matrix where rows are sentences and columns are coreferent entities. The matrix records grammatical roles (`S`=Subject, `O`=Object, `X`=Other, `-`=Absent).
Coherence can be scored using fixed-dimensional transition probabilities of sliding windows down the columns:

$$
p(t \mid G) = \frac{freq(t;G)}{\sum_{t' \in \mathcal{T}_n} freq(t';G)} \implies \Phi(x) = (p_1(x), \ldots, p_m(x))
$$

### Representation Learning Models

Rather than designing discrete linguistic grids or trees, modern methods treat coherence as **Topical similarity** using neural representations. Sentence adjacency functions as a supervision signal: naturally contiguous sentences should look more "similar" than randomized ones.

The **Local Coherence Discriminator (LCD)** constructs interaction feature vectors $z(s,t)$ between neighboring sentence embeddings $s$ and $t$:

$$
z(s,t)=\left[s;\; t;\; s-t;\; |s-t|;\; s \odot t\right]
$$

A feed-forward network converts this into a coherence score $f_\theta(s,t)$. LCD uses **self-supervision** via a ranking margin loss, penalizing the model if it fails to score a true adjacent pair $(s_i, s_{i+1})$ higher than a pair stitched together from random sentences in the same document $(s_i, s')$:

$$
L_\theta = \mathbb{E}_{p(s' \mid s_i)} \left[ \max(0, \eta - f_\theta(s_i,s_{i+1}) + f_\theta(s_i,s')) \right]
$$

### Global Discourse Coherence

Local pairs and entity grids cannot capture the overarching "plot" of a document. Global coherence relies upon genre-specific schemas dictating how specialized discourse flows.

#### Argument Mining

Persuasive essays follow schemas composed of **Claims** and **Premises** mapped through relations like `SUPPORT` and `ATTACK`. Argumentative coherence is highly non-local—a single thesis claim in sentence 1 may be supported recursively by premises in sentences 3, 5, and 12.

$$
\hat{\tau}(v) = \arg\max_{c \in \{\text{claim}, \text{premise}, \text{none}\}} P(c \mid v)
$$

#### Scientific Discourse Zoning

Scientific papers follow rigid epistemological pipelines (AIM $\to$ METHOD $\to$ RESULT $\to$ EXTERNAL COMPARISON). Argumentative zoning treats scientific coherence as a sequence classification task over rhetorical domains:

$$
\hat{z}_i = \arg\max_{z \in \mathcal{Z}} P(z \mid s_i), \quad \mathcal{Z} \in \{\text{AIM}, \text{OWN METHOD}, \text{SUPPORT}, \text{GAP WEAK}, \dots\}
$$

By fusing rhetorical spans, entity transitions, topical representations, and global argumentation graphs, modern NLP systems can analyze reading continuity and discourse-level logic simultaneously.

---

## 4.3 Conversation and Dialogue Acts

Conversation is not merely a sequence of isolated sentences; it is a collaborative, highly structured joint activity where multiple participants react to one another over time. This structure dictates how natural interaction unfolds, applying both to human-human conversations and human-LLM (Language Model) conversations.

Formally, we can view a conversation $D$ computationally as an ordered sequence of semantic interactions:

$$
D = \langle (s_1, u_1, b_1), (s_2, u_2, b_2), \dots, (s_n, u_n, b_n) \rangle
$$

where for each turn index $i$, $s_i$ is the speaker, $u_i$ is the spoken utterance, and $b_i$ denotes the boundary timings. Processing and annotating these elements provides the basis for building effective spoken dialogue systems.

### Properties of Human Conversation

Human conversations coordinate several complex structural mechanisms seamlessly. Modern conversational AI struggles precisely because these underlying properties are intricate.

#### Turn-Taking and Endpointing

Dialogue assumes alternating contributions. A system must therefore solve the **endpointing** problem: detecting when a user is done speaking so the system can reply without interrupting. This cannot rely on silence alone (due to natural mid-sentence pauses) and must be estimated probabilistically:

$$
\hat{e}_i = \arg\max_{t} P(done_i = 1 \mid x_{i,1:t}, c_i)
$$

where $x_{i,1:t}$ is the observed speech signal and $c_i$ is contextual dialogue state.

#### Speech Acts and Grounding

Utterances are not just vessels of meaning; they are **actions** ( $a_i = SA(u_i)$ ). Actions span from stating facts (**Constative**) and committing to the future (**Commissive**) to instructing actions (**Directive**).

Furthermore, conversation relies on **Common Ground**. An utterance must be grounded (accepted and understood) by the listener before the dialogue safely moves forward. Listeners use varied grounding methods ordered by strength:

$$
\mathcal{G} = \{\text{continued-attention} \prec \text{next-contribution} \prec \text{acknowledgment} \prec \text{demonstration} \prec \text{display}\}
$$

#### Adjacency Pairs and Subdialogues

Many dialogue acts project expectations onto the next turn, creating **Adjacency Pairs** (e.g., Question $\to$ Answer, Proposal $\to$ Acceptance). However, real conversations frequently feature interruptions or clarifying side-questions, known as subdialogues or side-sequences ( $\sigma_k^{*}$ ):

$$
AP_k = \langle FPP_k, \sigma_k^{*}, SPP_k \rangle
$$

#### Inference and Implicature

Gricean pragmatics show that a speaker's intention may not be stated literally. When asked "What day did you want to travel?", the response "I have a meeting that's from the 12th to the 15th" forces the system to infer the intended travel date using world knowledge ( $K$ ) and the maxim of relevance ( $M_{\mathrm{rel}}$ ):

$$
\hat{h}_i = \arg\max_{h \in \mathcal{H}} P(h \mid u_i, q_{i-1}, K, M_{\mathrm{rel}})
$$

### Dialog Acts and Annotation Corpora

To train machine learning systems, these conversational pragmatics are formalized into **Dialogue Act** labels:

$$
y_i = DA(u_i, c_i) \qquad y_i \in \mathcal{T}
$$

These tagsets ( $\mathcal{T}$ ) balance the needs of domain-specific tasks and generalized dialogue architectures.

#### Domain-Specific Tagsets

For highly constrained goal-oriented agents (e.g., meeting scheduling, restaurant recommendation), semantic slot-value content is bundled straight into the dialogue act class:

$$
y_i = \bigl(a_i, \theta_i\bigr)
$$

This binds conversational function ( $a_i$ ) to task constraints ( $\theta_i$ ):

<details><summary>Code</summary>

```text
# Domain-Specific Dialog Act Examples (HIS Task)
hello(task = find, type = restaurant)
inform(food = Italian, near = museum)
confirm(pricerange = moderate)
request(phone)
inform(name = "Roma", phone = "385456")
bye()
```

</details>

#### Domain-Independent Tagsets (DAMSL)

For general domain-independent dialogue, the standard architecture is **DAMSL** (Dialog Act Markup in Several Layers). DAMSL decomposes an utterance's function into two coordinated layers: what it projects onto the future, and how it responds to the past.

$$
y_i^{\mathrm{DAMSL}} = \bigl(f_i, b_i\bigr)
$$

- **Forward-looking ( $f_i$ )**: Actions driven towards the future discourse (e.g., `STATEMENT`, `INFO-REQUEST`, `CHECK`, `OPENING`).
- **Backward-looking ( $b_i$ )**: Grounding responses to the previous conversational turn (e.g., `ACCEPT`, `ANSWER`, `ACK`, `HOLD`, `SIGNAL-UNDERSTANDING`).

This multi-layer architecture natively captures grounding and overlapping dialogue intents within a single turn:

<details><summary>Code</summary>

```text
# DAMSL Double-Label Annotations
[info-req, ack] A2: And, what day in May did you want to travel?
[assert, answer] C3: OK uh I need to be there for a meeting that’s from the 12th to the 15th.
[check, hold] C7: Uh hmm I don’t think there’s many options for non-stop.
```

</details>

By decomposing sentences into these explicit computational dialogue acts, modern AI and spoken systems transition from passive sentence-processing toward proactive, context-aware interaction.
