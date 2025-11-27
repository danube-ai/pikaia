# Genetic Attention Ablation Summary

This document summarizes all genetic attention mechanisms attempted across different model architectures (encoder-only and decoder-only models). Each section represents a different iteration of the genetic attention concept, with varying approaches to incorporating evolutionary principles into attention computation.

## Overview

Genetic Attention mechanisms aim to replace traditional query-key similarity-based attention with biologically-inspired evolutionary fitness evaluation. The core idea is to modulate attention weights using genetic algorithm principles, where "genes" (feature dimensions) compete based on their evolutionary fitness, and "organisms" (sequence positions) interact through fitness-modulated attention.

## Attempt 1: Decoder-Only Genetic Attention (First Attempt)

### Multi-Head Genetic Attention (MGA) - Value-Only Projections

Multi-Head Genetic Attention (MGA) is a novel attention mechanism that applies genetic sorting principles to modulate attention weights. Unlike traditional attention mechanisms that compute similarities between queries and keys, MGA treats attention as an evolutionary process where token interactions are determined by genetic fitness scores.

#### Genetic Fitness Computation

The core innovation of MGA lies in its fitness computation, which uses a single-GEMM masked weighted aggregation to compute gene fitness scores across sequence positions.

Given input embeddings $X \in \mathbb{R}^{(B \times T \times D)}$, where $B$ is batch size, $T$ is sequence length, $D$ is embedding dimension, and $h$ is the number of heads:

The process follows:

1. **Value Projection**: Project inputs to Values (V) only:
   - $V = XW^V \in \mathbb{R}^{(B \times T \times D)}$
   - Reshape to heads: $V \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$
   - Where $d_{\text{head}} = \frac{D}{h}$

2. **Normalization**: Scale V to [0,1] range globally:
   - $V_{\text{min}} = \min(V)$, $V_{\text{max}} = \max(V)$
   - $V_{\text{scaled}} = \frac{V - V_{\text{min}}}{V_{\text{max}} - V_{\text{min}} + \epsilon}$

3. **Genetic Fitness Scores**: Compute fitness scores using masked weighted aggregation:
   - Create attention mask $M \in \mathbb{R}^{(T \times T)}$ (causal or sliding window)
   - Compute gene fitness $\gamma \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$ using single-GEMM:
     - $\gamma = \text{compute\_genetic\_fitness\_scores}(V_{\text{scaled}}, M)$
   - This involves masked matrix multiplication and weighted aggregation to compute evolutionary fitness

4. **Organism Fitness**: Compute pairwise fitness between all token pairs:
   - $\phi_{ij} = V_{\text{scaled}}[j] \cdot \gamma[i] \in \mathbb{R}^{(B \times h \times T \times T)}$
   - Apply masking to $\phi$ for invalid positions (future tokens)

5. **Attention Weights**: Apply softmax to organism fitness:
   - $\text{weights} = \text{softmax}(\phi) \in \mathbb{R}^{(B \times h \times T \times T)}$

6. **Weighted Sum**: Compute final output:
   - $\text{output} = \text{weights} \cdot V \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$

#### Key Differences from Standard Attention (Attempt 1)

- **Value-Only Projections**: Only computes Value projections, eliminating Query-Key interactions
- **Genetic Fitness Computation**: Uses evolutionary principles with GEMM-based fitness scoring
- **Direct Fitness Weighting**: Attention weights are derived from genetic fitness rather than QK similarities
- **Normalization**: Global min-max scaling ensures stable fitness computation

#### Variants and Optimizations (Attempt 1)

MGA supports several advanced optimizations:

- **Grouped-Query Attention (GQA)**: Uses fewer value heads ($h_v < h$) to reduce memory bandwidth
- **Multi-Head Latent Attention (MLA)**: Compresses inputs into a low-rank latent space before projection
- **Sliding Window Attention (SWA)**: Restricts attention to a fixed window for efficiency

This mechanism offers a biologically inspired alternative to standard attention, potentially capturing different contextual relationships through evolutionary fitness evaluation.

## Attempt 2: Decoder-Only Genetic Attention (Second Attempt)

### Multi-Head Genetic Attention (MGA) - QK-Based Genetic Matrix

Multi-Head Genetic Attention (MGA) is a novel attention mechanism that applies genetic sorting principles to modulate attention weights. Unlike traditional attention mechanisms that compute similarities between queries and keys, MGA treats attention as an evolutionary process where token interactions are determined by genetic fitness scores.

Given input embeddings $X \in \mathbb{R}^{(B \times T \times D)}$, where $B$ is batch size, $T$ is sequence length, $D$ is embedding dimension, and $h$ is the number of heads:

The process follows:

1. **Multi-Head Projections**: Project inputs using the same Q/K projection infrastructure as standard attention:

   - $O' = \text{RMSNorm}(XW^Q) \in \mathbb{R}^{(B \times T \times D)}$ (equivalent to Q)
   - $G' = \text{RMSNorm}(XW^K) \in \mathbb{R}^{(B \times T \times D)}$ (equivalent to Q)
   - $V = XW^V \in \mathbb{R}^{(B \times T \times D)}$

   Where $W^Q, W^V \in \mathbb{R}^{(D \times D)}$ are learned projection matrices.

2. **Reshape for Heads**: Split projections across $h$ attention heads:

   - $O', G', V \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$ where $d_{\text{head}} = D/h$

3. **Per-Head Genetic Matrix**: Compute pairwise genetic interactions for each head:

   - $\text{genetic\_matrix} = \frac{O'G'^T}{\sqrt{d_{\text{head}}}} \in \mathbb{R}^{(B \times h \times T \times T)}$
   - Apply causal attention mask per head

4. **Gene Fitness Scores**: Compute evolutionary fitness for each token position and head:

   - $\gamma = \sum_{j} \text{genetic\_matrix}[:, :, :, j] \in \mathbb{R}^{(B \times h \times T)}$
   - Where the sum is over valid (unmasked) positions $j$

5. **Fitness-Weighted Attention**: Apply fitness scores to modulate attention:

   - $\text{modulated} = \gamma.unsqueeze(-1) \odot \text{genetic\_matrix} \in \mathbb{R}^{(B \times h \times T \times T)}$
   - $\text{weights} = \text{softmax}(\text{modulated}) \in \mathbb{R}^{(B \times h \times T \times T)}$
   - $\text{output} = \text{weights} \cdot V \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$

6. **Concatenation and Output Projection**:

   - Concatenate head outputs: $\mathbb{R}^{(B \times T \times D)}$
   - Apply final projection: $W^O \in \mathbb{R}^{(D \times D)}$

#### Key Differences from Standard Attention (Attempt 2)

- **Genetic Projections**: Uses specialized $O'$ and $G'$ projections instead of $Q$ and $K$
- **Fitness Modulation**: Attention weights are modulated by genetic fitness scores
- **Simplified Computation**: Avoids complex GEMM operations for fitness computation
- **Evolutionary Interpretation**: Token interactions modeled as genetic selection

#### Variants and Optimizations (Attempt 2)

MGA supports several advanced optimizations:

- **Grouped-Query Attention (GQA)**: Uses fewer value heads ($h_v < h$) to reduce memory bandwidth
- **Multi-Head Latent Attention (MLA)**: Compresses inputs into a low-rank latent space before projection
- **Sliding Window Attention (SWA)**: Restricts attention to a fixed window for efficiency

This mechanism offers a biologically inspired alternative to standard attention, potentially capturing different contextual relationships through evolutionary fitness evaluation.

## Attempt 3: Decoder-Only Genetic Attention (Third Attempt)

### Multi-Head Genetic Attention (MGA) - Classical MHA with Genetic Modulation

Multi-Head Genetic Attention (MGA) is a novel attention mechanism that follows the classical Multi-Head Attention (MHA) pipeline but applies genetic sorting to modulate attention weights based on evolutionary fitness principles.

Given input embeddings $X \in \mathbb{R}^{(B \times T \times D)}$, where $B$ is batch size, $T$ is sequence length, $D$ is embedding dimension, and $h$ is the number of heads:

The process follows classical MHA with genetic modulation:

1. **Projections**: Project inputs to Queries (Q), Keys (K), and Values (V):
   - $Q = XW^Q \in \mathbb{R}^{(B \times T \times D)}$
   - $K = XW^K \in \mathbb{R}^{(B \times T \times D)}$
   - $V = XW^V \in \mathbb{R}^{(B \times T \times D)}$
   - Reshape to heads: $Q, K, V \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$

2. **QK RMS Normalization**: Always apply RMS normalization to stabilize training:
   - $Q' = \text{RMSNorm}(Q)$
   - $K' = \text{RMSNorm}(K)$

3. **Attention Scores**: Compute scaled dot-product attention within sliding window:
   - $\text{scores} = \frac{Q' K'^T}{\sqrt{d_{\text{head}}}} \in \mathbb{R}^{(B \times h \times T \times T)}$
   - Apply causal masking and windowing: $\text{scores}_{ij} = -\infty$ if $j > i$ or $|i-j| > w$

4. **Genetic Fitness Evaluation**: Apply genetic sorting to the attention matrix:
   - Scale scores to [0,1]: $\text{scaled} = \sigma(\text{scores})$ where $\sigma$ is sigmoid
   - Compute gene fitness per query token: $\text{fitness}_i = f(\text{scaled}_{i,:})$
   - Where $f$ implements genetic sorting based on evolutionary principles

5. **Genetic Weighting**: Modulate attention scores by fitness:
   - $\text{weighted}_{ij} = \text{fitness}_i \cdot \text{scores}_{ij}$

6. **Softmax Normalization**: Apply row-wise softmax:
   - $\text{weights}_{ij} = \frac{\exp(\text{weighted}_{ij})}{\sum_k \exp(\text{weighted}_{ik})}$

7. **Weighted Sum**: Compute final output:
   - $\text{output} = \text{weights} \cdot V$

#### Key Differences from Standard Attention (Attempt 3)

- **Genetic Modulation**: Attention weights are modulated by evolutionary fitness scores
- **Windowed Causal Attention**: Always applies sliding window and causal masking
- **Fitness-Based Weighting**: Uses genetic sorting to determine token importance
- **RMS Normalization**: Always applies QK RMS normalization for stability

#### Variants and Optimizations (Attempt 3)

MGA supports several advanced optimizations:

- **Grouped-Query Attention (GQA)**: Uses fewer value heads ($h_v < h$) to reduce memory bandwidth
- **Multi-Head Latent Attention (MLA)**: Compresses inputs into a low-rank latent space before projection
- **Gated DeltaNet Integration**: Applies gating mechanism to the output

This mechanism offers a biologically inspired alternative to standard attention, potentially capturing different contextual relationships through evolutionary fitness evaluation.

## Attempt 4: Decoder-Only Genetic Attention (Fourth Attempt)

### Multi-Head Genetic Attention (MGA) - Value-Only with GEMM Fitness (Repeat of Attempt 1)

Multi-Head Genetic Attention (MGA) is a novel attention mechanism that applies genetic sorting principles to modulate attention weights. Unlike traditional attention mechanisms that compute similarities between queries and keys, MGA treats attention as an evolutionary process where token interactions are determined by genetic fitness scores.

#### Genetic Fitness Computation (Attempt 4)

The core innovation of MGA lies in its fitness computation, which uses a single-GEMM masked weighted aggregation to compute gene fitness scores across sequence positions.

Given input embeddings $X \in \mathbb{R}^{(B \times T \times D)}$, where $B$ is batch size, $T$ is sequence length, $D$ is embedding dimension, and $h$ is the number of heads:

The process follows:

1. **Value Projection**: Project inputs to Values (V) only:
   - $V = XW^V \in \mathbb{R}^{(B \times T \times D)}$
   - Reshape to heads: $V \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$
   - Where $d_{\text{head}} = \frac{D}{h}$

2. **Normalization**: Scale V to [0,1] range globally:
   - $V_{\text{min}} = \min(V)$, $V_{\text{max}} = \max(V)$
   - $V_{\text{scaled}} = \frac{V - V_{\text{min}}}{V_{\text{max}} - V_{\text{min}} + \epsilon}$

3. **Genetic Fitness Scores**: Compute fitness scores using masked weighted aggregation:
   - Create attention mask $M \in \mathbb{R}^{(T \times T)}$ (causal or sliding window)
   - Compute gene fitness $\gamma \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$ using single-GEMM:
     - $\gamma = \text{compute\_genetic\_fitness\_scores}(V_{\text{scaled}}, M)$
   - This involves masked matrix multiplication and weighted aggregation to compute evolutionary fitness

4. **Organism Fitness**: Compute pairwise fitness between all token pairs:
   - $\phi_{ij} = V_{\text{scaled}}[j] \cdot \gamma[i] \in \mathbb{R}^{(B \times h \times T \times T)}$
   - Apply masking to $\phi$ for invalid positions (future tokens)

5. **Attention Weights**: Apply softmax to organism fitness:
   - $\text{weights} = \text{softmax}(\phi) \in \mathbb{R}^{(B \times h \times T \times T)}$

6. **Weighted Sum**: Compute final output:
   - $\text{output} = \text{weights} \cdot V \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$

#### Key Differences from Standard Attention (Attempt 4)

- **Value-Only Projections**: Only computes Value projections, eliminating Query-Key interactions
- **Genetic Fitness Computation**: Uses evolutionary principles with GEMM-based fitness scoring
- **Direct Fitness Weighting**: Attention weights are derived from genetic fitness rather than QK similarities
- **Normalization**: Global min-max scaling ensures stable fitness computation

#### Variants and Optimizations (Attempt 4)

MGA supports several advanced optimizations:

- **Grouped-Query Attention (GQA)**: Uses fewer value heads ($h_v < h$) to reduce memory bandwidth
- **Multi-Head Latent Attention (MLA)**: Compresses inputs into a low-rank latent space before projection
- **Sliding Window Attention (SWA)**: Restricts attention to a fixed window for efficiency

This mechanism offers a biologically inspired alternative to standard attention, potentially capturing different contextual relationships through evolutionary fitness evaluation.

## Attempt 5: Decoder-Only Genetic Attention (Fifth Attempt)

### Multi-Head Genetic Attention (MGA) - QK-Based with Simplified Fitness (Repeat of Attempt 2)

Multi-Head Genetic Attention (MGA) is a novel attention mechanism that applies genetic sorting principles to modulate attention weights. Unlike traditional attention mechanisms that compute similarities between queries and keys, MGA treats attention as an evolutionary process where token interactions are determined by genetic fitness scores.

Given input embeddings $X \in \mathbb{R}^{(B \times T \times D)}$, where $B$ is batch size, $T$ is sequence length, $D$ is embedding dimension, and $h$ is the number of heads:

The process follows:

1. **Multi-Head Projections**: Project inputs using the same Q/K projection infrastructure as standard attention:

   - $O' = \text{RMSNorm}(XW^Q) \in \mathbb{R}^{(B \times T \times D)}$ (equivalent to Q)
   - $G' = \text{RMSNorm}(XW^K) \in \mathbb{R}^{(B \times T \times D)}$ (equivalent to Q)
   - $V = XW^V \in \mathbb{R}^{(B \times T \times D)}$

   Where $W^Q, W^V \in \mathbb{R}^{(D \times D)}$ are learned projection matrices.

2. **Reshape for Heads**: Split projections across $h$ attention heads:

   - $O', G', V \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$ where $d_{\text{head}} = D/h$

3. **Per-Head Genetic Matrix**: Compute pairwise genetic interactions for each head:

   - $\text{genetic\_matrix} = \frac{O'G'^T}{\sqrt{d_{\text{head}}}} \in \mathbb{R}^{(B \times h \times T \times T)}$
   - Apply causal attention mask per head

4. **Gene Fitness Scores**: Compute evolutionary fitness for each token position and head:

   - $\gamma = \sum_{j} \text{genetic\_matrix}[:, :, :, j] \in \mathbb{R}^{(B \times h \times T)}$
   - Where the sum is over valid (unmasked) positions $j$

5. **Fitness-Weighted Attention**: Apply fitness scores to modulate attention:

   - $\text{modulated} = \gamma.unsqueeze(-1) \odot \text{genetic\_matrix} \in \mathbb{R}^{(B \times h \times T \times T)}$
   - $\text{weights} = \text{softmax}(\text{modulated}) \in \mathbb{R}^{(B \times h \times T \times T)}$
   - $\text{output} = \text{weights} \cdot V \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$

6. **Concatenation and Output Projection**:

   - Concatenate head outputs: $\mathbb{R}^{(B \times T \times D)}$
   - Apply final projection: $W^O \in \mathbb{R}^{(D \times D)}$

#### Key Differences from Standard Attention (Attempt 5)

- **Genetic Projections**: Uses specialized $O'$ and $G'$ projections instead of $Q$ and $K$
- **Fitness Modulation**: Attention weights are modulated by genetic fitness scores
- **Simplified Computation**: Avoids complex GEMM operations for fitness computation
- **Evolutionary Interpretation**: Token interactions modeled as genetic selection

#### Variants and Optimizations (Attempt 5)

MGA supports several advanced optimizations:

- **Grouped-Query Attention (GQA)**: Uses fewer value heads ($h_v < h$) to reduce memory bandwidth
- **Multi-Head Latent Attention (MLA)**: Compresses inputs into a low-rank latent space before projection
- **Sliding Window Attention (SWA)**: Restricts attention to a fixed window for efficiency

This mechanism offers a biologically inspired alternative to standard attention, potentially capturing different contextual relationships through evolutionary fitness evaluation.

## Attempt 6: Encoder-Only Genetic Attention (Final Implementation)

### Multi-Head Genetic Attention (MGA) - Encoder Version

Multi-Head Genetic Attention (MGA) is a novel attention mechanism that applies genetic sorting principles to modulate attention weights. Unlike traditional attention mechanisms that directly weight values based on query-key similarities, MGA incorporates evolutionary fitness scores derived from value projections to modulate the attention mechanism, creating a biologically inspired approach to contextual weighting.

Given input embeddings $X \in \mathbb{R}^{(B \times T \times D)}$, where $B$ is batch size, $T$ is sequence length, $D$ is embedding dimension, and $h$ is the number of heads:

The process follows:

1. **Optional Input Projection (MLA)**: If using Multi-Head Latent Attention, project input to compressed dimension:
   $$ X' = X W^{\text{in}} \in \mathbb{R}^{(B \times T \times d_c)} $$
   Where $d_c$ is the compressed dimension.

2. **Multi-Head Projections**: Project inputs to queries, keys, and values:
   - $Q = \text{reshape}(X' W^Q) \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$
   - $K = \text{reshape}(X' W^K) \in \mathbb{R}^{(B \times h_{kv} \times T \times d_{\text{head}})}$
   - $V = \text{reshape}(X' W^V) \in \mathbb{R}^{(B \times h_{kv} \times T \times d_{\text{head}})}$
   Where $h_{kv} \leq h$ for Grouped-Query Attention (GQA), and $d_{\text{head}} = D/h$.

3. **Grouped Query Handling (GQA)**: If $h_{kv} < h$, repeat K and V across groups:
   $$ K, V = \text{repeat\_interleave}(K, V, \frac{h}{h_{kv}}) \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})} $$

4. **QK Normalization**: Apply RMSNorm to queries and keys:
   $$ Q = \text{RMSNorm}(Q), \quad K = \text{RMSNorm}(K) $$

5. **Genetic Fitness Computation**: Compute gene fitness scores using genetic algorithm formulation:

   a. **Value Normalization**: Apply sigmoid to values:
      $$ V_{\text{norm}} = \sigma(V) \in [0, 1]^{(B \times h \times T \times d_{\text{head}})} $$

   b. **Gene Means**: Compute mean values for each gene (feature dimension) across valid organisms:
      $$ \widetilde{\Phi}_g = \frac{1}{T} \sum_{t=1}^T V_{\text{norm}}[:, :, t, g] \in \mathbb{R}^{(B \times h \times d_{\text{head}})} $$
      (Masked appropriately for padding tokens)

   c. **Gene Fitness**: Compute fitness for each gene using the genetic formulation:
      $$ \gamma_g^{*} = \left( \sum_{s=1}^{d_{\text{head}}} \frac{\widetilde{\Phi}_g + \tfrac{1}{2}}{\widetilde{\Phi}_s + \tfrac{1}{2}} \right)^{-1} \in \mathbb{R}^{(B \times h \times d_{\text{head}})} $$

   d. **Fitness-Weighted Values**: Modulate values by element-wise multiplication with gene fitness:
      $$ V_{\text{genetic}} = V \odot \gamma^{*} $$
      Where $\gamma^{*} \in \mathbb{R}^{(B \times h \times d_{\text{head}})}$ is broadcasted to match $V \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})}$ by repeating across the sequence dimension.

6. **Attention Computation**: Compute standard scaled dot-product attention:
   $$ \text{scores} = \frac{Q K^T}{\sqrt{d_{\text{head}}}} \in \mathbb{R}^{(B \times h \times T \times T)} $$
   $$ \text{weights} = \text{softmax}(\text{scores}) $$
   $$ \text{output} = \text{weights} \cdot V_{\text{genetic}} \in \mathbb{R}^{(B \times h \times T \times d_{\text{head}})} $$

7. **Concatenation and Output Projection**:
   $$ \text{Concat}(\text{output}) \in \mathbb{R}^{(B \times T \times D)} $$
   $$ \text{MGA}(X) = \text{Concat}(\text{output}) W^O \in \mathbb{R}^{(B \times T \times D)} $$

#### Key Differences from Standard Attention (Final Implementation)

- **Genetic Fitness Modulation**: Attention weights are computed from standard QK similarities, but applied to values that have been modulated by evolutionary gene fitness scores
- **Value-Based Fitness**: Fitness computation operates on value projections to determine feature importance through genetic algorithm principles
- **Element-wise Modulation**: Gene fitness scores are applied element-wise to value features, providing fine-grained control over attention contributions
- **Evolutionary Interpretation**: Feature importance is modeled through genetic fitness evaluation, where genes compete based on their average expression across the sequence

#### Variants and Optimizations (Final Implementation)

MGA supports several advanced optimizations:

- **Grouped-Query Attention (GQA)**: Uses fewer value heads ($h_{kv} < h$) to reduce memory bandwidth
- **Multi-Head Latent Attention (MLA)**: Compresses inputs into a low-rank latent space before projection
- **Sliding Window Attention (SWA)**: Restricts attention to a fixed window for efficiency
- **Genetic Disable Option**: Can fall back to standard attention by disabling fitness computation

This mechanism offers a biologically inspired alternative to standard attention, potentially capturing different contextual relationships through evolutionary fitness evaluation.

## Summary of Genetic Attention Evolution

The genetic attention mechanism has evolved through multiple iterations:

1. **Decoder Attempt 1**: Value-only projections with GEMM-based fitness computation for causal attention
2. **Decoder Attempt 2**: QK-based genetic matrix with simplified fitness computation
3. **Decoder Attempt 3**: Classical MHA with genetic sorting applied to attention scores
4. **Decoder Attempt 4**: Repeat of Attempt 1 (value-only with GEMM fitness)
5. **Decoder Attempt 5**: Repeat of Attempt 2 (QK-based with simplified fitness)
6. **Encoder-Only (Final)**: Full MHA pipeline with genetic fitness modulation of values based on evolutionary gene competition

Each attempt explores different ways to integrate evolutionary principles into attention computation, from value modulation to attention score modulation, with varying computational complexity and biological fidelity.
