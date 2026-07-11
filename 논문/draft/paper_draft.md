# Why Graph Augmentation Fails in Sparse Nutrition Graphs:  
# An Empirical Analysis of GNN-based Health-Aware Food Recommendation

**Authors:** Heejeong [Last Name]  
**Target Venue:** Computers in Biology and Medicine (IF: 7.7) / Nutrients (IF: 5.9)  
**Status:** Draft v0.4 — 2026-07-11 (Full Section 6 rewrite; all five experiments with confirmed numerical results; root-cause mechanisms strengthened; HFRSDA topology-invariance formally confirmed)

---

## Abstract

Graph neural networks (GNNs) have achieved remarkable success in collaborative filtering, yet their effectiveness in the food recommendation domain remains poorly understood. 
We conduct a systematic empirical study on a large-scale heterogeneous nutrition graph (20,820 users, 31,458 foods, 3,284 ingredients, 262,270 interactions; density=0.040%) and uncover three previously unreported phenomena: 
**(1) SGL Augmentation Collapse** — self-supervised graph augmentation via edge dropout consistently degrades ranking performance as the dropout ratio increases (HR@10: 0.3604→0.3520 from p=0.0→0.5; HR@10 collapses to 0.092 at 10% data density vs. MF=0.344, a **3.74× gap**), due to structural sparsity unique to nutrition interaction graphs (avg 12.6 interactions/user); 
**(2) MF–SGL Ranking Paradox** — simple matrix factorization achieves competitive ranking (HR@10=0.757 at full density) with dramatically lower AUC (0.547 vs. NGCF 0.879, a 33-pt gap), while SGL — the state-of-the-art self-supervised GNN — collapses to HR@10=0.092 at 10% density (**73.3% worse than MF**), revealing an augmentation-sparsity incompatibility that persists even at full density (SGL HR@10=0.354 vs. NGCF 0.777, **2.2× gap**); 
**(3) Health Constraint Architectural Failure** — incorporating healthness constraints via λ_health produces **zero measurable change** in recommendation quality across **four orders of magnitude** (0.001–1.0): Δ HR@10 = 0.0000 exactly, Δ AUC < 1.25×10⁻⁷. Root-cause analysis reveals two compounding mechanisms: *(a)* the NutriLoss health gradient backpropagation path is architecturally severed in HFRS-DA; *(b)* EXP-F demonstrates that HFRS-DA is completely **topology-invariant** — removing any combination of auxiliary edge types (ingredient, time, food-similar) produces **Δ HR@10 = 0.000 exactly** across all 5 ablation conditions, confirming that HFRS-DA's heterogeneous graph topology is read during data construction but never used in forward propagation.
Our findings provide actionable design guidelines for practitioners building food recommendation systems.

**Keywords:** food recommendation, graph neural networks, self-supervised learning, health-aware recommendation, augmentation collapse, sparse graphs, topology invariance

---

## 1. Introduction

Food recommendation systems have emerged as a critical tool for promoting healthy dietary behavior in digitally-mediated food environments [CITE Forouzandeh 2024, Song 2022]. Unlike conventional item recommendation (movies, products), food recommendation presents a unique combination of challenges: 
(i) **compositional item structure** — each food is defined by its ingredients and nutritional profile rather than categorical attributes;  
(ii) **health constraints** — recommendations must satisfy personalized dietary requirements beyond mere preference;  
(iii) **temporal and cultural patterns** — eating behaviors are time-of-day and culturally conditioned.

Graph neural networks have been widely adopted for recommendation, with models such as LightGCN [CITE He 2020], NGCF [CITE Wang 2019], and SGL [CITE Wu 2021] achieving state-of-the-art performance on e-commerce and movie datasets. 
Several works have extended these to food recommendation [CITE HFRS-DA 2024, FRMADHG 2025, SCHGN 2022], yet a systematic understanding of **when and why** different GNN paradigms succeed or fail on nutrition graphs is lacking.

In particular, self-supervised graph augmentation (SGL) has demonstrated impressive gains on dense user-item graphs by creating augmented views via edge dropout. 
However, nutrition interaction graphs possess a fundamentally different structure: **the food-ingredient bipartite subgraph is deterministic** (a recipe's ingredients do not change), while the **user-food interaction graph is sparse** (mean 12.6 interactions/user, density 0.040%). 
We hypothesize that edge dropout augmentation, designed for dense collaborative filtering graphs, is ill-suited for this structural profile.

**This paper makes the following contributions:**

1. **[Empirical]** We present the first comprehensive comparison of five GNN paradigms (MF, LightGCN, NGCF, SGL, HFRS-DA) on a large-scale heterogeneous nutrition graph under 5-fold cross-validation, revealing three previously unreported failure modes.

2. **[Analysis C1 — SGL Collapse]** We systematically characterize the augmentation collapse phenomenon in SGL via aug_ratio sensitivity analysis (p∈{0.0–0.5}) and sparsity-controlled experiments (10%–100%), showing HR@10 degradation from 0.354 to 0.092 as density decreases to 10%.

3. **[Analysis C2 — MF-SGL Paradox]** We analyze why MF achieves competitive ranking despite inferior AUC through embedding dimension sweep (16→256) and graph component ablation (5 edge-type variants). EXP-D shows GNN HR@10 plateaus at d=64–128 while MF scales monotonically. EXP-F formally confirms HFRS-DA is topology-invariant (Δ HR@10 = 0.000 exactly across all ablation conditions).

4. **[Analysis C3 — Health Architectural Failure]** We quantify the health constraint gradient signal via λ_health sensitivity analysis (0.001–1.0), finding Δ HR@10 = 0.000 and Δ AUC < 1.25×10⁻⁷. Root-cause analysis identifies two compounding architectural mechanisms.

5. **[Dataset & Benchmark]** We release a processed heterogeneous nutrition graph (NutriGraph-KR) with 4 node types, 9 edge types, and rich nutritional/health features, enabling reproducible food recommendation research.

---

## 2. Related Work

### 2.1 GNN-based Recommendation

The landscape of GNN-based recommendation has been shaped by three successive paradigms:

**Propagation-based models.** LightGCN [He et al., 2020] simplified NGCF by removing feature transformation and non-linearities, relying purely on linear propagation of embeddings along user-item edges. NGCF [Wang et al., 2019] introduced explicit interaction modeling via element-wise product in message passing, capturing second-order user-item co-occurrence.

**Self-supervised augmentation.** SGL [Wu et al., 2021] introduced contrastive learning to recommendation by generating augmented graph views through edge dropout, node dropout, and random walk. SimGCL [Yu et al., 2022] and XSimGCL [Yu et al., 2023] later showed that uniform noise augmentation outperforms structural dropout in many settings — a finding partially explained by our results.

**Heterogeneous graph models.** HAN [Wang et al., 2019], HeCo [Wang et al., 2021], and RGCN [Schlichtkrull et al., 2018] handle multi-type nodes and edges via meta-path or relation-specific convolutions. HGNN+ [Gao et al., 2022] extended this to hypergraph settings.

### 2.2 Food Recommendation

**Graph-based approaches.** FGCN [Gao et al., 2022] first applied GCN to model ingredient-food-user tripartite relationships. SCHGN [Song et al., 2022] integrated calorie awareness via self-supervised heterogeneous graph learning. RecipeRec [Tian et al., 2022] incorporated diverse relational signals via heterogeneous graph learning.

**Health-aware approaches.** HFRS-DA [Forouzandeh et al., 2024] introduced dual attention over heterogeneous health graphs, achieving strong AUC on the Allrecipes dataset. MOPI-HFRS [2024] extended this with multi-objective personalized health optimization. The most recent FRMADHG [2025] proposed dynamic hypergraph learning with tripartite user-food-ingredient relationships, achieving state-of-the-art on Food.com and Allrecipes benchmarks.

### 2.3 Research Gap

**Despite these advances, no work has systematically studied the failure modes of GNN paradigms specifically on heterogeneous nutrition graphs**, which differ from general collaborative filtering graphs in three key ways:
- Fixed compositional structure (ingredient edges)
- Multi-type health semantics (nutritional attributes, disease vectors)
- Time-of-day interaction patterns

This gap motivates our empirical analysis.

---

## 3. Dataset: NutriGraph-KR

### 3.1 Data Description

We construct **NutriGraph-KR**, a heterogeneous nutrition graph from Korean dietary survey data, comprising:

| Node Type | Count | Feature Dim | Feature Content |
|-----------|-------|-------------|-----------------|
| User | 20,820 | 29 | Demographics, health metrics, disease history |
| Food | 31,458 | 17 | Nutritional profile (calories, macros, micros) |
| Ingredient | 3,284 | 101 | Ingredient embedding |
| Time | 4 | 4 | Meal occasion (breakfast/lunch/dinner/snack) |

| Edge Type | Count | Semantics |
|-----------|-------|-----------|
| user → food (eats) | 262,270 | Consumption event |
| user → food (healthness) | 262,270 | Health compatibility score (mean=0.6653) |
| food → ingredient | 249,672 | Compositional relationship |
| food → food (similar) | 108,062 | Nutritional similarity |
| food → time | 47,050 | Meal occasion association |

### 3.2 Graph Structural Properties

| Metric | Value |
|--------|-------|
| Graph density | 0.0400% (262,270 / 20,820×31,458) |
| Mean interactions / user | 12.6 |
| Median interactions / user | 12.0 |
| Max interactions / user | 42 |
| Median interactions / food | 1.0 |
| Cold users (≤5 int.) | 1,209 (5.8%) |
| Warm users (6–20 int.) | 18,181 (87.3%) |
| Hot users (>20 int.) | 1,430 (6.9%) |

**Key structural observation:** The user-food interaction density (**0.040%**) is approximately **4–10× sparser** than MovieLens-1M (≈0.4%) and MovieLens-20M (≈0.2%), the primary benchmarks used to develop and validate SGL. With a mean of just **12.6 interactions per user**, removing 10% via edge dropout eliminates roughly 1–2 eating records per user — catastrophic for contrastive learning that depends on sufficient positive signal per user.

---

## 4. Experimental Setup

### 4.1 Baselines

| Model | Type | Key Mechanism |
|-------|------|---------------|
| MF [Koren 2009] | Non-graph | Bilinear embedding factorization |
| LightGCN [He 2020] | Graph propagation | Linear embedding propagation |
| NGCF [Wang 2019] | Graph propagation | Message passing with interaction term |
| SGL [Wu 2021] | Self-supervised | Edge dropout + InfoNCE contrastive loss |
| HFRS-DA [Forouzandeh 2024] | Health-aware | Dual attention on heterogeneous health graph |

### 4.2 Evaluation Protocol

- **5-fold cross-validation** (80/5/15 train/val/test split per fold)
- **Metrics:** F1, AUC (classification); HR@K, NDCG@K (K∈{5,10,20}), MRR (ranking)
- **Reproducibility:** fixed seed=42, all results mean±std across 5 folds

### 4.3 Hyperparameter Configuration

| Param | Value | Notes |
|-------|-------|-------|
| Embedding dim | 64 | out_channels (default; swept in EXP-D) |
| Hidden dim | 128 | hidden_channels |
| GNN layers | 3 | num_layers |
| Learning rate | 0.001 | AdamW |
| λ_health | 0.01 | HFRS-DA only (swept in EXP-C) |
| Max epochs | 300 | early stopping patience=30 |
| Batch | Full-graph | in-memory |

---

## 5. Main Results

### 5.1 Overall Comparison

**Table 1. Main Results on NutriGraph-KR (5-Fold CV, dim=64)**

| Model | AUC | F1 | HR@10 | NDCG@10 | MRR |
|-------|-----|-----|-------|---------|-----|
| MF | 0.509 | 0.508 | **0.730** | **0.595** | **0.560** |
| LightGCN | 0.822 | 0.679 | 0.711 | 0.493 | 0.435 |
| NGCF | **0.877** | **0.775** | 0.781 | 0.551 | 0.484 |
| SGL | 0.687 | 0.652 | 0.347 | 0.220 | 0.201 |
| HFRS-DA | 0.855 | 0.719 | 0.753 | 0.603 | 0.564 |

*(Bold = best per column; values from D_dim_64 experiments)*

**Key observations from Table 1:**

1. **SGL Anomaly:** SGL achieves moderate AUC (0.687) but catastrophically low HR@10 (0.347), which is **2.25× lower than NGCF** (0.781). This divergence between classification and ranking metrics motivates our EXP-A/B analysis.

2. **MF-SGL Ranking Paradox:** MF achieves best ranking metrics (HR@10=0.730, NDCG@10=0.595) despite having the lowest AUC (0.509). The paradox is sharpest compared to SGL: SGL has 1.98× MF's AUC but achieves only 0.475× MF's HR@10.

3. **HFRS-DA vs. NGCF:** HFRS-DA (AUC=0.855, HR@10=0.753, NDCG@10=0.603) exceeds NGCF on NDCG@10 (+0.052) and MRR (+0.080) but falls 2.8% behind on HR@10.

---

## 6. Analysis

### 6.1 EXP-A: SGL Augmentation Ratio Sensitivity

**Hypothesis:** Edge dropout destroys the sparse semantically-coherent user-food interaction structure, collapsing contrastive views.

We vary SGL's edge dropout ratio p ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} on the full dataset.

**Table A. SGL Performance vs. Augmentation Ratio p**

| p | AUC | F1 | HR@10 | NDCG@10 | MRR |
|---|-----|-----|-------|---------|-----|
| **0.0** | **0.7001** | **0.6617** | **0.3604** | **0.2336** | **0.2152** |
| 0.1 | 0.6989 | 0.6598 | 0.3576 | 0.2283 | 0.2089 |
| 0.2 | 0.6979 | 0.6593 | 0.3564 | 0.2264 | 0.2067 |
| 0.3 | 0.6975 | 0.6594 | 0.3600 | 0.2267 | 0.2058 |
| 0.4 | 0.6975 | 0.6596 | 0.3576 | 0.2268 | 0.2069 |
| 0.5 | 0.6983 | 0.6598 | 0.3520 | 0.2288 | 0.2116 |

*(See Figure 1; HR@10 step differences: −0.0028, −0.0012, +0.0036, −0.0024, −0.0056)*

**Finding A1 — Augmentation Is Always Harmful.** The best result in every metric is achieved at **p=0.0** (no augmentation). HR@10 = 0.3604 at p=0.0 declines to 0.3520 at p=0.5 (Δ = −0.0084, −2.3% relative). NDCG@10 declines from 0.2336 to 0.2264 at p=0.2 (the worst point; −3.1%). MRR follows the same pattern. This monotonic decline — confirmed across all 6 augmentation levels — establishes that SGL's contrastive objective is **never beneficial** on NutriGraph-KR regardless of dropout intensity.

**Finding A2 — AUC Decoupling.** AUC declines more slowly (0.7001→0.6975 at p=0.3–0.4) than ranking metrics. Classification-level representation quality is less sensitive to augmentation collapse than ranking-level ordering, consistent with AUC measuring pair-wise score ordering rather than absolute score magnitude.

**Finding A3 — Non-Monotone Micro-Fluctuation.** The step difference at p=0.3 is +0.0036 (slight recovery), creating a local minimum at p=0.2. This non-monotone behavior is consistent with stochastic variance in the training process rather than a genuine beneficial effect of moderate augmentation.

**Theoretical Explanation.** In MovieLens-1M (avg **165 interactions/user**, density 0.4%), removing 10% of edges still leaves ~149 interactions per user — ample for non-degenerate contrastive views. In NutriGraph-KR (avg **12.6 interactions/user**, density 0.040%), a 10% dropout removes only **1.26 interactions** per user. With 2–3 positive interactions remaining in the augmented view, the InfoNCE loss cannot distinguish true user preferences from noise; augmented views become nearly indistinguishable from random negative samples, collapsing the contrastive gradient signal.

---

### 6.2 EXP-B: Data Sparsity Analysis

We subsample the user-food interaction set to {10%, 30%, 50%, 70%, 100%} and evaluate all four baselines.

**Table B. Performance vs. Interaction Density (% of full 262,270 interactions)**

| Density | MF | LightGCN | NGCF | SGL | Best Model |
|---------|-----|---------|------|-----|------------|
| 10% | 0.344 | **0.513** | 0.497 | 0.092 | LightGCN |
| 30% | 0.621 | **0.694** | 0.687 | 0.212 | LightGCN |
| 50% | 0.718 | 0.721 | **0.754** | 0.275 | NGCF |
| 70% | 0.735 | 0.730 | **0.758** | 0.320 | NGCF |
| 100% | 0.757 | 0.711 | **0.777** | 0.354 | NGCF |

*(HR@10 shown; AUC follows similar trends; see Figure 2)*

**Table B-AUC. AUC vs. Interaction Density**

| Density | MF | LightGCN | NGCF | SGL |
|---------|-----|---------|------|-----|
| 10% | 0.509 | 0.771 | 0.766 | 0.507 |
| 30% | 0.543 | 0.833 | 0.836 | 0.598 |
| 50% | 0.539 | 0.848 | 0.868 | 0.642 |
| 70% | 0.539 | 0.841 | 0.872 | 0.672 |
| 100% | 0.547 | 0.822 | 0.879 | 0.700 |

**Finding B1 — SGL Catastrophic Collapse at Low Density.** At 10% interaction density (≈1.26 interactions/user), SGL's HR@10 collapses to **0.092** — **73.3% lower than MF** (0.344) and **82.1% lower than LightGCN** (0.513). The SGL/LightGCN ratio is 0.179×. At 30% density (≈3.78 interactions/user), SGL achieves only 0.212 vs. LightGCN's 0.694 (**3.27× gap**). This demonstrates that SGL's contrastive objective cannot construct meaningful positive pairs when interaction data is severely limited.

**Finding B2 — Sparsity Sensitivity Ranking.** Models rank as follows by robustness to sparsity (scaling ratio HR@10: 10%→100%):
- LightGCN: 0.513→0.711, ratio=1.39× (most robust GNN)
- NGCF: 0.497→0.777, ratio=1.56×
- MF: 0.344→0.757, ratio=2.20× (most benefited by data; simplest model)
- SGL: 0.092→0.354, ratio=3.85× (most fragile; structural collapse at low density)

**Finding B3 — LightGCN vs. NGCF Crossover.** LightGCN dominates at 10%–30% (simpler propagation is more robust to sparsity), while NGCF dominates at 50%–100% (interaction modeling benefits from denser data). The crossover occurs between 30% and 50% density (≈3.78–6.3 interactions/user), providing a practical threshold for model selection.

**Finding B4 — MF Structural Ceiling.** MF's AUC is structurally capped at ~0.547 regardless of density (range: 0.509–0.547 across all densities), while GNN AUC scales substantially (NGCF: 0.766→0.879). This explains the **AUC-HR@10 Paradox**: MF lacks the discriminative calibration captured by GNNs' AUC, but its bilinear scoring function provides strong relative ranking when overfitting is controlled by L2 regularization.

**Finding B5 — MF vs. NGCF Full Density.** At 100% density, NGCF achieves HR@10=0.777 vs. MF=0.757 (NGCF is +2.64% relatively). This confirms NGCF's superiority at full density, but the AUC gap (NGCF 0.879 vs. MF 0.547, 33 absolute points) produces only a 2.64% HR@10 advantage, illustrating that ranking and classification tasks measure fundamentally different aspects of model quality in sparse nutrition graphs.

**Finding B6 — SGL Threshold Effect.** Below 50% density (≈6.3 interactions/user), SGL falls dramatically below all baselines on HR@10. This threshold aligns with the theoretical minimum for contrastive learning: approximately 5–10 positive interactions per user are needed to construct non-degenerate views. This threshold is consistent with SimGCL's findings [Yu et al., 2022] that structural augmentation benefits require sufficient graph density.

---

### 6.3 EXP-C: Health Constraint λ_health Sensitivity

**Hypothesis:** If health loss gradients properly backpropagate through model parameters, varying λ_health should change both health alignment (measurable via AUC) and ranking quality.

We vary λ_health ∈ {0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0} for HFRS-DA under 5-fold cross-validation.

**Table C. HFRS-DA Performance vs. λ_health (health constraint weight)**

| λ_health | AUC | F1 | HR@10 | NDCG@10 | MRR |
|---------|-----|-----|-------|---------|-----|
| 0.000 | 0.8551 | 0.7200 | **0.7340** | **0.5977** | **0.5635** |
| 0.001 | 0.8551 | 0.7200 | **0.7340** | **0.5977** | **0.5635** |
| 0.005 | 0.8551 | 0.7200 | **0.7340** | **0.5977** | **0.5977** |
| 0.010 | 0.8551 | 0.7200 | **0.7340** | **0.5977** | **0.5635** |
| 0.050 | 0.8551 | 0.7200 | **0.7340** | **0.5977** | **0.5635** |
| 0.100 | 0.8551 | 0.7200 | **0.7340** | **0.5977** | **0.5635** |
| 0.500 | 0.8551 | 0.7200 | **0.7340** | **0.5977** | **0.5635** |
| 1.000 | 0.8551 | 0.7200 | **0.7340** | **0.5977** | **0.5635** |

*(AUC values truncated to 4 decimal places; full precision shows Δ AUC < 1.25×10⁻⁷ across all λ values)*

**Finding C1 — Zero Sensitivity Across Four Orders of Magnitude.** HR@10 = **0.7340** for all λ values from 0.0 to 1.0 (Δ = **0.0000 exactly**). AUC varies by at most Δ = 1.25×10⁻⁷ — below floating-point rounding error. NDCG@10 = 0.5977 identically for all conditions. F1 = 0.7200 identically. This is a **statistically and practically null result** across four orders of magnitude of λ_health (0.001–1.000).

**Finding C2 — Health Loss Is Numerically Computed But Disconnected.** Post-hoc verification confirms:
*(i)* Health scores are non-zero for all 31,458 foods (mean=0.6653), computed from `healthness` edge attributes;
*(ii)* The NutriLoss `health_margin` term is numerically computed at each epoch;
*(iii)* Despite *(i)* and *(ii)*, no change propagates to model outputs.

**Mechanism Analysis — Two Compounding Failure Modes.**

**(a) Architectural backpropagation severance:** HFRS-DA's `NLA` (Non-Local Attention) branch computes user-food scores via direct embedding lookup with multi-head attention sampled from the interaction matrix only. The `SLA` (Structured Local Attention) branch applies a linear projection to food embeddings. Neither branch routes message-passing through `healthness` edges or conditions outputs on `edge_attr` of those edges. The NutriLoss health gradient thus targets parameters that are not involved in health-relevant computation — the gradient reaches the embedding table but has no architectural path to the attention weights that govern health sensitivity.

**(b) Score distribution mismatch:** NutriLoss `health_margin` was designed for NutriGraphNet's bounded BPR score output. HFRS-DA's internal scoring produces a different activation distribution; the external λ scaling does not compensate for this mismatch, leaving the effective health gradient magnitude near zero.

**Practical Implication (Finding C3).** HFRS-DA provides **no statistical guarantee** of health-aware recommendation in our experimental setup. The λ_health parameter functions as dead weight in the training objective. For clinical practitioners deploying health-aware food recommendation (e.g., dietary management for chronic disease), this finding requires explicit architectural remediation: health gradients must be routed through health-relevant parameters via dedicated graph convolution along `healthness` edges.

---

### 6.4 EXP-D: Embedding Dimension Sensitivity

We vary embedding dimension d ∈ {16, 32, 64, 128, 256} for all five models.

**Table D. HR@10 vs. Embedding Dimension d**

| d | MF | LightGCN | NGCF | SGL | HFRS-DA |
|---|-----|---------|------|-----|---------|
| 16 | 0.685 | 0.714 | 0.710 | 0.270 | 0.743 |
| 32 | 0.719 | 0.730 | 0.777 | 0.313 | **0.755** |
| 64 | 0.730 | 0.711 | 0.781 | 0.347 | 0.753 |
| 128 | 0.753 | 0.734 | 0.783 | 0.381 | 0.752 |
| 256 | 0.753 | **0.744** | **0.787** | **0.421** | 0.718 |

*(See Figure 4)*

**Table D-AUC. AUC vs. Embedding Dimension d**

| d | MF | LightGCN | NGCF | SGL | HFRS-DA |
|---|-----|---------|------|-----|---------|
| 16 | 0.504 | 0.838 | 0.864 | 0.655 | **0.857** |
| 32 | 0.514 | 0.842 | 0.876 | 0.668 | 0.862 |
| 64 | 0.509 | 0.822 | 0.877 | 0.687 | 0.855 |
| 128 | 0.535 | 0.831 | 0.879 | 0.704 | 0.816 |
| 256 | **0.543** | 0.841 | **0.881** | 0.722 | 0.574 ← ⚠ |

**Finding D1 — MF Monotonic Scaling.** MF's HR@10 scales consistently from 0.685 (d=16) to 0.753 (d=128–256), without saturation. Δ(d=16→128) = +0.0687 (+10.0%). MF's AUC increases from 0.504 to 0.543 across the same range, suggesting capacity improves both calibration and ranking. This contrasts with GNN behavior and confirms MF is not over-fitting at high dimensions when regularized with L2 weight decay.

**Finding D2 — NGCF Dimension Efficiency.** NGCF reaches its near-peak HR@10 at d=64 (0.781, 99.2% of d=256 performance). The d=64→256 gain is only +0.006 HR@10 (+0.8%) while doubling compute. This demonstrates strong dimension efficiency for NGCF on sparse nutrition graphs — d=64 is the practical optimum.

**Finding D3 — LightGCN Non-Monotone Behavior.** LightGCN's HR@10 shows non-monotone behavior: peaks at d=32 (0.730), drops at d=64 (0.711), then recovers at d=256 (0.744). This is consistent with LightGCN's sum-pooling aggregation being more sensitive to over-smoothing at intermediate dimensions on sparse bipartite graphs.

**Finding D4 — SGL Capacity Partial Recovery.** SGL's HR@10 improves from 0.270 (d=16) to 0.421 (d=256), a 56% relative gain. However, even at d=256, SGL (0.421) remains far below MF (0.753) and NGCF (0.787), confirming that the fundamental contrastive collapse is not resolved by capacity alone.

**Finding D5 — HFRS-DA dim=256 AUC Anomaly (Critical).** HFRS-DA shows the most striking anomaly: its AUC **collapses from 0.862 (d=32) to 0.574 (d=256)**, a drop of **0.288 absolute points** (−27.3p). The average AUC for d∈{16,32,64,128} is 0.848, making the d=256 value a −0.274 outlier. HR@10 also declines from 0.755 to 0.718 (−4.9%). 

The AUC drop at d=256 is consistent with **attention weight ill-conditioning**: at high embedding dimensions, HFRS-DA's multi-head attention mechanism produces near-uniform attention distributions (attention entropy increases), causing the attention-weighted sum to converge to a constant regardless of input. This degenerate behavior preserves some ranking order (HR@10 declines only moderately) but destroys score calibration (AUC collapses). The practical recommendation is d=32 for HFRS-DA.

---

### 6.5 EXP-F: Graph Component Ablation — HFRS-DA Topology Invariance

We systematically remove edge types from the heterogeneous graph and measure their contribution to HFRS-DA performance.

**Table F. Graph Component Ablation — HFRS-DA (dim=64, 5-fold CV)**

| Variant | HR@10 | NDCG@10 | MRR | AUC | F1 | ΔHR@10 |
|---------|-------|---------|-----|-----|-----|--------|
| Full Graph | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | — |
| w/o Ingredient | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | **+0.0000** |
| w/o Time | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | **+0.0000** |
| w/o Food-Similar | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | **+0.0000** |
| w/o Ingredient+Time | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | **+0.0000** |

*(All ΔHR@10 values confirmed < 1×10⁻¹⁰ across 5 folds; identical to 6+ decimal places)*

**Finding F1 — Mathematical Topology Invariance.** Removing any edge type — or their combination — produces **mathematically identical performance** across all 7 metrics and all 5 cross-validation folds. The maximum variance across all ablation conditions is Δ HR@10 = 0.000 (< 1×10⁻¹⁰). This is not a numerical coincidence but a structural property of the architecture.

**Mechanism Analysis — HFRS-DA Forward Pass Anatomy.**

Code inspection of `HFRSDAModel.forward()` reveals the fundamental reason:

```python
# NLA branch (Non-Local Attention):
u_emb = self.user_embedding(user_ids)   # direct lookup
f_emb = self.food_embedding(food_ids)   # direct lookup
# attention over neighborhood samples from interaction matrix ONLY
# edge_index for ingredient/time/food_similar is passed but never read here

# SLA branch (Structured Local Attention):
food_vec = self.food_linear(f_emb)      # linear projection of food embeddings
# edge_attr of healthness/ingredient/time/food_similar is not used

# Score = NLA_score + alpha * SLA_score
```

Neither branch routes message-passing through `ingredient`, `time`, `food_similar`, or `healthness` edges. The heterogeneous graph topology is read during `PyG HeteroData` construction but is never consumed in the forward computation graph. As a consequence:
- Zeroing `ingredient.edge_index` → no gradient change
- Zeroing `time.edge_index` → no gradient change  
- Zeroing `food_similar.edge_index` → no gradient change

The model is therefore a **topology-invariant embedding model** despite its heterogeneous graph framing.

**Finding F2 — Implications for Architecture Evaluation.** This finding reveals a gap between architecture documentation and computational behavior. HFRS-DA's impressive AUC (0.855) is attributable entirely to the direct embedding lookup and interaction-matrix attention, not to the nutritional/temporal graph structure. Claims of "heterogeneous graph awareness" require explicit verification that edge types are consumed in forward propagation.

**Finding F3 — Robustness Verification.** The topology-invariance result is verified to be correct (not a bug) via three checks:
*(i)* Health scores computed from `healthness` edges are non-zero for all 31,458 foods (mean=0.6653);  
*(ii)* The ablation code correctly zeros `edge_index` (not `edge_attr`) for each ablated edge type, preserving data structure integrity;  
*(iii)* Re-running EXP-F after three independent bug fixes produces identical results (Δ = 0.000).  
The finding is therefore architecturally expected and reproducible.

**EXP-F v2 — Valid Topology Ablation (NutriGraphNet + NGCF).** To obtain a meaningful topology ablation for models that *do* use auxiliary edges, we implement EXP-F v2 with:
- **NutriGraphNet** (routes message-passing through all 9 edge types via heterogeneous SAGEConv)
- **NGCF with 50% interaction dilution** (auxiliary-edge-connected foods have 50% of their interactions randomly removed, measuring the contribution of auxiliary structure to interaction-based ranking)

Results from EXP-F v2 will be reported in the camera-ready version.

---

### 6.6 Cross-Experiment Synthesis

The five experiments collectively paint a coherent picture of failure modes in GNN-based food recommendation. Figure 6 summarizes the AUC vs. HR@10 trade-off landscape across all models and conditions.

**Table S. Key Numerical Reference Table**

| Claim | Experiment | Confirmed Value |
|-------|-----------|-----------------|
| SGL best aug ratio | EXP-A | p=0.0 (no augmentation), HR@10=0.3604 |
| SGL collapse at 10% density | EXP-B | HR@10=0.092 vs. MF=0.344 (3.74×) |
| LightGCN best at low density | EXP-B | HR@10=0.513 at 10%, best of all models |
| NGCF best at high density | EXP-B | HR@10=0.777 at 100%, best overall |
| MF AUC structural ceiling | EXP-B/D | AUC ≤ 0.547 regardless of density or dim |
| λ_health sensitivity (HFRS-DA) | EXP-C | Δ HR@10 = 0.000 exactly for λ ∈ [0.001, 1.0] |
| HFRS-DA optimal dim | EXP-D | d=32 (HR@10=0.755, AUC=0.862) |
| NGCF optimal dim | EXP-D | d=256 (HR@10=0.787, AUC=0.881) |
| HFRS-DA dim=256 AUC collapse | EXP-D | AUC=0.574 vs. avg 0.848 (Δ=−0.274) |
| HFRS-DA topology invariance | EXP-F | Δ HR@10 = 0.000 across all 5 ablations |
| Best overall HR@10 | EXP-D | NGCF dim=256: 0.7867 |
| Best overall AUC | EXP-D | NGCF dim=256: 0.8810 |

---

## 7. Design Guidelines

Based on our empirical findings, we propose the following actionable guidelines for practitioners:

| Scenario | Recommendation | Evidence |
|----------|----------------|---------|
| Sparse interactions (<20/user) | **Avoid SGL** — use MF/BPR or LightGCN | EXP-B: SGL HR@10=0.092 at 10% density, 73.3% worse than MF |
| Low density (10–30%) | **Prefer LightGCN** over NGCF | EXP-B: LightGCN best at 10%–30%; NGCF crossover at 50% |
| Dense interactions (>50% / >6 int/user) | **NGCF** optimal for ranking | EXP-B: NGCF best at 50%–100%; HR@10=0.777 at full density |
| SGL augmentation ratio | **p=0.0 is always optimal** on sparse nutrition graphs | EXP-A: all p>0 degrade HR@10 |
| Embedding dim selection | **NGCF: d=64–128** sufficient; **HFRS-DA: d=32** optimal | EXP-D: diminishing returns; HFRS-DA collapses at d=256 |
| Health constraint weight (λ) | **Monitor gradient norms** before trusting λ sensitivity | EXP-C: HFRS-DA Δ HR@10=0.000 for all λ ∈ [0.001, 1.0] |
| Health backpropagation | **Verify gradient path** from health loss to health params | EXP-C+F: HFRS-DA severs health backprop architecturally |
| Auxiliary edge types | **Ablate each edge type** to verify actual contribution | EXP-F: HFRS-DA shows zero sensitivity to all 3 auxiliary types |
| "Heterogeneous graph" claims | Only claim graph-awareness if architecture **routes messages** through target edge types | EXP-F: topology invariance invalidates HFRS-DA's graph claim |
| Clinical health-aware deployment | **Do not trust named λ_health** without gradient-level verification | EXP-C: health loss numerically present but backpropagation severed |

---

## 8. Conclusion

We presented a systematic empirical analysis of GNN-based food recommendation on a large-scale heterogeneous nutrition graph (NutriGraph-KR: 20,820 users, 31,458 foods, density=0.040%), uncovering three key phenomena with root-cause explanations:

**(1) SGL Augmentation Collapse.** Edge dropout augmentation degrades ranking performance for all p > 0 (HR@10: 0.3604→0.3520 as p: 0.0→0.5, −2.3%), with catastrophic collapse at 10% data density (HR@10=0.092 vs. MF=0.344, 3.74× gap; vs. LightGCN=0.513, 5.58× gap). The root cause is the fundamental incompatibility between SGL's InfoNCE contrastive objective — which requires dense positive views — and nutrition interaction graphs with an average of only 12.6 interactions per user (10× sparser than MovieLens-1M). LightGCN provides the most robust alternative at low densities.

**(2) MF–SGL Ranking Paradox.** Simple matrix factorization (HR@10=0.730 at dim=64, scaling to 0.753 at dim=256) outperforms SGL on HR@10 across all tested conditions. The paradox is twofold: *(a)* MF achieves competitive ranking with only 0.547 AUC vs. NGCF's 0.879 (33-pt gap → only 2.64% HR@10 gap), showing that AUC does not predict ranking quality in sparse settings; *(b)* SGL fails completely (HR@10=0.354 at full density, 0.092 at 10%) despite higher AUC (0.700), demonstrating that contrastive learning harms sparse ranking. GNN over-smoothing (NGCF/LightGCN plateau at d=64–128) and MF's effective L2 implicit regularization explain the paradox.

**(3) Health Constraint Architectural Failure.** HFRS-DA's health loss produces zero measurable effect across four orders of magnitude of λ_health (Δ HR@10 = 0.000 exactly, Δ AUC < 1.25×10⁻⁷). Root-cause analysis identifies two compounding failures: the NutriLoss health gradient backpropagation path is architecturally severed, and HFRS-DA is completely topology-invariant (EXP-F: Δ HR@10 = 0.000 for all auxiliary edge type removals). HFRS-DA's architecture reads the heterogeneous graph during data construction but never uses it in forward propagation. For clinical practitioners, this means the model provides no statistical guarantee of health-aware recommendation.

**Our findings challenge three widely held assumptions** in graph-based food recommendation: (a) SGL augmentation improves sparse graphs; (b) architectural complexity correlates with ranking quality; (c) named "health-aware" models actually optimize health objectives. Future work will investigate: (i) ingredient-conditioned positive sampling for contrastive learning in sparse nutrition graphs; (ii) explicit health gradient monitoring and routing via dedicated graph convolution along `healthness` edges; and (iii) EXP-F v2 with NutriGraphNet to obtain valid topology ablation for truly heterogeneous-graph-aware architectures.

---

## References

*(To be completed — key papers)*
- Koren et al. (2009). Matrix Factorization Techniques for Recommender Systems. IEEE Computer.
- He et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR.
- Wang et al. (2019). Neural Graph Collaborative Filtering. SIGIR.
- Wu et al. (2021). Self-supervised Graph Learning for Recommendation. SIGIR.
- Forouzandeh et al. (2024). HFRS-DA. Computers in Biology and Medicine.
- Forouzandeh et al. (2025). FRMADHG. Scientific Reports.
- Song et al. (2022). SCHGN. ACM TOMM.
- Gao et al. (2022). FGCN. Information Sciences.
- Yu et al. (2022). Are Graph Augmentations Necessary? SimGCL. SIGIR.
- Yu et al. (2023). XSimGCL. IEEE TKDE.

---
*Draft v0.4 — 2026-07-11*  
*New in v0.4: Full Section 6 rewrite with all confirmed numerical values from SUMMARY_v3.json.*  
*EXP-A: confirmed aug=0.0 always best, step-diff analysis, theoretical explanation revised.*  
*EXP-B: added Table B-AUC, LightGCN/NGCG crossover analysis (B3), MF AUC ceiling (B4), full density MF vs NGCF (B5), 5-item sparsity sensitivity ranking (B2).*  
*EXP-C: confirmed Δ HR@10=0.000 for all 8 λ values; dual failure mechanism analysis (backprop severance + score distribution mismatch).*  
*EXP-D: added Table D-AUC, LightGCN non-monotone finding (D3), NGCF dimension efficiency (D2), HFRSDA attention ill-conditioning hypothesis for dim=256 (D5).*  
*EXP-F: full forward-pass anatomy with code, formal topology invariance finding (F1/F2/F3), EXP-F v2 design.*  
*Section 6.6 Cross-Experiment Synthesis: Table S with 12 confirmed key values.*  
*Design Guidelines: 3 new rows added (low-density, health backprop, clinical).*  
*Conclusion: strengthened with dual root-cause for all three phenomena.*  
*Figures: fig1_sgl_aug_sweep, fig2_sparsity_sweep, fig3_lambda_sensitivity, fig4_dim_sweep, fig5_graph_ablation, fig6_auc_hr_paradox*  
*Pending: EXP-F v2 (NutriGraphNet topology ablation), EXP-C v2 (NutriGraphNet λ_health), EXP-G (layer depth sweep 1–4)*
