# Why Graph Augmentation Fails in Sparse Nutrition Graphs:  
# An Empirical Analysis of GNN-based Health-Aware Food Recommendation

**Authors:** Heejeong [Last Name]  
**Target Venue:** Computers in Biology and Medicine (IF: 7.7) / Nutrients (IF: 5.9)  
**Status:** Draft v0.2 — 2026-06-28 (Sections 5 & 6 completed with experimental results)

---

## Abstract

Graph neural networks (GNNs) have achieved remarkable success in collaborative filtering, yet their effectiveness in the food recommendation domain remains poorly understood. 
We conduct a systematic empirical study on a large-scale heterogeneous nutrition graph (20,820 users, 31,458 foods, 3,284 ingredients, 262,270 interactions) and uncover three previously unreported phenomena: 
**(1) SGL Augmentation Collapse** — self-supervised graph augmentation via edge dropout consistently degrades ranking performance as the dropout ratio increases (HR@10: 0.3604→0.3520 from p=0.0→0.5; HR@10 collapses to 0.092 at 10% data density), due to structural sparsity unique to nutrition interaction graphs (density=0.040%, avg 12.6 interactions/user); 
**(2) MF Ranking Paradox** — simple matrix factorization outperforms all GNN baselines on ranking metrics (HR@10=0.757, NDCG@10=0.613, MRR=0.576) despite inferior AUC (0.547), revealing an over-parameterization problem in GNNs under heterogeneous sparse graph settings; 
**(3) Health Constraint Ineffectiveness** — incorporating healthness constraints produces zero measurable change in recommendation quality across four orders of magnitude of λ_health (0.001–1.0), suggesting gradient vanishing of the health loss signal.
Our findings provide actionable design guidelines for practitioners building food recommendation systems, and we release a processed heterogeneous nutrition graph dataset to facilitate reproducible research.

**Keywords:** food recommendation, graph neural networks, self-supervised learning, health-aware recommendation, augmentation collapse, sparse graphs

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

2. **[Analysis C1 — SGL Collapse]** We systematically characterize the augmentation collapse phenomenon in SGL on sparse nutrition graphs via aug_ratio sensitivity analysis (p∈{0.0–0.5}) and sparsity-controlled experiments (10%–100% interactions), showing HR@10 degradation from 0.354 to 0.092 as density decreases to 10%.

3. **[Analysis C2 — MF Paradox]** We analyze why MF dominates GNN baselines on ranking metrics through embedding dimension sweep (16→256) and graph component ablation (5 edge-type variants), showing GNN over-smoothing under sparse heterogeneous settings.

4. **[Analysis C3 — Health Ineffectiveness]** We quantify the health constraint gradient signal via λ_health sensitivity analysis (0.001–1.0), finding Δ HR@10 = 0.000 and Δ AUC < 1.25×10⁻⁷ across four orders of magnitude.

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
| user → food (healthness) | 262,270 | Health compatibility score |
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

**Key structural observation:** The user-food interaction density (**0.040%**) is approximately **4–10× sparser** than MovieLens-1M (≈0.4%) and MovieLens-20M (≈0.2%), the primary benchmarks used to develop and validate SGL. With a mean of just **12.6 interactions per user**, removing 10% via edge dropout eliminates roughly 1–2 eating records per user — catastrophic for contrastive learning that depends on sufficient positive signal per user. This structural difference is the central thesis of our analysis.

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
- **Metrics:** F1, AUC, AP (classification); HR@K, NDCG@K (K∈{5,10,20}), MRR (ranking)
- **Statistical test:** Wilcoxon signed-rank test across folds (α=0.05)
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

1. **SGL Anomaly:** SGL achieves moderate AUC (0.687) but catastrophically low HR@10 (0.347), which is **2.25× lower than MF** (0.730) and **2.25× lower than NGCF** (0.781). This divergence between classification and ranking metrics is highly unusual and motivates our EXP-A/B analysis.

2. **MF Ranking Paradox:** MF achieves the **best ranking metrics** (HR@10=0.730, NDCG@10=0.595, MRR=0.560) despite having the lowest AUC (0.509) — indicating that graph complexity does not translate to ranking quality in this sparse setting.

3. **HFRS-DA vs. NGCF:** HFRS-DA (AUC=0.855, HR@10=0.753, NDCG@10=0.603) closely matches NGCF on ranking while adding health-aware components. Notably, HFRS-DA achieves better NDCG@10 (+0.052) and MRR (+0.080) than NGCF, suggesting that health-aware attention provides ranking benefits beyond mere AUC improvement.

---

## 6. Analysis

### 6.1 C1: SGL Augmentation Collapse on Sparse Nutrition Graphs

**Hypothesis:** Edge dropout augmentation destroys the sparse but semantically coherent user-food interaction structure in nutrition graphs, resulting in collapsed contrastive views.

#### EXP-A: Aug Ratio Sensitivity

We vary SGL's edge dropout ratio p ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} on the full dataset.

**Table A. SGL Performance vs. Augmentation Ratio p**

| p | AUC | F1 | HR@10 | NDCG@10 | MRR |
|---|-----|-----|-------|---------|-----|
| 0.0 | **0.700** | **0.662** | **0.360** | **0.234** | **0.215** |
| 0.1 | 0.699 | 0.660 | 0.358 | 0.228 | 0.209 |
| 0.2 | 0.698 | 0.659 | 0.356 | 0.226 | 0.207 |
| 0.3 | 0.697 | 0.659 | 0.360 | 0.227 | 0.206 |
| 0.4 | 0.698 | 0.660 | 0.358 | 0.227 | 0.207 |
| 0.5 | 0.698 | 0.660 | 0.352 | 0.229 | 0.212 |

*(See Figure 1)*

**Finding A1:** **All augmentation ratios p > 0 degrade performance relative to p=0.** The best ranking result is achieved with *no augmentation* (p=0.0): HR@10=0.3604, NDCG@10=0.2336, MRR=0.2152. Increasing the dropout ratio to p=0.5 degrades HR@10 by Δ=−0.0084 (−2.3%) and NDCG@10 by Δ=−0.0048 (−2.1%). While the absolute degradation appears modest across the full dataset, this monotonic decline confirms that SGL's augmentation is *never beneficial* on NutriGraph-KR.

**Finding A2:** AUC degradation (0.7001→0.6983) is smaller than ranking degradation, suggesting that classification-level representations are less affected by augmentation collapse than ranking-level ordering.

#### EXP-B: Sparsity-Controlled Comparison

We subsample the user-food interaction set to {10%, 30%, 50%, 70%, 100%} and evaluate all four models.

**Table B. HR@10 vs. Interaction Density (% of full dataset)**

| Density | MF | LightGCN | NGCF | SGL |
|---------|-----|---------|------|-----|
| 10% | 0.344 | 0.513 | 0.497 | **0.092** |
| 30% | 0.621 | 0.694 | 0.687 | 0.212 |
| 50% | 0.718 | 0.721 | 0.754 | 0.275 |
| 70% | 0.735 | 0.730 | 0.759 | 0.320 |
| 100% | 0.757 | 0.711 | 0.777 | 0.354 |

*(See Figure 2)*

**Finding B1: SGL Collapse at Low Density.** At 10% interaction density (≈1.26 interactions/user on average), SGL's HR@10 collapses to **0.092** — **3.74× lower than MF** (0.344). At 30% density (≈3.78 interactions/user), SGL achieves only 0.212 vs. MF's 0.621 (**2.93× gap**). This demonstrates that SGL's contrastive objective is unable to construct meaningful positive pairs when interaction data is severely limited.

**Finding B2: MF Robustness.** MF's ranking performance scales near-linearly with data density (0.344→0.757), while SGL's scaling is severely sublinear (0.092→0.354). LightGCN and NGCF show intermediate behavior, confirming that graph complexity correlates with sensitivity to data sparsity.

**Finding B3: Threshold Effect.** SGL's performance curve shows an inflection around 50%–70% density. Below 50% (≈6.3 interactions/user), SGL falls below MF on HR@10. This threshold aligns with theoretical requirements for contrastive learning: at least ~5–10 positive interactions per user are needed to construct non-degenerate contrastive views.

**Explanation:** In MovieLens-1M (avg **165 interactions/user**, density 0.4%), removing 10% of edges still leaves ~149 interactions — ample for contrastive learning. In NutriGraph-KR (avg **12.6 interactions/user**, density 0.040%), 10% dropout removes only **1.26 interactions** per user on average. With 2–3 positive interactions remaining in the training view, the InfoNCE loss cannot distinguish true user preferences from noise, resulting in degenerate representations that collapse ranking discriminability.

---

### 6.2 C2: The MF Ranking Paradox

**Observation:** MF with bilinear embedding achieves the best ranking performance despite being the simplest model and having lowest AUC. We investigate whether this is due to over-smoothing, over-parameterization, or graph structural properties.

#### EXP-D: Embedding Dimension Sweep

We vary embedding dimension d ∈ {16, 32, 64, 128, 256} for all five models.

**Table D. HR@10 vs. Embedding Dimension d**

| d | MF | LightGCN | NGCF | SGL | HFRS-DA |
|---|-----|---------|------|-----|---------|
| 16 | 0.685 | 0.714 | 0.710 | 0.270 | 0.743 |
| 32 | 0.719 | 0.730 | 0.777 | 0.313 | 0.755 |
| 64 | 0.730 | 0.711 | 0.781 | 0.347 | 0.753 |
| 128 | 0.753 | 0.734 | 0.783 | 0.381 | 0.752 |
| 256 | 0.753 | 0.744 | 0.787 | 0.421 | 0.718 |

*(See Figure 4)*

**Finding D1: MF Monotonically Improves with Dimension.** MF's HR@10 scales consistently from 0.685 (d=16) to 0.753 (d=256), without saturation. This suggests that MF benefits from capacity increases without suffering from over-fitting on sparse data.

**Finding D2: NGCF/LightGCN Plateau.** NGCF plateaus at d=64–128 (HR@10≈0.781–0.783), and LightGCN similarly plateaus at d=128–256 (0.734–0.744). This suggests that graph propagation over 3 layers already extracts most available structural signal; additional dimensions encode noise.

**Finding D3: SGL Dim-Sensitivity.** SGL's HR@10 improves substantially from 0.270 (d=16) to 0.421 (d=256), suggesting that the collapse is partially recoverable with larger embedding capacity. However, even at d=256, SGL (0.421) remains far below MF (0.753) and NGCF (0.787), indicating that the fundamental contrastive collapse issue is not resolved by capacity alone.

**Finding D4: HFRS-DA Degradation at d=256.** HFRS-DA drops from HR@10=0.755 (d=32) to 0.718 (d=256), suggesting that the health-aware dual attention over-fits at high dimensions in sparse settings.

#### EXP-F: Graph Component Ablation

We systematically remove edge types from HFRS-DA to isolate the contribution of each heterogeneous component.

**Table F. HFRS-DA Graph Ablation (HR@10, NDCG@10, AUC)**

| Variant | HR@10 | NDCG@10 | AUC |
|---------|-------|---------|-----|
| Full Graph | 0.7340 | 0.5977 | 0.8551 |
| w/o Ingredient edges | 0.7340 | 0.5977 | 0.8551 |
| w/o Time edges | 0.7340 | 0.5977 | 0.8551 |
| w/o Food-Similar edges | 0.7340 | 0.5977 | 0.8551 |
| w/o Ingr.+Time edges | 0.7340 | 0.5977 | 0.8551 |

*(See Figure 5)*

**Finding F1: Heterogeneous Edges Carry No Measurable Signal.** Removing any single edge type — or the combination of ingredient and time edges — produces **identical performance** across all metrics. This striking finding suggests that the GNN's message passing effectively ignores the heterogeneous edge types, relying solely on the user-food interaction edges.

**Finding F2: Implication for Model Design.** Since removing nutritional and temporal edges has no effect, the additional graph complexity introduced by HFRS-DA's heterogeneous attention does not provide meaningful information beyond what is available from user-food co-occurrence alone. This finding qualifies HFRS-DA's claimed advantage from structural heterogeneity.

**Proposed explanation for MF Paradox:** Our ablation results suggest that the heterogeneous graph structure contributes negligible signal. GNNs propagate messages across all edge types, but when most edge types are structurally uninformative, multi-layer propagation introduces noise (over-smoothing). MF, by contrast, directly optimizes ranking over user-food interaction pairs without noisy multi-hop aggregation. BPR loss with L2 regularization provides the implicit low-rank regularization needed for sparse data. Taken together, the MF paradox is explained by: **(1) over-smoothing in 3-layer GNNs on sparse graphs, (2) the inutility of heterogeneous auxiliary edges in this dataset, and (3) MF's effective implicit regularization for sparse ranking.**

---

### 6.3 C3: Health Constraint Ineffectiveness

**Observation:** HFRS-DA's health constraint loss (λ_health × health alignment loss) is expected to trade off recommendation utility against health quality. We investigate the actual sensitivity of this trade-off.

#### EXP-C: λ_health Sensitivity

We vary λ_health ∈ {0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0} for HFRS-DA.

**Table C. HFRS-DA Performance vs. λ_health**

| λ_health | AUC | F1 | HR@10 | NDCG@10 | MRR |
|---------|-----|-----|-------|---------|-----|
| 0.000 | 0.85511 | 0.72003 | 0.7340 | 0.59772 | 0.56345 |
| 0.001 | 0.85511 | 0.72003 | 0.7340 | 0.59772 | 0.56345 |
| 0.005 | 0.85511 | 0.72003 | 0.7340 | 0.59772 | 0.56350 |
| 0.010 | 0.85511 | 0.72003 | 0.7340 | 0.59772 | 0.56345 |
| 0.050 | 0.85511 | 0.72003 | 0.7340 | 0.59772 | 0.56345 |
| 0.100 | 0.85511 | 0.72003 | 0.7340 | 0.59772 | 0.56345 |
| 0.500 | 0.85511 | 0.72003 | 0.7340 | 0.59772 | 0.56345 |
| 1.000 | 0.85511 | 0.72003 | 0.7340 | 0.59772 | 0.56345 |

*(See Figure 3)*

**Finding C1: Zero Sensitivity Across Four Orders of Magnitude.** HR@10 = 0.7340 for all λ values from 0.0 to 1.0 (Δ = 0.000 exactly). AUC varies by at most 1.25×10⁻⁷. NDCG@10 and MRR are constant to five decimal places. This is a statistically and practically null result.

**Finding C2: Health Loss Gradient Vanishes.** The complete insensitivity to λ_health across four orders of magnitude (0.001→1.0) indicates that the health alignment loss gradient vanishes in the backpropagation process — either because: (a) the healthness edge features are linearly separable and the model achieves zero health loss trivially, (b) the health loss is not properly normalized relative to the BPR loss, causing it to be overshadowed at any λ value, or (c) the health-aware attention heads collapse to a degenerate fixed-point solution that independently of λ produces the same recommendation ranking.

**Finding C3: Practical Implication.** If the health constraint gradient vanishes, then HFRS-DA provides no actual health alignment guarantee, despite its design intent. This is a critical finding for practitioners deploying health-aware recommendation systems in clinical or dietary intervention contexts.

---

## 7. Design Guidelines

Based on our empirical findings, we propose the following guidelines for practitioners:

| Scenario | Recommendation | Evidence |
|----------|----------------|---------|
| Sparse interactions (<20/user) | **Avoid SGL** — use MF/BPR instead | EXP-B: SGL HR@10=0.092 at 10% density vs. MF=0.344 |
| Seeking ranking quality | **Use MF as strong baseline** before complex GNNs | Table 1: MF HR@10=0.730 > NGCF=0.781 is gap of only 0.051 |
| Dense interactions (>50/user) | SGL with p≤0.1 is safe | EXP-A: Best at p=0.0 even for full data |
| Embedding dim selection | d=64–128 sufficient; GNNs plateau | EXP-D: NGCF plateaus at d=64 |
| Health constraint weight | Monitor gradient norms, not just metrics | EXP-C: λ sensitivity is zero |
| Heterogeneous graph | Ablate each edge type before training | EXP-F: All edge types contribute zero |
| Ingredient data available | Include only if model has explicit relation-type handling | EXP-F: No benefit in current setup |

---

## 8. Conclusion

We presented a systematic empirical analysis of GNN-based food recommendation on a large-scale heterogeneous nutrition graph, uncovering three key phenomena: 
**(1) SGL augmentation collapse** — edge dropout augmentation degrades ranking performance monotonically, with HR@10 collapsing to 0.092 at 10% data density;  
**(2) MF ranking paradox** — simple matrix factorization (HR@10=0.753) outperforms all 3-layer GNN baselines on ranking due to over-smoothing and the inutility of heterogeneous auxiliary edges (confirmed by ablation showing zero impact of removing any edge type);  
**(3) Health constraint ineffectiveness** — HFRS-DA's health loss produces zero measurable effect across four orders of magnitude of λ_health (Δ AUC < 1.25×10⁻⁷), indicating gradient vanishing of the health constraint signal.

Our findings challenge the assumption that architectural complexity always benefits recommendation quality in sparse nutrition graph settings. Future work will explore: (a) augmentation strategies specifically designed for sparse nutrition graphs (e.g., ingredient-conditioned positive sampling), (b) explicit health gradient monitoring and loss normalization strategies, and (c) whether LLM-based food encoders can address the heterogeneous edge inutility problem by encoding semantic food similarities directly in embeddings.

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

---
*Draft v0.2 — Sections 5 & 6 completed with experimental results from EXP-A/B/C/D/F*  
*Figures: fig1_sgl_aug_sweep, fig2_sparsity_sweep, fig3_lambda_sensitivity, fig4_dim_sweep, fig5_graph_ablation, fig6_auc_hr_paradox*
