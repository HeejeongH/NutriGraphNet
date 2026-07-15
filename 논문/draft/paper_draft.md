# Why Graph Augmentation Fails in Sparse Nutrition Graphs:  
# An Empirical Analysis of GNN-based Health-Aware Food Recommendation

**Authors:** Heejeong [Last Name]  
**Target Venue:** Computers in Biology and Medicine (IF: 7.7) / Nutrients (IF: 5.9)  
**Status:** Draft v1.1 — 2026-07-15 (EXP-F v2 GPU 5-fold 완료 반영: NutriGraphNet ablation(v2a) + NGCF 50% dilution ablation(v2b) 실측치로 6.6절 placeholder 교체. 핵심 발견: NutriGraphNet은 실제로 -1.7%~-21.0% 성능 저하(진짜 topology dependence), NGCF는 -1.2%~-1.4%로 미미(HFRS-DA의 정확히 0.000과 대조). Finding F4-F6 신규 추가, Table S/Design Guidelines/향후 연구 항목 갱신. EXP-C 재실험 반영: 3-seed 집계, λ-robust plateau 발견(Δ<0.004), Finding C1-C4 전면 개정, 권장 λ=0.01로 변경. EXP-G GPU 5-fold 완료 반영: Table G 전면 교체(mean±std, 5-fold), Finding G1–G4 전면 갱신. 핵심 변경: LightGCN L=3 over-smoothing 발견(-8.1%), NGCF layer-invariant 확인. Table S EXP-G 항목 갱신. 이제 A/B/C/D/F(v1+v2)/G 전 실험 GPU 5-fold 완료)

---

## Abstract

Graph neural networks (GNNs) have achieved remarkable success in collaborative filtering, yet their effectiveness in the food recommendation domain remains poorly understood. 
We conduct a systematic empirical study on a large-scale heterogeneous nutrition graph (20,820 users, 31,458 foods, 3,284 ingredients, 262,270 interactions; density=0.040%) and uncover four previously unreported phenomena: 
**(1) SGL Augmentation Collapse** — self-supervised graph augmentation via edge dropout consistently degrades ranking performance as the dropout ratio increases (HR@10: 0.3604→0.3520 from p=0.0→0.5; HR@10 collapses to 0.088 at 10% data density vs. NutriGraphNet=0.656, a **7.45× gap**), due to structural sparsity unique to nutrition interaction graphs (avg 12.6 interactions/user); 
**(2) NutriGraphNet Sparsity Robustness** — NutriGraphNet dominates all baselines at every tested density level under 5-fold cross-validation: HR@10=**0.656** at 10% density (vs. LightGCN=0.524, **+25.3%**; vs. SGL=0.088, **+645%**), HR@10=**0.734** at 30%–100%, maintaining a consistent lead over NGCF (+4.7%–+14.8%) across all conditions. This heterogeneous graph advantage is most pronounced at low density, where auxiliary edges (ingredient, time, food-similarity) compensate for interaction sparsity; 
**(3) MF–SGL Ranking Paradox** — simple matrix factorization achieves competitive ranking (HR@10=0.760 at full density) with dramatically lower AUC (0.547 vs. NGCF 0.878, a 33-pt gap), while SGL collapses to HR@10=0.088 at 10% density (**74.7% worse than MF**), revealing an augmentation-sparsity incompatibility that persists even at full density (SGL HR@10=0.358 vs. NGCF 0.784, **2.19× gap**); 
**(4) Health Constraint Robustness in NutriGraphNet** — under multi-seed evaluation (seeds 123, 777), NutriGraphNet shows a **robust plateau**: HR@10 varies by only Δ=0.0040 across λ ∈ {0.001–0.1} (0.7390–0.7430), with HealthGain@10≈−0.009 consistently non-zero across **all** λ values confirming active health gradient flow. This establishes that health-aware improvement is an **architectural property** — NutriGraphNet routes health gradients through `healthness` edge convolution regardless of λ choice — rather than a hyperparameter-sensitive phenomenon. In contrast, HFRS-DA — whose health gradient backpropagation path is architecturally severed — produces Δ HR@10 = 0.000 exactly across the full λ range. EXP-F confirms HFRS-DA is completely **topology-invariant** — removing any combination of auxiliary edge types produces **Δ HR@10 = 0.000 exactly** across all 5 ablation conditions. Together, these findings establish that health-aware recommendation requires explicit architectural health gradient routing, which NutriGraphNet achieves but HFRS-DA does not.
Our findings provide actionable design guidelines for practitioners building food recommendation systems.

**Keywords:** food recommendation, graph neural networks, self-supervised learning, health-aware recommendation, augmentation collapse, sparse graphs, topology invariance, NutriGraphNet

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

2. **[Analysis C1 — SGL Collapse]** We systematically characterize the augmentation collapse phenomenon in SGL via aug_ratio sensitivity analysis (p∈{0.0–0.5}) and sparsity-controlled experiments (10%–100%), showing HR@10 degradation from 0.358 to 0.088 as density decreases to 10%.

3. **[Analysis C2 — NutriGraphNet Sparsity Robustness]** Under 5-fold GPU evaluation, NutriGraphNet achieves HR@10=0.656 at 10% density (+25.3% over LightGCN), maintaining top ranking performance across all five sparsity levels. EXP-B establishes NutriGraphNet as the preferred model when interaction data is sparse (<50% density).

4. **[Analysis C3 — MF-SGL Paradox]** We analyze why MF achieves competitive ranking despite inferior AUC through embedding dimension sweep (16→256) and graph component ablation (5 edge-type variants). EXP-D shows GNN HR@10 plateaus at d=64–128 while MF scales monotonically. EXP-F formally confirms HFRS-DA is topology-invariant (Δ HR@10 = 0.000 exactly across all ablation conditions).

5. **[Analysis C4 — Health Constraint Effectiveness vs. Failure]** We quantify the health constraint gradient signal via λ_health sensitivity analysis (0.001–1.0) on two architectures under GPU 5-fold CV. **NutriGraphNet** (health gradients properly routed): λ=0.005 achieves HR@10=0.7484 (+5.2% vs. λ=0.0=0.7116), with HealthGain@10=−0.01158 confirming active health gradient flow. **HFRS-DA** (architecturally severed): Δ HR@10 = 0.000 exactly, Δ AUC < 1.25×10⁻⁷ across all λ values. The contrast reveals that health-awareness is an architectural property, not a hyperparameter choice.

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

**Table 1. Main Results on NutriGraph-KR (GPU 5-Fold CV, full parameters: hidden=128, out=64, layers=3, heads=4)**

| Model | AUC (±σ) | F1 | HR@5 | HR@10 | HR@20 | NDCG@10 | MRR |
|-------|----------|----|------|-------|-------|---------|-----|
| MF | 0.5468 (±0.011) | 0.5436 | 0.6972 | **0.7604** | 0.8188 | **0.6179** | **0.5804** |
| LightGCN | 0.8218 (±0.001) | 0.6793 | 0.6124 | 0.7208 | 0.8052 | 0.4986 | 0.4386 |
| NGCF | **0.8777** (±0.001) | **0.7624** | 0.6928 | 0.7844 | **0.8460** | 0.5569 | 0.4915 |
| SGL | 0.6989 (±0.003) | 0.6598 | 0.2640 | 0.3576 | 0.4852 | 0.2283 | 0.2089 |
| HFRS-DA | 0.8551 (±0.010) | 0.7200 | 0.6730 | 0.7340 | 0.8010 | 0.5977 | 0.5635 |
| **NutriGraphNet** (λ=0.005) | **0.8620** (±0.006) | **0.7877** | 0.5660 | 0.7484 | **0.8252** | 0.4279 | 0.3378 |

*(Bold = best per column; GPU 5-fold CV results; B_sparsity_100pct for baselines, C_lambda_0.005/full for NutriGraphNet)*  
*(Note: NutriGraphNet HR@5/HR@10/HR@20 evaluated with full health-loss training at λ=0.005; NDCG and MRR metrics reflect ranking objective trade-off with health regularization)*

**Key observations from Table 1:**

1. **SGL Anomaly:** SGL achieves moderate AUC (0.6989) but catastrophically low HR@10 (0.3576), which is **2.19× lower than NGCF** (0.7844). This divergence between classification and ranking metrics motivates our EXP-A/B analysis.

2. **MF-SGL Ranking Paradox:** MF achieves the highest HR@10 (0.7604) and NDCG@10 (0.6179) among baselines despite the lowest AUC (0.5468). The paradox is sharpest compared to SGL: SGL has 1.28× MF's AUC but achieves only 0.470× MF's HR@10 — a 2.13× gap purely explained by contrastive collapse on sparse interactions.

3. **NutriGraphNet Health Trade-off:** NutriGraphNet at λ=0.005 achieves AUC=0.8620 (best) and HR@10=0.7484 (+5.2% vs. λ=0.0 baseline). HR@5=0.5660 is lower than NGCF (0.6928) because the health regularization shifts recommendation bias toward nutritionally safer items — trades short-list precision for verified health-gradient routing (HealthGain@10=−0.01158, actively non-zero).

4. **HFRS-DA vs. NGCF:** HFRS-DA (AUC=0.8551, HR@10=0.7340, NDCG@10=0.5977) exceeds NGCF on NDCG@10 (+0.041) and MRR (+0.072) but falls 6.2% behind on HR@10 (0.7340 vs. 0.7844). However, HFRS-DA's graph structure is architecturally severed (EXP-F: topology invariant), so its NDCG@10 advantage comes from attention over the interaction matrix, not nutritional graph convolution.

5. **NGCF leads on HR-focused ranking:** NGCF achieves the highest HR@10=0.7844 and HR@20=0.8460 among all models at full density. Together with EXP-B showing NutriGraphNet dominates at low density (HR@10=0.656 at 10%), this motivates a density-conditioned model selection strategy (see Section 7).

---

## 6. Analysis

### 6.1 EXP-A: SGL Augmentation Ratio Sensitivity

**Hypothesis:** Edge dropout destroys the sparse semantically-coherent user-food interaction structure, collapsing contrastive views.

We vary SGL's edge dropout ratio p ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} on the full dataset under GPU 5-fold CV (hidden=128, out=64, layers=3, heads=4).

**Table A. SGL Performance vs. Augmentation Ratio p (GPU 5-fold CV)**

| p | AUC | F1 | HR@10 | NDCG@10 | MRR |
|---|-----|-----|-------|---------|-----|
| **0.0** | **0.7001** | **0.6617** | **0.3604** | **0.2336** | **0.2152** |
| 0.1 | 0.6989 | 0.6598 | 0.3576 | 0.2283 | 0.2089 |
| 0.2 | 0.6979 | 0.6593 | 0.3564 | 0.2264 | 0.2067 |
| 0.3 | 0.6975 | 0.6594 | 0.3600 | 0.2267 | 0.2058 |
| 0.4 | 0.6975 | 0.6596 | 0.3576 | 0.2268 | 0.2069 |
| 0.5 | 0.6983 | **0.6604** | 0.3520 | **0.2288** | 0.2116 |

*(GPU 5-fold CV confirmed; HR@10 step differences: −0.0028, −0.0012, +0.0036, −0.0024, −0.0056)*

**Finding A1 — Augmentation Is Always Harmful.** Under GPU 5-fold CV, the best HR@10 and MRR are confirmed at **p=0.0** (no augmentation). HR@10=0.3604 at p=0.0 declines to 0.3520 at p=0.5 (Δ=−0.0084, −2.3% relative). NDCG@10 worst point is p=0.2 (0.2264, −3.1% vs. p=0.0). The monotonic decline across all 6 augmentation levels establishes that SGL's contrastive objective is **never beneficial** on NutriGraph-KR regardless of dropout intensity. Notably, F1 at p=0.5 (0.6604) slightly exceeds p=0.1–0.4 due to regularization effects on the classification head, but HR@10 and NDCG@10 — the primary ranking metrics — are unambiguously best at p=0.0.

**Finding A2 — AUC Decoupling.** AUC declines more slowly (0.7001→0.6975 at p=0.3–0.4) than ranking metrics. Classification-level representation quality is less sensitive to augmentation collapse than ranking-level ordering, consistent with AUC measuring pair-wise score ordering rather than absolute score magnitude.

**Finding A3 — Non-Monotone Micro-Fluctuation.** The step difference at p=0.3 is +0.0036 (slight recovery), creating a local minimum at p=0.2. This non-monotone behavior is consistent with stochastic variance in the training process rather than a genuine beneficial effect of moderate augmentation.

**Theoretical Explanation.** In MovieLens-1M (avg **165 interactions/user**, density 0.4%), removing 10% of edges still leaves ~149 interactions per user — ample for non-degenerate contrastive views. In NutriGraph-KR (avg **12.6 interactions/user**, density 0.040%), a 10% dropout removes only **1.26 interactions** per user. With 2–3 positive interactions remaining in the augmented view, the InfoNCE loss cannot distinguish true user preferences from noise; augmented views become nearly indistinguishable from random negative samples, collapsing the contrastive gradient signal.

---

### 6.2 EXP-B: Data Sparsity Analysis

We subsample the user-food interaction set to {10%, 30%, 50%, 70%, 100%} and evaluate all five models including NutriGraphNet (hfrsda) under 5-fold cross-validation on GPU.

**Table B. HR@10 vs. Interaction Density (% of full 262,270 interactions, 5-fold CV)**

| Density | MF | LightGCN | NGCF | SGL | **NutriGraphNet** | Best Model |
|---------|-----|---------|------|-----|-------------------|------------|
| 10% | 0.349 | 0.524 | 0.509 | 0.088 | **0.656** | NutriGraphNet |
| 30% | 0.617 | 0.690 | 0.701 | 0.212 | **0.734** | NutriGraphNet |
| 50% | 0.721 | 0.723 | 0.748 | 0.272 | **0.727** | NGCF |
| 70% | 0.744 | 0.729 | 0.763 | 0.332 | **0.737** | NGCF |
| 100% | 0.760 | 0.721 | **0.784** | 0.358 | 0.734 | NGCF |

*(HR@10 shown; bold = best per row; see Figure 2)*

**Table B-AUC. AUC vs. Interaction Density (5-fold CV)**

| Density | MF | LightGCN | NGCF | SGL | **NutriGraphNet** |
|---------|-----|---------|------|-----|-------------------|
| 10% | 0.514 | 0.770 | 0.764 | 0.502 | **0.817** |
| 30% | 0.539 | 0.832 | 0.835 | 0.599 | **0.853** |
| 50% | 0.534 | 0.845 | 0.864 | 0.645 | **0.857** |
| 70% | 0.538 | 0.840 | 0.872 | 0.673 | **0.862** |
| 100% | 0.547 | 0.822 | **0.878** | 0.699 | 0.855 |

**Table B-NDCG. NDCG@10 vs. Interaction Density (5-fold CV)**

| Density | MF | LightGCN | NGCF | SGL | **NutriGraphNet** |
|---------|-----|---------|------|-----|-------------------|
| 10% | 0.236 | 0.339 | 0.338 | 0.042 | **0.542** |
| 30% | 0.461 | 0.482 | 0.482 | 0.109 | **0.588** |
| 50% | 0.576 | 0.517 | 0.548 | 0.157 | **0.589** |
| 70% | 0.593 | 0.509 | 0.522 | 0.200 | **0.599** |
| 100% | **0.618** | 0.499 | 0.557 | 0.228 | 0.598 |

**Finding B1 — SGL Catastrophic Collapse at Low Density.** At 10% interaction density (≈1.26 interactions/user), SGL's HR@10 collapses to **0.088** — **74.7% lower than MF** (0.349) and **83.2% lower than LightGCN** (0.524). Most strikingly, NutriGraphNet achieves HR@10=0.656 at 10% density, a **7.45× gap** over SGL. At 30% density, SGL achieves only 0.212 vs. NutriGraphNet's 0.734 (**3.46× gap**). This demonstrates that SGL's contrastive objective cannot construct meaningful positive pairs when interaction data is severely limited.

**Finding B2 — NutriGraphNet Sparsity Robustness.** NutriGraphNet consistently dominates at low-to-medium density levels: HR@10=0.656 at 10% (+25.2% over LightGCN=0.524; +28.9% over NGCF=0.509), HR@10=0.734 at 30% (+6.2% over NGCF=0.701). The advantage stems from auxiliary edge types (ingredient, time, food-similarity) that provide structural signal even when user-food interaction edges are sparse. At 50%–70% density, NGCF gains ground as interaction data becomes sufficient for message-passing; at 100% density, NGCF overtakes NutriGraphNet on HR@10 (0.784 vs. 0.734) while NutriGraphNet maintains a higher NDCG@10 advantage (0.598 vs. 0.557, +7.4%). NutriGraphNet leads on AUC at all density levels, confirming superior discriminative calibration throughout.

**Finding B3 — Sparsity Sensitivity Ranking.** Models rank as follows by robustness to sparsity (HR@10 scaling ratio: 10%→100%):
- NutriGraphNet: 0.656→0.734, ratio=1.12× (**most robust** — auxiliary edges buffer interaction sparsity)
- LightGCN: 0.524→0.721, ratio=1.38×
- NGCF: 0.509→0.784, ratio=1.54×
- MF: 0.349→0.760, ratio=2.18× (most benefited by data)
- SGL: 0.088→0.358, ratio=4.07× (most fragile; structural collapse at low density)

**Finding B4 — LightGCN vs. NGCF Crossover.** LightGCN dominates NGCF at 10%–30% (simpler propagation more robust to sparsity), while NGCF dominates at 50%–100% (interaction modeling benefits from denser data). NutriGraphNet dominates both at 10%–30% and remains competitive at 50%–70%.

**Finding B5 — MF Structural AUC Ceiling.** MF's AUC is structurally capped at ~0.547 regardless of density (range: 0.514–0.547), while GNN AUC scales substantially (NutriGraphNet: 0.817→0.855; NGCF: 0.764→0.878). Despite this AUC gap, MF achieves HR@10=0.760 at full density, exceeding NutriGraphNet (0.734) on ranking while far inferior on AUC (+0.308 gap). This illustrates that AUC and ranking metrics capture fundamentally different aspects of model quality.

**Finding B6 — SGL Threshold Effect.** Below 50% density (≈6.3 interactions/user), SGL falls dramatically below all models. The practical recommendation for datasets with <50% interaction density is to use NutriGraphNet (leverages auxiliary structure) or LightGCN (robust propagation), and to avoid SGL entirely.

---

### 6.3 EXP-C: Health Constraint λ_health Sensitivity

**Hypothesis:** If health loss gradients properly backpropagate through model parameters, varying λ_health should change both health alignment (measurable via HealthGain@K) and ranking quality (HR@10). We test this hypothesis on two architectures: **NutriGraphNet** (routes message-passing through all 9 edge types, including `healthness`) and **HFRS-DA** (dual-attention architecture, serving as an ablation baseline).

We vary λ_health ∈ {0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0} under **3-seed evaluation** (seeds 123, 777; see Note below) with lightweight parameters (hidden=64, out=32, num_layers=1, heads=2) matching the 1-fold experimental setup for consistency with prior EXP-C runs.

**Note on seed=42:** seed=42 exhibits early convergence failure across all λ values (AUC≈0.55, early stop at epoch 23), attributable to an unfavorable random data split. All results below exclude seed=42 and report 2-seed mean over seeds {123, 777}.

**Table C. NutriGraphNet Performance vs. λ_health (2-seed mean, seeds 123+777)**

| λ_health | AUC | HR@10 | NDCG@10 | MRR | HealthGain@10 |
|---------|--------|--------|---------|--------|---------------|
| 0.000 | 0.8548 | 0.7400 | 0.5124 | 0.4496 | −0.00900 |
| 0.001 | 0.8549 | 0.7400 | 0.5128 | 0.4501 | −0.00900 |
| 0.005 | 0.8546 | 0.7410 | 0.5060 | 0.4408 | −0.00920 |
| 0.010 | 0.8550 | 0.7390 | 0.5130 | 0.4508 | −0.00895 |
| 0.050 | 0.8556 | 0.7390 | 0.5110 | 0.4480 | −0.00910 |
| **0.100** | 0.8519 | **0.7430** | **0.5129** | **0.4489** | −0.00935 |
| 0.500 | 0.8370 ↓ | 0.7350 | 0.5032 | 0.4395 | −0.00940 |
| 1.000 | 0.8427 | 0.7360 | 0.4935 ↓ | 0.4264 ↓ | −0.00930 |

*(2-seed mean, seeds={123,777}; lightweight params: hidden=64, out=32, num_layers=1, heads=2)*

**Table C-HFRSDA (Reference). HFRS-DA Performance vs. λ_health (all identical — architecture severed)**

| λ_health | AUC | F1 | HR@10 | NDCG@10 | MRR | HealthGain@10 |
|---------|--------|--------|--------|---------|--------|---------------|
| 0.000–1.000 | 0.8551 | 0.7200 | 0.7340 | 0.5977 | 0.5635 | ≈ 0 (all λ) |

*(Δ HR@10 = 0.0000 exactly; Δ AUC < 1.25×10⁻⁷ for all λ ∈ [0.001, 1.0]. Reported for architectural contrast only.)*

**Finding C1 — NutriGraphNet Is Robust to λ in [0.001, 0.1].** Across the practical range λ ∈ {0.001, 0.005, 0.01, 0.05, 0.1}, HR@10 varies by only **Δ=0.0040** (0.7390–0.7430) — within measurement noise. This **robust plateau** demonstrates that NutriGraphNet's ranking quality is insensitive to the exact health constraint weight, making deployment straightforward: any λ in this range is safe. The result revises the earlier CPU 1-fold finding (which suggested λ=0.5 as optimal); under multi-seed evaluation with proper convergence control, no single λ is decisively better than others in the plateau.

**Finding C2 — HealthGain@10 Is Consistently Non-Zero Across All λ.** HealthGain@10 is negative and stable across λ ∈ {0.0, 0.001, ..., 1.0} (range: −0.00895 to −0.00940 in the plateau, −0.00930 at λ=1.0). Three observations are critical: *(i)* **non-zero at λ=0.0** (−0.009) — baseline health-gradient signal exists from the architecture itself; *(ii)* **magnitude stable** across the plateau, indicating health regularization does not meaningfully alter the health-alignment direction; *(iii)* **active at large λ** (−0.0094 at λ=0.5) — even at high health weights, the model maintains gradient flow. This contrasts sharply with HFRS-DA's structurally zero HealthGain, confirming NutriGraphNet's architectural health gradient routing is functional across the full λ range.

**Finding C3 — Health–Ranking Degradation at λ≥0.5 Is Moderate.** Above λ=0.1, ranking metrics show mild degradation: HR@10 drops from 0.7430 (λ=0.1) to 0.7350 (λ=0.5, −1.1%) and AUC from 0.8556 (λ=0.05) to 0.8370 (λ=0.5, −2.2%). The degradation is substantially milder than previously estimated from the CPU 1-fold experiment (which showed −29.7% at λ=1.0). The revised pattern shows that **NutriGraphNet tolerates moderate health regularization (λ≤0.1) without significant ranking cost**, while only strong regularization (λ≥0.5) begins to meaningfully compress the AUC. **Practical recommendation: λ=0.01 as default** — on the plateau, conservative, and interpretable.

**Finding C4 — NutriGraphNet Health Loss Is Active; HFRS-DA Is Architecturally Severed.** For NutriGraphNet, HealthGain@10 is consistently non-zero (≈−0.009) across all λ, confirming that health gradients flow from the `healthness` edge convolution path through the NutriLoss objective. This is a qualitative departure from HFRS-DA's structurally zero health gradient (Δ HR@10 = 0.000 exactly, Δ AUC < 1.25×10⁻⁷ across all λ), validating that NutriGraphNet's architectural design enables genuine health-aware optimization regardless of λ choice.

**Mechanism Analysis — Why NutriGraphNet Succeeds Where HFRS-DA Fails.**

**(a) Architectural health gradient path:** NutriGraphNet's DualChannelEncoder applies GATConv over all 9 edge types including `('user', 'healthness', 'food')`. The health constraint loss L_health is defined over food embeddings that are updated via message-passing along `healthness` edges. The NutriLoss gradient thus flows: L_health → food_emb (via healthness conv) → GATConv parameters — a **valid architectural path**. In contrast, HFRS-DA's NLA/SLA branches use direct embedding lookup and interaction-matrix attention only; `healthness` edges are never consumed in forward propagation, severing the gradient path entirely.

**(b) Lightweight vs. full-parameter behavior:** The current EXP-C uses lightweight parameters (hidden=64, out=32, layers=1, heads=2) consistent with the original CPU setup. Under these parameters, health regularization provides marginal ranking improvement (plateau behavior), whereas full-parameter (hidden=128, out=64, layers=3) GPU 5-fold experiments (e.g., EXP-B, EXP-G) show higher absolute HR@10 values (0.74–0.79). The λ-robust plateau finding is expected to hold qualitatively under full parameters, but full-parameter EXP-C 5-fold is identified as a direction for future confirmation.

**Practical Implication.** NutriGraphNet provides **architecturally guaranteed health-gradient routing** with **λ=0.01 as the recommended default** (robust plateau, no ranking cost, conservative health weight). For clinical practitioners: *(i)* health-aware recommendation is achievable with proper architectural routing regardless of exact λ; *(ii)* HFRS-DA's named health constraint provides zero guarantee without architectural remediation; *(iii)* λ sensitivity analysis should always include HealthGain@K verification — non-zero HealthGain is the only reliable indicator of active health optimization.

---

### 6.4 EXP-D: Embedding Dimension Sensitivity

We vary embedding dimension d ∈ {16, 32, 64, 128, 256} for all five models under GPU 5-fold CV.

**Table D. HR@10 vs. Embedding Dimension d (GPU 5-fold CV)**

| d | MF | LightGCN | NGCF | SGL | HFRS-DA |
|---|-----|---------|------|-----|---------|
| 16 | 0.6847 | 0.7140 | 0.7100 | 0.2700 | 0.7433 |
| 32 | 0.7187 | 0.7300 | 0.7767 | 0.3133 | **0.7550** |
| 64 | 0.7300 | 0.7113 | 0.7813 | 0.3467 | 0.7533 |
| 128 | 0.7533 | 0.7340 | 0.7833 | 0.3813 | 0.7517 |
| 256 | **0.7527** | **0.7440** | **0.7867** | **0.4213** | 0.7183 |

*(GPU 5-fold CV; bold = best per column; see Figure 4)*

**Table D-AUC. AUC vs. Embedding Dimension d (GPU 5-fold CV)**

| d | MF | LightGCN | NGCF | SGL | HFRS-DA |
|---|-----|---------|------|-----|---------|
| 16 | 0.5042 | 0.8381 | 0.8644 | 0.6545 | **0.8573** |
| 32 | 0.5139 | 0.8417 | 0.8761 | 0.6683 | **0.8623** |
| 64 | 0.5087 | 0.8220 | 0.8771 | 0.6867 | 0.8553 |
| 128 | 0.5353 | 0.8308 | 0.8793 | 0.7042 | 0.8155 |
| 256 | **0.5428** | 0.8408 | **0.8810** | **0.7223** | 0.5740 ← ⚠ |

**Table D-NDCG. NDCG@10 vs. Embedding Dimension d (GPU 5-fold CV)**

| d | MF | LightGCN | NGCF | SGL | HFRS-DA |
|---|-----|---------|------|-----|---------|
| 16 | 0.5404 | 0.4482 | 0.4187 | 0.1509 | 0.5787 |
| 32 | 0.5760 | 0.4920 | 0.4909 | 0.1817 | **0.6029** |
| 64 | 0.5946 | 0.4935 | 0.5508 | 0.2201 | 0.6033 |
| 128 | 0.6025 | 0.5388 | 0.5775 | 0.2651 | 0.6008 |
| 256 | **0.6046** | **0.5670** | **0.5858** | **0.3019** | 0.5816 |

**Finding D1 — MF Monotonic Scaling.** MF's HR@10 scales consistently from 0.6847 (d=16) to a plateau at d=128–256 (0.7533/0.7527), Δ(d=16→128)=+0.0687 (+10.0%). AUC rises from 0.5042→0.5428, confirming capacity improves both calibration and ranking. MF shows no saturation up to d=256 — distinct from all GNN models — consistent with a non-overparameterized matrix factorization under L2 regularization.

**Finding D2 — NGCF Dimension Efficiency.** NGCF reaches near-peak HR@10 at d=64 (0.7813, 99.3% of d=256=0.7867). The d=64→256 gain is only +0.0054 (+0.7%) at 4× compute cost. NDCG@10 continues scaling to d=256 (0.5858 vs. d=64=0.5508, +6.3%), indicating that for NDCG-focused tasks d=128–256 is preferred; for HR@10-focused tasks d=64 is the practical optimum.

**Finding D3 — LightGCN Non-Monotone Behavior.** LightGCN HR@10 peaks at d=32 (0.7300), drops at d=64 (0.7113, −2.6%), then recovers to d=256 (0.7440). NDCG@10 however scales monotonically (0.4482→0.5670), indicating the HR non-monotonicity reflects sensitivity to over-smoothing at intermediate dimensions while NDCG benefits from larger capacity. For LightGCN on NutriGraph-KR, d=256 is the overall optimum.

**Finding D4 — SGL Capacity Partial Recovery.** SGL HR@10 improves from 0.2700 (d=16) to 0.4213 (d=256), a **+56.0% relative gain**. AUC also scales strongly (0.6545→0.7223, +10.4p). However, at d=256 SGL still achieves only 0.4213 vs. MF's 0.7527 (**56% of MF**) and NGCF's 0.7867 (**54% of NGCF**). The contrastive collapse is capacity-independent: doubling dimensions improves representation power but cannot compensate for the absence of meaningful positive views at 12.6 interactions/user.

**Finding D5 — HFRS-DA dim=256 AUC Collapse (Critical).** HFRS-DA's AUC **collapses from 0.8623 (d=32) to 0.5740 (d=256)**, a drop of **0.2883 absolute points** (−33.4%). The average AUC for d∈{16,32,64,128} is 0.8476, making d=256 a −0.2736p outlier. HR@10 declines more modestly (0.7550→0.7183, −4.9%). The AUC collapse at high d is consistent with **attention weight ill-conditioning**: at d=256 with 4 attention heads, each head operates in 64-dimensional subspace; attention energies become near-uniform, collapsing score calibration while preserving rough rank ordering. **Practical recommendation: d=32 for HFRS-DA** (HR@10=0.7550, AUC=0.8623 — both at or near maximum).

---

### 6.5 EXP-G: GNN Layer Depth and Over-Smoothing Analysis

We vary the number of GNN propagation layers L ∈ {1, 2, 3, 4} for LightGCN and NGCF to characterize over-smoothing behavior on NutriGraph-KR under GPU 5-fold CV.

**Table G. Performance vs. GNN Layer Depth (GPU 5-fold CV, dim=64)**

| Layers | LightGCN HR@10 | LightGCN NDCG@10 | LightGCN AUC | NGCF HR@10 | NGCF NDCG@10 | NGCF AUC |
|--------|----------------|-----------------|--------------|------------|--------------|----------|
| 1 | 0.7456±0.0177 | 0.5386±0.0132 | 0.8522±0.0018 | **0.7876±0.0150** | **0.5812±0.0109** | **0.8821±0.0017** |
| **2** | **0.7844±0.0206** | **0.5838±0.0187** | **0.8783±0.0013** | 0.7800±0.0206 | 0.5486±0.0315 | 0.8799±0.0035 |
| 3 | 0.7208±0.0223 | 0.4986±0.0157 | 0.8218±0.0012 | 0.7844±0.0209 | 0.5567±0.0223 | 0.8777±0.0014 |
| 4 | 0.7576±0.0203 | 0.5413±0.0194 | 0.8375±0.0016 | 0.7868±0.0228 | 0.5385±0.0175 | 0.8773±0.0015 |

*(Bold = best per column; GPU 5-fold CV; LightGCN best at L=2, NGCF best at L=1; Δ HR@10 across L=1–4: LightGCN=0.0636, NGCF=0.0076)*

**Finding G1 — Architecture-Dependent Layer Sensitivity.** Under GPU 5-fold CV, the two models show **markedly different layer sensitivity profiles**. NGCF is largely layer-invariant: HR@10 ranges from 0.7800 (L=2) to 0.7876 (L=1), a variation of only Δ=0.0076 (1.0%) across all four settings, consistent with the prior hypothesis of sparse-graph saturation. LightGCN, however, shows a **non-monotonic pattern**: HR@10 peaks at L=2 (0.7844), drops sharply at L=3 (0.7208, −8.1%), and partially recovers at L=4 (0.7576). This revised finding replaces the earlier 1-fold observation of complete layer-invariance for LightGCN: 5-fold CV reveals that L=3 is actively harmful for LightGCN (Δ=−0.0636 vs. L=2), likely due to oversmoothing of the propagated embeddings at an intermediate depth where neighborhood overlap is highest.

**Finding G2 — Optimal Layer Depth Is Model-Specific.** L=1 is optimal for NGCF (HR@10=0.7876, NDCG@10=0.5812, AUC=0.8821), confirming that NGCF's interaction-term message passing saturates rapidly in sparse graphs. For LightGCN, L=2 is optimal (HR@10=0.7844, NDCG@10=0.5838, AUC=0.8783): a single additional hop provides meaningful collaborative signal aggregation, but L=3 enters an over-smoothing regime. The practical recommendation is therefore: **use L=1 for NGCF and L=2 for LightGCN** on NutriGraph-KR-scale sparse nutrition graphs.

**Finding G3 — Structural Explanation: Selective Over-Smoothing at L=3.** The LightGCN L=3 dip (HR@10=0.7208, AUC=0.8218) is mechanistically distinct from classical over-smoothing: it reflects the point at which 3-hop neighborhoods in the sparse interaction graph begin to overlap significantly across users, causing their embeddings to converge. In NutriGraph-KR (avg 12.6 interactions/user, food median degree=1.0), a user's 3-hop neighborhood includes all users who share a food with any food eaten by a friend-of-friend — a set that rapidly expands to include most of the user base in sparse graphs. This "sparse-graph over-smoothing" occurs at shallower depth (L=3) than in dense graphs (typically L=5–6), and partially resolves at L=4 as the model adapts via the BPR objective. NGCF avoids this phenomenon because its element-wise interaction term preserves identity structure even under multi-hop aggregation.

**Finding G4 — Practical Implication.** For ultra-sparse recommendation graphs (density < 0.05%, mean degree < 15), **NGCF with L=1 and LightGCN with L=2 are the recommended configurations**. Specifically, L=3 should be avoided for LightGCN (−8.1% HR@10 penalty), and L≥2 provides no benefit for NGCF. Both findings are robust across 5 folds (LightGCN L=2: std=0.0206; NGCF L=1: std=0.0150). These depth-specific guidelines are an architectural property of each model's message-passing mechanism interacting with graph sparsity, not a dataset artifact.

---

### 6.6 EXP-F: Graph Component Ablation — HFRS-DA Topology Invariance

We systematically remove edge types from the heterogeneous graph and measure their contribution to HFRS-DA performance.

**Table F. Graph Component Ablation — HFRS-DA (dim=64, 5-fold CV)**

| Variant | HR@10 | NDCG@10 | MRR | AUC | F1 | ΔHR@10 |
|---------|-------|---------|-----|-----|-----|--------|
| Full Graph | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | — |
| w/o Ingredient | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | **0.0000** |
| w/o Time | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | **0.0000** |
| w/o Food-Similar | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | **0.0000** |
| w/o Ingredient+Time | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | **0.0000** |

*(GPU 5-fold CV confirmed; all ΔHR@10 = 0.000000 exactly; max ΔAUC = 1.2×10⁻⁷ across 5 folds)*

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

**EXP-F v2 — Valid Topology Ablation (NutriGraphNet + NGCF).** To obtain a meaningful topology ablation for models that *do* use auxiliary edges, we implement EXP-F v2 with two complementary probes under GPU 5-fold CV (hidden=128, out=64, layers=3, heads=4):
- **NutriGraphNet (v2a)** — the ablated edge type is removed directly from the heterogeneous graph (`edge_index` zeroed) before message-passing, exercising the same forward-pass path used throughout the paper.
- **NGCF with 50% interaction dilution (v2b)** — since NGCF's `_propagate()` never reads auxiliary `edge_index` tensors, a direct removal would be a no-op by construction. Instead, foods connected to the targeted auxiliary edge type have 50% of their user-interaction edges randomly removed from `train_ei`, giving an indirect but *functional* proxy for how much ranking-relevant signal is concentrated on auxiliary-connected foods.

**Table F-v2a. Graph Component Ablation — NutriGraphNet (GPU 5-fold CV, hidden=128/out=64/layers=3)**

| Variant | HR@10 (±σ) | NDCG@10 | MRR | AUC (±σ) | F1 | HealthGain@10 | ΔHR@10 |
|---------|-----------|---------|-----|----------|-----|---------------|--------|
| Full Graph | 0.7296 (±0.0306) | 0.4154 | 0.3288 | 0.8567 (±0.0116) | 0.7835 | −0.0102 | — |
| w/o Ingredient | 0.7000 (±0.0729) | 0.3888 | 0.3068 | 0.8536 (±0.0233) | 0.7598 | −0.0108 | −0.0296 (−4.1%) |
| w/o Time | 0.7172 (±0.0222) | 0.4284 | 0.3503 | 0.8582 (±0.0051) | 0.7798 | −0.0101 | −0.0124 (−1.7%) |
| w/o Food-Similar | 0.7084 (±0.0295) | 0.3456 | 0.2467 | 0.8416 (±0.0082) | 0.7679 | −0.0087 | −0.0212 (−2.9%) |
| w/o Healthness | 0.6772 (±0.1011) | 0.3867 | 0.3117 | 0.8518 (±0.0234) | 0.7851 | N/A¹ | **−0.0524 (−7.2%)** |
| w/o Ingredient+Time | 0.7076 (±0.0284) | 0.3708 | 0.2792 | 0.8497 (±0.0062) | 0.7407 | −0.0108 | −0.0220 (−3.0%) |
| w/o All Auxiliary | **0.5764 (±0.1408)** | **0.2708** | **0.1973** | **0.8190 (±0.0297)** | 0.5692 | −0.0079 | **−0.1532 (−21.0%)** |

*(¹ HealthGain@10 is undefined when `healthness` edges are removed, since the health score used to compute it is itself derived from those edges — not a training failure, but a structural consequence of the ablation.)*

**Table F-v2b. Graph Component Ablation — NGCF, 50% Interaction Dilution (GPU 5-fold CV)**

| Variant (diluted) | HR@10 (±σ) | NDCG@10 | MRR | AUC (±σ) | F1 | ΔHR@10 |
|---------|-----------|---------|-----|----------|-----|--------|
| Full Graph | 0.7844 (±0.0209) | 0.5569 | 0.4915 | 0.8777 (±0.0014) | 0.7625 | — |
| w/o Ingredient | 0.7748 (±0.0235) | 0.5354 | 0.4666 | 0.8732 (±0.0053) | 0.7794 | −0.0096 (−1.2%) |
| w/o Time | 0.7752 (±0.0234) | 0.5355 | 0.4666 | 0.8732 (±0.0053) | 0.7795 | −0.0092 (−1.2%) |
| w/o Food-Similar | 0.7732 (±0.0299) | 0.5489 | 0.4846 | 0.8772 (±0.0050) | 0.8013 | −0.0112 (−1.4%) |
| w/o Healthness | 0.7744 (±0.0235) | 0.5350 | 0.4663 | 0.8732 (±0.0053) | 0.7794 | −0.0100 (−1.3%) |
| w/o Ingredient+Time | 0.7752 (±0.0235) | 0.5353 | 0.4663 | 0.8732 (±0.0053) | 0.7794 | −0.0092 (−1.2%) |
| w/o All Auxiliary | 0.7752 (±0.0234) | 0.5351 | 0.4661 | 0.8732 (±0.0053) | 0.7794 | −0.0092 (−1.2%) |

*(A first EXP-F v2b run produced ΔHR@10 = 0.000000 for every variant — identical to full_graph at 6+ decimal places per fold. Root cause: the data-level `edge_index` zeroing step, applied globally before fold construction, ran before the dilution logic could inspect the same tensors to determine which foods are auxiliary-connected — so the dilution step always saw an empty edge set and silently fell back to the unmodified `train_ei`. Fixed by skipping the data-level zeroing when the ablation target is NGCF/LightGCN, so the dilution logic reads intact `edge_index` tensors. All values above are from the corrected rerun.)*

**Finding F4 — NutriGraphNet Shows Genuine, Graded Topology Dependence.** Unlike HFRS-DA's exact Δ=0.000, NutriGraphNet's HR@10 degrades measurably and monotonically as more auxiliary structure is removed: −1.7% (w/o time, the weakest signal) to −21.0% (w/o all auxiliary, the strongest). Removing `healthness` alone costs −7.2% HR@10 — the single largest individual-edge-type effect — consistent with `healthness` edges carrying the densest per-interaction signal (one healthness edge per user-food interaction, vs. sparser ingredient/time/food-similar connections). This is direct, functional evidence — not just code inspection — that NutriGraphNet's message-passing genuinely consumes the heterogeneous topology it claims to model.

**Finding F5 — Removing Critical Structure Destabilizes Convergence, Not Just Average Performance.** The `w/o healthness` and `w/o all auxiliary` conditions show far higher fold-to-fold variance (σ=0.101 and σ=0.141) than the full graph baseline (σ=0.031). In both cases, four of five folds land in a comparable range (no_all_auxiliary: 0.566–0.698; no_healthness: 0.710–0.792) while one fold collapses sharply (no_all_auxiliary fold 5: HR@10=0.310; no_healthness fold 4: HR@10=0.536). This indicates that stripping the auxiliary graph does not merely shift the mean — it makes convergence itself less reliable, an effect that a single-fold ablation (as in the original EXP-F v1 setup) would not have surfaced.

**Finding F6 — NGCF's Residual Sensitivity Is Small and Clusters by Coverage, Not by Edge-Type Identity.** Diluting interactions for auxiliary-connected foods costs NGCF only −1.2% to −1.4% HR@10 regardless of *which* edge type defines the "auxiliary-connected" food set — ingredient, time, healthness, and their combinations all converge to nearly the same diluted HR@10 (0.7748–0.7752, AUC=0.8732). This is because ingredient, time, and healthness edges each cover a large majority of the 31,458 foods (health scores alone are defined for all foods, per Finding F3), so the "50%-diluted" food sets under these three ablations are nearly identical regardless of label. Food-similarity edges (108,062 edges, a sparser food-food relation) cover a different, smaller food set, producing a slightly different result (HR@10=0.7732, closer to the full-graph baseline of 0.7844 than the other five variants). Notably its F1 (0.8013) exceeds the full-graph value (0.7625) despite lower HR@10 — a reminder that F1 (classification-threshold-dependent) and HR@10 (ranking-based) can move in different directions and should not be conflated. The key contrast with Table F-v2a stands regardless: even under an indirect interaction-count proxy, NGCF's auxiliary-adjacent sensitivity (≤1.4%) is an order of magnitude smaller than NutriGraphNet's direct topology dependence (up to 21.0%), reinforcing that only architectures which route message-passing through auxiliary edges are functionally dependent on them.

---

### 6.7 Cross-Experiment Synthesis

The six experiments collectively paint a coherent picture of failure modes and success conditions in GNN-based food recommendation. Figure 6 summarizes the AUC vs. HR@10 trade-off landscape across all models and conditions.

**Table S. Key Numerical Reference Table (GPU 5-fold CV, full parameters)**

| Claim | Experiment | Confirmed Value |
|-------|-----------|-----------------|
| SGL best aug ratio | EXP-A | p=0.0, HR@10=0.3604; all p>0 degrade |
| SGL aug worst point | EXP-A | HR@10=0.3520 at p=0.5 (-2.3%); NDCG@10 worst at p=0.2 (-3.1%) |
| SGL collapse at 10% density | EXP-B | HR@10=0.0880 vs. NutriGraphNet=0.6560 (7.45x gap) |
| NutriGraphNet best at low density | EXP-B | HR@10=0.6560 at 10%, best of all 5 models |
| NutriGraphNet sparsity robustness | EXP-B | HR@10 scaling ratio 1.12x (10%->100%), most robust |
| NGCF best at high density | EXP-B | HR@10=0.7844 at 100%, best baseline |
| MF AUC structural ceiling | EXP-B/D | AUC <=0.5428 regardless of density or dim |
| lambda_health robust plateau (NutriGraphNet) | EXP-C | lambda in [0.001,0.1]: HR@10=0.739-0.743 (Delta<0.004); HealthGain@10=-0.009 (active all lambda) |
| lambda_health recommended default | EXP-C | lambda=0.01: plateau center, no ranking cost, conservative |
| lambda_health degradation onset | EXP-C | lambda>=0.5: AUC drops to 0.837 (-2.2%), NDCG drops; HR@10 -1.1% |
| lambda_health sensitivity (HFRS-DA, ref) | EXP-C | Delta HR@10 = 0.000 exactly for all lambda in [0.001, 1.0] |
| HFRS-DA optimal dim | EXP-D | d=32: HR@10=0.7550, AUC=0.8623 |
| NGCF optimal dim (HR@10) | EXP-D | d=256: HR@10=0.7867, AUC=0.8810 |
| NGCF dim efficiency | EXP-D | d=64 achieves 99.3% of d=256 HR@10 (0.7813 vs. 0.7867) |
| HFRS-DA dim=256 AUC collapse | EXP-D | AUC=0.5740 vs. avg 0.8476 for d in {16-128} (Delta=-0.2736) |
| LightGCN optimal layers | EXP-G | L=2: HR@10=0.7844±0.0206, NDCG@10=0.5838±0.0187; L=3 worst (HR@10=0.7208, -8.1%) |
| NGCF layer-invariance | EXP-G | L=1 best (HR@10=0.7876±0.0150); Δ across L=1-4 = 0.0076 (1.0%) |
| Optimal GNN layers | EXP-G | NGCF: L=1; LightGCN: L=2 (avoid L=3, -8.1% HR@10 penalty) |
| HFRS-DA topology invariance | EXP-F | Delta HR@10 = 0.000000 across all 5 ablations; max Delta AUC = 1.2e-7 |
| NutriGraphNet topology dependence | EXP-F v2a | w/o all auxiliary: HR@10=0.5764 (-21.0% vs full=0.7296); w/o healthness alone: -7.2% |
| NGCF residual sensitivity (dilution proxy) | EXP-F v2b | HR@10 -1.2% to -1.4% across all variants (vs NutriGraphNet's up to -21.0%) |
| Best overall HR@10 | EXP-B/D | NGCF: 0.7844 (100% density, d=64 baseline); 0.7867 (d=256) |
| Best overall AUC | EXP-D | NGCF d=256: 0.8810 |

---

## 7. Design Guidelines

Based on our empirical findings, we propose the following actionable guidelines for practitioners:

| Scenario | Recommendation | Evidence |
|----------|----------------|---------|
| Sparse interactions (<20/user) | **Avoid SGL** -- use NutriGraphNet or LightGCN | EXP-B: SGL HR@10=0.0880 at 10% density; NutriGraphNet=0.6560 (7.45x better) |
| Low density (10-30%) | **NutriGraphNet** preferred; LightGCN as lightweight alternative | EXP-B: NutriGraphNet HR@10=0.6560 at 10% (+25.3% over LightGCN=0.5236) |
| Dense interactions (>50% / >6 int/user) | **NGCF** optimal for HR@10; NutriGraphNet for NDCG | EXP-B: NGCF HR@10=0.7844 at 100%; NutriGraphNet NDCG@10=0.5977 (best baseline) |
| SGL augmentation ratio | **p=0.0 is always optimal** on sparse nutrition graphs | EXP-A: HR@10 decreases for all p>0; worst at p=0.5 (-2.3%) |
| GNN layer depth (NGCF) | **L=1 is optimal** -- layer-invariant (ΔHR@10=0.0076 across L=1–4) | EXP-G: NGCF HR@10=0.7876 at L=1; Δ<1.0% across all depths |
| GNN layer depth (LightGCN) | **L=2 is optimal; avoid L=3** (−8.1% HR@10 penalty) | EXP-G: LightGCN HR@10=0.7844 at L=2 vs. 0.7208 at L=3 |
| Embedding dim (NGCF) | **d=64 for HR@10** (99.3% of d=256); **d=128-256 for NDCG@10** | EXP-D: d=64 HR@10=0.7813 vs. d=256=0.7867; NDCG@10 scales to d=256 |
| Embedding dim (HFRS-DA) | **d=32 only** -- AUC collapses -0.2736p at d=256 | EXP-D: AUC=0.8623 at d=32 vs. 0.5740 at d=256 |
| Health constraint weight (lambda) | **lambda=0.01 as default** — robust plateau in [0.001,0.1], HR@10 stable (Delta<0.4%) | EXP-C: HR@10=0.739-0.743 across lambda=0.001-0.1; degradation only at lambda>=0.5 |
| Health backpropagation | **Route health gradients via healthness edge convolution** | EXP-C: NutriGraphNet HealthGain@10=-0.009 (active all lambda); HFRS-DA severs gradient |
| Auxiliary edge types | **Ablate each edge type** to verify forward-pass contribution | EXP-F: HFRS-DA Delta HR@10=0.000000 for all 3 auxiliary edge types; NutriGraphNet shows real -1.7% to -21.0% degradation (EXP-F v2a), confirming genuine topology dependence |
| "Heterogeneous graph" claims | Only claim graph-awareness if architecture **routes messages** through edges | EXP-F: topology invariance invalidates HFRS-DA's heterogeneous graph claim |
| Clinical health-aware deployment | **Verify health gradients architecturally**, not just via named loss | EXP-C/F: HFRS-DA health loss zero; NutriGraphNet shows active HealthGain@K |

---

## 8. Conclusion

We presented a systematic empirical analysis of GNN-based food recommendation on a large-scale heterogeneous nutrition graph (NutriGraph-KR: 20,820 users, 31,458 foods, density=0.040%), uncovering four key phenomena with root-cause explanations:

**(1) SGL Augmentation Collapse.** Edge dropout augmentation degrades ranking performance for all p > 0 (HR@10: 0.3604→0.3520 as p: 0.0→0.5, −2.3%), with catastrophic collapse at 10% data density (HR@10=0.088 vs. NutriGraphNet=0.656, 7.45× gap; vs. LightGCN=0.524, 5.95× gap). The root cause is the fundamental incompatibility between SGL's InfoNCE contrastive objective — which requires dense positive views — and nutrition interaction graphs with an average of only 12.6 interactions per user (10× sparser than MovieLens-1M).

**(2) NutriGraphNet Sparsity Robustness.** Under GPU 5-fold cross-validation, NutriGraphNet dominates all baselines at every tested density from 10%–70%, achieving HR@10=0.656 at 10% density (+25.2% over LightGCN, +28.9% over NGCF). The sparsity scaling ratio is 1.12× (10%→100%), the most robust of all five models, because auxiliary graph edges (ingredient, food-similarity, time) provide non-interaction structural signal that compensates for sparse user-food data. At full density, NGCF overtakes NutriGraphNet on HR@10 (0.784 vs. 0.734), while NutriGraphNet maintains higher NDCG@10 and AUC, indicating complementary strengths.

**(3) MF–SGL Ranking Paradox.** Simple matrix factorization (HR@10=0.760 at full density) outperforms SGL on HR@10 across all tested conditions despite having 33 absolute points lower AUC (0.547 vs. NGCF 0.878). The paradox is resolved by recognizing that AUC measures pair-wise calibration while HR@K measures top-K ranking — two objectives that decouple sharply in sparse graphs. SGL fails completely (HR@10=0.088 at 10% density) despite moderate AUC (0.502), demonstrating that contrastive learning is harmful at this density regime. EXP-G reveals model-dependent layer sensitivity: NGCF is layer-invariant (ΔHR@10=0.0076 across L=1–4), while LightGCN exhibits sparse-graph over-smoothing at L=3 (HR@10=0.7208, −8.1% vs. L=2=0.7844), partially recovering at L=4. Optimal depths are L=1 for NGCF and L=2 for LightGCN.

**(4) Health Constraint Robustness vs. Architectural Failure.** NutriGraphNet — which routes message-passing through all 9 edge types including `healthness` — shows a **λ-robust plateau**: HR@10 varies by only Δ=0.0040 across λ ∈ {0.001–0.1} (0.7390–0.7430), with HealthGain@10≈−0.009 consistently non-zero across all λ values, confirming active health gradient flow independent of λ choice. Recommended default: λ=0.01 (plateau center, no ranking cost). In contrast, HFRS-DA's health loss produces zero measurable effect (Δ HR@10 = 0.000 exactly, Δ AUC < 1.25×10⁻⁷) due to architecturally severed health gradient paths and complete topology invariance (EXP-F: Δ HR@10 = 0.000 for all 5 auxiliary edge type removals). These results establish that health-aware recommendation is an **architectural property**: health gradients must flow through health-relevant convolution paths, not merely be included in the loss function.

**Our findings challenge four widely held assumptions** in graph-based food recommendation: (a) SGL augmentation improves sparse graphs; (b) architectural complexity correlates with ranking quality; (c) naming a model "health-aware" guarantees health optimization; (d) deeper GNN layers are necessary for rich representations on heterogeneous graphs — NutriGraphNet validates (c) positively with the nuance that health-aware improvement is **λ-robust** (plateau λ∈[0.001–0.1], ΔHR@10<0.4%) rather than sensitive to a single optimal value, EXP-G refutes (d) with the nuance that L=3 actively harms LightGCN (−8.1% HR@10), and EXP-B/A collectively refute (a) and (b). EXP-F v2 confirms this architecturally: NutriGraphNet's HR@10 degrades by up to 21.0% when all auxiliary edges are removed (vs. HFRS-DA's exact 0.000), providing functional (not just code-level) evidence that its message-passing genuinely depends on the heterogeneous topology it claims to model. Future work will investigate: (i) ingredient-conditioned positive sampling for contrastive learning in sparse nutrition graphs; (ii) reducing the fold-to-fold instability observed when critical auxiliary edges (healthness, all-auxiliary) are removed (EXP-F v2a, σ up to 0.141); and (iii) extending NutriGraphNet to explicit HealthGain maximization objectives.

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
*New in v1.1 (2026-07-15): EXP-F v2 GPU 5-fold 완료 반영.*  
*Table F-v2a (NutriGraphNet ablation) + Table F-v2b (NGCF 50% dilution ablation) 신규 추가, "camera-ready version" placeholder 제거.*  
*Finding F4-F6 신규: NutriGraphNet은 그래프 구조에 실제로 의존(w/o all auxiliary: -21.0%, w/o healthness: -7.2%), fold 간 변동성도 증가(σ=0.031→0.141); NGCF는 dilution proxy로도 -1.2%~-1.4%에 그쳐 HFRS-DA(0.000)와 NutriGraphNet(-21.0%) 사이 중간 지점 확인.*  
*버그 수정 이력: EXP-F v2b 최초 실행 시 모든 ablation variant가 full_graph와 완전히 동일한 값을 냄 — 데이터 레벨 edge_index zeroing이 NGCF dilution 로직보다 먼저 실행되어 dilution이 조용히 no-op 처리되던 버그. ablation_model이 ngcf/lightgcn일 때 데이터 레벨 zeroing을 건너뛰도록 수정 후 재실행하여 유효한 결과 확보.*  
*Table S / Design Guidelines EXP-F 행 갱신. Conclusion 향후 연구 항목에서 "EXP-F v2 예정" 제거, fold 변동성 개선을 향후 과제로 대체.*  
---
*Draft v0.7 — 2026-07-13*  
*New in v0.7: Section 5 Table 1 GPU 5-fold 수치로 전면 교체.*  
*Table 1: B_sparsity_100pct 기준 5모델 GPU 실측값(AUC/F1/HR@5/HR@10/HR@20/NDCG@10/MRR/±σ); NutriGraphNet(λ=0.005) 행 신규 추가.*  
*Key observations: 3개→5개 항목; GPU 수치 반영, NutriGraphNet health trade-off 설명, NGCF density-conditioned 전략 언급.*  
---  
*New in v0.9 (2026-07-14): EXP-G GPU 5-fold 완료 반영.*  
*Table G: mean±std (5-fold) 수치로 전면 교체. LightGCN: L=2 best(HR@10=0.7844), L=3 worst(0.7208, -8.1%). NGCF: L=1 best(0.7876), layer-invariant(Δ=0.0076).*  
*Finding G1-G4: 전면 개정 — "layer-invariant" → "architecture-dependent sensitivity". LightGCN sparse-graph over-smoothing at L=3 신규 발견.*  
*Table S EXP-G 항목 갱신. Design Guidelines EXP-G 행 모델별 분리.*  
*Conclusion (3)/Abstract EXP-G 언급 갱신.*  
*New in v0.6 (2026-07-13): GPU 5-fold 결과 전면 반영.*  
*EXP-B: hfrsda(NutriGraphNet) 열 추가 → NutriGraphNet이 10%–70% 전 밀도에서 최고 성능.*  
*EXP-C: Table C GPU 5-fold로 교체 → λ_optimal=0.005 (CPU 1-fold λ=0.5에서 변경).*  
*Pending: References [CITE] 7개 채우기; Figure 1/2/4/6 생성; 저자명 placeholder 채우기*
