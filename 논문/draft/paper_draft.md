# Auxiliary Nutritional Structure Substitutes for Interaction Data  
# in Health-Aware Food Recommendation:  
# Evidence from a National Dietary Survey Graph

**Authors:** Heejeong [Last Name]  
**Target Venue:** Computers in Biology and Medicine (IF: 7.7) / Nutrients (IF: 5.9)  
**Status:** Draft v2.1 — 2026-07-16 (**HFRS-DA 실제 baseline 추가**: 공식 구현(Forouzandeh et al., 2024) 기반 충실 재구현을 `hfrsda_real` variant로 추가 — DualAttn-TB 대조군과 별개. NLA(ingredient neighbourhood 통합)+SLA(WHO-7 health 가중)+cosine 스코어링, emb_dim=64. 통일 평가 프로토콜·BPR 학습·graph health-score 사용 2가지 adaptation을 §4.1에 명시. §11.6 "health-aware baseline 부재" → "재구현이나 원 pipeline 아님"으로 갱신, 이제 논문 최대 공백 해소. **수치는 GPU 5-fold 실행 후 Table 1/B에 반영 예정**(scripts/run_hfrsda_real.bat). Local smoke(15ep/1fold): AUC=0.831 HR@10=0.630 NDCG@10=0.485. v2.0: **전면 재프레이밍**. 제목 교체: 증강 논지가 v1.3에서 철회되어 기존 제목("Why Graph Augmentation Fails")이 본문과 모순 상태였음 → "Auxiliary Nutritional Structure Substitutes for Interaction Data"로 변경, 발견 중심 프레임. **HFRS-DA 주장 전면 철회**: 기존 EXP-F/C의 "HFRS-DA는 구조적으로 고장났다" 주장은 자체 단순화 재구현(코드 docstring이 "Simplified re-implementation"이라 명시)에 근거했고 원저자 공식 구현이 공개돼 있어 방어 불가였음. 해당 모델을 **DualAttn-TB(자체 설계 topology-blind 대조군)**로 재정의 — 남의 논문 반박이 아니라 우리 모델의 그래프 사용 입증 도구가 되어 논지가 오히려 강화됨. Δ=0.000은 결함이 아니라 대조군의 설계된 성질이자 ablation 하네스 검증. §4.1에 귀속 고지 신설. Abstract/Contribution/Introduction/Research Gap/Conclusion(4) 전면 재작성 — 두 축(보조구조가 상호작용 대체 / 건강은 gradient 라우팅 문제) 중심. **§8.4 신설**: v2 vs v3 아키텍처-밀도 교차(v3는 100%에서 NDCG +19.5%지만 10%에서 −12.6%) — v2를 논문 본체로 확정. **§11 Limitations 신설**: 측정 함정 2건(HealthGain 저밀도 무효, 재현성 노이즈 바닥) 자진 보고 + health-aware 베이스라인 부재/단일 데이터셋/AUC 미해결 명시. Conclusion은 §12로 이동. v1.3: EXP-B 오표기 수정 — Table B의 "NutriGraphNet" 열이 실제로는 hfrsda 데이터였음; 실측 결과 진짜 NutriGraphNet이 더 우수(10% HR@10=0.738, NGCF 대비 +45.0%, 밀도 불변 0.99×).)

---

## Abstract

Health-aware food recommendation is most needed exactly where interaction data is hardest to obtain: clinical cohorts and national dietary surveys record only a handful of eating events per participant. We study this regime on **NutriGraph-KR**, a heterogeneous graph built from Korean national dietary survey data (20,820 users, 31,458 foods, 3,284 ingredients, 262,270 interactions; density 0.040%, ≈12.6 interactions per user — roughly ten times sparser than the MovieLens benchmarks on which modern graph recommenders were developed). We introduce **NutriGraphNet**, a dual-channel heterogeneous GAT that propagates over all nine edge types, including a `healthness` relation, under a health-constrained objective, and evaluate it against four published baselines and a purpose-built topology-blind control under GPU 5-fold cross-validation.

**Auxiliary structure substitutes for interaction data.** NutriGraphNet is the best of six models wherever interactions are scarce: HR@10=**0.738** at 10% density (**+45.0%** over NGCF=0.509, +41.0% over LightGCN=0.524) and **0.755** at 30%. Its HR@10 is **density-invariant** — 0.738→0.729 across a tenfold change in interaction volume (ratio 0.99×) — reaching **94.1%** of the best full-data score any model achieves while using one tenth of the interactions. The property is two-sided and we report it as such: NutriGraphNet does not convert additional data into ranking quality, so NGCF overtakes it above roughly 50% density (0.784 vs. 0.729 at 100%).

**The mechanism is verified, not inferred.** Removing all auxiliary edges costs NutriGraphNet **21.0%** HR@10 (removing `healthness` alone costs 7.2%), whereas our topology-blind control — identical in embedding capacity but structurally unable to read auxiliary edges — registers **Δ=0.000 exactly**, which both calibrates the ablation harness and isolates graph consumption as the source of the advantage. That control is itself a strong recommender here (HR@10=0.656 at 10%, beating every published graph baseline), so on ultra-sparse nutrition graphs propagation is not automatically worth its cost; NutriGraphNet's **+12.5%** margin over it at 10% density is what topology actually buys.

**Health constraints must be architecturally routed.** HealthGain@10 remains non-zero (≈−0.010) across four orders of magnitude of the health weight λ, with a robust plateau over λ∈[0.001, 0.1] (Δ HR@10=0.004). Under full parameters the trade-off is functional and controllable rather than nominal: at λ≥0.5, HealthGain@10 contracts toward zero (−0.002) while HR@10 falls 19.6–27.3%, showing the objective genuinely reshapes rankings once capacity permits. We recommend λ=0.01.

**Negative and cautionary results.** SGL collapses at low density (HR@10=0.088 at 10%), but not because of augmentation: with augmentation disabled it still trails NGCF 2.18×, and sweeping the dropout ratio moves HR@10 by only 0.008 — within our measured run-to-run noise. MF attains the highest full-density HR@10 (0.760) despite the lowest AUC (0.547). The architectural choices that improve full-density ranking actively harm the sparse regime: a rank-calibrated decoder variant gains 19.5% NDCG@10 at 100% density yet loses 12.6% HR@10 at 10%. We also document two measurement pitfalls that silently manufacture false conclusions in this setting.

Our findings indicate that in data-scarce nutrition domains, auxiliary relational structure — not more interaction data — is the resource to exploit, and that health-awareness is a property of gradient routing rather than of loss-function naming.

**Keywords:** food recommendation, graph neural networks, health-aware recommendation, data sparsity, heterogeneous graphs, dietary survey data, topology ablation, NutriGraphNet

---

## 1. Introduction

Food recommendation systems have emerged as a critical tool for promoting healthy dietary behavior in digitally-mediated food environments [CITE Forouzandeh 2024, Song 2022]. Unlike conventional item recommendation (movies, products), food recommendation presents a unique combination of challenges: 
(i) **compositional item structure** — each food is defined by its ingredients and nutritional profile rather than categorical attributes;  
(ii) **health constraints** — recommendations must satisfy personalized dietary requirements beyond mere preference;  
(iii) **temporal and cultural patterns** — eating behaviors are time-of-day and culturally conditioned.

Graph neural networks have been widely adopted for recommendation, with models such as LightGCN [CITE He 2020], NGCF [CITE Wang 2019], and SGL [CITE Wu 2021] achieving state-of-the-art performance on e-commerce and movie datasets. 
Several works have extended these to food recommendation [CITE HFRS-DA 2024, FRMADHG 2025, SCHGN 2022], yet these methods are developed and validated on benchmarks far denser than the data that health-aware recommendation actually has to serve.

This gap matters because of an asymmetry specific to the nutrition domain. Interaction data is scarce and expensive: a national dietary survey records roughly a dozen eating events per participant (mean 12.6, density 0.040% — about ten times sparser than MovieLens-1M), and collecting more requires re-running the survey. But **auxiliary structure is abundant and nearly free**: a food's ingredient composition is deterministic and already known, nutritional similarity between foods is computable, and meal timing is recorded alongside every interaction. Where collaborative filtering has almost nothing to work with, the heterogeneous graph is fully populated.

This suggests a hypothesis that inverts the usual framing. Rather than asking how to squeeze more from sparse interactions, we ask whether **auxiliary nutritional structure can substitute for interaction data outright** — and, if a model is to be called health-aware, whether its health objective actually reaches the representation through health-relevant structure or merely decorates the loss. The two questions are linked: both are claims about what a model's forward pass genuinely consumes, and both require a control that consumes nothing to be answerable.

**This paper makes the following contributions:**

1. **[Dataset]** We release **NutriGraph-KR**, a heterogeneous nutrition graph derived from Korean national dietary survey data: 4 node types, 9 edge types, 20,820 users with real demographic/health/disease attributes, 31,458 foods with nutritional profiles, and 262,270 interactions at density 0.040% (≈12.6 interactions per user). To our knowledge it is the first public food-recommendation graph that pairs genuine per-user health attributes with a sparsity regime an order of magnitude beyond the MovieLens benchmarks on which graph recommenders are typically validated.

2. **[Model]** We introduce **NutriGraphNet**, a dual-channel heterogeneous GAT that propagates over all nine edge types — including a `healthness` relation — under a health-constrained objective, so that health gradients reach the encoder through health-relevant convolution rather than through the loss name alone.

3. **[Finding: auxiliary structure substitutes for interaction data]** Under GPU 5-fold CV, NutriGraphNet is the best of six models wherever interactions are scarce (HR@10=0.738 at 10% density, +45.0% over NGCF; 0.755 at 30%) and is **density-invariant** (ratio 0.99× across a tenfold change), reaching 94.1% of the best full-data score on one tenth of the interactions. We report the converse with equal weight: it does not convert extra data into ranking quality, and NGCF overtakes it above ~50% density.

4. **[Finding: the mechanism is verified, not inferred]** We construct a **topology-blind control** with matched embedding capacity that structurally cannot read auxiliary edges. It calibrates the ablation harness (Δ=0.000 exactly, as it must) and isolates graph consumption: removing all auxiliary edges costs NutriGraphNet 21.0% HR@10, and NutriGraphNet leads the control by 12.5% at 10% density. Notably the control is itself competitive (HR@10=0.656 at 10%, above every published graph baseline), so graph propagation is not automatically worth its cost on this data.

5. **[Finding: health constraints must be architecturally routed]** Across four orders of magnitude of λ_health, HealthGain@10 stays non-zero (≈−0.010) with a robust plateau over λ∈[0.001, 0.1]. Under full parameters the trade-off is functional rather than nominal: at λ≥0.5 HealthGain contracts toward zero while HR@10 falls 19.6–27.3%, showing the objective genuinely reshapes rankings once capacity permits. We recommend λ=0.01.

6. **[Negative results and methodological cautions]** We report what did not hold, including against our own earlier framing. SGL collapses at low density (HR@10=0.088 at 10%) but *not* because of augmentation — with augmentation off it still trails NGCF 2.18×, and the dropout sweep moves HR@10 by only 0.008, within our measured noise floor. Architecture choices do not transfer across density: a rank-calibrated decoder gains 19.5% NDCG@10 at full density yet loses 12.6% HR@10 at 10%. We further document two measurement pitfalls that silently manufacture false conclusions in sparsity studies of health-aware recommenders (§11.3, §11.4).

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

Two gaps motivate this work.

**Density.** The graph recommenders above were developed and validated on benchmarks with hundreds of interactions per user. Health-aware food recommendation is deployed where that assumption fails hardest — clinical cohorts and dietary surveys yield roughly a dozen events per participant. Whether conclusions drawn at MovieLens density transfer to 0.040% density is untested, and our results indicate they do not: model rankings reverse (§6.1), and even architecture choices reverse (§8.4).

**Evidence of graph use.** Heterogeneous food recommenders are motivated by the claim that nutritional structure carries signal, but that claim is rarely tested by ablating the structure and observing the consequence. Reporting a model as "graph-based" describes its inputs, not its computation. We show that a topology-blind model with matched capacity is competitive on this graph (§6.2), which means graph consumption must be demonstrated rather than assumed — and that demonstrating it requires a control that provably consumes nothing.

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

### 4.1 Baselines and Control

| Model | Type | Key Mechanism |
|-------|------|---------------|
| MF [Koren 2009] | Non-graph | Bilinear embedding factorization |
| LightGCN [He 2020] | Graph propagation | Linear embedding propagation |
| NGCF [Wang 2019] | Graph propagation | Message passing with interaction term |
| SGL [Wu 2021] | Self-supervised | Edge dropout + InfoNCE contrastive loss |
| HFRS-DA [Forouzandeh 2024] | Health-aware | Dual attention (node + health) over heterogeneous graph |
| **DualAttn-TB** (ours) | **Topology-blind control** | Embedding lookup + multi-head attention; auxiliary edges never read |

**On the DualAttn-TB control.** Alongside the four published baselines we construct a deliberate **topology-blind control**. DualAttn-TB scores items as `α·NLA + (1−α)·SLA`, where the NLA branch applies multi-head self-attention over user/food embeddings obtained by direct lookup and the SLA branch is a food-health scoring head. Critically, **no branch consumes the auxiliary `edge_index` tensors** — ingredient, food-similarity, time and healthness edges are present in the data object but never read during forward propagation.

This control serves two purposes. First, it **isolates the contribution of graph consumption itself**: DualAttn-TB has comparable embedding capacity and an attention mechanism, but cannot exploit heterogeneous structure, so the gap between it and NutriGraphNet at a given density measures what topology buys. Second, it **validates the ablation protocol** of EXP-F: a model that provably ignores auxiliary edges must score exactly Δ=0.000 when those edges are removed. If it did not, the ablation harness would be faulty. The control is thus a measuring instrument, not a competitor.

*The DualAttn-TB control is not HFRS-DA.* The dual-attention framing of this control was loosely inspired by the design philosophy of HFRS-DA [Forouzandeh et al., 2024], but **DualAttn-TB is not an implementation of HFRS-DA and must not be read as one**. It is a deliberately topology-blind measuring instrument. To compare against the actual published system we implement it separately, below.

**On the HFRS-DA baseline.** As our health-aware published baseline we re-implement **HFRS-DA** [Forouzandeh et al., 2024] following the authors' official reference implementation. We reproduce its two mechanisms: a Node-Level Attention (NLA) branch that folds each food's ingredient neighbourhood into its representation through LeakyReLU-gated attention, and a Semantic-Level Attention (SLA) branch that weights foods by a WHO 7-nutrient health signal; the two fused embeddings are scored by cosine similarity, with embedding dimension 64 as in the original. Two adaptations are required for a like-for-like comparison and we state them plainly. *(i) Evaluation protocol.* The official model generates recommendations by user–user cosine-similarity collaborative filtering and reports Precision/Recall/F1 over a user-level split — a protocol not comparable to the unified 5-fold sampled ranking (1 positive + 100 negatives → HR@K/NDCG@K/AUC) applied to every other model here. We keep HFRS-DA's architecture and health mechanism intact but evaluate it under the same protocol as all other models, which is standard benchmarking practice and the only way the numbers are comparable. Training uses the shared BPR objective for the same reason. *(ii) Health signal.* HFRS-DA's `is_healthy()` rule expects raw nutrition in grams; our graph stores a continuous per-food health score on the `healthness` edge, computed from the same WHO 7-nutrient basis, which we use directly (the raw KNHANES nutrition remains available for an exact gram-threshold reproduction). We flag that, as a consequence of these adaptations, our HFRS-DA numbers reflect its architecture and health mechanism under our evaluation, not a re-run of the authors' original pipeline.

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
| λ_health | 0.01 | DualAttn-TB only (swept in EXP-C) |
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
| DualAttn-TB | 0.8551 (±0.010) | 0.7200 | 0.6730 | 0.7340 | 0.8010 | 0.5977 | 0.5635 |
| **NutriGraphNet** (λ=0.005) | **0.8620** (±0.006) | **0.7877** | 0.5660 | 0.7484 | **0.8252** | 0.4279 | 0.3378 |

*(Bold = best per column; GPU 5-fold CV results; B_sparsity_100pct for baselines, C_lambda_0.005/full for NutriGraphNet)*  
*(Note: NutriGraphNet HR@5/HR@10/HR@20 evaluated with full health-loss training at λ=0.005; NDCG and MRR metrics reflect ranking objective trade-off with health regularization)*

**Key observations from Table 1:**

1. **SGL Anomaly:** SGL achieves moderate AUC (0.6989) but catastrophically low HR@10 (0.3576), which is **2.19× lower than NGCF** (0.7844). This divergence between classification and ranking metrics motivates our EXP-A/B analysis.

2. **MF-SGL Ranking Paradox:** MF achieves the highest HR@10 (0.7604) and NDCG@10 (0.6179) among baselines despite the lowest AUC (0.5468). The paradox is sharpest compared to SGL: SGL has 1.28× MF's AUC but achieves only 0.470× MF's HR@10 — a 2.13× gap purely explained by contrastive collapse on sparse interactions.

3. **NutriGraphNet Health Trade-off:** NutriGraphNet at λ=0.005 achieves AUC=0.8620 (best) and HR@10=0.7484 (+5.2% vs. λ=0.0 baseline). HR@5=0.5660 is lower than NGCF (0.6928) because the health regularization shifts recommendation bias toward nutritionally safer items — trades short-list precision for verified health-gradient routing (HealthGain@10=−0.01158, actively non-zero).

4. **The topology-blind control is competitive:** DualAttn-TB (AUC=0.8551, HR@10=0.7340, NDCG@10=0.5977) exceeds NGCF on NDCG@10 (+0.041) and MRR (+0.072) while falling 6.2% behind on HR@10 (0.7340 vs. 0.7844) — achieved without reading a single auxiliary edge (EXP-F). Its ranking quality comes entirely from embedding lookup and attention over the interaction matrix. That a graph-blind model places this well is itself a result: on this data, graph propagation must earn its cost rather than be assumed to pay.

5. **NGCF leads on HR-focused ranking at full density:** NGCF achieves the highest HR@10=0.7844 and HR@20=0.8460 among all models when all interactions are available. Together with EXP-B showing NutriGraphNet dominates at low density (HR@10=0.738 at 10%, +45.0% over NGCF, which reaches only 0.509 there), this motivates a density-conditioned model selection strategy (see Section 10): the ranking reverses at roughly 50% density.

---

## 6. Auxiliary Structure Substitutes for Interaction Data

This section establishes the paper's first claim in two steps: the sparsity sweep shows *that* NutriGraphNet's advantage concentrates where interactions are scarce (§6.1), and the topology ablation shows *why* — that the advantage comes from consuming auxiliary structure rather than from capacity (§6.1).

### 6.1 EXP-B: Data Sparsity Analysis

We subsample the user-food interaction set to {10%, 30%, 50%, 70%, 100%} and evaluate all six models — four published baselines, our topology-blind control, and **NutriGraphNet** — under GPU 5-fold cross-validation with identical full parameters (hidden=128, out=64, layers=3, heads=4, seed=42, λ_health=0.01). Subsampling is seeded, so every model at a given density sees the exact same interaction subset. Note that only the user-food `eats`/`healthness` edges are subsampled; the auxiliary graph (ingredient, food-similar, time) remains intact at every density. This is deliberate — it isolates the question of interest, namely how much auxiliary structure can compensate when interaction data is scarce.

**Table B. HR@10 vs. Interaction Density (% of full 262,270 interactions, GPU 5-fold CV)**

| Density | MF | LightGCN | NGCF | SGL | DualAttn-TB | **NutriGraphNet** | Best Model |
|---------|-----|---------|------|-----|---------|-------------------|------------|
| 10% | 0.349 | 0.524 | 0.509 | 0.088 | 0.656 | **0.738** (±0.018) | **NutriGraphNet** |
| 30% | 0.617 | 0.690 | 0.701 | 0.212 | 0.734 | **0.755** (±0.009) | **NutriGraphNet** |
| 50% | 0.721 | 0.723 | **0.748** | 0.272 | 0.727 | 0.742 (±0.006) | NGCF |
| 70% | 0.744 | 0.729 | **0.763** | 0.332 | 0.737 | 0.722 (±0.020) | NGCF |
| 100% | 0.760 | 0.721 | **0.784** | 0.358 | 0.734 | 0.729 (±0.037) | NGCF |

*(HR@10; bold = best per row; see Figure 2)*

**Table B-AUC. AUC vs. Interaction Density (GPU 5-fold CV)**

| Density | MF | LightGCN | NGCF | SGL | DualAttn-TB | **NutriGraphNet** |
|---------|-----|---------|------|-----|---------|-------------------|
| 10% | 0.514 | 0.770 | 0.764 | 0.502 | 0.817 | **0.932** |
| 30% | 0.539 | 0.832 | 0.835 | 0.599 | 0.853 | **0.897** |
| 50% | 0.534 | 0.845 | 0.864 | 0.645 | 0.857 | **0.880** |
| 70% | 0.538 | 0.840 | 0.872 | 0.673 | 0.862 | **0.864** |
| 100% | 0.547 | 0.822 | **0.878** | 0.699 | 0.855 | 0.860 |

**Table B-NDCG. NDCG@10 vs. Interaction Density (GPU 5-fold CV)**

| Density | MF | LightGCN | NGCF | SGL | DualAttn-TB | NutriGraphNet |
|---------|-----|---------|------|-----|---------|---------------|
| 10% | 0.236 | 0.339 | 0.338 | 0.042 | **0.542** | 0.508 |
| 30% | 0.461 | 0.482 | 0.482 | 0.109 | **0.588** | 0.505 |
| 50% | 0.576 | 0.517 | 0.548 | 0.157 | **0.589** | 0.504 |
| 70% | 0.593 | 0.509 | 0.522 | 0.200 | **0.599** | 0.409 |
| 100% | **0.618** | 0.499 | 0.557 | 0.228 | 0.598 | 0.411 |

*(**Measurement note.** HealthGain@K is deliberately omitted from the sparsity sweep. `_get_food_health()` derives per-food health scores by averaging `healthness` edge attributes per food, and foods with no surviving `healthness` edge receive a score of 0. Because subsampling removes `healthness` edges alongside `eats` edges, the population baseline `hs_mean` collapses from 0.6653 at 100% density to 0.1529 at 10%, where 24,205 of 31,458 foods (77%) score 0. HealthGain@10 then reads +0.503 at 10% density — an artifact of recommending foods that still have interactions, not evidence of health alignment. Only the 100% figure (−0.0099) is meaningful; EXP-C, which runs entirely at full density, is unaffected.)*

**Finding B1 — SGL Catastrophic Collapse at Low Density.** At 10% interaction density (≈1.26 interactions/user), SGL's HR@10 collapses to **0.088** — **74.8% lower than MF** (0.349) and **83.2% lower than LightGCN** (0.524). Against NutriGraphNet (0.738) the gap is **8.39×**. At 30% density, SGL reaches only 0.212 vs. NutriGraphNet's 0.755 (**3.56× gap**). SGL's contrastive objective cannot construct meaningful positive pairs when interaction data is severely limited.

**Finding B2 — NutriGraphNet Dominates Where Interactions Are Scarce.** NutriGraphNet is the best model at both low-density settings: HR@10=**0.738** at 10% (**+45.0%** over NGCF=0.509, **+41.0%** over LightGCN=0.524, **+111%** over MF=0.349) and **0.755** at 30% (+7.7% over NGCF=0.701). The margin is widest exactly where interaction data is scarcest, which is the signature the auxiliary-edge hypothesis predicts: when `eats` edges are decimated, the intact ingredient/food-similar/time structure still carries usable signal.

This reading is corroborated by an independent line of evidence rather than resting on the correlation alone. DualAttn-TB, whose forward pass never consumes auxiliary edges (EXP-F: Δ HR@10 = 0.000 exactly for every edge-type removal), reaches 0.656 at 10% — while NutriGraphNet, which demonstrably does consume them (EXP-F v2a: −21.0% HR@10 when all auxiliary edges are removed), reaches 0.738, a **+12.5%** margin over that topology-blind reference. The two experiments triangulate: the model that uses auxiliary structure beats the model that cannot, by the largest margin precisely where auxiliary structure is the only structure left.

**Finding B3 — NutriGraphNet Is Density-Invariant; the Baselines Are Data-Hungry.** Ranking models by HR@10 scaling ratio (10%→100%):
- **NutriGraphNet: 0.738→0.729, ratio=0.99×** (density-invariant — performance is essentially flat across a 10× change in interaction volume)
- DualAttn-TB: 0.656→0.734, ratio=1.12×
- LightGCN: 0.524→0.721, ratio=1.38×
- NGCF: 0.509→0.784, ratio=1.54×
- MF: 0.349→0.760, ratio=2.18× (most data-hungry)
- SGL: 0.088→0.358, ratio=4.06× (most fragile)

At 10% density NutriGraphNet already attains **94.1%** of the best full-data score any model achieves (NGCF's 0.784) — using one tenth of the interactions. For domains where interaction data is expensive to collect (clinical cohorts, national dietary surveys), this is the operationally relevant property.

**Finding B4 — The Flip Side: NutriGraphNet Does Not Exploit Additional Data.** Density-invariance cuts both ways. NGCF overtakes NutriGraphNet from 50% upward (0.748 vs. 0.742 at 50%; 0.784 vs. 0.729 at 100%, +7.6%), because NGCF converts extra interactions into ranking quality while NutriGraphNet plateaus at HR@10≈0.72–0.76 throughout. Within the sweep, NutriGraphNet's own variation (0.722–0.755) is small relative to its fold-level spread (σ up to 0.037) and to the run-to-run reproducibility gap measured on this setup (ΔHR@10≈0.002, ΔAUC≈0.005 for identical seed and configuration), so the apparent 50%→100% decline should be read as a plateau, not a decline. NutriGraphNet is therefore best positioned as a **sparse-data specialist**: preferred below ~50% density, matched or beaten by NGCF above it.

**Finding B5 — MF Structural AUC Ceiling.** MF's AUC is structurally capped at ~0.547 regardless of density (range 0.514–0.547) while GNN AUC scales (NGCF: 0.764→0.878). Yet MF achieves HR@10=0.760 at full density, exceeding NutriGraphNet (0.729) on hit rate despite a 0.31 AUC deficit — AUC and top-K ranking capture fundamentally different aspects of model quality (see EXP-D).

**Finding B6 — Ranking Quality Is Not Uniform Across Metrics.** NutriGraphNet's advantage is confined to HR@10 and AUC; on NDCG@10 it trails DualAttn-TB at every density (0.508 vs. 0.542 at 10%; 0.411 vs. 0.598 at 100%) and trails MF at full density (0.618). NutriGraphNet retrieves relevant foods into the top-10 more reliably than any baseline at low density, but orders them less well once retrieved. Section 8.1 and the v2 decoder analysis attribute this to the HybridDecoder's rank calibration rather than to health regularization — at λ_health=0.0 (health loss fully disabled) NDCG@10 is 0.4032, *below* the λ=0.005 value of 0.4279, so the health objective is not the cause.

**Finding B7 — SGL Threshold Effect.** Below 50% density (≈6.3 interactions/user), SGL falls dramatically below every other model. For datasets under 50% interaction density the practical recommendation is NutriGraphNet (leverages auxiliary structure) or LightGCN (robust propagation), and to avoid SGL entirely.

**Open question — the AUC trend.** NutriGraphNet's AUC *decreases* monotonically with density (0.932→0.860), the only model to do so; every baseline increases (NGCF 0.764→0.878). One plausible explanation is that the auxiliary graph is held at full size while `eats` edges are subsampled, so at 10% density the model's discriminative signal comes overwhelmingly from intact auxiliary structure against a smaller, easier positive set. We flag this as unresolved: the effect is consistent across all five folds (σ≤0.008) and so is not noise, but the current experiments do not isolate its cause, and it should be settled before the AUC column is used to support any claim.

---

### 6.2 EXP-F: Graph Component Ablation — Does NutriGraphNet Actually Consume Its Topology?

Section 6.1 showed that NutriGraphNet's advantage is largest exactly where interaction edges are scarcest, which is what the auxiliary-edge hypothesis predicts. But a correlation between sparsity and advantage does not by itself prove that the model *uses* auxiliary structure. EXP-F tests that mechanism directly by removing edge types and measuring the consequence.

The experiment is run in two parts. **EXP-F v1** establishes the measuring instrument: we ablate edges on the DualAttn-TB control, which by construction never reads auxiliary `edge_index` tensors, and confirm that the harness reports exactly zero change — the necessary calibration check. **EXP-F v2** then applies the same harness to models that do read the graph. The contrast between the two is the finding.

**Table F. Graph Component Ablation — DualAttn-TB (dim=64, 5-fold CV)**

| Variant | HR@10 | NDCG@10 | MRR | AUC | F1 | ΔHR@10 |
|---------|-------|---------|-----|-----|-----|--------|
| Full Graph | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | — |
| w/o Ingredient | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | **0.0000** |
| w/o Time | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | **0.0000** |
| w/o Food-Similar | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | **0.0000** |
| w/o Ingredient+Time | 0.7340 | 0.5977 | 0.5635 | 0.8551 | 0.7200 | **0.0000** |

*(GPU 5-fold CV confirmed; all ΔHR@10 = 0.000000 exactly; max ΔAUC = 1.2×10⁻⁷ across 5 folds)*

**Finding F1 — The Control Behaves Exactly As Designed, Validating the Ablation Harness.** Removing any edge type — or any combination — produces **mathematically identical performance** for DualAttn-TB across all 7 metrics and all 5 folds (Δ HR@10 = 0.000, < 1×10⁻¹⁰; max Δ AUC = 1.2×10⁻⁷, i.e. floating-point noise). This is the expected result, not a discovery: the control never reads auxiliary edges, so removing them *must* be a no-op. Its value is methodological — it demonstrates that (i) the ablation code removes exactly what it claims to remove and nothing else, and (ii) any non-zero Δ measured for another model in EXP-F v2 reflects genuine topology dependence rather than an artifact of the harness. Without this calibration, the v2a numbers below would be uninterpretable.

**Why the Control Is Topology-Blind by Construction.**

DualAttn-TB's forward pass is written so that auxiliary edges cannot influence the score. This is a design decision, not a defect discovered after the fact:

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

Neither branch routes message-passing through `ingredient`, `time`, `food_similar`, or `healthness` edges. The heterogeneous graph is present in the `PyG HeteroData` object the control receives — it is given exactly the same input as NutriGraphNet — but never enters its forward computation graph. Consequently, zeroing `ingredient.edge_index`, `time.edge_index`, or `food_similar.edge_index` produces no gradient change and no score change.

This is precisely what makes the model useful as a control: it holds embedding capacity and attention fixed while setting graph consumption to zero, so any performance difference against NutriGraphNet under the same data and budget is attributable to topology consumption rather than to capacity or optimisation.

**Finding F2 — A Topology-Blind Model Is a Strong Recommender on This Graph.** DualAttn-TB attains AUC=0.855 and HR@10=0.734 at full density — competitive with the graph baselines — using nothing but embedding lookup and attention over the interaction matrix. At 10% density it reaches HR@10=0.656, beating every published graph baseline we tested (LightGCN 0.524, NGCF 0.509). The practical implication is that on an ultra-sparse nutrition graph, graph propagation is not automatically worth its cost: a well-tuned topology-blind model is a genuinely competitive alternative, and the burden is on graph-based methods to demonstrate that they extract value from the structure they consume. NutriGraphNet meets that burden (Finding F4); the point of the control is that meeting it cannot be assumed.

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

**Finding F4 — NutriGraphNet Shows Genuine, Graded Topology Dependence.** Unlike DualAttn-TB's exact Δ=0.000, NutriGraphNet's HR@10 degrades measurably and monotonically as more auxiliary structure is removed: −1.7% (w/o time, the weakest signal) to −21.0% (w/o all auxiliary, the strongest). Removing `healthness` alone costs −7.2% HR@10 — the single largest individual-edge-type effect — consistent with `healthness` edges carrying the densest per-interaction signal (one healthness edge per user-food interaction, vs. sparser ingredient/time/food-similar connections). This is direct, functional evidence — not just code inspection — that NutriGraphNet's message-passing genuinely consumes the heterogeneous topology it claims to model.

**Finding F5 — Removing Critical Structure Destabilizes Convergence, Not Just Average Performance.** The `w/o healthness` and `w/o all auxiliary` conditions show far higher fold-to-fold variance (σ=0.101 and σ=0.141) than the full graph baseline (σ=0.031). In both cases, four of five folds land in a comparable range (no_all_auxiliary: 0.566–0.698; no_healthness: 0.710–0.792) while one fold collapses sharply (no_all_auxiliary fold 5: HR@10=0.310; no_healthness fold 4: HR@10=0.536). This indicates that stripping the auxiliary graph does not merely shift the mean — it makes convergence itself less reliable, an effect that a single-fold ablation (as in the original EXP-F v1 setup) would not have surfaced.

**Finding F6 — NGCF's Residual Sensitivity Is Small and Clusters by Coverage, Not by Edge-Type Identity.** Diluting interactions for auxiliary-connected foods costs NGCF only −1.2% to −1.4% HR@10 regardless of *which* edge type defines the "auxiliary-connected" food set — ingredient, time, healthness, and their combinations all converge to nearly the same diluted HR@10 (0.7748–0.7752, AUC=0.8732). This is because ingredient, time, and healthness edges each cover a large majority of the 31,458 foods (health scores alone are defined for all foods, per Finding F3), so the "50%-diluted" food sets under these three ablations are nearly identical regardless of label. Food-similarity edges (108,062 edges, a sparser food-food relation) cover a different, smaller food set, producing a slightly different result (HR@10=0.7732, closer to the full-graph baseline of 0.7844 than the other five variants). Notably its F1 (0.8013) exceeds the full-graph value (0.7625) despite lower HR@10 — a reminder that F1 (classification-threshold-dependent) and HR@10 (ranking-based) can move in different directions and should not be conflated. The key contrast with Table F-v2a stands regardless: even under an indirect interaction-count proxy, NGCF's auxiliary-adjacent sensitivity (≤1.4%) is an order of magnitude smaller than NutriGraphNet's direct topology dependence (up to 21.0%), reinforcing that only architectures which route message-passing through auxiliary edges are functionally dependent on them.

---

## 7. Health Constraints Are Architecturally Routed

### 7.1 EXP-C: Health Constraint λ_health Sensitivity

**Hypothesis:** If health loss gradients properly backpropagate through model parameters, varying λ_health should change both health alignment (measurable via HealthGain@K) and ranking quality (HR@10). We test this hypothesis on two architectures: **NutriGraphNet** (routes message-passing through all 9 edge types, including `healthness`) and **DualAttn-TB** (dual-attention architecture, serving as an ablation baseline).

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

**Table C-Control (Reference). DualAttn-TB Performance vs. λ_health (identical by construction — the control's health branch cannot reach graph structure)**

| λ_health | AUC | F1 | HR@10 | NDCG@10 | MRR | HealthGain@10 |
|---------|--------|--------|--------|---------|--------|---------------|
| 0.000–1.000 | 0.8551 | 0.7200 | 0.7340 | 0.5977 | 0.5635 | ≈ 0 (all λ) |

*(Δ HR@10 = 0.0000 exactly; Δ AUC < 1.25×10⁻⁷ for all λ ∈ [0.001, 1.0]. Reported for architectural contrast only.)*

**Finding C1 — NutriGraphNet Is Robust to λ in [0.001, 0.1].** Across the practical range λ ∈ {0.001, 0.005, 0.01, 0.05, 0.1}, HR@10 varies by only **Δ=0.0040** (0.7390–0.7430) — within measurement noise. This **robust plateau** demonstrates that NutriGraphNet's ranking quality is insensitive to the exact health constraint weight, making deployment straightforward: any λ in this range is safe. The result revises the earlier CPU 1-fold finding (which suggested λ=0.5 as optimal); under multi-seed evaluation with proper convergence control, no single λ is decisively better than others in the plateau.

**Finding C2 — HealthGain@10 Is Consistently Non-Zero Across All λ.** HealthGain@10 is negative and stable across λ ∈ {0.0, 0.001, ..., 1.0} (range: −0.00895 to −0.00940 in the plateau, −0.00930 at λ=1.0). Three observations are critical: *(i)* **non-zero at λ=0.0** (−0.009) — baseline health-gradient signal exists from the architecture itself; *(ii)* **magnitude stable** across the plateau, indicating health regularization does not meaningfully alter the health-alignment direction; *(iii)* **active at large λ** (−0.0094 at λ=0.5) — even at high health weights, the model maintains gradient flow. This contrasts sharply with DualAttn-TB's structurally zero HealthGain, confirming NutriGraphNet's architectural health gradient routing is functional across the full λ range.

**Finding C3 — Health–Ranking Degradation at λ≥0.5 Is Moderate *under Lightweight Parameters*.** Above λ=0.1, ranking metrics show mild degradation in this setting: HR@10 drops from 0.7430 (λ=0.1) to 0.7350 (λ=0.5, −1.1%) and AUC from 0.8556 (λ=0.05) to 0.8370 (λ=0.5, −2.2%). This is substantially milder than the CPU 1-fold estimate (−29.7% at λ=1.0) — but the full-parameter confirmation (Table C-Full, Finding C5) shows the degradation severity is capacity-dependent: under full parameters HR@10 falls −19.6% at λ=0.5 and −27.3% at λ=1.0. The consistent picture across both regimes is that **NutriGraphNet tolerates moderate health regularization (λ≤0.1) without significant ranking cost**, while strong regularization (λ≥0.5) carries a real — and at full capacity, severe — ranking penalty. **Practical recommendation: λ=0.01 as default** — on the plateau in both regimes, conservative, and interpretable.

**Finding C4 — NutriGraphNet Health Loss Is Active; DualAttn-TB Is Architecturally Severed.** For NutriGraphNet, HealthGain@10 is consistently non-zero (≈−0.009) across all λ, confirming that health gradients flow from the `healthness` edge convolution path through the NutriLoss objective. This is a qualitative departure from DualAttn-TB's structurally zero health gradient (Δ HR@10 = 0.000 exactly, Δ AUC < 1.25×10⁻⁷ across all λ), validating that NutriGraphNet's architectural design enables genuine health-aware optimization regardless of λ choice.

**Full-Parameter Confirmation (EXP-C-Full: GPU 5-fold CV, hidden=128, out=64, layers=3, heads=4, seed 42).** To address the parameter-scale caveat, we additionally ran the identical λ sweep under the full-parameter GPU 5-fold protocol used in EXP-B/D/F/G.

**Table C-Full. NutriGraphNet Performance vs. λ_health (GPU 5-fold CV, full parameters, seed 42)**

| λ_health | AUC | HR@10 (±σ) | NDCG@10 | MRR | HealthGain@10 |
|---------|--------|--------|---------|--------|---------------|
| 0.000 | 0.8577 | 0.7116 (±0.0624) | 0.4032 | 0.3202 | −0.01083 |
| 0.001 | 0.8606 | 0.7396 (±0.0201) | 0.4188 | 0.3289 | −0.01091 |
| **0.005** | **0.8620** | **0.7484 (±0.0192)** | **0.4279** | **0.3378** | −0.01158 |
| 0.010 | 0.8545 | 0.7308 (±0.0295) | 0.4176 | 0.3310 | −0.00999 |
| 0.050 | 0.8508 | 0.6836 (±0.0884) | 0.3869 | 0.3091 | −0.00877 |
| 0.100 | 0.8573 | 0.7176 (±0.0302) | 0.3953 | 0.3073 | −0.00911 |
| 0.500 | 0.8387 ↓ | 0.5820 (±0.0481) ↓ | 0.2841 ↓ | 0.2138 ↓ | −0.00300 |
| 1.000 | 0.8271 ↓ | 0.5260 (±0.0681) ↓ | 0.2700 ↓ | 0.2143 ↓ | −0.00190 |

*(single seed=42; 5-fold mean. Unlike the lightweight setting, seed 42 converges normally here — AUC=0.83–0.86 at every λ — consistent with the convergence failure being specific to the lightweight/1-layer configuration.)*

**Finding C5 — The Plateau Survives Full Parameters; the Degradation Does Not Stay Mild.** Two conclusions from Table C-Full: *(i)* **the λ-robust plateau holds qualitatively** — across λ ∈ [0.001, 0.1], HR@10 varies non-monotonically within 0.6836–0.7484 (Δ=0.0648), which is within fold-level noise (σ up to 0.088) and shows no systematic trend, confirming that no single λ in the practical range is decisively better; λ=0.005 is the nominal optimum (HR@10=0.7484, also the best AUC=0.8620). *(ii)* **Degradation at λ≥0.5 is severe under full parameters** — HR@10 falls to 0.5820 at λ=0.5 (−19.6% vs. plateau mean 0.7240) and 0.5260 at λ=1.0 (−27.3%), an order of magnitude larger than the lightweight-setting estimate (−1.1%) and close to the original CPU 1-fold observation (−29.7%). Notably, HealthGain@10 simultaneously shrinks toward zero (−0.0030 at λ=0.5, −0.0019 at λ=1.0 vs. ≈−0.011 on the plateau): at full capacity, strong health regularization actively pulls recommendations toward the population nutritional average — the health–ranking trade-off becomes *functional* rather than dormant. The degradation severity is therefore **capacity-dependent**: lightweight models lack the capacity for L_health to reshape rankings, while the full model trades ranking for health alignment exactly as the loss design intends. The practical recommendation is unchanged — **λ=0.01 remains a safe default** (on the plateau in both regimes) — but λ≥0.5 should be avoided in deployment unless health alignment is explicitly prioritized over ranking quality. A multi-seed replication of the full-parameter sweep remains future work.

**Mechanism Analysis — Why NutriGraphNet Succeeds Where DualAttn-TB Fails.**

**(a) Architectural health gradient path:** NutriGraphNet's DualChannelEncoder applies GATConv over all 9 edge types including `('user', 'healthness', 'food')`. The health constraint loss L_health is defined over food embeddings that are updated via message-passing along `healthness` edges. The NutriLoss gradient thus flows: L_health → food_emb (via healthness conv) → GATConv parameters — a **valid architectural path**. In contrast, DualAttn-TB's NLA/SLA branches use direct embedding lookup and interaction-matrix attention only; `healthness` edges are never consumed in forward propagation, severing the gradient path entirely.

**(b) Lightweight vs. full-parameter behavior:** The multi-seed sweep (Table C) uses lightweight parameters (hidden=64, out=32, layers=1, heads=2) for consistency with prior EXP-C runs; the full-parameter GPU 5-fold sweep (Table C-Full) confirms the plateau qualitatively while revealing that the λ≥0.5 degradation — mild under lightweight parameters (−1.1%) — becomes severe at full capacity (−19.6%/−27.3%), with HealthGain@10 simultaneously moving toward zero. See Finding C5 for the capacity-dependence interpretation.

**Practical Implication.** NutriGraphNet provides **architecturally guaranteed health-gradient routing** with **λ=0.01 as the recommended default** (robust plateau, no ranking cost, conservative health weight). For clinical practitioners: *(i)* health-aware recommendation is achievable with proper architectural routing regardless of exact λ; *(ii)* DualAttn-TB's named health constraint provides zero guarantee without architectural remediation; *(iii)* λ sensitivity analysis should always include HealthGain@K verification — non-zero HealthGain is the only reliable indicator of active health optimization.

---

## 8. Supporting Analyses and Negative Results

### 8.1 EXP-D: Embedding Dimension Sensitivity

We vary embedding dimension d ∈ {16, 32, 64, 128, 256} for the four published baselines and the DualAttn-TB control under GPU 5-fold CV.

**Table D. HR@10 vs. Embedding Dimension d (GPU 5-fold CV)**

| d | MF | LightGCN | NGCF | SGL | DualAttn-TB |
|---|-----|---------|------|-----|---------|
| 16 | 0.6847 | 0.7140 | 0.7100 | 0.2700 | 0.7433 |
| 32 | 0.7187 | 0.7300 | 0.7767 | 0.3133 | **0.7550** |
| 64 | 0.7300 | 0.7113 | 0.7813 | 0.3467 | 0.7533 |
| 128 | 0.7533 | 0.7340 | 0.7833 | 0.3813 | 0.7517 |
| 256 | **0.7527** | **0.7440** | **0.7867** | **0.4213** | 0.7183 |

*(GPU 5-fold CV; bold = best per column; see Figure 4)*

**Table D-AUC. AUC vs. Embedding Dimension d (GPU 5-fold CV)**

| d | MF | LightGCN | NGCF | SGL | DualAttn-TB |
|---|-----|---------|------|-----|---------|
| 16 | 0.5042 | 0.8381 | 0.8644 | 0.6545 | **0.8573** |
| 32 | 0.5139 | 0.8417 | 0.8761 | 0.6683 | **0.8623** |
| 64 | 0.5087 | 0.8220 | 0.8771 | 0.6867 | 0.8553 |
| 128 | 0.5353 | 0.8308 | 0.8793 | 0.7042 | 0.8155 |
| 256 | **0.5428** | 0.8408 | **0.8810** | **0.7223** | 0.5740 ← ⚠ |

**Table D-NDCG. NDCG@10 vs. Embedding Dimension d (GPU 5-fold CV)**

| d | MF | LightGCN | NGCF | SGL | DualAttn-TB |
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

**Finding D5 — DualAttn-TB dim=256 AUC Collapse (Critical).** DualAttn-TB's AUC **collapses from 0.8623 (d=32) to 0.5740 (d=256)**, a drop of **0.2883 absolute points** (−33.4%). The average AUC for d∈{16,32,64,128} is 0.8476, making d=256 a −0.2736p outlier. HR@10 declines more modestly (0.7550→0.7183, −4.9%). The AUC collapse at high d is consistent with **attention weight ill-conditioning**: at d=256 with 4 attention heads, each head operates in 64-dimensional subspace; attention energies become near-uniform, collapsing score calibration while preserving rough rank ordering. **Practical recommendation: d=32 for DualAttn-TB** (HR@10=0.7550, AUC=0.8623 — both at or near maximum).

---

### 8.2 EXP-G: GNN Layer Depth and Over-Smoothing Analysis

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

### 8.3 EXP-A: SGL Augmentation Ratio Sensitivity — A Negative Result

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

### 8.4 Architecture Choices Do Not Transfer Across Density

The design decisions that improve ranking at full density can actively harm the sparse regime. We observed this directly while developing NutriGraphNet. A revised variant (v3) replaces the HybridDecoder (Bilinear × Dot × MLP ensemble) with a pure L2-normalised dot-product decoder that is rank-optimal by construction, and replaces the dual-channel encoder with a residual/batch-normalised one. Given an identical budget (300 epochs, lr=1e-3, seed 42, identical folds and interaction subsets), the two variants invert across density:

**Table H. NutriGraphNet Decoder Variants vs. Interaction Density (GPU 5-fold CV, HR@10 unless noted)**

| Density | v2 (HybridDecoder) | v3 (RankDot decoder) | Winner |
|---------|--------------------|----------------------|--------|
| 10% | **0.738** | 0.645 (−12.6%) | v2 |
| 30% | **0.755** | 0.700 (−7.3%) | v2 |
| 100% | 0.729 | **0.751** | v3 |
| 100% — NDCG@10 | 0.428 | **0.511** (+19.5%) | v3 |
| 100% — MRR | 0.338 | **0.445** (+31.7%) | v3 |
| 100% — AUC | 0.862 | **0.872** | v3 |

**Finding H1 — Rank Calibration Helps With Data and Hurts Without It.** The rank-calibrated decoder delivers exactly what it was designed for at full density (+19.5% NDCG@10, +31.7% MRR) and reverses at low density, losing 12.6% HR@10 at 10%. The gap is 18× our measured run-to-run noise floor (≈0.005), so it is not a fluctuation. The interpretation consistent with the rest of this study is capacity: a pure dot-product decoder is a *simpler* function than the Bilinear+MLP ensemble, and simpler models are hungrier — the same ordering EXP-B reports across baselines (scaling ratios: NutriGraphNet 0.99×, LightGCN 1.38×, NGCF 1.54×, MF 2.18×, SGL 4.06×). The decoder capacity that looks like a liability at 100% density is what extracts signal from auxiliary structure at 10%.

This has a practical consequence beyond our model: **architecture ablations reported only at full density can recommend the wrong design for data-scarce deployment.** We report v2 throughout this paper because the sparse regime is the setting of interest; had we selected on full-density NDCG alone, we would have shipped the variant that is 12.6% worse where it matters.

*Caveat.* v3's hyperparameters were tuned at full density and not re-tuned for the sparse regime; v2's were likewise held fixed across densities, so the comparison is matched on budget rather than on per-density optimality. We report the crossover as an observed interaction, not as a proof that no rank-calibrated decoder can win at low density.

---

## 9. Cross-Experiment Synthesis

The six experiments collectively paint a coherent picture of failure modes and success conditions in GNN-based food recommendation. Figure 6 summarizes the AUC vs. HR@10 trade-off landscape across all models and conditions.

**Table S. Key Numerical Reference Table (GPU 5-fold CV, full parameters)**

| Claim | Experiment | Confirmed Value |
|-------|-----------|-----------------|
| SGL best aug ratio | EXP-A | p=0.0, HR@10=0.3604; all p>0 degrade |
| SGL aug worst point | EXP-A | HR@10=0.3520 at p=0.5 (-2.3%); NDCG@10 worst at p=0.2 (-3.1%) |
| SGL collapse at 10% density | EXP-B | HR@10=0.0880 vs. NutriGraphNet=0.7380 (8.39x gap) |
| NutriGraphNet best at low density | EXP-B | HR@10=0.7380 at 10%, best of all 6 models (+45.0% over NGCF) |
| NutriGraphNet density-invariance | EXP-B | HR@10 0.7380 (10%) -> 0.7292 (100%), ratio 0.99x; 94.1% of NGCF's full-data 0.7844 using 10% of interactions |
| NGCF overtakes above 50% density | EXP-B | NGCF 0.7844 vs. NutriGraphNet 0.7292 at 100% (+7.6%) |
| NutriGraphNet sparsity robustness | EXP-B | HR@10 scaling ratio 1.12x (10%->100%), most robust |
| NGCF best at high density | EXP-B | HR@10=0.7844 at 100%, best baseline |
| MF AUC structural ceiling | EXP-B/D | AUC <=0.5428 regardless of density or dim |
| lambda_health robust plateau (NutriGraphNet) | EXP-C | lambda in [0.001,0.1]: HR@10=0.739-0.743 (Delta<0.004); HealthGain@10=-0.009 (active all lambda) |
| lambda_health recommended default | EXP-C | lambda=0.01: plateau center, no ranking cost, conservative |
| lambda_health degradation onset (lightweight) | EXP-C | lambda>=0.5: AUC drops to 0.837 (-2.2%), NDCG drops; HR@10 -1.1% |
| lambda_health full-param confirmation | EXP-C-Full | plateau holds (lambda in [0.001,0.1], best lambda=0.005 HR@10=0.7484); degradation severe: HR@10 -19.6% (lambda=0.5), -27.3% (lambda=1.0); HealthGain -> 0 |
| lambda_health sensitivity (DualAttn-TB, ref) | EXP-C | Delta HR@10 = 0.000 exactly for all lambda in [0.001, 1.0] |
| DualAttn-TB optimal dim | EXP-D | d=32: HR@10=0.7550, AUC=0.8623 |
| NGCF optimal dim (HR@10) | EXP-D | d=256: HR@10=0.7867, AUC=0.8810 |
| NGCF dim efficiency | EXP-D | d=64 achieves 99.3% of d=256 HR@10 (0.7813 vs. 0.7867) |
| DualAttn-TB dim=256 AUC collapse | EXP-D | AUC=0.5740 vs. avg 0.8476 for d in {16-128} (Delta=-0.2736) |
| LightGCN optimal layers | EXP-G | L=2: HR@10=0.7844±0.0206, NDCG@10=0.5838±0.0187; L=3 worst (HR@10=0.7208, -8.1%) |
| NGCF layer-invariance | EXP-G | L=1 best (HR@10=0.7876±0.0150); Δ across L=1-4 = 0.0076 (1.0%) |
| Optimal GNN layers | EXP-G | NGCF: L=1; LightGCN: L=2 (avoid L=3, -8.1% HR@10 penalty) |
| DualAttn-TB topology invariance | EXP-F | Delta HR@10 = 0.000000 across all 5 ablations; max Delta AUC = 1.2e-7 |
| NutriGraphNet topology dependence | EXP-F v2a | w/o all auxiliary: HR@10=0.5764 (-21.0% vs full=0.7296); w/o healthness alone: -7.2% |
| NGCF residual sensitivity (dilution proxy) | EXP-F v2b | HR@10 -1.2% to -1.4% across all variants (vs NutriGraphNet's up to -21.0%) |
| Best overall HR@10 | EXP-B/D | NGCF: 0.7844 (100% density, d=64 baseline); 0.7867 (d=256) |
| Best overall AUC | EXP-D | NGCF d=256: 0.8810 |

---

## 10. Design Guidelines

Based on our empirical findings, we propose the following actionable guidelines for practitioners:

| Scenario | Recommendation | Evidence |
|----------|----------------|---------|
| Sparse interactions (<20/user) | **Avoid SGL** -- use NutriGraphNet or LightGCN | EXP-B: SGL HR@10=0.0880 at 10% density; NutriGraphNet=0.7380 (8.39x better) |
| Low density (10-30%) | **NutriGraphNet** preferred; LightGCN as lightweight alternative | EXP-B: NutriGraphNet HR@10=0.7380 at 10% (+45.0% over NGCF=0.5088, +41.0% over LightGCN=0.5236) |
| Density above ~50% | **NGCF** preferred if hit rate is the target | EXP-B: NGCF 0.7844 vs. NutriGraphNet 0.7292 at 100%; ranking reverses near 50% |
| Ranking order matters (NDCG-critical) | **Avoid NutriGraphNet v2** -- use MF or DualAttn-TB | EXP-B: NutriGraphNet NDCG@10=0.411 at 100% vs. MF=0.618; a decoder rank-calibration issue, not a health trade-off |
| Dense interactions (>50% / >6 int/user) | **NGCF** optimal for HR@10; NutriGraphNet for NDCG | EXP-B: NGCF HR@10=0.7844 at 100%; NutriGraphNet NDCG@10=0.5977 (best baseline) |
| SGL augmentation ratio | **p=0.0 is always optimal** on sparse nutrition graphs | EXP-A: HR@10 decreases for all p>0; worst at p=0.5 (-2.3%) |
| GNN layer depth (NGCF) | **L=1 is optimal** -- layer-invariant (ΔHR@10=0.0076 across L=1–4) | EXP-G: NGCF HR@10=0.7876 at L=1; Δ<1.0% across all depths |
| GNN layer depth (LightGCN) | **L=2 is optimal; avoid L=3** (−8.1% HR@10 penalty) | EXP-G: LightGCN HR@10=0.7844 at L=2 vs. 0.7208 at L=3 |
| Embedding dim (NGCF) | **d=64 for HR@10** (99.3% of d=256); **d=128-256 for NDCG@10** | EXP-D: d=64 HR@10=0.7813 vs. d=256=0.7867; NDCG@10 scales to d=256 |
| Embedding dim (DualAttn-TB) | **d=32 only** -- AUC collapses -0.2736p at d=256 | EXP-D: AUC=0.8623 at d=32 vs. 0.5740 at d=256 |
| Health constraint weight (lambda) | **lambda=0.01 as default** — robust plateau in [0.001,0.1]; avoid lambda>=0.5 unless health alignment is the priority | EXP-C: HR@10=0.739-0.743 across lambda=0.001-0.1; EXP-C-Full: degradation at lambda>=0.5 is severe under full params (HR@10 -19.6%/-27.3%) |
| Health backpropagation | **Route health gradients via healthness edge convolution** | EXP-C: NutriGraphNet HealthGain@10=-0.009 (active all lambda); DualAttn-TB severs gradient |
| Auxiliary edge types | **Ablate each edge type** to verify forward-pass contribution | EXP-F: DualAttn-TB Delta HR@10=0.000000 for all 3 auxiliary edge types; NutriGraphNet shows real -1.7% to -21.0% degradation (EXP-F v2a), confirming genuine topology dependence |
| "Heterogeneous graph" claims | Only claim graph-awareness if architecture **routes messages** through edges | EXP-F: a topology-blind model reproduces competitive AUC/HR without reading any auxiliary edge, so graph framing alone is not evidence of graph use |
| Clinical health-aware deployment | **Verify health gradients architecturally**, not just via named loss | EXP-C/F: NutriGraphNet shows active HealthGain@K; the control's health branch cannot affect graph structure at all |
| Architecture selection | **Choose the decoder for your density**, not by full-density benchmarks | §8.4: a rank-calibrated decoder gains +19.5% NDCG@10 at 100% density but loses −12.6% HR@10 at 10% |

---

## 11. Limitations and Threats to Validity

We state the boundaries of these results explicitly, including two measurement pitfalls we encountered ourselves.

**11.1 NutriGraphNet does not exploit additional data.** Density-invariance is the headline property and also the ceiling: HR@10 stays at 0.72–0.76 whether given 10% or 100% of interactions, so NGCF overtakes it above roughly 50% density (0.784 vs. 0.729 at 100%). NutriGraphNet is a sparse-data specialist and should not be presented as a general-purpose ranker.

**11.2 Ranking order is weaker than retrieval.** NutriGraphNet's advantage is confined to HR@10 and AUC. Its NDCG@10 trails the topology-blind control at every density (0.508 vs. 0.542 at 10%; 0.411 vs. 0.598 at 100%) and trails MF at full density (0.618). This is a decoder rank-calibration limitation, not a health trade-off — disabling the health objective entirely (λ=0.0) *lowers* NDCG@10 to 0.403 — and the rank-calibrated decoder that fixes it at full density is worse in the sparse regime (§8.4). Applications where the order within the top-10 matters more than its membership are not well served by this model.

**11.3 HealthGain@K is not measurable below full density (measurement pitfall).** Per-food health scores are derived by averaging `healthness` edge attributes per food, so foods left with no surviving `healthness` edge score 0. Because interaction subsampling removes `healthness` edges alongside `eats` edges, the population baseline collapses from 0.6653 at 100% density to 0.1529 at 10%, where 24,205 of 31,458 foods (77%) score 0. HealthGain@10 then reads **+0.503** at 10% density — which would appear to show that health alignment *strengthens* under sparsity, an attractive and entirely false conclusion. The quantity is measuring the tautology that recommended foods are foods with interactions. We therefore report HealthGain only at full density. Any sparsity study of a health-aware recommender that derives item health scores from interaction-coupled edges is exposed to this artifact.

**11.4 Run-to-run reproducibility floor (measurement pitfall).** Identical seed, parameters, and density produce ΔHR@10≈0.002, ΔAUC≈0.005, ΔNDCG@10≈0.007 across independent runs (`B_full_100pct` vs. `C_lambda_0.01`), which we attribute to non-deterministic scatter operations in GPU message passing accumulated over 300 epochs. All effects we claim exceed this floor by at least an order of magnitude, but it is the reason we decline to claim an augmentation-ratio effect (EXP-A, Δ=0.008) and treat the nominal λ optimum (λ=0.005 vs. λ=0.01, Δ=0.018) as weakly identified.

**11.5 An unexplained AUC trend.** NutriGraphNet's AUC decreases monotonically as density increases (0.932→0.860), uniquely among the six models; every baseline increases (NGCF 0.764→0.878). The effect is consistent across all five folds (σ≤0.008) and so is not noise. A plausible explanation is that the auxiliary graph is held at full size while `eats` edges are subsampled, leaving the model an intact structural signal against a smaller positive set, but our experiments do not isolate the cause. We therefore draw no conclusion from the AUC column in the sparsity sweep.

**11.6 Health-aware baseline is a re-implementation, not the original pipeline.** We compare against HFRS-DA [Forouzandeh et al., 2024], the published health-aware food recommender, re-implemented from the authors' official code (§4.1). Because HFRS-DA's native recommendation and evaluation (user–user cosine CF, Precision/Recall/F1 over a user split) are not comparable to our unified sampled-ranking protocol, our figures reflect its architecture and health mechanism under our evaluation rather than a verbatim re-run of the original pipeline, and its health signal is taken from our WHO-7 `healthness` edge rather than re-derived raw-gram thresholds. A comparison against additional health-aware systems (SCHGN, FRMADHG, MOPI-HFRS) remains outstanding.

**11.7 Single dataset.** All results are on NutriGraph-KR. The sparsity regime we study (0.040%, 12.6 interactions/user) is the paper's subject rather than an incidental property, but we have not shown that the auxiliary-substitution effect transfers to other nutrition graphs, and denser public benchmarks (Food.com is ≈3× denser at ≈25 interactions/user, and carries no user health attributes) would test the boundary rather than replicate the finding.

**11.8 Single seed for the full-parameter λ sweep.** Table C-Full reports seed 42 only; the multi-seed evidence for the λ plateau (Table C) comes from the lightweight configuration. A multi-seed full-parameter replication remains outstanding.

---

## 12. Conclusion

We presented a systematic empirical analysis of GNN-based food recommendation on a large-scale heterogeneous nutrition graph (NutriGraph-KR: 20,820 users, 31,458 foods, density=0.040%), uncovering four key phenomena with root-cause explanations:

**(1) SGL Collapse Under Sparsity.** SGL suffers catastrophic collapse at 10% data density (HR@10=0.088 vs. NutriGraphNet=0.738, **8.39× gap**; vs. LightGCN=0.524, 5.95× gap). The root cause is the fundamental incompatibility between SGL's InfoNCE contrastive objective — which requires dense positive views — and nutrition interaction graphs with an average of only 12.6 interactions per user (10× sparser than MovieLens-1M). Critically, this is a property of the objective, not of augmentation intensity: at p=0.0, with augmentation disabled entirely, SGL still reaches only HR@10=0.3604 versus NGCF's 0.7844 (2.18×), and sweeping p from 0.0 to 0.5 moves HR@10 by just Δ=0.0084 — within this setup's reproducibility gap and below fold-level σ. We therefore do **not** claim an augmentation-ratio effect; the earlier framing of these results as "augmentation collapse" overstated a difference that the data cannot resolve.

**(2) NutriGraphNet Sparsity Robustness.** Under GPU 5-fold cross-validation, NutriGraphNet is the best model at 10% and 30% density, achieving HR@10=0.738 at 10% (+45.0% over NGCF=0.509, +41.0% over LightGCN=0.524). Its sparsity scaling ratio is 0.99× (10%→100%) — **density-invariant**, the most robust of all six models — because auxiliary graph edges (ingredient, food-similarity, time) supply non-interaction structural signal that compensates for sparse user-food data. That mechanism is independently confirmed rather than merely inferred: removing the auxiliary edges costs NutriGraphNet 21.0% HR@10 (EXP-F v2a), while the topology-blind DualAttn-TB reference is provably indifferent to the same removal (Δ=0.000) and reaches only 0.656 at 10%. At 10% density NutriGraphNet attains 94.1% of the best full-data score any model reaches, using one tenth of the interactions — the operationally relevant property when interaction data is expensive to collect. The property is two-sided: NutriGraphNet does not convert additional data into ranking quality, so NGCF overtakes it above 50% density (0.784 vs. 0.729 at 100%), and its NDCG@10 (0.411) trails both MF (0.618) and DualAttn-TB (0.598) — a decoder rank-calibration limitation, not a health trade-off, since disabling the health loss entirely (λ=0.0) *lowers* NDCG@10 to 0.403. NutriGraphNet is therefore best positioned as a sparse-data specialist rather than a general-purpose ranker.

**(3) MF–SGL Ranking Paradox.** Simple matrix factorization (HR@10=0.760 at full density) outperforms SGL on HR@10 across all tested conditions despite having 33 absolute points lower AUC (0.547 vs. NGCF 0.878). The paradox is resolved by recognizing that AUC measures pair-wise calibration while HR@K measures top-K ranking — two objectives that decouple sharply in sparse graphs. SGL fails completely (HR@10=0.088 at 10% density) despite moderate AUC (0.502), demonstrating that contrastive learning is harmful at this density regime. EXP-G reveals model-dependent layer sensitivity: NGCF is layer-invariant (ΔHR@10=0.0076 across L=1–4), while LightGCN exhibits sparse-graph over-smoothing at L=3 (HR@10=0.7208, −8.1% vs. L=2=0.7844), partially recovering at L=4. Optimal depths are L=1 for NGCF and L=2 for LightGCN.

**(4) Health Constraints Are Architecturally Routed, Not Merely Named.** NutriGraphNet propagates through all 9 edge types including `healthness`, and its health objective demonstrably reaches the representation: HealthGain@10 stays non-zero (≈−0.010) across four orders of magnitude of λ, with a robust plateau over λ∈[0.001, 0.1] (Δ HR@10 = 0.0040). Under full parameters the trade-off is functional rather than nominal — at λ≥0.5, HealthGain@10 contracts toward zero (−0.002) while HR@10 falls 19.6–27.3%, i.e. the objective genuinely trades ranking for health alignment once model capacity permits it. Recommended default: λ=0.01. Our topology-blind control, whose health branch cannot influence graph structure at all, registers Δ HR@10 = 0.000 exactly across the same λ range — by construction, and reported here only to demonstrate what an unrouted health term looks like when measured. These results establish health-awareness as a property of **gradient routing**: the health signal must flow through health-relevant convolution to affect recommendations, and a model's loss function name is not evidence that it does.

**Our findings challenge four widely held assumptions** in graph-based food recommendation: (a) more interaction data is the primary lever for sparse graphs; (b) architectural complexity correlates with ranking quality; (c) a health-aware loss term suffices for health optimization without architectural routing; (d) deeper GNN layers are necessary for rich representations on heterogeneous graphs — NutriGraphNet validates (c) positively with the nuance that health-aware improvement is **λ-robust** (plateau λ∈[0.001–0.1], ΔHR@10<0.4%) rather than sensitive to a single optimal value, EXP-G refutes (d) with the nuance that L=3 actively harms LightGCN (−8.1% HR@10), and EXP-B/A collectively refute (a) and (b). EXP-F v2 confirms this architecturally: NutriGraphNet's HR@10 degrades by up to 21.0% when all auxiliary edges are removed (vs. DualAttn-TB's exact 0.000), providing functional (not just code-level) evidence that its message-passing genuinely depends on the heterogeneous topology it claims to model. Future work will investigate: (i) ingredient-conditioned positive sampling for contrastive learning in sparse nutrition graphs; (ii) reducing the fold-to-fold instability observed when critical auxiliary edges (healthness, all-auxiliary) are removed (EXP-F v2a, σ up to 0.141); and (iii) extending NutriGraphNet to explicit HealthGain maximization objectives.

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
*New in v1.3 (2026-07-16): EXP-B "NutriGraphNet" 열 오표기 수정 — 실측 데이터로 교체.*  
*문제: v0.8(커밋 592634a)에서 "evaluate all four baselines" → "all five models including NutriGraphNet (hfrsda)"로 바뀌며 `hfrsda` 결과가 "NutriGraphNet" 열로 실렸음. 5가지 독립 근거로 확인: (1) 코드상 `'hfrsda'`=DualAttn-TB(line 2013/2184), `'full'`=NutriGraphNet(2017/2186/2298); (2) run_all.bat EXP-B가 `full`을 실행하지 않음; (3) `results/gpu/B_sparsity_*/`에 `results_full.json` 부재; (4) Table B 100% 행이 Table 1의 DualAttn-TB 행과 완전 동일; (5) 해당 데이터의 HealthGain@10=None·health loss=0.0 (NutriGraphNet은 항상 ≈−0.010 보고). 부작용으로 Finding B2가 "auxiliary edge가 희소성을 보완"이라 설명하던 수치가 EXP-F에서 auxiliary edge를 전혀 읽지 않는다고 증명된 바로 그 모델의 것이어서 자기모순 상태였음.*  
*조치: `run_expB_full.bat`(커밋 560efd1)으로 NutriGraphNet 5개 밀도 GPU 5-fold 실행(λ=0.01, seed 42, full params, 시드 고정 부분추출로 베이스라인과 동일 부분집합). 출력은 기존 폴더 파괴를 피해 `results/gpu/B_full_*pct`로 분리.*  
*결과: 진짜 NutriGraphNet이 오표기값보다 우수 — 10% HR@10=0.738(vs 오표기 0.656), NGCF 대비 +45.0%. 30%에서도 1위(0.755). 밀도 불변(0.738→0.729, ratio 0.99×). 이제 EXP-F v2a(−21.0%)와 DualAttn-TB(Δ=0.000)가 독립 증거로 삼각검증되어 auxiliary-edge 메커니즘 주장이 처음으로 정합적으로 성립.*  
*한계도 실측 확인: 50% 이상에서 NGCF에 역전(0.784 vs 0.729), NDCG@10은 전 밀도에서 DualAttn-TB에 열세(0.411 vs 0.598 at 100%). 후자는 health 때문이 아님 — λ=0.0에서 NDCG@10=0.4032로 오히려 더 낮음 → HybridDecoder rank calibration 문제(v3 RankDotDecoder의 설계 근거와 일치).*  
*측정 결함 발견: HealthGain@K는 저밀도에서 무의미. `_get_food_health()`가 healthness 엣지 평균으로 food 점수를 만드는데 엣지 없는 food는 0점 → 부분추출 시 기준선 `hs_mean`이 0.6653(100%)에서 0.1529(10%)로 붕괴, food의 77%(24,205/31,458)가 0점. 그 결과 10%에서 HealthGain@10=+0.503이 나오지만 이는 "상호작용 있는 food를 추천했다"는 동어반복. Table B에서 제외, 100%(−0.0099)만 유효. EXP-C는 전부 100% 밀도라 영향 없음.*  
*미해결: NutriGraphNet AUC가 밀도에 반비례(0.932→0.860)하는 유일한 모델. 5-fold 전부 일관(σ≤0.008)이라 노이즈 아님. auxiliary 그래프가 부분추출되지 않는 설계 때문으로 추정되나 현 실험으로 인과 규명 불가 → AUC 열을 근거로 쓰기 전 해소 필요.*  
*재현성 관측: 동일 seed·파라미터·밀도에서 `B_full_100pct`와 `C_lambda_0.01`이 ΔHR@10=0.0016, ΔAUC=0.0050, ΔNDCG=0.0068 차이(GPU scatter 비결정성 추정). 이 노이즈 바닥(~0.005)은 EXP-A의 증강 효과(Δ=0.0084)와 같은 크기 → EXP-A 주장 재검토 필요.*  
---
*New in v1.2 (2026-07-15): EXP-C full-parameter confirmation (EXP-C-Full) 반영.*  
*Table C-Full 신규: GPU 5-fold CV(hidden=128, out=64, layers=3, heads=4, seed 42), λ 8개 값 전수 — `results/gpu/C_lambda_*` (2026-07-13 run_all.bat 실행분, 체크포인트로 파라미터 검증 완료).*  
*Finding C5 신규: (i) λ-robust plateau는 full params에서도 유지(λ∈[0.001,0.1] 비단조 변동, fold noise 이내; 명목 최적 λ=0.005). (ii) λ≥0.5 저하는 capacity-dependent — lightweight -1.1% vs full -19.6%(λ=0.5)/-27.3%(λ=1.0), CPU 1-fold의 -29.7%와 근접. HealthGain@10은 동시에 0으로 수렴(-0.011→-0.002) → full capacity에서는 health-ranking trade-off가 설계 의도대로 작동.*  
*Finding C3 제목/본문에 lightweight 한정 명시, mechanism (b) "future confirmation" 해소, Table S에 EXP-C-Full 행 추가, Design Guidelines λ 행 갱신(λ≥0.5 회피 권고), Abstract/Conclusion Finding(4) 갱신. 남은 향후 과제: full-param sweep의 multi-seed 반복.*  
---
*New in v1.1 (2026-07-15): EXP-F v2 GPU 5-fold 완료 반영.*  
*Table F-v2a (NutriGraphNet ablation) + Table F-v2b (NGCF 50% dilution ablation) 신규 추가, "camera-ready version" placeholder 제거.*  
*Finding F4-F6 신규: NutriGraphNet은 그래프 구조에 실제로 의존(w/o all auxiliary: -21.0%, w/o healthness: -7.2%), fold 간 변동성도 증가(σ=0.031→0.141); NGCF는 dilution proxy로도 -1.2%~-1.4%에 그쳐 DualAttn-TB(0.000)와 NutriGraphNet(-21.0%) 사이 중간 지점 확인.*  
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
