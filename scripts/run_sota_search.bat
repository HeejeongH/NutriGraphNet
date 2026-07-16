@echo off
echo ============================================================
echo  NutriGraphNet SOTA search: decoder diagnostic + candidates
echo ============================================================
echo  Started: %date% %time%
echo ============================================================

:: GOAL (2026-07-17)
::   Beat the faithful HFRS-DA baseline (at 100%%: AUC 0.880, HR@10 0.782,
::   NDCG@10 0.625) while keeping NutriGraphNet's low-density strength
::   (HR@10 0.738 at 10%%) and its tunable health-gradient routing.
::
::   Reference points already on disk:
::     v2 (full, HybridDecoder): 100%% HR@10 0.729 NDCG 0.411 | 10%% HR@10 0.738
::     HFRS-DA:                  100%% HR@10 0.782 NDCG 0.625 | 10%% HR@10 0.742
::
::   Two new variants, SAME DualChannelEncoder, only the decoder changes:
::     full_cos  = rank-optimal cosine decoder (isolates the decoder as the fix)
::     full_hcos = cosine + direct health-score gate (SOTA candidate)
::
::   All runs from the REPOSITORY ROOT. Same protocol/seed/params as every
::   other model, so numbers drop straight into Table 1 / Table B.
::
::   DECISION LOGIC (read after it finishes):
::     Step 1 -- does the cosine decoder fix NDCG at 100%% AND keep low-density
::       HR? Compare full_cos to v2. If full_cos 100%% NDCG >> 0.411 and 10%%
::       HR@10 ~ 0.738, the decoder was the whole problem.
::     Step 2 -- does the health gate add more? Compare full_hcos to full_cos.
::     Step 3 -- SOTA check: is full_hcos (or full_cos) >= HFRS-DA on HR@10 AND
::       NDCG@10 at 100%%, and still best-or-tied at 10%%? If yes -> SOTA.

echo.
echo === 100%% density (Table 1 comparison; the ranking-quality test) ===
echo [1/8] full_cos @ 100%%
python src\nutrigraphnet_v2.py --variants full_cos --lambda_health 0.01 --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA_full_cos_100pct
echo   Done full_cos 100%%
echo [2/8] full_hcos @ 100%%
python src\nutrigraphnet_v2.py --variants full_hcos --lambda_health 0.01 --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA_full_hcos_100pct
echo   Done full_hcos 100%%

echo.
echo === 10%% density (the low-density-survival test) ===
echo [3/8] full_cos @ 10%%
python src\nutrigraphnet_v2.py --variants full_cos --lambda_health 0.01 --interaction_ratio 0.1 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA_full_cos_10pct
echo   Done full_cos 10%%
echo [4/8] full_hcos @ 10%%
python src\nutrigraphnet_v2.py --variants full_hcos --lambda_health 0.01 --interaction_ratio 0.1 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA_full_hcos_10pct
echo   Done full_hcos 10%%

echo.
echo === 30/50/70%% for the winner-so-far (fill Table B) ===
echo [5/8] full_hcos @ 30%%
python src\nutrigraphnet_v2.py --variants full_hcos --lambda_health 0.01 --interaction_ratio 0.3 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA_full_hcos_30pct
echo [6/8] full_hcos @ 50%%
python src\nutrigraphnet_v2.py --variants full_hcos --lambda_health 0.01 --interaction_ratio 0.5 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA_full_hcos_50pct
echo [7/8] full_hcos @ 70%%
python src\nutrigraphnet_v2.py --variants full_hcos --lambda_health 0.01 --interaction_ratio 0.7 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA_full_hcos_70pct
echo [8/8] full_cos @ 30/50/70%% too (so both curves are complete)
python src\nutrigraphnet_v2.py --variants full_cos --lambda_health 0.01 --interaction_ratio 0.3 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA_full_cos_30pct
python src\nutrigraphnet_v2.py --variants full_cos --lambda_health 0.01 --interaction_ratio 0.5 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA_full_cos_50pct
python src\nutrigraphnet_v2.py --variants full_cos --lambda_health 0.01 --interaction_ratio 0.7 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA_full_cos_70pct

echo.
echo ============================================================
echo  SOTA search complete!
echo  Finished: %date% %time%
echo ============================================================
echo Compare (100%%):  v2 HR@10 0.729 NDCG 0.411  ^|  HFRS-DA HR@10 0.782 NDCG 0.625
echo Compare (10%%):   v2 HR@10 0.738            ^|  HFRS-DA HR@10 0.742
echo Read results/gpu/SOTA_*/results_*.json and apply the decision logic above.
pause
