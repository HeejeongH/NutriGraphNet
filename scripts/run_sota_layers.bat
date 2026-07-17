@echo off
echo ============================================================
echo  NutriGraphNet SOTA search round 2: over-smoothing + fusion
echo ============================================================
echo  Started: %date% %time%
echo ============================================================

:: WHY (2026-07-17)
::   Round 1 finding: swapping ONLY the decoder (full_cos) barely moved NDCG
::   (0.411 -> 0.437 at 100%%), so the decoder was NOT the lever. The ranking
::   gap to HFRS-DA (NDCG 0.625) comes from the encoder's embedding geometry.
::   full_cos retrieves well (HR@10 0.758) but orders badly (NDCG 0.437) --
::   the signature of OVER-SMOOTHING: 3 layers x 9 edge types make food
::   embeddings too similar to rank finely. This is the same effect EXP-G
::   already found for LightGCN at L=3 (-8.1%%).
::
::   TEST 1 (over-smoothing): run full_cos at num_layers = 1 and 2 at 100%%.
::     If NDCG rises sharply as layers drop, over-smoothing is confirmed and
::     shallower depth is the fix. (health branch off via lambda_health 0 so
::     only depth changes.)
::   TEST 2 (proper health fusion): full_hcos (now concat+project, not the
::     failed multiplicative gate) at the best depth.
::
::   All from REPOSITORY ROOT. 100%% density only (the Table 1 ranking test).
::   Reference: v2 NDCG 0.411 | full_cos(L3) 0.437 | v3 0.511 | HFRS-DA 0.625.

echo.
echo [1/5] full_cos  L=1  @100%%  (over-smoothing test)
python src\nutrigraphnet_v2.py --variants full_cos --lambda_health 0.0 --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 1 --heads 4 --seed 42 --output_dir results/gpu/SOTA2_cos_L1_100pct
echo   Done L=1

echo [2/5] full_cos  L=2  @100%%
python src\nutrigraphnet_v2.py --variants full_cos --lambda_health 0.0 --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 2 --heads 4 --seed 42 --output_dir results/gpu/SOTA2_cos_L2_100pct

echo [3/5] full_hcos L=2  @100%%  (proper concat+project fusion)
python src\nutrigraphnet_v2.py --variants full_hcos --lambda_health 0.01 --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 2 --heads 4 --seed 42 --output_dir results/gpu/SOTA2_hcos_L2_100pct

echo [4/5] full_hcos L=1  @100%%
python src\nutrigraphnet_v2.py --variants full_hcos --lambda_health 0.01 --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 1 --heads 4 --seed 42 --output_dir results/gpu/SOTA2_hcos_L1_100pct

echo [5/5] winner @10%%  (does the shallow model keep low-density strength?)
echo   -- run AFTER reading 1-4; set the variant/layers of the best 100%% model:
echo   python src\nutrigraphnet_v2.py --variants full_hcos --lambda_health 0.01 --interaction_ratio 0.1 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 2 --heads 4 --seed 42 --output_dir results/gpu/SOTA2_best_10pct

echo.
echo ============================================================
echo  Round 2 complete!  Finished: %date% %time%
echo ============================================================
echo Target NDCG@10 @100%%: beat HFRS-DA 0.625 (also HR@10 0.782).
echo If NDCG climbs as L drops: over-smoothing confirmed -> shallow model is SOTA path.
echo If NDCG stays ~0.44 regardless of L: over-smoothing is NOT it -> rethink.
pause
