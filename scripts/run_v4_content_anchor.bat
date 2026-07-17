@echo off
echo ============================================================
echo  NutriGraphNet v4: content-anchor branch (SOTA attempt)
echo ============================================================
echo  Started: %date% %time%
echo ============================================================

:: WHY (2026-07-17) -- grounded in a MEASUREMENT, not another guess.
::   Four tuning rounds failed to close the NDCG gap to HFRS-DA (0.625).
::   src/diag_food_collapse.py then measured the actual food embeddings from
::   checkpoints: NutriGraphNet's collaborative encoder squeezes food reprs
::   into ~11/64 effective dims vs HFRS-DA's ~20 -- user-side message passing
::   dilutes food identity, which fits good HR (0.729) but poor NDCG (0.411).
::   Every high-NDCG model (HFRS-DA 0.625, MF 0.618, DualAttn-TB 0.598) is
::   content/identity anchored; every low-NDCG model is collaboration-washed.
::
::   v4 (variant suffix _ca) adds an identity-preserving path: each food's own
::   nutrition features + an attention pool of its ingredient embeddings,
::   residual-fused into the encoder output. This is the property the good
::   rankers share, added ON TOP of the collaborative+auxiliary encoder that
::   wins at low density. Combined with the cosine decoder (full_ca_cos).
::
::   Run from REPOSITORY ROOT. Reference @100%%: v2 NDCG 0.411 | full_cos 0.437
::   | HFRS-DA 0.625 (target). @10%%: v2/HFRS-DA HR@10 ~0.74.
::
::   GO/NO-GO (this is the lead 2-week candidate, not a promise):
::     100%% NDCG@10 >= ~0.58 AND 10%% HR@10 >= ~0.73  -> real progress, push on
::       (tune gamma, add health fusion full_ca_hcos, full sparsity sweep).
::     100%% NDCG@10 still < ~0.48  -> identity-anchor is not enough; the honest
::       two-model paper is the outcome. No more blind rounds.

echo.
echo [1/3] full_ca_cos @100%%  (does the identity anchor fix NDCG?)
python src\nutrigraphnet_v2.py --variants full_ca_cos --lambda_health 0.01 --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/V4_ca_cos_100pct
echo   Done 100%%

echo [2/3] full_ca_hcos @100%%  (anchor + health fusion)
python src\nutrigraphnet_v2.py --variants full_ca_hcos --lambda_health 0.01 --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/V4_ca_hcos_100pct
echo   Done 100%%

echo [3/3] full_ca_cos @10%%  (does the anchor keep low-density strength?)
python src\nutrigraphnet_v2.py --variants full_ca_cos --lambda_health 0.01 --interaction_ratio 0.1 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/V4_ca_cos_10pct
echo   Done 10%%

echo.
echo ============================================================
echo  v4 content-anchor complete!  Finished: %date% %time%
echo ============================================================
echo Compare @100%%: v2 NDCG 0.411 ^| full_cos 0.437 ^| HFRS-DA 0.625 (target)
echo Compare @10%%:  HR@10 v2 0.738 ^| HFRS-DA 0.742
echo NDCG jumped toward 0.58+ ^&^& 10%% HR held ^>=0.73 -^> keep going. Else stop, go honest.
pause
