@echo off
echo ============================================================
echo  NutriGraphNet SOTA final attempt: InfoNCE off + cosine
echo ============================================================
echo  Started: %date% %time%
echo ============================================================

:: WHY (2026-07-17) -- the last principled lever.
::   Rounds 1-2 ruled out the decoder (cosine alone gave +0.026 NDCG) and
::   over-smoothing (fewer layers made it WORSE and unstable). But those runs
::   showed the InfoNCE contrastive loss stuck at cl=5.5 for all 300 epochs --
::   never decreasing, yet ~40%% of the loss at lambda_cl=0.05. A contrastive
::   term that never converges is pulling embeddings toward a geometry good for
::   AUC but bad for fine ranking (NDCG). This turns it off.
::
::   variant no_cl_cos = cosine decoder + InfoNCE OFF + DropEdge OFF
::   (verified: training log shows cl=0.0000). Everything else identical.
::
::   DECISION (this is the last attempt):
::     100%% NDCG@10 >= ~0.55  -> InfoNCE was the blocker. Pursue SOTA:
::        also run no_cl_hcos (health fusion) and the sparsity sweep.
::     100%% NDCG@10 still ~0.44 -> stop tuning. Switch to the honest
::        two-model framing (NutriGraphNet + HFRS-DA, health-routing as the
::        unique contribution). No more SOTA rounds.
::
::   Run from REPOSITORY ROOT. Reference: v2 NDCG 0.411 | full_cos 0.437 |
::   v3 0.511 | HFRS-DA 0.625 (target).

echo.
echo [1/3] no_cl_cos @100%%  (does killing InfoNCE fix NDCG?)
python src\nutrigraphnet_v2.py --variants no_cl_cos --lambda_health 0.01 --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA3_nocl_cos_100pct
echo   Done 100%%

echo [2/3] no_cl_hcos @100%%  (InfoNCE off + proper health fusion)
python src\nutrigraphnet_v2.py --variants no_cl_hcos --lambda_health 0.01 --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA3_nocl_hcos_100pct
echo   Done 100%%

echo [3/3] no_cl_cos @10%%  (does it keep low-density strength?)
python src\nutrigraphnet_v2.py --variants no_cl_cos --lambda_health 0.01 --interaction_ratio 0.1 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/SOTA3_nocl_cos_10pct
echo   Done 10%%

echo.
echo ============================================================
echo  Final attempt complete!  Finished: %date% %time%
echo ============================================================
echo Compare @100%%: v2 NDCG 0.411 ^| full_cos 0.437 ^| HFRS-DA 0.625 (target)
echo If NDCG jumped to ~0.55+: InfoNCE was it. If still ~0.44: stop, go honest.
pause
