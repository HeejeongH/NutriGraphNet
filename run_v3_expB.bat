@echo off
echo ============================================================
echo  NutriGraphNet v3 - EXP-B LOW-DENSITY PROBE
echo ============================================================
echo  Started: %date% %time%
echo ============================================================

:: WHY ONLY 10%% AND 30%% (2026-07-16)
::   The fair rerun settled that v3 beats v2 at full density on every metric
::   (AUC .8718 vs .8620, HR@10 .7512 vs .7484, NDCG@10 .5114 vs .4279,
::   MRR .4448 vs .3378) -- the RankDotDecoder thesis holds once v3 is given
::   v2's budget (300 epochs, lr=1e-3).
::
::   But full density is not where the paper's thesis lives. NGCF wins there
::   (HR@10 .7844) and always did. The entire argument rests on the sparse
::   regime, where v2 leads: HR@10 .7380 at 10%%, .7552 at 30%%.
::
::   So the only question worth 13 GPU-hours is: does v3 keep the low-density
::   crown? This probe answers it in 2 runs instead of 20. Commit to the full
::   v3 migration (EXP-B/C/D/F) only if v3 wins here.
::
:: WHAT THIS NEEDS
::   --interaction_ratio was ported from v2 into nutrigraphnet_v3.py for this.
::   Verified: identical (seed, ratio) draws the identical subset the v2
::   baselines saw, so the comparison is apples-to-apples.
::
:: STILL MISSING FOR A FULL v3 MIGRATION
::   v3 has no --ablate_* flags and no baselines, so EXP-F (the auxiliary-edge
::   ablation carrying the paper's mechanism claim) CANNOT run on v3 yet. That
::   port is the next code task if this probe succeeds.

echo.
echo [1/2] v3 at 10%% interaction density
python nutrigraphnet_v3.py --data data/processed_data/processed_data_GNN_v5.pkl --output results/v3_expB_10pct --folds 5 --epochs 300 --hidden 128 --out_dim 64 --layers 3 --heads 4 --dropout 0.2 --lr 1e-3 --lambda_health 0.005 --phase1_frac 0.8 --infonce_weight 0.1 --batch_size 4096 --device auto --seed 42 --interaction_ratio 0.1
echo   Done density=10%%

echo.
echo [2/2] v3 at 30%% interaction density
python nutrigraphnet_v3.py --data data/processed_data/processed_data_GNN_v5.pkl --output results/v3_expB_30pct --folds 5 --epochs 300 --hidden 128 --out_dim 64 --layers 3 --heads 4 --dropout 0.2 --lr 1e-3 --lambda_health 0.005 --phase1_frac 0.8 --infonce_weight 0.1 --batch_size 4096 --device auto --seed 42 --interaction_ratio 0.3
echo   Done density=30%%

echo.
echo ============================================================
echo  v3 low-density probe complete!
echo  Finished: %date% %time%
echo ============================================================
echo.
echo BENCHMARK -- v2 HR@10 (results/gpu/B_full_*pct), same seed/subset:
echo    10 pct : 0.7380   (best of all 6 models; NGCF 0.5088, HFRS-DA 0.6560)
echo    30 pct : 0.7552   (best of all 6 models; NGCF 0.7012)
echo.
echo DECISION:
echo   v3 HR@10 beats v2 at both densities  - migrate everything to v3.
echo     Next: port --ablate_* into v3, then rerun EXP-B/C/D/F.
echo   v3 loses at low density  - keep v2 as the paper's model for the sparse
echo     thesis, and report v3 only as a full-density decoder improvement.
echo.
echo IGNORE HealthGain@K in these two runs - it is invalid below full density
echo (health-score baseline dilution; see run_kfold_v3 note).
pause
