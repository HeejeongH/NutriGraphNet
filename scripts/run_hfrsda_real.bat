@echo off
echo ============================================================
echo  Faithful HFRS-DA baseline (variant hfrsda_real)
echo  Full density (Table 1) + sparsity sweep (Table B)
echo ============================================================
echo  Started: %date% %time%
echo ============================================================

:: WHY (2026-07-16)
::   Fills the paper's largest gap: no published health-aware baseline
::   (section 11.6). hfrsda_real is a faithful re-implementation of
::   Forouzandeh et al. 2024's official HFRS-DA -- NOT the DualAttn-TB control.
::   Same 5-fold sampled-ranking protocol, seed, and full parameters as every
::   other model, so its numbers drop straight into Table 1 and Table B.
::
::   All commands run from the REPOSITORY ROOT. Output goes to results/gpu.
::   HealthGain@K is invalid below full density (health-score baseline
::   dilution, see section 11.3) -- ignore it in the <100%% runs.

echo.
echo [1/6] HFRS-DA at 100%% density (Table 1 / Table B 100%%)
python src\nutrigraphnet_v2.py --variants hfrsda_real --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/HFRSDA_real_100pct
echo   Done 100%%

echo.
echo [2/6] HFRS-DA at 10%% density
python src\nutrigraphnet_v2.py --variants hfrsda_real --interaction_ratio 0.1 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/HFRSDA_real_10pct
echo   Done 10%%

echo.
echo [3/6] HFRS-DA at 30%% density
python src\nutrigraphnet_v2.py --variants hfrsda_real --interaction_ratio 0.3 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/HFRSDA_real_30pct
echo   Done 30%%

echo.
echo [4/6] HFRS-DA at 50%% density
python src\nutrigraphnet_v2.py --variants hfrsda_real --interaction_ratio 0.5 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/HFRSDA_real_50pct
echo   Done 50%%

echo.
echo [5/6] HFRS-DA at 70%% density
python src\nutrigraphnet_v2.py --variants hfrsda_real --interaction_ratio 0.7 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/HFRSDA_real_70pct
echo   Done 70%%

echo.
echo [6/6] lambda_health sensitivity is NOT run for HFRS-DA (it has no
echo       lambda_health knob; its health signal is fixed by the SLA branch).

echo.
echo ============================================================
echo  HFRS-DA baseline complete!
echo  Finished: %date% %time%
echo ============================================================
python src\generate_summary_gpu.py
echo.
echo Benchmark at 100%% density (NutriGraphNet lambda=0.005, results/gpu/C_lambda_0.005):
echo    NutriGraphNet:  AUC=0.8620  HR@10=0.7484  NDCG@10=0.4279
echo    DualAttn-TB:    AUC=0.8551  HR@10=0.7340  NDCG@10=0.5977
echo Local 1-fold/15-epoch smoke of hfrsda_real: AUC=0.831 HR@10=0.630 NDCG@10=0.485
echo (full 5-fold/300-epoch numbers will be higher).
echo.
echo SANITY: hfrsda_real must report distinct per-density HR@10 and a real AUC
echo (not 0.000). If any density crashes on ingredient attach, check that
echo food-contains-ingredient edges survive subsampling (they should -- only
echo user-food edges are subsampled).
pause
