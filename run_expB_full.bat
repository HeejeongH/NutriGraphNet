@echo off
echo ============================================================
echo  NutriGraphNet GPU Experiments - EXP-B MISSING RUN
echo  NutriGraphNet ('full') across the sparsity sweep
echo ============================================================
echo  Started: %date% %time%
echo ============================================================

:: WHY THIS EXISTS (2026-07-15)
::   The original EXP-B (run_all.bat) ran only:
::       --variants mf,lightgcn,ngcf,sgl,hfrsda
::   i.e. NutriGraphNet ('full') was NEVER run in the sparsity sweep.
::   The paper's Table B nonetheless carries a "NutriGraphNet" column,
::   populated with the 'hfrsda' (HFRS-DA baseline) numbers -- confirmed by
::   (a) results/gpu/B_sparsity_*/ containing no results_full.json,
::   (b) Table B's 100%% row being byte-identical to Table 1's HFRS-DA row,
::   (c) that data having HealthGain@10=None / health loss=0.0, which is the
::       HFRS-DA signature ('full' always reports HealthGain@10 ~= -0.010).
::   This script produces the real NutriGraphNet sparsity curve so Table B
::   (and Abstract claim 2, Contribution 3, Findings B1-B6, Table S,
::   Design Guidelines, Conclusion 2) can be rewritten on measured data.
::
:: NON-DESTRUCTIVE BY DESIGN
::   Output goes to results/gpu/B_full_*pct, NOT into B_sparsity_*pct.
::   Reason: nutrigraphnet_v2.py rewrites all_results.json with only the
::   variants of the current invocation, and generate_summary_gpu.py reads
::   all_results.json (not results_*.json). Writing 'full' into the existing
::   folders would silently drop mf/lightgcn/ngcf/sgl/hfrsda from SUMMARY.
::   The "B_" prefix still maps to the B_sparsity group, so the new folders
::   are picked up automatically as B_full_10pct/full, etc.
::
:: COMPARABILITY
::   Same seed (42), same full parameters (hidden=128, out=64, layers=3,
::   heads=4), same 5-fold protocol as the existing baselines. Subsampling is
::   seeded (torch.manual_seed(args.seed) before randperm), so each density
::   draws the exact same interaction subset the baselines saw.
::
:: LAMBDA CHOICE: 0.01
::   The paper's recommended default and the code default; sits on the
::   lambda-robust plateau. Deliberately NOT 0.005 (the nominal argmax at
::   100%% density) -- picking the best-performing lambda on full-density data
::   and then applying it across the sweep would be selection-biased.
::   NOTE: Table 1 currently reports NutriGraphNet at lambda=0.005
::   (HR@10=0.7484). If Table B uses lambda=0.01, the 100%% point will read
::   0.7308 instead. Decide whether to move Table 1 to lambda=0.01 for
::   consistency before writing the tables.

echo.
echo [1/5] EXP-B: NutriGraphNet at 10%% interaction density
python nutrigraphnet_v2.py --variants full --lambda_health 0.01 --interaction_ratio 0.1 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/B_full_10pct
echo   Done density=10%%

echo.
echo [2/5] EXP-B: NutriGraphNet at 30%% interaction density
python nutrigraphnet_v2.py --variants full --lambda_health 0.01 --interaction_ratio 0.3 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/B_full_30pct
echo   Done density=30%%

echo.
echo [3/5] EXP-B: NutriGraphNet at 50%% interaction density
python nutrigraphnet_v2.py --variants full --lambda_health 0.01 --interaction_ratio 0.5 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/B_full_50pct
echo   Done density=50%%

echo.
echo [4/5] EXP-B: NutriGraphNet at 70%% interaction density
python nutrigraphnet_v2.py --variants full --lambda_health 0.01 --interaction_ratio 0.7 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/B_full_70pct
echo   Done density=70%%

echo.
echo [5/5] EXP-B: NutriGraphNet at 100%% interaction density (reproducibility check)
python nutrigraphnet_v2.py --variants full --lambda_health 0.01 --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/B_full_100pct
echo   Done density=100%%

echo [EXP-B full] Complete!

echo.
echo ============================================================
echo  EXP-B NutriGraphNet sweep complete!
echo  Finished: %date% %time%
echo ============================================================
python run_analysis_experiments.py --exp summary
echo.
echo Results written to: results/gpu/B_full_*pct  (existing B_sparsity_* untouched)
echo.
echo SANITY CHECKS before touching the paper:
echo   1. B_full_100pct/full should reproduce C_lambda_0.01/full
echo      (HR@10=0.7308, AUC=0.8545). Same seed/params/density - if these
echo      differ, something is non-deterministic and must be resolved first.
echo   2. B_full_*/full must report HealthGain@10 (~ -0.010), NOT None.
echo      A None here means the wrong variant ran.
echo   3. The curve must NOT match the old "NutriGraphNet" row
echo      (0.656/0.734/0.727/0.737/0.734) - that row is HFRS-DA's data.
pause
