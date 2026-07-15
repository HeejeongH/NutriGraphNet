@echo off
echo ============================================================
echo  NutriGraphNet v3 - FAIR-BUDGET RERUN
echo ============================================================
echo  Started: %date% %time%
echo ============================================================

:: WHY THIS EXISTS (2026-07-16)
::   The first v3 run (run_analysis_experiments.py --exp V3) missed every
::   target it set for itself and came out WORSE than v2 on every metric:
::       AUC     0.8142  (target >=0.88,  v2=0.8620)
::       NDCG@10 0.3361  (target >=0.55,  v2=0.4279)
::       MRR     0.2767  (target >=0.45,  v2=0.3378)
::       HR@10   0.5908                  (v2=0.7308)
::   That is NOT an architecture verdict -- the two models were not given
::   the same training budget:
::       v2 (EXP-B/C/D/F):  epochs=300, lr=1e-3, patience=30
::       v3 (--exp V3):     epochs=100, lr=3e-4, patience=15
::   Cutting the learning rate to 1/3 requires MORE epochs, not fewer; v3 got
::   1/3 of both. Undertraining alone can explain the gap. Evaluation protocol
::   IS comparable (both use sampled ranking, 1 pos + 100 negs, 80/5/15 split),
::   so only the budget needs equalising.
::
:: TWO RUNS TO DISAMBIGUATE
::   A: epochs=300, lr=3e-4  -> v3's designed LR, fair epoch budget.
::                             Isolates "was it just too few epochs?"
::   B: epochs=300, lr=1e-3  -> fully matched to v2.
::                             Isolates "was it the LR?"
::   If BOTH still lose to v2 (NDCG@10 < 0.4279), the RankDotDecoder thesis
::   is genuinely wrong and v2 stays the paper's model.
::
:: KNOWN LIMITATION
::   v3 hardcodes early_stop_patience=15 (nutrigraphnet_v3.py:1008, no CLI
::   flag) vs v2's 30. With lr=3e-4 this may still stop prematurely. If run A
::   early-stops well before 300 epochs, patience -- not epochs -- is the
::   binding constraint and the code needs a --patience flag before any
::   conclusion is drawn. Watch the per-fold stopping epoch in the log.

echo.
echo [1/2] v3 fair budget: epochs=300, lr=3e-4 (designed LR)
python nutrigraphnet_v3.py --data data/processed_data/processed_data_GNN_v5.pkl --output results/v3_e300_lr3e4 --folds 5 --epochs 300 --hidden 128 --out_dim 64 --layers 3 --heads 4 --dropout 0.2 --lr 3e-4 --lambda_health 0.005 --phase1_frac 0.8 --infonce_weight 0.1 --batch_size 4096 --device auto --seed 42
echo   Done: epochs=300, lr=3e-4

echo.
echo [2/2] v3 fully matched to v2: epochs=300, lr=1e-3
python nutrigraphnet_v3.py --data data/processed_data/processed_data_GNN_v5.pkl --output results/v3_e300_lr1e3 --folds 5 --epochs 300 --hidden 128 --out_dim 64 --layers 3 --heads 4 --dropout 0.2 --lr 1e-3 --lambda_health 0.005 --phase1_frac 0.8 --infonce_weight 0.1 --batch_size 4096 --device auto --seed 42
echo   Done: epochs=300, lr=1e-3

echo.
echo ============================================================
echo  v3 fair rerun complete!
echo  Finished: %date% %time%
echo ============================================================
echo.
echo Compare against v2 at 100%% density, lambda=0.005 (results/gpu/C_lambda_0.005):
echo    v2:  AUC=0.8620  HR@10=0.7484  NDCG@10=0.4279  MRR=0.3378
echo Prior undertrained v3 (epochs=100, lr=3e-4, results/v3):
echo    v3:  AUC=0.8142  HR@10=0.5908  NDCG@10=0.3361  MRR=0.2767
echo.
echo DECISION RULE:
echo   NDCG@10 >= 0.55 and AUC >= 0.86  -> v3 becomes the paper's model.
echo                                       EXP-B/C/D/F must then be rerun on v3.
echo   NDCG@10 improves but stays < v2  -> v2 stays the model; report v3 as a
echo                                       negative architectural result.
echo   No improvement over the 100-epoch run -> budget was not the cause;
echo                                       check the patience limitation above.
pause
