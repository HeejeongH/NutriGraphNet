@echo off
echo ============================================================
echo  NutriGraphNet GPU Experiments - EXP-F v2 (Valid Ablation)
echo ============================================================
echo  Started: %date% %time%
echo ============================================================

:: ===== EXP-F v2a: NutriGraphNet graph component ablation =====
:: NutriGraphNet routes message-passing through all 9 edge types,
:: so removing an edge type at the data level directly changes its
:: forward pass (unlike HFRS-DA, which is topology-invariant).
echo.
echo [1/2] EXP-F v2a: NutriGraphNet ablation (full graph model)

python nutrigraphnet_v2.py --variants full --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ablation_full_graph
echo   Done: full_graph

python nutrigraphnet_v2.py --variants full --ablate_no_ingredient --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ablation_no_ingredient
echo   Done: no_ingredient

python nutrigraphnet_v2.py --variants full --ablate_no_time --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ablation_no_time
echo   Done: no_time

python nutrigraphnet_v2.py --variants full --ablate_no_healthness --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ablation_no_healthness
echo   Done: no_healthness

python nutrigraphnet_v2.py --variants full --ablate_no_food_similar --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ablation_no_food_similar
echo   Done: no_food_similar

python nutrigraphnet_v2.py --variants full --ablate_no_ingredient --ablate_no_time --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ablation_no_ingredient_time
echo   Done: no_ingredient_time

python nutrigraphnet_v2.py --variants full --ablate_no_ingredient --ablate_no_time --ablate_no_food_similar --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ablation_no_all_auxiliary
echo   Done: no_all_auxiliary

echo [EXP-F v2a] Complete!

:: ===== EXP-F v2b: NGCF topology ablation (50%% interaction dilution) =====
:: NGCF's _propagate() only consumes user-food edges, so auxiliary edges
:: cannot be ablated directly. Instead --ablation_model ngcf dilutes 50%%
:: of the interactions belonging to foods connected to the targeted
:: auxiliary edge type, giving a valid (if indirect) topology ablation.
echo.
echo [2/2] EXP-F v2b: NGCF topology ablation (auxiliary-edge interaction dilution)

python nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_full_graph
echo   Done: full_graph

python nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --ablate_no_ingredient --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_no_ingredient
echo   Done: no_ingredient

python nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --ablate_no_time --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_no_time
echo   Done: no_time

python nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --ablate_no_healthness --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_no_healthness
echo   Done: no_healthness

python nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --ablate_no_food_similar --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_no_food_similar
echo   Done: no_food_similar

python nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --ablate_no_ingredient --ablate_no_time --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_no_ingredient_time
echo   Done: no_ingredient_time

python nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --ablate_no_ingredient --ablate_no_time --ablate_no_food_similar --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_no_all_auxiliary
echo   Done: no_all_auxiliary

echo [EXP-F v2b] Complete!

:: ===== 결과 요약 =====
echo.
echo ============================================================
echo  EXP-F v2 complete!
echo  Finished: %date% %time%
echo ============================================================
python run_analysis_experiments.py --exp summary
echo.
echo Results saved to: results/gpu/F_ablation_* and results/gpu/F_ngcf_dilution_*
echo Update paper Section 6.6 (EXP-F v2) with these results!
pause
