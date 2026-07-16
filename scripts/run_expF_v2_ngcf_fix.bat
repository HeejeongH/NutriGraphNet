@echo off
echo ============================================================
echo  NutriGraphNet GPU Experiments - EXP-F v2b RERUN (bug fix)
echo  (NutriGraphNet ablation results from the first run are valid
echo   and do NOT need to be rerun - only NGCF dilution is redone)
echo ============================================================
echo  Started: %date% %time%
echo ============================================================

:: Bug fix applied in nutrigraphnet_v2.py: the data-level edge removal
:: (meant only for NutriGraphNet/'full') was running before the NGCF/
:: LightGCN dilution logic could read the same edge_index to find which
:: foods are aux-connected, so dilution silently became a no-op and every
:: NGCF ablation variant matched full_graph exactly. Now ablation_model
:: in ('ngcf','lightgcn') skips the data-level zeroing so dilution works.

echo.
echo [1/1] EXP-F v2b (fixed): NGCF topology ablation (auxiliary-edge interaction dilution)

python src\nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_full_graph
echo   Done: full_graph

python src\nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --ablate_no_ingredient --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_no_ingredient
echo   Done: no_ingredient

python src\nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --ablate_no_time --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_no_time
echo   Done: no_time

python src\nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --ablate_no_healthness --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_no_healthness
echo   Done: no_healthness

python src\nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --ablate_no_food_similar --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_no_food_similar
echo   Done: no_food_similar

python src\nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --ablate_no_ingredient --ablate_no_time --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_no_ingredient_time
echo   Done: no_ingredient_time

python src\nutrigraphnet_v2.py --variants ngcf --ablation_model ngcf --ablate_no_ingredient --ablate_no_time --ablate_no_food_similar --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/F_ngcf_dilution_no_all_auxiliary
echo   Done: no_all_auxiliary

echo [EXP-F v2b RERUN] Complete!

echo.
echo ============================================================
echo  EXP-F v2b rerun complete!
echo  Finished: %date% %time%
echo ============================================================
python src\run_analysis_experiments.py --exp summary
echo.
echo Results overwritten in: results/gpu/F_ngcf_dilution_*
echo Sanity check: values should now DIFFER across ablation variants
echo (unlike the previous buggy run where every variant == full_graph).
pause
