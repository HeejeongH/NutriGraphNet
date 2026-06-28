# Allrecipes 5-fold 최종 확정 수치 (λ_health=0.01)
# Source: nutrigraphnet_v2.py --variants full,mf,lightgcn,ngcf,sgl,hfrsda --epochs 300 --n_folds 5
# Date: 2026-06-28
#
# NOTE: 이전 논문 버전(ALLRECIPES_OLD)의 baselines(MF, LightGCN)은 Sampled-100 protocol로
#       HR@K 수치가 낮았음(e.g., MF HR@10=0.052). 신규 실험은 동일 full-ranking evaluation
#       protocol을 사용하여 모든 모델을 공정하게 비교함.
#
# 신규 결과 요약 (5-fold CV, Allrecipes):
#   MF:       AUC=0.8125, F1=0.7448, HR@10=0.7550, NDCG@10=0.5530, MRR=0.4968
#   LightGCN: AUC=0.8914, F1=0.8431, HR@10=0.8020, NDCG@10=0.6140, MRR=0.5604
#   NGCF:     AUC=0.8921, F1=0.8394, HR@10=0.8000, NDCG@10=0.5960, MRR=0.5372
#   SGL:      AUC=0.7953, F1=0.7008, HR@10=0.6040, NDCG@10=0.4009, MRR=0.3505
#   HFRSDA:   AUC=0.8827, F1=0.8251, HR@10=0.4760, NDCG@10=0.3281, MRR=0.3012
#   FULL:     AUC=0.8521, F1=0.7811, HR@10=0.7020, NDCG@10=0.3820, MRR=0.2956
#             HealthGain@10=-0.0090

ALLRECIPES = {
    'mf': {
        'auc':   (0.8125, 0.0037), 'f1':   (0.7448, 0.0047),
        'HR@5':  (0.6640, 0.0265), 'HR@10': (0.7550, 0.0255), 'HR@20': (0.8230, 0.0294),
        'NDCG@5': (0.5233, 0.0194), 'NDCG@10': (0.5530, 0.0154), 'NDCG@20': (0.5705, 0.0156),
        'MRR':   (0.4968, 0.0147),
        'HealthGain@10': (None, None),
    },
    'lightgcn': {
        'auc':   (0.8914, 0.0013), 'f1':   (0.8431, 0.0016),
        'HR@5':  (0.7330, 0.0264), 'HR@10': (0.8020, 0.0284), 'HR@20': (0.8650, 0.0270),
        'NDCG@5': (0.5915, 0.0241), 'NDCG@10': (0.6140, 0.0252), 'NDCG@20': (0.6300, 0.0251),
        'MRR':   (0.5604, 0.0272),
        'HealthGain@10': (None, None),
    },
    'ngcf': {
        'auc':   (0.8921, 0.0026), 'f1':   (0.8394, 0.0030),
        'HR@5':  (0.7270, 0.0284), 'HR@10': (0.8000, 0.0263), 'HR@20': (0.8620, 0.0304),
        'NDCG@5': (0.5724, 0.0289), 'NDCG@10': (0.5960, 0.0305), 'NDCG@20': (0.6119, 0.0297),
        'MRR':   (0.5372, 0.0327),
        'HealthGain@10': (None, None),
    },
    'sgl': {
        'auc':   (0.7953, 0.0019), 'f1':   (0.7008, 0.0010),
        'HR@5':  (0.4900, 0.0409), 'HR@10': (0.6040, 0.0361), 'HR@20': (0.6900, 0.0341),
        'NDCG@5': (0.3639, 0.0260), 'NDCG@10': (0.4009, 0.0258), 'NDCG@20': (0.4227, 0.0253),
        'MRR':   (0.3505, 0.0219),
        'HealthGain@10': (None, None),
    },
    'hfrsda': {
        'auc':   (0.8827, 0.0017), 'f1':   (0.8251, 0.0030),
        'HR@5':  (0.3740, 0.0532), 'HR@10': (0.4760, 0.0483), 'HR@20': (0.6130, 0.0461),
        'NDCG@5': (0.2950, 0.0314), 'NDCG@10': (0.3281, 0.0294), 'NDCG@20': (0.3625, 0.0274),
        'MRR':   (0.3012, 0.0224),
        'HealthGain@10': (None, None),
    },
    'full': {
        'auc':   (0.8521, 0.0117), 'f1':   (0.7811, 0.0184),
        'HR@5':  (0.4930, 0.1376), 'HR@10': (0.7020, 0.0760), 'HR@20': (0.8230, 0.0294),
        'NDCG@5': (0.3138, 0.1048), 'NDCG@10': (0.3820, 0.0825), 'NDCG@20': (0.4127, 0.0659),
        'MRR':   (0.2956, 0.0827),
        'HealthGain@10': (-0.0090, 0.0033),
    },
}

# Wilcoxon signed-rank 유의성 (5-fold, one-tailed: full > baseline, from experiment output)
# full > mf:    AUC p=0.0312*, F1 p=0.0312*
# full > sgl:   AUC p=0.0312*, F1 p=0.0312*, HR@20 p=0.0312*
# full > hfrsda: HR@10 p=0.0312*, HR@20 p=0.0312*
# full > lightgcn: 모두 불유의 (FULL이 낮음)
# full > ngcf:    모두 불유의 (FULL이 낮음)
SIG = {
    'mf':       {'auc': '*', 'f1': '*', 'HR@5': None, 'HR@10': None, 'HR@20': None,
                 'NDCG@5': None, 'NDCG@10': None, 'NDCG@20': None, 'MRR': None},
    'lightgcn': {'auc': None, 'f1': None, 'HR@5': None, 'HR@10': None, 'HR@20': None,
                 'NDCG@5': None, 'NDCG@10': None, 'NDCG@20': None, 'MRR': None},
    'ngcf':     {'auc': None, 'f1': None, 'HR@5': None, 'HR@10': None, 'HR@20': None,
                 'NDCG@5': None, 'NDCG@10': None, 'NDCG@20': None, 'MRR': None},
    'sgl':      {'auc': '*', 'f1': '*', 'HR@5': None, 'HR@10': None, 'HR@20': '*',
                 'NDCG@5': None, 'NDCG@10': None, 'NDCG@20': None, 'MRR': None},
    'hfrsda':   {'auc': None, 'f1': None, 'HR@5': None, 'HR@10': '*', 'HR@20': '*',
                 'NDCG@5': None, 'NDCG@10': None, 'NDCG@20': None, 'MRR': None},
}

print("Allrecipes data loaded OK")
print(f"  Models: {list(ALLRECIPES.keys())}")
