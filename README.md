# LoL Challenger Ranked Games Win Prediction

이 프로젝트는 LoL 챌린저 랭크 경기의 10분/15분 데이터를 이용해 **블루팀의 최종 승패(`blueWins`)**를 예측하는 이진 분류 프로젝트입니다.

담당 모델은 다음 두 가지입니다.

1. **XGBoost**
2. **MLP, Multi-Layer Perceptron**

핵심 분석 질문은 다음과 같습니다.

- 10분 데이터와 15분 데이터 중 어느 시점의 승패 예측 성능이 더 높은가?
- XGBoost와 MLP 중 tabular game data에 더 적합한 모델은 무엇인가?
- 승패 예측에 가장 크게 기여하는 초반 지표는 무엇인가?

---

## 1. 프로젝트 구조

```text
lol_project/
│
├── data/
│   ├── Challenger_Ranked_Games_10minute.csv
│   └── Challenger_Ranked_Games_15minute.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_xgboost.ipynb
│   ├── 03_mlp.ipynb
│   └── 04_improved_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── metrics_utils.py
│   ├── train_xgboost.py
│   ├── train_mlp.py
│   ├── make_model_comparison.py
│   ├── make_ensemble.py
│   ├── run_experiments.py
│   └── run_improved_experiments.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── models/
├── requirements.txt
└── README.md
```

---

## 2. 데이터 설명

두 데이터는 각각 경기 시작 후 10분, 15분 시점의 팀별 정보를 담고 있습니다.

| 파일 | 행 개수 | 컬럼 개수 | 타깃 |
|---|---:|---:|---|
| `Challenger_Ranked_Games_10minute.csv` | 26,409 | 51 | `blueWins` |
| `Challenger_Ranked_Games_15minute.csv` | 26,834 | 51 | `blueWins` |

`blueWins`는 다음 의미를 가집니다.

| 값 | 의미 |
|---:|---|
| 0 | 블루팀 패배 |
| 1 | 블루팀 승리 |

중요한 주의점:

- `gameId`: 경기 식별자이므로 모델 입력에서 제거합니다.
- `blueWins`: 정답값이므로 모델 입력에서 제거합니다.
- `redWins`: `blueWins`의 반대값이므로 모델 입력에 넣으면 데이터 누수가 발생합니다. 반드시 제거합니다.

---

## 3. 전처리 및 Feature Engineering

`src/feature_engineering.py`에서 다음 피처를 생성합니다.

### 3.1 블루팀-레드팀 차이 피처

승패 예측에서는 절대값보다 팀 간 차이가 중요합니다.

예시:

```text
diff_totalGolds = blueTotalGolds - redTotalGolds
diff_kill = blueKill - redKill
diff_totalLevel = blueTotalLevel - redTotalLevel
diff_dragon = blueDragon - redDragon
diff_towerKills = blueTowerKills - redTowerKills
```

### 3.2 비율 및 요약 피처

추가로 다음 피처를 생성합니다.

```text
blueKDA, redKDA, diff_KDA
blueGoldPerMinion, redGoldPerMinion, diff_goldPerMinion
blueObjectiveScore, redObjectiveScore, diff_objectiveScore
```

### 3.3 문자열 리스트 컬럼 인코딩

다음 컬럼은 원본에서 `[]`, `['MID_LANE']` 같은 문자열 리스트 형태입니다.

```text
blueFirstTowerLane
redFirstTowerLane
blueDragnoType
redDragnoType
```

코드에서는 이 컬럼을 binary indicator column으로 변환한 뒤, 원본 문자열 컬럼은 제거합니다.

---

## 4. 모델 1: XGBoost

XGBoost는 tabular data에서 강력한 성능을 보이는 gradient boosting tree 모델입니다.

### 실행 방법

```bash
python src/train_xgboost.py \
  --data-file data/Challenger_Ranked_Games_10minute.csv \
  --time-label 10minute

python src/train_xgboost.py \
  --data-file data/Challenger_Ranked_Games_15minute.csv \
  --time-label 15minute
```

빠른 테스트용 실행:

```bash
python src/train_xgboost.py \
  --data-file data/Challenger_Ranked_Games_10minute.csv \
  --time-label 10minute \
  --quick
```

### 주요 하이퍼파라미터

| 파라미터 | 의미 |
|---|---|
| `n_estimators` | 생성할 트리 개수 |
| `max_depth` | 각 트리의 최대 깊이 |
| `learning_rate` | 한 번에 학습하는 정도 |
| `subsample` | 각 트리에 사용할 데이터 비율 |
| `colsample_bytree` | 각 트리에 사용할 피처 비율 |
| `min_child_weight` | 리프 노드 분할을 제한하는 값 |
| `reg_lambda` | L2 정규화 강도 |

### 저장 결과

```text
results/tables/xgboost_10minute_metrics.csv
results/tables/xgboost_10minute_feature_importance.csv
results/figures/xgboost_10minute_feature_importance.png
results/figures/xgboost_10minute_confusion_matrix.png
results/figures/xgboost_10minute_roc_curve.png
models/xgboost_10minute.pkl
```

---

## 5. 모델 2: MLP

MLP는 여러 개의 fully connected layer를 사용해 피처 간 비선형 관계를 학습하는 신경망 모델입니다.

이 프로젝트에서는 다음 구조를 기본값으로 사용합니다.

```text
Input
→ Linear(64) → ReLU → BatchNorm → Dropout(0.2)
→ Linear(32) → ReLU → BatchNorm → Dropout(0.2)
→ Linear(16) → ReLU → BatchNorm → Dropout(0.2)
→ Linear(1)
```

출력층에 sigmoid를 직접 넣지 않고, 학습 시 `BCEWithLogitsLoss`를 사용합니다.

### 실행 방법

```bash
python src/train_mlp.py \
  --data-file data/Challenger_Ranked_Games_10minute.csv \
  --time-label 10minute \
  --epochs 100

python src/train_mlp.py \
  --data-file data/Challenger_Ranked_Games_15minute.csv \
  --time-label 15minute \
  --epochs 100
```

빠른 테스트용 실행:

```bash
python src/train_mlp.py \
  --data-file data/Challenger_Ranked_Games_10minute.csv \
  --time-label 10minute \
  --quick
```

### 저장 결과

```text
results/tables/mlp_10minute_metrics.csv
results/tables/mlp_10minute_history.csv
results/figures/mlp_10minute_loss_curve.png
results/figures/mlp_10minute_confusion_matrix.png
results/figures/mlp_10minute_roc_curve.png
models/mlp_10minute.pt
models/mlp_10minute_scaler.pkl
```

---

## 6. 최종 비교표 만들기

각 모델 스크립트를 실행하면 결과가 `results/tables/`에 저장됩니다.

```text
xgboost_10minute_metrics.csv
mlp_10minute_metrics.csv
xgboost_15minute_metrics.csv
mlp_15minute_metrics.csv
```

이 네 파일의 `accuracy`, `f1`, `roc_auc`를 모아 최종 발표 표를 만들 수 있습니다. 자동 생성은 다음 명령으로 실행합니다.

```bash
python src/make_model_comparison.py
```

전체 실험을 한 번에 다시 실행하려면 다음 명령을 사용합니다.

```bash
python src/run_experiments.py
```

XGBoost 설치가 안 된 환경에서 MLP와 비교표만 빠르게 확인하려면 다음처럼 실행합니다.

```bash
python src/run_experiments.py --skip-xgboost --quick
```

생성 결과:

```text
results/tables/model_comparison.csv
results/tables/model_summary.json
results/figures/model_comparison_metrics.png
results/xgboost_mlp_presentation_notes.md
```

### 노트북 구성

| 노트북 | 목적 |
|---|---|
| `01_eda.ipynb` | 데이터 크기, 타깃 분포, 데이터 누수 위험, 주요 차이 피처를 확인합니다. |
| `02_xgboost.ipynb` | XGBoost baseline 학습, feature importance 확인, threshold tuning 추가 실험을 수행합니다. |
| `03_mlp.ipynb` | MLP baseline 학습, loss history 확인, activation/AdamW/threshold tuning 실험을 수행합니다. |
| `04_improved_experiments.ipynb` | Baseline, threshold tuning, soft voting ensemble 결과를 종합 비교하고 최종 결론을 정리합니다. |

---

## 7. 실험 결과 요약

현재 저장된 실험 결과는 다음과 같습니다.

| 모델 | 시점 | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | 10분 | 0.7490 | 0.7597 | 0.7299 | 0.7445 | 0.8235 |
| MLP | 10분 | 0.7463 | 0.7458 | 0.7491 | 0.7475 | 0.8218 |
| XGBoost | 15분 | 0.8094 | 0.8087 | 0.8108 | 0.8097 | 0.8957 |
| MLP | 15분 | 0.8073 | 0.8126 | 0.7993 | 0.8059 | 0.8957 |

핵심 해석:

- 10분 데이터만 사용해도 약 74~75% 정확도로 최종 승패를 예측할 수 있습니다.
- 15분 데이터에서는 약 80~81% 정확도까지 상승하여, 추가 5분의 게임 흐름 정보가 승패 예측에 유의미하다는 것을 확인했습니다.
- XGBoost가 MLP보다 근소하게 높은 성능을 보였습니다. 이 데이터는 이미지/텍스트가 아닌 tabular data이므로, 트리 기반 앙상블이 신경망보다 더 안정적인 선택일 수 있습니다.
- XGBoost 15분 모델의 주요 피처는 `diff_totalGolds`, `diff_avgLevel`, `diff_totalLevel`, `diff_dragon`, `diff_objectiveScore` 등으로, 골드/레벨 격차와 오브젝트 주도권이 승패 예측에 강하게 작용했습니다.

---

## 8. 추가 개선 실험

기존 baseline 결과를 유지한 상태에서, 추가 실험 결과는 별도 suffix인 `threshold`로 저장했습니다.

```bash
python src/run_improved_experiments.py --quick
```

추가한 개선 방식:

- Validation split에서 decision threshold를 탐색하여 0.5 고정 기준과 비교
- XGBoost hyperparameter search 범위 확장 (`gamma`, `reg_alpha`, 더 넓은 `reg_lambda` 등)
- XGBoost와 MLP의 예측 확률을 결합하는 soft voting ensemble 추가
- 각 모델의 test prediction probability를 CSV로 저장하여 후속 분석 가능하게 구성

현재 추가 실험 결과는 다음과 같습니다.

| 모델 | 실험 | 시점 | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| MLP | threshold | 10분 | 0.7488 | 0.7762 | 0.7008 | 0.7365 | 0.8218 |
| XGBoost | threshold | 10분 | 0.7491 | 0.7631 | 0.7242 | 0.7432 | 0.8233 |
| Soft Voting Ensemble | threshold | 10분 | 0.7456 | 0.7516 | 0.7352 | 0.7433 | 0.8240 |
| MLP | threshold | 15분 | 0.8068 | 0.7975 | 0.8227 | 0.8099 | 0.8957 |
| XGBoost | threshold | 15분 | 0.8092 | 0.8287 | 0.7799 | 0.8035 | 0.8960 |
| Soft Voting Ensemble | threshold | 15분 | 0.8099 | 0.8129 | 0.8056 | 0.8092 | 0.8971 |

추가 실험 해석:

- 10분 MLP는 threshold tuning으로 Accuracy가 `0.7463`에서 `0.7488`로 소폭 향상되었습니다.
- 15분 MLP는 Accuracy는 약간 낮아졌지만 F1-score가 `0.8059`에서 `0.8099`로 향상되어, 승리/패배 예측 균형이 개선되었습니다.
- 15분 Soft Voting Ensemble은 ROC-AUC `0.8971`로 가장 높은 구분 성능을 보였습니다.
- 다만 threshold tuning은 validation 기준에서만 수행해야 하며, test set으로 threshold를 맞추면 데이터 누수가 됩니다. 본 코드에서는 ensemble threshold를 test set에 맞추지 않고 0.5로 고정하여 평가했습니다.

---

## 9. 결과 분석 및 결론

### 9.1 10분 데이터와 15분 데이터 비교

가장 뚜렷한 결과는 **15분 데이터가 10분 데이터보다 일관되게 높은 성능을 보였다는 점**입니다.

```text
XGBoost baseline
- 10분 Accuracy: 0.7490
- 15분 Accuracy: 0.8094

MLP baseline
- 10분 Accuracy: 0.7463
- 15분 Accuracy: 0.8073
```

10분 데이터만 사용해도 약 75% 정확도로 최종 승패 예측이 가능했지만, 15분 데이터에서는 약 81%까지 상승했습니다. 이는 경기 시간이 조금 더 진행되면서 골드 차이, 레벨 차이, 드래곤/타워와 같은 오브젝트 차이가 더 명확해지기 때문입니다.

따라서 이 프로젝트의 첫 번째 결론은 다음과 같습니다.

> 경기 초반 10분 정보만으로도 승패 예측이 가능하지만, 15분 시점의 정보가 추가되면 최종 승패를 훨씬 더 안정적으로 예측할 수 있다.

### 9.2 XGBoost와 MLP 비교

XGBoost와 MLP는 모두 비슷한 수준의 ROC-AUC를 보였지만, baseline 기준으로는 XGBoost가 MLP보다 근소하게 더 높은 성능을 보였습니다.

```text
15분 XGBoost baseline
- Accuracy: 0.8094
- F1-score: 0.8097
- ROC-AUC: 0.8957

15분 MLP baseline
- Accuracy: 0.8073
- F1-score: 0.8059
- ROC-AUC: 0.8957
```

이 데이터는 이미지나 텍스트가 아니라 골드, 킬, 레벨, 오브젝트 수치로 구성된 **정형 tabular data**입니다. 이런 데이터에서는 XGBoost 같은 트리 기반 앙상블 모델이 피처 간 조건과 상호작용을 안정적으로 학습하는 경우가 많습니다.

MLP도 경쟁력 있는 성능을 보였지만, XGBoost를 명확하게 앞서지는 못했습니다. 따라서 모델 비교 관점에서는 다음과 같이 해석할 수 있습니다.

> 본 데이터셋에서는 MLP보다 XGBoost가 약간 더 안정적인 성능을 보였으며, tabular game data에서는 복잡한 신경망이 항상 더 좋은 성능을 내는 것은 아님을 확인했다.

### 9.3 Threshold Tuning 분석

기본 분류 기준은 예측 확률이 0.5 이상이면 블루팀 승리, 0.5 미만이면 블루팀 패배로 판단하는 방식입니다. 추가 실험에서는 validation split을 이용해 decision threshold를 조정했습니다.

10분 MLP에서는 threshold tuning으로 Accuracy가 소폭 상승했습니다.

```text
MLP 10분 baseline Accuracy: 0.7463
MLP 10분 threshold Accuracy: 0.7488
```

반면 15분 MLP에서는 Accuracy는 약간 낮아졌지만 F1-score가 개선되었습니다.

```text
MLP 15분 baseline F1-score: 0.8059
MLP 15분 threshold F1-score: 0.8099
```

이는 threshold를 조정하면 Precision과 Recall의 균형이 달라지기 때문입니다. 즉, threshold tuning은 항상 Accuracy를 높이는 방법은 아니지만, 모델이 승리/패배를 예측하는 균형을 조정하는 데 도움이 될 수 있습니다.

중요한 점은 threshold를 test set에 맞추면 데이터 누수가 된다는 것입니다. 본 프로젝트에서는 threshold를 validation 기준으로만 탐색하고, test set은 최종 평가에만 사용했습니다.

### 9.4 Soft Voting Ensemble 분석

추가 실험에서는 XGBoost와 MLP의 예측 확률을 결합한 soft voting ensemble도 적용했습니다.

```text
XGBoost 15분 baseline
- Accuracy: 0.8094
- ROC-AUC: 0.8957

Soft Voting Ensemble 15분
- Accuracy: 0.8099
- ROC-AUC: 0.8971
```

15분 데이터에서는 soft voting ensemble이 가장 높은 ROC-AUC를 기록했습니다. 이는 XGBoost와 MLP가 서로 다른 방식으로 데이터를 학습하기 때문에, 두 모델의 예측 확률을 결합했을 때 구분 성능이 약간 개선될 수 있음을 보여줍니다.

다만 10분 데이터에서는 ensemble이 단일 XGBoost보다 Accuracy가 높지 않았습니다. 따라서 ensemble은 모든 상황에서 항상 좋은 방법이라기보다, 여러 모델이 서로 보완적인 예측을 할 때 효과가 있는 방법으로 해석하는 것이 적절합니다.

### 9.5 Feature Importance 분석

XGBoost 15분 모델의 feature importance에서 중요한 피처는 다음과 같은 팀 간 격차 지표였습니다.

```text
diff_totalGolds
diff_avgLevel
diff_totalLevel
diff_dragon
diff_objectiveScore
diff_kill
```

이 피처들은 실제 LoL 경기에서 승패와 직관적으로 연결되는 요소입니다. 골드 차이와 레벨 차이는 팀의 성장 격차를 나타내고, 드래곤/타워/전령과 같은 오브젝트 차이는 맵 주도권을 나타냅니다.

따라서 모델은 단순히 무작위 패턴을 학습한 것이 아니라, 실제 경기 흐름에서 중요한 요소를 활용해 승패를 예측했다고 볼 수 있습니다.

### 9.6 최종 결론

본 프로젝트에서는 LoL 챌린저 랭크 경기의 10분/15분 데이터를 이용해 블루팀의 최종 승패를 예측했습니다. 10분 데이터만으로도 약 75% 정확도를 달성했으며, 15분 데이터에서는 약 81%까지 성능이 향상되었습니다.

단일 모델 기준으로는 XGBoost 15분 모델이 Accuracy 0.8094, ROC-AUC 0.8957로 가장 안정적인 성능을 보였습니다. 추가 실험에서는 XGBoost와 MLP를 결합한 Soft Voting Ensemble이 15분 데이터에서 ROC-AUC 0.8971로 가장 높은 구분 성능을 기록했습니다.

최종적으로, 보고서와 발표에서는 **XGBoost 15분 모델을 메인 모델**로 사용하고, **Soft Voting Ensemble을 추가 개선 실험**으로 소개하는 것이 가장 적절합니다. XGBoost는 feature importance를 통해 해석이 가능하고, ensemble은 추가적인 성능 개선 시도를 보여줄 수 있기 때문입니다.

---

## 10. 평가 지표

이진 분류 문제이므로 다음 지표를 사용합니다.

| 지표 | 의미 |
|---|---|
| Accuracy | 전체 예측 중 맞춘 비율 |
| Precision | 승리라고 예측한 것 중 실제 승리 비율 |
| Recall | 실제 승리 경기 중 맞춘 비율 |
| F1-score | Precision과 Recall의 조화 평균 |
| ROC-AUC | threshold 변화 전체에서의 분류 성능 |
| Confusion Matrix | TP, TN, FP, FN 개수 확인 |

---

## 11. 발표용 핵심 해석 문장

### XGBoost

XGBoost는 tabular data에서 강한 성능을 보이는 트리 기반 앙상블 모델이다. 본 프로젝트에서는 골드 차이, 레벨 차이, 킬/어시스트 차이, 오브젝트 차이와 같은 팀 간 격차 피처를 잘 활용할 수 있으며, feature importance를 통해 승패 예측에 중요한 지표를 해석할 수 있다.

### MLP

MLP는 여러 피처 간의 비선형 관계를 학습할 수 있는 신경망 모델이다. 다만 이 데이터는 이미지나 텍스트가 아닌 tabular data이므로, 복잡한 딥러닝 모델이 항상 XGBoost보다 우수하다고 볼 수는 없다. 따라서 본 프로젝트에서는 MLP와 XGBoost를 비교하여 tabular game data에서 어떤 모델이 더 적합한지 확인한다.

### 결론 문장

본 실험에서는 10분 데이터보다 15분 데이터의 성능이 일관되게 높았고, 단일 모델 기준으로는 XGBoost 15분 모델이 Accuracy 0.8094, ROC-AUC 0.8957로 가장 안정적인 성능을 보였다. 추가 실험에서는 XGBoost와 MLP를 결합한 Soft Voting Ensemble이 15분 데이터에서 ROC-AUC 0.8971을 기록해 가장 높은 구분 성능을 보였다. 이는 초반 골드/레벨 격차와 오브젝트 주도권이 최종 승패에 강한 예측 신호로 작용한다는 점을 보여준다.

### 10분 vs 15분 비교

15분 데이터는 10분 데이터보다 게임 진행 정보가 더 많이 포함되어 있어 일반적으로 예측 성능이 높을 가능성이 크다. 특히 골드 차이, 타워 파괴, 드래곤, 리프트 헤럴드 등 오브젝트 관련 지표가 15분 시점에서 더 명확하게 벌어질 수 있다.

---

## 12. 권장 최종 보고서 구성

```text
1. Introduction
   - 프로젝트 목표
   - LoL 경기 승패 예측 문제 정의

2. Data Description
   - 10분/15분 데이터 설명
   - 타깃 분포
   - 데이터 누수 방지: redWins 제거

3. Preprocessing & Feature Engineering
   - 불필요 컬럼 제거
   - 팀 간 차이 피처 생성
   - 문자열 리스트 컬럼 인코딩
   - MLP용 StandardScaler 적용

4. Modeling
   - XGBoost 구조 및 하이퍼파라미터
   - MLP 구조 및 학습 설정

5. Results
   - 10분/15분 성능 비교
   - XGBoost vs MLP 성능 비교
   - Confusion Matrix, ROC Curve

6. Feature Importance Analysis
   - XGBoost feature importance Top 10
   - 승패 예측에 중요한 지표 해석

7. Discussion
   - 15분 데이터가 더 좋은 이유
   - XGBoost와 MLP 차이
   - 한계점 및 개선 방향
```

---

## 13. 설치 방법

가상환경을 만든 뒤 아래 명령어로 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

Jupyter Notebook 실행:

```bash
jupyter notebook
```

---

## 14. 빠른 시작

```bash
cd lol_project
pip install -r requirements.txt

python src/train_xgboost.py --data-file data/Challenger_Ranked_Games_10minute.csv --time-label 10minute --quick
python src/train_mlp.py --data-file data/Challenger_Ranked_Games_10minute.csv --time-label 10minute --quick --epochs 20
```

15분 데이터도 같은 방식으로 `--data-file`과 `--time-label`만 바꿔서 실행하면 됩니다.
