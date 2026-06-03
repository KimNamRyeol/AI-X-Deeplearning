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
│   └── 03_mlp.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_xgboost.py
│   └── train_mlp.py
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

이 네 파일의 `accuracy`, `f1`, `roc_auc`를 모아 최종 발표 표를 만들면 됩니다.
압축파일에는 예시 실행 결과로 `results/tables/model_comparison.csv`도 함께 포함되어 있습니다.

---

## 7. 평가 지표

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

## 8. 발표용 핵심 해석 문장

### XGBoost

XGBoost는 tabular data에서 강한 성능을 보이는 트리 기반 앙상블 모델이다. 본 프로젝트에서는 골드 차이, 레벨 차이, 킬/어시스트 차이, 오브젝트 차이와 같은 팀 간 격차 피처를 잘 활용할 수 있으며, feature importance를 통해 승패 예측에 중요한 지표를 해석할 수 있다.

### MLP

MLP는 여러 피처 간의 비선형 관계를 학습할 수 있는 신경망 모델이다. 다만 이 데이터는 이미지나 텍스트가 아닌 tabular data이므로, 복잡한 딥러닝 모델이 항상 XGBoost보다 우수하다고 볼 수는 없다. 따라서 본 프로젝트에서는 MLP와 XGBoost를 비교하여 tabular game data에서 어떤 모델이 더 적합한지 확인한다.

### 10분 vs 15분 비교

15분 데이터는 10분 데이터보다 게임 진행 정보가 더 많이 포함되어 있어 일반적으로 예측 성능이 높을 가능성이 크다. 특히 골드 차이, 타워 파괴, 드래곤, 리프트 헤럴드 등 오브젝트 관련 지표가 15분 시점에서 더 명확하게 벌어질 수 있다.

---

## 9. 권장 최종 보고서 구성

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

## 10. 설치 방법

가상환경을 만든 뒤 아래 명령어로 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

Jupyter Notebook 실행:

```bash
jupyter notebook
```

---

## 11. 빠른 시작

```bash
cd lol_project
pip install -r requirements.txt

python src/train_xgboost.py --data-file data/Challenger_Ranked_Games_10minute.csv --time-label 10minute --quick
python src/train_mlp.py --data-file data/Challenger_Ranked_Games_10minute.csv --time-label 10minute --quick --epochs 20
```

15분 데이터도 같은 방식으로 `--data-file`과 `--time-label`만 바꿔서 실행하면 됩니다.
