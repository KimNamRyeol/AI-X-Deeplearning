# XGBoost / MLP Results Notes

## Key Results

- Best setting: SoftVotingEnsemble with 15minute data.
- Best test accuracy: 80.99%.
- Best ROC-AUC: 0.8971.
- Accuracy improves when moving from 10-minute to 15-minute data: {'MLP': 0.06103294234960066, 'XGBoost': 0.06043199331858695}.

## Interpretation

The result supports the project hypothesis that early-game information can predict the final winner better than chance. The 15-minute data consistently performs better than the 10-minute data, which means the additional five minutes contain meaningful signals such as objective control, accumulated gold gap, and level gap.

XGBoost is slightly stronger than MLP in this experiment. This is reasonable because the dataset is structured tabular data, where tree-based ensemble models often learn feature thresholds and interactions efficiently. MLP still performs competitively after standardization and dropout, but it does not clearly outperform XGBoost.

For feature importance, the strongest 15-minute XGBoost signals are: diff_totalGolds, diff_avgLevel, diff_totalLevel, blueObjectiveScore, diff_dragon. These features are interpretable in game terms: gold and level gaps summarize lane and jungle advantages, while dragon/objective variables represent map control.

## Presentation Takeaway

Even though the topic is game-related, the project is framed as a binary classification and feature-importance analysis task. The important academic point is not the game itself, but the full machine-learning workflow: leakage prevention, feature engineering, model comparison, threshold-based and threshold-free evaluation, and interpretation of predictive factors.
