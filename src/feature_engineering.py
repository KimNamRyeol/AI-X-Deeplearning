"""Feature engineering utilities for the LoL win prediction project.

The raw data contains blue-team and red-team versions of many variables.
For win prediction, the difference between teams is often more informative
than each team's absolute value. This module adds those difference features
and encodes list-like categorical columns such as dragon type and tower lane.
"""

from __future__ import annotations

import ast
from typing import Iterable

import numpy as np
import pandas as pd


# Raw blue/red metric pairs. The suffix becomes the diff feature name.
DIFF_FEATURES: list[tuple[str, str, str]] = [
    ("blueTotalGolds", "redTotalGolds", "diff_totalGolds"),
    ("blueCurrentGolds", "redCurrentGolds", "diff_currentGolds"),
    ("blueTotalLevel", "redTotalLevel", "diff_totalLevel"),
    ("blueAvgLevel", "redAvgLevel", "diff_avgLevel"),
    ("blueTotalMinionKills", "redTotalMinionKills", "diff_totalMinionKills"),
    ("blueTotalJungleMinionKills", "redTotalJungleMinionKills", "diff_totalJungleMinionKills"),
    ("blueFirstBlood", "redFirstBlood", "diff_firstBlood"),
    ("blueKill", "redKill", "diff_kill"),
    ("blueDeath", "redDeath", "diff_death"),
    ("blueAssist", "redAssist", "diff_assist"),
    ("blueWardPlaced", "redWardPlaced", "diff_wardPlaced"),
    ("blueWardKills", "redWardKills", "diff_wardKills"),
    ("blueFirstTower", "redFirstTower", "diff_firstTower"),
    ("blueFirstInhibitor", "redFirstInhibitor", "diff_firstInhibitor"),
    ("blueTowerKills", "redTowerKills", "diff_towerKills"),
    ("blueMidTowerKills", "redMidTowerKills", "diff_midTowerKills"),
    ("blueTopTowerKills", "redTopTowerKills", "diff_topTowerKills"),
    ("blueBotTowerKills", "redBotTowerKills", "diff_botTowerKills"),
    ("blueInhibitor", "redInhibitor", "diff_inhibitor"),
    ("blueFirstDragon", "redFirstDragon", "diff_firstDragon"),
    ("blueDragon", "redDragon", "diff_dragon"),
    ("blueRiftHeralds", "redRiftHeralds", "diff_riftHeralds"),
]

LIST_LIKE_COLUMNS = [
    "blueFirstTowerLane",
    "redFirstTowerLane",
    "blueDragnoType",  # original column has this typo in the dataset
    "redDragnoType",
]


def parse_list_cell(value) -> list[str]:
    """Parse a cell that looks like "['MID_LANE']" or "[]" into a Python list.

    Invalid or missing values are treated as empty lists so the pipeline does
    not break during modeling.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text == "[]":
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
            return [str(parsed)]
        except (ValueError, SyntaxError):
            return [text]
    return [str(value)]


def add_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add blue-minus-red difference features."""
    out = df.copy()
    for blue_col, red_col, new_col in DIFF_FEATURES:
        if blue_col in out.columns and red_col in out.columns:
            out[new_col] = out[blue_col] - out[red_col]
    return out


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few compact ratio/KDA features.

    These are intentionally simple and interpretable for presentation.
    """
    out = df.copy()
    if {"blueKill", "blueAssist", "blueDeath", "redKill", "redAssist", "redDeath"}.issubset(out.columns):
        out["blueKDA"] = (out["blueKill"] + out["blueAssist"]) / (out["blueDeath"] + 1)
        out["redKDA"] = (out["redKill"] + out["redAssist"]) / (out["redDeath"] + 1)
        out["diff_KDA"] = out["blueKDA"] - out["redKDA"]
    if {"blueTotalGolds", "blueTotalMinionKills", "redTotalGolds", "redTotalMinionKills"}.issubset(out.columns):
        out["blueGoldPerMinion"] = out["blueTotalGolds"] / (out["blueTotalMinionKills"] + 1)
        out["redGoldPerMinion"] = out["redTotalGolds"] / (out["redTotalMinionKills"] + 1)
        out["diff_goldPerMinion"] = out["blueGoldPerMinion"] - out["redGoldPerMinion"]
    if {"blueTowerKills", "blueDragon", "blueRiftHeralds", "redTowerKills", "redDragon", "redRiftHeralds"}.issubset(out.columns):
        out["blueObjectiveScore"] = out["blueTowerKills"] + out["blueDragon"] + out["blueRiftHeralds"]
        out["redObjectiveScore"] = out["redTowerKills"] + out["redDragon"] + out["redRiftHeralds"]
        out["diff_objectiveScore"] = out["blueObjectiveScore"] - out["redObjectiveScore"]
    return out


def expand_list_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert list-like categorical columns into binary indicator columns."""
    out = df.copy()
    for col in LIST_LIKE_COLUMNS:
        if col not in out.columns:
            continue
        parsed = out[col].apply(parse_list_cell)
        unique_values = sorted({item for items in parsed for item in items})
        safe_prefix = col.replace("Dragno", "Dragon")
        for value in unique_values:
            safe_value = str(value).replace(" ", "_").replace("'", "").replace('"', "")
            out[f"{safe_prefix}_{safe_value}"] = parsed.apply(lambda items, v=value: int(v in items))
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all project feature engineering steps."""
    out = df.copy()
    out = add_difference_features(out)
    out = add_ratio_features(out)
    out = expand_list_like_columns(out)
    return out
