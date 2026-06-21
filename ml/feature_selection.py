import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif

from ml.config import RANDOM_STATE, RF_IMPORTANCE_K, SELECT_KBEST_K


def build_selector(feature_set: str):
    if feature_set == "ALL_FEATURES":
        return "passthrough"

    if feature_set == f"SELECT_KBEST_{SELECT_KBEST_K}":
        return SelectKBest(mutual_info_classif, k=SELECT_KBEST_K)

    if feature_set == f"RF_IMPORTANCE_{RF_IMPORTANCE_K}":
        return SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            max_features=RF_IMPORTANCE_K,
            threshold=-np.inf,
        )

    raise ValueError(f"Feature set inconnu : {feature_set}")
