import numpy as np
from lightgbm import LGBMClassifier
from mord import LogisticAT
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow import keras

from ml.config import RANDOM_STATE

MODEL_REGISTRY = {
    "LogisticRegression": lambda: LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "RandomForestClassifier": lambda: RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "LGBMClassifier": lambda: LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "OrdinalLogistic": lambda: LogisticAT(alpha=1.0, max_iter=10000),
}


def build_model(model_name: str, params: dict | None = None):
    model = MODEL_REGISTRY[model_name]()
    if params:
        model.set_params(**params)
    return model


def compute_balanced_sample_weight(y) -> np.ndarray:
    return compute_sample_weight(class_weight="balanced", y=y)


def build_neural_net(
    n_features: int,
    units: tuple[int, int] = (128, 64),
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(units[0], activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(units[1], activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_early_stopping(patience: int = 10) -> keras.callbacks.EarlyStopping:
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
