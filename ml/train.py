import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (  # noqa: E402
    CV_SPLITS,
    DATASET_PATH,
    FEATURE_SETS,
    ID_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS,
    NEURAL_NET_SEEDS,
    RANDOM_STATE,
    SAMPLE_WEIGHT_MODELS,
    SKLEARN_MODELS,
    TARGET,
    TEST_SIZE,
)
from ml.feature_selection import build_selector  # noqa: E402
from ml.metrics import classification_metrics, plot_confusion_matrix, plot_roc_curves  # noqa: E402
from ml.models import (  # noqa: E402
    build_early_stopping,
    build_model,
    build_neural_net,
    compute_balanced_sample_weight,
)
from ml.preprocessing import build_preprocessor  # noqa: E402

ARTIFACTS_DIR = DATASET_PATH.parent / "artifacts"

NEURAL_NET_MAX_EPOCHS = 100
NEURAL_NET_BATCH_SIZE = 32


def build_transform_pipeline(feature_set: str) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("selector", build_selector(feature_set)),
        ]
    )


def build_pipeline(feature_set: str, model_name: str, model_params: dict | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("selector", build_selector(feature_set)),
            ("model", build_model(model_name, model_params)),
        ]
    )


def evaluate_pipeline(pipeline: Pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    return y_pred, y_proba, classification_metrics(y_test, y_pred, y_proba)


def loggable_params(params: dict) -> dict:
    return {
        f"model__{key}": value
        for key, value in params.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def aggregate_metrics(metrics_list: list[dict], prefix: str) -> dict:
    aggregated = {}
    for key in metrics_list[0].keys():
        values = np.array([m[key] for m in metrics_list], dtype=float)
        aggregated[f"{prefix}_{key}_mean"] = float(np.nanmean(values))
        aggregated[f"{prefix}_{key}_std"] = float(np.nanstd(values))
    return aggregated


def run_sklearn_combo(feature_set, model_name, X_train, y_train, X_test, y_test, run_name):
    logger.info(f"[{run_name}] Validation croisée ({CV_SPLITS} folds)...")
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    use_sw = model_name in SAMPLE_WEIGHT_MODELS

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        pipeline = build_pipeline(feature_set, model_name)
        fit_kwargs = {}
        if use_sw:
            fit_kwargs["model__sample_weight"] = compute_balanced_sample_weight(y_train.iloc[train_idx])
        pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], **fit_kwargs)

        y_val = y_train.iloc[val_idx]
        y_pred = pipeline.predict(X_train.iloc[val_idx])
        y_proba = pipeline.predict_proba(X_train.iloc[val_idx])

        metrics = classification_metrics(y_val, y_pred, y_proba)
        fold_metrics.append(metrics)
        logger.info(f"[{run_name}] Fold {fold}/{CV_SPLITS} - f1_macro={metrics['f1_macro']:.4f}")

    cv_summary = aggregate_metrics(fold_metrics, prefix="cv")

    logger.info(f"[{run_name}] Entraînement final sur le train complet...")
    pipeline = build_pipeline(feature_set, model_name)
    fit_kwargs = {}
    if use_sw:
        fit_kwargs["model__sample_weight"] = compute_balanced_sample_weight(y_train)
    pipeline.fit(X_train, y_train, **fit_kwargs)

    y_pred, y_proba, test_metrics = evaluate_pipeline(pipeline, X_test, y_test)

    model_params = loggable_params(pipeline.named_steps["model"].get_params())

    return pipeline, cv_summary, test_metrics, y_pred, y_proba, model_params


def train_neural_net_seeds(
    X_train_t, y_train_arr, X_test_t, y_test_arr, n_features, run_name, model_kwargs=None
):
    seed_metrics = []
    final_model = None
    final_y_pred = None
    final_y_proba = None

    for seed in NEURAL_NET_SEEDS:
        logger.info(f"[{run_name}] Entraînement NeuralNet (seed={seed})...")
        keras.utils.set_random_seed(seed)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_t,
            y_train_arr,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=y_train_arr,
        )

        sample_weight = compute_balanced_sample_weight(y_tr)

        model = build_neural_net(n_features, **(model_kwargs or {}))
        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            sample_weight=sample_weight,
            epochs=NEURAL_NET_MAX_EPOCHS,
            batch_size=NEURAL_NET_BATCH_SIZE,
            callbacks=[build_early_stopping()],
            verbose=0,
        )

        y_proba = model.predict(X_test_t, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
        metrics = classification_metrics(y_test_arr, y_pred, y_proba)
        seed_metrics.append(metrics)
        logger.info(
            f"[{run_name}] seed={seed} - {len(history.history['loss'])} epochs"
            f" - f1_macro={metrics['f1_macro']:.4f}"
        )

        if seed == NEURAL_NET_SEEDS[0]:
            final_model = model
            final_y_pred = y_pred
            final_y_proba = y_proba

    seed_summary = aggregate_metrics(seed_metrics, prefix="seed")
    test_metrics = seed_metrics[0]

    return final_model, seed_summary, test_metrics, final_y_pred, final_y_proba


def run_neural_net_combo(feature_set, X_train, y_train, X_test, y_test, run_name):
    logger.info(f"[{run_name}] Préprocessing + sélection de features...")
    transform_pipeline = build_transform_pipeline(feature_set)

    X_train_t = np.asarray(transform_pipeline.fit_transform(X_train, y_train), dtype=np.float32)
    X_test_t = np.asarray(transform_pipeline.transform(X_test), dtype=np.float32)
    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()

    n_features = X_train_t.shape[1]
    logger.info(f"[{run_name}] {n_features} features après sélection")

    final_model, seed_summary, test_metrics, final_y_pred, final_y_proba = train_neural_net_seeds(
        X_train_t, y_train_arr, X_test_t, y_test_arr, n_features, run_name
    )

    return transform_pipeline, final_model, seed_summary, test_metrics, final_y_pred, final_y_proba, n_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None, help="Filtrer les modèles à entraîner")
    parser.add_argument("--feature_sets", nargs="+", default=None, help="Filtrer les feature sets")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"❌ {DATASET_PATH} introuvable. Lancez d'abord : uv run python -m ml.extract_dataset"
        )

    logger.info(f"Chargement de {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    logger.info(f"{df.shape[0]} lignes, {df.shape[1]} colonnes")

    feature_columns = [c for c in df.columns if c not in ID_COLUMNS + [TARGET]]
    X = df[feature_columns]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train : {len(X_train)} lignes - Test : {len(X_test)} lignes")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    feature_sets_to_run = args.feature_sets or FEATURE_SETS
    models_to_run = args.models or MODELS

    results = []

    for feature_set in feature_sets_to_run:
        for model_name in models_to_run:
            run_name = f"{feature_set}__{model_name}"
            logger.info(f"=== {run_name} ===")

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("feature_set", feature_set)
                mlflow.log_param("model", model_name)

                if model_name in SKLEARN_MODELS:
                    pipeline, agg_summary, test_metrics, y_pred, y_proba, model_params = run_sklearn_combo(
                        feature_set, model_name, X_train, y_train, X_test, y_test, run_name
                    )
                    mlflow.log_params(model_params)
                    mlflow.sklearn.log_model(pipeline, artifact_path="model")
                else:
                    transform_pipeline, model, agg_summary, test_metrics, y_pred, y_proba, n_features = (
                        run_neural_net_combo(feature_set, X_train, y_train, X_test, y_test, run_name)
                    )
                    mlflow.log_params(
                        {
                            "n_features": n_features,
                            "epochs_max": NEURAL_NET_MAX_EPOCHS,
                            "batch_size": NEURAL_NET_BATCH_SIZE,
                            "seeds": str(NEURAL_NET_SEEDS),
                        }
                    )
                    mlflow.sklearn.log_model(transform_pipeline, artifact_path="preprocessing_pipeline")
                    mlflow.tensorflow.log_model(model, artifact_path="model")

                mlflow.log_metrics(agg_summary)
                mlflow.log_metrics(test_metrics)

                cm_path = ARTIFACTS_DIR / f"{run_name}_confusion_matrix.png"
                roc_path = ARTIFACTS_DIR / f"{run_name}_roc_curve.png"
                plot_confusion_matrix(y_test, y_pred, cm_path)
                plot_roc_curves(y_test, y_proba, roc_path)
                mlflow.log_artifact(str(cm_path))
                mlflow.log_artifact(str(roc_path))

                logger.success(
                    f"[{run_name}] test f1_macro={test_metrics['f1_macro']:.4f}"
                    f" - critical_undertriage_rate={test_metrics['critical_undertriage_rate']:.4f}"
                )

                row = {"feature_set": feature_set, "model": model_name}
                row.update(test_metrics)
                results.append(row)

    new_rows = pd.DataFrame(results)
    summary_path = ARTIFACTS_DIR / "summary.csv"

    if (args.models or args.feature_sets) and summary_path.exists():
        existing = pd.read_csv(summary_path)
        mask = existing["model"].isin(new_rows["model"]) & existing["feature_set"].isin(new_rows["feature_set"])
        existing = existing[~mask]
        summary = pd.concat([existing, new_rows], ignore_index=True)
    else:
        summary = new_rows

    summary = summary.sort_values(by=["f1_macro", "critical_undertriage_rate"], ascending=[False, True])
    logger.info("Tableau récapitulatif (test) :\n" + summary.to_string(index=False))
    summary.to_csv(summary_path, index=False)
    logger.success(f"Tableau récapitulatif sauvegardé dans {summary_path}")


if __name__ == "__main__":
    main()
