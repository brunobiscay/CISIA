import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (  # noqa: E402
    DATASET_PATH,
    ID_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    NEURAL_NET_MODEL,
    OPTUNA_N_TRIALS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    TUNE_FEATURE_SET,
    TUNE_MODELS,
)
from ml.metrics import classification_metrics, plot_confusion_matrix, plot_roc_curves  # noqa: E402
from ml.models import (  # noqa: E402
    build_early_stopping,
    build_model,
    build_neural_net,
    compute_balanced_sample_weight,
)
from ml.train import (  # noqa: E402
    ARTIFACTS_DIR,
    build_pipeline,
    build_transform_pipeline,
    evaluate_pipeline,
    train_neural_net_seeds,
)

TUNE_NN_MAX_EPOCHS = 60
TUNE_NN_PATIENCE = 8

optuna.logging.set_verbosity(optuna.logging.WARNING)


def log_trial(study, trial):
    logger.info(f"  trial {trial.number}: score={trial.value:.4f} - params={trial.params}")


def sklearn_search_space(trial, model_name):
    if model_name == "RandomForestClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20, 30]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if model_name == "LGBMClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    raise ValueError(f"Modèle inconnu : {model_name}")


def objective_sklearn(trial, model_name, X_tr_t, y_tr, X_val_t, y_val):
    params = sklearn_search_space(trial, model_name)
    model = build_model(model_name, params)
    model.fit(X_tr_t, y_tr)
    y_pred = model.predict(X_val_t)
    metrics = classification_metrics(y_val, y_pred)
    return metrics["f1_macro"] - metrics["critical_undertriage_rate"]


def objective_neural_net(trial, X_tr_t, y_tr, X_val_t, y_val, n_features):
    units1 = trial.suggest_int("units1", 32, 256, step=32)
    units2 = trial.suggest_int("units2", 16, 128, step=16)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    model = build_neural_net(
        n_features, units=(units1, units2), dropout=dropout, learning_rate=learning_rate
    )
    sample_weight = compute_balanced_sample_weight(y_tr)
    model.fit(
        X_tr_t,
        y_tr,
        validation_data=(X_val_t, y_val),
        sample_weight=sample_weight,
        epochs=TUNE_NN_MAX_EPOCHS,
        batch_size=32,
        callbacks=[build_early_stopping(patience=TUNE_NN_PATIENCE)],
        verbose=0,
    )

    y_proba = model.predict(X_val_t, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = classification_metrics(y_val, y_pred, y_proba)
    return metrics["f1_macro"] - metrics["critical_undertriage_rate"]


def run_study(objective, n_trials, label):
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    logger.info(f"[{label}] Lancement de {n_trials} essais Optuna...")
    study.optimize(objective, n_trials=n_trials, callbacks=[log_trial])
    logger.success(f"[{label}] meilleur score (val) = {study.best_value:.4f} - params={study.best_params}")
    return study.best_params, study.best_value


def main():
    logger.info(f"Chargement de {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)

    feature_columns = [c for c in df.columns if c not in ID_COLUMNS + [TARGET]]
    X = df[feature_columns]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )

    logger.info(f"[{TUNE_FEATURE_SET}] Préprocessing + sélection de features (fit sur X_tr)...")
    search_transform = build_transform_pipeline(TUNE_FEATURE_SET)
    X_tr_t = np.asarray(search_transform.fit_transform(X_tr, y_tr), dtype=np.float32)
    X_val_t = np.asarray(search_transform.transform(X_val), dtype=np.float32)
    y_tr_arr = y_tr.to_numpy()
    y_val_arr = y_val.to_numpy()
    n_features = X_tr_t.shape[1]
    logger.info(f"[{TUNE_FEATURE_SET}] {n_features} features après sélection")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    tuned_rows = []

    for model_name in TUNE_MODELS:
        run_name = f"{TUNE_FEATURE_SET}__{model_name}__tuned"
        logger.info(f"=== {run_name} ===")

        if model_name == NEURAL_NET_MODEL:
            best_params, best_value = run_study(
                lambda trial: objective_neural_net(trial, X_tr_t, y_tr_arr, X_val_t, y_val_arr, n_features),
                OPTUNA_N_TRIALS[model_name],
                run_name,
            )
        else:
            best_params, best_value = run_study(
                lambda trial: objective_sklearn(trial, model_name, X_tr_t, y_tr_arr, X_val_t, y_val_arr),
                OPTUNA_N_TRIALS[model_name],
                run_name,
            )

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("feature_set", TUNE_FEATURE_SET)
            mlflow.log_param("model", model_name)
            mlflow.log_param("tuned", True)
            mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})
            mlflow.log_metric("optuna_best_val_score", best_value)

            if model_name == NEURAL_NET_MODEL:
                logger.info(f"[{run_name}] Entraînement final (3 seeds) avec les meilleurs hyperparams...")
                transform_pipeline = build_transform_pipeline(TUNE_FEATURE_SET)
                X_train_t = np.asarray(transform_pipeline.fit_transform(X_train, y_train), dtype=np.float32)
                X_test_t = np.asarray(transform_pipeline.transform(X_test), dtype=np.float32)

                model_kwargs = {
                    "units": (best_params["units1"], best_params["units2"]),
                    "dropout": best_params["dropout"],
                    "learning_rate": best_params["learning_rate"],
                }
                model, seed_summary, test_metrics, y_pred, y_proba = train_neural_net_seeds(
                    X_train_t,
                    y_train.to_numpy(),
                    X_test_t,
                    y_test.to_numpy(),
                    X_train_t.shape[1],
                    run_name,
                    model_kwargs=model_kwargs,
                )
                mlflow.log_metrics(seed_summary)
                mlflow.sklearn.log_model(transform_pipeline, artifact_path="preprocessing_pipeline")
                mlflow.tensorflow.log_model(model, artifact_path="model")
            else:
                logger.info(f"[{run_name}] Entraînement final sur le train complet avec les meilleurs hyperparams...")
                pipeline = build_pipeline(TUNE_FEATURE_SET, model_name, model_params=best_params)
                pipeline.fit(X_train, y_train)
                y_pred, y_proba, test_metrics = evaluate_pipeline(pipeline, X_test, y_test)
                mlflow.sklearn.log_model(pipeline, artifact_path="model")

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

            row = {"feature_set": TUNE_FEATURE_SET, "model": f"{model_name}__tuned"}
            row.update(test_metrics)
            tuned_rows.append(row)

    summary_path = ARTIFACTS_DIR / "summary.csv"
    baseline = pd.read_csv(summary_path)
    baseline_subset = baseline[
        (baseline["feature_set"] == TUNE_FEATURE_SET) & (baseline["model"].isin(TUNE_MODELS))
    ].copy()

    comparison = pd.concat([baseline_subset, pd.DataFrame(tuned_rows)], ignore_index=True)
    comparison["base_model"] = comparison["model"].str.replace("__tuned", "", regex=False)
    comparison = comparison.sort_values(by=["base_model", "model"]).drop(columns=["base_model"])

    logger.info("Comparaison baseline vs tuned (test) :\n" + comparison.to_string(index=False))

    comparison_path = ARTIFACTS_DIR / "tuning_summary.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.success(f"Comparaison sauvegardée dans {comparison_path}")


if __name__ == "__main__":
    main()
