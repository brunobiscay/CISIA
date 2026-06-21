import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from ml.config import CLASS_LABELS


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mae_ordinal": mean_absolute_error(y_true, y_pred),
        "undertriage_rate": float(np.mean(y_pred < y_true)),
    }

    critical_mask = y_true == max(CLASS_LABELS)
    if critical_mask.sum() > 0:
        metrics["critical_undertriage_rate"] = float(
            np.mean(y_pred[critical_mask] < y_true[critical_mask])
        )
    else:
        metrics["critical_undertriage_rate"] = float("nan")

    if y_proba is not None:
        try:
            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro", labels=CLASS_LABELS
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = float("nan")

    return metrics


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice de confusion")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_curves(y_true, y_proba, out_path):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_true_bin = label_binarize(y_true, classes=CLASS_LABELS)

    fig, ax = plt.subplots(figsize=(6, 6))

    for i, label in enumerate(CLASS_LABELS):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"Classe {label} (AUC = {auc:.3f})")

    fpr_macro, tpr_macro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    auc_macro = roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="macro")
    ax.plot(fpr_macro, tpr_macro, linestyle="--", color="black", label=f"Macro (AUC = {auc_macro:.3f})")

    ax.plot([0, 1], [0, 1], linestyle=":", color="grey")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbes ROC (one-vs-rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
