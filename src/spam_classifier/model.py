from sklearn.model_selection import train_test_split
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

import json
from sklearn.pipeline import Pipeline
import joblib
from spam_classifier.logger import get_logger
from spam_classifier.exceptions import ModelArtifactError
from spam_classifier.config import TrainConfig
from spam_classifier.preprocessing import clean_batch

logger = get_logger(__name__)


def build_pipeline(train_cfg: TrainConfig) -> Pipeline:
    """Returns complete pipeline with all cleaning and preprocessing done"""

    return Pipeline([("clean",FunctionTransformer(clean_batch)),("tfidf", TfidfVectorizer(stop_words="english",max_features=train_cfg.max_features)),("mnb",MultinomialNB(alpha=train_cfg.alpha))])

def eval_and_export(pipeline:Pipeline,x_test: pd.Series, y_test: pd.Series, metrics_path:Path, cm_path:Path,error_analysis_path:Path) -> dict[str, Any]:
    predicted_y = pipeline.predict(x_test)
    spam_prob = pipeline.predict_proba(x_test)[:,1]

    report_text = classification_report(y_test,predicted_y,target_names=["ham","spam"])
    report_dict = classification_report(y_test,predicted_y,target_names=["ham","spam"],output_dict=True)

    logger.info("Classification Report: \n%s",report_text)

    cm = confusion_matrix(y_test,predicted_y)
    metrics: dict[str, Any] = {
        "roc_auc": round(float(roc_auc_score(y_test, spam_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, spam_prob)), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # Save misclassified rows for error analysis
    results_df = pd.DataFrame(
        {
            "text": x_test.values,
            "actual_y": y_test.values,
            "predicted_y": predicted_y,
            "spam_probability": spam_prob.round(4),
        }
    )
    errors_df = results_df.loc[results_df["actual_y"] != results_df["predicted_y"]].copy()
    error_analysis_path.parent.mkdir(parents=True, exist_ok=True)
    errors_df.to_csv(error_analysis_path, index=False)
    logger.info("Saved %d misclassified samples to %s", len(errors_df), error_analysis_path)

    return metrics


def save_model(pipeline: Pipeline, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Saved pipeline artifact to %s", model_path)


def load_model(model_path: Path) -> Pipeline:
    if not model_path.exists():
        raise ModelArtifactError(f"Model artifact not found at: {model_path}")
    return joblib.load(model_path)