"""
Generate SHAP feature-importance plots for the trained rain classifier.

Usage:
    python -m src.evaluation.explain
"""
import joblib
import matplotlib.pyplot as plt
import shap

from src.training.train_classifier import get_feature_matrix
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.splitting import time_based_split

log = get_logger(__name__)


def run():
    cfg = load_config()
    clf_dir = resolve_path("models/classification")
    model = joblib.load(clf_dir / "rain_classifier.joblib")
    feature_cols = joblib.load(clf_dir / "feature_columns.joblib")

    df = get_feature_matrix(cfg)
    _, _, test_df = time_based_split(df, cfg["training"]["test_size_days"], cfg["training"]["val_size_days"])
    X_sample = test_df[feature_cols].sample(min(500, len(test_df)), random_state=cfg["random_seed"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig_dir = resolve_path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(vals, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary_rain_classifier.png", dpi=150)
    log.info(f"SHAP summary plot saved to {fig_dir / 'shap_summary_rain_classifier.png'}")


if __name__ == "__main__":
    run()
