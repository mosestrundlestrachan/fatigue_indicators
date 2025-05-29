from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE #for the imbalance in fatigued moments to non fatigued moments
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def train_fatigue_classifier(df: pd.DataFrame, show_plot: bool = True, threshold: float = 0.5) -> XGBClassifier:
    feature_cols = [
        "early_ff_velocity",
        "early_ff_spin",
        "early_strike_ratio",
        "early_pitch_count"
    ]

    X = df[feature_cols]
    y = df["fatigued"]

    # Stratified split for class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply SMOTE to balance training data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        n_jobs=1,
        verbosity=1,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train_balanced, y_train_balanced)

    # Predict probabilities and apply custom threshold
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > threshold).astype(int)

    print("=== Classifier Evaluation ===")
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if show_plot:
        plot_importance(model)
        plt.title("Feature Importance (Fatigue Classifier)")
        plt.tight_layout()
        plt.savefig("classifier_feature_importance.png")
        print("📊 Saved feature importance plot as 'classifier_feature_importance.png'")

    return model
