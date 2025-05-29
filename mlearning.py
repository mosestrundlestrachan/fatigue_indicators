import pandas as pd
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def train_fatigue_model(df: pd.DataFrame, show_plot: bool = True) -> XGBClassifier:
    """
    Trains an XGBoost classifier on the given fatigue DataFrame.

    Parameters:
    - df: A DataFrame with features and 'fatigued' column as target
    - show_plot: If True, displays feature importance plot

    Returns:
    - Trained XGBClassifier
    """
    feature_cols = [
        "early_ff_velocity",
        "early_ff_spin",
        "early_strike_ratio",
        "early_pitch_count"
    ]
    
    X = df[feature_cols].dropna()
    y = df.loc[X.index, "fatigued"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    print("=== Evaluation Results ===")
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    if show_plot:
        plot_importance(model)
        plt.title("Feature Importance for Fatigue Prediction")
        plt.tight_layout()
        plt.show()

    return model