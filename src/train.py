from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Crop_recommendation.csv"
MODEL_PATH = ROOT / "models" / "crop_recommendation_pipeline.joblib"
REPORT_DIR = ROOT / "reports"

FEATURES = [
    "N", "P", "K",
    "temperature", "humidity",
    "ph", "rainfall"
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download Crop_recommendation.csv from Kaggle and place it in data/."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    target = "label" if "label" in df.columns else "crop" if "crop" in df.columns else None
    if target is None:
        raise ValueError(
            "Target column not found. Expected 'label' or 'crop'. "
            f"Found: {list(df.columns)}"
        )

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    print("\nDataset shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nMissing values:\n", df[FEATURES + [target]].isnull().sum())

    df = df.dropna(subset=FEATURES + [target])

    X = df[FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Scaling is included to satisfy the lab preprocessing workflow.
    # Random Forest itself does not require feature scaling.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)

    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    joblib.dump(
        {
            "pipeline": pipeline,
            "features": FEATURES,
        },
        MODEL_PATH,
    )

    with open(REPORT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Crop Recommendation - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
