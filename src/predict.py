from pathlib import Path
import argparse
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "crop_recommendation_pipeline.joblib"

FEATURES = [
    "N", "P", "K",
    "temperature", "humidity",
    "ph", "rainfall"
]


def main():
    parser = argparse.ArgumentParser(description="Predict the best crop.")
    for feature in FEATURES:
        parser.add_argument(f"--{feature}", type=float, required=True)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python src/train.py"
        )

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]

    row = pd.DataFrame([{
        feature: getattr(args, feature)
        for feature in FEATURES
    }])

    prediction = pipeline.predict(row)[0]
    probabilities = pipeline.predict_proba(row)[0]
    classes = pipeline.named_steps["model"].classes_

    top_indices = probabilities.argsort()[::-1][:3]

    print(f"\nRecommended Crop: {prediction}")
    print("\nTop 3 predictions:")
    for idx in top_indices:
        print(f"{classes[idx]}: {probabilities[idx] * 100:.2f}%")


if __name__ == "__main__":
    main()
