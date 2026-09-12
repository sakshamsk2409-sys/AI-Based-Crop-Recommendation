# 🌾 Agri-AI: Crop Recommendation System

An AI-based crop recommendation system built for the AISA Lab Assignment 06.

The system predicts the most suitable crop from:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

## Dataset

Kaggle: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

The dataset contains the seven input features above and a crop label.

## Model

The main model is **Random Forest Classifier**, matching the lab write-up's Module A architecture.

Pipeline:

`Agricultural Inputs → Preprocessing → Random Forest → Crop Prediction → Evaluation`

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Project Structure

```text
agri-ai-crop-recommendation/
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── notebooks/
├── reports/
├── src/
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/agri-ai-crop-recommendation.git
cd agri-ai-crop-recommendation
```

## 2. Create virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Download the dataset

Download `Crop_recommendation.csv` from Kaggle and place it here:

```text
data/Crop_recommendation.csv
```

Do **not** upload the dataset to GitHub if you want to keep the repository lightweight. The repository contains instructions for obtaining it.

## 5. Train the model

```bash
python src/train.py
```

This creates:

```text
models/crop_recommendation_pipeline.joblib
reports/metrics.txt
reports/confusion_matrix.png
```

## 6. Test a prediction from terminal

```bash
python src/predict.py --N 90 --P 42 --K 43 --temperature 25 --humidity 80 --ph 6.5 --rainfall 200
```

## 7. Run the web application

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit.

## Expected Output

For suitable agricultural conditions, the application returns a recommended crop such as:

```text
Recommended Crop: rice
```



## Important

This is an educational ML project. A crop recommendation should not be treated as a standalone agronomic prescription; real deployment would require local soil tests, weather data, crop economics, and agronomist validation.
