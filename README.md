# Airline Passenger Satisfaction — Streamlit app

## Project structure

```text
airline_satisfaction_streamlit/
├── Home.py
├── cleaned_data.csv
├── xgboost_model.pkl
├── requirements.txt
└── pages/
    ├── 1_Visualization.py
    └── 2_Prediction.py
```

## Important model note

The notebook saves the fitted pipeline with:

```python
joblib.dump(xgboost_pipeline, "xgboost_model.pkl")
```

Copy that exact `xgboost_model.pkl` file into this project root. The prediction page loads the complete fitted pipeline, including preprocessing. Do not encode or scale the inputs again in the Streamlit page.

The notebook uses 21 input features. `id` is excluded because the notebook reads the training CSV with `index_col=0`; `satisfaction` is the target and must not be entered as an input.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run from this folder:

```bash
streamlit run Home.py
```

Place `cleaned_data.csv` and `xgboost_model.pkl` beside `Home.py`.

## Target meaning

In the uploaded dataset, `satisfaction=True` means **satisfied** and `False` means **neutral or dissatisfied**.
