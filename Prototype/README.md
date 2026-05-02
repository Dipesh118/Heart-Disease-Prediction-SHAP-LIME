# Heart Disease Prediction System

This project is a machine learning-based clinical decision support system that predicts the risk of heart disease using patient health data.

## Features
- Logistic Regression model
- SHAP explanation (global & local)
- LIME explanation (local interpretability)
- Interactive Streamlit web application
- ROC Curve and Confusion Matrix
- Risk classification (Low, Medium, High)

## Model Performance
- Accuracy: ~0.84
- Recall: ~0.91
- AUC: ~0.93

## Technologies Used
- Python
- Scikit-learn
- Streamlit
- SHAP
- LIME

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
