# Heart Disease Prediction – Model Development Notebook

This folder contains the Jupyter Notebook used for developing the machine learning model for heart disease prediction.

The notebook demonstrates the complete workflow, from data preprocessing to model evaluation and explainability.

---

## Objective

The objective of this notebook is to:

- Perform data preprocessing and cleaning
- Train machine learning models for heart disease prediction
- Evaluate model performance using classification metrics
- Apply explainable AI techniques (SHAP and LIME)
- Identify the most suitable model for deployment

---

## Workflow

The notebook follows a structured machine learning pipeline:

### 1. Data Loading
- UCI Heart Disease dataset (downloaded via Kaggle)
- Initial data inspection

### 2. Data Preprocessing
- Removal of irrelevant features (e.g., ID column)
- Conversion of target variable (`num → target`)
- Handling missing values
- Encoding categorical variables
- Feature scaling using StandardScaler
- Train-test split (80/20 with stratification)

### 3. Exploratory Data Analysis (EDA)
- Distribution of target variable
- Feature relationships
- Statistical summaries

### 4. Model Development
The following models were implemented:

- Logistic Regression
- Random Forest
- XGBoost

### 5. Model Evaluation
Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC score
- Confusion Matrix
- ROC Curve

---

## Key Results

- **Best Model:** Logistic Regression  
- **Accuracy:** ~0.84  
- **Recall:** ~0.91  
- **AUC Score:** ~0.93  

The model demonstrates strong performance, particularly in identifying positive heart disease cases, which is critical in healthcare applications.

---

## Explainability

To improve model transparency, the following techniques were applied:

### SHAP (SHapley Additive Explanations)
- Global feature importance
- Local explanation for individual predictions

### LIME (Local Interpretable Model-agnostic Explanations)
- Instance-level interpretation
- Feature contribution analysis

These methods help explain how different features influence predictions.

---

## Outcome

The notebook establishes a reliable and interpretable machine learning model, which is later deployed as a Streamlit web application.

---

## How to Run

You can run this notebook using:

### Option 1: Google Colab
Upload the `.ipynb` file to Colab and run all cells.

### Option 2: Local Jupyter Notebook

```bash
pip install pandas numpy scikit-learn matplotlib shap lime xgboost
jupyter notebook
