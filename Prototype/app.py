import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# PAGE CONFIG

st.set_page_config(
    page_title="Heart Disease Prediction System",
    layout="wide"
)

# CUSTOM STYLE

st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 700;
    color: #b30000;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 18px;
    color: #444444;
    margin-bottom: 1.2rem;
}
.section-box {
    background-color: #f7f9fc;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e6eaf0;
    margin-bottom: 16px;
}
.result-high {
    background-color: #ffe6e6;
    padding: 20px;
    border-radius: 12px;
    border-left: 8px solid #d62728;
    font-size: 18px;
    font-weight: 600;
}
.result-medium {
    background-color: #fff4e5;
    padding: 20px;
    border-radius: 12px;
    border-left: 8px solid #ff9800;
    font-size: 18px;
    font-weight: 600;
}
.result-low {
    background-color: #eaf7ea;
    padding: 20px;
    border-radius: 12px;
    border-left: 8px solid #2ca02c;
    font-size: 18px;
    font-weight: 600;
}
.metric-card {
    background-color: #ffffff;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e6eaf0;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.small-note {
    color: #666666;
    font-size: 14px;
}
.sample-btn-text {
    font-size: 14px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)


# SESSION STATE DEFAULTS

if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded = False


# LOAD DATA

@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_uci.csv")
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1)
    df = df.drop(columns=["num"])
    df = df.drop_duplicates()
    return df

df = load_data()

X = df.drop(columns=["target"])
y = df["target"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()


# PREPROCESSING
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# TRAIN MODEL

@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return model, X_train, X_test, y_train, y_test, accuracy, auc, cm, fpr, tpr

model, X_train, X_test, y_train, y_test, accuracy, auc, cm, fpr, tpr = train_model()


# HEADER

st.markdown('<div class="main-title">Heart Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A prototype clinical decision support system using Logistic Regression, SHAP, and LIME.</div>',
    unsafe_allow_html=True
)

# DASHBOARD METRICS

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Dataset Size</h4>
        <h2>{len(df)}</h2>
        <p class="small-note">Patient records</p>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Model Accuracy</h4>
        <h2>{accuracy:.3f}</h2>
        <p class="small-note">Test set performance</p>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>AUC Score</h4>
        <h2>{auc:.3f}</h2>
        <p class="small-note">Discrimination ability</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# DASHBOARD CHARTS

c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    ax_roc.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic")
    ax_roc.legend()
    ax_roc.grid(True)
    st.pyplot(fig_roc, clear_figure=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax_cm, colorbar=False)
    ax_cm.set_title("Test Set Confusion Matrix")
    st.pyplot(fig_cm, clear_figure=True)
    st.markdown('</div>', unsafe_allow_html=True)


# DATA PREVIEW
with st.expander("View cleaned dataset preview"):
    st.dataframe(df.head())

# OPTIONS
sex_options = sorted(df["sex"].dropna().unique().tolist())
dataset_options = sorted(df["dataset"].dropna().unique().tolist())
cp_options = sorted(df["cp"].dropna().unique().tolist())
restecg_options = sorted(df["restecg"].dropna().unique().tolist())
slope_options = sorted(df["slope"].dropna().unique().tolist())
thal_options = sorted(df["thal"].dropna().unique().tolist())


# SAMPLE PATIENTS
sample_patients = {
    "Low-risk sample": {
        "age": 40,
        "sex": sex_options[0],
        "dataset": dataset_options[0],
        "cp": cp_options[0],
        "trestbps": 120.0,
        "chol": 200.0,
        "fbs": False,
        "restecg": restecg_options[0],
        "thalch": 170.0,
        "exang": False,
        "oldpeak": 0.2,
        "slope": slope_options[0],
        "ca": 0.0,
        "thal": thal_options[0]
    },
    "Medium-risk sample": {
        "age": 55,
        "sex": sex_options[0],
        "dataset": dataset_options[0],
        "cp": cp_options[min(1, len(cp_options)-1)],
        "trestbps": 140.0,
        "chol": 250.0,
        "fbs": False,
        "restecg": restecg_options[0],
        "thalch": 145.0,
        "exang": True,
        "oldpeak": 1.5,
        "slope": slope_options[0],
        "ca": 1.0,
        "thal": thal_options[0]
    },
    "High-risk sample": {
        "age": 63,
        "sex": sex_options[0],
        "dataset": dataset_options[0],
        "cp": cp_options[-1],
        "trestbps": 150.0,
        "chol": 290.0,
        "fbs": True,
        "restecg": restecg_options[0],
        "thalch": 120.0,
        "exang": True,
        "oldpeak": 2.8,
        "slope": slope_options[-1],
        "ca": 2.0,
        "thal": thal_options[-1]
    }
}

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Quick Demo Sample Patients")
sb1, sb2, sb3 = st.columns(3)

with sb1:
    if st.button("Load Low-risk Sample"):
        for k, v in sample_patients["Low-risk sample"].items():
            st.session_state[k] = v

with sb2:
    if st.button("Load Medium-risk Sample"):
        for k, v in sample_patients["Medium-risk sample"].items():
            st.session_state[k] = v

with sb3:
    if st.button("Load High-risk Sample"):
        for k, v in sample_patients["High-risk sample"].items():
            st.session_state[k] = v

st.caption("These buttons auto-fill the form for quick supervisor/demo presentation.")
st.markdown('</div>', unsafe_allow_html=True)


# HELPER FOR DEFAULT VALUES
def get_default(key, fallback):
    return st.session_state[key] if key in st.session_state else fallback

# INPUT FORM
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Patient Clinical Information")

with st.form("prediction_form"):
    c1, c2 = st.columns(2)

    with c1:
        age = st.slider("Age", 20, 100, int(get_default("age", 50)))
        sex = st.selectbox("Sex", sex_options, index=sex_options.index(get_default("sex", sex_options[0])))
        cp = st.selectbox("Chest Pain Type", cp_options, index=cp_options.index(get_default("cp", cp_options[0])))
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=50.0,
            max_value=250.0,
            value=float(get_default("trestbps", 130.0))
        )
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=50.0,
            max_value=700.0,
            value=float(get_default("chol", 240.0))
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [True, False],
            index=[True, False].index(get_default("fbs", False))
        )
        restecg = st.selectbox(
            "Resting ECG Result",
            restecg_options,
            index=restecg_options.index(get_default("restecg", restecg_options[0]))
        )

    with c2:
        thalch = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=50.0,
            max_value=250.0,
            value=float(get_default("thalch", 150.0))
        )
        exang = st.selectbox(
            "Exercise-Induced Angina",
            [True, False],
            index=[True, False].index(get_default("exang", False))
        )
        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=float(get_default("oldpeak", 1.0)),
            step=0.1
        )
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            slope_options,
            index=slope_options.index(get_default("slope", slope_options[0]))
        )
        ca = st.number_input(
            "Number of Major Vessels",
            min_value=0.0,
            max_value=4.0,
            value=float(get_default("ca", 0.0)),
            step=1.0
        )
        thal = st.selectbox(
            "Thalassemia",
            thal_options,
            index=thal_options.index(get_default("thal", thal_options[0]))
        )
        dataset_source = st.selectbox(
            "Hospital / Dataset Source",
            dataset_options,
            index=dataset_options.index(get_default("dataset", dataset_options[0]))
        )

    submitted = st.form_submit_button("Predict Heart Disease Risk")

st.markdown('</div>', unsafe_allow_html=True)


# RISK DISPLAY FUNCTION
def get_risk_label(probability):
    if probability < 0.35:
        return "Low Risk", "result-low"
    elif probability < 0.65:
        return "Medium Risk", "result-medium"
    else:
        return "High Risk", "result-high"

# Readable Format
def make_feature_names_readable(feature_names):
    readable_names = []

    for name in feature_names:
        name = name.replace("num__", "")
        name = name.replace("cat__", "")

        replacements = {
            "age": "Age",
            "sex_Male": "Sex: Male",
            "sex_Female": "Sex: Female",
            "dataset_Cleveland": "Dataset source: Cleveland",
            "dataset_Hungary": "Dataset source: Hungary",
            "dataset_Switzerland": "Dataset source: Switzerland",
            "dataset_VA Long Beach": "Dataset source: VA Long Beach",
            "cp_asymptomatic": "Chest pain: Asymptomatic",
            "cp_atypical angina": "Chest pain: Atypical angina",
            "cp_non-anginal": "Chest pain: Non-anginal pain",
            "cp_typical angina": "Chest pain: Typical angina",
            "trestbps": "Resting blood pressure",
            "chol": "Serum cholesterol",
            "fbs_False": "Fasting blood sugar: No",
            "fbs_True": "Fasting blood sugar: Yes",
            "restecg_lv hypertrophy": "Resting ECG: LV hypertrophy",
            "restecg_normal": "Resting ECG: Normal",
            "restecg_st-t abnormality": "Resting ECG: ST-T abnormality",
            "thalch": "Maximum heart rate",
            "exang_False": "Exercise angina: No",
            "exang_True": "Exercise angina: Yes",
            "oldpeak": "ST depression",
            "slope_downsloping": "ST slope: Downsloping",
            "slope_flat": "ST slope: Flat",
            "slope_upsloping": "ST slope: Upsloping",
            "ca": "Number of major vessels",
            "thal_fixed defect": "Thalassemia: Fixed defect",
            "thal_normal": "Thalassemia: Normal",
            "thal_reversable defect": "Thalassemia: Reversible defect"
        }

        readable_names.append(replacements.get(name, name.replace("_", " ").title()))

    return readable_names

# PREDICTION SECTION
if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "dataset": dataset_source,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    risk_label, risk_class = get_risk_label(prob)

    st.subheader("Prediction Outcome")

    r1, r2 = st.columns([2, 1])

    with r1:
        message = "Heart disease detected" if pred == 1 else "No heart disease detected"
        st.markdown(
            f'<div class="{risk_class}">{risk_label}: {message}<br><span class="small-note">Predicted probability: {prob:.2%}</span></div>',
            unsafe_allow_html=True
        )

    with r2:
        st.metric("Disease Probability", f"{prob:.2%}")

    
    # TRAFFIC LIGHT RISK GAUGE
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Traffic-light Risk Gauge")

    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
    with gauge_col1:
        st.markdown("🟢 **Low Risk**  \nProbability < 35%")
    with gauge_col2:
        st.markdown("🟠 **Medium Risk**  \nProbability 35% to 64%")
    with gauge_col3:
        st.markdown("🔴 **High Risk**  \nProbability ≥ 65%")

    st.progress(float(prob))
    st.caption(f"Current patient estimated disease probability: {prob:.2%}")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("View entered patient data"):
        st.dataframe(input_df)

    # TRANSFORM DATA FOR EXPLAINABILITY
    X_train_transformed = model.named_steps["preprocessor"].transform(X_train)
    input_transformed = model.named_steps["preprocessor"].transform(input_df)

    if hasattr(X_train_transformed, "toarray"):
        X_train_dense = X_train_transformed.toarray()
    else:
        X_train_dense = X_train_transformed

    if hasattr(input_transformed, "toarray"):
        input_dense = input_transformed.toarray()
    else:
        input_dense = input_transformed

    feature_names_raw = model.named_steps["preprocessor"].get_feature_names_out()
    feature_names = make_feature_names_readable(feature_names_raw)
    lr_model = model.named_steps["model"]

  
    # SHAP
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("SHAP Explanation")
    st.caption("This explains which features pushed the prediction higher or lower for this patient.")

    try:
        explainer = shap.Explainer(lr_model, X_train_dense, feature_names=feature_names)
        shap_values = explainer(input_dense)

        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        fig_shap = plt.gcf()
        fig_shap.set_size_inches(10, 5)
        st.pyplot(fig_shap, clear_figure=True)

    except Exception as e:
        st.warning(f"SHAP explanation could not be generated: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

  
    # LIME
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("LIME Explanation")
    st.caption("This provides a local explanation showing the top contributing features for the current prediction.")

    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_dense,
            feature_names=feature_names,
            class_names=["No Disease", "Disease"],
            mode="classification"
        )

        exp = lime_explainer.explain_instance(
            data_row=input_dense[0],
            predict_fn=lr_model.predict_proba,
            num_features=10
        )

        lime_df = pd.DataFrame(exp.as_list(), columns=["Feature", "Contribution"])
        st.dataframe(lime_df, use_container_width=True)

        fig_lime = exp.as_pyplot_figure()
        fig_lime.set_size_inches(8, 4)
        st.pyplot(fig_lime, clear_figure=True)

    except Exception as e:
        st.warning(f"LIME explanation could not be generated: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# FOOTER
st.markdown("---")
st.caption(
    "Prototype developed for MSc project demonstration. "
    "This system is for academic use only and does not replace clinical diagnosis."
)