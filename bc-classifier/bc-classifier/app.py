import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from metrics_utils import compute_metrics

st.set_page_config(page_title="Breast Cancer Classifier Lab", page_icon="🔬", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
CLASS_NAMES = ["Malignant (0)", "Benign (1)"]


@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.title("🔬 Breast Cancer Diagnosis — Classifier Comparison")
st.caption("Pre-trained models on the Breast Cancer Wisconsin (Diagnostic) dataset. "
           "Upload a test CSV; models are only loaded and evaluated, never retrained.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV (features + `target` column)", type="csv")
    choice = st.selectbox("Select model", list(MODEL_FILES))
    st.markdown("Don't have a file? Use `test_data.csv` from the GitHub repo.")

if uploaded is None:
    st.info("⬅️ Upload the test CSV in the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded)
if "target" not in df.columns:
    st.error("The CSV must contain a `target` column (0 = malignant, 1 = benign).")
    st.stop()

model = load_model(MODEL_FILES[choice])
expected = list(model.feature_names_in_)
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()

X, y = df[expected], df["target"]
with st.expander(f"Preview of uploaded data ({len(df)} rows)"):
    st.dataframe(df.head(10), use_container_width=True)

tab1, tab2, tab3 = st.tabs([f"📊 {choice}: metrics", "🧮 Confusion matrix & report", "🏆 All models"])

with tab1:
    m = compute_metrics(model, X, y)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")

with tab2:
    left, right = st.columns(2)
    pred = model.predict(X)
    with left:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=CLASS_NAMES, cmap="Blues", ax=ax)
        st.pyplot(fig)
    with right:
        rep = classification_report(y, pred, target_names=CLASS_NAMES, output_dict=True)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

with tab3:
    rows = [{"ML Model Name": n, **compute_metrics(load_model(p), X, y)} for n, p in MODEL_FILES.items()]
    comp = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"), use_container_width=True)
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
