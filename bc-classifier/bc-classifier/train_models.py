"""Train five classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset,
save them to model/, write test_data.csv and metrics_summary.csv."""
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from metrics_utils import compute_metrics

SEED = 42
data = load_breast_cancer(as_frame=True)
df = data.frame  # 'target': 0 = malignant, 1 = benign

X, y = df.drop(columns="target"), df["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
pd.concat([X_te, y_te], axis=1).to_csv("test_data.csv", index=False)

models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=SEED))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED),
    "KNN": Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=7))]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED))]),
}
rows = []
for name, mdl in models.items():
    mdl.fit(X_tr, y_tr)
    joblib.dump(mdl, f"model/{name.lower().replace(' ', '_')}.joblib")
    rows.append({"ML Model Name": name, **compute_metrics(mdl, X_te, y_te)})
res = pd.DataFrame(rows).round(4)
res.to_csv("metrics_summary.csv", index=False)
print(res.to_markdown(index=False))
