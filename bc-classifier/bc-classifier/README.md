# Breast Cancer Diagnosis – Classification Model Comparison

## a. Problem statement
Predict whether a breast tumour is **malignant (0)** or **benign (1)** from measurements of cell nuclei taken from a digitised fine-needle-aspirate image. Five classical ML classifiers are trained on the same data, compared on six metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and exposed through an interactive Streamlit app that evaluates the saved models on an uploaded test CSV.

## b. Dataset description
- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset (UCI Machine Learning Repository; bundled with scikit-learn as `load_breast_cancer`).
- **Instances:** 569  |  **Features:** 30 numeric (mean, standard error and "worst" value of 10 nucleus measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- **Target:** `target` – 0 = malignant (212), 1 = benign (357). Binary classification, mildly imbalanced.
- **Split:** 80/20 stratified (`random_state=42`) → 455 train / 114 test. `test_data.csv` holds the 114 test rows. LogReg, KNN and SVM include a `StandardScaler` inside their pipeline.

## c. GitHub repository link
https://github.com/manishy108/ML_ASSIGNMENT2

## d. Models used
Evaluation on the held-out test set (114 rows):

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9825 | 0.9954 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |
| Decision Tree          |     0.9035 | 0.9373 |      0.9420 |   0.9028 | 0.9220 | 0.7969 |
| KNN                    |     0.9737 | 0.9884 |      0.9600 |   1.0000 | 0.9796 | 0.9442 |
| Naive Bayes            |     0.9386 | 0.9878 |      0.9452 |   0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine |     0.9825 | 0.9950 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Excellent and stable: 98.25% accuracy and the joint-best AUC (0.9954). After scaling, the classes are close to linearly separable, so a linear boundary suffices. Only 2 of 114 cases misclassified. |
| Decision Tree | Weakest model (90.35% accuracy, MCC 0.797). A depth-limited single tree is high-variance and cannot match smooth boundaries; its AUC (0.937) is lowest because leaf probabilities are coarse. |
| KNN | Strong (97.37% accuracy) with perfect recall (1.00): no benign case was missed, but lower precision (0.96) means a few malignant cases were labelled benign. Sensitive to feature scaling and the choice of k. |
| Naive Bayes | Good (93.86% accuracy) with high AUC (0.988) despite correlated features violating its independence assumption; its probability ranking is good but hard-label decisions are less accurate. Fastest to train. |
| Support Vector Machine | Ties Logistic Regression on accuracy, precision, recall, F1 and MCC (0.9623) with AUC 0.9950. The RBF kernel handles mild non-linearity well. |
| **Overall winner** | **Logistic Regression** – ties with SVM on threshold metrics but has the highest AUC, and is simpler, faster and more interpretable. SVM is a very close second; Decision Tree is clearly the weakest. |

*Results come from a single 114-sample split, so differences of one or two predictions (LR vs SVM vs KNN) are not statistically significant.*

## Repository structure
```
app.py                  Streamlit app (upload CSV, choose model, metrics, confusion matrix, comparison)
train_models.py         Trains the 5 models, saves them, writes test_data.csv
metrics_utils.py        Shared metric computation
model/                  Saved models (.joblib)
test_data.csv           Held-out test data (features + target)
requirements.txt
```

## Run locally
```
pip install -r requirements.txt
python train_models.py     # optional, regenerates models and test_data.csv
streamlit run app.py
```
Live app: https://mlassignment2-a2cqbrxmdgam3hwokc2tbe.streamlit.app/
