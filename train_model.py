# ================================================================
#  GlycoSense — Diabetes Risk Predictor
#  Dataset : Pima Indians Diabetes Database (UCI / Kaggle)
#  File    : diabetes.csv  (768 rows × 9 columns)
#
#  NOTE ON KNN IN THIS PROJECT:
#  KNN is used in TWO different roles:
#
#  1. KNNImputer (preprocessing)  — fills fake/missing zeros
#     using the 5 nearest patient neighbors. Lives inside
#     EVERY model's pipeline as a data-cleaning step.
#
#  2. KNeighborsClassifier (model) — one of our 4 classifiers.
#     Uses the same K-Nearest Neighbors algorithm but for
#     PREDICTION: classifies a patient as diabetic or not
#     based on similarity to training patients.
#
#  Models trained:
#    1. Gradient Boosting Classifier
#    2. Random Forest Classifier
#    3. Logistic Regression
#    4. KNN Classifier  (k=11, distance-weighted)
# ================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier   # KNN as CLASSIFIER
from sklearn.impute import KNNImputer                # KNN as IMPUTER
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
import joblib, json, warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ────────────────────────────────────────────────────────────────
df = pd.read_csv('diabetes.csv')

print("=" * 65)
print("  DATASET LOADED FROM diabetes.csv")
print("=" * 65)
print(f"\n📌 Shape         : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"📌 Columns       : {list(df.columns)}")
print(f"\n📌 First 5 rows  :\n{df.head()}")
print(f"\n📌 Class balance :\n{df['Outcome'].value_counts()}")
print(f"   → {df['Outcome'].mean()*100:.1f}% diabetic in dataset")

# ────────────────────────────────────────────────────────────────
# STEP 2 — FIX FAKE ZEROS → NaN
# ────────────────────────────────────────────────────────────────
# In the Pima dataset, 0 is used as a placeholder for MISSING
# data in 5 columns. These zeros are biologically impossible:
#   Glucose=0, BMI=0, BloodPressure=0 cannot exist in a living person.
# We replace them with NaN so the KNN Imputer can fill them properly.

print("\n" + "=" * 65)
print("  STEP 2 : FIXING FAKE ZEROS → NaN")
print("=" * 65)

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("\n📌 Zero counts BEFORE cleaning (these are FAKE zeros):")
for col in zero_cols:
    count = (df[col] == 0).sum()
    print(f"   {col:<28}: {count:>3} zeros ({count/len(df)*100:.1f}% of data)")

df_clean = df.copy()
for col in zero_cols:
    df_clean[col] = df_clean[col].replace(0, np.nan)

print("\n📌 Missing values AFTER replacing zeros with NaN:")
print(df_clean[zero_cols].isnull().sum().to_string())
print("""
✅ Fake zeros fixed!
   These NaN values will be filled by KNNImputer (Role 1 of KNN)
   inside every model pipeline before training begins.
""")

# ────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 3 : FEATURE ENGINEERING")
print("=" * 65)

G = df_clean['Glucose'].fillna(df_clean['Glucose'].median())
B = df_clean['BMI'].fillna(df_clean['BMI'].median())
I = df_clean['Insulin'].fillna(df_clean['Insulin'].median())

df_clean['Glucose_BMI']      = G * B
df_clean['Age_Pregnancies']  = df_clean['Age'] * df_clean['Pregnancies']
df_clean['Insulin_Glucose']  = I / (G + 1)
df_clean['BMI_Age']          = B * df_clean['Age']
df_clean['Glucose_category'] = pd.cut(
    G, bins=[0, 70, 100, 125, 300], labels=[0, 1, 2, 3]
).astype(float)

print("\n   New features created from clinical domain knowledge:")
print("   ✅ Glucose_BMI      = Glucose × BMI  (compound metabolic risk)")
print("   ✅ Age_Pregnancies   = Age × Pregnancies  (gestational DM history)")
print("   ✅ Insulin_Glucose   = Insulin ÷ Glucose  (insulin resistance proxy)")
print("   ✅ BMI_Age           = BMI × Age  (age-weighted obesity risk)")
print("   ✅ Glucose_category  = WHO bins: 0=low, 1=normal, 2=pre-DM, 3=DM")

feature_cols = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age',
    'Glucose_BMI', 'Age_Pregnancies', 'Insulin_Glucose',
    'BMI_Age', 'Glucose_category'
]
print(f"\n📌 Total features: {len(feature_cols)} (8 original + 5 engineered)")

X = df_clean[feature_cols]
y = df_clean['Outcome']

# ────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN / TEST SPLIT
# ────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\n📌 Training set : {X_train.shape[0]} samples (80%)")
print(f"📌 Test set     : {X_test.shape[0]} samples (20%)")
print(f"📌 Stratified   : yes — class ratio preserved in both sets")

# ────────────────────────────────────────────────────────────────
# STEP 5 — DEFINE 4 MODELS
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 5 : DEFINING 4 MODELS")
print("=" * 65)
print("""
  All 4 pipelines share the same preprocessing:
  ┌─────────────────────────────────────────────────────┐
  │  KNNImputer(k=5)  →  RobustScaler  →  Classifier   │
  │  (fill NaN/fake        (scale with      (predict    │
  │   zeros using 5        median+IQR,      diabetic    │
  │   nearest patients)    robust to        or not)     │
  │                        outliers)                    │
  └─────────────────────────────────────────────────────┘

  KNN appears TWICE in this project:
  ─ KNNImputer   → data cleaning step (inside every pipeline)
  ─ KNeighborsClassifier → one of the 4 prediction models
""")

models_def = {
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08,
        max_depth=4, subsample=0.8,
        min_samples_split=10, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6,
        min_samples_split=10, random_state=42,
        class_weight='balanced'
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=0.5,
        class_weight='balanced', random_state=42
    ),
    "KNN Classifier": KNeighborsClassifier(
        n_neighbors=11,          # odd number avoids ties
        weights='distance',      # closer patients vote more
        metric='euclidean'
    ),
}

# ────────────────────────────────────────────────────────────────
# STEP 6 — TRAIN ALL 4 MODELS
# ────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 6 : TRAINING ALL 4 MODELS")
print("=" * 65)

trained_pipelines = {}
model_results     = {}
cv                = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models_def.items():
    print(f"\n  🔧 Training: {name}")

    # Every pipeline: KNNImputer → RobustScaler → Classifier
    pipe = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),   # KNN Role 1: fill missing
        ('scaler',  RobustScaler()),
        ('model',   clf)                           # KNN Role 2 (if KNN Classifier)
    ])

    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    f1     = f1_score(y_test, y_pred)

    trained_pipelines[name] = pipe
    model_results[name] = {
        "accuracy":    round(acc, 4),
        "roc_auc":     round(auc, 4),
        "f1_score":    round(f1, 4),
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std":  round(float(cv_auc.std()), 4),
    }
    print(f"     Accuracy  : {acc*100:.2f}%")
    print(f"     ROC-AUC   : {auc:.4f}")
    print(f"     F1 Score  : {f1:.4f}")
    print(f"     CV AUC    : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Non-Diabetic','Diabetic'])}")

best_name = max(model_results, key=lambda k: model_results[k]['roc_auc'])
print(f"\n{'='*65}")
print(f"  🏆 Best Model by ROC-AUC: {best_name}")
print(f"     AUC = {model_results[best_name]['roc_auc']}")
print(f"{'='*65}")

# Save all 4 pipelines
print("\n  Saving model files...")
for name, pipe in trained_pipelines.items():
    fname = name.lower().replace(" ", "_")
    joblib.dump(pipe, f'model_{fname}.pkl')
    print(f"   ✅ model_{fname}.pkl saved")

# ────────────────────────────────────────────────────────────────
# STEP 7 — FEATURE DEPENDENCY ANALYSIS
# ────────────────────────────────────────────────────────────────
# Drop one feature at a time → retrain all 4 models → measure
# how much ROC-AUC drops. Bigger drop = feature was more important.

print("\n" + "=" * 65)
print("  STEP 7 : FEATURE DEPENDENCY ANALYSIS")
print("  Drop one feature → retrain → measure AUC drop")
print("=" * 65)

dependency_results = {}
baseline = {name: model_results[name] for name in models_def}

for feat in feature_cols:
    remaining  = [f for f in feature_cols if f != feat]
    X_tr_drop  = X_train[remaining]
    X_te_drop  = X_test[remaining]
    dependency_results[feat] = {}

    for name, clf_def in models_def.items():
        pipe_drop = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler',  RobustScaler()),
            ('model',   clf_def.__class__(**clf_def.get_params()))
        ])
        pipe_drop.fit(X_tr_drop, y_train)
        y_prob_d = pipe_drop.predict_proba(X_te_drop)[:, 1]
        y_pred_d = pipe_drop.predict(X_te_drop)

        auc_d = roc_auc_score(y_test, y_prob_d)
        acc_d = accuracy_score(y_test, y_pred_d)

        dependency_results[feat][name] = {
            "accuracy": round(acc_d, 4),
            "roc_auc":  round(auc_d, 4),
            "acc_drop": round(baseline[name]['accuracy'] - acc_d, 4),
            "auc_drop": round(baseline[name]['roc_auc']  - auc_d, 4),
        }

    avg_drop = np.mean([dependency_results[feat][n]['auc_drop'] for n in models_def])
    print(f"   {feat:<30} avg AUC drop: {avg_drop:+.4f}")

print("\n✅ Feature dependency analysis complete")

# ────────────────────────────────────────────────────────────────
# STEP 8 — FEATURE IMPORTANCES (tree models only)
# ────────────────────────────────────────────────────────────────
feature_importances = {}
for name in ["Gradient Boosting", "Random Forest"]:
    imps = trained_pipelines[name].named_steps['model'].feature_importances_
    feature_importances[name] = {
        feat: round(float(imp), 4)
        for feat, imp in zip(feature_cols, imps)
    }

# ────────────────────────────────────────────────────────────────
# STEP 9 — SAVE ALL METADATA
# ────────────────────────────────────────────────────────────────
meta = {
    "feature_cols":        feature_cols,
    "zero_cols":           zero_cols,
    "best_model":          best_name,
    "model_results":       model_results,
    "dependency_results":  dependency_results,
    "feature_importances": feature_importances,
    "train_samples":       int(X_train.shape[0]),
    "test_samples":        int(X_test.shape[0]),
    "dataset_source":      "Pima Indians Diabetes Database — UCI / Kaggle",
    "knn_note":            "KNN is used TWICE: as KNNImputer (data cleaning) and as KNeighborsClassifier (prediction model)"
}
json.dump(meta, open('model_meta.json', 'w'), indent=2)

print("\n✅ model_meta.json saved")
print("\n" + "=" * 65)
print("  ALL DONE!")
print("  Models saved: model_gradient_boosting.pkl")
print("                model_random_forest.pkl")
print("                model_logistic_regression.pkl")
print("                model_knn_classifier.pkl")
print("  Run: python3 app.py → open http://localhost:5000")
print("=" * 65)
