"""
Model Training and Performance Evaluation Module
Provides clinical dataset generation, multi-algorithm training
(Random Forest, SVM, KNN, Decision Tree, Logistic Regression),
hyperparameter management, model persistence, and comprehensive evaluation
(Confusion Matrix, Multi-Class ROC Curves, Classification Report, Feature Importance).
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

CLASS_NAMES = [
    "Normal Sinus Rhythm",
    "Sinus Bradycardia",
    "Sinus Tachycardia",
    "Cardiac Arrhythmia",
]

FEATURE_NAMES = [
    "heart_rate_bpm",
    "mean_rr_ms",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",
    "signal_std",
    "signal_skew",
    "signal_kurtosis",
    "signal_energy",
    "signal_entropy",
]

def generate_clinical_dataset(n_samples_per_class=350, random_state=42):
    """
    Generate a medically realistic, calibrated dataset of ECG features
    for 4 primary heart rate & rhythm diagnostic categories:
    1. Normal Sinus Rhythm (NSR: 60-100 BPM)
    2. Sinus Bradycardia (BRAD: <60 BPM)
    3. Sinus Tachycardia (TACH: >100 BPM)
    4. Cardiac Arrhythmia (ARR: High RR variance, irregular beats)
    """
    np.random.seed(random_state)
    data = []
    labels = []

    for class_idx in range(4):
        n = n_samples_per_class
        if class_idx == 0:  # Normal Sinus Rhythm
            hr = np.random.normal(74.0, 7.0, n)
            hr = np.clip(hr, 60.0, 98.0)
            rr_mean = 60000.0 / hr + np.random.normal(0, 10, n)
            sdnn = np.random.normal(42.0, 8.0, n)
            rmssd = np.random.normal(36.0, 7.0, n)
            pnn50 = np.random.normal(12.0, 4.0, n)
            sig_std = np.random.normal(1.0, 0.05, n)
            sig_skew = np.random.normal(0.85, 0.15, n)
            sig_kurt = np.random.normal(2.6, 0.4, n)
            sig_energy = np.random.normal(1.05, 0.1, n)
            sig_entropy = np.random.normal(2.4, 0.15, n)

        elif class_idx == 1:  # Sinus Bradycardia
            hr = np.random.normal(49.0, 5.0, n)
            hr = np.clip(hr, 35.0, 58.5)
            rr_mean = 60000.0 / hr + np.random.normal(0, 20, n)
            sdnn = np.random.normal(32.0, 7.0, n)
            rmssd = np.random.normal(26.0, 6.0, n)
            pnn50 = np.random.normal(7.0, 3.0, n)
            sig_std = np.random.normal(0.92, 0.06, n)
            sig_skew = np.random.normal(0.65, 0.18, n)
            sig_kurt = np.random.normal(2.1, 0.35, n)
            sig_energy = np.random.normal(0.88, 0.1, n)
            sig_entropy = np.random.normal(2.1, 0.2, n)

        elif class_idx == 2:  # Sinus Tachycardia
            hr = np.random.normal(126.0, 12.0, n)
            hr = np.clip(hr, 102.0, 165.0)
            rr_mean = 60000.0 / hr + np.random.normal(0, 12, n)
            sdnn = np.random.normal(24.0, 5.0, n)
            rmssd = np.random.normal(18.0, 4.5, n)
            pnn50 = np.random.normal(4.0, 2.0, n)
            sig_std = np.random.normal(1.08, 0.06, n)
            sig_skew = np.random.normal(1.15, 0.2, n)
            sig_kurt = np.random.normal(3.2, 0.45, n)
            sig_energy = np.random.normal(1.22, 0.12, n)
            sig_entropy = np.random.normal(2.65, 0.18, n)

        else:  # Cardiac Arrhythmia / Irregular Rhythm
            hr = np.random.normal(88.0, 18.0, n)
            hr = np.clip(hr, 55.0, 140.0)
            rr_mean = 60000.0 / hr + np.random.normal(0, 45, n)
            sdnn = np.random.normal(95.0, 22.0, n)  # hallmark: high SDNN
            rmssd = np.random.normal(85.0, 20.0, n)  # hallmark: high RMSSD
            pnn50 = np.random.normal(38.0, 10.0, n)  # hallmark: elevated pNN50
            sig_std = np.random.normal(1.25, 0.15, n)
            sig_skew = np.random.normal(0.40, 0.3, n)
            sig_kurt = np.random.normal(4.1, 0.8, n)
            sig_energy = np.random.normal(1.35, 0.2, n)
            sig_entropy = np.random.normal(3.2, 0.25, n)

        # Ensure no negative or impossible physiological values
        sdnn = np.clip(sdnn, 5.0, 250.0)
        rmssd = np.clip(rmssd, 5.0, 250.0)
        pnn50 = np.clip(pnn50, 0.0, 100.0)
        sig_energy = np.clip(sig_energy, 0.1, 5.0)

        class_features = np.column_stack([
            hr, rr_mean, sdnn, rmssd, pnn50,
            sig_std, sig_skew, sig_kurt, sig_energy, sig_entropy
        ])

        data.append(class_features)
        labels.extend([class_idx] * n)

    X = np.vstack(data)
    y = np.array(labels, dtype=int)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    df["diagnosis"] = [CLASS_NAMES[i] for i in y]

    return df

def get_classifier(algorithm_name, **kwargs):
    """Factory to create classifier instances based on name."""
    algo = algorithm_name.lower().replace(" ", "_")
    if "random_forest" in algo:
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", 10)
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif "svm" in algo or "support_vector" in algo:
        C = kwargs.get("C", 1.0)
        kernel = kwargs.get("kernel", "rbf")
        return SVC(C=C, kernel=kernel, probability=True, random_state=42)
    elif "knn" in algo or "k_nearest" in algo:
        n_neighbors = kwargs.get("n_neighbors", 5)
        return KNeighborsClassifier(n_neighbors=n_neighbors)
    elif "decision_tree" in algo:
        max_depth = kwargs.get("max_depth", 6)
        return DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif "logistic" in algo:
        return LogisticRegression(max_iter=1000, random_state=42)
    else:
        return RandomForestClassifier(n_estimators=100, random_state=42)

def train_and_evaluate_model(
    df=None,
    algorithm="Random Forest",
    test_size=0.2,
    random_state=42,
    **algo_kwargs
):
    """
    Full training and evaluation pipeline:
    1. Splits dataset into train and test sets
    2. Applies standard scaling
    3. Trains selected Machine Learning model
    4. Computes Accuracy, Precision, Recall, F1-Score
    5. Computes Confusion Matrix
    6. Computes Multi-Class ROC curves & AUC
    7. Computes Feature Importances (if supported)
    Returns:
        dict: Full model artifact bundle and evaluation metrics
    """
    if df is None:
        df = generate_clinical_dataset(n_samples_per_class=350, random_state=random_state)

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Classifier
    clf = get_classifier(algorithm, **algo_kwargs)
    clf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled) if hasattr(clf, "predict_proba") else None

    # Performance Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    # ROC Curves (One-vs-Rest for multi-class)
    roc_data = {}
    if y_proba is not None:
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
        for i, class_name in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = float(auc(fpr, tpr))
            roc_data[class_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}

    # Feature Importances (if model has them)
    feature_importances = {}
    if hasattr(clf, "feature_importances_"):
        for name, imp in zip(FEATURE_NAMES, clf.feature_importances_):
            feature_importances[name] = float(imp)
    elif hasattr(clf, "coef_"):
        mean_coef = np.mean(np.abs(clf.coef_), axis=0)
        for name, imp in zip(FEATURE_NAMES, mean_coef):
            feature_importances[name] = float(imp)

    bundle = {
        "model": clf,
        "scaler": scaler,
        "algorithm": algorithm,
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "metrics": {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "precision_weighted": prec_weighted,
            "recall_weighted": rec_weighted,
            "f1_weighted": f1_weighted,
        },
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "roc_data": roc_data,
        "feature_importances": feature_importances,
        "test_size": test_size,
        "n_samples": len(df),
    }

    return bundle

def save_model(bundle, filepath="models/trained_ecg_model.joblib"):
    """Persist model bundle to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(bundle, filepath)
    return filepath

def load_model(filepath="models/trained_ecg_model.joblib"):
    """Load model bundle from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)

def predict_single_sample(model_bundle, feature_vector_or_dict):
    """
    Run diagnostic prediction for an input sample.
    Returns:
        dict: {
            'predicted_class': str,
            'class_index': int,
            'probabilities': dict {class_name: float},
            'confidence': float
        }
    """
    clf = model_bundle["model"]
    scaler = model_bundle["scaler"]
    class_names = model_bundle["class_names"]
    feature_names = model_bundle["feature_names"]

    if isinstance(feature_vector_or_dict, dict):
        vec = np.array([feature_vector_or_dict[k] for k in feature_names], dtype=np.float32).reshape(1, -1)
    else:
        vec = np.array(feature_vector_or_dict, dtype=np.float32).reshape(1, -1)

    vec_scaled = scaler.transform(vec)
    pred_idx = int(clf.predict(vec_scaled)[0])

    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(vec_scaled)[0]
        prob_dict = {class_names[i]: float(probas[i]) for i in range(len(class_names))}
        confidence = float(probas[pred_idx])
    else:
        prob_dict = {class_names[i]: 1.0 if i == pred_idx else 0.0 for i in range(len(class_names))}
        confidence = 1.0

    return {
        "predicted_class": class_names[pred_idx],
        "class_index": pred_idx,
        "probabilities": prob_dict,
        "confidence": confidence,
    }
