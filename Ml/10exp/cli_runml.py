"""
Command-Line Interface (CLI) Runner for ECG Heart Rate & Arrhythmia Classification
Allows running the entire pipeline from the terminal:
Load image -> Preprocess -> Feature Extraction -> BPM Estimation -> ML Diagnostic Prediction.

Usage:
    # 1. Train model from CLI:
    python cli_run.py --train --algo "Random Forest"

    # 2. Predict on an ECG image:
    python cli_run.py --image sample_images/normal_ecg.png
    python cli_run.py --image sample_images/tachycardia_ecg.png
    python cli_run.py --image sample_images/bradycardia_ecg.png
    python cli_run.py --image sample_images/arrhythmia_ecg.png
"""

import argparse
import os
import sys
import numpy as np

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from preprocessing import preprocess_pipeline
from feature_extraction import extract_all_features
from model_trainer import (
    train_and_evaluate_model,
    save_model,
    load_model,
    predict_single_sample,
    CLASS_NAMES,
)

def run_training_pipeline(algorithm="Random Forest", model_path="models/trained_ecg_model.joblib"):
    print("\n=======================================================")
    print("      HEART RATE ML MODEL TRAINING & BENCHMARKING      ")
    print("=======================================================")
    print(f"Algorithm selected: {algorithm}")
    print("Generating calibrated clinical benchmark dataset (1,400 samples across 4 classes)...")
    
    bundle = train_and_evaluate_model(algorithm=algorithm)
    save_model(bundle, model_path)
    
    metrics = bundle["metrics"]
    print(f"\n[OK] Model successfully trained and saved to: {model_path}")
    print("-------------------------------------------------------")
    print(f"Overall Accuracy:        {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:         {metrics['precision_macro'] * 100:.2f}%")
    print(f"Macro Recall:            {metrics['recall_macro'] * 100:.2f}%")
    print(f"Macro F1-Score:          {metrics['f1_macro'] * 100:.2f}%")
    print("-------------------------------------------------------")
    print("Confusion Matrix:")
    print(bundle["confusion_matrix"])
    print("-------------------------------------------------------")
    return bundle

def run_inference_pipeline(image_path, model_path="models/trained_ecg_model.joblib", sampling_rate=250):
    print("\n=======================================================")
    print("   END-TO-END ECG HEART RATE ANALYSIS & ML DIAGNOSIS   ")
    print("=======================================================")
    if not os.path.exists(image_path):
        print(f"[ERROR] Image path does not exist: {image_path}")
        sys.exit(1)

    print(f"Step 1: Loading input image: {image_path}")
    preproc = preprocess_pipeline(image_path)
    h, w, c = preproc["original_rgb"].shape
    print(f"        Image dimensions: {w}x{h} ({c} channels)")

    print("Step 2: Preprocessing pipeline completed:")
    print("        - Grayscale conversion [OK]")
    print("        - Contrast enhancement (CLAHE) [OK]")
    print("        - Noise filtering (Gaussian blur) [OK]")
    print("        - Otsu thresholding & Binarization [OK]")
    print("        - Morphological waveform skeletonization [OK]")

    print("Step 3: Extracting 1D Cardiac Waveform & Features:")
    feat = extract_all_features(preproc["skeleton"], sampling_rate=sampling_rate)
    hr_metrics = feat["hr_metrics"]
    stats = feat["stat_features"]

    print("\n------------------- VITAL SIGNS -----------------------")
    print(f"  Detected R-Peaks:     {hr_metrics['num_beats']} beats")
    print(f"  Estimated Heart Rate: {hr_metrics['heart_rate_bpm']:.1f} BPM")
    print(f"  Mean RR Interval:     {hr_metrics['mean_rr_ms']:.1f} ms")
    print(f"  Min / Max RR:         {hr_metrics['min_rr_ms']:.1f} ms / {hr_metrics['max_rr_ms']:.1f} ms")
    print(f"  SDNN (HRV variance):  {hr_metrics['sdnn_ms']:.2f} ms")
    print(f"  RMSSD:                {hr_metrics['rmssd_ms']:.2f} ms")
    print(f"  pNN50:                {hr_metrics['pnn50']:.1f} %")
    print("----------------- SIGNAL STATISTICS -------------------")
    print(f"  Signal Std Dev:       {stats['signal_std']:.4f}")
    print(f"  Signal Skewness:      {stats['signal_skew']:.4f}")
    print(f"  Signal Kurtosis:      {stats['signal_kurtosis']:.4f}")
    print(f"  Signal Energy:        {stats['signal_energy']:.4f}")
    print(f"  Shannon Entropy:      {stats['signal_entropy']:.4f}")
    print("-------------------------------------------------------")

    print("\nStep 4: Machine Learning Diagnostic Classification:")
    if not os.path.exists(model_path):
        print(f"[INFO] Model file {model_path} not found. Training default model now...")
        run_training_pipeline("Random Forest", model_path)

    bundle = load_model(model_path)
    result = predict_single_sample(bundle, feat["feature_dict"])

    print(f"\n>>> DIAGNOSTIC RESULT:  {result['predicted_class'].upper()}")
    print(f">>> MODEL CONFIDENCE:   {result['confidence'] * 100:.2f}%\n")
    print("Class Probabilities:")
    for c_name, prob in result["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"  - {c_name:<24}: {prob * 100:5.1f}%  |{bar}")
    print("=======================================================\n")

def main():
    parser = argparse.ArgumentParser(description="ECG Heart Rate ML Analysis CLI")
    parser.add_argument("--image", type=str, help="Path to ECG image file to analyze")
    parser.add_argument("--train", action="store_true", help="Train a new ML model")
    parser.add_argument("--algo", type=str, default="Random Forest", help="Algorithm: Random Forest, SVM, KNN, Decision Tree")
    parser.add_argument("--model", type=str, default="models/trained_ecg_model.joblib", help="Model file path")
    parser.add_argument("--fs", type=int, default=250, help="ECG sampling rate in Hz (default: 250)")

    args = parser.parse_args()

    if args.train:
        run_training_pipeline(args.algo, args.model)
    elif args.image:
        run_inference_pipeline(args.image, args.model, args.fs)
    else:
        # Default behavior: if sample images exist, analyze normal_ecg.png
        default_img = "sample_images/normal_ecg.png"
        if os.path.exists(default_img):
            print("[INFO] No arguments specified. Running demonstration on sample normal_ecg.png:")
            run_inference_pipeline(default_img, args.model, args.fs)
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
