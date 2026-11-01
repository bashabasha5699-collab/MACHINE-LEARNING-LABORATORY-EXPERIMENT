"""
Streamlit Web Application: Medical Heart Rate Analysis & ML Pipeline
Unifies all stages into a single-frame interactive medical dashboard:
1. Load Input Image (ECG Strip / Paper scan)
2. Image Preprocessing (Grayscale, CLAHE, Denoising, Otsu Binarization, Skeletonization)
3. Feature Extraction (1D Signal Reconstruction, R-Peak Detection, BPM, HRV Metrics)
4. ML Model Training (Random Forest, SVM, KNN, Decision Tree, Logistic Regression)
5. Diagnostic Classification & Model Performance Evaluation (Confusion Matrix, ROC Curves, F1-Scores)
"""

import os
import io
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from preprocessing import preprocess_pipeline, load_image
from feature_extraction import extract_all_features
from model_trainer import (
    train_and_evaluate_model,
    save_model,
    load_model,
    predict_single_sample,
    generate_clinical_dataset,
    CLASS_NAMES,
    FEATURE_NAMES,
)

# Page Configuration
st.set_page_config(
    page_title="CardioVision ML - Heart Rate & ECG Analyzer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Medical Aesthetic Styling
st.markdown(
    """
    <style>
        /* General page layout and header */
        .main-header {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            padding: 24px;
            border-radius: 12px;
            color: #ffffff;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        .main-header h1 {
            color: #ffffff !important;
            font-size: 28px;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .main-header p {
            color: #b0bec5;
            font-size: 14px;
            margin-top: 6px;
            margin-bottom: 0;
        }
        
        /* Metric card styling */
        .card-metric {
            background: #ffffff;
            border-left: 5px solid #0288d1;
            padding: 16px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 12px;
        }
        .card-metric h4 {
            margin: 0;
            font-size: 13px;
            color: #546e7a;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .card-metric .val {
            font-size: 26px;
            font-weight: bold;
            color: #102027;
            margin: 6px 0 2px 0;
        }
        .card-metric .sub {
            font-size: 12px;
            color: #78909c;
        }
        
        /* Diagnosis Alert Badges */
        .diag-box-normal {
            background-color: #e8f5e9;
            border-left: 6px solid #2e7d32;
            padding: 18px;
            border-radius: 8px;
            color: #1b5e20;
            font-size: 18px;
            font-weight: 600;
            margin: 15px 0;
        }
        .diag-box-warning {
            background-color: #fff3e0;
            border-left: 6px solid #ef6c00;
            padding: 18px;
            border-radius: 8px;
            color: #e65100;
            font-size: 18px;
            font-weight: 600;
            margin: 15px 0;
        }
        .diag-box-danger {
            background-color: #ffebee;
            border-left: 6px solid #c62828;
            padding: 18px;
            border-radius: 8px;
            color: #b71c1c;
            font-size: 18px;
            font-weight: 600;
            margin: 15px 0;
        }
        
        .stage-pill {
            display: inline-block;
            background: #e1f5fe;
            color: #0277bd;
            font-weight: 600;
            font-size: 12px;
            padding: 3px 10px;
            border-radius: 20px;
            margin-bottom: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.markdown(
    """
    <div class="main-header">
        <h1>❤️ CardioVision ML: Clinical Heart Rate & ECG Analysis</h1>
        <p>Unified Machine Learning Pipeline: Image Acquisition ➔ Preprocessing ➔ Waveform Feature Extraction ➔ Model Training ➔ Diagnostic Performance Evaluation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Helper function to get or train model
@st.cache_resource
def get_cached_model(algo_name="Random Forest", n_estimators=100, max_depth=10, c_val=1.0, k_val=5):
    model_path = f"models/cached_{algo_name.lower().replace(' ', '_')}.joblib"
    if os.path.exists(model_path):
        try:
            return load_model(model_path)
        except Exception:
            pass
    
    # Train fresh
    bundle = train_and_evaluate_model(
        algorithm=algo_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        C=c_val,
        n_neighbors=k_val,
    )
    save_model(bundle, model_path)
    return bundle

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    selected_algo = st.selectbox(
        "Classifier Algorithm",
        ["Random Forest", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)", "Decision Tree", "Logistic Regression"],
        index=0
    )
    
    st.subheader("Model Hyperparameters")
    if selected_algo == "Random Forest":
        rf_trees = st.slider("Number of Trees", 20, 300, 100, step=10)
        rf_depth = st.slider("Max Tree Depth", 3, 25, 10)
        algo_kwargs = {"n_estimators": rf_trees, "max_depth": rf_depth}
    elif "SVM" in selected_algo:
        svm_c = st.slider("Regularization (C)", 0.1, 10.0, 1.0, step=0.1)
        algo_kwargs = {"c_val": svm_c}
    elif "KNN" in selected_algo:
        knn_k = st.slider("Number of Neighbors (K)", 1, 15, 5, step=2)
        algo_kwargs = {"k_val": knn_k}
    elif "Decision Tree" in selected_algo:
        dt_depth = st.slider("Max Depth", 2, 15, 6)
        algo_kwargs = {"max_depth": dt_depth}
    else:
        algo_kwargs = {}

    st.subheader("ECG Signal Settings")
    sampling_rate = st.number_input("ECG Sampling Rate (Hz)", min_value=100, max_value=1000, value=250, step=25)
    
    st.markdown("---")
    st.markdown(
        """
        **Medical Reference Bounds:**
        - **Bradycardia**: < 60 BPM
        - **Normal**: 60 – 100 BPM
        - **Tachycardia**: > 100 BPM
        - **Arrhythmia**: High RR Variance
        """
    )
    
    # Retrain button in sidebar
    if st.button("🔄 Retrain ML Model with Current Settings", use_container_width=True):
        with st.spinner("Retraining classifier..."):
            bundle = train_and_evaluate_model(algorithm=selected_algo, **algo_kwargs)
            save_model(bundle, f"models/cached_{selected_algo.lower().replace(' ', '_')}.joblib")
            st.success(f"Trained {selected_algo} (Accuracy: {bundle['metrics']['accuracy']*100:.1f}%)")

# Fetch active model bundle
model_bundle = get_cached_model(selected_algo, **algo_kwargs)

# App Navigation Tabs
tab_unified, tab_preproc, tab_features, tab_training, tab_docs, tab_code = st.tabs([
    "🖥️ One-Frame Unified Pipeline",
    "🖼️ Image Preprocessing Studio",
    "📈 Feature Extraction & Waveform",
    "🏋️ Model Training & Benchmarks",
    "🩺 Clinical Formulas & Guide",
    "💻 Code & Execution Manual",
])

# ==============================================================================
# TAB 1: ONE-FRAME UNIFIED PIPELINE (The exact core requirement)
# ==============================================================================
with tab_unified:
    st.markdown("### 📋 Unified End-to-End Analysis Workflow")
    st.info("💡 Everything runs right here in one screen: upload or select an ECG image, inspect automatic preprocessing, view digitized waveform & detected R-peaks, see calculated BPM vital signs, and receive instant Machine Learning diagnosis with performance statistics.")

    # Image Selection / Upload Section
    col_upload_left, col_upload_right = st.columns([1, 1])

    with col_upload_left:
        st.markdown('<span class="stage-pill">STAGE 1: LOAD INPUT IMAGE</span>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload ECG Strip Image (.png, .jpg, .jpeg, .bmp)",
            type=["png", "jpg", "jpeg", "bmp"],
            help="Upload a standard Lead-II ECG paper scan or synthetic strip image"
        )

    with col_upload_right:
        st.markdown('<span class="stage-pill">OR SELECT PRE-LOADED CLINICAL SAMPLE</span>', unsafe_allow_html=True)
        sample_choice = st.selectbox(
            "Quick Test Samples",
            [
                "None (Use Uploaded Image)",
                "Normal Sinus Rhythm (~75 BPM)",
                "Sinus Bradycardia (~48 BPM)",
                "Sinus Tachycardia (~130 BPM)",
                "Cardiac Arrhythmia (Irregular RR)",
            ],
            index=1 if uploaded_file is None else 0
        )

    # Determine image source
    active_image = None
    sample_mapping = {
        "Normal Sinus Rhythm (~75 BPM)": "sample_images/normal_ecg.png",
        "Sinus Bradycardia (~48 BPM)": "sample_images/bradycardia_ecg.png",
        "Sinus Tachycardia (~130 BPM)": "sample_images/tachycardia_ecg.png",
        "Cardiac Arrhythmia (Irregular RR)": "sample_images/arrhythmia_ecg.png",
    }

    if uploaded_file is not None:
        active_image = uploaded_file
    elif sample_choice != "None (Use Uploaded Image)" and sample_choice in sample_mapping:
        sample_path = sample_mapping[sample_choice]
        if os.path.exists(sample_path):
            active_image = sample_path
        else:
            # Fallback if samples weren't created yet
            from generate_samples import generate_all_sample_assets
            generate_all_sample_assets()
            active_image = sample_path

    if active_image is not None:
        # Run Complete Pipeline
        with st.spinner("Processing ECG image through clinical ML pipeline..."):
            preproc_stages = preprocess_pipeline(active_image)
            feat_results = extract_all_features(preproc_stages["skeleton"], sampling_rate=sampling_rate)
            pred_results = predict_single_sample(model_bundle, feat_results["feature_dict"])

        hr_metrics = feat_results["hr_metrics"]
        stats = feat_results["stat_features"]
        bpm_val = hr_metrics["heart_rate_bpm"]
        pred_label = pred_results["predicted_class"]
        confidence = pred_results["confidence"] * 100.0

        st.markdown("---")

        # ----------------------------------------------------
        # STAGE 2: PREPROCESSOR VIEW (IN ONE FRAME)
        # ----------------------------------------------------
        st.markdown('<span class="stage-pill">STAGE 2: PREPROCESSOR (SIDE-BY-SIDE TRANSFORMATION)</span>', unsafe_allow_html=True)
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            st.image(preproc_stages["original_rgb"], caption="1. Original ECG Input", use_container_width=True)
        with pcol2:
            st.image(preproc_stages["clahe"], caption="2. Contrast Enhanced (CLAHE)", use_container_width=True, clamp=True)
        with pcol3:
            st.image(preproc_stages["binary"], caption="3. Otsu Binarization", use_container_width=True, clamp=True)
        with pcol4:
            st.image(preproc_stages["skeleton"], caption="4. Morphological Skeleton Trace", use_container_width=True, clamp=True)

        st.markdown("---")

        # ----------------------------------------------------
        # STAGE 3 & 4: FEATURE EXTRACTION & VITAL SIGNS (IN ONE FRAME)
        # ----------------------------------------------------
        st.markdown('<span class="stage-pill">STAGE 3 & 4: FEATURE EXTRACTION & HEART RATE (BPM) ANALYSIS</span>', unsafe_allow_html=True)

        # Plot 1D Digitized ECG Signal with R-Peaks
        fig_ecg, ax_ecg = plt.subplots(figsize=(12, 3.2), dpi=150)
        signal_data = feat_results["signal_1d"]
        r_peaks = feat_results["r_peaks"]
        time_axis = np.arange(len(signal_data)) / float(sampling_rate)

        ax_ecg.plot(time_axis, signal_data, color="#0288d1", lw=1.5, label="Digitized ECG Waveform y(t)")
        if len(r_peaks) > 0:
            ax_ecg.plot(
                time_axis[r_peaks],
                signal_data[r_peaks],
                "ro",
                markersize=8,
                label=f"Detected R-Peaks ({len(r_peaks)} beats)",
            )
            # Annotate R-peaks with beat numbers
            for idx, p in enumerate(r_peaks):
                ax_ecg.annotate(
                    f"R{idx+1}",
                    (time_axis[p], signal_data[p]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#c62828",
                )

        ax_ecg.set_title("Reconstructed Continuous Cardiac Waveform & QRS Complex Detection", fontsize=11, fontweight="bold")
        ax_ecg.set_xlabel("Time (Seconds)", fontsize=9)
        ax_ecg.set_ylabel("Normalized Voltage Amplitude (mV)", fontsize=9)
        ax_ecg.grid(True, linestyle="--", alpha=0.5)
        ax_ecg.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_ecg)
        plt.close(fig_ecg)

        # Quantitative Metric Cards
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.metric("❤️ Heart Rate", f"{bpm_val:.1f} BPM", delta=f"{bpm_val - 75:.1f} vs normal 75")
        with mcol2:
            st.metric("⏱️ Mean RR Interval", f"{hr_metrics['mean_rr_ms']:.1f} ms")
        with mcol3:
            st.metric("📊 SDNN (HRV)", f"{hr_metrics['sdnn_ms']:.1f} ms")
        with mcol4:
            st.metric("🔬 RMSSD", f"{hr_metrics['rmssd_ms']:.1f} ms")
        with mcol5:
            st.metric("⚡ Shannon Entropy", f"{stats['signal_entropy']:.2f}")

        st.markdown("---")

        # ----------------------------------------------------
        # STAGE 5: ML PREDICTION & PERFORMANCE EVALUATION (IN ONE FRAME)
        # ----------------------------------------------------
        st.markdown('<span class="stage-pill">STAGE 5: MACHINE LEARNING DIAGNOSTIC EVALUATION</span>', unsafe_allow_html=True)

        res_col_left, res_col_right = st.columns([1.2, 1.0])

        with res_col_left:
            st.subheader(f"Diagnostic Assessment ({model_bundle['algorithm']})")
            
            # Clinical status banner
            if "Normal" in pred_label:
                st.markdown(
                    f"""
                    <div class="diag-box-normal">
                        ✅ DIAGNOSIS: {pred_label.upper()}
                        <div style="font-size: 13px; font-weight: normal; margin-top: 4px;">
                            Heart rate is within normal clinical limits (60–100 BPM) with stable beat-to-beat variability.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif "Bradycardia" in pred_label:
                st.markdown(
                    f"""
                    <div class="diag-box-warning">
                        ⚠️ DIAGNOSIS: {pred_label.upper()}
                        <div style="font-size: 13px; font-weight: normal; margin-top: 4px;">
                            Abnormally low heart rate detected (&lt; 60 BPM). Physiological during deep rest or indicative of sinus node dysfunction.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif "Tachycardia" in pred_label:
                st.markdown(
                    f"""
                    <div class="diag-box-warning">
                        ⚠️ DIAGNOSIS: {pred_label.upper()}
                        <div style="font-size: 13px; font-weight: normal; margin-top: 4px;">
                            Elevated heart rate detected (&gt; 100 BPM). Indicative of sinus tachycardia, stress, or sympathetic overactivity.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="diag-box-danger">
                        🚨 DIAGNOSIS: {pred_label.upper()}
                        <div style="font-size: 13px; font-weight: normal; margin-top: 4px;">
                            High beat-to-beat irregularity (elevated SDNN/RMSSD). Suggests cardiac arrhythmia or atrial fibrillation.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write(f"**Model Classification Confidence:** `{confidence:.2f}%`")
            st.progress(min(1.0, max(0.0, confidence / 100.0)))

            # Extracted Feature Vector Table
            st.write("**Extracted Feature Vector Fed to Classifier:**")
            feat_df = pd.DataFrame([feat_results["feature_dict"]])
            st.dataframe(feat_df.style.format("{:.3f}"), use_container_width=True)

        with res_col_right:
            st.subheader("Class Probability Distribution")
            prob_dict = pred_results["probabilities"]
            prob_df = pd.DataFrame({
                "Cardiac Rhythm": list(prob_dict.keys()),
                "Probability (%)": [v * 100 for v in prob_dict.values()]
            })

            fig_prob, ax_prob = plt.subplots(figsize=(6, 3.8), dpi=130)
            colors = ["#2e7d32" if "Normal" in c else "#ff9800" if ("Brady" in c or "Tachy" in c) else "#d32f2f" for c in prob_df["Cardiac Rhythm"]]
            bars = ax_prob.barh(prob_df["Cardiac Rhythm"], prob_df["Probability (%)"], color=colors, height=0.55)
            ax_prob.set_xlim(0, 100)
            ax_prob.set_xlabel("Confidence (%)", fontsize=9)
            ax_prob.grid(axis="x", linestyle="--", alpha=0.6)
            for bar in bars:
                w = bar.get_width()
                ax_prob.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_prob)
            plt.close(fig_prob)

    else:
        st.warning("Please upload an ECG image or select one of the pre-loaded test samples to begin analysis.")

# ==============================================================================
# TAB 2: IMAGE PREPROCESSING STUDIO
# ==============================================================================
with tab_preproc:
    st.header("🖼️ Image Preprocessing Laboratory")
    st.write("Examine each individual image processing transformation applied to extract clean cardiac signals from raw medical paper strips.")
    
    if active_image is not None:
        p_res = preprocess_pipeline(active_image)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("1. Original RGB Input")
            st.image(p_res["original_rgb"], use_container_width=True)
            st.caption("Raw ECG strip with paper grid lines and potential scanning grain.")
        with c2:
            st.subheader("2. Grayscale Conversion")
            st.image(p_res["grayscale"], use_container_width=True, clamp=True)
            st.caption("Single-channel luminance channel reduction (8-bit depth).")
            
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("3. Contrast Enhancement (CLAHE)")
            st.image(p_res["clahe"], use_container_width=True, clamp=True)
            st.caption("Contrast Limited Adaptive Histogram Equalization sharpens ECG trace against background.")
        with c4:
            st.subheader("4. Denoised Image")
            st.image(p_res["denoised"], use_container_width=True, clamp=True)
            st.caption("Gaussian/Median spatial filtering removes high-frequency scanner noise.")
            
        c5, c6 = st.columns(2)
        with c5:
            st.subheader("5. Otsu Adaptive Binarization")
            st.image(p_res["binary"], use_container_width=True, clamp=True)
            st.caption("Optimal binarization threshold separating foreground waveform from paper.")
        with c6:
            st.subheader("6. Morphological Skeletonization")
            st.image(p_res["skeleton"], use_container_width=True, clamp=True)
            st.caption("1-pixel width skeleton ready for 1D numerical signal coordinate conversion.")
    else:
        st.info("Select or upload an image in Tab 1 to see preprocessing studio results.")

# ==============================================================================
# TAB 3: FEATURE EXTRACTION & WAVEFORM ANALYSIS
# ==============================================================================
with tab_features:
    st.header("📈 1D Waveform Reconstruction & HRV Extraction")
    st.write("Detailed breakdown of peak detection, RR interval calculation, and Heart Rate Variability (HRV) feature engineering.")
    
    if active_image is not None:
        p_res = preprocess_pipeline(active_image)
        f_res = extract_all_features(p_res["skeleton"], sampling_rate=sampling_rate)
        
        # Waveform with interactive prominence
        sig = f_res["signal_1d"]
        peaks = f_res["r_peaks"]
        hr = f_res["hr_metrics"]
        
        fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
        t = np.arange(len(sig)) / float(sampling_rate)
        ax.plot(t, sig, color="#1565c0", label="Reconstructed Waveform", lw=1.5)
        ax.plot(t[peaks], sig[peaks], "rx", markersize=10, markeredgewidth=2.5, label="R-Peaks")
        ax.set_title(f"ECG Signal Trace ({len(peaks)} R-peaks detected | Calculated BPM: {hr['heart_rate_bpm']:.1f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (Seconds)")
        ax.set_ylabel("Normalized Voltage Amplitude")
        ax.grid(True, alpha=0.4)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        
        # Detailed RR interval breakdown table
        if len(hr["rr_intervals_sec"]) > 0:
            st.subheader("Beat-to-Beat RR Interval Timeline")
            rr_table = pd.DataFrame({
                "Beat Transition": [f"Beat {i+1} ➔ Beat {i+2}" for i in range(len(hr["rr_intervals_sec"]))],
                "RR Interval (seconds)": hr["rr_intervals_sec"],
                "RR Interval (milliseconds)": hr["rr_intervals_sec"] * 1000.0,
                "Instantaneous Heart Rate (BPM)": hr["instant_bpm"],
            })
            st.dataframe(rr_table.style.format({"RR Interval (seconds)": "{:.3f}", "RR Interval (milliseconds)": "{:.1f}", "Instantaneous Heart Rate (BPM)": "{:.1f}"}), use_container_width=True)
            
        # Statistical moments summary
        st.subheader("Calculated Statistical Moments & Non-Linear Features")
        stat_df = pd.DataFrame([f_res["stat_features"]]).T
        stat_df.columns = ["Value"]
        st.table(stat_df.style.format("{:.4f}"))
    else:
        st.info("Select or upload an image in Tab 1 to inspect waveform features.")

# ==============================================================================
# TAB 4: MODEL TRAINING & PERFORMANCE BENCHMARKS
# ==============================================================================
with tab_training:
    st.header("🏋️ Machine Learning Model Training & Evaluation Studio")
    st.write("Train and evaluate classifiers across 1,400+ clinical ECG feature samples. View Confusion Matrices, ROC Curves, and Accuracy metrics.")
    
    t_col1, t_col2 = st.columns([1, 2])
    
    with t_col1:
        st.subheader("Model Evaluation Summary")
        m = model_bundle["metrics"]
        st.metric("Model Architecture", model_bundle["algorithm"])
        st.metric("Test Accuracy", f"{m['accuracy'] * 100:.2f}%")
        st.metric("Macro Precision", f"{m['precision_macro'] * 100:.2f}%")
        st.metric("Macro Recall", f"{m['recall_macro'] * 100:.2f}%")
        st.metric("Macro F1-Score", f"{m['f1_macro'] * 100:.2f}%")
        st.caption(f"Evaluated on {int(model_bundle['n_samples'] * model_bundle['test_size'])} independent test cases.")

    with t_col2:
        st.subheader("Confusion Matrix")
        cm = model_bundle["confusion_matrix"]
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5), dpi=140)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[c.split()[0] for c in CLASS_NAMES],
            yticklabels=[c.split()[0] for c in CLASS_NAMES],
            ax=ax_cm,
            cbar=False
        )
        ax_cm.set_title(f"Confusion Matrix ({model_bundle['algorithm']})", fontsize=11, fontweight="bold")
        ax_cm.set_xlabel("Predicted Diagnosis", fontsize=10)
        ax_cm.set_ylabel("True Ground Truth", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # Multi-Class ROC Curves
    st.markdown("---")
    st.subheader("Multi-Class Receiver Operating Characteristic (ROC) Curves")
    roc_data = model_bundle["roc_data"]
    if roc_data:
        fig_roc, ax_roc = plt.subplots(figsize=(9, 4.2), dpi=140)
        palette = ["#2e7d32", "#1976d2", "#f57c00", "#d32f2f"]
        for idx, (c_name, c_roc) in enumerate(roc_data.items()):
            ax_roc.plot(
                c_roc["fpr"],
                c_roc["tpr"],
                label=f"{c_name} (AUC = {c_roc['auc']:.3f})",
                color=palette[idx % len(palette)],
                lw=2
            )
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.50)")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
        ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
        ax_roc.set_title("One-vs-Rest Multi-Class ROC Curves & Area Under Curve (AUC)", fontsize=11, fontweight="bold")
        ax_roc.grid(True, linestyle="--", alpha=0.5)
        ax_roc.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    # Feature Importance Plot
    if model_bundle["feature_importances"]:
        st.markdown("---")
        st.subheader("Feature Importance Ranking (Gini / Weight Impact)")
        fi_dict = model_bundle["feature_importances"]
        fi_df = pd.DataFrame({
            "Feature": list(fi_dict.keys()),
            "Importance": list(fi_dict.values())
        }).sort_values(by="Importance", ascending=True)

        fig_fi, ax_fi = plt.subplots(figsize=(8, 3.8), dpi=140)
        ax_fi.barh(fi_df["Feature"], fi_df["Importance"], color="#00838f", height=0.6)
        ax_fi.set_xlabel("Relative Importance Weight", fontsize=9)
        ax_fi.set_title("Which extracted ECG features drive the ML diagnosis?", fontsize=10, fontweight="bold")
        ax_fi.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close(fig_fi)

# ==============================================================================
# TAB 5: CLINICAL FORMULAS & MEDICAL REFERENCE
# ==============================================================================
with tab_docs:
    st.header("🩺 Clinical Diagnostic Guide & Mathematical Formulations")
    
    st.markdown(
        r"""
        ### 1. Heart Rate (BPM) Calculation
        Heart rate measures the number of ventricular contractions per minute. When analyzing ECG strips, it is computed from the consecutive interval between $R$-peaks ($RR$ intervals):
        
        $$\text{Heart Rate (BPM)} = \frac{60}{\overline{RR} \text{ (seconds)}} = \frac{60 \times f_s}{\overline{\Delta x_R}}$$
        
        Where:
        - $f_s$: Sampling frequency in Hertz (Hz)
        - $\overline{RR}$: Average duration between consecutive R-peaks in seconds
        - $\overline{\Delta x_R}$: Average horizontal pixel distance between consecutive R-peaks

        ---

        ### 2. Clinical Rhythm Categories
        | Diagnostic Class | Heart Rate (BPM) | Clinical Hallmark & Pathophysiology |
        | :--- | :--- | :--- |
        | **Normal Sinus Rhythm** | $60 \le \text{BPM} \le 100$ | Regular P-Q-R-S-T morphology, pacemaker origin in SA node. |
        | **Sinus Bradycardia** | $\text{BPM} < 60$ | Prolonged RR interval ($>1000\text{ ms}$). Normal in trained athletes or pathological in nodal disease. |
        | **Sinus Tachycardia** | $\text{BPM} > 100$ | Shortened RR interval ($<600\text{ ms}$). Physiological during exercise/fever or pathological. |
        | **Cardiac Arrhythmia** | Variable | Marked beat-to-beat variability, elevated SDNN & RMSSD, abnormal entropy. |

        ---

        ### 3. Heart Rate Variability (HRV) Formulas
        - **SDNN** (Standard Deviation of NN/RR Intervals):
          $$\text{SDNN} = \sqrt{\frac{1}{N - 1} \sum_{i=1}^{N} (RR_i - \overline{RR})^2}$$
        - **RMSSD** (Root Mean Square of Successive Differences):
          $$\text{RMSSD} = \sqrt{\frac{1}{N - 1} \sum_{i=1}^{N-1} (RR_{i+1} - RR_i)^2}$$
        - **pNN50**: Percentage of adjacent RR intervals differing by $> 50\text{ ms}$.

        ---

        ### 4. Image Preprocessing & Otsu's Thresholding
        Otsu's thresholding automatically finds the binarization threshold $t^*$ that maximizes between-class variance $\sigma_B^2(t)$:
        
        $$\sigma_B^2(t) = \omega_0(t) \omega_1(t) \big[\mu_0(t) - \mu_1(t)\big]^2$$
        """
    )

# ==============================================================================
# TAB 6: CODE & EXECUTION MANUAL
# ==============================================================================
with tab_code:
    st.header("💻 How to Compile, Configure, and Run This Project")
    
    st.markdown(
        """
        ### Step-by-Step Execution Guide

        #### 1. Open Terminal or Command Prompt
        Navigate into the project directory:
        ```bash
        cd C:\\Users\\acer\\.gemini\\antigravity\\scratch\\heart_rate_ml_app
        ```

        #### 2. Create and Activate Virtual Environment (Recommended)
        ```bash
        python -m venv venv
        # On Windows:
        .\\venv\\Scripts\\activate
        ```

        #### 3. Install Dependencies
        ```bash
        pip install -r requirements.txt
        ```

        #### 4. Generate Sample ECG Images & Baseline Model
        ```bash
        python generate_samples.py
        ```

        #### 5. Launch the Web Application (One-Frame Interface)
        ```bash
        streamlit run app.py
        ```
        The dashboard will automatically open in your default browser at `http://localhost:8501`.

        #### 6. Run from Terminal (CLI Mode)
        If you want to run the program directly in terminal:
        ```bash
        # Analyze an ECG image
        python cli_run.py --image sample_images/normal_ecg.png
        python cli_run.py --image sample_images/tachycardia_ecg.png
        python cli_run.py --image sample_images/bradycardia_ecg.png
        python cli_run.py --image sample_images/arrhythmia_ecg.png

        # Train a new model algorithm
        python cli_run.py --train --algo "Support Vector Machine"
        ```
        """
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #78909c; font-size: 13px;">
        CardioVision ML Research System • Medical Machine Learning & Computer Vision Pipeline • Python 3.x • Streamlit • Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True,
)
