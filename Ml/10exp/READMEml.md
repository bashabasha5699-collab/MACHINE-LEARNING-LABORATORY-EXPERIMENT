# ❤️ CardioVision ML: Medical Heart Rate Analysis & ML Pipeline

An end-to-end medical machine learning application for clinical heart rate estimation and cardiac rhythm diagnosis from ECG paper images and signal strips. 

All 5 core stages—**Load Input Image**, **Preprocessor**, **Feature Extraction**, **Model Training**, and **Performance Evaluation**—are unified in a **single-frame interactive web application** powered by Streamlit and Scikit-Learn.

---

## 📑 Table of Contents
1. [Key Features](#-key-features)
2. [End-to-End Pipeline Architecture](#-end-to-end-pipeline-architecture)
3. [Project Directory Structure](#-project-directory-structure)
4. [Mathematical & Clinical Formulation](#-mathematical--clinical-formulation)
5. [Prerequisites & System Requirements](#-prerequisites--system-requirements)
6. [How to Compile and Run the Code](#-how-to-compile-and-run-the-code)
7. [Running in Terminal (CLI Mode)](#-running-in-terminal-cli-mode)
8. [Machine Learning Algorithms Supported](#-machine-learning-algorithms-supported)
9. [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🌟 Key Features

1. **Load Input Image**:
   - Supports medical ECG paper scans and synthetic strips in PNG, JPG, JPEG, and BMP.
   - Built-in one-click sample gallery for **Normal Sinus Rhythm**, **Sinus Bradycardia**, **Sinus Tachycardia**, and **Cardiac Arrhythmia**.

2. **Preprocessor**:
   - Grayscale conversion and 8-bit channel reduction.
   - Contrast enhancement via **CLAHE** (Contrast Limited Adaptive Histogram Equalization).
   - Noise reduction (Gaussian, Median, and Bilateral spatial filtering) to suppress ECG grid grain.
   - **Otsu's Optimal Binarization** to separate signal waveform from background.
   - **Morphological Skeletonization** to thin the signal into a continuous 1-pixel-wide path.

3. **Feature Extraction**:
   - 2D-to-1D continuous voltage waveform reconstruction $y(t)$.
   - Baseline wander correction and smoothing.
   - **QRS Complex & R-Peak Detection** with adaptive prominence and physiological constraints.
   - **Heart Rate (BPM)** estimation from $RR$ intervals:
     $$\text{BPM} = \frac{60 \times f_s}{\overline{\Delta x_R}}$$
   - Clinical **Heart Rate Variability (HRV)** metrics: Mean RR, Min/Max RR, **SDNN**, **RMSSD**, and **pNN50**.
   - Higher-order statistical moments: Variance, Skewness, Kurtosis, Energy, and Shannon Entropy.

4. **Model Training**:
   - Multi-algorithm training suite: **Random Forest**, **Support Vector Machine (SVM)**, **K-Nearest Neighbors (KNN)**, **Decision Tree**, and **Logistic Regression**.
   - Built-in calibrated clinical dataset of 1,400+ samples reflecting true physiological distributions.
   - Hyperparameter sliders (number of trees, tree depth, SVM regularization $C$, KNN $k$).
   - Automatic model persistence (`.joblib`).

5. **Performance Evaluation**:
   - Real-time diagnostic prediction on uploaded images with confidence percentage.
   - Comprehensive model evaluation:
     - **Accuracy**, **Macro Precision**, **Recall**, and **F1-Score**.
     - **Confusion Matrix** heatmap visualization.
     - Multi-Class **Receiver Operating Characteristic (ROC)** curves and Area Under Curve (AUC).
     - **Feature Importance** ranking chart.

6. **Unified Single-Frame Web Interface**:
   - Everything visible side-by-side on one unified screen: image input, preprocessing filters, reconstructed waveform with marked peaks, vital metric cards, and ML diagnosis with probability bars.

---

## 📐 End-to-End Pipeline Architecture

```
+--------------------+
|  Input ECG Image   | (Medical paper strip / file upload)
+---------+----------+
          |
          v
+--------------------+
|   Preprocessing    | Grayscale ➔ CLAHE ➔ Gaussian Denoising ➔ Otsu Binarization ➔ Skeletonization
+---------+----------+
          |
          v
+--------------------+
| Feature Extraction | 2D➔1D Waveform Digitization ➔ R-Peak Detection ➔ BPM & HRV (SDNN/RMSSD)
+---------+----------+
          |
          v
+--------------------+
| Model Training /   | Random Forest / SVM / KNN / Decision Tree
| Diagnostic Predict | ➔ Normal Rhythm / Bradycardia / Tachycardia / Arrhythmia
+---------+----------+
          |
          v
+--------------------+
| Evaluation Metrics | Confusion Matrix Heatmap, Multi-Class ROC Curves, Accuracy & F1-Score
+--------------------+
```

---

## 📂 Project Directory Structure

```
heart_rate_ml_app/
│
├── app.py                   # Main Streamlit single-frame web application
├── preprocessing.py          # Image processing: Grayscale, CLAHE, Otsu, Skeletonization
├── feature_extraction.py     # 1D signal digitization, R-peak detection, BPM & HRV metrics
├── model_trainer.py          # ML algorithms, training, testing, ROC & Confusion Matrix
├── generate_samples.py       # Generates sample ECG strip images & baseline trained model
├── cli_run.py                # Command-line interface for terminal execution
├── requirements.txt          # Python library dependencies
├── README.md                 # Project documentation and manual
│
├── sample_images/            # Pre-generated ECG sample images
│   ├── normal_ecg.png        # Normal sinus rhythm (~75 BPM)
│   ├── bradycardia_ecg.png   # Sinus bradycardia (~48 BPM)
│   ├── tachycardia_ecg.png   # Sinus tachycardia (~130 BPM)
│   └── arrhythmia_ecg.png    # Cardiac arrhythmia (irregular RR)
│
└── models/                   # Persisted trained model artifacts (.joblib)
    └── trained_ecg_model.joblib
```

---

## 🔬 Mathematical & Clinical Formulation

### 1. Heart Rate (BPM) Calculation
Given sampling frequency $f_s$ (Hz) and consecutive detected R-peaks $R = [r_1, r_2, \dots, r_N]$:
$$\Delta t_i = \frac{r_{i+1} - r_i}{f_s} \quad \text{(seconds)}$$

$$\overline{RR} = \frac{1}{N - 1} \sum_{i=1}^{N - 1} \Delta t_i$$

$$\text{Heart Rate (BPM)} = \frac{60}{\overline{RR}}$$

### 2. Heart Rate Variability (HRV)
- **SDNN** (Standard Deviation of NN/RR intervals):
  $$\text{SDNN} = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(RR_i - \overline{RR})^2}$$
- **RMSSD** (Root Mean Square of Successive Differences):
  $$\text{RMSSD} = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N-1}(RR_{i+1} - RR_i)^2}$$
- **pNN50**: Percentage of adjacent RR intervals that differ by more than 50 milliseconds.

### 3. Clinical Diagnostic Thresholds
- **Normal Sinus Rhythm**: $60 \le \text{BPM} \le 100$, regular rhythm ($\text{SDNN} < 65\text{ ms}$).
- **Sinus Bradycardia**: $\text{BPM} < 60$.
- **Sinus Tachycardia**: $\text{BPM} > 100$.
- **Cardiac Arrhythmia**: Irregular rhythm characterized by elevated $\text{SDNN} > 70\text{ ms}$ and $\text{RMSSD} > 60\text{ ms}$.

---

## ⚙️ Prerequisites & System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python**: Python 3.8, 3.9, 3.10, 3.11, or 3.12 installed on your system.
  - To check your version, run: `python --version`

---

## 🚀 How to Compile and Run the Code

Follow these exact steps in your Windows PowerShell or Command Prompt:

### Step 1: Open Terminal and Navigate to Project Directory
```powershell
cd C:\Users\acer\.gemini\antigravity\scratch\heart_rate_ml_app
```

### Step 2: Create a Virtual Environment (Recommended)
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
- **On Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an Execution Policy error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*
- **On Windows (Command Prompt `cmd`)**:
  ```cmd
  venv\Scripts\activate.bat
  ```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Generate Sample Images & Baseline Model
Run the generator script to create sample ECG images and pre-train the default model:
```powershell
python generate_samples.py
```
*Output:*
```
Generating sample ECG images...
  [OK] Saved normal_ecg.png - Normal Sinus Rhythm (~75 BPM)
  [OK] Saved bradycardia_ecg.png - Sinus Bradycardia (~48 BPM)
  [OK] Saved tachycardia_ecg.png - Sinus Tachycardia (~130 BPM)
  [OK] Saved arrhythmia_ecg.png - Cardiac Arrhythmia (Irregular RR)

Training and persisting baseline Random Forest model...
  [OK] Model saved to models/trained_ecg_model.joblib (Accuracy: 98.21%)
```

### Step 6: Launch the Web Application
```powershell
streamlit run app.py
```
Your default browser will immediately open to:
```
http://localhost:8501
```

---

## 💻 Running in Terminal (CLI Mode)

You can also run analysis and training directly from the command line without opening the browser:

### 1. Analyze an ECG Image via CLI
```powershell
python cli_run.py --image sample_images/normal_ecg.png
```
```powershell
python cli_run.py --image sample_images/tachycardia_ecg.png
```
```powershell
python cli_run.py --image sample_images/bradycardia_ecg.png
```
```powershell
python cli_run.py --image sample_images/arrhythmia_ecg.png
```

### 2. Train a Specific Machine Learning Algorithm via CLI
```powershell
# Train Random Forest
python cli_run.py --train --algo "Random Forest"

# Train Support Vector Machine
python cli_run.py --train --algo "Support Vector Machine"

# Train K-Nearest Neighbors
python cli_run.py --train --algo "KNN"

# Train Decision Tree
python cli_run.py --train --algo "Decision Tree"
```

---

## 🧠 Machine Learning Algorithms Supported

| Algorithm | Strengths in ECG Heart Rate Classification | Key Tunable Parameters |
| :--- | :--- | :--- |
| **Random Forest** | Ensemble of decision trees; handles non-linear relationships and resists overfitting. High precision on HRV features. | `n_estimators`, `max_depth` |
| **Support Vector Machine (SVM)** | Effective in high-dimensional feature spaces with RBF kernel; creates optimal separating hyperplanes. | Regularization `C`, `kernel` |
| **K-Nearest Neighbors (KNN)** | Non-parametric instance-based classifier; identifies rhythm similarities in feature distance space. | Number of neighbors `k` |
| **Decision Tree** | Highly interpretable white-box clinical decision paths. | `max_depth` |
| **Logistic Regression** | Baseline linear model; provides calibrated class probabilities. | `max_iter` |

---

## ❓ Troubleshooting & FAQs

1. **`python: The term 'python' is not recognized`**:
   - Ensure Python is installed and checked "Add Python to PATH" during installation.
   - If installed via Microsoft Store, run `python3` or `py`.

2. **PowerShell `Script Execution Disabled`**:
   - Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

3. **Port 8501 is already in use**:
   - Run Streamlit on an alternative port:
     ```powershell
     streamlit run app.py --server.port 8502
     ```
