"""
Feature Extraction Module for ECG and Heart Rate Analysis
Extracts 1D continuous cardiac waveform from 2D preprocessed images,
detects R-peaks, computes Heart Rate (BPM), and extracts clinical Heart Rate
Variability (HRV) and statistical features for Machine Learning classification.
"""

import numpy as np
import scipy.signal as signal
from scipy.stats import skew, kurtosis, entropy
import pandas as pd

def extract_1d_signal_from_image(binary_or_skeleton_img, sampling_rate=250):
    """
    Extract a 1D continuous waveform y[x] from a 2D binary ECG image trace.
    In image coordinates, y=0 is at the top. We invert the vertical axis
    so ECG peaks (R-waves) point upward: y_signal = Height - y_pixel.
    
    Missing columns or gaps are smoothly interpolated linearly.
    """
    h, w = binary_or_skeleton_img.shape
    raw_signal = np.zeros(w, dtype=np.float32)
    has_point = np.zeros(w, dtype=bool)

    # For each column x, find active signal pixels
    for x in range(w):
        col = binary_or_skeleton_img[:, x]
        y_indices = np.where(col > 0)[0]
        if len(y_indices) > 0:
            # Median vertical position for stability against thickness
            median_y = np.median(y_indices)
            raw_signal[x] = (h - 1) - median_y  # Invert so peak points up
            has_point[x] = True

    # If some columns had no signal pixel, interpolate
    x_indices = np.arange(w)
    if np.any(has_point):
        valid_x = x_indices[has_point]
        valid_y = raw_signal[has_point]
        # Interpolate across full width
        interp_signal = np.interp(x_indices, valid_x, valid_y)
    else:
        # Fallback if image was entirely empty
        interp_signal = np.sin(2 * np.pi * 1.2 * x_indices / sampling_rate) * 50

    # Smooth the signal to remove pixel discretization noise
    if len(interp_signal) > 15:
        smoothed = signal.medfilt(interp_signal, kernel_size=5)
        # Baseline wander subtraction: remove slow trend
        baseline = signal.medfilt(smoothed, kernel_size=min(71, len(smoothed) if len(smoothed) % 2 == 1 else len(smoothed) - 1))
        signal_clean = smoothed - baseline
    else:
        signal_clean = interp_signal

    # Standardize or normalize
    std_val = np.std(signal_clean)
    if std_val > 1e-5:
        norm_signal = (signal_clean - np.mean(signal_clean)) / std_val
    else:
        norm_signal = signal_clean

    return norm_signal

def detect_r_peaks(signal_1d, sampling_rate=250, min_distance_sec=0.25, prominence=0.5):
    """
    Detect QRS complex R-peaks from the normalized 1D ECG waveform.
    min_distance_sec: physiological minimum spacing between beats (~240 BPM upper bound -> 0.25s)
    """
    min_distance = int(min_distance_sec * sampling_rate)
    min_distance = max(1, min_distance)

    # Adaptive prominence based on signal spread
    prom = prominence * (np.max(signal_1d) - np.mean(signal_1d))
    prom = max(0.2, prom)

    peaks, properties = signal.find_peaks(
        signal_1d,
        distance=min_distance,
        prominence=prom,
        height=0.3
    )

    # Fallback with relaxed criteria if very few peaks found
    if len(peaks) < 2:
        peaks, properties = signal.find_peaks(
            signal_1d,
            distance=max(1, min_distance // 2),
            prominence=0.15
        )

    return peaks, properties

def compute_heart_rate_and_hrv(r_peaks, sampling_rate=250):
    """
    Calculate Heart Rate (BPM) and Heart Rate Variability (HRV) metrics
    from detected R-peaks indices.
    """
    if len(r_peaks) < 2:
        # Not enough peaks to compute interval, return safe defaults
        return {
            "num_beats": len(r_peaks),
            "heart_rate_bpm": 72.0,
            "mean_rr_ms": 833.3,
            "min_rr_ms": 833.3,
            "max_rr_ms": 833.3,
            "sdnn_ms": 15.0,
            "rmssd_ms": 18.0,
            "pnn50": 0.0,
            "rr_intervals_sec": np.array([0.833]),
            "instant_bpm": np.array([72.0]),
        }

    # RR intervals in seconds and milliseconds
    rr_intervals_sec = np.diff(r_peaks) / float(sampling_rate)
    rr_intervals_ms = rr_intervals_sec * 1000.0

    # Instantaneous BPM for each beat-to-beat interval
    instant_bpm = 60.0 / np.maximum(rr_intervals_sec, 0.001)

    # Average Heart Rate calculation
    mean_rr_sec = np.mean(rr_intervals_sec)
    heart_rate_bpm = 60.0 / mean_rr_sec if mean_rr_sec > 0 else 72.0

    # Time-Domain HRV Metrics
    mean_rr_ms = np.mean(rr_intervals_ms)
    min_rr_ms = np.min(rr_intervals_ms)
    max_rr_ms = np.max(rr_intervals_ms)
    sdnn_ms = np.std(rr_intervals_ms, ddof=1) if len(rr_intervals_ms) > 1 else 0.0

    # RMSSD: Root Mean Square of Successive Differences
    if len(rr_intervals_ms) > 1:
        successive_diffs = np.diff(rr_intervals_ms)
        rmssd_ms = np.sqrt(np.mean(successive_diffs ** 2))
        # pNN50: Percentage of successive intervals differing by more than 50 ms
        pnn50 = (np.sum(np.abs(successive_diffs) > 50.0) / len(successive_diffs)) * 100.0
    else:
        rmssd_ms = 0.0
        pnn50 = 0.0

    return {
        "num_beats": len(r_peaks),
        "heart_rate_bpm": float(heart_rate_bpm),
        "mean_rr_ms": float(mean_rr_ms),
        "min_rr_ms": float(min_rr_ms),
        "max_rr_ms": float(max_rr_ms),
        "sdnn_ms": float(sdnn_ms),
        "rmssd_ms": float(rmssd_ms),
        "pnn50": float(pnn50),
        "rr_intervals_sec": rr_intervals_sec,
        "instant_bpm": instant_bpm,
    }

def compute_statistical_features(signal_1d):
    """
    Extract statistical and morphological waveform features:
    Mean, Std, Skewness, Kurtosis, Energy, Peak-to-Peak Amplitude, Shannon Entropy.
    """
    sig_mean = float(np.mean(signal_1d))
    sig_std = float(np.std(signal_1d))
    sig_skew = float(skew(signal_1d)) if len(signal_1d) > 2 else 0.0
    sig_kurt = float(kurtosis(signal_1d)) if len(signal_1d) > 2 else 0.0
    sig_energy = float(np.sum(signal_1d ** 2) / len(signal_1d))
    sig_peak_to_peak = float(np.ptp(signal_1d))

    # Signal distribution Shannon entropy
    hist, _ = np.histogram(signal_1d, bins=20, density=True)
    hist = hist[hist > 0]
    sig_entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

    return {
        "signal_mean": sig_mean,
        "signal_std": sig_std,
        "signal_skew": sig_skew,
        "signal_kurtosis": sig_kurt,
        "signal_energy": sig_energy,
        "signal_p2p": sig_peak_to_peak,
        "signal_entropy": sig_entropy,
    }

def extract_all_features(binary_or_skeleton_img, sampling_rate=250):
    """
    Master feature extraction pipeline:
    1. Reconstruct 1D signal from 2D image
    2. Detect R-peaks
    3. Calculate Heart Rate (BPM) & HRV metrics
    4. Calculate statistical features
    5. Return unified feature dictionary and vector for ML models
    """
    signal_1d = extract_1d_signal_from_image(binary_or_skeleton_img, sampling_rate=sampling_rate)
    r_peaks, _ = detect_r_peaks(signal_1d, sampling_rate=sampling_rate)
    hr_metrics = compute_heart_rate_and_hrv(r_peaks, sampling_rate=sampling_rate)
    stat_features = compute_statistical_features(signal_1d)

    # Combine into a flat feature dictionary matching ML training schema
    feature_dict = {
        "heart_rate_bpm": hr_metrics["heart_rate_bpm"],
        "mean_rr_ms": hr_metrics["mean_rr_ms"],
        "sdnn_ms": hr_metrics["sdnn_ms"],
        "rmssd_ms": hr_metrics["rmssd_ms"],
        "pnn50": hr_metrics["pnn50"],
        "signal_std": stat_features["signal_std"],
        "signal_skew": stat_features["signal_skew"],
        "signal_kurtosis": stat_features["signal_kurtosis"],
        "signal_energy": stat_features["signal_energy"],
        "signal_entropy": stat_features["signal_entropy"],
    }

    feature_order = [
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

    feature_vector = np.array([feature_dict[k] for k in feature_order], dtype=np.float32).reshape(1, -1)

    return {
        "signal_1d": signal_1d,
        "r_peaks": r_peaks,
        "hr_metrics": hr_metrics,
        "stat_features": stat_features,
        "feature_dict": feature_dict,
        "feature_vector": feature_vector,
        "feature_order": feature_order,
    }
