"""
Sample ECG Image & Pretrained Model Generator
Generates realistic medical ECG paper strip images (Lead II) for testing:
1. normal_ecg.png (~75 BPM)
2. bradycardia_ecg.png (~48 BPM)
3. tachycardia_ecg.png (~130 BPM)
4. arrhythmia_ecg.png (irregular RR intervals)

Also trains and caches the baseline Machine Learning model.
"""

import os
import numpy as np
import cv2
from model_trainer import train_and_evaluate_model, save_model

def draw_ecg_grid(width=1000, height=300):
    """
    Draw a standard medical ECG paper grid:
    Pink/Peach background with fine 1mm gridlines and bold 5mm major gridlines.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    # Light pink base
    img[:, :] = (245, 245, 255)  # BGR order: slight pinkish hue (R high, B/G slightly lower)
    img[:, :, 0] = 235  # B
    img[:, :, 1] = 235  # G
    img[:, :, 2] = 255  # R

    # Minor grid lines (every 10 pixels)
    for x in range(0, width, 10):
        cv2.line(img, (x, 0), (x, height), (210, 210, 250), 1)
    for y in range(0, height, 10):
        cv2.line(img, (0, y), (width, y), (210, 210, 250), 1)

    # Major grid lines (every 50 pixels)
    for x in range(0, width, 50):
        cv2.line(img, (x, 0), (x, height), (170, 170, 240), 1)
    for y in range(0, height, 50):
        cv2.line(img, (0, y), (width, y), (170, 170, 240), 1)

    return img

def synthesize_ecg_beat(t_norm):
    """
    Synthesize one normalized P-Q-R-S-T cardiac cycle between t in [0, 1].
    t_norm is beat-relative phase from 0 to 1.
    """
    val = 0.0
    # P-wave: gentle atrial depolarization hump around 0.15 - 0.25
    val += 0.20 * np.exp(-((t_norm - 0.20) ** 2) / (2 * (0.03 ** 2)))
    # Q-wave: slight negative dip before R
    val -= 0.15 * np.exp(-((t_norm - 0.36) ** 2) / (2 * (0.015 ** 2)))
    # R-peak: sharp high-voltage ventricular spike
    val += 1.80 * np.exp(-((t_norm - 0.40) ** 2) / (2 * (0.012 ** 2)))
    # S-wave: sharp negative deflection right after R
    val -= 0.35 * np.exp(-((t_norm - 0.44) ** 2) / (2 * (0.015 ** 2)))
    # T-wave: broader ventricular repolarization wave around 0.65
    val += 0.35 * np.exp(-((t_norm - 0.65) ** 2) / (2 * (0.05 ** 2)))
    return val

def generate_ecg_strip(bpm=75, is_arrhythmia=False, width=1000, height=300, sampling_rate=250):
    """
    Generate an ECG strip image with specified heart rate or arrhythmia irregularity.
    """
    grid = draw_ecg_grid(width, height)
    baseline_y = height // 2 + 30
    voltage_scale = 55.0  # pixels per unit amplitude

    signal_points = []
    total_samples = width

    # Determine beat timings
    if not is_arrhythmia:
        rr_sec = 60.0 / bpm
        rr_samples = int(rr_sec * sampling_rate)
        # Sequence of R-peak centers
        beat_starts = list(range(20, total_samples + rr_samples, rr_samples))
    else:
        # Arrhythmia: irregular intervals, premature ectopic beats
        beat_starts = [15]
        intervals = [120, 240, 100, 280, 110, 220, 90, 260]  # highly irregular pixel intervals
        idx = 0
        while beat_starts[-1] < total_samples - 50:
            step = intervals[idx % len(intervals)]
            beat_starts.append(beat_starts[-1] + step)
            idx += 1

    # Compute amplitude for each pixel x along time
    for x in range(total_samples):
        # Find which beat cycle x belongs to
        current_beat_idx = -1
        for i in range(len(beat_starts) - 1):
            if beat_starts[i] <= x < beat_starts[i + 1]:
                current_beat_idx = i
                break

        if current_beat_idx != -1:
            start_x = beat_starts[current_beat_idx]
            end_x = beat_starts[current_beat_idx + 1]
            period = max(1, end_x - start_x)
            phase = (x - start_x) / float(period)
            amp = synthesize_ecg_beat(phase)
        else:
            amp = 0.0

        # Small physiological baseline noise
        noise = np.random.normal(0, 0.02)
        total_amp = amp + noise
        y_pixel = int(baseline_y - (total_amp * voltage_scale))
        y_pixel = np.clip(y_pixel, 5, height - 6)
        signal_points.append((x, y_pixel))

    # Draw continuous ECG trace (dark medical black/charcoal line)
    for i in range(len(signal_points) - 1):
        pt1 = signal_points[i]
        pt2 = signal_points[i + 1]
        cv2.line(grid, pt1, pt2, (20, 20, 25), 2, cv2.LINE_AA)

    return grid

def generate_all_sample_assets(output_dir="sample_images", model_dir="models"):
    """Generate all sample ECG images and train default model bundle."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    samples = [
        ("normal_ecg.png", 75, False, "Normal Sinus Rhythm (~75 BPM)"),
        ("bradycardia_ecg.png", 48, False, "Sinus Bradycardia (~48 BPM)"),
        ("tachycardia_ecg.png", 130, False, "Sinus Tachycardia (~130 BPM)"),
        ("arrhythmia_ecg.png", 85, True, "Cardiac Arrhythmia (Irregular RR)"),
    ]

    print("Generating sample ECG images...")
    for filename, bpm, is_arrhythmia, desc in samples:
        path = os.path.join(output_dir, filename)
        img = generate_ecg_strip(bpm=bpm, is_arrhythmia=is_arrhythmia)
        cv2.imwrite(path, img)
        print(f"  [OK] Saved {filename} - {desc}")

    print("\nTraining and persisting baseline Random Forest model...")
    bundle = train_and_evaluate_model(algorithm="Random Forest", n_estimators=100)
    model_path = os.path.join(model_dir, "trained_ecg_model.joblib")
    save_model(bundle, model_path)
    print(f"  [OK] Model saved to {model_path} (Accuracy: {bundle['metrics']['accuracy']*100:.2f}%)")

if __name__ == "__main__":
    generate_all_sample_assets()
