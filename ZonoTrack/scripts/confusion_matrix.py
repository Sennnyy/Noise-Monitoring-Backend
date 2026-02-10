"""
Confusion Matrix Visualization for ZonoTrack K-Means Model

Generates a visual confusion matrix and displays model evaluation metrics.

Run this script after training the model with /api/train endpoint.
"""

import os
import math
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ============================================
# CONSTANTS (same as app.py)
# ============================================
SAMPLE_RATE = 22050
NUM_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512
NUM_SEGMENTS = 10
N_CLUSTERS = 256

# Base directory (parent of scripts folder = ZonoTrack root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
KMEANS_PATH = os.path.join(MODELS_DIR, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
CLUSTER_LABELS_PATH = os.path.join(MODELS_DIR, 'cluster_labels.npy')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')


def extract_mfcc_features(file_path):
    """Extract MFCC features from audio file."""
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=NUM_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    mfccs = mfccs.T
    if mfccs.shape[0] < NUM_SEGMENTS:
        mfccs = np.pad(mfccs, ((0, NUM_SEGMENTS - mfccs.shape[0]), (0, 0)))
    else:
        mfccs = mfccs[:NUM_SEGMENTS]
    
    return mfccs.flatten()


def euclidean_distance(x, centroid):
    """Compute Euclidean distance between feature vector and centroid."""
    squared_diff = 0.0
    for i in range(len(x)):
        diff = x[i] - centroid[i]
        squared_diff += diff * diff
    return math.sqrt(squared_diff)


def compute_all_distances(x, centroids):
    """Compute distances from feature vector to all centroids."""
    return [euclidean_distance(x, c) for c in centroids]


def find_nearest_cluster(distances):
    """Find the cluster with minimum distance."""
    min_distance = distances[0]
    min_index = 0
    for i in range(1, len(distances)):
        if distances[i] < min_distance:
            min_distance = distances[i]
            min_index = i
    return min_index, min_distance


def main():
    print("=" * 60)
    print(" ZonoTrack K-Means Confusion Matrix Visualization")
    print("=" * 60)
    
    # Load model
    if not os.path.exists(KMEANS_PATH) or not os.path.exists(SCALER_PATH):
        print("\nError: Model not found. Run /api/train first.")
        return
    
    print("\nLoading model...")
    kmeans_model = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)
    cluster_to_label = np.load(CLUSTER_LABELS_PATH, allow_pickle=True).item()
    
    print("Evaluating on dataset...")
    
    # Collect predictions
    y_true = []
    y_pred = []
    
    total_files = 0
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                total_files += 1
    
    processed = 0
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                file_path = os.path.join(root, file)
                true_label = os.path.basename(root)
                
                try:
                    features = extract_mfcc_features(file_path)
                    features_scaled = scaler.transform([features])[0]
                    
                    distances = compute_all_distances(features_scaled, kmeans_model.cluster_centers_)
                    cluster_idx, _ = find_nearest_cluster(distances)
                    
                    if cluster_idx in cluster_to_label:
                        pred_label = cluster_to_label[cluster_idx]
                    else:
                        pred_label = f"cluster_{cluster_idx}"
                    
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                    
                    processed += 1
                    if processed % 100 == 0:
                        print(f"  Processed {processed}/{total_files} files...")
                        
                except Exception as e:
                    print(f"  Error: {file}: {e}")
    
    print(f"\nTotal samples evaluated: {len(y_true)}")
    
    # Get labels
    all_labels = sorted(list(set(y_true + y_pred)))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Print metrics
    print("\n" + "=" * 60)
    print(" MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"\n  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Per-class metrics
    print("\n" + "-" * 60)
    print(" PER-CLASS METRICS")
    print("-" * 60)
    per_class_precision = precision_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    
    print(f"\n  {'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("  " + "-" * 66)
    for i, label in enumerate(all_labels):
        support = sum(1 for y in y_true if y == label)
        print(f"  {label:<20} {per_class_precision[i]:<12.4f} {per_class_recall[i]:<12.4f} {per_class_f1[i]:<12.4f} {support:<10}")
    
    # Create results directory
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    
    # Normalized confusion matrix (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=all_labels,
        yticklabels=all_labels,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Prediction Rate'}
    )
    
    plt.title(f'K-Means Sound Classification Confusion Matrix\n'
              f'Accuracy: {accuracy*100:.2f}% | F1 Score: {f1:.4f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(RESULTS_PATH, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Confusion matrix saved to: {output_path}")
    
    # Also create a raw counts version
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=all_labels,
        yticklabels=all_labels,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(f'K-Means Sound Classification Confusion Matrix (Counts)\n'
              f'Total Samples: {len(y_true)}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path_counts = os.path.join(RESULTS_PATH, 'confusion_matrix_counts.png')
    plt.savefig(output_path_counts, dpi=150, bbox_inches='tight')
    print(f"  Counts matrix saved to: {output_path_counts}")
    
    # Show the plots
    print("\n" + "=" * 60)
    print(" Displaying confusion matrix visualization...")
    print("=" * 60)
    plt.show()


if __name__ == '__main__':
    main()
