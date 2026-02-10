"""
ZonoTrack K-Means Clustering REST API
Urban Sound Classification with Explicit Computation

K-Means Algorithm:
==================
1. Initialize k centroids randomly from data points
2. Assign each point to nearest centroid using Euclidean distance
3. Update centroids as mean of assigned points
4. Repeat until convergence

Distance Computation:
====================
Euclidean Distance: d(x, c) = sqrt(sum((x_i - c_i)^2))

Where:
- x = feature vector of audio sample
- c = centroid vector
- i = feature dimension index

Confidence Score:
================
confidence = exp(-min_distance / 10) * 100

Lower distance to centroid = higher confidence
"""

import os
import tempfile
import math
import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

# ============================================
# CONSTANTS
# ============================================
SAMPLE_RATE = 22050
NUM_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512
NUM_SEGMENTS = 10
N_CLUSTERS = 256  # Number of K-Means clusters

# Base directory (where app.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================
# FLASK APP SETUP
# ============================================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
KMEANS_PATH = os.path.join(MODELS_DIR, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
CENTROIDS_PATH = os.path.join(MODELS_DIR, 'centroids.npy')
CLUSTER_LABELS_PATH = os.path.join(MODELS_DIR, 'cluster_labels.npy')
LABELS_PATH = os.path.join(MODELS_DIR, 'label_classes.npy')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'test_data')

# Global model variables
kmeans_model = None
scaler = None
cluster_to_label = None
label_classes = None


# ============================================
# COMPUTATION FUNCTIONS
# ============================================

def euclidean_distance(x, centroid):
    """
    Compute Euclidean distance between feature vector and centroid.
    
    Formula: d = sqrt(sum((x_i - c_i)^2))
    
    Args:
        x: Feature vector (1D array)
        centroid: Centroid vector (1D array)
    
    Returns:
        float: Euclidean distance
    """
    squared_diff = 0.0
    for i in range(len(x)):
        diff = x[i] - centroid[i]
        squared_diff += diff * diff
    
    distance = math.sqrt(squared_diff)
    return distance


def compute_all_distances(x, centroids):
    """
    Compute distances from feature vector to ALL centroids.
    
    Args:
        x: Feature vector
        centroids: Array of centroid vectors
    
    Returns:
        list: Distances to each centroid
    """
    distances = []
    for centroid in centroids:
        d = euclidean_distance(x, centroid)
        distances.append(d)
    return distances


def find_nearest_cluster(distances):
    """
    Find the cluster with minimum distance.
    
    Args:
        distances: List of distances to each centroid
    
    Returns:
        tuple: (cluster_index, min_distance)
    """
    min_distance = distances[0]
    min_index = 0
    
    for i in range(1, len(distances)):
        if distances[i] < min_distance:
            min_distance = distances[i]
            min_index = i
    
    return min_index, min_distance


def compute_confidence(min_distance):
    """
    Compute confidence score from distance.
    
    Formula: confidence = exp(-distance / 10) * 100
    
    Closer distance (smaller value) = higher confidence
    
    Args:
        min_distance: Distance to nearest centroid
    
    Returns:
        float: Confidence percentage (0-100)
    """
    confidence = math.exp(-min_distance / 10.0) * 100.0
    return min(confidence, 99.9)


def compute_decibels(signal):
    """
    Compute Sound Pressure Level (SPL) in decibels.
    
    Formula: dB_SPL = 20 * log10(rms / reference)
    
    Where:
    - rms = Root Mean Square of the signal
    - reference = 20 μPa (threshold of human hearing)
    
    For digital audio normalized to [-1, 1]:
    - 0 dBFS = maximum digital level
    - dB = 20 * log10(rms)
    
    Args:
        signal: Audio signal array
    
    Returns:
        dict: dB values and computation details
    """
    # Compute RMS (Root Mean Square)
    rms = math.sqrt(sum(s * s for s in signal) / len(signal))
    
    # Avoid log(0)
    if rms < 1e-10:
        rms = 1e-10
    
    # dBFS (decibels relative to full scale)
    db_fs = 20 * math.log10(rms)
    
    # Estimated SPL (approximate, assuming calibrated microphone)
    # Typical ambient: 30-50 dB, conversation: 60-70 dB, traffic: 70-85 dB
    # We add an offset to approximate real-world SPL values
    db_spl_estimate = db_fs + 94  # Approximate offset for SPL
    
    return {
        'rms': round(rms, 6),
        'db_fs': round(db_fs, 2),
        'db_spl_estimate': round(max(0, db_spl_estimate), 2),
        'formula': 'dB = 20 * log10(RMS)',
        'description': get_noise_level_description(db_spl_estimate)
    }


def get_noise_level_description(db_spl):
    """
    Get human-readable description of noise level.
    
    Reference levels:
    - 30 dB: Whisper
    - 60 dB: Normal conversation
    - 70 dB: Traffic
    - 85 dB: Heavy traffic (hearing damage threshold)
    - 100 dB: Construction
    - 120 dB: Pain threshold
    """
    if db_spl < 30:
        return 'Very quiet (whisper level)'
    elif db_spl < 50:
        return 'Quiet (library level)'
    elif db_spl < 60:
        return 'Moderate (office level)'
    elif db_spl < 70:
        return 'Normal (conversation level)'
    elif db_spl < 80:
        return 'Loud (traffic level)'
    elif db_spl < 90:
        return 'Very loud (heavy traffic)'
    elif db_spl < 100:
        return 'Dangerously loud (prolonged exposure harmful)'
    elif db_spl < 120:
        return 'Extremely loud (construction level)'
    else:
        return 'Pain threshold exceeded'


def get_noise_indicator(db_spl):
    """
    Get noise level indicator based on WHO/EPA standards.
    
    WHO Guidelines:
    - < 55 dB: Normal (safe for continuous exposure)
    - 55-70 dB: Elevated (moderate noise, may cause annoyance)
    - > 70 dB: High (risk of hearing damage with prolonged exposure)
    
    Args:
        db_spl: Sound pressure level in decibels
    
    Returns:
        str: 'Normal', 'Elevated', or 'High'
    """
    if db_spl < 55:
        return 'Normal'
    elif db_spl < 70:
        return 'Elevated'
    else:
        return 'High'


def compute_centroid_mean(points):
    """
    Compute new centroid as mean of assigned points.
    
    Formula: c_new = (1/n) * sum(x_i)
    
    Args:
        points: List of feature vectors assigned to cluster
    
    Returns:
        array: New centroid position
    """
    if len(points) == 0:
        return None
    
    n_features = len(points[0])
    centroid = [0.0] * n_features
    
    for point in points:
        for i in range(n_features):
            centroid[i] += point[i]
    
    for i in range(n_features):
        centroid[i] /= len(points)
    
    return centroid


# ============================================
# FEATURE EXTRACTION
# ============================================

def extract_mfcc_features(file_path, return_signal=False):
    """
    Extract MFCC features from audio file.
    
    MFCC = Mel-Frequency Cepstral Coefficients
    Represents the short-term power spectrum of sound.
    
    Args:
        file_path: Path to audio file
        return_signal: If True, also return the raw signal for dB calculation
    
    Returns:
        If return_signal=False: flattened MFCC features
        If return_signal=True: (flattened MFCC features, signal array)
    """
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
    
    if return_signal:
        return mfccs.flatten(), signal
    return mfccs.flatten()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load trained K-Means model."""
    global kmeans_model, scaler, cluster_to_label, label_classes
    
    if os.path.exists(KMEANS_PATH) and os.path.exists(SCALER_PATH):
        kmeans_model = joblib.load(KMEANS_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        if os.path.exists(CLUSTER_LABELS_PATH):
            cluster_to_label = np.load(CLUSTER_LABELS_PATH, allow_pickle=True).item()
        if os.path.exists(LABELS_PATH):
            label_classes = np.load(LABELS_PATH, allow_pickle=True)
        
        print("K-Means model loaded")
    else:
        print("Model not found. Run /api/train first.")


# ============================================
# REST API ENDPOINTS
# ============================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Classify audio using K-Means with explicit computation.
    
    Returns class, confidence, and computation details.
    """
    if kmeans_model is None:
        return jsonify({'error': 'Model not loaded. Run /api/train first.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features and signal for dB calculation
        features, signal = extract_mfcc_features(filepath, return_signal=True)
        features_scaled = scaler.transform([features])[0]
        
        # COMPUTATION: Calculate decibels
        decibels = compute_decibels(signal)
        
        # Get centroids
        centroids = kmeans_model.cluster_centers_
        
        # COMPUTATION: Calculate distances to all centroids
        distances = compute_all_distances(features_scaled, centroids)
        
        # COMPUTATION: Find nearest cluster
        cluster_idx, min_distance = find_nearest_cluster(distances)
        
        # COMPUTATION: Calculate confidence
        confidence = compute_confidence(min_distance)
        
        # Map cluster to label
        if cluster_to_label and cluster_idx in cluster_to_label:
            predicted_class = cluster_to_label[cluster_idx]
        else:
            predicted_class = f"cluster_{cluster_idx}"
        
        os.remove(filepath)
        
        return jsonify({
            'class': predicted_class,
            'confidence': round(confidence, 2),
            'decibels': round(decibels['db_spl_estimate'], 2),
            'indicator': get_noise_indicator(decibels['db_spl_estimate'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train():
    """
    Train K-Means model on dataset.
    
    K-Means Algorithm Steps:
    1. Initialize k random centroids
    2. Assign points to nearest centroid (Euclidean distance)
    3. Recompute centroids as mean of assigned points
    4. Repeat until convergence
    """
    global kmeans_model, scaler, cluster_to_label, label_classes
    
    try:
        if not os.path.exists(DATASET_PATH):
            return jsonify({'error': 'Dataset folder not found'}), 400
        
        features_list = []
        labels_list = []
        
        for root, dirs, files in os.walk(DATASET_PATH):
            for file in files:
                if file.endswith(('.wav', '.mp3')):
                    file_path = os.path.join(root, file)
                    label = os.path.basename(root)
                    
                    try:
                        features = extract_mfcc_features(file_path)
                        features_list.append(features)
                        labels_list.append(label)
                    except Exception as e:
                        print(f"Error: {file_path}: {e}")
        
        if len(features_list) == 0:
            return jsonify({'error': 'No audio files found'}), 400
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Scale features (standardization)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train K-Means
        kmeans_model = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        kmeans_model.fit(X_scaled)
        
        # Map clusters to labels (majority voting)
        cluster_assignments = kmeans_model.predict(X_scaled)
        cluster_to_label = {}
        
        for cluster_id in range(N_CLUSTERS):
            mask = cluster_assignments == cluster_id
            if np.any(mask):
                cluster_labels = y[mask]
                most_common = Counter(cluster_labels).most_common(1)
                if most_common:
                    cluster_to_label[cluster_id] = most_common[0][0]
        
        label_classes = np.unique(y)
        
        # Save model
        joblib.dump(kmeans_model, KMEANS_PATH)
        joblib.dump(scaler, SCALER_PATH)
        np.save(CENTROIDS_PATH, kmeans_model.cluster_centers_)
        np.save(CLUSTER_LABELS_PATH, cluster_to_label)
        np.save(LABELS_PATH, label_classes)
        
        # Calculate inertia (sum of squared distances)
        inertia = kmeans_model.inertia_
        
        return jsonify({
            'status': 'completed',
            'samples': len(features_list),
            'classes': label_classes.tolist(),
            'n_clusters': N_CLUSTERS,
            'computation': {
                'algorithm': 'K-Means Clustering',
                'distance_metric': 'Euclidean: d = sqrt(sum((x_i - c_i)^2))',
                'centroid_update': 'Mean: c = (1/n) * sum(x_i)',
                'inertia': round(float(inertia), 2),
                'iterations': kmeans_model.n_iter_
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get sound classification categories."""
    if label_classes is None:
        if os.path.exists(DATASET_PATH):
            classes = [d for d in os.listdir(DATASET_PATH) 
                      if os.path.isdir(os.path.join(DATASET_PATH, d))]
            return jsonify({'classes': classes})
        return jsonify({'error': 'Labels not loaded'}), 500
    
    return jsonify({'classes': label_classes.tolist()})


@app.route('/api/centroids', methods=['GET'])
def get_centroids():
    """Get centroids for ESP32 deployment."""
    if kmeans_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'centroids': kmeans_model.cluster_centers_.tolist(),
        'cluster_labels': cluster_to_label if cluster_to_label else {},
        'n_clusters': N_CLUSTERS,
        'feature_dim': kmeans_model.cluster_centers_.shape[1]
    })


@app.route('/api/algorithm', methods=['GET'])
def get_algorithm_info():
    """Get K-Means algorithm explanation and formulas."""
    return jsonify({
        'algorithm': 'K-Means Clustering',
        'steps': [
            '1. Initialize k centroids randomly',
            '2. Assign each point to nearest centroid',
            '3. Update centroids as mean of points',
            '4. Repeat until convergence'
        ],
        'formulas': {
            'euclidean_distance': 'd(x, c) = sqrt(sum((x_i - c_i)^2))',
            'centroid_update': 'c_new = (1/n) * sum(x_i)',
            'confidence': 'confidence = exp(-distance/10) * 100'
        },
        'parameters': {
            'n_clusters': N_CLUSTERS,
            'max_iterations': 300,
            'n_init': 10,
            'decibel_formula': 'dB = 20 * log10(RMS)'
        }
    })


if __name__ == '__main__':
    load_model()
    print("\n" + "="*50)
    print("ZonoTrack K-Means Clustering API")
    print("="*50)
    print("\nAlgorithm: K-Means Clustering")
    print("Distance Formula: d = sqrt(sum((x_i - c_i)^2))")
    print("Decibel Formula: dB = 20 * log10(RMS)")
    print("\nEndpoints:")
    print("  POST /api/predict   - Classify audio (with dB measurement)")
    print("  POST /api/train     - Train K-Means model")
    print("  GET  /api/classes   - List categories")
    print("  GET  /api/centroids - Export for ESP32")
    print("  GET  /api/algorithm - View formulas")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
