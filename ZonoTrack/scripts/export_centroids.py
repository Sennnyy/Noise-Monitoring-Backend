"""
Export K-Means Centroids to C++ Header File

Converts trained K-Means model to C++ arrays for ESP32 edge computing.
Run this after training the model with /api/train.
"""

import os
import numpy as np
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to ZonoTrack root
MODELS_DIR = os.path.join(BASE_DIR, 'models')
KMEANS_PATH = os.path.join(MODELS_DIR, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
CLUSTER_LABELS_PATH = os.path.join(MODELS_DIR, 'cluster_labels.npy')
LABELS_PATH = os.path.join(MODELS_DIR, 'label_classes.npy')
OUTPUT_PATH = os.path.join(BASE_DIR, 'esp32', 'esp32_edge', 'src', 'centroids.h')


def export_centroids():
    """Export centroids and model parameters to C++ header file."""
    
    print("=" * 60)
    print(" Exporting K-Means Model to C++ Header")
    print("=" * 60)
    
    # Load model
    if not os.path.exists(KMEANS_PATH):
        print("\nError: Model not found. Run /api/train first.")
        return
    
    print("\nLoading model...")
    kmeans = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)
    cluster_to_label = np.load(CLUSTER_LABELS_PATH, allow_pickle=True).item()
    label_classes = np.load(LABELS_PATH, allow_pickle=True)
    
    centroids = kmeans.cluster_centers_
    n_clusters, n_features = centroids.shape
    
    print(f"  Clusters: {n_clusters}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {list(label_classes)}")
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Generate C++ header
    with open(OUTPUT_PATH, 'w') as f:
        f.write("// Auto-generated K-Means model parameters\n")
        f.write("// DO NOT EDIT MANUALLY\n\n")
        f.write("#ifndef CENTROIDS_H\n")
        f.write("#define CENTROIDS_H\n\n")
        
        # Constants
        f.write(f"#define N_CLUSTERS {n_clusters}\n")
        f.write(f"#define N_FEATURES {n_features}\n")
        f.write(f"#define N_CLASSES {len(label_classes)}\n\n")
        
        # Scaler parameters (mean and std)
        f.write("// Scaler parameters (StandardScaler)\n")
        f.write("const float SCALER_MEAN[N_FEATURES] = {\n")
        for i, val in enumerate(scaler.mean_):
            f.write(f"    {val:.8f}{',' if i < len(scaler.mean_)-1 else ''}\n")
        f.write("};\n\n")
        
        f.write("const float SCALER_STD[N_FEATURES] = {\n")
        for i, val in enumerate(scaler.scale_):
            f.write(f"    {val:.8f}{',' if i < len(scaler.scale_)-1 else ''}\n")
        f.write("};\n\n")
        
        # Centroids (flattened 2D array)
        f.write("// K-Means centroids (flattened)\n")
        f.write("const float CENTROIDS[N_CLUSTERS][N_FEATURES] = {\n")
        for i, centroid in enumerate(centroids):
            f.write("    {")
            for j, val in enumerate(centroid):
                f.write(f"{val:.8f}{',' if j < len(centroid)-1 else ''}")
            f.write(f"}}{',' if i < len(centroids)-1 else ''}\n")
        f.write("};\n\n")
        
        # Cluster to label mapping
        f.write("// Cluster to label mapping\n")
        f.write("const char* CLUSTER_LABELS[N_CLUSTERS] = {\n")
        for i in range(n_clusters):
            label = cluster_to_label.get(i, f"cluster_{i}")
            f.write(f'    "{label}"{"," if i < n_clusters-1 else ""}\n')
        f.write("};\n\n")
        
        # Class names
        f.write("// Sound class names\n")
        f.write("const char* CLASS_NAMES[N_CLASSES] = {\n")
        for i, label in enumerate(label_classes):
            f.write(f'    "{label}"{"," if i < len(label_classes)-1 else ""}\n')
        f.write("};\n\n")
        
        f.write("#endif // CENTROIDS_H\n")
    
    # Calculate file size
    file_size = os.path.getsize(OUTPUT_PATH) / 1024  # KB
    
    print(f"\n✓ Export complete!")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Size: {file_size:.2f} KB")
    
    # Memory estimate for ESP32
    centroid_memory = n_clusters * n_features * 4  # 4 bytes per float
    total_memory = centroid_memory + (n_features * 2 * 4)  # + scaler params
    
    print(f"\n  ESP32 Memory Usage:")
    print(f"    Centroids: {centroid_memory / 1024:.2f} KB")
    print(f"    Total: {total_memory / 1024:.2f} KB")
    
    if total_memory > 100 * 1024:  # 100 KB
        print("\n  ⚠ Warning: Model size is large for ESP32")
        print("    Consider reducing N_CLUSTERS in app.py")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    export_centroids()
