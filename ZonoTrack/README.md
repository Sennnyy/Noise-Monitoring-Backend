# ZonoTrack - K-Means Sound Classification System

Urban sound classification using K-Means clustering with ESP32 integration.

## 📁 Project Structure

```
ZonoTrack/
│
├── 📄 app.py                    # Flask REST API (main application)
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # This file
│
├── 📁 deployment/               # Cloud deployment files
│   ├── Procfile                 # Gunicorn server config
│   ├── render.yaml              # Render service definition
│   └── DEPLOYMENT.md            # Deployment instructions
│
├── 📁 scripts/                  # Utility scripts
│   ├── confusion_matrix.py      # Model evaluation & visualization
│   └── export_centroids.py      # Export model for ESP32
│
├── 📁 docs/                     # Documentation
│   └── ESP32_README.md          # ESP32 integration guide
│
├── 📁 models/                   # Trained model files
│   ├── kmeans_model.joblib      # K-Means model
│   ├── scaler.joblib            # Feature scaler
│   ├── centroids.npy            # Cluster centroids
│   ├── cluster_labels.npy       # Cluster-to-label mapping
│   └── label_classes.npy        # Sound class names
│
├── 📁 dataset/                  # Training data
│   ├── children playing/        # 403 audio files
│   ├── construction/            # 900 audio files
│   ├── dog_sound/               # 384 audio files
│   └── vehicle_sound/           # 237 audio files
│
├── 📁 results/                  # Output files
│   ├── confusion_matrix.png
│   └── confusion_matrix_counts.png
│
└── 📁 esp32/                    # ESP32 projects
    └── esp32_cloud/             # Cloud-based approach
        ├── platformio.ini
        └── src/
            ├── config.h
            └── main.cpp
```

## 🚀 Quick Start

### 1. Train the Model

```bash
# Install dependencies
pip install -r requirements.txt

# Start Flask API
python app.py

# Train model (via API)
curl -X POST http://localhost:5000/api/train
```

### 2. Evaluate Model

```bash
# Generate confusion matrix
python scripts/confusion_matrix.py
```

### 3. Test Classification

```bash
# Classify audio file
curl -X POST -F "file=@audio.wav" http://localhost:5000/api/predict
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Classify audio + get decibels |
| POST | `/api/train` | Train K-Means model |
| GET | `/api/classes` | List sound categories |
| GET | `/api/centroids` | Export centroids for ESP32 |
| GET | `/api/algorithm` | View K-Means formulas |

## 🔧 Features

- **K-Means Clustering** - 256 clusters for high-resolution classification
- **Decibel Measurement** - Real-time SPL estimation
- **Confusion Matrix** - Visual model evaluation
- **ESP32 Integration** - Cloud and edge computing support
- **REST API** - Production-ready Flask server

## 🌐 Deployment

Deploy to Render cloud platform:

```bash
# See deployment/DEPLOYMENT.md for detailed instructions
git init
git add .
git commit -m "Initial commit"
git push origin main
# Deploy via Render dashboard
```

## 🔌 ESP32 Integration

Connect INMP441 microphone to ESP32 for real-time classification:

```
INMP441 → ESP32
─────────────────
VDD → 3.3V
GND → GND
SD  → GPIO 41
WS  → GPIO 42
SCK → GPIO 40
L/R → GND
```

See `docs/ESP32_README.md` for complete setup guide.

## 📈 Model Performance

- **Accuracy**: Run `python scripts/confusion_matrix.py` to evaluate
- **Classes**: 4 sound categories (children playing, construction, dog, vehicle)
- **Features**: 130 MFCC features per sample
- **Sample Rate**: 22050 Hz

## 🛠️ Development

### Run Locally

```bash
python app.py
# API available at http://localhost:5000
```

### Export Model for ESP32

```bash
python scripts/export_centroids.py
# Generates esp32/esp32_edge/src/centroids.h
```

### Flash ESP32

```bash
cd esp32/esp32_cloud
# Edit src/config.h with WiFi credentials
pio run --target upload
pio device monitor
```

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Support

For issues or questions, see the troubleshooting section in `docs/ESP32_README.md`
