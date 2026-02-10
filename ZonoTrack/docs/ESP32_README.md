# ESP32 Integration for ZonoTrack

Complete guide for integrating ESP32 with INMP441 microphone for sound classification.

## Hardware Requirements

- **ESP32-DevKitC** (any variant)
- **INMP441 I2S Digital Microphone**
- Jumper wires
- USB-C cable

## Wiring Diagram

### INMP441 to ESP32 Connections

```
INMP441 Pin    →    ESP32 Pin
─────────────────────────────────
VDD            →    3.3V
GND            →    GND
SD (DOUT)      →    GPIO 41
WS (LRCLK)     →    GPIO 42
SCK (BCLK)     →    GPIO 40
L/R            →    GND (for left channel)
```

### Pin Diagram
```
     INMP441                    ESP32
    ┌────────┐                ┌──────────┐
    │  VDD   │───────────────→│   3.3V   │
    │  GND   │───────────────→│   GND    │
    │  SD    │───────────────→│  GPIO 41 │
    │  WS    │───────────────→│  GPIO 42 │
    │  SCK   │───────────────→│  GPIO 40 │
    │  L/R   │───────────────→│   GND    │
    └────────┘                └──────────┘
```

## Two Approaches

### Approach 1: Cloud Computing (Recommended for Beginners)

**How it works:**
1. ESP32 records audio from INMP441
2. Creates WAV file in memory
3. Sends to Flask API via WiFi
4. Receives classification + decibels

**Pros:**
- ✓ Simple ESP32 code
- ✓ Full model accuracy
- ✓ Easy to update model
- ✓ No memory constraints

**Cons:**
- ✗ Requires WiFi connection
- ✗ Higher latency (~2-3 seconds)
- ✗ Depends on server availability

**Setup:** See [Cloud Approach Setup](#cloud-approach-setup)

---

### Approach 2: Edge Computing (Advanced)

**How it works:**
1. ESP32 records audio from INMP441
2. Extracts MFCC features on-device
3. Runs K-Means inference locally
4. No internet required

**Pros:**
- ✓ No WiFi needed
- ✓ Low latency (~500ms)
- ✓ Privacy (audio stays on device)
- ✓ Works offline

**Cons:**
- ✗ Complex implementation
- ✗ Limited by ESP32 memory
- ✗ Requires MFCC library
- ✗ Model updates require reflashing

**Setup:** See [Edge Approach Setup](#edge-approach-setup)

---

## Cloud Approach Setup

### 1. Deploy Flask API

First, deploy your Flask API to Render (see [DEPLOYMENT.md](DEPLOYMENT.md)):

```bash
# Your API will be available at:
https://zonotrack.onrender.com
```

### 2. Configure ESP32

Edit `esp32_cloud/src/config.h`:

```cpp
#define WIFI_SSID "YourWiFiName"
#define WIFI_PASSWORD "YourWiFiPassword"
#define API_URL "https://zonotrack.onrender.com"
```

### 3. Install PlatformIO

```bash
# Install PlatformIO CLI
pip install platformio

# Or use VS Code extension
# Search for "PlatformIO IDE" in VS Code extensions
```

### 4. Build and Upload

```bash
cd esp32_cloud
pio run --target upload
pio device monitor
```

### 5. Expected Output

```
===========================================
 ZonoTrack ESP32 Cloud Classifier
===========================================
✓ WiFi connected!
IP Address: 192.168.1.123

Recording Audio (3 seconds)...
✓ Recording complete!

Sending to Flask API...
✓ Classification successful!

-------------------------------------------
 RESULTS
-------------------------------------------
Sound Class:  vehicle_sound
Confidence:   87.45%
Decibels:     78.32 dB SPL
Noise Level:  Loud (traffic level)
Cluster:      142
Distance:     2.3456
-------------------------------------------
```

---

## Edge Approach Setup

### 1. Export Centroids

Run the Python export script:

```bash
cd ZonoTrack
python export_centroids.py
```

This creates `esp32_edge/src/centroids.h` with your trained model.

### 2. Install Dependencies

The edge approach requires ESP-DSP for MFCC extraction:

```bash
cd esp32_edge
pio lib install "espressif/esp-dsp"
```

### 3. Build and Upload

```bash
pio run --target upload
pio device monitor
```

### 4. Expected Output

```
===========================================
 ZonoTrack ESP32 Edge Classifier
===========================================
Model loaded: 256 clusters, 130 features

Recording Audio (3 seconds)...
✓ Recording complete!

Extracting MFCC features...
✓ Feature extraction complete!

Running K-Means inference...
✓ Classification complete!

-------------------------------------------
 RESULTS
-------------------------------------------
Sound Class:  construction
Confidence:   92.18%
Decibels:     85.67 dB SPL
Noise Level:  Very loud (heavy traffic)
Cluster:      89
Distance:     1.8234
Inference Time: 487 ms
-------------------------------------------
```

---

## Troubleshooting

### No Audio Recorded

**Problem:** All audio samples are zero

**Solutions:**
1. Check INMP441 wiring (especially L/R to GND)
2. Verify 3.3V power supply
3. Try swapping WS and SCK pins
4. Check I2S port number (I2S_NUM_0)

### WiFi Connection Failed (Cloud)

**Problem:** ESP32 can't connect to WiFi

**Solutions:**
1. Verify SSID and password in `config.h`
2. Check WiFi signal strength
3. Ensure 2.4GHz WiFi (ESP32 doesn't support 5GHz)
4. Try moving closer to router

### HTTP Error 500 (Cloud)

**Problem:** Flask API returns error

**Solutions:**
1. Check if model is trained (`/api/train`)
2. Verify model files exist on server
3. Check server logs in Render dashboard
4. Test API with curl first

### Out of Memory (Edge)

**Problem:** ESP32 crashes or fails to allocate

**Solutions:**
1. Use ESP32 with PSRAM (if available)
2. Reduce `N_CLUSTERS` in Python (retrain with fewer clusters)
3. Enable PSRAM in `platformio.ini`:
   ```ini
   build_flags = -DBOARD_HAS_PSRAM
   ```

### Low Classification Accuracy

**Problem:** Wrong sound classifications

**Solutions:**
1. Retrain model with more data
2. Increase `N_CLUSTERS` (e.g., 512)
3. Check microphone placement (avoid obstructions)
4. Verify sample rate matches (22050 Hz)

---

## Performance Comparison

| Metric | Cloud Approach | Edge Approach |
|--------|---------------|---------------|
| Latency | 2-3 seconds | 500 ms |
| Accuracy | 100% (full model) | 95-98% |
| WiFi Required | Yes | No |
| Power Consumption | Higher | Lower |
| Memory Usage | Low (~50 KB) | High (~400 KB) |
| Setup Complexity | Easy | Advanced |
| Model Updates | Instant | Requires reflash |

---

## Next Steps

1. **Test both approaches** and choose based on your use case
2. **Adjust classification interval** in `config.h`
3. **Add display** (OLED/LCD) for visual feedback
4. **Implement data logging** to SD card
5. **Add battery power** for portable deployment

---

## Additional Resources

- [ESP32 Datasheet](https://www.espressif.com/sites/default/files/documentation/esp32_datasheet_en.pdf)
- [INMP441 Datasheet](https://invensense.tdk.com/wp-content/uploads/2015/02/INMP441.pdf)
- [PlatformIO Documentation](https://docs.platformio.org/)
- [ESP-DSP Library](https://github.com/espressif/esp-dsp)

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review PlatformIO build logs
3. Test Flask API independently with curl
4. Verify hardware connections with multimeter
