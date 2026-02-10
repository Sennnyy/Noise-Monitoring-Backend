# ESP32 Edge Computing - Arduino IDE Setup Guide

Complete guide for testing the ESP32 edge project in Arduino IDE.

## Prerequisites

### 1. Install Arduino IDE
- Download from: https://www.arduino.cc/en/software
- Install version 2.0 or later

### 2. Install ESP32 Board Support

**In Arduino IDE:**
1. Go to `File` → `Preferences`
2. Add to "Additional Board Manager URLs":
   ```
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
3. Go to `Tools` → `Board` → `Boards Manager`
4. Search for "esp32"
5. Install **"esp32 by Espressif Systems"** (version 2.0.11 or later)

### 3. Install Required Library

**ESP-DSP Library:**
1. Go to `Sketch` → `Include Library` → `Manage Libraries`
2. Search for "ESP-DSP"
3. Install **"ESP-DSP by Espressif"**

---

## Project Setup for Arduino IDE

### Step 1: Create Arduino Project Structure

Create a new folder structure:
```
esp32_edge_arduino/
├── esp32_edge_arduino.ino  (main sketch)
├── config.h
├── centroids.h
└── README.txt
```

### Step 2: Copy Files

1. **Create main sketch file:**
   - Copy `esp32_edge/src/main.cpp` content
   - Rename to `esp32_edge_arduino.ino`
   - Remove `#include <Arduino.h>` (Arduino IDE adds this automatically)

2. **Copy header files:**
   - Copy `esp32_edge/src/config.h` to sketch folder
   - Copy `esp32_edge/src/centroids.h` to sketch folder

### Step 3: Configure Arduino IDE

1. **Select Board:**
   - `Tools` → `Board` → `ESP32 Arduino` → `ESP32S3 Dev Module`

2. **Configure Settings:**
   ```
   Board: "ESP32S3 Dev Module"
   Upload Speed: "921600"
   USB Mode: "Hardware CDC and JTAG"
   USB CDC On Boot: "Enabled"
   USB Firmware MSC On Boot: "Disabled"
   USB DFU On Boot: "Disabled"
   Upload Mode: "UART0 / Hardware CDC"
   CPU Frequency: "240MHz (WiFi)"
   Flash Mode: "QIO 80MHz"
   Flash Size: "8MB (64Mb)"
   Partition Scheme: "8M with spiffs (3MB APP/1.5MB SPIFFS)"
   Core Debug Level: "Info"
   PSRAM: "OPI PSRAM"
   Arduino Runs On: "Core 1"
   Events Run On: "Core 1"
   ```

3. **Select Port:**
   - Connect ESP32 via USB
   - `Tools` → `Port` → Select your COM port (e.g., COM3)

---

## Quick Start Files

### Create: `esp32_edge_arduino.ino`

```cpp
// Copy the entire content from main.cpp here
// Remove the line: #include <Arduino.h>
```

### Verify `config.h` and `centroids.h` are in the same folder

---

## Upload and Test

### Step 1: Verify Code
1. Click **Verify** (✓) button
2. Wait for compilation (may take 1-2 minutes first time)
3. Check for errors in output window

### Step 2: Upload
1. Click **Upload** (→) button
2. Wait for upload to complete
3. You should see "Hard resetting via RTS pin..."

### Step 3: Open Serial Monitor
1. Click **Serial Monitor** button (top right)
2. Set baud rate to **115200**
3. You should see output like:

```
===========================================
 ZonoTrack ESP32 Edge Classifier
===========================================
Model: 256 clusters, 130 features
Classes: 4
===========================================

[SETUP] Allocating memory...
[SETUP] Audio buffer: 129 KB
[SETUP] MFCC buffer: 520 bytes

[SETUP] Configuring I2S...
[SETUP] I2S configured successfully

[READY] System initialized. Starting classification...

[AUDIO] Recording audio...
[AUDIO] Recording complete (3000 ms)

[MFCC] Extracting features...
[MFCC] Extraction complete (245 ms)

[INFERENCE] Running K-Means classification...

===========================================
 CLASSIFICATION RESULTS
===========================================
Sound Class:  vehicle_sound
Confidence:   85.23%
Decibels:     72.45 dB SPL
Cluster:      142
Distance:     2.3456
Inference:    187 ms
===========================================
```

---

## Troubleshooting

### Upload Failed

**Problem:** "Failed to connect to ESP32"

**Solutions:**
1. Hold **BOOT** button while clicking upload
2. Try different USB cable (data cable, not charge-only)
3. Install CP210x or CH340 USB driver
4. Try lower upload speed (460800 or 115200)

### Compilation Errors

**Problem:** "centroids.h: No such file"

**Solution:** Ensure all files are in the same folder as the `.ino` file

**Problem:** "PSRAM not found"

**Solution:** 
- Check if your ESP32 has PSRAM (optional)
- Change PSRAM setting to "Disabled" if not available
- Reduce buffer sizes in `config.h`

### No Serial Output

**Problem:** Serial monitor shows nothing

**Solutions:**
1. Check baud rate is 115200
2. Press **RESET** button on ESP32
3. Select correct COM port
4. Enable "USB CDC On Boot" in Tools menu

### Memory Errors

**Problem:** "Memory allocation failed"

**Solutions:**
1. Enable PSRAM in Tools menu
2. Reduce `BUFFER_SIZE` in config.h
3. Use ESP32 with PSRAM if available (improves performance)

---

## Hardware Wiring

Connect INMP441 to ESP32:

```
INMP441    →    ESP32
─────────────────────────
VDD        →    3.3V
GND        →    GND
SD         →    GPIO 41
WS         →    GPIO 42
SCK        →    GPIO 40
L/R        →    GND
```

**Important:**
- Use short wires (< 10cm)
- Connect L/R to GND for left channel
- Ensure stable 3.3V power supply

---

## Testing Tips

### 1. Test Audio Recording
- Make sounds near the microphone
- Check if "Recording complete" appears every 5 seconds
- Verify recording time is ~3000ms

### 2. Test Classification
- Try different sounds:
  - Talk (should detect "children playing" or similar)
  - Play vehicle sounds from phone
  - Make construction-like noises
  - Dog barking sounds

### 3. Monitor Performance
- Check inference time (should be < 500ms)
- Monitor memory usage
- Verify decibel readings make sense

### 4. Debug Mode
Enable debug in `config.h`:
```cpp
#define DEBUG_AUDIO       1  // Print audio samples
#define DEBUG_MFCC        1  // Print MFCC features
#define DEBUG_INFERENCE   1  // Print inference details
```

---

## Performance Expectations

| Metric | Expected Value |
|--------|---------------|
| Recording Time | ~3000 ms |
| MFCC Extraction | 200-400 ms |
| K-Means Inference | 150-300 ms |
| Total Latency | ~500 ms |
| Memory Usage | ~400 KB |
| Classification Interval | 5 seconds |

---

## Next Steps

1. **Calibrate Microphone:** Adjust gain if needed
2. **Tune Parameters:** Modify `config.h` for better accuracy
3. **Add Display:** Connect OLED for visual feedback
4. **Log Data:** Add SD card logging
5. **Optimize MFCC:** Implement full MFCC with ESP-DSP

---

## Support

For issues:
1. Check wiring connections
2. Verify board settings in Arduino IDE
3. Enable debug output in `config.h`
4. Check ESP32 has sufficient power
5. Review Serial Monitor output for errors
