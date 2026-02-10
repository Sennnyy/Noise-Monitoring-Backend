/*
 * ZonoTrack ESP32 Cloud Configuration
 * 
 * Update these settings for your network and server
 */

#ifndef CONFIG_H
#define CONFIG_H

// ============================================
// WiFi Configuration
// ============================================
#define WIFI_SSID "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

// ============================================
// Flask API Server
// ============================================
// For local testing: "http://192.168.1.100:5000"
// For Render deployment: "https://zonotrack.onrender.com"
#define API_URL "http://192.168.1.100:5000"
#define API_ENDPOINT "/api/predict"

// ============================================
// INMP441 I2S Microphone Pins
// ============================================
#define I2S_WS    42    // Word Select (LRCLK)
#define I2S_SD    41    // Serial Data (DOUT)
#define I2S_SCK   40    // Serial Clock (BCLK)

// ============================================
// Audio Recording Parameters
// ============================================
#define SAMPLE_RATE   22050    // Must match Python model (22050 Hz)
#define RECORD_TIME   3        // Recording duration in seconds
#define I2S_PORT      I2S_NUM_0
#define BITS_PER_SAMPLE 16

// Buffer size calculation
#define BUFFER_SIZE (SAMPLE_RATE * RECORD_TIME)

// ============================================
// Classification Settings
// ============================================
#define CLASSIFICATION_INTERVAL 5000  // Classify every 5 seconds (ms)

#endif // CONFIG_H
