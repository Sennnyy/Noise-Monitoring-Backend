// Configuration for ESP32 Edge Computing
// On-device K-Means sound classification

#ifndef CONFIG_H
#define CONFIG_H

// ============================================
// I2S Microphone Configuration (INMP441)
// ============================================
#define I2S_WS    42    // Word Select (LRCLK)
#define I2S_SD    41    // Serial Data (DOUT)
#define I2S_SCK   40    // Serial Clock (BCLK)
#define I2S_PORT  I2S_NUM_0

// ============================================
// Audio Parameters (MUST match Python model)
// ============================================
#define SAMPLE_RATE       22050    // Hz
#define RECORD_TIME       3        // seconds
#define BUFFER_SIZE       (SAMPLE_RATE * RECORD_TIME)

// ============================================
// MFCC Feature Extraction Parameters
// ============================================
#define NUM_MFCC          13       // Number of MFCC coefficients
#define N_FFT             2048     // FFT window size
#define HOP_LENGTH        512      // Hop length for STFT
#define NUM_SEGMENTS      10       // Number of time segments
#define TOTAL_FEATURES    (NUM_MFCC * NUM_SEGMENTS)  // 130 features

// ============================================
// Mel Filter Bank Parameters
// ============================================
#define NUM_MEL_FILTERS   40       // Number of mel filters
#define MEL_MIN_FREQ      0.0f     // Minimum frequency (Hz)
#define MEL_MAX_FREQ      11025.0f // Maximum frequency (Hz)

// ============================================
// Classification Parameters
// ============================================
#define CLASSIFICATION_INTERVAL  5000  // ms between classifications

// ============================================
// Debug Settings
// ============================================
#define DEBUG_AUDIO       0        // Print audio samples
#define DEBUG_MFCC        0        // Print MFCC features
#define DEBUG_INFERENCE   1        // Print inference results

#endif // CONFIG_H
