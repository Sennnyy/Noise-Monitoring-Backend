/*
 * ZonoTrack ESP32 Edge Computing
 * On-device K-Means Sound Classification with MFCC Extraction
 * 
 * Hardware: ESP32 + INMP441 I2S Microphone
 * Features: Real-time audio processing, MFCC extraction, K-Means inference
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>
#include "config.h"
#include "centroids.h"

// ============================================
// Global Variables
// ============================================
int16_t* audioBuffer = nullptr;
float* mfccFeatures = nullptr;

// ============================================
// I2S Configuration
// ============================================
void setupI2S() {
    Serial.println("\n[SETUP] Configuring I2S...");
    
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };
    
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_zero_dma_buffer(I2S_PORT);
    
    Serial.println("[SETUP] I2S configured successfully");
}

// ============================================
// Audio Recording
// ============================================
bool recordAudio() {
    Serial.println("\n[AUDIO] Recording audio...");
    unsigned long startTime = millis();
    
    size_t bytesRead = 0;
    int32_t sample32;
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        i2s_read(I2S_PORT, &sample32, sizeof(sample32), &bytesRead, portMAX_DELAY);
        
        // Convert 32-bit to 16-bit (shift right by 14 bits for INMP441)
        audioBuffer[i] = (int16_t)(sample32 >> 14);
        
        #if DEBUG_AUDIO
        if (i % 1000 == 0) {
            Serial.printf("Sample[%d]: %d\n", i, audioBuffer[i]);
        }
        #endif
    }
    
    unsigned long duration = millis() - startTime;
    Serial.printf("[AUDIO] Recording complete (%lu ms)\n", duration);
    
    return true;
}

// ============================================
// Helper: Convert Hz to Mel scale
// ============================================
float hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

// ============================================
// Helper: Convert Mel to Hz scale
// ============================================
float melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// ============================================
// MFCC Feature Extraction (Simplified)
// ============================================
void extractMFCC() {
    Serial.println("\n[MFCC] Extracting features...");
    unsigned long startTime = millis();
    
    // Initialize features to zero
    for (int i = 0; i < TOTAL_FEATURES; i++) {
        mfccFeatures[i] = 0.0f;
    }
    
    // Calculate number of frames
    int numFrames = (BUFFER_SIZE - N_FFT) / HOP_LENGTH + 1;
    int framesPerSegment = numFrames / NUM_SEGMENTS;
    
    // Process each segment
    for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
        float segmentMFCC[NUM_MFCC] = {0};
        int frameCount = 0;
        
        // Average MFCCs over frames in this segment
        for (int f = 0; f < framesPerSegment && (seg * framesPerSegment + f) < numFrames; f++) {
            int frameStart = (seg * framesPerSegment + f) * HOP_LENGTH;
            
            // Simple energy-based features (approximation of MFCC)
            float energy = 0.0f;
            for (int i = 0; i < N_FFT && (frameStart + i) < BUFFER_SIZE; i++) {
                float sample = (float)audioBuffer[frameStart + i] / 32768.0f;
                energy += sample * sample;
            }
            
            // Distribute energy across MFCC coefficients (simplified)
            segmentMFCC[0] += log10f(energy + 1e-10f);
            
            // Add spectral variation for other coefficients
            for (int m = 1; m < NUM_MFCC; m++) {
                float variation = 0.0f;
                for (int i = 0; i < N_FFT / NUM_MFCC && (frameStart + i) < BUFFER_SIZE; i++) {
                    int idx = frameStart + i + m * (N_FFT / NUM_MFCC);
                    if (idx < BUFFER_SIZE) {
                        float sample = (float)audioBuffer[idx] / 32768.0f;
                        variation += sample * sample;
                    }
                }
                segmentMFCC[m] += log10f(variation + 1e-10f);
            }
            
            frameCount++;
        }
        
        // Average and store in feature vector
        if (frameCount > 0) {
            for (int m = 0; m < NUM_MFCC; m++) {
                mfccFeatures[seg * NUM_MFCC + m] = segmentMFCC[m] / frameCount;
            }
        }
    }
    
    #if DEBUG_MFCC
    Serial.println("[MFCC] Feature vector:");
    for (int i = 0; i < TOTAL_FEATURES; i++) {
        if (i % 13 == 0) Serial.printf("\nSegment %d: ", i / 13);
        Serial.printf("%.3f ", mfccFeatures[i]);
    }
    Serial.println();
    #endif
    
    unsigned long duration = millis() - startTime;
    Serial.printf("[MFCC] Extraction complete (%lu ms)\n", duration);
}

// ============================================
// Feature Scaling (StandardScaler)
// ============================================
void scaleFeatures() {
    for (int i = 0; i < TOTAL_FEATURES; i++) {
        mfccFeatures[i] = (mfccFeatures[i] - SCALER_MEAN[i]) / SCALER_STD[i];
    }
}

// ============================================
// K-Means Inference
// ============================================
void classifySound() {
    Serial.println("\n[INFERENCE] Running K-Means classification...");
    unsigned long startTime = millis();
    
    // Scale features
    scaleFeatures();
    
    // Find nearest centroid
    float minDistance = INFINITY;
    int nearestCluster = -1;
    
    for (int c = 0; c < N_CLUSTERS; c++) {
        float distance = 0.0f;
        
        // Calculate Euclidean distance
        for (int f = 0; f < N_FEATURES; f++) {
            float diff = mfccFeatures[f] - CENTROIDS[c][f];
            distance += diff * diff;
        }
        
        distance = sqrtf(distance);
        
        if (distance < minDistance) {
            minDistance = distance;
            nearestCluster = c;
        }
    }
    
    // Get class from cluster
    int predictedClass = CLUSTER_TO_CLASS[nearestCluster];
    const char* className = CLASS_LABELS[predictedClass];
    
    // Calculate confidence (inverse of distance, normalized)
    float confidence = 100.0f / (1.0f + minDistance);
    
    // Calculate decibels (RMS of audio)
    float rms = 0.0f;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        float sample = (float)audioBuffer[i] / 32768.0f;
        rms += sample * sample;
    }
    rms = sqrtf(rms / BUFFER_SIZE);
    
    float dbFS = 20.0f * log10f(rms + 1e-10f);
    float dbSPL = dbFS + 94.0f;
    
    unsigned long duration = millis() - startTime;
    
    // Print results
    Serial.println("\n===========================================");
    Serial.println(" CLASSIFICATION RESULTS");
    Serial.println("===========================================");
    Serial.printf("Sound Class:  %s\n", className);
    Serial.printf("Confidence:   %.2f%%\n", confidence);
    Serial.printf("Decibels:     %.2f dB SPL\n", dbSPL);
    Serial.printf("Cluster:      %d\n", nearestCluster);
    Serial.printf("Distance:     %.4f\n", minDistance);
    Serial.printf("Inference:    %lu ms\n", duration);
    Serial.println("===========================================\n");
}

// ============================================
// Compute Decibels
// ============================================
float computeDecibels() {
    float rms = 0.0f;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        float sample = (float)audioBuffer[i] / 32768.0f;
        rms += sample * sample;
    }
    rms = sqrtf(rms / BUFFER_SIZE);
    
    float dbFS = 20.0f * log10f(rms + 1e-10f);
    return dbFS + 94.0f; // Approximate SPL
}

// ============================================
// Setup
// ============================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n\n");
    Serial.println("===========================================");
    Serial.println(" ZonoTrack ESP32 Edge Classifier");
    Serial.println("===========================================");
    Serial.printf("Model: %d clusters, %d features\n", N_CLUSTERS, N_FEATURES);
    Serial.printf("Classes: %d\n", N_CLASSES);
    Serial.println("===========================================\n");
    
    // Allocate memory
    Serial.println("[SETUP] Allocating memory...");
    audioBuffer = (int16_t*)ps_malloc(BUFFER_SIZE * sizeof(int16_t));
    mfccFeatures = (float*)ps_malloc(TOTAL_FEATURES * sizeof(float));
    
    if (!audioBuffer || !mfccFeatures) {
        Serial.println("[ERROR] Memory allocation failed!");
        while (1) delay(1000);
    }
    
    Serial.printf("[SETUP] Audio buffer: %d KB\n", (BUFFER_SIZE * sizeof(int16_t)) / 1024);
    Serial.printf("[SETUP] MFCC buffer: %d bytes\n", TOTAL_FEATURES * sizeof(float));
    
    // Setup I2S
    setupI2S();
    
    Serial.println("\n[READY] System initialized. Starting classification...\n");
    delay(1000);
}

// ============================================
// Main Loop
// ============================================
void loop() {
    // Record audio
    if (recordAudio()) {
        // Extract MFCC features
        extractMFCC();
        
        // Classify sound
        classifySound();
    }
    
    // Wait before next classification
    delay(CLASSIFICATION_INTERVAL);
}
