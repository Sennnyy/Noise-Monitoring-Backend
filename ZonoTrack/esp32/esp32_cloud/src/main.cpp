/*
 * ZonoTrack ESP32 Cloud Classifier
 * 
 * Records audio from INMP441 microphone and sends to Flask API
 * for K-Means classification and decibel measurement.
 * 
 * Hardware:
 * - ESP32-DevKitC
 * - INMP441 I2S Microphone
 * 
 * Wiring:
 * INMP441 -> ESP32
 * ----------------------
 * WS  (LRCLK) -> GPIO 42
 * SD  (DOUT)  -> GPIO 41
 * SCK (BCLK)  -> GPIO 40
 * VDD         -> 3.3V
 * GND         -> GND
 * L/R         -> GND (left channel)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <driver/i2s.h>
#include <ArduinoJson.h>
#include "config.h"

int16_t* audioBuffer = nullptr;
bool isRecording = false;

void setupI2S() {
    Serial.println("Configuring I2S...");
    
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
    
    esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("Failed to install I2S driver: %d\n", err);
        return;
    }
    
    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("Failed to set I2S pins: %d\n", err);
        return;
    }
    
    Serial.println("I2S configured successfully");
}

void connectWiFi() {
    Serial.println("\n===========================================");
    Serial.println(" Connecting to WiFi");
    Serial.println("===========================================");
    Serial.printf("SSID: %s\n", WIFI_SSID);
    
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n✓ WiFi connected!");
        Serial.printf("IP Address: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n✗ WiFi connection failed!");
    }
}

bool recordAudio() {
    Serial.println("\n===========================================");
    Serial.println(" Recording Audio");
    Serial.println("===========================================");
    Serial.printf("Duration: %d seconds\n", RECORD_TIME);
    Serial.printf("Sample Rate: %d Hz\n", SAMPLE_RATE);
    
    isRecording = true;
    
    if (audioBuffer == nullptr) {
        audioBuffer = (int16_t*)ps_malloc(BUFFER_SIZE * sizeof(int16_t));
        if (audioBuffer == nullptr) {
            Serial.println("✗ Failed to allocate audio buffer!");
            return false;
        }
    }
    
    memset(audioBuffer, 0, BUFFER_SIZE * sizeof(int16_t));
    
    size_t bytesRead = 0;
    int32_t sample32;
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        i2s_read(I2S_PORT, &sample32, sizeof(int32_t), &bytesRead, portMAX_DELAY);
        
        audioBuffer[i] = (int16_t)(sample32 >> 16);
        
        if (i % (SAMPLE_RATE / 4) == 0) {
            Serial.print(".");
        }
    }
    
    Serial.println("\n✓ Recording complete!");
    isRecording = false;
    
    return true;
}

uint8_t* createWAV(size_t* wavSize) {
    const int headerSize = 44;
    const int dataSize = BUFFER_SIZE * sizeof(int16_t);
    *wavSize = headerSize + dataSize;
    
    uint8_t* wavData = (uint8_t*)ps_malloc(*wavSize);
    if (wavData == nullptr) {
        Serial.println("✗ Failed to allocate WAV buffer!");
        return nullptr;
    }
    
    memcpy(wavData + 0, "RIFF", 4);
    *(uint32_t*)(wavData + 4) = *wavSize - 8;
    memcpy(wavData + 8, "WAVE", 4);
    memcpy(wavData + 12, "fmt ", 4);
    *(uint32_t*)(wavData + 16) = 16;
    *(uint16_t*)(wavData + 20) = 1;
    *(uint16_t*)(wavData + 22) = 1;
    *(uint32_t*)(wavData + 24) = SAMPLE_RATE;
    *(uint32_t*)(wavData + 28) = SAMPLE_RATE * 2;
    *(uint16_t*)(wavData + 32) = 2;
    *(uint16_t*)(wavData + 34) = 16;
    memcpy(wavData + 36, "data", 4);
    *(uint32_t*)(wavData + 40) = dataSize;
    
    memcpy(wavData + headerSize, audioBuffer, dataSize);
    
    return wavData;
}

void classifyAudio() {
    Serial.println("\n===========================================");
    Serial.println(" Sending to Flask API");
    Serial.println("===========================================");
    
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("✗ WiFi not connected!");
        return;
    }
    
    size_t wavSize;
    uint8_t* wavData = createWAV(&wavSize);
    if (wavData == nullptr) {
        return;
    }
    
    Serial.printf("WAV Size: %d bytes\n", wavSize);
    
    HTTPClient http;
    String url = String(API_URL) + String(API_ENDPOINT);
    Serial.printf("URL: %s\n", url.c_str());
    
    http.begin(url);
    http.setTimeout(15000);
    
    String boundary = "----ESP32Boundary";
    String contentType = "multipart/form-data; boundary=" + boundary;
    http.addHeader("Content-Type", contentType);
    
    String header = "--" + boundary + "\r\n";
    header += "Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n";
    header += "Content-Type: audio/wav\r\n\r\n";
    
    String footer = "\r\n--" + boundary + "--\r\n";
    
    size_t totalSize = header.length() + wavSize + footer.length();
    uint8_t* postData = (uint8_t*)ps_malloc(totalSize);
    
    if (postData == nullptr) {
        Serial.println("✗ Failed to allocate POST buffer!");
        free(wavData);
        return;
    }
    
    memcpy(postData, header.c_str(), header.length());
    memcpy(postData + header.length(), wavData, wavSize);
    memcpy(postData + header.length() + wavSize, footer.c_str(), footer.length());
    
    Serial.println("Sending HTTP POST...");
    int httpCode = http.POST(postData, totalSize);
    
    if (httpCode == HTTP_CODE_OK) {
        String response = http.getString();
        Serial.println("\n✓ Classification successful!");
        Serial.println("\n-------------------------------------------");
        Serial.println(" RESULTS");
        Serial.println("-------------------------------------------");
        
        JsonDocument doc;
        DeserializationError error = deserializeJson(doc, response);
        
        if (!error) {
            const char* soundClass = doc["class"];
            float confidence = doc["confidence"];
            
            JsonObject decibels = doc["decibels"];
            float dbSpl = decibels["db_spl_estimate"];
            const char* description = decibels["description"];
            
            Serial.printf("Sound Class:  %s\n", soundClass);
            Serial.printf("Confidence:   %.2f%%\n", confidence);
            Serial.printf("Decibels:     %.2f dB SPL\n", dbSpl);
            Serial.printf("Noise Level:  %s\n", description);
            
            JsonObject computation = doc["computation"];
            int cluster = computation["cluster_assigned"];
            float distance = computation["distance_to_centroid"];
            
            Serial.printf("\nCluster:      %d\n", cluster);
            Serial.printf("Distance:     %.4f\n", distance);
        } else {
            Serial.println("Raw response:");
            Serial.println(response);
        }
        
        Serial.println("-------------------------------------------");
    } else {
        Serial.printf("✗ HTTP Error: %d\n", httpCode);
        if (httpCode > 0) {
            Serial.println(http.getString());
        }
    }
    
    http.end();
    free(postData);
    free(wavData);
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n\n");
    Serial.println("===========================================");
    Serial.println(" ZonoTrack ESP32 Cloud Classifier");
    Serial.println("===========================================");
    Serial.println("Approach: Cloud Computing (Flask API)");
    Serial.println();
    
    setupI2S();
    
    connectWiFi();
    
    Serial.println("\n===========================================");
    Serial.println(" Ready!");
    Serial.println("===========================================");
    Serial.printf("Classification interval: %d ms\n", CLASSIFICATION_INTERVAL);
}

void loop() {
    if (WiFi.status() == WL_CONNECTED) {
        if (recordAudio()) {
            classifyAudio();
        }
        
        Serial.printf("\nWaiting %d seconds...\n", CLASSIFICATION_INTERVAL / 1000);
        delay(CLASSIFICATION_INTERVAL);
    } else {
        Serial.println("WiFi disconnected. Reconnecting...");
        connectWiFi();
        delay(5000);
    }
}
