/*
 * ZonoTrack ESP32 Cloud Configuration
 * 
 * Update these settings for your network and server
 */

#ifndef CONFIG_H
#define CONFIG_H

#define WIFI_SSID "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

#define API_URL "http://192.168.1.100:5000"
#define API_ENDPOINT "/api/predict"

#define I2S_WS    42
#define I2S_SD    41
#define I2S_SCK   40

#define SAMPLE_RATE   22050
#define RECORD_TIME   3
#define I2S_PORT      I2S_NUM_0
#define BITS_PER_SAMPLE 16

#define BUFFER_SIZE (SAMPLE_RATE * RECORD_TIME)

#define CLASSIFICATION_INTERVAL 5000

#endif // CONFIG_H
