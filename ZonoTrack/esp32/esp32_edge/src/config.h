// Configuration for ESP32 Edge Computing
// On-device K-Means sound classification

#ifndef CONFIG_H
#define CONFIG_H

#define I2S_WS    42
#define I2S_SD    41
#define I2S_SCK   40
#define I2S_PORT  I2S_NUM_0

#define SAMPLE_RATE       22050
#define RECORD_TIME       3
#define BUFFER_SIZE       (SAMPLE_RATE * RECORD_TIME)

#define NUM_MFCC          13
#define N_FFT             2048
#define HOP_LENGTH        512
#define NUM_SEGMENTS      10
#define TOTAL_FEATURES    (NUM_MFCC * NUM_SEGMENTS)

#define NUM_MEL_FILTERS   40
#define MEL_MIN_FREQ      0.0f
#define MEL_MAX_FREQ      11025.0f

#define CLASSIFICATION_INTERVAL  5000

#define DEBUG_AUDIO       0
#define DEBUG_MFCC        0
#define DEBUG_INFERENCE   1

#endif // CONFIG_H
