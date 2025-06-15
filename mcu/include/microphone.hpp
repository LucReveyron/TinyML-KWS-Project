#ifndef MICROPHONE_HPP
#define MICROPHONE_HPP

#include <Arduino.h>
#include <driver/i2s.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

#define SAMPLE_BUFFER_SIZE 640  // Same value as in training preprocessing
#define SAMPLE_RATE 16000       // Based on dataset SpeechCommand.v1 

#define I2S_MIC_SERIAL_CLOCK GPIO_NUM_26
#define I2S_MIC_LEFT_RIGHT_CLOCK GPIO_NUM_32
#define I2S_MIC_SERIAL_DATA GPIO_NUM_33

static int16_t pcm_buffer[SAMPLE_BUFFER_SIZE];
static QueueHandle_t pcm_queue;

void setup_mic();
void read_mic(int32_t* raw_samples, size_t* bytes_read);
void start_mic_task();

#endif // MICROPHONE_HPP