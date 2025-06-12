#ifndef MICROPHONE_HPP
#define MICROPHONE_HPP

#include <Arduino.h>
#include <driver/i2s.h>

#define SAMPLE_BUFFER_SIZE 512
#define SAMPLE_RATE 40000 // Fixed by hardware 

#define I2S_MIC_SERIAL_CLOCK GPIO_NUM_26
#define I2S_MIC_LEFT_RIGHT_CLOCK GPIO_NUM_32
#define I2S_MIC_SERIAL_DATA GPIO_NUM_33

void setup_mic();
void read_mic(int32_t* raw_samples, size_t* bytes_read);

#endif // EXAMPLE_HPP