#include <Arduino.h>
#include "microphone.hpp"

int32_t raw_samples[SAMPLE_BUFFER_SIZE] = {0}; // Store microphone raw output
size_t bytes_read = 0; // store number of bytes read

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  delay(500);
  Serial.println("Start ESP-EYE...");

  setup_mic();
}

void loop() {
  // put your main code here, to run repeatedly:
  
  read_mic(raw_samples, &bytes_read);

  int samples_read = bytes_read / sizeof(int32_t);
  // dump the samples out to the serial channel.
  for (int i = 0; i < 10; i++)
  {
    Serial.printf("%ld\n", raw_samples[i]);
  }
}
