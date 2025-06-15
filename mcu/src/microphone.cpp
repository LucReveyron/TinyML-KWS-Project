#include "microphone.hpp"

TaskHandle_t micTaskHandle = NULL;

extern int16_t shared_buffer[SAMPLE_BUFFER_SIZE];
extern SemaphoreHandle_t buffer_mutex;

// Init. Microphone I2S communication
void setup_mic()
{
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0};

    i2s_pin_config_t i2s_mic_pins = {
        .bck_io_num = I2S_MIC_SERIAL_CLOCK,
        .ws_io_num = I2S_MIC_LEFT_RIGHT_CLOCK,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_MIC_SERIAL_DATA};

        // Install and start I2S driver
        esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
        if (err != ESP_OK) {
            Serial.printf("Failed to install I2S driver: %d\n", err);
            esp_restart();
        }

        // Configure I2S pins
        err = i2s_set_pin(I2S_NUM_0, &i2s_mic_pins);
        if (err != ESP_OK) {
            Serial.printf("Failed to set I2S pins: %d\n", err);
            esp_restart();
        }

        i2s_zero_dma_buffer(I2S_NUM_0);
}

// Read current microphone output, use for debug
void read_mic(int16_t* raw_samples, size_t* bytes_read)
{
    esp_err_t ret = i2s_read(I2S_NUM_0, raw_samples, sizeof(int16_t) * SAMPLE_BUFFER_SIZE, bytes_read, portMAX_DELAY);
    if (ret != ESP_OK) {
        Serial.printf("I2S read error: %d\n", ret);
        esp_restart();
    }
}

// Collect data of microphone
void mic_task(void* parameter) 
{
  int16_t local_buffer[SAMPLE_BUFFER_SIZE];
  size_t bytes_read = 0;

  while (true) {
    esp_err_t result = i2s_read(I2S_NUM_0, (void*)local_buffer, sizeof(local_buffer), &bytes_read, portMAX_DELAY);

    if (result == ESP_OK) {
      if (xSemaphoreTake(buffer_mutex, portMAX_DELAY)) {
        memcpy(shared_buffer, local_buffer, sizeof(local_buffer));
        xSemaphoreGive(buffer_mutex);
      }
    } 
  else {
    Serial.println("I2S read error");
  }
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}

void start_mic_task() 
{
  xTaskCreatePinnedToCore(
    mic_task,
    "MicTask",
    4096,
    NULL,
    1,
    &micTaskHandle,
    1  // Run on core 1
  );
}