#include <Arduino.h>
#include "microphone.hpp"
#include "model_data.h"

#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"

// Audio settings
constexpr int kSampleRate = 16000;
constexpr int kWindowSizeMs = 40;    // 40 ms
constexpr int kWindowStepMs = 20;    // 20 ms
constexpr int kWindowSize = (kSampleRate * kWindowSizeMs) / 1000;   // 640 samples
constexpr int kWindowStep = (kSampleRate * kWindowStepMs) / 1000;   // 320 samples

// MFCC settings
constexpr int kNumMelBins = 40;
constexpr float kLowerFreqLimit = 20.0f;
constexpr float kUpperFreqLimit = 4000.0f;
constexpr int  kNumMfccCoeffs = 13;         // MFCC coefficients per slice (matches your model input shape)
constexpr int kNumMfccSlices = 49;          // Number of MFCC slices needed by your model input
int feature_index = 0;                      // Tracks current filled slice index

// TFLite arena size
constexpr int kTensorArenaSize = 56000;     // Try and error
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Globals
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// Frontend state
FrontendState frontend_state;
FrontendConfig frontend_config;

// Buffers
int16_t audio_window[kWindowSize];  // sliding window
float mfcc_out[kNumMfccCoeffs];

// Shared mic buffer (define here to match microphone.cpp externs)
int16_t shared_buffer[SAMPLE_BUFFER_SIZE];
SemaphoreHandle_t buffer_mutex;

// Compute DCT-II for MFCC
void ComputeDCT(const float* in, float* out, int in_size, int out_size) 
{
  for (int k = 0; k < out_size; ++k) {
    float sum = 0.0f;
    for (int n = 0; n < in_size; ++n) {
      sum += in[n] * cosf(PI * k * (2 * n + 1) / (2 * in_size));
    }
    float scale = (k == 0) ? sqrtf(1.0f / in_size) : sqrtf(2.0f / in_size);
    out[k] = sum * scale;
  }
}

struct QuantParams {
  float scale;
  int zero_point;
};

QuantParams GetInputQuantParams() 
{
  QuantParams q;
  q.scale = input_tensor->params.scale;
  q.zero_point = input_tensor->params.zero_point;
  return q;
}

// Quantize MFCC to int8
int8_t QuantizeMFCC(float val) 
{
  // Adjust these to your model's quantization params
  const float scale = 0.1f;
  const int32_t zero_point = 0;
  int32_t q = static_cast<int32_t>(roundf(val / scale)) + zero_point;
  return static_cast<int8_t>(q < -128 ? -128 : (q > 127 ? 127 : q));
}

void SetupFrontend() 
{
  LogScaleConfig log_cfg;
  NoiseReductionConfig nr_cfg;
  PcanGainControlConfig pcan_cfg;
  LogScaleFillConfigWithDefaults(&log_cfg);
  NoiseReductionFillConfigWithDefaults(&nr_cfg);
  PcanGainControlFillConfigWithDefaults(&pcan_cfg);

  frontend_config.window = WindowConfig{kWindowSizeMs, kWindowStepMs};
  frontend_config.filterbank = FilterbankConfig{kNumMelBins, kUpperFreqLimit, kLowerFreqLimit};
  frontend_config.noise_reduction = nr_cfg;
  frontend_config.pcan_gain_control = pcan_cfg;
  frontend_config.log_scale = log_cfg;

  if (!FrontendPopulateState(&frontend_config, &frontend_state, kSampleRate)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to init frontend");
    while (1);
  }
}

void SetupModel() 
{

  model = tflite::GetModel(model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model schema mismatch");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      static_cast<size_t>(kTensorArenaSize),
      /* resource_variables */ nullptr,
      /* profiler */ nullptr);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    while (1);
  }
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // Print input quantization parameters for debugging
   Serial.printf("Input scale=%.6f, zero_point=%d", input_tensor->params.scale,input_tensor->params.zero_point);
}

void setup() {
  Serial.begin(9600);

  // Initialize microphone and mutex
  buffer_mutex = xSemaphoreCreateMutex();
  if (!buffer_mutex) {
    Serial.println("Failed to create buffer mutex");
    while (1);
  }
  setup_mic();
  start_mic_task();

  SetupFrontend();
  SetupModel();

  Serial.println("Initialization complete");
}

void loop() 
{
  if (xSemaphoreTake(buffer_mutex, portMAX_DELAY)) {
    // Slide audio window
    memmove(audio_window, audio_window + kWindowStep, (kWindowSize - kWindowStep) * sizeof(int16_t));

    for (int i = 0; i < kWindowStep; ++i) {
        int32_t s = shared_buffer[i];
        audio_window[kWindowSize - kWindowStep + i] = static_cast<int16_t>(max(min(s, 32767), -32768));
      }

      xSemaphoreGive(buffer_mutex);
    }
    // Process frontend to get filterbank energies (mel bins)
    size_t samples_read = 0;
    FrontendOutput fe_out = FrontendProcessSamples(&frontend_state, audio_window, kWindowSize, &samples_read);

    // Check if frontend output matches mel bins count
    if (fe_out.size == kNumMelBins) {
      // Copy mel bins to float array

      float mel[kNumMelBins];
      for (size_t i = 0; i < kNumMelBins; ++i) {
        mel[i] = static_cast<float>(fe_out.values[i]);
      }

      // Compute MFCC coefficients from mel bins
      ComputeDCT(mel, mfcc_out, kNumMelBins, kNumMfccCoeffs);

      // Quantize MFCC and store in input tensor slice at feature_index
      QuantParams q = GetInputQuantParams();
      int8_t* in_data = input_tensor->data.int8;
      int base = feature_index * kNumMfccCoeffs;
      for (int i = 0; i < kNumMfccCoeffs; ++i) {
        int32_t quant_val = (int32_t)roundf(mfcc_out[i] / q.scale) + q.zero_point;
        if (quant_val < -128) quant_val = -128;
        else if (quant_val > 127) quant_val = 127;
        in_data[base + i] = static_cast<int8_t>(quant_val);
      }
      feature_index++;

      // Once enough slices collected, run inference
      if (feature_index >= kNumMfccSlices) {
        TfLiteStatus status = interpreter->Invoke();
        if (status != kTfLiteOk) {
          TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
          feature_index = 0;  // reset on failure
          return;
        }

        // Print output tensor results
        int8_t* out = output_tensor->data.int8;
        Serial.print("Output: ");
        for (int i = 0; i < output_tensor->dims->data[1]; ++i) {
          Serial.printf("%d ", out[i]);
        }
        Serial.println();

        // Slide the window of MFCC slices in input tensor for next inference
        memmove(in_data,in_data + kNumMfccCoeffs,(kNumMfccSlices - 1) * kNumMfccCoeffs);

        feature_index = kNumMfccSlices - 1;
      }
    }

  delay(kWindowStepMs);
}
