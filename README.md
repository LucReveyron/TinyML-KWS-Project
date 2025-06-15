# TinyML Keyword Spotting on ESP-EYE

## ✅ Project Workflow Checklist

- [x] Preprocessing dataset
- [x] Training
- [x] Conversion (to TFLite int8 model)
- [x] Microphone capture on ESP-EYE
- [ ] Embedded preprocessing (MFCC)
- [ ] Inference on device


This project demonstrates speech command recognition on the [ESP-EYE](https://www.espressif.com/en/products/devkits/esp-eye/overview) microcontroller. It includes two main components:

- `ml_mcu_project/`: Training pipeline for a quantized speech command model using TensorFlow and the [Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands).
- `mcu/`: Embedded firmware using PlatformIO to deploy the model on ESP-EYE using [Chirale_TensorFlowLite](https://github.com/spaziochirale/Chirale_TensorFlowLite.git).

## 📁 Repository Structure
```
ml-mcu-project
│   README.md
│
├── ml_mcu_project
│   │   # Python training pipeline using Poetry
│   │___ convert 
│   │
│   └─── model
│
└── mcu # PlatformIO project for ESP-EYE firmware
```


## 🔊 Model Training (`ml_mcu_project/`)

This folder trains a quantized int8 model suitable for embedded inference.

### Features

- Uses the Speech Commands dataset
- MFCC preprocessing
- Quantization-aware training (int8)
- Export to `.tflite` for deployment

### Setup

```bash
cd ml_mcu_project
poetry install
```
### Usage

Train and export the model:
```bash
poetry run python train.py
poetry run python export_tflite.py --quantize int8 --output model.tflite
```

## 🎛 Embedded Inference (mcu/)

This folder contains the firmware for the ESP-EYE board.

### Features

- Real-time audio capture at 16 kHz using I2S

- On-device MFCC feature extraction

- Inference with Chirale_TensorFlowLite

- Recognizes predefined speech commands

### Setup

- Install PlatformIO

- Connect your ESP-EYE board

- Build and upload:
```bash
cd mcu
pio run --target upload
```
⚠️ Make sure the model.tflite is converted into a C array and included in the firmware.

## 🧠 Dependencies

    ml_mcu_project/: Python ≥3.8, TensorFlow, NumPy, SciPy (managed by Poetry)

    mcu/: ESP-IDF via PlatformIO, Chirale_TensorFlowLite, CMSIS-DSP

## 📎 References

- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [ESP-EYE Board](https://www.espressif.com/en/products/devkits/esp-eye/overview)
- [Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)

📜 License

MIT License
