import os
import tensorflow as tf
import numpy as np
import argparse

def test_tflite_model(tflite_model_path, test_data, test_labels):
    """
    Evaluates a quantized TensorFlow Lite model on test data.

    This function loads a TFLite model, prepares the interpreter, and runs inference on each
    sample of the provided test dataset. It handles the quantization of input data according 
    to the model's input scale and zero-point, and compares predictions to the true labels.

    Args:
        tflite_model_path (str): Path to the .tflite model file.
        test_data (np.ndarray): Numpy array of test input data (e.g., MFCC features).
        test_labels (np.ndarray): Numpy array of true labels corresponding to test_data.

    Returns:
        None. Prints the accuracy of the model on the test set.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']

    correct = 0
    total = len(test_data)

    for i in range(total):
        input_data = test_data[i]

        # Expand dims to match model input shape
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)
        elif input_data.ndim == 2:
            input_data = np.expand_dims(input_data, axis=(0, -1))

        # Quantize input float32 to int8 using scale and zero_point
        input_int8 = np.round(input_data / input_scale + input_zero_point)
        input_int8 = np.clip(input_int8, -128, 127).astype(np.int8)

        # Set the quantized input tensor
        interpreter.set_tensor(input_details[0]['index'], input_int8)
        interpreter.invoke()

        # Get the output and predict label
        output = interpreter.get_tensor(output_details[0]['index'])
        pred_label = np.argmax(output, axis=1)[0]

        if pred_label == test_labels[i]:
            correct += 1

    accuracy = correct / total
    print(f"TFLite model accuracy: {accuracy:.4f}")

def main():
    # Collect arguments
    parser = argparse.ArgumentParser(description="Convert a Keras model to a quantized TFLite model.")
    parser.add_argument("--model", required=True, help="Path to the .keras model")
    parser.add_argument("--dataset", required=True, help="Path to .npy file with representative MFCC samples")
    parser.add_argument("--output", default="model_quant.tflite", help="Path to save the TFLite model")
    
    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(args.model)

    # Test the current model
    test_data = np.load('mfcc_test_samples.npy')    
    test_labels = np.load('mfcc_test_labels.npy')  

    loss, accuracy = model.evaluate(test_data, test_labels, batch_size=32)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

    # Load train set 
    train_data = np.load('mfcc_samples.npy')

    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for input_value in train_data[:1000]:
            input_value = np.expand_dims(input_value, axis=0)  # add batch dim
            if input_value.ndim == 3: 
                input_value = np.expand_dims(input_value, axis=-1)  # add channels dim
            input_value = input_value.astype(np.float32)
            yield [input_value]
        
    converter.representative_dataset = representative_data_gen
    # Specify supported operations to ensure int8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Set input and output tensors to int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    # Save the TFLite model
    with open('model_quant.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model saved as 'model_quant.tflite'")

    # Test converted model
    test_tflite_model('model_quant.tflite', test_data, test_labels)

    # Compare size 
    original_size =  os.path.getsize(args.model)
    quantized_size = os.path.getsize('model_quant.tflite')

    print(f"Original model size: {original_size / 1e6:.2f} MB")
    print(f"Quantized model size: {quantized_size / 1e6:.2f} MB")
    print(f"Size reduction: {(original_size - quantized_size) / original_size * 100:.1f}%")

if __name__ == "__main__":
    main()