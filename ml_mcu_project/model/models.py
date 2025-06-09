import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.utils.class_weight import compute_class_weight

def build_lstm_keyword_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Optional: Normalize input (if needed)
    x = layers.BatchNormalization()(inputs)

    # LSTM layer (you can use return_sequences=True if stacking more LSTMs)
    x = layers.LSTM(128, return_sequences=False, 
                    recurrent_activation='sigmoid',  # use peephole equivalent behavior
                    activation='tanh',
                    name='lstm')(x)

    # Projection layer to reduce dimensionality (simulates output projection)
    x = layers.Dense(64, activation='relu', name='projection')(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_cnn_mfcc_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=(*input_shape, 1)) 

    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

