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
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)

    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

def build_cnn_lstm_mfcc_model(input_shape, num_classes):
    """
    CNN + LSTM hybrid model for keyword spotting.
    
    Args:
        input_shape: Tuple of shape (time_steps, feature_bins)
        num_classes: Number of output classes

    Returns:
        A compiled tf.keras.Model
    """
    # Input shape: (time_steps, feature_bins)
    inputs = tf.keras.Input(shape=(*input_shape, 1))  # Add channel dimension

    # 1st Conv block
    x = layers.Conv2D(60, (10, 4), strides=(1, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 2nd Conv block with stride to reduce time resolution
    x = layers.Conv2D(76, (10, 4), strides=(2, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Reshape for LSTM: (batch, time_steps, features)
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    # LSTM layer
    x = layers.LSTM(58, activation='tanh', recurrent_activation='sigmoid')(x)

    # Dense layer
    x = layers.Dense(128, activation='relu')(x)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)

