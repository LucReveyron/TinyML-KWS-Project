import os
import re
import urllib.request
import tarfile
import hashlib
import random
import numpy as np
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Constants 
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SAMPLING_FREQUENCY = 16000 # Hz
NBR_MEL_BANDS = 40
LOWER_FREQ_BOUND = 20.0 # Hz
UPPER_FREQ_BOUND = 4000.0 # Hz
STFT_FRAME_LENGTH = 640 # 40 ms window
STFT_HOP_LENGTH = 320 # 20 ms step
NBR_MFCC_COEFF_KEPT = 13 

# Functions

def download_and_extract_dataset(url, download_path, extract_path):
    """
    Downloads and extracts a dataset from a specified URL.

    Args:
        url (str): URL pointing to the dataset archive (e.g., a .zip or .tar.gz file).
        download_path (str): Directory path where the downloaded archive will be saved.
        extract_path (str): Directory path where the contents of the archive will be extracted.

    Returns:
        None
    """
    
    # Create directories if they don't exist
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

    filename = url.split('/')[-1]
    file_path = os.path.join(download_path, filename)

    # Download the file if it doesn't exist
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    else:
        print(f"{filename} already downloaded.")

    # Extract the dataset
    print(f"Extracting {filename}...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print(f"Extraction completed to {extract_path}")

# Inspired from the README of the Speech Commands Data Set v0.01
def which_set_wrapper(filename, validation_percentage, testing_percentage):
    """
    Determines the dataset split (training, validation, or testing) for a given audio filename,
    based on a deterministic hash of the filename. This ensures consistent data splits.

    Args:
        filename (str): Path to the audio file.
        validation_percentage (float): Percentage of data to use for validation (0–100).
        testing_percentage (float): Percentage of data to use for testing (0–100).

    Returns:
        str: One of 'training', 'validation', or 'testing', indicating the split assignment.
    """
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # Encode the string to bytes here before hashing:
    hash_name_bytes = hash_name.encode('utf-8')
    hash_name_hashed = hashlib.sha1(hash_name_bytes).hexdigest()
    
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

# Preproccessing functions

def decode_audio(filename):
    """
    Loads and decodes a WAV audio file into a 1D float32 Tensor.

    Args:
        filename (tf.Tensor or str): Path to the .wav audio file.

    Returns:
        tf.Tensor: A 1D Tensor of type float32 representing the audio waveform.
    """
    audio_binary = tf.io.read_file(filename)
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    return audio

def get_mfcc(waveform):
    """
    Computes Mel-frequency cepstral coefficients (MFCCs) from an audio waveform.

    Args:
        waveform (tf.Tensor): A 1D Tensor representing the raw audio waveform.

    Returns:
        tf.Tensor: A 2D Tensor of shape (num_time_steps, NBR_MFCC_COEFF_KEPT) containing the MFCCs.
    
    The computation includes:
        - Padding/truncating the waveform to 1 second (based on SAMPLING_FREQUENCY).
        - Short-Time Fourier Transform (STFT) to create a spectrogram.
        - Conversion to a Mel spectrogram using mel filter banks.
        - Logarithmic scaling of the Mel spectrogram.
        - Extraction of MFCCs and keeping the first NBR_MFCC_COEFF_KEPT coefficients.
        - Ensuring a fixed number of time steps, calculated as:
              num_time_steps = (SAMPLING_FREQUENCY - STFT_FRAME_LENGTH) // STFT_HOP_LENGTH + 1
    """
    desired_length = SAMPLING_FREQUENCY # e.g. 16000 for 1s of audio
    waveform = waveform[:desired_length]
    padding = desired_length - tf.shape(waveform)[0]
    waveform = tf.pad(waveform, paddings=[[0, padding]])

    spectrogram = tf.signal.stft(waveform, frame_length=STFT_FRAME_LENGTH, frame_step=STFT_HOP_LENGTH)
    mel_spectrogram = tf.abs(spectrogram)

    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins =NBR_MEL_BANDS, 
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate = SAMPLING_FREQUENCY, 
        lower_edge_hertz=LOWER_FREQ_BOUND, 
        upper_edge_hertz=UPPER_FREQ_BOUND
    )

    mel_spectrogram = tf.tensordot(mel_spectrogram, mel_weights, axes=1)
    mel_spectrogram.set_shape(mel_spectrogram.shape[:-1].concatenate([NBR_MEL_BANDS]))

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(tf.math.log(mel_spectrogram + 1e-6))
    mfccs = mfccs[..., :NBR_MFCC_COEFF_KEPT]

    num_time_steps = (desired_length - STFT_FRAME_LENGTH) // STFT_HOP_LENGTH + 1
    mfccs = tf.ensure_shape(mfccs, [num_time_steps, NBR_MFCC_COEFF_KEPT])

    return mfccs

def preprocess(filename, label):
    """
    Loads an audio file, decodes it, and computes its MFCC features.

    Args:
        filename (tf.Tensor): A string tensor representing the path to the audio file.
        label (tf.Tensor): The corresponding label for the audio sample.

    Returns:
        tuple: A tuple (mfcc, label) where:
            - mfcc (tf.Tensor): The MFCC features extracted from the audio.
            - label (tf.Tensor): The original label passed in.
    """
    audio = decode_audio(filename)
    mfcc = get_mfcc(audio)
    return mfcc, label


def prepare_dataset(filenames, labels, is_training=True, batch_size=32):
    """
    Prepares a TensorFlow dataset by mapping preprocessing, batching, shuffling, and prefetching.

    Args:
        filenames (list or tf.Tensor): A list or tensor of audio file paths.
        labels (list or tf.Tensor): A list or tensor of corresponding labels.

    Returns:
        tf.data.Dataset: A TensorFlow dataset containing preprocessed MFCC features and labels,
                         batched and ready for training or evaluation.
    """

    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Data augmentations

def load_bg_noises(bg_noise_folder):
    noises = []
    for f in Path(bg_noise_folder).glob("*.wav"):
        samples, sr = sf.read(f)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)  # convert to mono
        samples = samples.astype(np.float32)
        noises.append((samples, sr))
    return noises

def get_random_noise_clip(noises, length):
    noise, sr = random.choice(noises)
    if len(noise) < length:
        # loop noise to make sure it's long enough
        repeats = int(np.ceil(length / len(noise)))
        noise = np.tile(noise, repeats)
    start_idx = random.randint(0, len(noise) - length)
    return noise[start_idx:start_idx + length]

def mix_with_noise(speech, noise, snr_db):
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))  # normalize

    if len(noise) != len(speech):
        noise = noise[:len(speech)] if len(noise) > len(speech) else np.pad(noise, (0, len(speech) - len(noise)))

    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    desired_noise_power = speech_power / (10 ** (snr_db / 10))
    scale_factor = np.sqrt(desired_noise_power / (noise_power + 1e-10))
    
    augmented = speech + scale_factor * noise
    return np.clip(augmented, -1.0, 1.0)

def augment_dataset_with_noise(train_files, train_labels, bg_noises, augmented_dir,
                               augment_factor=2, snr_range=(15, 25)):
    os.makedirs(augmented_dir, exist_ok=True)
    
    augmented_files = []
    augmented_labels = []
    
    for filepath, label in zip(train_files, train_labels):
        audio, sr = sf.read(filepath)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # mono
        audio = audio.astype(np.float32)
        
        for i in range(augment_factor):
            noise_clip = get_random_noise_clip(bg_noises, len(audio))
            snr_db = random.uniform(*snr_range)
            augmented_audio = mix_with_noise(audio, noise_clip, snr_db)
            
            orig_stem = Path(filepath).stem
            new_filename = f"{orig_stem}_aug{i}.wav"
            save_path = Path(augmented_dir) / new_filename
            
            sf.write(save_path, augmented_audio, sr)
            
            augmented_files.append(str(save_path))
            augmented_labels.append(label)
    
    return augmented_files, augmented_labels

# Plotting functions

def save_training_plots(history, output_path_prefix):
    """
    Saves training and validation accuracy/loss plots as a PNG image.

    Args:
        history (dict): Dictionary containing training history.
        output_path_prefix (str): Path prefix for saving the image (without extension).
    """

    # Ensure the directory exists
    output_dir = os.path.dirname(output_path_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Save image
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_training_plot.png")
    plt.close()

def save_confusion_matrix(model, test_ds, output_path_prefix, class_names=None):
    """
    Computes and saves the confusion matrix as a PNG image.

    Args:
        model (tf.keras.Model): Trained model.
        test_ds (tf.data.Dataset): Batched test dataset.
        output_path_prefix (str): Path prefix for saving the image.
        class_names (list, optional): List of class names for labels.
    """
    y_true, y_pred = [], []

    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred_batch = np.argmax(preds, axis=1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(y_pred_batch)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_path_prefix}_confusion_matrix.png")
    plt.close()

