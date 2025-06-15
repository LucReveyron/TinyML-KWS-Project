import os
import json
import numpy as np
import datetime
import random
import tensorflow as tf
from tensorflow.keras import callbacks
import utils
import models

# Constant
EPOCHS = 100
BATCH = 64

IS_AUG_NEEDED = True
INCREASE_SAMPLES = False

def main():

    dataset_url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
    download_dir = os.path.join("..", "data")
    extract_dir = download_dir + "/speech_commands"
    utils.download_and_extract_dataset(dataset_url, download_dir, extract_dir)

    subset_labels = ['yes', 'no', 'stop', 'go', 'on', 'off']

    subset_files = []
    for label in subset_labels:
        folder_path = os.path.join(extract_dir, label)
        if os.path.exists(folder_path):
            files = tf.io.gfile.glob(os.path.join(folder_path, '*.wav'))
            subset_files.extend([(f, label) for f in files])
        else:
            print(f"Warning: Label folder '{label}' not found!")
    print(f"Total files in subset: {len(subset_files)}")

    all_folders = [f for f in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, f))]
    all_labels = [label for label in all_folders if label != "_background_noise_"]

    unknown_labels = [label for label in all_labels if label not in subset_labels]
    print(f"Available unknown labels: {unknown_labels}")

    unknown_files = []
    for label in unknown_labels:
        folder_path = os.path.join(extract_dir, label)
        files = tf.io.gfile.glob(os.path.join(folder_path, '*.wav'))
        unknown_files.extend([(f, label) for f in files])
    print(f"Total unknown files found: {len(unknown_files)}")

    validation_percentage = 15.0
    testing_percentage = 15.0

    # Split known files by set type
    train_files = []
    val_files = []
    test_files = []

    for file_path, label in subset_files:
        set_type = utils.which_set_wrapper(file_path, validation_percentage, testing_percentage)
        if set_type == 'training':
            train_files.append((file_path, label))
        elif set_type == 'validation':
            val_files.append((file_path, label))
        else:
            test_files.append((file_path, label))

    print(f"Training set (known): {len(train_files)} files")
    print(f"Validation set (known): {len(val_files)} files")
    print(f"Testing set (known): {len(test_files)} files")

    # Function to count known samples per split (total count, not per label)
    def count_label_samples(files):
        counts = {}
        for _, label in files:
            counts[label] = counts.get(label, 0) + 1
        return counts

    train_counts = count_label_samples(train_files)
    val_counts = count_label_samples(val_files)
    test_counts = count_label_samples(test_files)

    # Total known samples per split (sum over known labels)
    train_known_total = sum(train_counts.values())
    val_known_total = sum(val_counts.values())
    test_known_total = sum(test_counts.values())

    # Shuffle unknown files once for all splits
    random.shuffle(unknown_files)

    # Split unknown files according to percentages
    unknown_train_files = []
    unknown_val_files = []
    unknown_test_files = []

    for file_path, label in unknown_files:
        set_type = utils.which_set_wrapper(file_path, validation_percentage, testing_percentage)
        if set_type == 'training':
            unknown_train_files.append((file_path, label))
        elif set_type == 'validation':
            unknown_val_files.append((file_path, label))
        else:
            unknown_test_files.append((file_path, label))

    # Sample unknown files to match known counts per split
    def sample_unknown(unknown_list, target_count):
        # If unknown_list is smaller, just use all
        return unknown_list[:target_count] if len(unknown_list) >= target_count else unknown_list

    unknown_train_sampled = sample_unknown(unknown_train_files, train_known_total)
    unknown_val_sampled = sample_unknown(unknown_val_files, val_known_total)
    unknown_test_sampled = sample_unknown(unknown_test_files, test_known_total)

    # Label unknown files as 'unknown'
    unknown_train_sampled = [(f, "unknown") for f, _ in unknown_train_sampled]
    unknown_val_sampled = [(f, "unknown") for f, _ in unknown_val_sampled]
    unknown_test_sampled = [(f, "unknown") for f, _ in unknown_test_sampled]

    # Add unknown to known split files
    train_files.extend(unknown_train_sampled)
    val_files.extend(unknown_val_sampled)
    test_files.extend(unknown_test_sampled)

    print(f"Training set after adding unknown: {len(train_files)} files")
    print(f"Validation set after adding unknown: {len(val_files)} files")
    print(f"Testing set after adding unknown: {len(test_files)} files")

    # Prepare paths and labels lists
    train_paths = [f for f, l in train_files]
    train_labels = [l for f, l in train_files]

    val_paths = [f for f, l in val_files]
    val_labels = [l for f, l in val_files]

    test_paths = [f for f, l in test_files]
    test_labels = [l for f, l in test_files]

    # Create label list with unknown
    all_labels = sorted(set(train_labels + val_labels + test_labels))

    label_to_index = {label: idx for idx, label in enumerate(all_labels)}
    print(f"Labels used for training: {all_labels}")

    train_labels_int = [label_to_index[label] for label in train_labels]
    val_labels_int = [label_to_index[label] for label in val_labels]
    test_labels_int = [label_to_index[label] for label in test_labels]

    # Load background noises for augmentation
    bg_noise_dir = os.path.join(extract_dir, "_background_noise_")
    bg_noises = utils.load_bg_noises(bg_noise_dir)

    # Prepare augmentation output directory
    augmented_dir = os.path.join(extract_dir, "augmented_train")

    augment_factor = 1
    snr_range = (15, 25)  # SNR in dB
    aug_ratio = 0.1       # 10% augmentation

    if IS_AUG_NEEDED:
        print("Starting data augmentation...")
        aug_files, aug_labels = utils.augment_dataset_with_noise(
            train_paths, train_labels, bg_noises, augmented_dir, augment_factor, snr_range
        )
        print(f"Generated {len(aug_files)} augmented samples")

        # Sample 10% of original training size from augmented data
        num_aug_samples = int(len(train_paths) * aug_ratio)

        combined = list(zip(aug_files, aug_labels))
        random.shuffle(combined)
        aug_files_subset, aug_labels_subset = zip(*combined[:num_aug_samples])

        train_paths += list(aug_files_subset)
        train_labels += list(aug_labels_subset)

        # Final shuffle
        combined_final = list(zip(train_paths, train_labels))
        random.shuffle(combined_final)
        train_paths, train_labels = zip(*combined_final)
        train_paths, train_labels = list(train_paths), list(train_labels)

        # Update integer labels after augmentation
        train_labels_int = [label_to_index[label] for label in train_labels]

    # Prepare TensorFlow datasets

    train_ds = utils.prepare_dataset(train_paths, train_labels_int, is_training=True, batch_size=BATCH)
    val_ds = utils.prepare_dataset(val_paths, val_labels_int, is_training=True, batch_size=BATCH)
    test_ds = utils.prepare_dataset(test_paths, test_labels_int, is_training=True, batch_size=BATCH)

    # Save MFCC features for quantization
    mfcc_train_save_path = "../convert/mfcc_samples.npy"
    mfcc_test_save_path = "../convert/mfcc_test_samples.npy"
    test_labels_save_path = "../convert/mfcc_test_labels.npy"

    if not os.path.exists(mfcc_train_save_path):
        mfcc_train_list = []
        for features, _ in train_ds.take(1000):
            mfcc_train_list.extend(features.numpy())
        np.save(mfcc_train_save_path, np.array(mfcc_train_list))
        print(f"Saved {len(mfcc_train_list)} MFCC training samples to {mfcc_train_save_path}")
    else:
        print(f"Found existing file: {mfcc_train_save_path}, skipping save.")

    if not (os.path.exists(mfcc_test_save_path) and os.path.exists(test_labels_save_path)):
        mfcc_test_list = []
        test_labels_list = []
        for features, labels in test_ds:
            mfcc_test_list.extend(features.numpy())
            test_labels_list.extend(labels.numpy())
        np.save(mfcc_test_save_path, np.array(mfcc_test_list))
        np.save(test_labels_save_path, np.array(test_labels_list))
        print(f"Saved {len(mfcc_test_list)} test MFCCs to {mfcc_test_save_path}")
        print(f"Saved {len(test_labels_list)} test labels to {test_labels_save_path}")
    else:
        print(f"Found existing test dataset files, skipping save.")

    # Prepare model
    num_classes = len(all_labels)

    #model = models.build_lstm_keyword_model((49, 13), num_classes)
    model = models.build_cnn_mfcc_model((49, 13), num_classes)
    #model = models.build_cnn_lstm_mfcc_model((49, 13), num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[lr_scheduler, early_stop],
        verbose=1   
    )   

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.3f}")

    # Save model and training steps
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = f"saved_models"
    history_json_path = f"training_history/history_{run_name}.json"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(history_json_path), exist_ok=True)

    model.save(f"{model_dir + f'/model_{run_name}'}.keras")

    with open(history_json_path, "w") as f:
        json.dump(history.history, f)

    output_prefix = f"saved_images/metrics_{run_name}"

    try:
        utils.save_training_plots(history.history, output_prefix)
    except Exception as e:
        print(f"Warning: Failed to save training plots: {e}")

    try:
        utils.save_confusion_matrix(model, test_ds, output_prefix, class_names=all_labels)
    except Exception as e:
        print(f"Warning: Failed to save confusion matrix: {e}")

if __name__ == "__main__":
    main()

"""" Old version
def main():
    
    # Download and extract dataset if needed
    dataset_url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
    download_dir = os.path.join("..", "data")
    extract_dir = download_dir + "/speech_commands"
    utils.download_and_extract_dataset(dataset_url, download_dir, extract_dir)

    # Keep only the subset of labels we want to classify
    subset_labels = ['yes', 'no', 'stop', 'go', 'on', 'off']

    subset_files = []
    for label in subset_labels:
        folder_path = os.path.join(extract_dir, label)
        if os.path.exists(folder_path):
            files = tf.io.gfile.glob(os.path.join(folder_path, '*.wav'))
            subset_files.extend([(f, label) for f in files])
        else:
            print(f"Warning: Label folder '{label}' not found!")

    print(f"Total files in subset: {len(subset_files)}")

    # Split between Training, validation and testing
    validation_percentage = 15.0
    testing_percentage = 15.0

    train_files = []
    val_files = []
    test_files = []

    for file_path, label in subset_files:
        set_type = utils.which_set_wrapper(file_path, validation_percentage, testing_percentage)
        if set_type == 'training':
            train_files.append((file_path, label))
        elif set_type == 'validation':
            val_files.append((file_path, label))
        else:
            test_files.append((file_path, label))

    print(f"Training set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    print(f"Testing set: {len(test_files)} files")

    # Data augmentation 
    
    # Load background noises for augmentation
    bg_noise_dir = os.path.join(extract_dir, "_background_noise_")
    bg_noises = utils.load_bg_noises(bg_noise_dir)

    # Prepare augmentation output directory
    augmented_dir = os.path.join(extract_dir, "augmented_train")

    # Extract train files and labels separately
    train_paths = [f for f, l in train_files]
    train_labels = [l for f, l in train_files]

    # Augment training data
    augment_factor = 1  # Number of augmented versions per original
    snr_range = (15, 25) # SNR range in dB

    if(IS_AUG_NEEDED):
        print("Starting data augmentation...")
        aug_files, aug_labels = utils.augment_dataset_with_noise(
            train_paths, train_labels, bg_noises, augmented_dir, augment_factor, snr_range
        )
        print(f"Generated {len(aug_files)} augmented samples")

    if(INCREASE_SAMPLES):
        # Combine original and augmented training data
        train_paths = train_paths + aug_files
        train_labels = train_labels + aug_labels
    else:
        train_paths = aug_files
        train_labels = aug_labels
    
    augment_factor = 1
    snr_range = (15, 25)  # SNR in dB
    aug_ratio = 0.1       # 10% augmentation

    if IS_AUG_NEEDED:
        print("Starting data augmentation...")
        aug_files, aug_labels = utils.augment_dataset_with_noise(
            train_paths, train_labels, bg_noises, augmented_dir, augment_factor, snr_range
        )
        print(f"Generated {len(aug_files)} augmented samples")

        # Sample 10% of the original dataset size from the augmented data
        num_aug_samples = int(len(train_paths) * aug_ratio)

        combined = list(zip(aug_files, aug_labels))
        random.shuffle(combined)

        aug_files_subset, aug_labels_subset = zip(*combined[:num_aug_samples])

        # Combine subset of augmented data with original training data
        train_paths += list(aug_files_subset)
        train_labels += list(aug_labels_subset)

        # Final shuffle
        combined_final = list(zip(train_paths, train_labels))
        random.shuffle(combined_final)
        train_paths, train_labels = zip(*combined_final)
        train_paths, train_labels = list(train_paths), list(train_labels)

    # Prepare tensorflow pipeline 
    # Unzip the lists of (filepath, label) tuples
    #train_paths, train_labels = zip(*train_files)
    val_paths, val_labels = zip(*val_files)
    test_paths, test_labels = zip(*test_files)

    # Convert from tuples to lists if needed
    train_paths, train_labels = list(train_paths), list(train_labels)
    val_paths, val_labels = list(val_paths), list(val_labels)
    test_paths, test_labels = list(test_paths), list(test_labels)

    # Build label to index mapping from all labels combined
    all_labels = sorted(set(train_labels + val_labels + test_labels))
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    # Convert string labels to int
    train_labels_int = [label_to_index[label] for label in train_labels]
    val_labels_int = [label_to_index[label] for label in val_labels]
    test_labels_int = [label_to_index[label] for label in test_labels]

    # Call your prepare_dataset function with correct inputs
    train_ds = utils.prepare_dataset(train_paths, train_labels_int, is_training=True, batch_size=BATCH)
    val_ds = utils.prepare_dataset(val_paths, val_labels_int, is_training=True, batch_size=BATCH)
    test_ds = utils.prepare_dataset(test_paths, test_labels_int, is_training=True, batch_size=BATCH)

    # Save the dataset for quantization 
    # Paths
    mfcc_train_save_path = "../convert/mfcc_samples.npy"
    mfcc_test_save_path = "../convert/mfcc_test_samples.npy"
    test_labels_save_path = "../convert/mfcc_test_labels.npy"

    # --- Save MFCCs from training set (for quantization) ---
    if not os.path.exists(mfcc_train_save_path):
        mfcc_train_list = []
        for features, _ in train_ds.take(1000):  # Adjust sample count if needed
            mfcc_train_list.extend(features.numpy())
        np.save(mfcc_train_save_path, np.array(mfcc_train_list))
        print(f"Saved {len(mfcc_train_list)} MFCC training samples to {mfcc_train_save_path}")
    else:
        print(f"Found existing file: {mfcc_train_save_path}, skipping save.")

    # --- Save MFCCs and labels from test set (for TFLite evaluation) ---
    if not (os.path.exists(mfcc_test_save_path) and os.path.exists(test_labels_save_path)):
        mfcc_test_list = []
        test_labels_list = []

        for features, labels in test_ds:
            mfcc_test_list.extend(features.numpy())
            test_labels_list.extend(labels.numpy())

        np.save(mfcc_test_save_path, np.array(mfcc_test_list))
        np.save(test_labels_save_path, np.array(test_labels_list))
        print(f"Saved {len(mfcc_test_list)} test MFCCs to {mfcc_test_save_path}")
        print(f"Saved {len(test_labels_list)} test labels to {test_labels_save_path}")
    else:
        print(f"Found existing test dataset files, skipping save.")


    # Prepare model
    num_classes = len(subset_labels)
    #model = models.build_lstm_keyword_model((None, 13), num_classes)

    # if your MFCCs are (49 time steps, 13 features)
    model = models.build_cnn_mfcc_model((49, 13), num_classes)
    #model = models.build_cnn_lstm_mfcc_model((49, 13), num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[lr_scheduler, early_stop],
        verbose=1
    )

    # Test model trained
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.3f}") 

    # Save model and training steps
    # Generate timestamped run name
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = f"saved_models"
    history_json_path = f"training_history/history_{run_name}.json"

    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(history_json_path), exist_ok=True)

    # Save the model
    model.save(f"{model_dir + f"/model_{run_name}"}.keras")

    # Save the training history as JSON
    with open(history_json_path, "w") as f:
        json.dump(history.history, f)

    output_prefix = f"saved_images/metrics_{run_name}"

    try:
        utils.save_training_plots(history.history, output_prefix)
    except Exception as e:
        print(f"Warning: Failed to save training plots: {e}")

    try:
        utils.save_confusion_matrix(model, test_ds, output_prefix, class_names=subset_labels)
    except Exception as e:
        print(f"Warning: Failed to save confusion matrix: {e}")

if __name__ == "__main__":
    main()
"""
