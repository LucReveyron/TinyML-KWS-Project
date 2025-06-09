import os
import json
import datetime
import tensorflow as tf
from tensorflow.keras import callbacks
import utils
import models

# Constant
EPOCHS = 40
BATCH = 128

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

    print("Starting data augmentation...")
    aug_files, aug_labels = utils.augment_dataset_with_noise(
        train_paths, train_labels, bg_noises, augmented_dir, augment_factor, snr_range
    )
    print(f"Generated {len(aug_files)} augmented samples")

    # Combine original and augmented training data
    train_paths = train_paths + aug_files
    train_labels = train_labels + aug_labels

    # Prepare tensorflow pipeline 
    # # Unzip the lists of (filepath, label) tuples
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

    # Prepare model
    num_classes = len(subset_labels)
    #model = models.build_cnn_mfcc_model((None, 13), num_classes)

    # if your MFCCs are (49 time steps, 13 features)
    model = models.build_cnn_mfcc_model((49, 13), num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
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
    model.save(f"{model_dir + "/model_{run_name}"}.keras")

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