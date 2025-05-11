# datasets.py
import tensorflow as tf

def load_dataset(directory, image_size=(64, 64), batch_size=32):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int',  # for sparse_categorical_crossentropy
        shuffle=True
    )
    return dataset