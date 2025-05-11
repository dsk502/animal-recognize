# train.py
import tensorflow as tf
from model import build_model
from datasets import load_dataset

# Parameters
train_dir = 'animal_dataset/train'
eval_dir = 'animal_dataset/test'
image_size = (64, 64)
batch_size = 32
epochs = 10

# Load datasets
train_ds = load_dataset(train_dir, image_size, batch_size)
eval_ds = load_dataset(eval_dir, image_size, batch_size)

# Determine number of classes from dataset
class_names = train_ds.class_names
num_classes = len(class_names)

# Build and compile model
model = build_model(input_shape=image_size + (3,), num_classes=num_classes)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=epochs
)

# Save model
model.save('saved_model/animal_classifier')
