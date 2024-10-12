import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as td
import os
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
from keras import layers, models


print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices("GPU")[0], True
)

metadata = pd.read_csv(
    "C:\\Users\\ansh_p2jfg5q\\Downloads\\archive\\HAM10000_metadata.csv"
)
metadata["image_id"] = metadata["image_id"] + ".jpg"

data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=-0.1,
    horizontal_flip=True,
    validation_split=0.2,
)

image_folder = "C:\\Users\\ansh_p2jfg5q\\Downloads\\archive\\HAM10000_images_part_1"

train_generator = data_gen.flow_from_dataframe(
    dataframe=metadata,
    directory=image_folder,
    x_col='image_id',
    y_col='dx',
    target_size=(450, 600),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = data_gen.flow_from_dataframe(
    dataframe=metadata,
    directory=image_folder,
    x_col='image_id',
    y_col='dx',
    target_size=(450, 600),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(450, 600, 3)),  # Feature extraction
    layers.MaxPooling2D((2, 2)),                                            # Downsampling
    layers.Conv2D(64, (3, 3), activation='relu'),                           # More feature extraction
    layers.MaxPooling2D((2, 2)),                                            # Further downsampling
    layers.Flatten(),                                                        # Convert 2D feature maps to 1D
    layers.Dense(64, activation='relu'),                                    # Intermediate representation
    layers.Dense(7, activation='softmax')                                   # Final classification (7 classes)
])

model.compile(optimizer='adam', #the "best" optimizer out there
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10 
)