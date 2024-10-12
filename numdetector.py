import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers, models
import matplotlib.pyplot as plt


(train, test), info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label


train = train.map(normalize_image).batch(32)
test = test.map(normalize_image).batch(32)

model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",  # the "best" optimizer out there
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(train, validation_data=test, epochs=5)
