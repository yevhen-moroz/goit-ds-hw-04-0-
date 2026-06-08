import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

train_images = tf.expand_dims(train_images, axis=-1)
test_images = tf.expand_dims(test_images, axis=-1)

train_images = tf.image.grayscale_to_rgb(train_images)
test_images = tf.image.grayscale_to_rgb(test_images)

train_images = tf.image.resize(train_images, (48, 48))
test_images = tf.image.resize(test_images, (48, 48))

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

conv_base = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(48, 48, 3)
)

conv_base.trainable = False

model = models.Sequential([
    conv_base,

    layers.Flatten(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=2e-5
    ),
    metrics=["accuracy"]
)

history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

test_loss, test_acc = model.evaluate(
    test_images,
    test_labels
)

print("Feature extraction accuracy:", test_acc)

conv_base.trainable = True

set_trainable = False

for layer in conv_base.layers:
    if layer.name == "block5_conv1":
        set_trainable = True

    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

test_loss, test_acc = model.evaluate(
    test_images,
    test_labels
)

print("Fine tuning accuracy:", test_acc)

acc = history_fine.history["accuracy"]
val_acc = history_fine.history["val_accuracy"]

loss = history_fine.history["loss"]
val_loss = history_fine.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, acc, label="Training accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.title("VGG16 Fine Tuning Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title("VGG16 Fine Tuning Loss")
plt.legend()
plt.grid(True)
plt.show()

import os
import pickle

os.makedirs("models", exist_ok=True)
os.makedirs("history", exist_ok=True)

model.save("models/vgg16_model.keras")

with open("history/vgg16_history.pkl", "wb") as file:
    pickle.dump(history_fine.history, file)