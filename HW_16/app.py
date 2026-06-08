import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from PIL import Image


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


def load_history(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def preprocess_for_cnn(image):
    image = image.convert("L")
    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32") / 255.0

    if image_array.mean() > 0.5:
        image_array = 1.0 - image_array

    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


def preprocess_for_vgg16(image):
    image = image.convert("L")
    image = image.resize((48, 48))

    image_array = np.array(image).astype("float32") / 255.0

    if image_array.mean() > 0.5:
        image_array = 1.0 - image_array

    image_array = np.stack([image_array, image_array, image_array], axis=-1)
    image_array = image_array.reshape(1, 48, 48, 3)

    return image_array


def plot_history(history):
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(history["loss"], label="Training loss")
    ax_loss.plot(history["val_loss"], label="Validation loss")
    ax_loss.set_title("Функція втрат")
    ax_loss.set_xlabel("Епоха")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    st.pyplot(fig_loss)

    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(history["accuracy"], label="Training accuracy")
    ax_acc.plot(history["val_accuracy"], label="Validation accuracy")
    ax_acc.set_title("Точність моделі")
    ax_acc.set_xlabel("Епоха")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    st.pyplot(fig_acc)


st.title("Класифікація зображень Fashion MNIST")

st.write(
    "Веб-застосунок дозволяє завантажити зображення та класифікувати його "
    "за допомогою однієї з двох моделей: звичайної CNN або моделі на основі VGG16."
)

model_choice = st.selectbox(
    "Оберіть модель",
    ["Згорткова нейромережа CNN", "Модель на основі VGG16"]
)

uploaded_file = st.file_uploader(
    "Завантажте зображення",
    type=["png", "jpg", "jpeg"]
)

if model_choice == "Згорткова нейромережа CNN":
    model = load_model("models/cnn_model.keras")
    history = load_history("history/cnn_history.pkl")
else:
    model = load_model("models/vgg16_model.keras")
    history = load_history("history/vgg16_history.pkl")

st.subheader("Графіки навчання моделі")
plot_history(history)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Завантажене зображення")
    st.image(image, caption="Вхідне зображення", use_container_width=True)

    if model_choice == "Згорткова нейромережа CNN":
        processed_image = preprocess_for_cnn(image)
    else:
        processed_image = preprocess_for_vgg16(image)

    predictions = model.predict(processed_image)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[predicted_index]

    st.subheader("Результат класифікації")

    st.write(f"Передбачений клас: **{predicted_class}**")
    st.write(f"Ймовірність: **{confidence:.2%}**")

    results_df = pd.DataFrame({
        "Клас": CLASS_NAMES,
        "Ймовірність": predictions
    })

    results_df["Ймовірність (%)"] = results_df["Ймовірність"] * 100
    results_df = results_df.sort_values(by="Ймовірність", ascending=False)

    st.dataframe(results_df[["Клас", "Ймовірність (%)"]], use_container_width=True)

    st.subheader("Ймовірності для кожного класу")
    st.bar_chart(results_df.set_index("Клас")["Ймовірність (%)"])
else:
    st.info("Завантажте зображення для класифікації.")