# (HW_16) Fashion MNIST Image Classification Web App

This project is a Streamlit web application for visualizing the work of trained neural networks for image classification using the Fashion MNIST dataset.

The application allows users to upload an image, select a trained model, and receive a prediction with class probabilities.

## Project Structure

```text
HW_16/
│
├── history/
│   ├── cnn_history.json
│   └── vgg16_history.json
│
├── image_for_test/
│   └── test images
│
├── models/
│   ├── cnn_model.keras
│   └── vgg16_model.keras
│
├── result/
│   ├── cnn_accuracy.png
│   ├── cnn_loss.png
│   ├── vgg16_accuracy.png
│   └── vgg16_loss.png
│
├── .gitignore
└── app.py
```

## Project Description

The goal of this project is to create a web application that demonstrates image classification using neural networks trained on the Fashion MNIST dataset.

The application supports two models:

* Custom Convolutional Neural Network from Part 1;
* VGG16-based model from Part 2.

The user can upload an image, choose one of the available models, and get the predicted class with probabilities for all classes.

## Features

* Upload an image for classification;
* Display the uploaded image on the web page;
* Choose between two models:

  * CNN model;
  * VGG16-based model;
* Predict the class of the uploaded image;
* Display class probabilities;
* Display training accuracy and loss graphs.

## Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* NumPy
* Pillow
* Matplotlib
* Fashion MNIST Dataset

## Models

### CNN Model

The custom CNN model was trained on the Fashion MNIST dataset and saved as:

```text
models/cnn_model.keras
```

This model is included in the project repository.

### VGG16-based Model

The VGG16-based model was also trained for image classification, but the file:

```text
models/vgg16_model.keras
```

is not included in the GitHub repository because its size is too large.

GitHub has a file size limit, and the VGG16 model file exceeds this limit.
Because of this, the file was added to `.gitignore`.

To run the VGG16 model option locally, you need to train the model again or manually place the file into the `models/` folder:

```text
HW_16/models/vgg16_model.keras
```

If this file is missing, only the CNN model can be used.

## Installation

Install the required dependencies:

```bash
pip install streamlit tensorflow numpy pillow matplotlib
```

## Running the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

After running the command, the application will open in the browser.

## How to Use

1. Run the Streamlit application.
2. Upload an image for classification.
3. Select the model:

   * CNN model;
   * VGG16-based model.
4. View the uploaded image.
5. Check the predicted class.
6. Review the probabilities for each class.
7. View the training accuracy and loss graphs.

## Fashion MNIST Classes

The model predicts one of the following classes:

```text
0 - T-shirt/top
1 - Trouser
2 - Pullover
3 - Dress
4 - Coat
5 - Sandal
6 - Shirt
7 - Sneaker
8 - Bag
9 - Ankle boot
```

## Important Note

The file `vgg16_model.keras` is not included in this repository because it is too large for GitHub.

To use the VGG16-based model in the application, add the trained model manually to:

```text
models/vgg16_model.keras
```

Otherwise, use the CNN model option.
