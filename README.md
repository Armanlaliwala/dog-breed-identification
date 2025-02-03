Image Classification with VGG16

This project demonstrates image classification using the VGG16 model pre-trained on the ImageNet dataset. The application allows users to classify images using both a command-line script and a Streamlit web interface.

Features

Uses VGG16 for image classification

Supports JPEG, PNG image formats

Provides a command-line script for image classification

Includes a Streamlit web app for user-friendly interaction

Installation

Prerequisites

Ensure you have Python 3.7+ installed. Then, install the required dependencies:

pip install tensorflow streamlit pillow numpy

Usage

1️⃣ Command-Line Script

Run the script to classify an image:

python classify.py

2️⃣ Streamlit Web App

Run the Streamlit app to classify images via a web interface:

streamlit run streamlit_app.py

Upload an image, and the model will predict its category.

File Structure

📂 vgg16_image_classification
├── classify.py          # Command-line script for image classification
├── streamlit_app.py     # Streamlit web application
├── requirements.txt     # Required dependencies
└── README.md            # Documentation

Example Output

🎯 Command Line

Predicted: Golden Retriever

🌍 Streamlit App

Upload an image.

The model displays the predicted category with confidence score.

Credits

Uses VGG16 from tensorflow.keras.applications

Developed using Streamlit and TensorFlow

License

This project is open-source under the MIT License.

