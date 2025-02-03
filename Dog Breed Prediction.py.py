from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load a pre-trained model
model = VGG16()

# Load and preprocess the image
image = load_img('dog1.jpg', target_size=(224, 224))  # Resize image
image_array = img_to_array(image)                   # Convert to array
image_array = preprocess_input(image_array)         # Prepare image for model
image_array = image_array.reshape((1, 224, 224, 3)) # Add batch dimension

# Predict the class
predictions = model.predict(image_array)
decoded_predictions = decode_predictions(predictions)

# Print the top prediction
print("Predicted:", decoded_predictions[0][0][1])
