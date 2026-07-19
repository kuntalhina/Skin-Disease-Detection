import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Load model
model = tf.keras.models.load_model("skin_disease_model.h5")

# Get class names from training folder
class_names = sorted(os.listdir("dataset/train"))
print("Classes:", class_names)

# Image path (change if needed)
img_path = "masa.jpeg"

# Load image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print("Prediction probabilities:", prediction)
print("Predicted Class:", predicted_class)