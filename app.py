import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = tf.keras.models.load_model("skin_disease_model.h5")

# Get class names
class_names = sorted(os.listdir("dataset/train"))

# Create main window
root = tk.Tk()
root.title("Skin Lension Detection")
root.geometry("700x600")
root.resizable(False, False)

# ===== Background Image =====
bg_image = Image.open("background.jpg")
bg_image = bg_image.resize((700, 600))
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# ===== Heading =====
heading = tk.Label(
    root,
    text="Skin Lension Detection",
    font=("Arial", 24, "bold"),
    bg="white",
    fg="black"
)
heading.pack(pady=20)

# ===== Image Display =====
image_label = tk.Label(root, bg="white")
image_label.pack(pady=20)

# ===== Result Label =====
result_label = tk.Label(
    root,
    text="Upload an image to classify",
    font=("Arial", 14, "bold"),
    bg="white",
    fg="darkblue"
)
result_label.pack(pady=20)

def upload_image():
    file_path = filedialog.askopenfilename()

    if file_path:
        # Display selected image
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img_display = ImageTk.PhotoImage(img)

        image_label.config(image=img_display)
        image_label.image = img_display

        # Prepare for model
        img_for_model = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img_for_model)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        confidence = np.max(prediction) * 100
        predicted_class = class_names[np.argmax(prediction)]

        # Display result
        result_label.config(
            text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"
        )

# ===== Upload Button =====
upload_btn = tk.Button(
    root,
    text="Upload Image",
    command=upload_image,
    font=("Arial", 14, "bold"),
    bg="#4CAF50",
    fg="white",
    padx=20,
    pady=10
)
upload_btn.pack(pady=20)

root.mainloop()