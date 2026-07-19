import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("skin_disease_model.h5")
class_names = ['bcc', 'melanoma', 'nevus']

IMG_SIZE = 224
camera_running = False

# Main Window
root = tk.Tk()
root.title("Skin Lension Detection")
root.geometry("900x700")
root.configure(bg="#f4f6f7")

# Heading
heading = Label(
    root,
    text="Skin Lension Detection",
    font=("Arial", 28, "bold"),
    bg="#f4f6f7",
    fg="#2c3e50"
)
heading.pack(pady=20)

# Video Frame
video_label = Label(root)
video_label.pack()

# Prediction Label
result_label = Label(
    root,
    text="Prediction: ",
    font=("Arial", 18, "bold"),
    bg="#f4f6f7",
    fg="black"
)
result_label.pack(pady=10)

confidence_label = Label(
    root,
    text="Confidence: ",
    font=("Arial", 16),
    bg="#f4f6f7"
)
confidence_label.pack()

cap = None


def start_camera():
    global cap, camera_running
    cap = cv2.VideoCapture(0)
    camera_running = True
    update_frame()


def stop_camera():
    global camera_running
    camera_running = False
    if cap:
        cap.release()
    video_label.config(image='')


def update_frame():
    global camera_running
    if camera_running:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            # Resize for prediction
            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img_array = np.expand_dims(img, axis=0) / 255.0

            prediction = model.predict(img_array, verbose=0)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            # Update labels
            result_label.config(text=f"Prediction: {predicted_class}")

            if predicted_class == "melanoma":
                result_label.config(fg="red")
            else:
                result_label.config(fg="green")

            confidence_label.config(text=f"Confidence: {confidence:.2f}%")

            # Convert frame to Tkinter format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)

            video_label.imgtk = img_tk
            video_label.configure(image=img_tk)

        video_label.after(10, update_frame)


# Buttons
start_btn = Button(
    root,
    text="Start Camera",
    font=("Arial", 14, "bold"),
    bg="#27ae60",
    fg="white",
    command=start_camera
)
start_btn.pack(pady=10)

stop_btn = Button(
    root,
    text="Stop Camera",
    font=("Arial", 14, "bold"),
    bg="#c0392b",
    fg="white",
    command=stop_camera
)
stop_btn.pack()

root.mainloop()