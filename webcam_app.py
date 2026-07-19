import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("skin_disease_model.h5")

class_names = ['bcc', 'melanoma', 'nevus']

IMG_SIZE = 224

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'p' to predict | Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show live frame
    cv2.imshow("Live Skin Lesion Detection", frame)

    key = cv2.waitKey(1)

    # Press P to predict
    if key == ord('p'):
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")

    # Press Q to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()