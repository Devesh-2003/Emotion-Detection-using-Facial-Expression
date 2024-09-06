import cv2
from keras.models import model_from_json
import numpy as np

# Load the pre-trained model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = image.reshape(1, 48, 48, 1) / 255.0
    return feature

# Open the webcam
webcam = cv2.VideoCapture(0)

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, frame = webcam.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))

        # Extract features and make prediction
        img = extract_features(face_roi)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, prediction_label, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

    # Display the result
    cv2.imshow("Emotion Detection", frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
