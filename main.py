import os
import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras
from keras import layers

dataset_path = "./"

def load_dataset(path):
    data = []
    labels = []
    emotions = os.listdir(path)
    for emotion in emotions:
        emotion_path = os.path.join(path, emotion)
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            data.append(img)
            labels.append(emotions.index(emotion))
    return np.array(data), np.array(labels)

X_train, y_train = load_dataset(os.path.join(dataset_path, "train"))

X_train = X_train / 255.0

model = keras.Sequential([
    layers.Flatten(input_shape=(48, 48)),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

cap = cv2.VideoCapture(0)

emotions = ["angry", "fear", "disgust", "happy", "neutral", "sad", "surprise"]

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48))

        prediction = model.predict(roi_gray)
        emotion_label = emotions[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
