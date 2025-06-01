import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
from PIL import Image
import tempfile

# Load model
with open("faceEmotion.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("faceEmotion.h5")

# Haar Cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

# Feature extraction function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature / 255.0

# Streamlit interface
st.title("Face Emotion Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    img_array = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_features = extract_features(roi_resized)
        prediction = model.predict(roi_features)
        label = labels[prediction.argmax()]
        
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Predicted Emotion", use_column_width=True)
