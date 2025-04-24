import cv2
import pickle
import getpass
import numpy as np
from keras.models import load_model
import os

current_directory = os.getcwd()
FACE_CASCADE_PATH = os.path.join(current_directory, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')
EYE_CASCADE_PATH = os.path.join(current_directory, 'cascades', 'data', 'haarcascade_eye.xml')
SMILE_CASCADE_PATH = os.path.join(current_directory, 'cascades', 'data', 'haarcascade_smile.xml')


def preprocess_image(img):
    # Preprocess the image according to the requirements of VGGFaceRecoModel
    # (e.g., resize to 224x224, normalize pixel values)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize pixel values
    return img

def perform_facial_recognition():
    username = getpass.getuser()
    print(f"The Python script is running under the username: {username}")

    # Load the VGGFaceRecoModel
    model = load_model('VGGFaceRecoModel.h5')

    # Labels mapping
    labels = {0: "person_name_1", 1: "person_name_2", ...}  # Update with your label mapping

    cap = cv2.VideoCapture(0)
    smallest_area = float(1000000)
    smallest_name = ""

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                face_area = w * h

                # Preprocess the image
                processed_img = preprocess_image(roi_color)

                # Pass the preprocessed image through the VGGFaceRecoModel
                pred = model.predict(np.array([processed_img]))

                # Decode predictions
                id_ = np.argmax(pred)
                conf = np.max(pred)
                name = labels[id_]

                # Display recognized name and confidence
                if conf >= 0.5:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    display_text = f"{name} - {conf}"
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(frame, display_text, (x, y), font, 1, color, stroke, cv2.LINE_AA)

            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    perform_facial_recognition()
