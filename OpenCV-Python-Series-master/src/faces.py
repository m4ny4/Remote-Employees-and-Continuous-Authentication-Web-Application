import cv2
import pickle
import getpass
import os

current_directory = os.getcwd()

def perform_facial_recognition():
    username = getpass.getuser()
    print(f"The Python script is running under the username: {username}")

    face_cascade = cv2.CascadeClassifier(os.path.join(current_directory, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml'))
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(current_directory, 'recognizers', 'face-trainner.yml'))
    labels = {"person_name": 1}
    with open(os.path.join(current_directory, 'pickles', 'face-labels.pickle'), 'rb') as f:
         og_labels = pickle.load(f)
         labels = {v:k for k,v in og_labels.items()}
    print(og_labels)

# Print key and value for each item in the labels dictionary
    for key, value in labels.items():
         print(f"Key: {key}, Value: {value}")
    cap = cv2.VideoCapture(0)
    smallest_area=float(1000000)
    smallest_name=""

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                face_area = w * h

                id_, conf = recognizer.predict(roi_gray)
                if conf >= 12 and conf <= 85:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    display_text = f"{name} - {conf}"
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(frame, display_text, (x, y), font, 1, color, stroke, cv2.LINE_AA)

            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

            return name

    cap.release()
    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    #perform_facial_recognition()
    nam = perform_facial_recognition()
    print(nam)
    print(type(nam))