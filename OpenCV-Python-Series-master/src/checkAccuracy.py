import cv2
import os
import numpy as np
from PIL import Image
import pickle
from facestrain import train_classifier, retRecognizer, retLabel
import os
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input
import os
import pickle
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import tensorflow as tf

def test_LBPHclassifier():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(BASE_DIR, "imagesTest")

    face_cascade = cv2.CascadeClassifier('D:\TCS_YF\Research\OpenCV-Python-Series-master\OpenCV-Python-Series-master\src\cascades\data\haarcascade_frontalface_alt2.xml')
    recognizer = retRecognizer()
    label_ids = retLabel()
    print(f"Label ids ALREADY HAD ARE: {label_ids}")

    # Load label_ids from training
    with open("D:\\TCS_YF\\Research\\OpenCV-Python-Series-master\\OpenCV-Python-Series-master\\src\\pickles\\face-labels.pickle", 'rb') as f:
        label_ids = pickle.load(f)
    print(f"After loading from pickle Label ids are: {label_ids}")

    # Create a dictionary to store label_ids during testing
    test_label_ids = {}

    correct_predictions = 0
    total_predictions = 0

    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                print(f"Testing: {label} - {path}")

                # Assign a new ID for unseen labels during testing
                if label not in label_ids:
                    label_ids[label] = len(label_ids)
                if label not in test_label_ids:
                    test_label_ids[label] = len(test_label_ids)

                # id_ = label_ids[label]
                # print(f"Stored id is: {id_}")

                # for i in test_label_ids:
                #     print(f"i in test_label_id is: {i}")

                pil_image = Image.open(path).convert("L")  # grayscale
                size = (550, 550)
                final_image = pil_image.resize(size, Image.LANCZOS)
                image_array = np.array(final_image, "uint8")

                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    id_, confidence = recognizer.predict(roi)

                    # print(f"Printed id is: {id_}")

                    print(f"FOR ELEMENT IN LABELS ID:")
                    print(f"FOR ELEMENT IN LABELS ID: {list(label_ids.keys())[list(label_ids.values()).index(4)]}")


                    # Use test_label_ids to get the label for the ID
                    #predicted_label = test_label_ids[id_]
                    if (list(label_ids.keys())[list(label_ids.values()).index(4)]) in test_label_ids:
                        print("DETECTED")
                        predicted_label = list(label_ids.keys())[list(label_ids.values()).index(4)]
                    elif (list(label_ids.keys())[list(label_ids.values()).index(4)]) in label_ids:
                        print("NOT DETECTED")
                        predicted_label = list(label_ids.keys())[list(label_ids.values()).index(4)]
                    # predicted_label = test_label_ids[id_] if id_ in test_label_ids else label_ids[id_]

                    print(f"Actual: {label}, Predicted: {predicted_label}, Confidence: {confidence}")

                    if label == predicted_label:
                        correct_predictions += 1

                    total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy for LBPH is: {accuracy}%")

def test_VGG16classifier():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(BASE_DIR, "imageTest2")

    # Load the trained VGG16 model
    saved_model_path = 'D:\TCS_YF\Research\OpenCV-Python-Series-master\OpenCV-Python-Series-master\src\VGGFaceRecoModel.h5'
    model = load_model(saved_model_path)

    # Load label-to-id mapping dictionary from pickle
    # label_to_id_path = 'D:\\TCS_YF\\Research\\OpenCV-Python-Series-master\\OpenCV-Python-Series-master\\src\\pickles\\face-labelsVGG16.pickle'
    # with open(label_to_id_path, 'rb') as f:
    #     label_to_id = pickle.load(f)
    label_to_id_path = 'D:\\TCS_YF\\Research\\OpenCV-Python-Series-master\\OpenCV-Python-Series-master\\src\\pickles\\face-labelsVGG16.pickle'
    with open(label_to_id_path, 'rb') as f:
        label_to_id = pickle.load(f)

    correct_predictions = 0
    total_predictions = 0

    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                print(f"Testing: {label} - {path}")

                pil_image = Image.open(path).convert("RGB") 
                img = pil_image.resize((224, 224))
                #img_gray = tf.image.rgb_to_grayscale(np.array(img))
                image_array = np.array(img)
                # Reshape the image to have a single channel
                image_array = preprocess_input(image_array)  # preprocess input for VGG16

                # Perform prediction using the VGG16 model
                prediction = model.predict(np.expand_dims(image_array, axis=0))
                predicted_label_id = np.argmax(prediction)
                predicted_label = [k for k, v in label_to_id.items() if v == predicted_label_id][0]

                print(f"Actual: {label}, Predicted: {predicted_label}")

                if label == predicted_label.lower():
                    correct_predictions += 1

                total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy for VGG16 is: {accuracy}%")

def test_ResNet50_classifier():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(BASE_DIR, "imageTest2")

    # Load the saved ResNet50 model
    saved_model_path = 'D:\\TCS_YF\\Research\\OpenCV-Python-Series-master\\OpenCV-Python-Series-master\\src\\recognizers\\ResNet50.keras'
    model = load_model(saved_model_path)

    # Load label-to-id mapping dictionary from pickle
    label_to_id_path = 'D:\\TCS_YF\\Research\\OpenCV-Python-Series-master\\OpenCV-Python-Series-master\\src\\pickles\\face-labelsResNet50.pickle'
    with open(label_to_id_path, 'rb') as f:
        label_to_id = pickle.load(f)

    correct_predictions = 0
    total_predictions = 0

    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                print(f"Testing: {label} - {path}")

                pil_image = Image.open(path).convert("RGB") 
                img = pil_image.resize((224, 224))
                image_array = np.array(img)

                # Preprocess the image
                image_array = preprocess_input(image_array)  # preprocess input for ResNet50

                # Perform prediction using the ResNet50 model
                prediction = model.predict(np.expand_dims(image_array, axis=0))
                predicted_label_id = np.argmax(prediction)
                predicted_label = [k for k, v in label_to_id.items() if v == predicted_label_id][0]

                print(f"Actual: {label}, Predicted: {predicted_label}")

                if label == predicted_label.lower():
                    correct_predictions += 1

                total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy for ResNet50 is: {accuracy}%")




if __name__ == "__main__":
    print("In main")
    #train_classifier()
    #test_LBPHclassifier()
    test_VGG16classifier()
    test_ResNet50_classifier()
