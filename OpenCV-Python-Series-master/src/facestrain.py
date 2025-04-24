import cv2
import os
import numpy as np
from PIL import Image
import pickle
import time
from timeit import default_timer as timer
from PIL import Image
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import os
import tempfile
tempfile.tempdir = 'D:/temp'


recognizer = None
label_ids = None
current_directory = os.getcwd()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
best_accuracy = 0


def augment_data(image, label):
    augmented_images = []
    augmented_labels = []
    
    # Apply rotation
    for angle in [-10, 0, 10]:
        rotated_image = rotate_image(image, angle)
        augmented_images.append(rotated_image)
        augmented_labels.append(label)
    
    # Apply horizontal flip
    flipped_image = cv2.flip(image, 1)
    augmented_images.append(flipped_image)
    augmented_labels.append(label)
    
    # Apply shifting
    for shift in [(-20, 0), (20, 0), (0, -20), (0, 20)]:
        shifted_image = shift_image(image, shift)
        augmented_images.append(shifted_image)
        augmented_labels.append(label)
    
    return augmented_images, augmented_labels

def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

def shift_image(image, shift):
    shift_matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    shifted_image = cv2.warpAffine(image, shift_matrix, (image.shape[1], image.shape[0]))
    return shifted_image

def train_classifier2():
    global recognizer
    global label_ids
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(current_directory, "images")

    #face_cascade = cv2.CascadeClassifier(os.path.join(current_directory, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml'))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(BASE_DIR, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                #print(f"Processing: {label} - {path}")

                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]

                pil_image = Image.open(path).convert("L")  # grayscale
                size = (550, 550)
                final_image = pil_image.resize(size, Image.LANCZOS)
                image_array = np.array(final_image, "uint8")

                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    
                    # Augment data
                    augmented_images, augmented_labels = augment_data(roi, id_)
                    
                    # Add augmented data to training set
                    for augmented_image in augmented_images:
                        x_train.append(augmented_image)
                    y_labels.extend(augmented_labels)

    
    # with open(os.path.join(current_directory, 'pickles', 'face-labels.pickle'),'wb') as f:
    with open(os.path.join(BASE_DIR, 'pickles', 'face-labels.pickle'), 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    # recognizer.save(os.path.join(current_directory, 'recognizers', 'face-trainner.yml'))
    recognizer.save(os.path.join(BASE_DIR, 'recognizers', 'face-trainner.yml'))

# def perform_facial_recognition_on_directory(directory_path,scale_factor,min_neighbours):
#     #print(f"The Python script is running under the username: {username}")

#     # cascade_path = os.path.join(current_directory, 'src', 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')
#     # face_cascade = cv2.CascadeClassifier(cascade_path)

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     cascade_path = os.path.join(BASE_DIR, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')
#     face_cascade = cv2.CascadeClassifier(cascade_path)

#     #face_cascade = cv2.CascadeClassifier(cv2.CascadeClassifier(os.path.join(current_directory, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')))
#     #side_cascade = cv2.CascadeClassifier
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     # recognizer.read(os.path.join(current_directory, 'recognizers', 'face-trainner.yml'))
#     recognizer.read(os.path.join(BASE_DIR, 'recognizers', 'face-trainner.yml'))

#     labels = {"person_name": 1}
#     # with open(os.path.join(current_directory, 'pickles', 'face-labels.pickle'), 'rb') as f:
#     with open(os.path.join(BASE_DIR, 'pickles', 'face-labels.pickle'), 'rb') as f:
#         og_labels = pickle.load(f)
#         labels = {v: k for k, v in og_labels.items()}
#     #print(og_labels)

#     correct_predictions = 0
#     total_predictions = 0

#     for person_folder in os.listdir(directory_path):
#         person_path = os.path.join(directory_path, person_folder)
#         if os.path.isdir(person_path):
#             for file in os.listdir(person_path):
#                 if file.endswith("png") or file.endswith("jpg"):
#                     path = os.path.join(person_path, file)

#                     pil_image = Image.open(path).convert("L")
#                     size = (550, 550)
#                     final_image = pil_image.resize(size, Image.LANCZOS)
#                     image_array = np.array(final_image, "uint8")

#                     # faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
#                     #nope
#                     #was this: before im changing on 6/9/2024
#                     faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.8, minNeighbors=4)
#                     #faces = face_cascade.detectMultiScale(image_array, scaleFactor=scale_factor, minNeighbors=min_neighbours)

#                     for (x, y, w, h) in faces:
#                         roi_gray = image_array[y:y + h, x:x + w]
#                         id_, conf = recognizer.predict(roi_gray)
#                         predicted_label = labels[id_]

#                         actual_label = person_folder.lower()  # Assuming the folder name is the person's label

#                         #print(f"Actual: {actual_label}, Predicted: {predicted_label}, Confidence: {conf}")

#                         total_predictions += 1
#                         if actual_label == predicted_label:
#                             #and conf >= 50 and conf <= 85:
#                             correct_predictions += 1

#     accuracy = (correct_predictions / total_predictions) * 100
#     #print(f"Accuracy: {accuracy}%")
#     return accuracy, total_predictions

def perform_facial_recognition_on_directory(directory_path, scale_factor, min_neighbors, radius, neighbors, grid_x, grid_y):
    import os
    import pickle
    import numpy as np
    from PIL import Image
    import cv2

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(BASE_DIR, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=radius,
        neighbors=neighbors,
        grid_x=grid_x,
        grid_y=grid_y
    )
    recognizer.read(os.path.join(BASE_DIR, 'recognizers', 'face-trainner.yml'))

    with open(os.path.join(BASE_DIR, 'pickles', 'face-labels.pickle'), 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    correct_predictions = 0
    total_predictions = 0

    for person_folder in os.listdir(directory_path):
        person_path = os.path.join(directory_path, person_folder)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(person_path, file)
                    image = Image.open(path).convert("L").resize((550, 550), Image.LANCZOS)
                    image_array = np.array(image, "uint8")

                    faces = face_cascade.detectMultiScale(image_array, scaleFactor=scale_factor, minNeighbors=min_neighbors)

                    for (x, y, w, h) in faces:
                        roi = image_array[y:y + h, x:x + w]
                        id_, _ = recognizer.predict(roi)
                        predicted_label = labels.get(id_, "unknown")
                        actual_label = person_folder.lower()

                        total_predictions += 1
                        if predicted_label == actual_label:
                            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions else 0
    return accuracy, total_predictions



def retRecognizer():
    global recognizer
    return recognizer

def retLabel():
    global label_ids
    return label_ids

def train_classifier():
    global recognizer
    global label_ids
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")

    # face_cascade = cv2.CascadeClassifier(os.path.join(current_directory, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml'))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(BASE_DIR, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)


    
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                #print(f"Processing: {label} - {path}")

                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]

                pil_image = Image.open(path).convert("L")  # grayscale
                size = (550, 550)
                final_image = pil_image.resize(size, Image.LANCZOS)
                image_array = np.array(final_image, "uint8")

                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)

    # with open(os.path.join(current_directory, 'pickles', 'face-labels.pickle'), 'wb') as f:
    with open(os.path.join(BASE_DIR, 'pickles', 'face-labels.pickle'), 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    # recognizer.save(os.path.join(current_directory, 'recognizers', 'face-trainner.yml'))
    recognizer.save(os.path.join(BASE_DIR, 'recognizers', 'face-trainner.yml'))

def train_classifier_without_aug(scale_factor,min_neighbours):
    global recognizer
    global label_ids
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")

    #face_cascade = cv2.CascadeClassifier(cv2.CascadeClassifier(os.path.join(current_directory, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(BASE_DIR, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                #print(f"Processing: {label} - {path}")

                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]

                pil_image = Image.open(path).convert("L")  # grayscale
                size = (550, 550)
                final_image = pil_image.resize(size, Image.LANCZOS)
                image_array = np.array(final_image, "uint8")

                faces = face_cascade.detectMultiScale(image_array, scaleFactor=scale_factor, minNeighbors=min_neighbours)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)

    # with open(os.path.join(current_directory, 'pickles', 'face-labels.pickle'), 'wb') as f:
    with open(os.path.join(BASE_DIR, 'pickles', 'face-labels.pickle'), 'wb') as f:
        pickle.dump(label_ids, f)
 
    recognizer.train(x_train, np.array(y_labels))
    # recognizer.save(os.path.join(current_directory, 'recognizers', 'face-trainner.yml'))
    recognizer.save(os.path.join(BASE_DIR, 'recognizers', 'face-trainner.yml'))

# def find_optimal(directory):
#     accuracy = 0
#     best_accuracy=0
#     finalSF = 0
#     finalMN = 0
#     for scale_factor in np.arange(1.01,2.0,0.1):
#         for min_neighbours in range(1,11):
#             print("trainingt classifier WITH aug & standard VALUE of directory sf and mn")
#             train_classifier_optimal(scale_factor,min_neighbours)
#             print("predicting it")
#             accuracy,totalPred = perform_facial_recognition_on_directory(directory,scale_factor,min_neighbours)
#             print("accuracy received, checking if optimal")
#             if (accuracy > best_accuracy) and (totalPred >=19):
#                 best_accuracy = accuracy
#                 finalSF = scale_factor
#                 finalMN = min_neighbours
#                 print(f"Best accuracy is: {accuracy}")
#                 print(f"For SF {finalSF}")
#                 print(f"For MN {finalMN}")
#     print(f"FINAL Best accuracy is: {accuracy}")
#     print(f"FINAL For SF {finalSF}")
#     print(f"FINAL For MN {finalMN}")

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import os
from skopt.space import Categorical

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define hyperparameter space
space = [
    Real(1.1, 1.9, name='scale_factor'),
    Integer(3, 10, name='min_neighbors'),
    Integer(1, 3, name='radius'),
    Categorical([8, 16], name='neighbors'),
    Integer(8, 10, name='grid_x'),
    Integer(8, 10, name='grid_y'),
]

@use_named_args(space)
def objective(scale_factor, min_neighbors, radius, neighbors, grid_x, grid_y):
    global best_accuracy  # access the shared variable
    print(f"\nTrying: SF={scale_factor}, MN={min_neighbors}, R={radius}, N={neighbors}, GX={grid_x}, GY={grid_y}")

    try:
        train_classifier_optimal(scale_factor, min_neighbors, radius, neighbors, grid_x, grid_y)
        accuracy, total_pred = perform_facial_recognition_on_directory(
            os.path.join(BASE_DIR, "imagesTest"),
            scale_factor,
            min_neighbors,
            radius,
            neighbors,
            grid_x,
            grid_y
        )

        print(f"Accuracy: {accuracy:.2f}%, Predictions: {total_pred}")

        if accuracy > best_accuracy and total_pred >= 15:
            best_accuracy = accuracy
            print(f"🎯 NEW BEST ACCURACY: {accuracy:.2f}%")

        if total_pred < 15:
            return 1.0
        return 1.0 - accuracy / 100.0

    except Exception as e:
        print("❌ Error:", e)
        return 1.0


# def train_classifier_optimal(scale_factor,min_neighbours):
#     global recognizer
#     global label_ids
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     image_dir = os.path.join(BASE_DIR, "images")

#     #face_cascade = cv2.CascadeClassifier(cv2.CascadeClassifier(os.path.join(current_directory, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')))
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     cascade_path = os.path.join(BASE_DIR, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')
#     face_cascade = cv2.CascadeClassifier(cascade_path)


    
#     recognizer = cv2.face.LBPHFaceRecognizer_create()

#     current_id = 0
#     label_ids = {}
#     y_labels = []
#     x_train = []

#     for root, dirs, files in os.walk(image_dir):
#         for file in files:
#             if file.endswith("png") or file.endswith("jpg"):
#                 path = os.path.join(root, file)
#                 label = os.path.basename(root).replace(" ", "-").lower()
#                 #print(f"Processing: {label} - {path}")

#                 if not label in label_ids:
#                     label_ids[label] = current_id
#                     current_id += 1
#                 id_ = label_ids[label]

#                 pil_image = Image.open(path).convert("L")  # grayscale
#                 size = (550, 550)
#                 final_image = pil_image.resize(size, Image.LANCZOS)
#                 image_array = np.array(final_image, "uint8")


#                 faces = face_cascade.detectMultiScale(image_array, scaleFactor=scale_factor, minNeighbors=min_neighbours)

#                 for (x, y, w, h) in faces:
#                     roi = image_array[y:y + h, x:x + w]

#                     #print("augmenting data")
                    
#                     # Augment data
#                     augmented_images, augmented_labels = augment_data(roi, id_)
                    
#                     # Add augmented data to training set
#                     for augmented_image in augmented_images:
#                         x_train.append(augmented_image)
#                     y_labels.extend(augmented_labels)

#     # with open(os.path.join(current_directory, 'pickles', 'face-labels.pickle'), 'wb') as f:
#     with open(os.path.join(BASE_DIR, 'pickles', 'face-labels.pickle'), 'wb') as f:
#         pickle.dump(label_ids, f)

#     recognizer.train(x_train, np.array(y_labels))
#     print("saving it")
#     # recognizer.save(os.path.join(current_directory, 'recognizers', 'face-trainner.yml'))
#     recognizer.save(os.path.join(BASE_DIR, 'recognizers', 'face-trainner.yml'))

def train_classifier_optimal(scale_factor, min_neighbors, radius, neighbors, grid_x, grid_y):
    import os
    import pickle
    import numpy as np
    from PIL import Image
    import cv2

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")
    cascade_path = os.path.join(BASE_DIR, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=radius,
        neighbors=neighbors,
        grid_x=grid_x,
        grid_y=grid_y
    )

    current_id = 0
    label_ids = {}
    x_train, y_labels = [], []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()

                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]

                image = Image.open(path).convert("L").resize((550, 550), Image.LANCZOS)
                image_array = np.array(image, "uint8")

                faces = face_cascade.detectMultiScale(image_array, scaleFactor=scale_factor, minNeighbors=min_neighbors)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)

    with open(os.path.join(BASE_DIR, 'pickles', 'face-labels.pickle'), 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(os.path.join(BASE_DIR, 'recognizers', 'face-trainner.yml'))


def perform_facial_recognition_with_multiple_predictions(directory_path):
    face_cascade = cv2.CascadeClassifier(cv2.CascadeClassifier(os.path.join(current_directory, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')))
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(current_directory, 'recognizers', 'face-trainer.yml'))

    labels = {}
    # with open(os.path.join(current_directory, 'pickles', 'face-labels.pickle'), 'rb') as f:
    with open(os.path.join(BASE_DIR, 'pickles', 'face-labels.pickle'), 'wb') as f:
        labels = pickle.load(f)

    correct_predictions = 0
    total_predictions = 0

    for person_folder in os.listdir(directory_path):
        person_path = os.path.join(directory_path, person_folder)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(person_path, file)

                    pil_image = Image.open(path).convert("L")
                    size = (550, 550)
                    final_image = pil_image.resize(size, Image.LANCZOS)
                    image_array = np.array(final_image, "uint8")

                    faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.8, minNeighbors=4)

                    for (x, y, w, h) in faces:
                        roi_gray = image_array[y:y + h, x:x + w]
                        predictions = []

                        # Get top predictions for the detected face
                        id_confidences = recognizer.predict(roi_gray, return_labels=True)
                        for id_, conf in zip(id_confidences[0], id_confidences[1]):
                            predicted_label = labels[id_]
                            predictions.append((predicted_label, conf))

                        # Sort predictions based on confidence scores
                        predictions.sort(key=lambda x: x[1], reverse=True)

                        # Print top predictions
                        for i, (predicted_label, confidence) in enumerate(predictions[:2], start=1):
                            actual_label = person_folder.lower()
                            print(f"Actual: {actual_label}, Predicted {i}: {predicted_label}, Confidence: {confidence}")
                            total_predictions += 1
                            if actual_label == predicted_label:
                                correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy}%")
    return accuracy, total_predictions

def perform_facial_recognition_with_multiple_predictions2(directory_path):
    face_cascade = cv2.CascadeClassifier(cv2.CascadeClassifier(os.path.join(current_directory, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')))
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(current_directory, 'recognizers', 'face-trainer.yml'))

    labels = {}
    # with open(os.path.join(current_directory, 'pickles', 'face-labels.pickle'), 'rb') as f:
    with open(os.path.join(BASE_DIR, 'pickles', 'face-labels.pickle'), 'wb') as f:
        labels = pickle.load(f)

    correct_predictions = 0
    total_predictions = 0

    for person_folder in os.listdir(directory_path):
        person_path = os.path.join(directory_path, person_folder)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(person_path, file)

                    pil_image = Image.open(path).convert("L")
                    size = (550, 550)
                    final_image = pil_image.resize(size, Image.LANCZOS)
                    image_array = np.array(final_image, "uint8")

                    faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.8, minNeighbors=4)

                    for (x, y, w, h) in faces:
                        roi_gray = image_array[y:y + h, x:x + w]
                        predictions = []

                        # Get top predictions for the detected face
                        id_, conf = recognizer.predict(roi_gray)
                        # print("Predicted ID:", id_) 
                        # print("Predicted id TYPE:", type(id_))
                        # print("KEY associated is: ", list(labels.keys())[list(labels.values()).index(id_)])
                        predicted_label = list(labels.keys())[list(labels.values()).index(id_)]
                        #labels[id_]
                        predictions.append((predicted_label, conf))

                        # Print top predictions
                        for i, (predicted_label, confidence) in enumerate(predictions[:2], start=1):
                            actual_label = person_folder.lower()
                            print(f"Actual: {actual_label}, Predicted {i}: {predicted_label}, Confidence: {confidence}")
                            total_predictions += 1
                            if actual_label == predicted_label:
                                correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy}%")
    return accuracy, total_predictions

def perform_facial_recognition_extended(directory_path):
    #print(f"The Python script is running under the username: {username}")

    face_cascade = cv2.CascadeClassifier(cv2.CascadeClassifier(os.path.join(current_directory, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml')))
    #side_cascade = cv2.CascadeClassifier
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(current_directory, 'recognizers', 'face-trainer.yml'))

    labels = {"person_name": 1}
    with open(os.path.join(current_directory, 'pickles', 'face-labels.pickle'), 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}
    #print(og_labels)

    correct_predictions = 0
    total_predictions = 0

    for person_folder in os.listdir(directory_path):
        person_path = os.path.join(directory_path, person_folder)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(person_path, file)

                    pil_image = Image.open(path).convert("L")
                    size = (550, 550)
                    final_image = pil_image.resize(size, Image.LANCZOS)
                    image_array = np.array(final_image, "uint8")

                    # faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
                    #nope
                    print("SF: 1.8 and MN: 10")
                    faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.01, minNeighbors=10)

                    for (x, y, w, h) in faces:
                        roi_gray = image_array[y:y + h, x:x + w]
                        id_, conf = recognizer.predict(roi_gray)
                        predicted_label = labels[id_]

                        actual_label = person_folder.lower()  # Assuming the folder name is the person's label

                        print(f"Actual: {actual_label}, Predicted: {predicted_label}, Confidence: {conf}")

                        total_predictions += 1
                        if actual_label == predicted_label:
                            #and conf >= 50 and conf <= 85:
                            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy}%")
    return accuracy, total_predictions




# if __name__ == "__main__":
#     print("In main")
#     if not os.path.exists('D:/temp'):
#         os.makedirs('D:/temp')

#     start = timer()
#     #train_classifier2()
#     end = timer()
#     directory=os.path.join(BASE_DIR, "imagesTest")
#     find_optimal(directory)
#     #test_classifier()

if __name__ == "__main__":
    from skopt.plots import plot_convergence
    import matplotlib.pyplot as plt

    print("🚀 Running Bayesian optimization with skopt...")
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=15,
        random_state=42,
        verbose=True
    )

    print("\n🏆 Best Parameters:")
    for name, val in zip([dim.name for dim in space], result.x):
        print(f"{name}: {val}")
    print(f"\n📈 Best Accuracy: {100 * (1 - result.fun):.2f}%")

