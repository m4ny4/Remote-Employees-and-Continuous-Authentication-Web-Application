import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from PIL import Image
import pickle
import time
from timeit import default_timer as timer
import keras 
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from tensorflow.python.keras.models import Sequential
import numpy as np 
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import keras.optimizers
from keras import regularizers


# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "images")
print("path of images is:",train_path)
#train_path = 'path_to_your_training_data_directory'
saved_model_path = 'D:\\TCS_YF\\Research\\OpenCV-Python-Series-master\\OpenCV-Python-Series-master\\src\\recognizers\\VGGtrainer.h5'
IMAGE_SIZE = (224, 224)

# Image data generator for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Generate training data from the directory
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

# Extract class labels from the data generator
class_labels = list(training_set.class_indices.keys())

# Build a dictionary to map class labels to IDs
label_to_id = {label: idx for idx, label in enumerate(class_labels)}

# Load the pre-trained VGG16 model (excluding top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Freeze the base model layers
for layer in base_model.layers[4:]:
    layer.trainable = True

# Add custom classification layers on top of VGG16
x = Flatten()(base_model.output)
folders = glob('D:\TCS_YF\Research\OpenCV-Python-Series-master\OpenCV-Python-Series-master\src\images\*')
print("length of folders is: ", len(folders))
predictions = Dense(56, activation='softmax')(x)  # num_classes is the number of output classes

# predictions = Dense(256, activation='relu')(x)  # Add a dense layer with 256 neurons
# predictions = Dropout(0.01)(predictions)  # Apply dropout with 50% probability
predictions = Dense(56, activation='softmax')(predictions)  # Final softmax layer with 56 classes

#predictions = Dense(56, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)


# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    # optimizer='adam',
    optimizer = Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    training_set,
    epochs=7,  # Set the number of epochs
    steps_per_epoch=len(training_set)
)

# Save the trained model
model.save(saved_model_path)

# Save the label-to-id mapping dictionary to a pickle file
label_to_id_path = 'D:\\TCS_YF\\Research\\OpenCV-Python-Series-master\\OpenCV-Python-Series-master\\src\\pickles\\face-labelsVGG16.pickle'
with open(label_to_id_path, 'wb') as f:
    pickle.dump(label_to_id, f)

# Plot training loss and accuracy
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['accuracy'], label='train accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

###########################################################################################

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.src.legacy.preprocessing.image import ImageDataGenerator
# from keras.applications.vgg16 import VGG16
# from keras.layers import Flatten, Dense, Dropout
# from keras.models import Model
# from keras.optimizers import Adam
# import pickle
# import keras
# from keras.src.legacy.preprocessing.image import ImageDataGenerator
# from keras.applications.vgg16 import VGG16
# from keras.layers import Flatten, Dense, Dropout
# from keras.models import Model
# import matplotlib.pyplot as plt
# import cv2
# import os
# import numpy as np
# from PIL import Image
# import pickle
# import time
# from timeit import default_timer as timer
# import keras 
# from keras.layers import Input, Lambda, Dense, Flatten
# from keras.models import Model
# from keras.applications.vgg16 import VGG16
# from keras.preprocessing import image
# from tensorflow.python.keras.models import Sequential
# import numpy as np 
# import tensorflow as tf
# from glob import glob
# import matplotlib.pyplot as plt
# from keras.optimizers import Adam
# import keras.optimizers

# # Define paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# train_path = os.path.join(BASE_DIR, "imageTemp")
# saved_model_path = 'D:\\TCS_YF\\Research\\OpenCV-Python-Series-master\\OpenCV-Python-Series-master\\src\\recognizers\\VGGtrainer.h5'
# label_to_id_path = 'D:\\TCS_YF\\Research\\OpenCV-Python-Series-master\\OpenCV-Python-Series-master\\src\\pickles\\face-labelsVGG16.pickle'
# test_path = 'D:\\TCS_YF\\Research\\OpenCV-Python-Series-master\\OpenCV-Python-Series-master\\src\\imagesTest'
# IMAGE_SIZE = (224, 224)

# # Image data generator for training, validation, and testing
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2  # 20% of data will be used for validation
# )

# # Generate training data from the directory
# train_generator = datagen.flow_from_directory(
#     train_path,
#     target_size=IMAGE_SIZE,
#     batch_size=32,
#     class_mode='categorical',
#     subset='training'  # Use subset 'training' for training data
# )

# # Generate validation data from the same directory
# validation_generator = datagen.flow_from_directory(
#     train_path,
#     target_size=IMAGE_SIZE,
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'  # Use subset 'validation' for validation data
# )

# # Create a separate ImageDataGenerator for the test set
# test_datagen = ImageDataGenerator(rescale=1./255)

# # Load test data from the directory
# test_generator = test_datagen.flow_from_directory(
#     test_path,  # Path to your test data directory
#     target_size=IMAGE_SIZE,
#     batch_size=32,
#     class_mode='categorical'
# )

# # Extract class labels from the data generator
# class_labels = list(train_generator.class_indices.keys())

# # Build a dictionary to map class labels to IDs
# label_to_id = {label: idx for idx, label in enumerate(class_labels)}

# # Load the pre-trained VGG16 model (excluding top layers)
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# # Freeze the base model layers
# for layer in base_model.layers:
#     layer.trainable = False

# # Add custom classification layers on top of VGG16
# x = Flatten()(base_model.output)
# predictions = Dense(256, activation='relu')(x)  # Add a dense layer with 256 neurons
# predictions = Dropout(0.01)(predictions)  # Apply dropout with 1% probability
# predictions = Dense(18, activation='softmax')(predictions)  # Final softmax layer with 56 classes

# # Create the model
# model = Model(inputs=base_model.input, outputs=predictions)

# # Compile the model
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=Adam(learning_rate=0.0001),
#     metrics=['accuracy']
# )

# print("len(train_generator): ", len(train_generator))
# print("len(validation_steps): ", len(validation_generator))

# # Train the model using both training and validation data
# history = model.fit(
#     train_generator,
#     epochs=7,  # Set the number of epochs
#     steps_per_epoch=len(train_generator),
#     validation_data=validation_generator,
#     validation_steps=len(validation_generator)
# )

# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")

# # Save the trained model
# model.save(saved_model_path)

# # Save the label-to-id mapping dictionary to a pickle file
# with open(label_to_id_path, 'wb') as f:
#     pickle.dump(label_to_id, f)

# # Plot training and validation loss and accuracy
# plt.plot(history.history['loss'], label='train loss')
# plt.plot(history.history['accuracy'], label='train accuracy')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.plot(history.history['val_accuracy'], label='val accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Loss/Accuracy')
# plt.legend()
# plt.show()

#####################################################
# import keras
# from keras.src.legacy.preprocessing.image import ImageDataGenerator
# from keras.applications.vgg16 import VGG16
# from keras.layers import Flatten, Dense, Dropout
# from keras.models import Model
# import matplotlib.pyplot as plt
# import cv2
# import os
# import numpy as np
# from PIL import Image
# import pickle
# import time
# from timeit import default_timer as timer
# import keras 
# from keras.layers import Input, Lambda, Dense, Flatten
# from keras.models import Model
# from keras.applications.vgg16 import VGG16
# from keras.preprocessing import image
# from tensorflow.python.keras.models import Sequential
# import numpy as np 
# import tensorflow as tf
# from glob import glob
# import matplotlib.pyplot as plt
# from keras.optimizers import Adam
# import keras.optimizers
# from keras import regularizers

# # Define paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# train_path = os.path.join(BASE_DIR, "images")
# saved_model_path = os.path.join(BASE_DIR, "VGGtrainer.h5")
# label_to_id_path = os.path.join(BASE_DIR, "face-labelsVGG16.pickle")
# IMAGE_SIZE = (224, 224)

# # Image data generator for training and validation
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2  # Splitting 20% of data for validation
# )

# # Generate training data from the directory
# train_generator = datagen.flow_from_directory(
#     train_path,
#     target_size=IMAGE_SIZE,
#     batch_size=32,
#     class_mode='categorical',
#     subset='training'  # Use subset 'training' for training data
# )

# # Generate validation data from the same directory
# validation_generator = datagen.flow_from_directory(
#     train_path,
#     target_size=IMAGE_SIZE,
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'  # Use subset 'validation' for validation data
# )

# # Extract class labels from the data generator
# class_labels = list(train_generator.class_indices.keys())

# # Build a dictionary to map class labels to IDs
# label_to_id = {label: idx for idx, label in enumerate(class_labels)}

# # Load the pre-trained VGG16 model (excluding top layers)
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# # Freeze the base model layers
# for layer in base_model.layers:
#     layer.trainable = False

# # Add custom classification layers on top of VGG16
# x = Flatten()(base_model.output)
# predictions = Dense(256, activation='relu')(x)
# predictions = Dropout(0.5)(predictions)
# predictions = Dense(len(class_labels), activation='softmax')(predictions)

# # Create the model
# model = Model(inputs=base_model.input, outputs=predictions)

# # Compile the model
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=Adam(learning_rate=0.0001),
#     metrics=['accuracy']
# )

# # Train the model using both training and validation data
# history = model.fit(
#     train_generator,
#     epochs=10,  # Increase the number of epochs
#     steps_per_epoch=len(train_generator),
#     validation_data=validation_generator,
#     validation_steps=len(validation_generator)
# )

# # Save the trained model
# model.save(saved_model_path)

# # Save the label-to-id mapping dictionary to a pickle file
# with open(label_to_id_path, 'wb') as f:
#     pickle.dump(label_to_id, f)

# # Plot training and validation loss and accuracy
# plt.plot(history.history['loss'], label='train loss')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.plot(history.history['accuracy'], label='train accuracy')
# plt.plot(history.history['val_accuracy'], label='val accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Loss/Accuracy')
# plt.legend()
# plt.show()

