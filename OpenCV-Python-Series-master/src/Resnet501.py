import os
import keras
import pickle
import time
import numpy as np
import tensorflow as tf
from glob import glob
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.layers import MaxPooling2D, BatchNormalization
# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the paths to the training and test datasets
train_path = 'D:/TCS_YF/Research/OpenCV-Python-Series-master/OpenCV-Python-Series-master/src/images'
test_path = 'D:/TCS_YF/Research/OpenCV-Python-Series-master/OpenCV-Python-Series-master/src/imagesTest'

# Define the image size
IMAGE_SIZE = [224, 224]

# Load the ResNet-50 model with pre-trained weights
resnet = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# Freeze the pre-trained layers
for layer in resnet.layers:
    layer.trainable = False

for layer in resnet.layers[-4:]:
    layer.trainable = True

# Define additional layers on top of ResNet-50
# x = Flatten()(resnet.output)
# x = Dense(4096, activation='relu')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)  # Add max pooling
# prediction = Dense(56, activation='softmax')(x)
x = resnet.output
x = MaxPooling2D(pool_size=(2, 2))(x)  # Add max pooling
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
prediction = Dense(56, activation='softmax')(x)

# Create the model
model = Model(inputs=resnet.input, outputs=prediction)

# Display the model summary
model.summary()

# Compile the model
learning_rate = 0.00025
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=20.0,
    preprocessing_function=keras.applications.resnet50.preprocess_input
)

# Normalization for test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of augmented data for training and normalized data for validation
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("length of training_set: ", len(training_set))

# Train the model
start_time = time.time()
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set)//32,
    validation_steps=len(test_set)//32
)
finish = time.time()
print("Total time:", finish - start_time)

# Save the trained model
model.save('D:\\TCS_YF\\Research\\OpenCV-Python-Series-master\\OpenCV-Python-Series-master\\src\\recognizers\\ResNet50.keras')

# Plot the training history
import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))

# Plotting training loss
plt.subplot(1, 2, 1)
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plotting training and validation accuracy
plt.figure(figsize=(12, 6))

# Plotting training accuracy
plt.subplot(1, 2, 1)
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save class labels and their corresponding IDs to a pickle file
class_labels = list(training_set.class_indices.keys())
label_to_id = {label: idx for idx, label in enumerate(class_labels)}
label_to_id_path = 'D:\\TCS_YF\\Research\\OpenCV-Python-Series-master\\OpenCV-Python-Series-master\\src\\pickles\\face-labelsResNet50.pickle'
with open(label_to_id_path, 'wb') as f:
    pickle.dump(label_to_id, f)
