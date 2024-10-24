
import keras 
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.preprocessing import image
from tensorflow.python.keras.models import Sequential
import numpy as np 
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
import keras.optimizers
from keras.layers import Dense,Flatten,Conv2D,Activation,Dropout
from keras import backend as K
import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import MaxPool2D
from pathlib import Path
import os
import keras
import pickle
from keras.models import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import os

current_directory = os.getcwd()

np.random.seed(42)
tf.random.set_seed(42)

train_path = os.path.join(current_directory, 'images','*')
test_path=os.path.join(current_directory, 'imagesTest','*')
IMAGE_SIZE = [224,224]

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet',include_top=False)

# %%
for layer in vgg.layers:
    layer.trainable = False

for layer in vgg.layers[-4:]:
    layer.trainable = True

train_path = current_directory / 'images' / '*'
test_path = current_directory / 'imagesTest' / '*'

folders = glob(train_path)


# %%
x = Flatten()(vgg.output)
#print(vgg.output.shape)
s = Dense(4096, activation='relu')(x)
#x = Dropout(0.2)(x)
# prediction = Dense(4096, activation='relu')
prediction = Dense(55,activation='softmax')(x)

print("Number of folders in images",len(folders))
model = Model(inputs=vgg.input, outputs = prediction)
root_path = train_path


###############################################################
def count_total_images_in_folders(root_path):
    total_image_count = 0
    
    # Iterate over each directory in the root_path
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Count the number of image files in the current directory
        image_count = len([filename for filename in filenames if filename.endswith(('.jpg', '.jpeg', '.png'))])
        
        # Add the count to the total_image_count
        total_image_count += image_count
    
    return total_image_count

# Call the function to count total images in all directories under train_path
total_images_count = count_total_images_in_folders(train_path)

print(f"Total number of images in all directories: {total_images_count}")   


# %%
model.summary()

learning_rate = 0.00025

# Initialize the Adam optimizer with the specified learning rate
optimizer = Adam(learning_rate=learning_rate)

# %%
from keras.optimizers import Adam, SGD, RMSprop
model.compile(
    loss='categorical_crossentropy',
    #optimizer=Adam(),
    optimizer = optimizer,
    metrics=['accuracy']
)


# %%
#data aug: helps rotate image if face is sideway

train_datagen = ImageDataGenerator(rescale = 1/255,
                                   shear_range = 0.2,
                                   zoom_range=0.2,
                                   horizontal_flip = True,
                                   rotation_range=30,  # Randomly rotate images in the range (degrees, 0 to 180)
                                   width_shift_range=0.1,  # Randomly shift images horizontally (fraction of total width)
                                   height_shift_range=0.1,  # Randomly shift images vertically (fraction of total height)
                                   #brightness_range=[0.8, 1.2],  # Adjust brightness range
                                   channel_shift_range=20.0,  # Randomly shift channels
                                   #fill_mode='nearest',  # Strategy used for filling in newly created pixels
                                   #preprocessing_function = lambda x: keras.applications.vgg16.preprocess_input(tf.image.rgb_to_grayscale(x))
                                   preprocessing_function=keras.applications.vgg16.preprocess_input
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(224,224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# %%
print("LENGTH OF TRAINING SET IS:",len(training_set))
print("LENGTH OF TRAINING SET IS:",len(test_set))

print(training_set.classes.shape)
print(test_set.classes.shape)


# %%
import time 
start_time = time.time()
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=8, 
    steps_per_epoch=len(training_set)//32,
    validation_steps=len(test_set)//32
)

finish = time.time()
print("total time: ")
print(finish-start_time)

model.save(os.path.join(current_directory, 'VGGFaceRecoModel.keras'))

# %%
plt.plot(r.history['loss'],label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))

# Plotting training loss
plt.subplot(1, 2, 1)  # Create subplot for training loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plotting training and validation accuracy
plt.figure(figsize=(12, 6))

# Plotting training accuracy
plt.subplot(1, 2, 1)  # Create subplot for training accuracy
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# %%
model.save(os.path.join(current_directory, 'VGGFaceRecoModel.keras'))

# Extract class labels from the data generator
class_labels = list(training_set.class_indices.keys())

# Build a dictionary to map class labels to IDs
label_to_id = {label: idx for idx, label in enumerate(class_labels)}


label_to_id_path = os.path.join(current_directory, 'pickles','face-labelsVGG16.pickle')
with open(label_to_id_path, 'wb') as f:
    pickle.dump(label_to_id, f)

