import os
import cv2
import numpy as np
import pandas as pd
import pickle
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import *
from keras.regularizers import *
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt

current_directory = os.getcwd()

# Define the path to your training dataset directory
dataset_path = os.path.join(current_directory, 'images')

# Initialize an empty dictionary to store the mapping of folder names to numerical labels
label_ids = {}

# function for creating a block
bnEps = 2e-5
bnMom = 0.9
chanDim = 1

# Loop through each directory (celebrity folder) in the training dataset path
current_id = 0
for root, dirs, files in os.walk(dataset_path):
    for folder_name in dirs:
        label = folder_name.lower()
        if label not in label_ids:
            label_ids[label] = current_id
            current_id += 1

# Save the label mapping dictionary to a pickle file
with open('label_mapping.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

print("Label mapping saved to label_mapping.pickle")

# Initialize lists to store images and labels
images = []
label = []

# Function to crop faces (assuming you have this function implemented)
def cropFaces(image):
    #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(os.path.join(current_directory, 'cascades', 'data', 'haarcascade_frontalface_alt2.xml'))
    # Convert the image to grayscale (required for face detection)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    
    # If no faces are detected, return None
    if len(faces) == 0:
        return None
    
    # Assuming there's only one face in the image, extract the coordinates
    x, y, w, h = faces[0]
    
    # Crop the face region from the original image
    cropped_image = image[y:y+h, x:x+w]
    
    return cropped_image

# Loop through each directory (celebrity folder) in the dataset path
# Loop through each directory (celebrity folder) in the dataset path
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        if filename.endswith(("png", "jpg")):  # Check if the file is an image
            img_path = os.path.join(dirname, filename)
            #print("Processing image:", img_path)  # Print the image path for debugging
            img = cv2.imread(img_path)
            if img is None:
                print("Error: Failed to read image:", img_path)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] == 0 or img.shape[1] == 0 or len(img) == 0:
                #print("Error: Invalid image shape for image:", img_path)
                continue
            img = cv2.resize(img, (224, 224)) / 255
            images.append(img)
            lbl = os.path.basename(dirname.lower())  # Extract the label from the directory name
            #print("Label:", lbl)  # Print the label for debugging
            label.append(label_ids[lbl.lower()])  # Use the label mapping dictionary

# Create a DataFrame and shuffle the dataset
df = pd.DataFrame({'image': images, 'label': label})
df = df.sample(frac=1)

# Print unique labels
print("Unique Labels:", df['label'].unique())

# # Convert DataFrame to numpy arrays
# X_train = np.array(df['image'].tolist())
# y_train = to_categorical(np.array(df['label'].tolist()))

# # Create a DataFrame and shuffle the dataset
# df = pd.DataFrame({'image': images, 'label': label})
# df = df.sample(frac=1)

# Convert DataFrame to numpy arrays
X_train = np.array(df['image'].tolist())
y_train = to_categorical(np.array(df['label'].tolist()))

def resnet(layer_in, n_filters, s):
    data = layer_in
    stride = (s, s)
    
    # 1st Convolutional Layer
    merge_input = Conv2D(n_filters, (1, 1), strides=stride)(layer_in)        
    bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(merge_input)
    act2 = Activation('relu')(bn2)
    
    # 2nd Convolutional Layer
    conv2 = Conv2D(n_filters, (3, 3), strides=(1, 1), use_bias=False, padding='same', kernel_initializer='he_normal')(act2)  
    bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
    act3 = Activation('relu')(bn3)
    
    # 3rd Convolutional layer
    conv3 = Conv2D(n_filters, (1, 1), use_bias=False, kernel_initializer='he_normal')(act3)  
    
    # Adjusting the input size according to 3rd convolutional layer
    if data.shape[-1] != n_filters:
        data = Conv2D(n_filters, (1, 1), padding='same', strides=stride)(data)
    
    # Adjusting spatial dimensions of data if needed
    if data.shape[1:] != conv3.shape[1:]:
        data = Conv2D(n_filters, (1, 1), padding='same')(data)
    
    # Add filters, assumes filters/channels last
    layer_out = Add()([conv3, data])
    layer_out = Activation('relu')(layer_out)
    
    return layer_out



# Define the model input
visible = Input(shape=(224, 224, 3))

# Define the ResNet layers
layer1 = resnet(visible, 64, 3)
layer2 = resnet(layer1, 128, 1)
layer4 = resnet(layer2, 256, 1)
layer5 = resnet(layer4, 256, 2)
layer6 = resnet(layer5, 512, 2)
layer7 = resnet(layer6, 512, 2)
layer8 = resnet(layer7, 1024, 2)
layert = Dropout(0.5)(layer8)
layer9 = resnet(layert, 2048, 2)
layert2 = Dropout(0.5)(layer9)
layer10 = resnet(layert2, 4096, 2)

# Add global average pooling and dense layers
x = GlobalAveragePooling2D()(layer10)
x = Dropout(0.7)(x)
den = Dense(2048, activation='sigmoid')(x)
final = Dense(len(label_ids), activation='softmax')(den)
print("label length",len(lbel_ids))

# Create the model
model = Model(inputs=visible, outputs=final)

# Compile the model
lr = 1e-5
decay = 1e-7
optimizer = RMSprop(lr=lr, decay=decay)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, steps_per_epoch=20, shuffle=True, epochs=9)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()
plt.show()

# Evaluate the model (replace X_test and y_test with your test data)
# score = model.evaluate(X_test
