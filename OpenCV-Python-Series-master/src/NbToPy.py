# %%
import keras 
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from tensorflow.python.keras.models import Sequential
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt
import keras.optimizers

# %%
from keras.layers import Dense,Flatten,Conv2D,Activation,Dropout
from keras import backend as K
import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import MaxPool2D

# %%
train_path='D:\TCS_YF\Research\OpenCV-Python-Series-master\OpenCV-Python-Series-master\src\imageTemp'
#train_path = 'D:\TCS_YF\Research\OpenCV-Python-Series-master\OpenCV-Python-Series-master\src\images'
test_path='D:\TCS_YF\Research\OpenCV-Python-Series-master\OpenCV-Python-Series-master\src\imagesTest'
IMAGE_SIZE = [224,224]

# %%
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet',include_top=False)

# %%
for layer in vgg.layers:
    layer.trainable = False

# %%
#folders = glob(train_path)
folders = glob('D:\TCS_YF\Research\OpenCV-Python-Series-master\OpenCV-Python-Series-master\src\imageTemp\*')

# %%
#folders

# %%
# model = Sequential()

# model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name='VGG16'))

# model.add(Flatten(name='flatten'))

# model.add(Dense(256, activation="relu", name="fc1"))

# model.add(Dense(128, activation="relu", name="fc2"))

# model.add(Dense(196, activation="softmax", name="output"))

# %%
x = Flatten()(vgg.output)
#print(vgg.output.shape)
prediction = Dense(len(folders),activation='softmax')(x)
#prediction = Dense(19,activation='softmax')(x)
print(len(folders))
model = Model(inputs=vgg.input, outputs = prediction)

# %%
model.summary()

# %%
from keras.optimizers import Adam, SGD, RMSprop
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

# %%
#data aug: helps rotate image if face is sideways
import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# %%
train_datagen = ImageDataGenerator(rescale = 1/255,
                                   shear_range = 0.2,
                                   zoom_range=0.2,
                                   horizontal_flip = True)

# %%
test_datagen = ImageDataGenerator(rescale=1./255)

# %%
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(224,224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# %%
print(training_set.classes.shape)
print(test_set.classes.shape)


# %%
import time 
start_time = time.time()
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=9, 
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)
finish = time.time()
print("total time: ")
print(finish-start_time)

# %%
plt.plot(r.history['loss'],label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# %%
model.save('VGGFaceRecoModel.h5')


