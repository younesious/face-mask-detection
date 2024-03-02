import os.path
import numpy as np
from cv2 import cv2
from scipy.spatial import distance
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

face_model = cv2.CascadeClassifier(os.path.abspath('./kaggle/input/haarcascades/haarcascade_frontalface_default.xml'))

img = cv2.imread('./kaggle/input/face-mask-detection/images/maksssksksss244.png')

img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)  # returns a list of (x,y,w,h) tuples

# Load train and test set
train_dir = './kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset/Train'
val_dir = './kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset/Validation'

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)

train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(128, 128),
                                                    class_mode='categorical', batch_size=32)
val_generator = train_datagen.flow_from_directory(directory=val_dir, target_size=(128, 128), class_mode='categorical',
                                                  batch_size=32)

vgg19 = VGG19(weights='./vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False,
              input_shape=(128, 128, 3))

for layer in vgg19.layers:
    layer.trainable = False

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=len(train_generator) // 32,
                              epochs=20, validation_data=val_generator,
                              validation_steps=len(val_generator) // 32)

model.save('masknet.h5')

mask_label = {0: 'MASK', 1: 'NO MASK'}
dist_label = {0: (0, 255, 0), 1: (255, 0, 0)}
MIN_DISTANCE = 130

if len(faces) >= 2:
    label = [0 for i in range(len(faces))]
    for i in range(len(faces) - 1):
        for j in range(i + 1, len(faces)):
            dist = distance.euclidean(faces[i][:2], faces[j][:2])
            if dist < MIN_DISTANCE:
                label[i] = 1
                label[j] = 1
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # colored output image
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        crop = new_img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (128, 128))
        crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
        mask_result = model.predict(crop)
        cv2.putText(new_img, mask_label[mask_result.argmax()], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    dist_label[label[i]], 2)
        cv2.rectangle(new_img, (x, y), (x + w, y + h), dist_label[label[i]], 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(new_img)
    plt.show()

else:
    print("No. of faces detected is less than 2")
