import os
import numpy as np
from os import listdir
from scipy.misc import imread, imresize
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import tensorflow as tf
np.set_printoptions(threshold=np.inf)


# Settings:
img_size = 64
channel_size = 1 # 1: Grayscale, 3: RGB
num_class = 10
test_size = 0.2


def get_img(data_path):
    # Getting image array from path:
    img = imread(data_path, flatten= True if channel_size == 1 else False)
    img = imresize(img, (img_size, img_size, channel_size))
    return img

def get_dataset(dataset_path='/path/to/data'):
    # Getting all data from data path:
    try:
        X = np.load('Data/npy_dataset/X.npy')
        Y = np.load('Data/npy_dataset/Y.npy')
    except:
        labels = listdir(dataset_path) # Geting labels
        X, Y = [], []
        for label in labels:
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = get_img(datas_path+'/'+data)
                X.append(img)
                Y.append(label)
        # Create dateset:
        X = 1-np.array(X).astype('float32')/255.
        X = X.reshape(X.shape[0], img_size, img_size, channel_size)
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_class)
        if not os.path.exists('Data/npy_dataset/'):
            os.makedirs('Data/npy_dataset/')
        np.save('Data/npy_dataset/X.npy', X)
        np.save('Data/npy_dataset/Y.npy', Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X, X_test, Y, Y_test



if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = get_dataset()


tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
#model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(256, (3,3), activation='relu'))
#model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(512, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['acc'])

history = model.fit(X_train, Y_train, epochs=32, validation_split=0.25, callbacks=[tbCallBack])

print(model.summary())

# pred = model.predict(X_test)
# print(pred)


score = model.evaluate(X_test, Y_test)
print("Test loss: ", score[0])
print("Test Accuracy: ", score[1])
