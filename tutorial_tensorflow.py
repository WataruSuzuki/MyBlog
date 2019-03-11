from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense,Dropout
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import os


folder = os.listdir("datasets")
folder.pop(-1)
image_size = 50
dense_size  = len(folder)

X = []
Y = []
for index, name in enumerate(folder):
    dir = "./datasets/" + name
    files = glob.glob(dir + "/*.png")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)
X = X.astype('float32')
X = X / 255.0

Y = np_utils.to_categorical(Y, dense_size)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(dense_size))
model.add(Activation('softmax'))

model.summary()
optimizers ="Adadelta"
results = {}

epochs = 200
model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy'])
results[0]= model.fit(X_train, y_train, validation_split=0.2, epochs=epochs)

model_json_str = model.to_json()
open('mnist_mlp_model.json', 'w').write(model_json_str)
model.save_weights('mnist_mlp_weights.h5');

x = range(epochs)
for k, result in results.items():
    plt.plot(x, result.history['acc'], label=k)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),borderaxespad=0, ncol=2)

name = 'acc.jpg'
plt.savefig(name, bbox_inches='tight')
plt.close()

for k, result in results.items():
    plt.plot(x, result.history['val_acc'], label=k)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),borderaxespad=0, ncol=2)

name = 'val_acc.jpg'
plt.savefig(name, bbox_inches='tight')
