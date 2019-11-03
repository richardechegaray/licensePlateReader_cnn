from keras import layers
from keras import models
from keras import optimizers
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from keras.utils import plot_model
from keras import backend
import sklearn
from sklearn import metrics


VALIDATION_SPLIT = 0.2

def reset_weights(model):
    session = backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def to_one_hot(char):
    ind = ord(char)
    ind = ind - 48 if ind < 58 else ind - 55
    row = [0] * 36
    row[ind] = 1
    return row

paths = np.array(glob.glob('/home/fizzer/enph353_cnn_lab/data/*.png'))
np.random.shuffle(paths)

x_data = np.array([cv2.imread(path)[:,:,2][:,:,np.newaxis]/255.0 for path in paths])
y_data = np.array([to_one_hot(path[-8]) for path in paths])

print(y_data.shape)
print(x_data.shape)


# actual code xd
conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(85, 53, 1)))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Flatten())
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(512, activation='relu'))
conv_model.add(layers.Dense(36, activation='softmax'))

conv_model.summary()

LEARNING_RATE = 1e-4
conv_model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                metrics=['acc'])

reset_weights(conv_model)

history_conv = conv_model.fit(x_data, y_data, 
                              validation_split=VALIDATION_SPLIT, 
                              epochs=5, 
                              batch_size=16)

conv_model.save('showMiti.h5')


y_pred = conv_model.predict(x_data)
y_pred = [np.argmax(i) for i in y_pred]
y_data = [np.argmax(i) for i in y_data]

confusion_matrix = sklearn.metrics.confusion_matrix(y_data, y_pred)
confusion_matrix = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
print(confusion_matrix)


plt.plot(history_conv.history['loss'])
plt.plot(history_conv.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(history_conv.history['acc'])
plt.plot(history_conv.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()

