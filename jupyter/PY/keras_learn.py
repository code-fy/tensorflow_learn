
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

#%%

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1,28*28) / 255.0
test_images = test_images[:1000].reshape(-1,28*28) / 255.0

#%%

# class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#%%

# train_images.shape

#%%

# train_images

#%%

# train_labels

#%%

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#%%
#
# train_images = train_images / 255.0
# test_images = test_images / 255.0

#%%

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_name[train_labels[i]])
# plt.show()

#%%

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(128,activation='relu'),
#     keras.layers.Dense(10)
# ])

#%%

# model.compile(optimizer='adam', loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics = ['accuracy'])

#%%

# model.fit(train_images,train_labels,epochs=10)

#%%

# test_loss, test_acc = model.evaluate(test_images,test_labels, verbose=2)
# print('\nTest accuracy:', test_acc)
#
# probability_model = tf.keras.Sequential([model,
#                                          tf.keras.layers.Softmax()])
#
# predictions = probability_model.predict(test_images)
#
# predictions[0]
#
# np.argmax(predictions[0])


# def plot_image(i, predictions_array, true_label, img):
#   predictions_array, true_label, img = predictions_array, true_label[i], img[i]
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
#
#   plt.imshow(img, cmap=plt.cm.binary)
#
#   predicted_label = np.argmax(predictions_array)
#   if predicted_label == true_label:
#     color = 'blue'
#   else:
#     color = 'red'
#
#   plt.xlabel("{} {:2.0f}% ({})".format(class_name[predicted_label],
#                                 100*np.max(predictions_array),
#                                 class_name[true_label]),
#                                 color=color)

# def plot_value_array(i, predictions_array, true_label):
#   predictions_array, true_label = predictions_array, true_label[i]
#   plt.grid(False)
#   plt.xticks(range(10))
#   plt.yticks([])
#   thisplot = plt.bar(range(10), predictions_array, color="#777777")
#   plt.ylim([0, 1])
#   predicted_label = np.argmax(predictions_array)
#
#   thisplot[predicted_label].set_color('red')
#   thisplot[true_label].set_color('blue')
#
#   i = 0
#   plt.figure(figsize=(6, 3))
#   plt.subplot(1, 2, 1)
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(1, 2, 2)
#   plot_value_array(i, predictions[i], test_labels)
#   plt.show()
#
#   #------------------------------------------------------------------------------------


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu',input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )
    return model

model = create_model()
model.summary()
model.fit(train_images, train_labels, epochs=5)
model.save('saved_model/my_model')