from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#load MNIST dataset provided by Keras
dataset = keras.datasets.fashion_mnist

# labels for the images
class_names = ['T-shirt/top','Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# load the training and test data
(img_rows, img_cols) = (28,28)
(x_train, y_train), (x_test, y_test) = dataset.load_data()

#plotting data samples
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.gray)
    plt.xlabel(class_names[y_test[i]])

plt.show()

