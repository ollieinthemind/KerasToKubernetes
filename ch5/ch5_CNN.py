from keras.utils import to_categorical
from ch5_classifying_FashionImages import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D


# one hot encode the results
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# normalize the data to values between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0
# customize data for CNN - make a 3D array
x_train_cnn = np.expand_dims(x_train, -1)
x_test_cnn = np.expand_dims(x_test, -1)


model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu',input_shape=(28,28,1), padding='same'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

#assign the optimizer for the model and define loss function

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# model.summary()

#run the actual training
history = model.fit(x_train_cnn, y_train, epochs=1)

#evaluate on test data
