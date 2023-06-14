from keras.models import Sequential
from keras.layers import Dense, Flatten
from ch4_handwriting  import *

# build a simple  neural network
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(512, activation= 'relu'))
model.add(Dense(10, activation='softmax'))

# show summary
model.summary()

# assign the optimizer for the model and define loss function
model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics=['accuracy'])

# run the actual training
history = model.fit(x_train, y_train, epochs=1, validation_split=0.33)

# evaluate on test data
model.evaluate(x_test, y_test)



