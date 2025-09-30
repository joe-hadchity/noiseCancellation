import numpy as np
from numpy import genfromtxt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# ==========================
# Load CSV data
# ==========================
x_train = genfromtxt('compressed_dataset/train_data.csv', delimiter=',')
y_train = genfromtxt('compressed_dataset/train_labels.csv', delimiter=',')
x_test = genfromtxt('compressed_dataset/test_data.csv', delimiter=',')
y_test = genfromtxt('compressed_dataset/test_labels.csv', delimiter=',')

print('\nShape train CSV:', x_train.shape)

# ==========================
# Convert labels to one-hot
# ==========================
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# ==========================
# Reshape for CNN input
# ==========================
x_train = x_train.reshape(-1, 40, 5, 1)
x_test = x_test.reshape(-1, 40, 5, 1)
print('Shape train for CNN:', x_train.shape)

# ==========================
# Build CNN model
# ==========================
model = Sequential()

model.add(Conv2D(64, kernel_size=(5,5), strides=1, padding="same", activation="relu", input_shape=(40,5,1)))
model.add(MaxPooling2D(pool_size=(2,1), padding="same"))
model.add(Conv2D(128, kernel_size=(5,5), strides=1, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,1), padding="same"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation="softmax"))

# ==========================
# Compile the model
# ==========================
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ==========================
# Train the model
# ==========================
model.fit(x_train, y_train, batch_size=50, epochs=40, validation_data=(x_test, y_test))

# ==========================
# Save the trained model
# ==========================
model.save('model.h5')
print('\nModel saved as model.h5\n')

# ==========================
# Evaluate
# ==========================
train_score = model.evaluate(x_train, y_train)
test_score = model.evaluate(x_test, y_test)
print('Train Loss & Accuracy:', train_score)
print('Test Loss & Accuracy:', test_score)
