import os
from keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error


def prediction(img):

    single_image_batch = np.expand_dims(img, axis=0)

    # Make predictions on the batch (which contains only one image)
    predictions = model.predict(single_image_batch)
    cv2.circle(img, (int(predictions[0][0]), int(predictions[0][1])), 4, (0, 255, 0), -1)
    cv2.circle(img, (int(predictions[0][2]), int(predictions[0][3])), 4, (255, 0, 0), -1)
    cv2.circle(img, (int(predictions[0][4]), int(predictions[0][5])), 4, (255, 255, 255), -1)
    cv2.circle(img, (int(predictions[0][6]), int(predictions[0][7])), 4, (0, 0, 255), -1)

    cv2.imshow("prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return predictions


# data preparation
os.chdir('/Users/sara/Desktop/part/computervision_task3/merged_data')
list_images = sorted(os.listdir('images'))
list_coordinates = sorted(os.listdir('txtFiles'))

images = []
coordinates = []
for image_file, txt_file in zip(list_images, list_coordinates):

    image = cv2.imread("images/" + image_file)
    images.append(image)
    # Parse coordinates from the text file
    with open("txtFiles/" + txt_file, "r") as file:
        for line in file:
            x, y = map(float, line.strip().split(", "))
            coordinates.append((x, y))

images = np.array(images)
coordinates = np.array(coordinates).reshape(-1, 8)

test_size = 0.2
valid_size = 0.5
random_state = 42
X_train, X_valid_test, y_train, y_valid_test = train_test_split(images, coordinates, test_size=test_size, random_state=random_state)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=valid_size, random_state=random_state)

# model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False
regression_head = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(7, 7, 1280)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(8)
])
# Combine the base model and regression head
model = tf.keras.Sequential([
    base_model,
    regression_head
])

# Compile the model
# Define a custom learning rate
custom_learning_rate = 0.001
epochs = 100
batch_size = 16
lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=4,
    min_lr=1e-7
)
# Create a custom optimizer with the specified learning rate
custom_optimizer = Adam(learning_rate=custom_learning_rate)
model.compile(optimizer=custom_optimizer, loss="mean_squared_error")
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), batch_size=batch_size, callbacks=[lr_callback])


predicted_coordinates = model.predict(X_test)


mse = mean_squared_error(y_test, predicted_coordinates)
print(f"The final MSE is {mse}")

coordinates = prediction(X_test[1])

print(f"The coordinates are: {coordinates}")
