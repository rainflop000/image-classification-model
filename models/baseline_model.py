import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
cifar100 = tf.keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

# Define model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),  # Input layer
    layers.Conv2D(16, (7, 7), activation="relu", padding="valid", strides=1),  # Conv2D layer
    layers.MaxPooling2D((2, 2), strides=2),  # MaxPooling2D layer
    layers.Conv2D(32, (5, 5), activation="relu", padding="valid", strides=1),  # Conv2D layer
    layers.MaxPooling2D((2, 2), strides=2),  # MaxPooling2D layer
    layers.Flatten(),  # Flatten layer
    layers.Dense(128, activation="relu"),  # Dense layer
    layers.Dense(100)  # Output layer
])

model.summary()

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)