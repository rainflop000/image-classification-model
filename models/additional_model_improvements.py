import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.python.eager.context import PhysicalDevice
from typing import List

def main():

    try:
        devices: List[PhysicalDevice] = tf.config.list_physical_devices('GPU')
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Use devices:", list(map(lambda d: d.name, devices)))
    except IndexError:
        print("Use CPU")

    # Load dataset
    cifar100 = tf.keras.datasets.cifar100
    (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip(mode="horizontal", seed=100),
        layers.RandomContrast(0.1),
    ])

    # Define normalization layer
    normalization_layer = layers.Normalization(axis=-1)
    normalization_layer.adapt(train_images)

    # Define model
    inputs = layers.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    x = normalization_layer(x)

    x = layers.Conv2D(24, (3, 3), activation="relu", padding="valid", strides=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LayerNormalization(axis=-1)(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)

    x = inception_module(x, [32, 32, 64, 16, 32, 32])

    x = layers.Conv2D(64, (4, 4), activation="relu", padding="valid", strides=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LayerNormalization(axis=-1)(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)

    x = inception_module(x, [64, 64, 128, 32, 64, 64])

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(100)(x)

    model = models.Model(inputs, outputs)

    model.summary()

    learning_rate = 1e-3
    lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, decay_steps=100, decay_rate=0.96, staircase=True)

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_decay),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("\nTest accuracy:", test_acc)


# Define Inception module
def inception_module(x, filters):
    # Branch 1
    branch1x1 = layers.Conv2D(filters[0], (1, 1), padding="same", activation="relu")(x)
    # Branch 2
    branch3x3 = layers.Conv2D(filters[1], (1, 1), padding="same", activation="relu")(x)
    branch3x3 = layers.Conv2D(filters[2], (3, 3), padding="same", activation="relu")(branch3x3)
    # Branch 3
    branch5x5 = layers.Conv2D(filters[3], (1, 1), padding="same", activation="relu")(x)
    branch5x5 = layers.Conv2D(filters[4], (5, 5), padding="same", activation="relu")(branch5x5)
    # Branch 4
    branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = layers.Conv2D(filters[5], (1, 1), padding="same", activation="relu")(branch_pool)
    # Concatenate branches
    x = layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1)
    return x

if __name__ == "__main__":
    main()