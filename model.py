import tensorflow as tf
from keras import layers, models
import os
from keras.optimizers import Adam


LABELS = {
    "apple_fruit": 0,
    "banana_fruit": 1,
    "cherry_fruit": 2,
    "chickoo_fruit": 3,
    "grapes_fruit": 4,
    "kiwi_fruit": 5,
    "mango_fruit": 6,
    "orange_fruit": 7,
    "strawberry_fruit": 8,
}
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


def create_model(input_shape=(128, 128, 3)):
    model = models.Sequential()

    # First convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(2, 2))

    # Second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))

    # Third convolutional layer
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(9, activation="softmax"))

    return model


def localization_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true[:, :4] - y_pred[:, :4]))


def classification_loss(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true[:, 4], y_pred[:, 4])


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image


def get_label_from_path(path):
    parts = tf.strings.split(path, os.sep)
    label_name = parts[-2]  # Get the folder name
    return LABELS[label_name.numpy().decode("utf-8")]


def load_dataset(data_dir):
    # Get all image paths
    all_image_paths = []
    all_labels = []

    for fruit_name in LABELS.keys():
        fruit_dir = os.path.join(data_dir, fruit_name)
        image_paths = [
            os.path.join(fruit_dir, fname)
            for fname in os.listdir(fruit_dir)
            if fname.endswith(".jpg")
        ]
        all_image_paths.extend(image_paths)
        all_labels.extend([LABELS[fruit_name]] * len(image_paths))

    # Convert to tf.data.Dataset
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    label_ds = tf.data.Dataset.from_tensor_slices(all_labels)

    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    dataset = dataset.shuffle(buffer_size=len(all_image_paths))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    model = create_model()
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    dataset = load_dataset("images")
    model.fit(dataset, epochs=10, batch_size=32)

    model.save("fruit_model.h5")
