
import os
import csv
import numpy as np
import cv2

# Project configuration
IMAGE_WIDTH = 180
IMAGE_HEIGHT = 60
CAPTCHA_LENGTH = 5

CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Character-to-number mapping
char_to_num = {
    char: index
    for index, char in enumerate(CHARACTERS)
}

num_to_char = {
    index: char
    for char, index in char_to_num.items()
}

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def load_dataset(split):

    dataset_path = os.path.join(
        BASE_DIR, "dataset", split
    )

    images = []
    labels = []

    csv_path = os.path.join(
        dataset_path, "labels.csv"
    )

    with open(csv_path, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            filename = row["filename"]
            text = row["label"]

            image_path = os.path.join(
                dataset_path, filename
            )

            # Step 1: Load image
            image = cv2.imread(image_path)

            if image is None:
                raise ValueError(
                    f"Cannot load image: {image_path}"
                )

            # Step 2: Convert RGB/BGR to grayscale
            image = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

            # Step 3: Resize image
            image = cv2.resize(
                image,
                (IMAGE_WIDTH, IMAGE_HEIGHT)
            )

            # Step 4: Normalize pixel values
            image = image.astype(
                np.float32
            ) / 255.0

            # Step 5: Encode text label
            encoded_label = [
                char_to_num[char]
                for char in text
            ]

            images.append(image)
            labels.append(encoded_label)

    # Convert lists into NumPy arrays
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Add grayscale channel dimension
    X = np.expand_dims(X, axis=-1)

    return X, y


if __name__ == "__main__":

    print("Loading training dataset...")

    X_train, y_train = load_dataset("train")

    print("Loading testing dataset...")

    X_test, y_test = load_dataset("test")

    print("\nPreprocessing completed!")

    print("\nDataset Information:")
    print("--------------------------")

    print("Training images:", X_train.shape)
    print("Training labels:", y_train.shape)

    print("Testing images:", X_test.shape)
    print("Testing labels:", y_test.shape)

    print("\nPixel range:")
    print("Minimum:", X_train.min())
    print("Maximum:", X_train.max())

    # Save processed datasets
    output_dir = os.path.join(
        BASE_DIR, "outputs"
    )

    os.makedirs(output_dir, exist_ok=True)

    np.save(
        os.path.join(output_dir, "X_train.npy"),
        X_train
    )

    np.save(
        os.path.join(output_dir, "y_train.npy"),
        y_train
    )

    np.save(
        os.path.join(output_dir, "X_test.npy"),
        X_test
    )

    np.save(
        os.path.join(output_dir, "y_test.npy"),
        y_test
    )

    print("\nProcessed datasets saved in outputs/")
