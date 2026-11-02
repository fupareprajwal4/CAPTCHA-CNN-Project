
import os
import sys
import cv2
import numpy as np
import tensorflow as tf

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "captcha_cnn.keras"
)

CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

IMAGE_WIDTH = 180
IMAGE_HEIGHT = 60
CAPTCHA_LENGTH = 5

# =====================================================
# 2. LOAD TRAINED CNN MODEL
# =====================================================

print("\nLoading trained CNN model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# =====================================================
# 3. IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"CAPTCHA image not found: {image_path}"
        )

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "Unable to read CAPTCHA image."
        )

    # Convert to grayscale
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Resize image
    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    # Normalize pixel values
    image = image.astype(
        np.float32
    ) / 255.0

    # Add channel dimension
    image = np.expand_dims(
        image,
        axis=-1
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image

# =====================================================
# 4. CAPTCHA PREDICTION
# =====================================================

def predict_captcha(image_path):

    image = preprocess_image(image_path)

    # CNN prediction
    predictions = model.predict(
        image,
        verbose=0
    )

    predicted_text = ""
    confidences = []

    # Process five output heads
    for i, prediction in enumerate(predictions):

        # Extract probabilities for one image
        probabilities = prediction[0]

        # Find predicted character index
        predicted_index = int(
            np.argmax(probabilities)
        )

        # Convert index into character
        predicted_character = CHARACTERS[
            predicted_index
        ]

        # Get confidence
        confidence = float(
            probabilities[predicted_index]
        )

        predicted_text += predicted_character

        confidences.append(confidence)

    return predicted_text, confidences

# =====================================================
# 5. DISPLAY PREDICTION RESULT
# =====================================================

def display_result(predicted_text, confidences):

    print("\n========== CAPTCHA RESULT ==========")

    print(
        "\nPredicted CAPTCHA:",
        predicted_text
    )

    print("\nCharacter Confidence:")

    for i, confidence in enumerate(confidences):

        print(
            f"Character {i+1}: "
            f"{confidence * 100:.2f}%"
        )

    average_confidence = np.mean(confidences)

    print(
        "\nAverage Confidence:",
        f"{average_confidence * 100:.2f}%"
    )

    print("\n====================================")

# =====================================================
# 6. MAIN PROGRAM
# =====================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")

        print(
            "python src/predict.py "
            "dataset/test/captcha_00000.png"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    # Resolve relative paths
    if not os.path.isabs(image_path):

        image_path = os.path.join(
            BASE_DIR,
            image_path
        )

    print("\nRecognizing CAPTCHA...")

    try:

        predicted_text, confidences = predict_captcha(
            image_path
        )

        display_result(
            predicted_text,
            confidences
        )

    except Exception as error:

        print("\nPrediction failed!")
        print("Error:", error)

        sys.exit(1)
