
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 180
CAPTCHA_LENGTH = 5
NUM_CLASSES = 36

BATCH_SIZE = 64
EPOCHS = 30

# =====================================================
# 2. LOAD PREPROCESSED DATASET
# =====================================================

print("\nLoading processed dataset...")

X_train = np.load(
    os.path.join(OUTPUT_DIR, "X_train.npy")
)

y_train = np.load(
    os.path.join(OUTPUT_DIR, "y_train.npy")
)

X_test = np.load(
    os.path.join(OUTPUT_DIR, "X_test.npy")
)

y_test = np.load(
    os.path.join(OUTPUT_DIR, "y_test.npy")
)

print("Dataset loaded successfully!")

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)

# =====================================================
# 3. PREPARE LABELS
# =====================================================

y_train_outputs = [
    y_train[:, i] for i in range(CAPTCHA_LENGTH)
]

y_test_outputs = [
    y_test[:, i] for i in range(CAPTCHA_LENGTH)
]

# =====================================================
# 4. BUILD CNN ARCHITECTURE
# =====================================================

def build_model():

    inputs = layers.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    )

    # Convolution block 1
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(inputs)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    # Convolution block 2
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    # Convolution block 3
    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    # Feature extraction
    x = layers.Flatten()(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.3)(x)

    # Five character prediction heads
    outputs = []

    for i in range(CAPTCHA_LENGTH):

        output = layers.Dense(
            NUM_CLASSES,
            activation="softmax",
            name=f"character_{i+1}"
        )(x)

        outputs.append(output)

    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )

    return model


# =====================================================
# 5. CREATE AND COMPILE MODEL
# =====================================================

print("\nBuilding CNN model...")

model = build_model()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics={
        f"character_{i+1}": ["accuracy"]
        for i in range(CAPTCHA_LENGTH)
    }
)

model.summary()

print("\nCNN architecture created successfully!")

# Save initial architecture
model.save(
    os.path.join(
        MODEL_DIR,
        "captcha_cnn_untrained.keras"
    )
)

# =====================================================
# 6. AUTOMATIC CHECKPOINTING
# =====================================================

checkpoint = ModelCheckpoint(
    filepath=os.path.join(
        MODEL_DIR,
        "captcha_best.weights.h5"
    ),
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Save training history
csv_logger = CSVLogger(
    os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    ),
    append=False
)

callbacks = [
    checkpoint,
    csv_logger
]

# =====================================================
# 7. TRAIN CNN
# =====================================================

print("\nStarting CNN training...")

history = model.fit(
    X_train,
    y_train_outputs,
    validation_data=(
        X_test,
        y_test_outputs
    ),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\nCNN training completed!")

# =====================================================
# 8. SAVE FINAL WEIGHTS
# =====================================================

print("\nSaving final model weights...")

model.save_weights(
    os.path.join(
        MODEL_DIR,
        "captcha_final.weights.h5"
    )
)

print("Final weights saved successfully!")

# =====================================================
# 9. SAVE COMPLETE MODEL
# =====================================================

print("\nSaving complete CNN model...")

model.save(
    os.path.join(
        MODEL_DIR,
        "captcha_cnn.keras"
    )
)

print("Complete model saved successfully!")

# =====================================================
# 10. TRAINING SUMMARY
# =====================================================

print("\n========== TRAINING SUMMARY ==========")

print("Total epochs:", len(history.epoch))

print(
    "Final training loss:",
    history.history["loss"][-1]
)

print(
    "Final validation loss:",
    history.history["val_loss"][-1]
)

for i in range(CAPTCHA_LENGTH):

    key = f"character_{i+1}_accuracy"

    print(
        f"Character {i+1} training accuracy:",
        history.history[key][-1]
    )

print("\nAll training files saved successfully!")
print("======================================")
