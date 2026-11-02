# AI-Based CAPTCHA Generation and Recognition System Using CNN

## 1. Project Overview

This project implements an AI-based CAPTCHA generation and text recognition system using a Convolutional Neural Network (CNN).

The system generates synthetic CAPTCHA images, preprocesses them, trains a CNN model and recognizes text from uploaded CAPTCHA images through a Gradio web interface.

The project is developed as part of the **Introduction to Neural Networks (INN)** academic course.

The long-term objective is to extend the system towards recognizing text across different image formats, fonts, backgrounds and visual distortions using more advanced OCR architectures.

## 2. Objectives

- Generate synthetic CAPTCHA images automatically.
- Create and preprocess a labelled image dataset.
- Develop a CNN-based character recognition model.
- Predict five-character CAPTCHA sequences.
- Evaluate model performance using character-level and complete-sequence accuracy.
- Develop an interactive web interface using Gradio.
- Explore advanced OCR techniques for recognizing text in diverse images.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | CNN development and training |
| OpenCV | Image preprocessing |
| NumPy | Numerical operations |
| Pillow | Image generation and manipulation |
| Matplotlib | Training visualization |
| Scikit-learn | Machine learning evaluation |
| Gradio | Interactive web interface |

## 4. Project Architecture

```text
CAPTCHA_CNN_Project/
│
├── dataset/
│   ├── train/
│   │   ├── CAPTCHA images
│   │   └── labels.csv
│   │
│   └── test/
│       ├── CAPTCHA images
│       └── labels.csv
│
├── models/
│   ├── captcha_cnn.keras
│   ├── captcha_best.weights.h5
│   ├── captcha_final.weights.h5
│   └── captcha_cnn_untrained.keras
│
├── outputs/
│   ├── plots/
│   ├── predictions/
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   └── training_history.csv
│
├── src/
│   ├── __init__.py
│   ├── generate_captcha.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Dataset Description

The project uses a synthetically generated CAPTCHA dataset.

| Parameter | Value |
|---|---|
| Total images | 10,000 |
| Training images | 8,000 |
| Testing images | 2,000 |
| Image dimensions | 180 × 60 pixels |
| CAPTCHA length | 5 characters |
| Character set | A–Z and 0–9 |
| Total character classes | 36 |

### Dataset characteristics

The generated CAPTCHA images include:

- Random uppercase letters and digits.
- Character rotation.
- Random interference lines.
- Pixel noise.
- Image smoothing.

Each image has a corresponding text label stored in `labels.csv`.

## 6. Image Preprocessing

The preprocessing pipeline performs:

1. Image loading.
2. BGR-to-grayscale conversion.
3. Image resizing to 180 × 60 pixels.
4. Pixel normalization to the range 0 to 1.
5. Character-to-numeric label encoding.
6. Conversion into NumPy arrays.

### Processed dataset dimensions

| Dataset | Image shape | Label shape |
|---|---|---|
| Training | (8000, 60, 180, 1) | (8000, 5) |
| Testing | (2000, 60, 180, 1) | (2000, 5) |

## 7. CNN Model Architecture

The current model uses a CNN with five independent output heads.

| Layer | Configuration |
|---|---|
| Input | 60 × 180 × 1 |
| Conv2D | 32 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 128 filters, 3 × 3, ReLU |
| Flatten | Feature extraction |
| Dense | 256 neurons, ReLU |
| Dropout | 0.30 |
| Output | Five independent heads, 36 classes each |
| Activation | Softmax |

### Training configuration

- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Batch size: 64
- Training epochs: 30
- Regularization: Dropout

The model predicts each of the five CAPTCHA character positions independently.

## 8. Training Results

The completed training run produced the following results:

| Metric | Result |
|---|---:|
| Training epochs | 30 |
| Final training loss | 0.8748 |
| Final validation loss | 0.2608 |
| Mean character-level validation accuracy | 98.45% |

### Character-wise validation accuracy

| Character position | Accuracy |
|---|---:|
| Character 1 | 98.65% |
| Character 2 | 98.55% |
| Character 3 | 98.25% |
| Character 4 | 98.65% |
| Character 5 | 98.30% |

**Evaluation note:** The 2,000-image test split was also used as validation data during training. Therefore, these results should be treated as validation results rather than independent final test performance.

Character-level accuracy is different from complete CAPTCHA accuracy. Complete-sequence accuracy requires all five characters to be correct.

## 9. Installation and Setup

### Prerequisites

- Python 3.10 or later, subject to library compatibility.
- pip package manager.
- Windows, Linux or macOS.
- Recommended: virtual environment.

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CAPTCHA_CNN_Project
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Generate the dataset

```bash
python src/generate_captcha.py
```

### Step 6: Preprocess the dataset

```bash
python src/preprocess.py
```

### Step 7: Train the CNN model

```bash
python src/train.py
```

### Step 8: Test CAPTCHA recognition

```bash
python src/predict.py dataset/test/captcha_00000.png
```

### Step 9: Launch the web application

```bash
python app.py
```

Open the following URL in your browser:

```text
http://127.0.0.1:7860
```

## 10. Web Application Features

The Gradio application provides three main sections.

### CAPTCHA Generator

- Generate new synthetic CAPTCHA images.
- Display the corresponding generated text.
- Test generated CAPTCHA samples.

### CAPTCHA Recognition

- Upload a CAPTCHA image.
- Generate character predictions.
- Display character-wise model confidence.
- Display average model confidence.

### About Project

Displays project information, CNN architecture, input dimensions and technologies used.

## 11. Current Limitations

The current CNN model has the following limitations:

- Supports exactly five characters.
- Recognizes only uppercase English letters and digits.
- Expects a fixed image size of 180 × 60 pixels.
- Was trained on a synthetic dataset with limited visual variation.
- May perform poorly on images with unfamiliar fonts, backgrounds, layouts or distortions.
- Does not currently support arbitrary-length text recognition.

The current model is an academic demonstration and is not a universal OCR solution.

## 12. Future Scope

The project can be extended towards more general-purpose OCR using:

1. **CRNN:** Combine convolutional feature extraction with recurrent sequence recognition.
2. **CTC Loss:** Recognize variable-length text without manually segmenting individual characters.
3. **Advanced preprocessing:** Improve handling of different backgrounds, noise, rotation and image quality.
4. **Dataset expansion:** Include diverse fonts, image dimensions, colours and text arrangements.
5. **OCR evaluation:** Measure character error rate (CER), word error rate (WER) and exact-sequence accuracy.
6. **Improved interface:** Support additional image formats and configurable OCR processing.

These improvements would expand the system beyond its current fixed-length synthetic CAPTCHA recognition task.

## 13. Applications

The underlying image recognition techniques have applications in:

- OCR research and education.
- Document text extraction.
- Image-based text recognition.
- Computer vision experiments.
- Neural network learning and evaluation.

## 14. Academic Project Information

**Project Title:** AI-Based CAPTCHA Generation and Recognition System Using CNN

**Course:** Introduction to Neural Networks (INN)

**Model:** Convolutional Neural Network

**Programming Language:** Python

**Deep Learning Framework:** TensorFlow / Keras

**Interface Framework:** Gradio

---

## 15. License

This project is developed for academic and educational purposes. Further usage and distribution should follow the applicable dataset, dependency and repository licenses.
