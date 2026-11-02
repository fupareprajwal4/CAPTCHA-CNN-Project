
from pathlib import Path

PROJECT_NAME = "CAPTCHA_CNN_Project"

folders = [
    "dataset/train",
    "dataset/test",
    "models",
    "outputs",
    "outputs/plots",
    "outputs/predictions",
    "src",
    "src/__pycache__"
]

files = [
    "src/__init__.py",
    "src/generate_captcha.py",
    "src/preprocess.py",
    "src/train.py",
    "src/predict.py",
    "src/evaluate.py",
    "app.py",
    "requirements.txt",
    "README.md",
    ".gitignore"
]

root = Path(PROJECT_NAME)
root.mkdir(parents=True, exist_ok=True)

# Create directories
for folder in folders:
    (root / folder).mkdir(parents=True, exist_ok=True)

# Create files only if they do not already exist
for filename in files:
    file_path = root / filename

    if not file_path.exists():
        file_path.touch()
        print(f"Created: {filename}")
    else:
        print(f"Already exists: {filename}")

# Update requirements
requirements = """tensorflow
numpy
opencv-python
matplotlib
Pillow
scikit-learn
gradio
"""

(root / "requirements.txt").write_text(requirements)

# Create gitignore
gitignore = """venv/
.venv/
__pycache__/
*.pyc
*.pyo
.DS_Store

# Generated datasets and model files
dataset/train/*.png
dataset/test/*.png
models/*.keras
models/*.h5

# Generated outputs
outputs/*.npy
outputs/*.csv
"""

(root / ".gitignore").write_text(gitignore)

# Display structure
print("\nPROJECT STRUCTURE")
print("=" * 45)

def display_tree(path, prefix=""):
    items = sorted(
        path.iterdir(),
        key=lambda p: (p.is_file(), p.name.lower())
    )

    for index, item in enumerate(items):
        is_last = index == len(items) - 1
        connector = "└── " if is_last else "├── "

        print(prefix + connector + item.name)

        if item.is_dir():
            extension = "    " if is_last else "│   "
            display_tree(item, prefix + extension)

print(root.name + "/")
display_tree(root)

print("\nUpdated project structure created successfully!")
