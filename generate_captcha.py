
import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# Configuration
CAPTCHA_LENGTH = 5
IMAGE_WIDTH = 180
IMAGE_HEIGHT = 60
TOTAL_IMAGES = 10000

CHARACTERS = string.ascii_uppercase + string.digits

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Font selection
def get_font():
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf"
    ]

    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, 35)

    return ImageFont.load_default()


def generate_text():
    return ''.join(
        random.choices(CHARACTERS, k=CAPTCHA_LENGTH)
    )


def generate_captcha(text):
    image = Image.new(
        "RGB",
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        color=(245, 245, 245)
    )

    draw = ImageDraw.Draw(image)
    font = get_font()

    # Draw random interference lines
    for _ in range(6):
        x1 = random.randint(0, IMAGE_WIDTH)
        y1 = random.randint(0, IMAGE_HEIGHT)
        x2 = random.randint(0, IMAGE_WIDTH)
        y2 = random.randint(0, IMAGE_HEIGHT)

        draw.line(
            (x1, y1, x2, y2),
            fill=(random.randint(100, 180),) * 3,
            width=1
        )

    # Draw CAPTCHA characters
    char_width = IMAGE_WIDTH // CAPTCHA_LENGTH

    for i, char in enumerate(text):
        char_image = Image.new("RGBA", (45, 50), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_image)

        char_draw.text(
            (5, 3),
            char,
            font=font,
            fill=(random.randint(0, 100),) * 3
        )

        angle = random.randint(-20, 20)
        char_image = char_image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False
        )

        x = i * char_width + random.randint(2, 8)
        y = random.randint(3, 10)

        image.paste(char_image, (x, y), char_image)

    # Add noise
    pixels = np.array(image)

    noise = np.random.randint(
        -15, 16, pixels.shape, dtype=np.int16
    )

    pixels = np.clip(
        pixels.astype(np.int16) + noise,
        0, 255
    ).astype(np.uint8)

    image = Image.fromarray(pixels)

    image = image.filter(
        ImageFilter.SMOOTH
    )

    return image


def create_dataset(split, count):
    output_dir = os.path.join(BASE_DIR, "dataset", split)
    os.makedirs(output_dir, exist_ok=True)

    labels_path = os.path.join(
        output_dir, "labels.csv"
    )

    with open(labels_path, "w") as file:
        file.write("filename,label\n")

        for i in range(count):
            text = generate_text()
            image = generate_captcha(text)

            filename = f"captcha_{i:05d}.png"

            image.save(
                os.path.join(output_dir, filename)
            )

            file.write(f"{filename},{text}\n")

            if (i + 1) % 1000 == 0:
                print(f"{split}: {i + 1}/{count} generated")

    print(f"{split} dataset completed!")


if __name__ == "__main__":

    random.seed(42)
    np.random.seed(42)

    create_dataset("train", 8000)
    create_dataset("test", 2000)

    print("Dataset generation completed!")
