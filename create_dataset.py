import os
import random

from PIL import Image, ImageDraw

# Set image dimensions and square size
IMG_WIDTH, IMG_HEIGHT = 512, 512
SQUARE_SIZE = 32

TOP_Y_MIN = 0
TOP_Y_MAX = IMG_HEIGHT // 2 - SQUARE_SIZE
BOTTOM_Y_MIN = IMG_HEIGHT // 2
BOTTOM_Y_MAX = IMG_HEIGHT - SQUARE_SIZE 

base_dir = "dataset"
splits = {
    "train": {
        "class1_region": "top",
        "class1_count": 1000,
        "class2_region": "bottom",
        "class2_count": 1000,
    },
    "validation": {
        "class1_region": "top",
        "class1_count": 500,
        "class2_region": "bottom",
        "class2_count": 500,
    },
    "test_similar": {
        "class1_region": "top",
        "class1_count": 500,
        "class2_region": "bottom",
        "class2_count": 500,
    },
    "test_dissimilar": {
        "class1_region": "bottom",
        "class1_count": 500,
        "class2_region": "top",
        "class2_count": 500,
    },
}
for split, cfg in splits.items():
    for class_label in [1, 2]:
        dir_path = os.path.join(base_dir, split, f"class{class_label}")
        os.makedirs(
            dir_path, exist_ok=True
        )

used_coords_class1 = set()
used_coords_class2 = set()


def generate_image(class_label, region):
    if region == "top":
        y_min, y_max = TOP_Y_MIN, TOP_Y_MAX
    else:
        y_min, y_max = BOTTOM_Y_MIN, BOTTOM_Y_MAX

    if class_label == 1:
        # Class 1: Red on left, Green on right
        red_x = random.randint(
            0, IMG_WIDTH - 2 * SQUARE_SIZE
        )
        red_y = random.randint(y_min, y_max)
        green_x_min = red_x + SQUARE_SIZE
        green_x = random.randint(green_x_min, IMG_WIDTH - SQUARE_SIZE)
        green_y = random.randint(y_min, y_max)
        coords = (red_x, red_y, green_x, green_y)
        if coords in used_coords_class1:
            return None
        used_coords_class1.add(coords)
    else:
        # Class 2: Green on left, Red on right
        green_x = random.randint(0, IMG_WIDTH - 2 * SQUARE_SIZE)
        green_y = random.randint(y_min, y_max)
        red_x_min = green_x + SQUARE_SIZE
        red_x = random.randint(red_x_min, IMG_WIDTH - SQUARE_SIZE)
        red_y = random.randint(y_min, y_max)
        coords = (red_x, red_y, green_x, green_y)
        if coords in used_coords_class2:
            return None
        used_coords_class2.add(coords)

    image = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    red_x, red_y, green_x, green_y = coords
    if class_label == 1:
        # Draw red square (left) and green square (right)
        draw.rectangle(
            [red_x, red_y, red_x + SQUARE_SIZE - 1, red_y + SQUARE_SIZE - 1], fill=(255, 0, 0)
        )
        draw.rectangle(
            [green_x, green_y, green_x + SQUARE_SIZE - 1, green_y + SQUARE_SIZE - 1],
            fill=(0, 255, 0),
        )
    else:
        # Draw green square (left) and red square (right)
        draw.rectangle(
            [green_x, green_y, green_x + SQUARE_SIZE - 1, green_y + SQUARE_SIZE - 1],
            fill=(0, 255, 0),
        )
        draw.rectangle(
            [red_x, red_y, red_x + SQUARE_SIZE - 1, red_y + SQUARE_SIZE - 1], fill=(255, 0, 0)
        )
    return image


for split, cfg in splits.items():
    for class_label in [1, 2]:
        region = cfg[f"class{class_label}_region"]
        num_images = cfg[f"class{class_label}_count"]
        output_dir = os.path.join(base_dir, split, f"class{class_label}")
        count = 0
        while count < num_images:
            img = generate_image(class_label, region)
            if img is None:
                continue
            # Save image with zero-padded index in filename (PNG format)
            img.save(os.path.join(output_dir, f"img_{count:04d}.png"))
            count += 1

print("Dataset generation complete.")
