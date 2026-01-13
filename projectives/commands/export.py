import base64
import click
import hashlib
import json
import numpy as np
import os
import scipy
import tqdm
from PIL import Image, ImageDraw

from ..utils import end, start, status

def polygon_to_binary_mask(polygon, image):
    # Convert polygon to list of pairs
    polygon = [(x, y) for x, y in polygon]
    
    # Step 1: Create a blank mask (black) with the same dimensions as the image
    mask = Image.new("L", image.size, 0)  # "L" mode is grayscale (0-255)

    # Step 2: Draw the polygon on the mask with white (255)
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)  # outline and fill with 1 (white)

    # Step 3: Convert mask to a NumPy array and make it binary
    binary_mask = np.array(mask, dtype=np.uint8)  # Array with 0s and 1s

    return binary_mask

def extract_masked_region(image: Image, binary_mask):
    image.save("before.png")

    # Convert binary_mask (PIL image) to a NumPy array
    binary_mask_np = np.array(binary_mask)
    
    # Step 1: Find the bounding box of the mask
    mask_indices = np.argwhere(binary_mask_np == 1)  # Find white pixels (mask region)
    if not mask_indices.size:
        return None
    (y_min, x_min), (y_max, x_max) = mask_indices.min(0), mask_indices.max(0) + 1

    # Step 2: Crop the original image to this bounding box
    cropped_image = image.crop((x_min, y_min, x_max, y_max))

    # Step 3: Crop the corresponding part of the binary mask
    cropped_mask = binary_mask_np[y_min:y_max, x_min:x_max]

    # Step 4: Apply the mask to retain only the masked pixels in the cropped area
    cropped_image_np = np.array(cropped_image)
    cropped_image_np[cropped_mask == 0] = 255 # Set unmasked areas to black
    
    return Image.fromarray(cropped_image_np)

@click.group()
def _export():
    pass

@_export.command()
@click.argument("directory", type=click.Path(exists=True), nargs=-1)
@click.option("--ext", default=".json", help="File extension to search for (default: .json)")
@click.option("--exif-ext", default=".jpg", help="File extension to extract EXIF from (default: .jpg)")
@click.option("--output-dir", default=".", help="Output directory for extracted masks (default: current directory)")
def export(directory, ext, exif_ext, output_dir):
    start("Scanning for files")
    todo = list(directory)
    found = []
    while todo:
        dir = todo.pop()
        for entry in os.listdir(dir):
            entry_path = os.path.join(dir, entry)
            if os.path.isdir(entry_path):
                todo.append(entry_path)
            elif entry.endswith(ext):
                found.append(entry_path)
    status(len(found), end='')
    end()

    data = {}
    for entry in tqdm.tqdm(found, desc="Parsing JSON and checking for shapes data"):
        with open(entry, 'rt') as f:
            d = json.load(f)
            if "shapes" in d:
                data[entry] = d
            else:
                print(f"Warning: {entry} has no shapes data")
    status(len(data), end='') 
    end()

    start("Extracing masked regions")
    for entry, d in tqdm.tqdm(data.items(), desc="Extracting masks and labels"):
        image_path = os.path.join(os.path.dirname(entry), d["imagePath"])
        
        # Check if the image has EXIF data
        if image_path.endswith(exif_ext):
            image = Image.open(image_path)
            
            for i, shape in enumerate(d["shapes"]):
                poly = shape["points"]
                label = shape['label']
                binary_mask = polygon_to_binary_mask(poly, image)
                binary_mask = scipy.ndimage.binary_erosion(np.array(binary_mask).astype(int), structure=np.ones((3,3)), iterations=10)
                result_image = extract_masked_region(image, binary_mask)
                if result_image is not None:
                    hash_value = base64.urlsafe_b64encode(hashlib.sha256(result_image.tobytes()).digest()).decode()
                    file_name = f"{label}_{hash_value}.png"
                    result_image.save(os.path.join(output_dir, file_name))
    end()
