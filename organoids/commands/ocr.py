import torch
import click
import json
import numpy as np
import os
import re
import cv2
import scipy.ndimage
import tqdm
from PIL import Image, ImageDraw
from .cifarx import CifarXModel
from shapely.geometry import Polygon, MultiPoint
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import time
from organoids.utils import end, start, status
import time

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


def resize_image_pil(image, target_size, method='bicubic'):
    """
    Resize image using PIL with different interpolation methods.
    
    Parameters:
    image: PIL Image or numpy array
    target_size: tuple of (width, height)
    method: str, one of ['nearest', 'box', 'bilinear', 'hamming', 'bicubic', 'lanczos']
    
    Returns:
    PIL Image
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Map method names to PIL constants
    methods = {
        'nearest': Image.NEAREST,  # Fastest but lowest quality
        'box': Image.BOX,         # Box sampling
        'bilinear': Image.BILINEAR,  # Linear interpolation
        'hamming': Image.HAMMING,    # Hamming windowed sinc filter
        'bicubic': Image.BICUBIC,    # Cubic spline interpolation
        'lanczos': Image.LANCZOS     # High-quality Lanczos filter
    }
    
    # Resize the image
    resized = image.resize(target_size, methods.get(method, Image.BICUBIC))
    return resized

def add_padding(image, padding=50):
    """
    Add white padding around the image to improve Tesseract recognition.
    
    Args:
        image (PIL.Image): Input grayscale image
        padding (int): Number of pixels to pad on each side
    Returns:
        PIL.Image: Padded image
    """
    # Create a new white image with padding
    new_size = (image.width + 2*padding, image.height + 2*padding)
    padded_image = Image.new('L', new_size, 255)
    
    # Paste the original image in the center
    padded_image.paste(image, (padding, padding))
    return padded_image

def smooth_edges(image, kernel_size):
    """Smooths edges using Gaussian blur and morphological operations."""
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred

def extract_masked_region(image: Image, binary_mask, verbose):
    image.save("before.png")

    # Convert binary_mask (PIL image) to a NumPy array
    binary_mask_np = np.array(binary_mask)
    
    # Step 1: Find the bounding box of the mask
    mask_indices = np.argwhere(binary_mask_np == 1)  # Find white pixels (mask region)
    (y_min, x_min), (y_max, x_max) = mask_indices.min(0), mask_indices.max(0) + 1

    # Step 2: Crop the original image to this bounding box
    cropped_image = image.crop((x_min, y_min, x_max, y_max))

    # Step 3: Crop the corresponding part of the binary mask
    cropped_mask = binary_mask_np[y_min:y_max, x_min:x_max]

    # Step 4: Apply the mask to retain only the masked pixels in the cropped area
    cropped_image_np = np.array(cropped_image)
    cropped_image_np[cropped_mask == 0] = 255 # Set unmasked areas to black

    # Step 6: Convert the result back to a PIL image
    result_image = Image.fromarray(cropped_image_np)
    return result_image

@click.group()
def _ocr():
    pass

@_ocr.command()
@click.argument("directory", type=click.Path(exists=True), nargs=-1)
@click.option("--ext", default=".json", help="File extension to search for (default: .json)")
@click.option("--exif-ext", default=".jpg", help="File extension to extract EXIF from (default: .jpg)")
@click.option("--verbose", default=True, help="Enable verbose output (default: True)")
@click.option("--evaluate", default=None, type=click.Path(exists=True), help="Provide directory with GT JSON. Must have have name as images")
def ocr(directory, ext, exif_ext, verbose, evaluate):
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
                print(d["shapes"], d["imageWidth"], d["imageHeight"])
            else:
                print(f"Warning: {entry} has no shapes data")
    status(len(data), end='') 
    end()

    cifar_model = CifarXModel()
    cifar_model.load_state_dict(torch.load('organoids/commands/final_digit_model.pth')['model_state_dict'])
    cifar_model.eval()
    
    start("Extracing masked regions and performing OCR")
    for entry, d in tqdm.tqdm(data.items(), desc="Performing OCR on masked regions"):
        image_path = os.path.join(os.path.dirname(entry), d["imagePath"])
        
        # Check if the image has EXIF data
        if image_path.endswith(exif_ext):
            image = Image.open(image_path).convert("RGB")
            
            if evaluate:
                gt_basename = image_path.replace(".jpg", ".json")
                gt_path = os.path.join(evaluate, os.path.basename(gt_basename)) 
                print("gt_path: ", gt_path)

                with open(gt_path, 'rt') as f:
                    gt = json.load(f)
                    gt_shapes = gt["shapes"] # load all shapes from given images

            for i, shape in enumerate(d["shapes"]):
                poly = shape["points"]
                binary_mask = polygon_to_binary_mask(poly, image)
                binary_mask = scipy.ndimage.binary_erosion(np.array(binary_mask).astype(int), structure=np.ones((3,3)), iterations=10)
                
                result_image = extract_masked_region(image, binary_mask, verbose=verbose)    
                result_image = result_image.convert('L')
                if verbose:
                    result_image.save("grayscale.png")
                
                result_image = image.convert('L')
                result_image = Image.open("grayscale.png")
                prediction, probabilities = cifar_model.classify(result_image)
                shape["label"] = f"{str(prediction)}"
        
    end()
    start("Writing recognized numbers to disk")
    for entry, d in tqdm.tqdm(data.items(), desc="Writing areas to disk"):
        print(f"entry: {entry}")
        with open(entry, 'wt') as f:
            json.dump(d, f, indent=2)
    end()


if __name__ == "__main__":
    print("HELLO")
    ocr()

    # threshold = 55
    # threshold = np.mean(np.array(padded_image))
    # #binary_image = padded_image.point(lambda x: 0 if x < threshold else 255, '1')
    # #binary_image.save("black_white.png")

    # result = pytesseract.image_to_string(padded_image, config=r'--oem 3 --psm 10 -c tessedit_char_whitelist=01234567891012')
    # # number = ''.join(filter(str.isdigit, result))
    # print("TESSER CLASSI: ", result)
    # python3 -m organoids ocr /Users/jacobnielsen/Documents/PROJECTS/Projectives/organoids/gt_testing-images10