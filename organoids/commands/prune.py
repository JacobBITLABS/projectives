import PIL
import click
import cv2
import exif
import json
import labelme
import numpy as np
import os
import pickle
import scipy.ndimage
import shapely
import tqdm
import zstandard as zstd

from organoids.utils import end, start, status


@click.group()
def _prune():
    pass
@_prune.command()
@click.argument("file-or-directory", type=click.Path(exists=True), nargs=-1)
@click.option("--ext", default=".jpg", help="File extension to search for (default: .jpg)")
@click.option("--json-ext", default=".json", help="File extension to save meta data to (default: .json)")
@click.option("--pickle-ext", default=".pickle", help="File extension to load masks from (default: .pickle)")
@click.option("--eps", default=0.0005, help="Epsilon for polygon approximation (default: .001)")
@click.option("--min-size", default=0.00005, help="Minimum size of mask as fraction of total area (default: 0.001)")
@click.option("--max-size", default=0.35, help="Maximum size of mask as fraction of total area (default: 0.5)")
@click.option("--brightness-threshold", default=1.5, help="Brightness threshold for masks (default: 1.25)")
@click.option("--regularity-threshold", default=5, help="Regularity threshold for masks (default: 2)")
@click.option("--subsumption-threshold", default=0.9, help="Threshold for how much overlap with other circles(!) is tolerated (default: 0.9)")
@click.option("--empty-threshold", default=192, help="Threshold for how dark pixels there should (default: 192)")
def prune(file_or_directory, ext, json_ext, pickle_ext, eps, min_size, max_size, brightness_threshold, regularity_threshold, subsumption_threshold, empty_threshold):
    start("Scanning for files")
    todo = list(file_or_directory)
    found = []

    while todo:
        file_or_directory = todo.pop()
        if os.path.isfile(file_or_directory) and file_or_directory.endswith(ext):
            found.append(file_or_directory)
        elif os.path.isdir(file_or_directory):
            todo.extend(os.path.join(file_or_directory, entry) for entry in os.listdir(file_or_directory))
    found.sort()
    status(len(found), end='')
    end()
    meta = {}
    for entry in tqdm.tqdm(found, desc="Reading EXIF data and checking for user_comments and pixel_dimension fields"):
        with open(entry, 'rb') as f:
            try:
                e = exif.Image(f)
            except:
                e = None
            if e is not None and e.has_exif:
                if hasattr(e, "pixel_x_dimension") and hasattr(e, "pixel_y_dimension") and hasattr(e, "user_comment"):
                    user_comment = json.loads(e.user_comment)
                    meta[entry] = dict(
                        width=e.pixel_x_dimension,
                        height=e.pixel_y_dimension,
                        pix_size=user_comment["effectivePixelSize"],
                        mag=user_comment["objectiveMag"],
                    )
                else:
                    print(f"Warning: {entry} has no pixel dimensions in EXIF data")
                    meta[entry] = dict(
                        width=e.pixel_x_dimension,
                        height=e.pixel_y_dimension,
                        pix_size=1,
                        mag=1,
                    )
            else:
                print(f"Warning: {entry} has no EXIF data")
                e = PIL.Image.open(f)
                x_dimension, y_dimension = e.size
                meta[entry] = dict(
                    width=x_dimension,
                    height=y_dimension,
                    pix_size=1,
                    mag=1,
                )
    status(len(meta), end='')
    end()
    start("Pruning masks")
    for entry in tqdm.tqdm(meta, desc="Pruning masks"):
        width = meta[entry]["width"]
        height = meta[entry]["height"]
        print(f"Loading {entry}")
        image = PIL.Image.open(entry).convert("RGB")
        with open(os.path.splitext(entry)[0]+pickle_ext+".zst", 'rb') as f:
            g = zstd.ZstdDecompressor().stream_reader(f)
            loaded_masks = pickle.load(g)
        masks = []
        for mask in loaded_masks:
            if any(row[0] for row in mask) or any(row[-1] for row in mask):
                print("Skipping as mask touches left or right image border")
                continue
            masked = sum(1 for row in mask for pixel in row if pixel)
            total = sum(len(row) for row in mask)
            if masked/total < min_size:
                print(f"Skipping as mask is less than {min_size*100}% of total area: {masked/total*100}%")
                continue
            if masked/total > max_size:
                print(f"Skipping as mask is more than {max_size*100}% of total area: {masked/total*100}%")
                continue
            print(f"Mask is {masked/total*100}% of total area")
            masks.append(mask)
        super_masks = []
        for i, m1 in enumerate(masks):
            for j, m2 in enumerate(masks):
                if i != j and m2[m1].all():
                    print("Skipping as mask is subsumed by another mask")
                    break
            else:
                super_masks.append(m1)
        polygons = []
        for mask in super_masks:
            image_pixels = np.array(image)
            masked_pixels = image_pixels[mask]
            masked_pixels_mean = masked_pixels.mean()
            shrinked_mask = scipy.ndimage.binary_erosion(np.array(mask).astype(int), structure=np.ones((3,3)), iterations=10)
            shrinked_pixels = image_pixels[shrinked_mask]
            if len([pix for row in shrinked_pixels for pix in row if pix < empty_threshold]) < 10:
                print(f"Skipping as masked area appears empty")
                continue
            image_pixels_mean = image_pixels[height//3:2*height//3,width//3:2*width//3].mean()
            if masked_pixels_mean > brightness_threshold*image_pixels_mean:
                print(f"Skipping as mask is too bright: {masked_pixels_mean} > {brightness_threshold} * {image_pixels_mean}")
                continue
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            obj = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(obj, True)
            approx = cv2.approxPolyDP(obj, eps*peri, True)
            approx_x = approx[:,:,0]
            approx_y = approx[:,:,1]
            center, _ = cv2.minEnclosingCircle(approx)
            distances = np.sqrt((approx_x-center[0])**2 + (approx_y-center[1])**2)
            min_dist, max_dist = distances.min(), distances.max()
            if max_dist > regularity_threshold*min_dist:
                print(f"Skipping as mask is too irregular: {max_dist/min_dist} > {regularity_threshold}")
                continue
            poly = [
                [int(p[0][0]), int(p[0][1])] for p in approx
            ]
            polygons.append(poly)
        # compute overlap matrix
        while True:
            shapely_polygons = [shapely.geometry.Polygon(poly) for poly in polygons]
            overlap_matrix = [[poly1.intersection(poly2).area/poly1.area for poly2 in shapely_polygons] for poly1 in shapely_polygons]
            for i in range(len(overlap_matrix)):
                overlap_matrix[i][i] = 0
            #print("\n".join(str(row) for row in overlap_matrix))
            #print(f"subsumption_threshold = {subsumption_threshold}")
            #print([sum(row) for row in overlap_matrix])
            indices_to_delete = [i for i in range(len(polygons)) if sum(overlap_matrix[i]) >= subsumption_threshold and all(shapely_polygons[j].area < shapely_polygons[i].area for j in range(len(polygons)) if overlap_matrix[i][j] > 0)]
            if not indices_to_delete:
                break
            print(f"Deleting {indices_to_delete[0]}")
            del polygons[indices_to_delete[0]]
        areas = []
        for poly in polygons:
            poly = shapely.geometry.Polygon(poly)
            area = poly.area*meta[entry]["pix_size"]/10**6/meta[entry]["mag"]
            areas.append(area)
        lf = labelme.LabelFile()
        shapes = [dict(
            label="auto",
            points=poly,
            group_id=None,
            description=f"{area} mm²",
            shape_type="polygon",
            flags={},
            mask=None,
        ) for poly, area in zip(polygons, areas)]
        flags = {}
        imagePath = os.path.basename(entry)
        imageData = open(entry, 'rb').read()
        lf.save(
            filename=os.path.splitext(entry)[0]+json_ext,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=height,
            imageWidth=width,
            otherData=None,
            flags=flags,
        )
    end()


if __name__ == "__main__":
    print("HELLO")
    prune()
