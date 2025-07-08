import PIL
import click
import gc
import os
import pickle
import torch
import tqdm
import transformers
import numpy as np
import matplotlib.pyplot as plt
from zstandard import ZstdCompressor
from organoids.utils import end, start, status

@click.group()
def _segment():
    pass
@_segment.command()
@click.argument("file-or-directory", type=click.Path(exists=True), nargs=-1)
@click.option("--model", default="facebook/sam-vit-base", help="Model to use for segmentation (default: facebook/sam-vit-base)")
@click.option("--ext", default=".jpg", help="File extension to search for (default: .jpg)")
@click.option("--viz", default=True, help="Visualize)")
@click.option("--pickle-ext", default=".pickle", help="File extension to save masks to (default: .pickle)")
@click.option("--points-per-crop", default=60, help="Number of points per crop (default: 24)")
@click.option("--device", default="cpu", help="Device to use for segmentation (cpu, mps, cuda) (default: cpu)")
def segment(file_or_directory, model, ext, viz, pickle_ext, points_per_crop, device):
    backend = getattr(torch, device.split(":")[0])
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
    start("Loading segmentation model")
    generator = transformers.pipeline("mask-generation", model=model, device=device, torch_dtype=torch.float32)
    end()
    start("Segmenting images")
    for entry in tqdm.tqdm(found, desc="Segmenting images"):
        print(f"Segmenting {entry}")
        image = PIL.Image.open(entry).convert("RGB")
        outputs = generator(image, points_per_batch=points_per_crop, points_per_crop=points_per_crop)
        masks = []
        for mask in outputs["masks"]:
            masks.append(mask)

        if viz:
            plt.imshow(np.array(image))
            ax = plt.gca()
            for mask in outputs["masks"]:
                show_mask(mask, ax=ax, random_color=True)
            plt.axis("off")
            plt.savefig(f"/Users/jacobnielsen/Documents/PROJECTS/Projectives/organoids/organoids/commands/seg_outputs/{os.path.basename(entry)}")
            
        pickle_path = os.path.splitext(entry)[0]+pickle_ext+".zst"
        print("pickled path: ", pickle_path)
        with open(pickle_path, 'wb') as f:
            compressor = ZstdCompressor()
            with compressor.stream_writer(f) as compressed_f:
                pickle.dump(masks, compressed_f)
        del outputs
        del masks
        del image
        gc.collect()
        backend.empty_cache()
    end()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


if __name__ == "__main__":
    print("HELLO")
    segment()
