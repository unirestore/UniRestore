import os
import csv
import numpy as np

from PIL import Image
# from corruption import pixelate
from pathlib import Path

from tqdm import tqdm
# from corruption import corruptions
from threading import Thread
from multiprocessing import Process, Queue


def open_image(imagepath):
    if isinstance(imagepath, Path):
        imagepath = imagepath.resolve().as_posix()

    im = Image.open(imagepath, "r")
    im = im.convert("RGB")
    return im


def crop_to_bounding_box(image, bbox):
    x, y, w, h = bbox
    w = w + x
    h = y + h
    bbox = (x, y, w, h)
    cropped_image = image.crop(bbox)
    return cropped_image


def to_image(image):
    return Image.fromarray(np.uint8(image))


def main():
    pass


def read_csv(label_path):
    print(f"loading form {label_path}")
    with open(label_path) as csv_file:
        csv_file_rows = csv.reader(csv_file, delimiter=",")
        for row in csv_file_rows:
            yield row

def generate_C_dataset(image_path, id, divider=2):

    image_path = Path(image_path).iterdir()
    image_path = [d for d in image_path]
    image_path = np.array_split(image_path, divider)[id]

    for image_pth in tqdm(image_path):
        image_pth = image_pth.resolve().as_posix()
        im = open_image(image_pth)
        im = im.resize((256, 256))
        image_pth = "/".join(image_pth.split("/")[-1:])

        for crp in crp_list:
            crp_func = corruptions[crp]
            for sev in [3]:
                crp_image = crp_func(im, sev)
                savedir = Path(f"/data/ILSVRC2012-C/train/{crp}/{sev}/{image_pth}")
                if not savedir.parents[0].exists():
                    Path.mkdir(savedir.parents[0], parents=True)
                elif savedir.exists():
                    continue
                crp_image = restore_image_from_numpy(crp_image)
                crp_image.save(savedir.resolve().as_posix())


if __name__ == "__main__":
    crp_list = ["gaussian_noise", "shot_noise", "impulse_noise",
                    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                    "snow", "frost", "fog",
                    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
                    "speckle_noise", "gaussian_blur", "spatter", "saturate"]
    image_path = Path(os.environ.get("IMAGENET_TRAIN_IMAGE"))

    threads = []
    divide = 80
    result = Queue()
    for i in range(divide):
        threads.append(Process(target=generate_C_dataset, args=(image_path, i, divide)))

    for t in threads:
        t.start()

    for t in threads:
        t.join()


