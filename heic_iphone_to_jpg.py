# convert heic to jpg

import os
import sys
import glob
import PIL
import PIL.Image
import cv2 as opencv
from pillow_heif import register_heif_opener

register_heif_opener()


def heic_to_jpg(folder):
    for path in os.listdir(folder):
        if path.endswith(".HEIC"):
            im = PIL.Image.open(os.path.join(folder, path))
            im.save(folder + "/" + path[:-5] + ".jpg", "JPEG")
            os.remove(folder + "/" + path)
            print(im)


heic_to_jpg("test")
