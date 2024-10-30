from PIL import Image
import os
import shutil

trash_folder = os.path.join(os.getcwd(), "trash")

"""
This script ensures that all files are of the same dimension. 
Result: all files are 48×48.
Used Dataset: https://www.kaggle.com/datasets/msambare/fer2013
"""

def check_image_dimensions(path, width, height):
    for filename in os.listdir(path):

        file_path = os.path.join(path, filename)

        try:
            with Image.open(file_path) as img:
                if img.size == (48, 48):
                    continue
                else:
                    shutil.move(file_path, trash_folder)
        except:
            continue


directory = os.path.join(os.getcwd(), "data", "train")

for root, dirs, files in os.walk(directory):
    for dir_name in dirs:
        check_image_dimensions(os.path.join(directory, dir_name), 48, 48)