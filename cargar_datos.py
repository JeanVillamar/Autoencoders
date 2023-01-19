
from PIL import Image, ImageFilter, ImageChops
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os

PATH_GOOD = "./crops_good/"
PATH_BAD = "./crops_bad/"

SIZE_INPUT = 128

def load_dataset():
    array_good =[] 
    array_bad = [] 
    
    good_photos = sorted(os.listdir(PATH_GOOD)) # Nombres de las fotos
    
    bad_photos = sorted(os.listdir(PATH_BAD)) 
  
    
    
    total = 0
    size = len(good_photos)
    print(good_photos)
    for name in good_photos:
        image_good = Image.open(PATH_GOOD+name)
        image_bad = Image.open(PATH_BAD+name)
        print(image_good)
        array_good.append(np.array(image_good))
        array_bad.append(np.array(image_bad))
        
        total +=1
        print("Loaded {} of {}".format(total, size))
    print(array_good)
    return (array_good, array_bad)
        
def load_image(path):
    image = Image.open(path)
    arr_boxes = []
    size_x = image.width
    size_y = image.height
    
    batches_width = size_x // SIZE_INPUT
    batches_height = size_y // SIZE_INPUT
    
    print("{}x{}".format(batches_height, batches_width))
        
    for i in range(128, size_y, SIZE_INPUT):
        for j in range(128, size_x, SIZE_INPUT):
            crop_good = image.crop((j-128, i-128, j, i))
            arr_boxes.append(np.array(crop_good))

    return (arr_boxes, batches_width, batches_height)