import _PIGS
from PIL import Image
import numpy as np

filepath = '../data/flower.jpg'
image = Image.open()
ndarray_image = np.array(image)

pig = PIGS(ndarray_image)

pig.compute_gram()


