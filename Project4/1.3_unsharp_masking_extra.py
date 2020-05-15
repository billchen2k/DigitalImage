import imageio
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter, maximum_filter, minimum_filter
from skimage import img_as_float

def UM(input, output, op):
    radius = 5
    amount = 2
    image = imageio.imread(input)
    image = img_as_float(image) # ensuring float values for computations
    if op == 1:
        blurred_image = gaussian_filter(image, sigma=radius)
    elif op == 2:
        blurred_image = minimum_filter(image, size=20)
    elif op == 3:
        blurred_image = maximum_filter(image, size=20)
    mask = image - blurred_image # keep the edges created by the filter
    sharpened_image = image + mask * amount
    sharpened_image = np.clip(sharpened_image, float(0), float(1)) # Interval [0.0, 1.0]
    sharpened_image = (sharpened_image*255).astype(np.uint8) # Interval [0,255]
    imageio.imwrite(output, sharpened_image[:, :, 0])

UM('input/胸透X.jpg', 'output/胸透X_output_1.jpg', 1)
UM('input/胸透X.jpg', 'output/胸透X_output_2.jpg', 2)
UM('input/胸透X.jpg', 'output/胸透X_output_3.jpg', 3)
UM('input/胸透X2.jpg', 'output/胸透X2_output_1.jpg', 1)
UM('input/胸透X2.jpg', 'output/胸透X2_output_2.jpg', 2)
UM('input/胸透X2.jpg', 'output/胸透X2_output_3.jpg', 3)