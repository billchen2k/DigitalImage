from PIL import Image
import numpy as np
import matplotlib.pyplot as histogram

img=np.array(Image.open('equalize_hisyogram.jpg'))
 
histogram.figure("histogram")
arr=img.flatten()
patches=histogram.hist(arr,bins=256, normed=1, facecolor='black', alpha=0.75)  
histogram.show()