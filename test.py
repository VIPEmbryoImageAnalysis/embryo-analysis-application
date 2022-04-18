import numpy as np
from PIL import Image
import os
import cv2
from natsort import natsorted

dir = "/Embryo Image Analysis Python/4240Dataset_ICM_New/ELabeled"
#os.makedirs('/Embryo Image Analysis Python/212Dataset_TE/TestingSet/New')
dir = natsorted(os.listdir(dir))
print(dir)
for filename in dir:
    filename1 = os.path.join("/Embryo Image Analysis Python/4240Dataset_ICM_New/ELabeled", filename)
    print(filename1)
    im = cv2.imread(filename1)
    im[im == 1] = 0 # change everything to white where pixel is not black
    im[im == 2] = 1
    path = os.path.join("/Embryo Image Analysis Python/4240Dataset_ICM_New1/ELabeled", filename)
    cv2.imwrite(path, im)