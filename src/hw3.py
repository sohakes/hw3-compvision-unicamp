import cv2
import numpy as np
import math
from utils import *

################  HW2  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################


def harris(imgpath):
    img1 = cv2.imread(imgpath)

    img_g_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    corner1 = cv2.cornerHarris(img_g_1, 2, 3, 0.04)

    #corner1 = cv2.normalize(corner1, corner1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    #corner1 = cv2.convertScaleAbs(corner1)
    h, w = corner1.shape
    for y in range(h):
        for x in range(w):
            if corner1[y][x] > 10e-05:
                img1 = cv2.circle(img1,(x, y), 2, (0,255,0), 1)

    debug('corner1', img1)

#seen in this tutorial https://pythonprogramming.net/corner-detection-python-opencv-tutorial/
def corners(imgpath):
    img1 = cv2.imread(imgpath)

    img_g_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(img_g_1, 100, 0.01, 10)
    corners = np.int0(corners)
    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(img1,(x,y), 2, (0,255,0), 1)

    debug('corner1', img1)

def main():
    corners('input/templeRing/templeR0031.png')
    corners('input/templeRing/templeR0030.png')
    corners('input/templeRing/templeR0029.png')
    

if __name__ == '__main__':
   main()


