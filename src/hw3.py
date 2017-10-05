import cv2
import numpy as np
import math
from utils import *
from KLT import KLT
from Sfm import Sfm

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
    #corners('input/dinoRing/dinoR0001.png')
    #corners('input/dinoRing/dinoR0002.png')
    #corners('input/dinoRing/dinoR0003.png')
    imgs = [cv2.imread('input/dinoRing/dinoR0001.png'), cv2.imread('input/dinoRing/dinoR0002.png'),
            cv2.imread('input/dinoRing/dinoR0003.png'),cv2.imread('input/dinoRing/dinoR0004.png'),
            cv2.imread('input/dinoRing/dinoR0005.png'),cv2.imread('input/dinoRing/dinoR0006.png')]
    #imgs = [cv2.imread('input/p1-1-1.png'), cv2.imread('input/p1-1-2.png')]
    klt = KLT()
    corners = None
    try:
        corners = np.load('corners.npy')
        print('loaded')
    except IOError:
        corners = klt.feature_tracking(imgs)
        np.save('corners.npy', corners)
        print('ioerror')
    sfm = Sfm()
    sfm.structure_from_motion(imgs, corners)
    

if __name__ == '__main__':
   main()


