import cv2
import numpy as np
import math
from utils import *
from KLT import KLT
from Sfm import Sfm
from VideoSfm import VideoSfm

################  HW2  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################


def run_sfm(imgs, iterations=40, dosfm=True):
    klt = KLT()
    corners = None
    corners = klt.feature_tracking(imgs, iterations)
    if dosfm:
        sfm = Sfm()
        sfm.structure_from_motion(imgs, corners)



def main():
    #corners('input/dinoR0001.png')
    #corners('input/dinoR0002.png')
    #corners('input/dinoR0003.png')
    imgs1 = [cv2.imread('input/p3-1-0.png'), cv2.imread('input/p3-1-1.png')]
    imgs2 = [cv2.imread('input/dinoR0004.png'), cv2.imread('input/dinoR0005.png'),
            cv2.imread('input/dinoR0006.png')]    
    imgs3 = [cv2.imread('input/templeR0001.png'), cv2.imread('input/templeR0002.png'),
            cv2.imread('input/templeR0003.png'), cv2.imread('input/templeR0004.png'),
            cv2.imread('input/templeR0005.png')]
    #imgs = [cv2.imread('input/templeR0013.png'), cv2.imread('input/templeR0014.png'),
    #        cv2.imread('input/templeR0015.png'), cv2.imread('input/templeR0016.png'),
    #        cv2.imread('input/templeR0017.png'), cv2.imread('input/templeR0018.png')]

    run_sfm(imgs1, 1, False)
    run_sfm(imgs1)
    run_sfm(imgs2)
    run_sfm(imgs3)


    """
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
    """
    
    #VideoSfm('input/out.mp4')
    

if __name__ == '__main__':
   main()


