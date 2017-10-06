import cv2
import numpy as np
from utils import *
from KLT import KLT
from Sfm import Sfm


class VideoSfm:
    def __init__(self, src_video_file_path):
        print(src_video_file_path)
        cap = cv2.VideoCapture(src_video_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + 0  # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 0# float
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(width, height, length)
        it = 0
        first_frame = None
        print("Skipping first 30 frames...")
        frames = []
        print("lenght:", 15)
        itin = 0
        while(cap.isOpened()):
            #print('in')
            ret, frame = cap.read()
            it+=1
            if it < 40 or it %1 != 0:
                print("continuing", it, itin)
                continue
            if (itin > 8):
                break
            
            if ret==True:
                itin += 1
                
                frames.append(frame)
            else:
                break

        klt = KLT()
        corners = None
        try:
            corners = np.load('corners.npy')
            print('loaded')
        except IOError:
            corners = klt.feature_tracking(frames)
            np.save('corners.npy', corners)
            print('ioerror')
        sfm = Sfm()
        sfm.structure_from_motion(frames, corners)

        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()