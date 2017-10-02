from utils import *
import cv2
import numpy as np
import math
from scipy import interpolate

class KLT:
    #gotten from here https://stackoverflow.com/questions/17931613/how-to-decide-a-whether-a-matrix-is-singular-in-python-numpy
    def _is_invertible(self, a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

    def _corners(self, img_g_1):
        

        corners = cv2.goodFeaturesToTrack(img_g_1, 100, 0.01, 10)
        corners = np.int0(corners)
        img_g_1t = img_g_1.copy()
        for corner in corners:
            x,y = corner.ravel()
            cv2.circle(img_g_1t,(x,y), 2, (0,255,0), 1)

        debug('corner1', img_g_1t)
        return [c.ravel() for c in corners]

    def _normto1(self, v):
        return (v.astype('float'))/255

    def _getval(self,im, x, y):
        restx = x - math.floor(x)
        resty = y - math.floor(y)
        f00 = im[math.floor(y)][math.floor(x)]
        f10 = im[math.floor(y)][math.ceil(x)]
        f01 = im[math.ceil(y)][math.floor(x)]
        f11 = im[math.ceil(y)][math.ceil(x)]
        val = f00*(1-restx)*(1-resty) + f10*restx*(1-resty) + f01*(1-restx)*resty + f11*restx*resty
        #print(x, y, restx, resty, f00, f10, f01, f11, val)
        return val


    def _gradtpoint(self, im1, im2, x, y, u, v):
        h, w = im2.shape
        if x + u < 1 or x + u >= w-1 or y + v < 1 or y + v >= h-1:
            return None
        #print(im1[y,x])
        #print(im2[int(round(y + v)),int(round(x + u))])
        return ((float(im1[y,x])) - (float(self._getval(im2, x+u, y+v))))/255

    def _get_A_b(self, im1, im2, px, py, derivx, derivy, u=0, v=0, neigh_size=15):
        assert neigh_size % 2 == 1

        #derivx = self._normto1(derivx)
        #derivy = self._normto1(derivy)
        #derivt = self._normto1(derivt)

        h, w = derivx.shape
        rang = math.floor(neigh_size / 2)

        if py > h - rang or py < rang or px > w - rang or px < rang:
            return None
        
        sixx = sixy = siyy = sixt = siyt = 0

        for y in range(py - rang, py + rang + 1):
            for x in range(px - rang, px + rang + 1):
                sixx += derivx[y,x] ** 2
                siyy += derivy[y,x] ** 2
                derivtn = self._gradtpoint(im1, im2, x, y, u, v)
                if derivtn is None:
                    return None
                
                sixy += derivx[y,x] * derivy[y,x]
                sixt += derivx[y,x] * derivtn
                siyt += derivy[y,x] * derivtn

        #print(sixx, siyy, sixy, sixt, siyt)

        A = np.matrix([[sixx, sixy], [sixy, siyy]])
        b = np.matrix([[sixt], [siyt]])
        return (A, b)

    def _gradxy(self, im):
        imgradx = self._normto1(im)
        imgrady = self._normto1(im)
        im = self._normto1(im)
        #im = im.astype('float')
        #imgradx = im.astype('float')
        #imgrady = im.astype('float')
        h, w = im.shape
        for y in range(h):
            for x in range(w):
                if x == w - 1 or x == 0:
                    imgradx[y,x] = 0
                    continue
                if y == h - 1 or y == 0:
                    imgrady[y,x] = 0
                    continue
                imgradx[y,x] = (im[y,x + 1] - im[y,x - 1])/2
                imgrady[y,x] = (im[y + 1,x] - im[y - 1,x])/2
                print(im[y,x + 1],im[y,x - 1],imgradx[y,x])
        return imgradx, imgrady


    
    def _new_xy(self, im1o, im2o, corners):

        #sobelx = cv2.Sobel(img_g_1,cv2.CV_64F,1,0,ksize=3)
        #sobely = cv2.Sobel(img_g_1,cv2.CV_64F,0,1,ksize=3)
        im1 = cv2.cvtColor(im1o, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2o, cv2.COLOR_BGR2GRAY)
        img_g_1 = im1.copy()
        sobelx, sobely = self._gradxy(img_g_1)
        debug('sobelx', sobelx)
        debug('sobely', sobely)
        #derivt = imgsbw[0] - imgsbw[1]

        #im1 = imgs[0].copy()
        #im2 = imgs[1].copy()
        moving_u_v = [(0,0)]*len(corners)
        
        for k in range(30):
            count = 0
            sumu = 0
            sumv = 0
            #corners_new = []
            u = v = 0
            for i in range(len(corners)):
                x, y = corners[i]
                res1 = moving_u_v[i]
                if res1 is None:
                    continue
                u, v = res1
                res = self._get_A_b(im1, im2, x, y, sobelx, sobely, u, v)
                if res is None:
                    moving_u_v[i] = None
                    continue
                A, b = res
                if self._is_invertible(A):
                    uv = np.linalg.inv(A) * b
                    #print('inv, b', np.linalg.inv(A), b)
                    #print(uv)
                    u, v = np.asscalar(uv[0][0]), np.asscalar(uv[1][0])
                    moving_u_v[i] = (moving_u_v[i][0] + u, moving_u_v[i][1] + v)
                    u, v = moving_u_v[i]
                    #from_to.append(((x, y), (x + u, y + v)))
                    #corners_new.append((int(round(x + u)),int(round(y + v))))
                    
                    
                    #print('uv', u, v)
                    count+=1
                    sumu+=u
                    sumv+=v
            #corners = corners_new

            #print('means u v', sumu/count, sumv/count)
        moved_u_v = []
        for i in range(len(corners)):
            x, y = corners[i]
            if moving_u_v[i] is None:
                continue
            u, v = moving_u_v[i]
            #GET SUBPIXEL IF YOU WANNA FIX THIS
            moved_u_v.append((int(round(x + u)),int(round(y + v)))) 
        
        return moved_u_v



    def feature_tracking(self, imgs):
        assert len(imgs) > 1
        #imgsbw = []
        #for im in imgs:
        #    imgsbw.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        #im1 = imgs[0].copy()
        img_g_1 = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
        corners = self._corners(img_g_1)
        
        for idx in range(len(imgs) - 1):
            uvs = self._new_xy(imgs[idx], imgs[idx + 1], corners)
            print(uvs)
            im1 = imgs[idx].copy()
            im2 = imgs[idx + 1].copy()

            for i in range(len(corners)):
                x, y = corners[i]                
                cv2.circle(im1,(x,y), 2, (0,255,0), 1)

                if i < len(uvs):
                    x1, y1 = uvs[i]
                    cv2.circle(im2,(x1, y1), 2, (0,255,0), 1)

            debug('im1', im1)
            debug('im2', im2)
            corners = uvs



    def __init__(self):
        pass
