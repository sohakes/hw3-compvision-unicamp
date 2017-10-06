from utils import *
import cv2
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Sfm:
    def _get_W(self, all_corners):
        U = []
        V = []
        avgus = []
        avgvs = []
        W = []
        for corners in all_corners:
            avgu = sum([c[0] for c in corners])/len(corners)
            avgv = sum([c[1] for c in corners])/len(corners)
            row_U = [c[0] - avgu for c in corners]
            row_V = [c[1] - avgv for c in corners]
            U.append(row_U)
            V.append(row_V)
            avgus.append(avgu)
            avgvs.append(avgv)
            W.append(row_U)
            W.append(row_V)

        return np.matrix(W), avgus, avgvs

    def _get_Gc(self, R):
        mid = math.floor(len(R)/2)
        its = R[:mid]
        jts = R[mid:]
        G = []
        siz = len(its)
        for i in range(len(its)):
            G.append(self._gt(its[i], its[i]))
        for i in range(len(its)):
            G.append(self._gt(jts[i], jts[i]))
        for i in range(len(its)):
            G.append(self._gt(its[i], jts[i]))

        c = [1]*(2*siz) + [0] * siz
        
        return np.matrix(G), np.transpose(np.matrix(c))


    def _gt(self, a, b):
        a = a.tolist()[0]
        b = b.tolist()[0]
        return [a[0]*b[0], a[0]*b[1] + a[1]*b[0], a[0]*b[2] +
                a[2]*b[0], a[1]*b[1], a[1]*b[2] + a[2]*b[1], a[2]*b[2]]

    def _get_l(self, lvs):
        return np.matrix([  [lvs[0], lvs[1], lvs[2]],
                            [lvs[1], lvs[3], lvs[4]],
                            [lvs[2], lvs[4], lvs[5]]])

    def structure_from_motion(self, imgs, all_corners):
        W, avgus, avgvs = self._get_W(all_corners)
        U, D, V = np.linalg.svd(W, full_matrices=False)

        #Get only the first 3
        U = np.matrix(U)
        D = np.matrix(np.diag(D[:3]))
        V = np.matrix(V)

        U = U[:,:3]
        V = V[:3,]

        Rp = U
        Sp = D * V      

        G, c = self._get_Gc(Rp)
        lvs, res, rank, s = np.linalg.lstsq(G, c)

        l = self._get_l(lvs.flatten().tolist()[0])
        Q = np.linalg.cholesky(l)

        R = np.dot(Rp, Q)
        S = np.dot(np.linalg.pinv(Q), Sp)

        Rs = []
        Ts = []
        #recover R and T
        for i in range(len(imgs)):
            idx = 2*i
            curr_r = R[idx]
            curr_mult = np.matrix([[avgus[i]], [avgvs[i]], [1]])
            curr_t = curr_r * curr_mult


        r1 = np.transpose(R[0])
        r2 = np.transpose(R[1])
        R0 = np.hstack((r1, r2, np.transpose(np.cross(np.transpose(r1), np.transpose(r2)))))
        R = np.dot(R, R0)
        S = np.dot(np.transpose(R0), S)

        xs = S[0,:].tolist()[0]
        ys = S[1,:].tolist()[0]
        zs = S[2,:].tolist()[0]
        maxxs = max(xs)
        minxs = min(xs)
        maxys = max(ys)
        minys = min(ys)
        rngx = abs(minxs) + abs(maxxs)
        rngy = abs(maxys) + abs(minys)
        f = open(get_file_name(2), 'w')
        f2 = open(get_file_name(2), 'w')
        hei, wid, _ = imgs[0].shape
        ply_header = '''ply
                        format ascii 1.0
                        element vertex %d
                        property float x
                        property float y
                        property float z
                        property uchar red
                        property uchar green
                        property uchar blue
                        end_header
                        '''
        f2.write(ply_header % (len(imgs)))
        f.write(ply_header % (len(xs)))


        for x, y, z in zip(xs, ys, zs):
            f.write("%f %f %f %d %d %d\n" % (x, y, z, 0, 255, 255))

        for i in range(len(imgs)):
            idx = i*2
            x1 = R[idx, :].tolist()[0]
            y1 = R[idx+1, :].tolist()[0]
            pos =  np.cross(x1, y1)
            f2.write("%f %f %f %d %d %d\n" % (pos[0], pos[1], pos[2], 255, 0, 0))





    def __init__(self):
        pass