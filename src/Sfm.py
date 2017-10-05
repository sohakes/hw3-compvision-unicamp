from utils import *
import cv2
import numpy as np
import math


class Sfm:
    def _get_W(self, all_corners):
        U = []
        V = []
        avgus = []
        avgvs = []
        for corners in all_corners:
            avgu = sum([c[0] for c in corners])/len(corners)
            avgv = sum([c[1] for c in corners])/len(corners)
            row_U = [c[0] - avgu for c in corners]
            row_V = [c[1] - avgv for c in corners]
            U.append(row_U)
            V.append(row_V)
            avgus.append(avgu)
            avgvs.append(avgv)
        return np.vstack((U, V)), avgus, avgvs

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
        print('more lvs', lvs)
        return np.matrix([  [lvs[0], lvs[1], lvs[2]],
                            [lvs[1], lvs[3], lvs[4]],
                            [lvs[2], lvs[4], lvs[5]]])

    def structure_from_motion(self, imgs, all_corners):
        W, avgus, avgvs = self._get_W(all_corners)
        U, D, V = np.linalg.svd(W, full_matrices=False)
        print('shapes',U.shape, D.shape, V.shape)
        #Get only the first 3
        U = np.matrix(U)
        D = np.matrix(np.diag(D[:3]))
        V = np.matrix(V)
        #print(np.dot(U, np.dot(D, V)))
        print("V", V, V.shape)
        U = U[:,:3]
        V = V[:3,]
        print("W", W, W.shape)
        print("U", U, U.shape)
        print("D", D, D.shape)
        print("V", V, V.shape)
        Rp = U
        Sp = D * V      
        print("R", Rp, Rp.shape)
        print("S", Sp, Sp.shape)
        #print("RS", Rp*np.transpose(V))

        G, c = self._get_Gc(Rp)
        lvs, res, rank, s = np.linalg.lstsq(G, c)
        print('lvs', lvs)
        l = self._get_l(lvs.flatten().tolist()[0])
        Q = np.linalg.cholesky(l)
        print(Q)
        R = np.dot(Rp, Q)
        S = np.dot(np.linalg.pinv(Q), Sp)
        print('R', R)
        print('S', S)
        Rs = []
        Ts = []
        #recover R and T
        for i in range(len(imgs)):
            idx = 2*i
            curr_r = R[idx]
            curr_mult = np.matrix([[avgus[i]], [avgvs[i]], [1]])
            curr_t = curr_r * curr_mult
            print(curr_t)

        xs = S[0,:].tolist()[0]
        ys = S[1,:].tolist()[0]
        zs = S[2,:].tolist()[0]
        f = open('xyzmeshlab.asc', 'w')
        for x, y, z in zip(xs, ys, zs):
            #print(x, y, z)
            f.write("%f %f %f\n" % (x, y, z))





    def __init__(self):
        pass