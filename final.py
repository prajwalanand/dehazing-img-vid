# -*- coding:utf-8 -*-
import cv2
import numpy as np
import scipy
import scipy.ndimage

from GuidedFilter import GuidedFilter


no_of_threads = 4


def calDepthMap(I, r):

    hsvI = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    s = hsvI[:,:,1] / 255.0
    v = hsvI[:,:,2] / 255.0

    sigma = 0.041337
    sigmaMat = np.random.normal(0, sigma, (I.shape[0], I.shape[1]))

    output =  0.121779 + 0.959710 * v - 0.780245 * s + sigmaMat
    outputPixel = output
    output = scipy.ndimage.filters.minimum_filter(output,(r,r))
    outputRegion = output
    
    return outputRegion, outputPixel

def estA(img, Jdark):


    h,w,c = img.shape
    if img.dtype == np.uint8:
        img = np.float32(img) / 255
    
    n_bright = int(np.ceil(0.001*h*w))

    reshaped_Jdark = Jdark.reshape(1,-1)
    #Y = np.sort(reshaped_Jdark) 
    Loc = np.argsort(reshaped_Jdark)
    
    Ics = img.reshape(1, h*w, 3)
    ix = img.copy()
    #dx = Jdark.reshape(1,-1)
    
    Acand = np.zeros((1, n_bright, 3), dtype=np.float32)
    Amag = np.zeros((1, n_bright, 1), dtype=np.float32)
    
    for i in range(n_bright):
        x = Loc[0,h*w-1-i]
        ix[x//w, x%w, 0] = 0
        ix[x//w, x%w, 1] = 0
        ix[x//w, x%w, 2] = 1
        
        Acand[0, i, :] = Ics[0, Loc[0, h*w-1-i], :]
        Amag[0, i] = np.linalg.norm(Acand[0,i,:])
    
    reshaped_Amag = Amag.reshape(1,-1)
    Y2 = np.sort(reshaped_Amag) 
    Loc2 = np.argsort(reshaped_Amag)

    if len(Y2) > 20:
        A = Acand[0, Loc2[0, n_bright-19:n_bright],:]
    else:
        A = Acand[0, Loc2[0,n_bright-len(Y2):n_bright],:]
    
    print(A)
    
    return A

def sceneRecovery(I, a, dR, refineDR):
    tR = np.exp(-beta * refineDR)
    
    h,w,c = I.shape
    J = np.zeros((h, w, c), dtype=np.float32)
    
    J[:,:,0] = I[:,:,0] - a[0,0]
    J[:,:,1] = I[:,:,1] - a[0,1]
    J[:,:,2] = I[:,:,2] - a[0,2]

    t = tR
    t0, t1 = 0.05, 1
    t = t.clip(t0, t1)

    J[:, :, 0] = J[:, :, 0]  / t
    J[:, :, 1] = J[:, :, 1]  / t
    J[:, :, 2] = J[:, :, 2]  / t

    J[:, :, 0] = J[:, :, 0]  + a[0, 0]
    J[:, :, 1] = J[:, :, 1]  + a[0, 1]
    J[:, :, 2] = J[:, :, 2]  + a[0, 2]
    
    return J

if __name__ == "__main__":
    
    r = 15
    beta = 1.3
    gimfiltR = 60
    eps = 10**-3
    
    vidcap = cv2.VideoCapture('data/video.mp4')
    vidcap.set(cv2.CAP_PROP_FPS, 30)

    while vidcap.isOpened():
        if True:
            success, I = vidcap.read()

            dR,dP = calDepthMap(I, r)
            
            guided_filter = GuidedFilter(I, gimfiltR, eps, no_of_threads)
            refineDR = guided_filter.filter(dR)
        
            a = estA(I, refineDR)

            if I.dtype == np.uint8:
                I = np.float32(I) / 255

            J = sceneRecovery(I, a, dR, refineDR)

            collage = np.concatenate((I,J), axis=1)
            imS = cv2.resize(collage, (1460, 640))
            cv2.imshow("collage", imS)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
