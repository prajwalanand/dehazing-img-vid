# -*- coding:utf-8 -*-
import cv2
import numpy as np
import scipy
import scipy.ndimage
import sys
import threading
import pylab
from random import randint
#import matplotlib.pyplot as plt
val = 0
class MyThread(threading.Thread):
    def __init__(self, ID, func, args):
        threading.Thread.__init__(self)
        self.ID = ID
        self.func = func
        self.args = args
    def run(self):
        self.func(self.args)

class GuidedFilter:
    
    def __init__(self, I, radius=60, epsilon=10**-3):

        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = self._toFloatImg(I)
        
        self._initFilter()

    def _toFloatImg(self, img):
        if img.dtype == np.float32:
            return img
        return ( 1.0 / 255.0 ) * np.float32(img)

    def _initFilter(self):
        I = self._I
        r = self._radius
        eps = self._epsilon

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        self._Ir_mean = cv2.blur(Ir, (r, r))
        self._Ig_mean = cv2.blur(Ig, (r, r))
        self._Ib_mean = cv2.blur(Ib, (r, r))

        Irr_var = cv2.blur(Ir ** 2, (r, r)) - self._Ir_mean ** 2 + eps                                       
        Irg_var = cv2.blur(Ir * Ig, (r, r)) - self._Ir_mean * self._Ig_mean                                  
        Irb_var = cv2.blur(Ir * Ib, (r, r)) - self._Ir_mean * self._Ib_mean                                  
        Igg_var = cv2.blur(Ig * Ig, (r, r)) - self._Ig_mean * self._Ig_mean + eps                            
        Igb_var = cv2.blur(Ig * Ib, (r, r)) - self._Ig_mean * self._Ib_mean                                  
        Ibb_var = cv2.blur(Ib * Ib, (r, r)) - self._Ib_mean * self._Ib_mean + eps                                                                                     


        self._Ir_mean = cv2.blur(Ir, (r, r))
        self._Ig_mean = cv2.blur(Ig, (r, r))
        self._Ib_mean = cv2.blur(Ib, (r, r))

        Irr_var = cv2.blur(Ir ** 2, (r, r)) - self._Ir_mean ** 2 + eps                                       
        Irg_var = cv2.blur(Ir * Ig, (r, r)) - self._Ir_mean * self._Ig_mean                                  
        Irb_var = cv2.blur(Ir * Ib, (r, r)) - self._Ir_mean * self._Ib_mean                                  
        Igg_var = cv2.blur(Ig * Ig, (r, r)) - self._Ig_mean * self._Ig_mean + eps                            
        Igb_var = cv2.blur(Ig * Ib, (r, r)) - self._Ig_mean * self._Ib_mean                                  
        Ibb_var = cv2.blur(Ib * Ib, (r, r)) - self._Ib_mean * self._Ib_mean + eps                                                       


        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var                                                      
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var                                                      
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var                                                      
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var                                                      
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var                                                      
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var                                                      
        
        I_cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var                                    
        Irr_inv /= I_cov                                                                                     
        Irg_inv /= I_cov                                                                                     
        Irb_inv /= I_cov                                                                                     
        Igg_inv /= I_cov                                                                                     
        Igb_inv /= I_cov                                                                                     
        Ibb_inv /= I_cov                                                                                     
        
        self._Irr_inv = Irr_inv                                                                              
        self._Irg_inv = Irg_inv                                                                              
        self._Irb_inv = Irb_inv                                                                              
        self._Igg_inv = Igg_inv                                                                              
        self._Igb_inv = Igb_inv                                                                              
        self._Ibb_inv = Ibb_inv                  

    def _computeCoefficients(self, p):
        r = self._radius                                                             
        I = self._I                                                                 
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]                                                          
        

        p_mean = cv2.blur(p, (r, r))                             
        Ipr_mean = cv2.blur(Ir * p, (r, r))                                                         
        Ipg_mean = cv2.blur(Ig * p, (r, r))                                                    
        Ipb_mean = cv2.blur(Ib * p, (r, r))             



        Ipr_cov = Ipr_mean - self._Ir_mean * p_mean                                                 
        Ipg_cov = Ipg_mean - self._Ig_mean * p_mean                                                     
        Ipb_cov = Ipb_mean - self._Ib_mean * p_mean                                                       
                                                                                                                 
        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov                 
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov                
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov    

        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean                                                                                                                                         

        ar_mean = cv2.blur(ar, (r, r))          
        ag_mean = cv2.blur(ag, (r, r))                                                                   
        ab_mean = cv2.blur(ab, (r, r))                                                                      
        b_mean = cv2.blur(b, (r, r))                                                                                                                                              

        return ar_mean, ag_mean, ab_mean, b_mean            

    def _computeOutput(self, ab, I):
    
        ar_mean, ag_mean, ab_mean, b_mean = ab
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]
        q = ar_mean * Ir + ag_mean * Ig + ab_mean * Ib + b_mean
        return q

    def filter(self, p):
        #p_32F = self._toFloatImg(p)
        ab = self._computeCoefficients(p)
        
        #print("Before estA")
        th = threading.Thread(target=estA, args=[self._I, self._computeOutput(ab, self._I),])
        th.start()
        th.join(timeout=5)

def calDepthMap(I, r=15):
    #print("In calDepthMap")
    hsvI = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    s = hsvI[:,:,1] / 255.0
    v = hsvI[:,:,2] / 255.0

    sigma = 0.041337
    sigmaMat = np.random.normal(0, sigma, (I.shape[0], I.shape[1]))

    output =  0.121779 + 0.959710 * v - 0.780245 * s + sigmaMat

    output = scipy.ndimage.filters.minimum_filter(output,(r,r))
    outputRegion = output
    
    #print("Before guided filter")
    th = threading.Thread(target=GuidedFilter(I).filter, args=[outputRegion,])
    th.start()
    th.join(timeout=5)
    
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
    
    #print("Before sceneRecovery")
    th = threading.Thread(target=sceneRecovery, args=[img, A, Jdark,])
    th.start()
    th.join(timeout=5)
    
def sceneRecovery(I, a, refineDR):
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
    
    if I.dtype == np.uint8:
        I = np.float32(I) / 255
    
    collage = np.concatenate((I,J), axis=1)
    imS = cv2.resize(collage, (1460, 640))
    global val
    val = val + 1
    cv2.imwrite("data/outfiles/frames"+val+".png", imS*255)
    #pylab.ion()
    #pylab.imshow(imS)
    #pylab.draw()
    '''try:
        cv2.imshow("output", imS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)
    except:
        pass'''

if __name__ == "__main__":
    beta = 1.3
    
    #th1 = threading.Thread(target=printImg)
    #th1.start()
    
    vidcap = cv2.VideoCapture('data/video.mp4')
    print(vidcap.set(cv2.CAP_PROP_FPS, 1))

    while vidcap.isOpened():
        if randint(1, 100) == 1:
            success, I = vidcap.read()

            th = threading.Thread(target=calDepthMap, args=[I,])
            th.start()
            th.join(timeout=5)
