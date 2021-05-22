import matplotlib.pyplot as plt
import numpy as np
import cv2 
from skimage import data,color
img=(data.camera())
an1 = 45+   45
an2 = 45-   45
m1 = np.array ([0.2,0.2,0.2,0.2,0.2])
def Conv(image,mi,angle):
    K = mi.size
    H,W = image.shape
    Cos = np.cos(angle)
    Sin = np.sin(angle)
    out = np.zeros((H,W))
    for k in range(K):
        H=(H-k*Cos).astype(int)
        W=(W-k*Sin).astype(int)
        for i in range(H):
            for j in range(W):
                I = (i+k*Cos).astype(int)
                J = (j+k*Sin).astype(int)
                out[i][j] = out[i][j]+mi[k]*image[I][J]
    return out    
def Q(image, degree, angle):
  image = np.array(image)
  M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
  motion_blur_kernel = np.diag(np.ones(degree))
  motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
  motion_blur_kernel = motion_blur_kernel / degree
  blurred = cv2.filter2D(image, -1, motion_blur_kernel)
  blurred = np.array(blurred, dtype=np.uint8)
  return blurred
def De(image,p=1.25):
    h,w = image.shape
    fx = np.zeros(image.shape, np.uint8)
    fy = np.zeros(image.shape, np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            sx = (image[i + 1][j - 1] + image[i + 1][j] + image[i + 1][j + 1]) - \
                 (image[i - 1][j - 1] + image[i - 1][j] + image[i - 1][j + 1])
            sy = (image[i - 1][j + 1] + image[i][j + 1] + image[i + 1][j + 1]) - \
                 (image[i - 1][j - 1] + image[i][j - 1] + image[i + 1][j - 1])
            fx[i][j] = p*np.power(sx,p-1)*np.sign(sx)
            fy[i][j] = p*np.power(sy,p-1)*np.sign(sy)
    return fx,fy
def Update(image1,image2,m1,m2,a1,a2,L=0.004):
    f = 0.5*image1+0.5*image2
    f = f + 0.5*Conv((image1-Conv(f,m1,a1)),m1,a1)+0.5*Conv((image2-Conv(f,m2,a2)),m2,a2)
    fx,fy = De(f)
    _,fx = De(fx)
    fy,_ = De(fy)
    f = f - L*(fx+fy)
    return f
def Error1(image1,image2,K1,K2,a1,a2):
    H,W = image1.shape
    H=(H-K2).astype(int)
    W=(W-K2).astype(int)
    out = np.zeros(K2)
    c1 = np.cos(a1)
    s1 = np.sin(a1)
    c2 = np.cos(a2)
    s2 = np.sin(a2)
    for i in range(H):
            for j in range(W): 
                for k in range(K2):
                    out[k] = out[k] + (image2[i+k*c1][j+k*s1]-image2[i+k*c2][j+k*s2])
    return out
def Error2(image1,image2,K1,K2,a1,a2):
    H,W = image1.shape
    H=(H-K1).astype(int)
    W=(W-K1).astype(int)
    out = np.zeros(K1)
    c1 = np.cos(a1)
    s1 = np.sin(a1)
    c2 = np.cos(a2)
    s2 = np.sin(a2)
    for i in range(H):
            for j in range(W): 
                for k in range(K1):
                    out[k] = out[k] +  (image1[i+k*c2][j+k*s2]-image1[i+k*c1][j+k*s1])
    return out
def Amp(image1,image2,a1,a2):
    out1 = out2 = -1
    for k1 in range(14,30):
        for k2 in range(14,30):
            O1 = np.zeros(k2)
            O2 = np.zeros(k1)
            O1 = Error1(image1,image2,k1,k2,a1,a2)
            O2 = Error2(image1,image2,k1,k2,a1,a2)
            for i in range(k2):
                if O1[i] == 0:
                    out2 = k2
            for i in range(k1):
                if O2[i] == 0:
                    out1 = k1
    return out1,out2
def Direct(image1,image2,m1,m2,amin=-90,amax=90):
    Fuck = 50000
    for a1 in range(amin,amax):
        for a2 in range(amin,amax):
            img1 = Conv(image1,m2,a2)
            img2 = Conv(image2,m1,a1)
            H,W = img1.shape
            for i in range(H):
                for j in range(W):
                    output = img1[i][j]-img2[i][j]
                    output = output*output
                    if output < Fuck:
                        Fuck = output
                        o1=a1
                        o2=a2
    return o1,o2



dst1 = Q(img,50,45)
dst2 = Q(img,50,45)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
axes = axes.ravel()
ax1, ax2= axes
ax1.set_title("Raw Image")
ax1.imshow(img, 'prism')
ax2.set_title("After Processing")
ax2.imshow(dst2, 'gray')
for ax in axes:
    ax.axis('on')
fig.tight_layout()
plt.show()