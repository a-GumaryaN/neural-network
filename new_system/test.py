import numpy as np

from convolutional import Convolution
from neural_network import  Neural_network



conv1 = Convolution(filters=1,kernel_size=3,activation="relu",mode="valid")
kernels=np.array([
    [[-1,0,1],
    [-10,0,10],
    [-1,0,1]],
    [[-1,-10,-1],
    [0,0,0],
    [1,10,1]]
    ])
conv1.kernels=kernels

from PIL import Image

im = Image.open('test.jpeg').convert('L')
# im=np.array([im])
im=np.array(im)
im_row,im_col=im.shape
conv1.init_params(1,(im_row,im_col,1))

# print(im.size)
# print(pix[10,10])

import matplotlib.image

out1=conv1.convolution2d(im,kernels[0])
out2=conv1.convolution2d(im,kernels[1])
out3= np.minimum(out1,out2)

matplotlib.image.imsave('name1.png', out1)
matplotlib.image.imsave('name2.png', out2)
matplotlib.image.imsave('name3.png', out3)