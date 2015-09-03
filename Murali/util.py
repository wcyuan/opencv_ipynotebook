#!/bin/env python
# Stuff from the book
# http://programmingcomputervision.com/downloads/ProgrammingComputerVision_CCdraft.pdf
#
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image
def imresize(im, sz):
    """ Resize an image array using PIL. """
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))

# Histogram Equalization makes all brightness levels equally likely
# Supposedly increases contrast
def histeq(im,nbr_bins=256):
    """ Histogram equalization of a grayscale image. """
    # get image histogram
    imhist,bins = sp.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf

