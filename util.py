import cv2
import datetime
import io
import math
import numpy
import pylab
import PIL
import scipy
import time
import urllib2
from pylab import *
import scipy.signal

# Read an image from a url and display it
# http://stackoverflow.com/a/12020860/1808109
def show_url(url):
    fd = urllib2.urlopen(url)
    image_file = io.BytesIO(fd.read())
    image = PIL.Image.open(image_file)
    pylab.imshow(array(image))
    return image
# Appears to return UTC time
def ts(*args):
    now = datetime.datetime.now()
    ts = datetime.datetime.strftime(now, "%Y%m%d %H:%M:%S.%f")
    return "[{0}] {1}".format(ts, " ".join(str(a) for a in args))

# This reads all the frames of tennis.mp4 info a numpy array.
def read_video_file(filename, max_frame=500):
    # setup video capture
    cap = cv2.VideoCapture(filename)
    
    frames = []
    # get frame, store in array
    for ii in xrange(max_frame):
        #print ts("Frame:", ii)
        ret, im = cap.read()
        if not ret:
            break
        frames.append(im)
    cap.release()
    print ts("Converting to np array")
    frames = array(frames)
    
    # check the sizes
    print frames[-1].shape
    print frames.shape
    print ts("Done")

    return frames


def run(cmd):
    import subprocess
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.communicate()

def commit(file=NOTEBOOK_FILE_NAME, msg="Periodic Update"):
    print run(["git", "commit", "-m", msg, file])
    # Pushes to https://github.com/wcyuan/opencv_ipynotebook
    print run(["git", "push"])

# https://github.com/jesolem/PCV/blob/master/pcv_book/imtools.py

def histeq(im,nbr_bins=256):
    """ Histogram equalization of a grayscale image. """
    # get image histogram
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf

# https://github.com/jesolem/PCV/blob/master/pcv_book/pca.py

def pca(X):
    """    Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean.
    """
    
    # get dimensions
    num_data,dim = X.shape
    
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    
    if dim>num_data:
        # PCA - compact trick used
        M = dot(X,X.T) # covariance matrix
        e,EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
    
    # return the projection matrix, the variance and the mean
    return V,S,mean_X


def center(X):
    """    Center the square matrix X (subtract col and row means). """
    
    n,m = X.shape
    if n != m:
        raise Exception('Matrix is not square.')
    
    colsum = X.sum(axis=0) / n
    rowsum = X.sum(axis=1) / n
    totalsum = X.sum() / (n**2)
    
    #center
    Y = array([[ X[i,j]-rowsum[i]-colsum[j]+totalsum for i in range(n) ] for j in range(n)])
    
    return Y

# Murali's formula to calculate initial velocity, from 
# http://donthireddy.us/tennis/speed.html

import math
# published numbers  from a now dead link
KNOWN_SERVE_SPEED= 120.
KNOWN_SPEED_AT_BOUNCE= 87.0
DISTANCE_TRAVELED=60.

# A constant for formula:
K= math.log(120./87)/60

def init_speed(nf, d=60, fps=29.97):
    speed = ( math.exp(K*d) - 1) / (5280.*K * nf) * fps * 3600
    return round(speed*100)/100

def avg_speed(nf, d=60, fps=29.97):
    speed = d / (nf / fps ) * (3600./5280.)
    return round(speed*100)/100

# Create a single grayscale image where for each pixel you figure out the
# median brightness in the entire video for that pixel.  Then in your
# image, your pixel color is the brightness that is most different
# from the median.

# Increase contrast to make the non-zero pixels show up stronger.
contrast = 5

def diff_median(frames, start, end):
    # convert a portion of the video to grayscale
    gray = array([cv2.cvtColor(frames[ii], cv2.COLOR_RGB2GRAY)
                  for ii in xrange(start, end)])
    # Find medians across the video segment
    medians = numpy.median(gray, 0)
    # take differences
    diffs = gray - medians
    # for each pixel, keep only the greatest difference from the median
    abses = numpy.abs(diffs)
    maxes = (1.0-((1.0-numpy.max(abses, 0)/numpy.max(abses))**contrast))
    pylab.imshow(maxes)
    

 
def diff_filtered_median(frames, start, end, clean = False, contrast = 5):
    # convert a portion of the video to grayscale
    gray = array([cv2.cvtColor(frames[ii], cv2.COLOR_RGB2GRAY)
                  for ii in xrange(start, end)])
    # Find medians across the video segment
    medians = numpy.median(gray, 0)
    
    # Figure out where noise from camera movement happens.
    # This is a hack.  The court lines are different from their neighbors.
    fmeds = numpy.abs(medians-scipy.signal.medfilt2d(medians,21))
    
    # Need to "smooth" to push out the lines.
    fmeds = scipy.signal.convolve2d(fmeds, numpy.ones([3,3])/9, mode="same")
    
    # Convert to filter (mark as 1 non line pixels and 0 for line pixels)
    level = 10.0
    fmeds[fmeds<level] = 0.0
    fmeds[fmeds>=level] = 1.0
    
    
    # Take differences
    diffs = gray - medians
    # For each pixel, keep only the greatest difference from the median
    abses = numpy.abs(diffs)
    maxes = (1.0-((1.0-numpy.max(abses, 0)/numpy.max(abses))**contrast))*(1-fmeds)
    
    if clean:
    # Just the top points.
        maxes[maxes<0.3] = 0.0
    
    #pylab.imshow(medians)
    #pylab.figure()
    pylab.imshow(maxes)

