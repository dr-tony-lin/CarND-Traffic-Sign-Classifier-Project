import cv2
import numpy as np
import matplotlib.pyplot as plt

def randomize(image, scale):
    '''
    Randomize the image

    Arguments:
    image: the image
    scale: the scale of randomization
    '''
    noise = cv2.randu(np.empty(image.shape, dtype=np.float32), 0, scale)
    return noise + image

def normalize(img):
    '''
    Normalize the grayscale image

    Arguments:
    a: the grayscale image to normalize
    Return: the normalized grayscale image
    '''
    if img.dtype != np.float32:
        img = np.array(img, dtype=np.float32)
    low = np.amin(img, axis=(0,1))
    high = np.amax(img, axis=(0,1))
    mid = (high + low) * 0.5
    dis = (high - low + 0.1) * 0.5  # +0.1 in case min = max
    return (img - mid) / dis

def grayscale(img):
    '''
    Convert the image to grayscale

    Arguments:
    img: the image to convert to grayscale
    Return: the converted grayscale image

    '''
    return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

def normal_gray(images, randomize_scale=None):
    '''
    Convert the images into normalized grayscale images for processing

    Arguments:
    images: an array of RGB images
    Returns: an array of normalized grayscale images
    '''
    if randomize_scale is not None:
        return np.array([normalize(randomize(grayscale(img), randomize_scale)) for img in images], dtype=np.float32)
    else:
        return np.array([normalize(grayscale(img)) for img in images], dtype=np.float32)

def rand_visual(images, rows=4, columns=16, fig=None, indices=None):
    '''
    Randomly show images from images

    Arguments:
        images: the image array
        rows: number of rows to show
        columns: number of columns to show
        grey: True if the images are greyscale
        fig: the matplotlib figure to show the images
    '''
    nTypes = len(images)
    nImages = len(images[1])
    if fig is None:
        fig = plt.figure()
        fig.set_size_inches(columns, rows*nTypes)

    for i in range(rows):
        if indices is None:
            rands = np.random.randint(nImages-1, size=columns)
        else:
            rands = indices[i]
        for j in range(nTypes):
            for k in range(min(len(rands), columns)):
                plot_idx = (nTypes*i + j) * columns + k + 1
                plt.subplot(rows*nTypes, columns, plot_idx)
                if len(images[j][rands[k]].shape) < 3:
                    plt.imshow(images[j][rands[k]], cmap='gray')
                else:
                    plt.imshow(images[j][rands[k]])
    return plt
