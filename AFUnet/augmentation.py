import scipy
import nibabel as nib
import numpy as np
from skimage.transform import resize


def aug(volume, angle, axis):
    H, W, D = volume.shape
    if axis == (0, 1):
        test = scipy.ndimage.rotate(volume[:, :, 0], angle=angle)
        H1, W1 = test.shape
        out = np.zeros((H1, W1, D))
        for i in range(D):
            rotated = scipy.ndimage.rotate(volume[:, :, i], angle=angle)
            out[:, :, i] = rotated
    elif axis == (0, 2):
        test = scipy.ndimage.rotate(volume[:, 0, :], angle=angle)
        H1, D1 = test.shape
        out = np.zeros((H1, W, D1))
        for i in range(W):
            rotated = scipy.ndimage.rotate(volume[:, i, :], angle=angle)
            out[:, i, :] = rotated
    else:
        test = scipy.ndimage.rotate(volume[0, :, :], angle=angle)
        W1, D1 = test.shape
        out = np.zeros((H, W1, D1))
        for i in range(H):
            rotated = scipy.ndimage.rotate(volume[i, :, :], angle=angle)
            out[i, :, :] = rotated
    
    out = resize(out, (240, 240, 155))
    return out

