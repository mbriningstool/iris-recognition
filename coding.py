#################################################################
#iris-recognition original code base was created by
#github user mokosaur https://github.com/mokosaur/iris-recognition
#code was forked from original on 4-4-2019
################################################################

import math
import numpy as np

from skimage.util import view_as_blocks

def polar2cart( radius , xCoordinate , yCoordinate , polarAngle ):
    """Changes polar coordinates to cartesian coordinate system.

    :param xCoordinate: x coordinate of the origin
    :param yCoordinate: y coordinate of the origin
    :param theta: Angle
    :return: Cartesian coordinates
    :rtype: tuple (int, int)
    """
    xInCartesian = int( xCoordinate + radius * math.cos( polarAngle ) )
    yInCartesian = int( yCoordinate + radius * math.sin( polarAngle ) )

    return xInCartesian , yInCartesian


def unravel_iris(imageOfEye, centerOfPupilXCoordinate, centerOfPupilYPupilCoordinate, pupilRadius, xi, yi, ri, phase_width=300, iris_width=150):
    """Unravels the iris from the image and transforms it into a rectangular representation.

    :param imageOfEye: Image of an eye
    :param centerOfPupilXCoordinate: x coordinate of the pupil centre
    :param centerOfPupilYPupilCoordinate: y coordinate of the pupil centre
    :param pupilRadius: Radius of the pupil
    :param xi: x coordinate of the iris centre
    :param yi: y coordinate of the iris centre
    :param ri: Radius of the iris
    :param phase_width: Length of the transformed iris
    :param iris_width: Width of the transformed iris
    :return: Straightened image of the iris
    :rtype: ndarray
    """
    if imageOfEye.ndim > 2:
        imageOfEye = imageOfEye[:, :, 0].copy()

    iris = np.zeros((iris_width, phase_width))

    theta = np.linspace(0, 2 * np.pi, phase_width)
    
    for i in range(phase_width):
        begin = polar2cart(pupilRadius, centerOfPupilXCoordinate, centerOfPupilYPupilCoordinate, theta[i])
        end = polar2cart(ri, xi, yi, theta[i])
        xspace = np.linspace(begin[0], end[0], iris_width)
        yspace = np.linspace(begin[1], end[1], iris_width)
        iris[:, i] = [255 - imageOfEye[int(y), int(x)]
                      if 0 <= int(x) < imageOfEye.shape[1] and 0 <= int(y) < imageOfEye.shape[0]
                      else 0
                      for x, y in zip(xspace, yspace)]
    return iris


def gabor(rho, phi, w, theta0, r0, alpha, beta):
    """Calculates gabor wavelet.

    :param rho: Radius of the input coordinates
    :param phi: Angle of the input coordinates
    :param w: Gabor wavelet parameter (see the formula)
    :param theta0: Gabor wavelet parameter (see the formula)
    :param r0: Gabor wavelet parameter (see the formula)
    :param alpha: Gabor wavelet parameter (see the formula)
    :param beta: Gabor wavelet parameter (see the formula)
    :return: Gabor wavelet value at (rho, phi)
    """
    return np.exp(-w * 1j * (theta0 - phi)) * np.exp(-(rho - r0) ** 2 / alpha ** 2) * \
           np.exp(-(phi - theta0) ** 2 / beta ** 2)


def gabor_convolve(imageOfEye, w, alpha, beta):
    """Uses gabor wavelets to extract iris features.

    :param imageOfEye: Image of an iris
    :param w: w parameter of Gabor wavelets
    :param alpha: alpha parameter of Gabor wavelets
    :param beta: beta parameter of Gabor wavelets
    :return: Transformed image of the iris (real and imaginary)
    :rtype: tuple (ndarray, ndarray)
    """
    rho = np.array([np.linspace(0, 1, imageOfEye.shape[0]) for i in range(imageOfEye.shape[1])]).T
    x = np.linspace(0, 1, imageOfEye.shape[0])
    y = np.linspace(-np.pi, np.pi, imageOfEye.shape[1])
    xx, yy = np.meshgrid(x, y)
    return rho * imageOfEye * np.real(gabor(xx, yy, w, 0, 0.5, alpha, beta).T), \
           rho * imageOfEye * np.imag(gabor(xx, yy, w, 0, 0.5, alpha, beta).T)


def iris_encode(imageOfEye, dr=15, dtheta=15, alpha=0.4):
    """Encodes the straightened representation of an iris with gabor wavelets.

    :param imageOfEye: Image of an iris
    :param dr: Width of image patches producing one feature
    :param dtheta: Length of image patches producing one feature
    :param alpha: Gabor wavelets modifier (beta parameter of Gabor wavelets becomes inverse of this number)
    :return: Iris code and its mask
    :rtype: tuple (ndarray, ndarray)
    """
    # mean = np.mean(imageOfEye)
    # std = imageOfEye.std()
    mask = view_as_blocks(np.logical_and(100 < imageOfEye, imageOfEye < 230), (dr, dtheta))
    norm_iris = (imageOfEye - imageOfEye.mean()) / imageOfEye.std()
    patches = view_as_blocks(norm_iris, (dr, dtheta))
    code = np.zeros((patches.shape[0] * 3, patches.shape[1] * 2))
    code_mask = np.zeros((patches.shape[0] * 3, patches.shape[1] * 2))
    for i, row in enumerate(patches):
        for j, p in enumerate(row):
            for k, w in enumerate([8, 16, 32]):
                wavelet = gabor_convolve(p, w, alpha, 1 / alpha)
                code[3 * i + k, 2 * j] = np.sum(wavelet[0])
                code[3 * i + k, 2 * j + 1] = np.sum(wavelet[1])
                code_mask[3 * i + k, 2 * j] = code_mask[3 * i + k, 2 * j + 1] = \
                    1 if mask[i, j].sum() > dr * dtheta * 3 / 4 else 0
    code[code >= 0] = 1
    code[code < 0] = 0
    return code, code_mask


if __name__ == '__main__':
    import cv2
    from datasets import load_utiris
    import matplotlib.pyplot as plt

    data = load_utiris()['data']
    image = cv2.imread(data[0])

    iris = unravel_iris(image, 444, 334, 66, 450, 352, 245)
    code, mask = iris_encode(iris)

    plt.subplot(211)
    plt.imshow(iris, cmap=plt.cm.gray)
    plt.subplot(223)
    plt.imshow(code, cmap=plt.cm.gray, interpolation='none')
    plt.subplot(224)
    plt.imshow(mask, cmap=plt.cm.gray, interpolation='none')
    plt.show()
