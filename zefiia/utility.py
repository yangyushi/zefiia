import numpy as np
from numba import njit
from scipy import ndimage, stats
from scipy.optimize import minimize


def ransac(features, degree, sigma, N, T=31, alpha=0.95):
    """
    Use RANSAC algorithm to estimate a polynomial fit
        for a set of 2D points

    RANSAC algorith:
        1. randomly select [S] data points to initialise the mode
        2. Determine the set of data points [Si] within a distance
            thresold t of the model. [Si] is called the "consensus set"
        3. If the size of [Si] is less than [T], select a new [S] and
            repeat 2
        4. If the size of [Si] is greater than [T], estimate model and
            terminate
        5. After [N] trials, report the model with the largest [Si]

    See Multiple View Geometry in Computer Vision Chapter 4.7

    TODO: use analytical solution to fit the polynomial

    Args:
        features (np.ndarray): shape (n, 2)
        degree (int): the degree of the polynomial fit
        sigma (float): the sigma of the gaussian distribution of the error
        N (int): number of repeats
        T (int): the targeted support size
        alpha (float): the confidence level of the chi2 distribution
    """
    nf = features.shape[0]
    t2 = stats.chi2.ppf(alpha, df=1) * sigma**2
    Si_ensemble = []
    for trail_idx in tqdm(range(N)):
        Si = []
        while len(Si) < T:
            S = features[np.random.randint(0, nf, degree)]
            par = np.polyfit(*S.T, degree)
            dists = [
                minimize(
                    fun=lambda a, x, y: (a - x)**2 + (y - np.polyval(par, x))**2,
                    x0=0,
                    args=(f[0], f[1])
                ).fun for f in features
            ]
            Si = features[dists < t2]
        Si_ensemble.append(Si)
    Si_counts = [len(s) for s in Si_ensemble]
    return Si_ensemble[np.argmax(Si_counts)


@njit
def draw_disk(canvas, r, c, val=1):
    """
    Draw a disk inside a canvas

    Args:
        canvas (np.ndarray): a 2D array whose value will be modified in-place
        r (float): the radius of the disk to be drawn
        c (iterable): the centre of the disk
        val (float or int): the value of the pixels belonging to the disk

    Return:
        None

    Example:
        >>> canvas = np.zeros((5, 5))
        >>> draw_disk(canvas, r=1.2, c=(2, 2), val=-1)
        >>> canvas.sum()
        -5.0
    """
    r2 = r * r
    for row in range(canvas.shape[0]):
        for col in range(canvas.shape[1]):
            if (row - c[0])**2 + (col - c[0])**2 < r2:
                canvas[row, col] = val


def get_local_feature(image, xy, win_size):
    """
    Calculate statistical descriptors of the local intensity distribution

    Args:
        xy (np.array): the central location to perform the analysis, shape (2,)
        win_size (int): the window size of the local area

    Return:
        tuple: the mean, std, skewness and kurtosis of the local
            intensity distribution

    Example:
        >>> image = np.zeros((40, 40))
        >>> positions = np.array((20, 20))
        >>> feature = get_local_feature(image, xy=positions, win_size=5)
        >>> len(feature)
        4
        >>> feature[0]  # mean of blank image is 0.0
        0.0
    """
    x, y = xy.astype(int)
    half = win_size // 2
    residual = win_size % 2
    box = (
        slice(x - half - residual, x + half, 1),
        slice(y - half - residual, y + half, 1)
    )
    sub_image = image[box]
    stat_result = stats.describe(sub_image.ravel())[2:]
    return stat_result


def profile_poly(image, src, dst, parameters, linewidth=1, method=np.mean):
    """
    Return the intensity profile of an image measure alogn a
        polymonial curve. The code is inspired by the linear
        equivalent from package scikit-image.

    The polynomial is measured in the X-Y coordinates of the image.
        (not the row-column order)

    Args:
        image (np.ndarray): a 2d image shape (x, y).
        src (array_like): the start point (x, y) of the curve, shape (2, ).
        dst (array_like): the end point (x, y) of the curve, shape (2, ).
        parameters (array_like): the coefficient of the polynomial curve.
            Its length is used as the order of the polynomial fit
        linewidth (int): the width of the scan
        method (callable): the function to operate on the intensity profile
            whose shape is (length, linewidth)

    Return:
        return of the method: the result method on the measured profile
    """
    order = len(parameters) - 1
    par_der = [  # parameters of the derivative
        (order - i) * p for i, p in enumerate(parameters[:-1])
    ]

    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    d_x, d_y = dst - src
    length = int(np.ceil(np.hypot(d_x, d_y) + 1))

    line_x = np.linspace(src[0], dst[0], length)
    line_y = np.polyval(parameters, line_x)

    theta = np.arctan(np.polyval(par_der, line_x))

    width_x = (linewidth - 1) * np.sin(theta) / 2
    width_y = - (linewidth - 1) * np.cos(theta) / 2

    prep_x = np.stack([
        np.linspace(x - wx, x + wx, linewidth) for x, wx in zip(line_x, width_x)
    ])  # (length, linewidth)

    prep_y = np.stack([
        np.linspace(y - wy, y + wy, linewidth) for y, wy in zip(line_y, width_y)
    ])  # (length, linewidth)

    prep_lines = np.stack([prep_x, prep_y])

    # shape (n_pixels, width)
    pixels = ndimage.map_coordinates(image.T, prep_lines, order=2)

    return method(pixels, axis=1)


def measure_asymmetry(pixels, axis):
    """
    measure intensity symmetry along axis

    Args:
        pixels (np.array): shape (length, width)
    """
    l, w = pixels.shape
    c = w // 2 - w % 2 + 1
    left = pixels[:, :c]
    right = pixels[:, c+1:]
    diff = left - right[:, ::-1]
    return np.abs(diff.mean(axis)).mean() - pixels.mean()


def get_feature_cost(image, features, size, deg=4, axis=0):
    """
    Args:
        image (np.array): image
        features (np.array): 2, n
        size (int): the width of the profile
        deg (int): for poly
        axis (int): along axis which to measure
    """
    poly_par = np.polyfit(*features, deg=deg)
    src_idx = np.argmin(features[axis])
    dst_idx = np.argmax(features[axis])
    cost = profile_poly(
        image, features[:, src_idx], features[:, dst_idx],
        poly_par,
        linewidth=size - size % 2 + 1,
        method=measure_symmetry
    )
    return cost


def get_feature_profile(image, features, size, deg=4, axis=0):
    """
    Args:
        image (np.array): image
        features (np.array): 2, n
        size (int): the width of the profile
        deg (int): for poly
        axis (int): along axis which to measure
    """
    poly_par = np.polyfit(*features, deg=deg)
    src_idx = np.argmin(features[axis])
    dst_idx = np.argmax(features[axis])
    profile = profile_poly(
        image, features[:, src_idx], features[:, dst_idx],
        poly_par,
        linewidth=size - size % 2 + 1,
        method=lambda x, axis: x
    )
    return profile


def get_poly_profile(image, src, dst, poly_par, size):
    """
    Args:
        image (np.array): image
        size (int): the width of the profile
        axis (int): along axis which to measure
    """
    profile = profile_poly(
        image, src, dst, poly_par,
        linewidth=size - size % 2 + 1,
        method=lambda x, axis: x
    )
    return profile
