import copy
import numpy as np
from tqdm import tqdm
from numba import njit
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from joblib import Parallel, delayed


def ransac(
        features, degree, sigma, N, T=31, alpha=0.95,
        max_iter=100, report=False
    ):
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

    Args:
        features (np.ndarray): shape (n, 2)
        degree (int): the degree of the polynomial fit
        sigma (float): the sigma of the gaussian distribution of the error
        N (int): number of repeats
        T (int): the targeted support size
        alpha (float): the confidence level of the chi2 distribution

    Return:
        np.ndarray: features selected by the RANSAC algorithm, the fitting
            result should be estimated using `np.polyfit` with these features
            (the shape is n, 2)
    """
    nf = features.shape[0]
    scale = features.std(axis=0).max()
    t2 = stats.chi2.ppf(alpha, df=1) * (sigma / scale)**2
    Si_ensemble = []
    f_norm = features / scale

    if report:
        to_iter = tqdm(range(N))
    else:
        to_iter = range(N)

    for iter_idx in to_iter:
        Si = []
        count = 0
        while len(Si) < T:
            x, y = f_norm[np.random.randint(0, nf, degree)].T
            # the shape of x_poly is (n_degree, n_sample),
            #     but where n_degree == n_sample
            x_poly = x[np.newaxis, :] ** np.arange(degree + 1)[:, np.newaxis]
            par = np.linalg.pinv(x_poly @ x_poly.T) @ x_poly @ y
            par = par[::-1]
            dists = np.array([
                minimize(
                    fun=lambda a, x, y: (a - x)**2 + (y - np.polyval(par, x))**2,
                    x0=0,
                    args=(f[0], f[1])
                ).fun for f in f_norm
            ])  # revert back to the origional unit
            Si = np.where(dists < t2)[0]
            count += 1
            if count == max_iter:
                break
        Si_ensemble.append(Si)
    Si_counts = [len(s) for s in Si_ensemble]
    chosen = Si_ensemble[np.argmax(Si_counts)]
    return features[chosen]


def ransac_step(N, T, alpha, max_iter, f_norm, nf, t2, degree):
    """
    a single step for the ransac algorithm, exclusively designed for ransac_mp
    """
    Si = []
    count = 0
    while len(Si) < T:
        x, y = f_norm[np.random.randint(0, nf, degree)].T
        # the shape of x_poly is (n_degree, n_sample),
        #     but where n_degree == n_sample
        x_poly = x[np.newaxis, :] ** np.arange(degree + 1)[:, np.newaxis]
        par = np.linalg.pinv(x_poly @ x_poly.T) @ x_poly @ y
        par = par[::-1]
        dists = np.array([
            minimize(
                fun=lambda a, x, y: (a - x)**2 + (y - np.polyval(par, x))**2,
                x0=0,
                args=(f[0], f[1])
            ).fun for f in f_norm
        ])  # revert back to the origional unit
        Si = np.where(dists < t2)[0]
        count += 1
        if count == max_iter:
            break
    return Si


def ransac_mp(
        features, degree, sigma, N, T, n_jobs=4,
        alpha=0.95, max_iter=100, report=False
    ):
    """
    Use RANSAC algorithm to estimate a polynomial fit
        for a set of 2D points using multiple processors

    Args:
        features (np.ndarray): shape (n, 2)
        degree (int): the degree of the polynomial fit
        sigma (float): the sigma of the gaussian distribution of the error
        N (int): number of repeats
        T (int): the targeted support size
        alpha (float): the confidence level of the chi2 distribution

    Return:
        np.ndarray: features selected by the RANSAC algorithm, the fitting
            result should be estimated using `np.polyfit` with these features
            (the shape is n, 2)
    """
    nf = features.shape[0]
    scale = features.std(axis=0).max()
    t2 = stats.chi2.ppf(alpha, df=1) * (sigma / scale)**2
    f_norm = features / scale

    if report:
        to_iter = tqdm(range(N))
    else:
        to_iter = range(N)

    Si_ensemble = Parallel(n_jobs=n_jobs)(
        delayed(ransac_step)(
            N, T, alpha, max_iter, f_norm, nf, t2, degree
        ) for iter_idx in to_iter
    )
    Si_counts = [len(s) for s in Si_ensemble]
    chosen = Si_ensemble[np.argmax(Si_counts)]
    return features[chosen]


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


def get_similar_features(image, features, size):
    """
    Get the features that are similar in terms of intensity stats

    Args:
        image (np.ndarray): a 2D image
        features (np.ndarray): the coordinates of the image, shape (2, n)
        size (int): the window size inwhich pixel intensity distribution will
            be calculated
    """
    intensity_features = np.array([
        get_local_feature(
            image, f, win_size=size
        ) for f in features.T.astype(int)
    ])
    return features[:, get_largest_cluster(intensity_features)]


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
    return (np.abs(diff.mean(axis)) / pixels.mean(axis)).mean()


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
    src = features[:, src_idx]
    dst = features[:, dst_idx]
    cost = profile_poly(
        image, src, dst, poly_par,
        linewidth=size - size % 2 + 1,
        method=measure_asymmetry
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
        image, features[:, src_idx], features[:, dst_idx], poly_par,
        linewidth=size - size % 2 + 1, method=lambda x, axis: x
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


def get_pca_angle(image, threshold, weight=True, max_points=1000):
    """
    Getting the rotate angle to rotate the main feature in the image
        so that its main component axis align with the x-axis

    Args:
        image (np.ndarray): a 2D image
        threshold (float): the percentage threshold value, the pixles whose
            intensity is below.
        weight (bool): if true, weight the pixels in the image according to
            their intensities/brightness.
        max_points (int): the maximum points used to calculate the covariance
            matrix.
    """
    points = np.array(np.where(
        image > image.max() * threshold
    )).astype(float)  # shape (2, n)
    dim, n = points.shape
    if n > max_points:
        indices = np.arange(n)
        np.random.shuffle(indices)
        indices = indices[:max_points]
    else:
        indices = np.arange(n)
    if weight:
        weight = image[image > image.max() * threshold][indices]
    else:
        weight = np.ones(n)
    points = points[:, indices]
    mu = np.mean(points, axis=1)
    points -= mu[:, None]
    cov = points @ np.diag(weight) @ points.T / np.sum(weight)
    u, s, vh = np.linalg.svd(cov)
    return np.arcsin(u[0][0]) / np.pi * 180


def get_largest_cluster(features, eps=1, min_samples=4, pca_plot=True):
    """
    Use the DBSCAN algorithm to find the largest connected cluster
        in the feature space.

    Args:
        features (np.ndarray): features in any dimension, shape (n, dimension)
        eps (float): parameter for the DBSCAN algorithm
        min_samples (int): parameter for the DBSCAN algorithm
        pca_plot (bool): if True, plot the distribution of clusters along the first
            2 princple axes

    Return:
        np.ndarray: the indices of the feature that corresponding
        to the largest cluster
    """
    f = (features - features.mean(0)) / features.std(0)
    db = DBSCAN(eps=1, min_samples=4).fit(f)
    valid_labels = db.labels_[db.labels_ >= 0]
    valid_labels = list(set(db.labels_[db.labels_ >= 0]))
    label_counts = [
        np.sum(db.labels_ == val) for val in valid_labels
    ]
    chosen_label = valid_labels[np.argmax(label_counts)]
    return np.where(db.labels_ == chosen_label)[0]


    # Plot the result on the PCA axes
    if pac_plot:
        cov = f.T @ f
        u, s, vh = np.linalg.svd(cov)
        f_reduced = u[:, :2].T @ f.T
        plt.scatter(*f_reduced, color='teal', facecolor='none')
        for label_val in set(db.labels_):
            if label_val >= 0:
                chosen = f_reduced[:, db.labels_ == label_val]
                plt.scatter(*chosen, marker='+')
                plt.text(
                    *chosen.mean(1), f'#{label_val}',
                    bbox=dict(boxstyle='round', fc="w", ec="k", alpha=0.8)
                )
        plt.xlabel('Principle Axis #1')
        plt.ylabel('Principle Axis #2')
        plt.gcf().set_size_inches(5, 5)
        plt.show()


def remap(scatter, image, src, dst, parameters, linewidth=1):
    """
    Remapping coordinates in the profile image back to the coordinates
        in the origional image

    Args:
        scatter (np.ndarray): the coordinates in the profile image, shape (2, n)
        src (np.ndarray): the start point of the profile, shape (2,)
        dst (np.ndarray): the end point of the profile, shape (2,)
        parameters (np.ndarray): the fitting parameters of the profile, from
            numpy.polyfit
        linewidth (int): the width of the profile image

    Return:
        np.ndarray: the coordinates in the origional image, shape (2, n)
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

    prep_lines = np.stack([prep_x, prep_y])  # (2, length, width)
    remap_x = prep_x[tuple(scatter.astype(int))]
    remap_y = prep_y[tuple(scatter.astype(int))]
    return np.array((remap_x, remap_y))


def get_poly_parameters(features, axis, degree):
    """
    Get the src, dst, and polynormial parameters from features

    Args:
        features (np.ndarray): the features to fit, shape (2, n)
        axis (int): the x-componenet for the fit
        degree (int): the degee of the polynomial fit

    Return:
        tuple: src, dst, and the poly_parameters
    """
    fx = features[axis]
    src = features[:, np.argmin(fx)]
    dst = features[:, np.argmax(fx)]
    poly_par = np.polyfit(*features, deg=degree)
    return src, dst, poly_par


def optimise_spine_features(
        features, image, size, degree, template, sigma, N, T
    ):
    """
    Args:
        image (np.ndarray): a 2D image
        features (np.ndarray): (2, n)
        size (int): the width of the profile
        degree (int): the degree of the polynomial fit
        template (np.ndarray): a small 2D image as template for the vertebrae
        sigma (float): sigma parameter for the RANSAC fit
        N (int): sigma parameter for the RANSAC fit
        T (int): sigma parameter for the RANSAC fit

    Return:
        tuple: refined feature (2, n) and the asymmetry cost
    """
    src, dst, poly_par = get_poly_parameters(features, axis=0, degree=degree)
    profile = get_poly_profile(image, src, dst, poly_par, size=size)
    corr = fftconvolve(profile - profile.mean(), template, mode='same')
    corr[corr < 0] = 0
    maxima = get_maxima(corr, size//4)
    remapped = remap(maxima, image, src, dst, poly_par, linewidth=size)
    optimised = ransac(remapped.T, degree, sigma=sigma, N=N, T=T).T
    cost = get_feature_cost(image, optimised, size=size, deg=degree)
    return optimised, cost


def get_doughnut(size):
    """
    Draw a 2D doughnut shaped template, used as template for the vertebrae.

    Args:
        size (int): the radius of the doughnut.

    Return:
        (np.ndarray): the 2D picture containing the doughnut
    """
    r0 = size // 8
    r1 = size // 2
    doughnut = np.zeros((2 * r1 + 1, 2 * r1 + 1))
    draw_disk(doughnut, r1, c=(r1, r1), val=1)
    draw_disk(doughnut, r0, c=(r1, r1), val=-1)
    return doughnut


def get_circle_template(size):
    """
    Draw a 2D doughnut shaped template, used as template for the vertebrae.

    Args:
        size (int): the radius of the doughnut.

    Return:
        (np.ndarray): the 2D picture containing the doughnut
    """
    r0 = size // 2
    r1 = size // 2 + size // 8
    circle = np.ones((2 * r1 + 1, 2 * r1 + 1)) * -1
    draw_disk(circle, r0, (r1, r1), 1)
    return circle


def get_maxima(image, size, threshold='mean'):
    """
    Find local intensity maxima inside an image. There is no sub-pixel accuracy.

    Args:
        image (np.ndarray): a n-dimensional image
        size (int): the radius of the local region
        threshold (float or str): the maxima whose intensity is smaller
            than the threshold will be discarded.

    Return:
        np.ndarray: the coordinates of local intensity maxima, shape (dimension, n)
    """
    maxima_img = ndimage.maximum_filter(image, size) == image
    if threshold == 'mean':
        maxima_img[image < image[image > 0].mean()] = 0
    else:
        maxima_img[image < threshold] = 0
    local_maxima = np.array(np.where(maxima_img > 0))
    return local_maxima


def get_spine_features(
        image, size, degree, sigma=0.5, N=50, optimise_cycle=10,
        n_jobs=1, report=True, see_cc=False, see_features=False
    ):
    """
    Get the coordinates corresponding the spine inside the fish
    """
    win_size = 2 * size + 1
    template_1 = get_doughnut(size)
    template_2 = get_circle_template(size)
    cc = fftconvolve(image - image[image > 0].mean(), template_1, mode='same')
    cc[cc < 0] = 0
    cc = fftconvolve(cc, template_2, mode='same')
    cc[cc < 0] = 0
    maxima = get_maxima(cc, size)
    xy = np.flip(maxima, axis=0)
    if see_cc:
        plt.scatter(*xy, color='w', marker='o', facecolor='none')
    maxima = get_similar_features(image, maxima, size=win_size)
    xy = np.flip(maxima, axis=0)
    if see_cc:
        plt.imshow(cc, cmap='bwr')
        plt.scatter(*xy, color='teal', marker='+')
        plt.axis('off')
        plt.show()
    if n_jobs == 1:
        spine_features_xy = ransac(
            xy.T, degree=degree, sigma=sigma, N=N,
            T=maxima.shape[1]//2, report=report
        ).T
    else:
        spine_features_xy = ransac_mp(
            xy.T, degree=degree, sigma=sigma, N=N,
            T=maxima.shape[1]//2, report=report, n_jobs=n_jobs
        ).T
    history_opt = [spine_features_xy]
    history_cost = [get_feature_cost(
        image, spine_features_xy, size=win_size, deg=degree
    )]
    if report:
        to_iter = tqdm(range(optimise_cycle))
    else:
        to_iter = range(optimise_cycle)
    for _ in to_iter:
        optimised, cost = optimise_spine_features(
            history_opt[-1], image, size=win_size, degree=degree,
            template=template_2, sigma=sigma, N=N, T=history_opt[-1].shape[1]//2
        )
        history_opt.append(optimised)
        history_cost.append(cost)
    spine_features_xy = history_opt[np.argmin(history_cost)]
    if see_features:
        plt.imshow(image, cmap='gray')
        plt.scatter(
            *spine_features_xy, color='tomato', marker='+',
            label='Spine Features'
        )
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return spine_features_xy


def get_spine_from_features(image, features, size, degree, blur=2, threshold=2):
    """
    Generate a mask image just for the spine from spine features
    """
    src, dst, poly_par = get_poly_parameters(features, axis=0, degree=degree)
    win_size=size * 4 + 1
    prof_2d = get_feature_profile(
        image, features, size=win_size, deg=degree
    )
    h_centre = prof_2d.shape[1]//2
    centre_mask = np.zeros(prof_2d.shape)
    centre_mask[:, h_centre] = 1
    centre_mask = ndimage.gaussian_filter1d(centre_mask, size//2, axis=1)
    lap = ndimage.laplace(ndimage.gaussian_filter(prof_2d, blur))
    lap = lap.max() - lap
    prof_weighted = centre_mask * prof_2d * lap
    prof_mask = prof_weighted > prof_weighted.std() * threshold
    positions = np.array(np.where(prof_mask > 0))
    pos_rec = remap(positions, image, src, dst, poly_par, linewidth=win_size)
    pos_rec = np.flip(pos_rec, axis=0)
    mask_spine = np.zeros(image.shape, dtype=np.uint8)
    mask_spine[tuple(pos_rec.astype(int))] = 1
    mask_spine = ndimage.binary_closing(mask_spine)
    return mask_spine


def imshow_with_mask(image, mask, cmap_img='gray', cmap_mask='magma', figsize=(6, 4)):
    """
    Overlap a binary mask on an image and show them with matplotlib.imshow.

    Args:
        image (np.ndarray): a 2D image
        mask (np.ndarray): a binary mask, with background value assigned to 0
        cmap_img (str): the name of the colourmap for the image
        cmap_mask (str): the name of the colourmap for the mask, the central
            colours will be taken to plot the mask.
        figsize (tuple): the size of the output figure in inches

    Return:
        None
    """
    my_cmap = eval(f'cm.{cmap_mask}')
    my_cmap = copy.copy(my_cmap) # do not modify default cmap
    my_cmap.set_under('k', alpha=0)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.imshow(image, cmap=cmap_img)
    ax.imshow(mask, cmap=my_cmap, clim=[0.1, 2], interpolation='none')
    plt.axis('off')
    plt.show()
