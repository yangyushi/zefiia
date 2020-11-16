from . import utility
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import scikit_posthocs as sp



def find_notochord_clusters(image, threshold, classifier, conditions, axis='y', nmax=1000):
    """
    Find the clusters that belongs to the notochord using unsuperisverd
        machine-learning methods

    Args:
        image (np.ndarray): 2D image that contains the notochord
        threshold (float): the intensity threshold for the notochord
        classifier (sklearn.cluster objects): the classifier
            to be used to cluster the foreground points
        conditions (iterable): a list of functions to check the quality
            of the clusters, the format would be:
            `func(cluster, image) -> bool`
        axis (str): the axis where the data will be used for clustering.
            valid options are 'x', 'y', or 'xy'.
        nmax (int): if there are more than [nmax] pixels for a notochord,
            randomly select [nmax] pixels for the analysis
    
    Return:
        list of np.ndarray: a list of clusters that represent the pixels
            belonging to the notochord
    """
    binary = image > (image.max() * threshold)
    points_xy = np.flip(np.nonzero(binary), axis=0)  # shape (2, n)
    if axis == 'x':
        points = np.array(points_xy[0])[:, np.newaxis]  # shape (n, 2)
    elif axis == 'y':
        points = np.array(points_xy[1])[:, np.newaxis]
    elif axis == 'xy':
        points = np.array(points_xy).T
    else:
        raise ValueError("Invalid axis option", axis)
    if points.shape[0] > nmax:
        rand_idx = np.arange(points.shape[0])
        np.random.shuffle(rand_idx)
        rand_idx = rand_idx[:nmax]
        points = points[rand_idx]
        points_xy = points_xy[:, rand_idx]
    classifier.fit(points)
    clusters = []
    labels = classifier.labels_
    for val in set(labels):
        cluster = points_xy[:, labels == val].T  # shape (n, 2)
        is_notochord = True
        for check_func in conditions:
            is_notochord *= check_func(cluster, image)
        if is_notochord:
            clusters.append(cluster)
    return clusters   


def measure_notochord(image, points, method, order=5, width=20, etol=1e12, plot=False):
    """
    Measure the intensity profile of the notochord of the zebrafish baby
        under confocal images
        
    Args:
        image (np.ndarray): the 2D image where only one notochord can be seen
        threshold (np.ndarray): the intensity threshold for the notochord
        order (int): the order of polynomial fit for the notochord
        width (int): the width of the curve when measuring the profile 
            this is similar to the `line_width` in software ImageJ
        etol (float): the tolarance of the fitting. If the fitting residual is greater
            than `etol`, then return None
        plot (bool): if True, plot the profile along with the image.
            This option is designed to inspect the fitting result.
            
    Result:
        np.ndarray: the fluorescent intensity profile along the notochord
    """
    idx_min = np.argmin(points[:, 0])
    idx_max = np.argmax(points[:, 0])

    par, residual, _, _, _ = np.polyfit(
        *points.T, deg=order, full=True,
    )
    
    profile = utility.profile_poly(
        image, points[idx_min], points[idx_max], par, linewidth=width,
        method=method
    )
    
    if plot:
        x = np.linspace(points[idx_min, 0], points[idx_max, 0], 100)
        y = np.polyval(par, x)
        px = np.linspace(x[0], x[-1], len(profile))
        py = np.polyval(par, px)
        plt.imshow(image)
        plt.plot(x, y, color='tomato', lw=2, ls='-')
        plt.plot(px, py - profile, color='pink', lw=1)
        plt.ylim(py.max() + 100, py.min() - np.max(profile) - 100)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gcf().set_size_inches(12, 4)
        plt.show()
        
    if residual > etol:
        print("Notochord fitting failed, check the raw image!")
    else:
        return profile


def measure_notochord_ransac(
    image, points, method, order, width,
    sigma, N, T, alpha=0.95,
    etol=1e12, plot=False
    ):
    """
    Measure the intensity profile of the notochord of the zebrafish baby
        under confocal images using RANSAC algorithm
        
    Args:
        image (np.ndarray): the 2D image where only one notochord can be seen
        threshold (np.ndarray): the intensity threshold for the notochord
        order (int): the order of polynomial fit for the notochord
        width (int): the width of the curve when measuring the profile 
            this is similar to the `line_width` in software ImageJ
            than `etol`, then return None
        sigma (float): the sigma of the gaussian distribution of the error
        N (int): number of repeats
        T (int): the targeted support size
        alpha (float): the confidence level of the chi2 distribution
        etol (float): the tolarance of the fitting. If the fitting residual is greater
        plot (bool): if True, plot the profile along with the image.
            This option is designed to inspect the fitting result.
            
    Result:
        np.ndarray: the fluorescent intensity profile along the notochord
    """
    idx_min = np.argmin(points[:, 0])
    idx_max = np.argmax(points[:, 0])

    indices = np.arange(points.shape[0])
    np.random.shuffle(indices)

    ransac_support = utility.ransac(
        points[indices][:T * 2],
        degree=order, sigma=sigma,
        N=N, T=T, alpha=alpha
    )

    par, residual, _, _, _ = np.polyfit(
        *ransac_support.T, deg=order, full=True,
    )

    profile = utility.profile_poly(
        image, points[idx_min], points[idx_max], par, linewidth=width,
        method=method
    )
    
    if plot:
        x = np.linspace(points[idx_min, 0], points[idx_max, 0], 100)
        y = np.polyval(par, x)
        px = np.linspace(x[0], x[-1], len(profile))
        py = np.polyval(par, px)
        plt.imshow(image)
        plt.plot(x, y, color='tomato', lw=2, ls='-')
        plt.plot(px, py - profile, color='pink', lw=1)
        plt.ylim(py.max() + 100, py.min() - np.max(profile) - 100)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gcf().set_size_inches(12, 4)
        plt.show()
        
    if residual > etol:
        print("Notochord fitting failed, check the raw image!")
    else:
        return profile
