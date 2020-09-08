import numpy as np

from .constants import BETA, W11, W12, W21, W22, WC1, WC2, WC3
from .wavelet import get_wavelet_features


class Index:
    """
    Image index using wavelet features
    """
    def __init__(self, wavelet, level, entries=None):

        self.level = level
        self.wavelet = wavelet

        if entries is None:
            entries = []

        self.entries = entries

    def search(self, img, n_results):
        results = []
        wc, sigma_c, l5_wc = get_wavelet_features(img, self.wavelet, self.level)
        for e in self.entries:
            if not acceptance(e.sigma_c, sigma_c):
                continue
            results.append((dist(e.wc, wc), e))

        return sorted(results, key=lambda r: r[0])[:n_results]


class Entry:
    """
    Wavelet features index entry
    """
    def __init__(self, path='', wc=None, sigma_c=None, l5_wc=None):
        """
        :param path: The image path
        :param wc: 16x16 4-layer 2-D fast wavelet transform submatrices
        :param sigma_c: 8x􏰊8 corner submatrices standard deviations
        :param l5_wc: 8x8 5-level 2-D fast wavelet transform
        """

        self.path = path
        self.wc = wc
        self.sigma_c = sigma_c
        self.l5_wc = l5_wc

    def __repr__(self):
        return 'Entry(path={0}, sigma_c={1})'.format(self.path, self.sigma_c)


def acceptance(idx_sigma_c, sigma_c):
    """
    Computes the acceptance criteria
    :param idx_sigma_c: Indexed image standard deviation
    :param sigma_c: Query image standard deviation
    :return: If passes the acceptance criteria
    """
    return sigma_c[0] * BETA < idx_sigma_c[0] < sigma_c[0] / BETA or \
           ((sigma_c[1] * BETA < idx_sigma_c[1] < sigma_c[1] / BETA) and
            (sigma_c[2] * BETA < idx_sigma_c[2] < sigma_c[2] / BETA))


def dist(idx_wc, wc):
    """
    Computes the distance between the wavelet feature vectors
    :param idx_wc: Indexed image 2 last wavelet features
    :param wc: Query image 2 last wavelet features
    :return: Euclidean distance
    """
    wci = np.array([WC1, WC2, WC3])

    idx_w11 = np.array([idx_wc[0][0], idx_wc[1][0], idx_wc[2][0]])
    w11 = np.array([wc[0][0], wc[1][0], wc[2][0]])

    idx_w12 = np.array([idx_wc[0][1][0], idx_wc[1][1][0], idx_wc[2][1][0]])
    w12 = np.array([wc[0][1][0], wc[1][1][0], wc[2][1][0]])

    idx_w21 = np.array([idx_wc[0][1][1], idx_wc[1][1][1], idx_wc[2][1][1]])
    w21 = np.array([wc[0][1][1], wc[1][1][1], wc[2][1][1]])

    idx_w22 = np.array([idx_wc[0][1][2], idx_wc[1][1][2], idx_wc[2][1][2]])
    w22 = np.array([wc[0][1][2], wc[1][1][2], wc[2][1][2]])

    return W11 * np.sum(wci * matrix_norm(w11 - idx_w11)) \
           + W12 * np.sum(wci * matrix_norm(w12 - idx_w12)) \
           + W21 * np.sum(wci * matrix_norm(w21 - idx_w21)) \
           + W22 * np.sum(wci * matrix_norm(w22 - idx_w22))


def matrix_norm(arr):
    """
    :param arr: Array of matrices
    :return:
    """
    return np.linalg.norm(arr, axis=(1, 2))
