import numpy as np

from .constants import BETA, W11, W12, W21, W22, WC1, WC2, WC3, L5_FACTOR
from .wavelet import get_wavelet_features

"""
LAB color channels
"""
WB = 0
RG = 1
BY = 2


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
        WCi, sigma_ci, l5_WCi = get_wavelet_features(img, self.wavelet, self.level)
        for e in self.entries:
            if acceptance(e.sigma_ci, sigma_ci):
                if euclidean(min_max(e.l5_WCi), min_max(l5_WCi)) > L5_FACTOR:
                    continue
                else:
                    results.append((dist(e.WCi, WCi), e))

        return sorted(results, key=lambda r: r[0])[:n_results]


class Entry:
    """
    Wavelet features index entry
    """
    def __init__(self, path='', WCi=None, sigma_ci=None, l5_WCi=None):
        """
        :param path: The image path
        :param WCi: 16x16 4-layer 2-D fast wavelet transform submatrices
        :param sigma_ci: 8x􏰊8 corner submatrices standard deviations
        :param l5_WCi: 8x8 5-level 2-D fast wavelet transform
        """
        self.path = path
        self.WCi = WCi
        self.sigma_ci = sigma_ci
        self.l5_WCi = l5_WCi

    def __repr__(self):
        return 'Entry(path={0}, sigma_c={1})'.format(self.path, self.sigma_ci)


def acceptance(idx_sigma_ci, sigma_ci):
    """
    Computes the acceptance criteria
    :param idx_sigma_ci: Indexed image standard deviation
    :param sigma_ci: Query image standard deviation
    :return: If passes the acceptance criteria
    """
    return sigma_ci[WB] * BETA < idx_sigma_ci[WB] < sigma_ci[WB] / BETA or \
           ((sigma_ci[RG] * BETA < idx_sigma_ci[RG] < sigma_ci[RG] / BETA) and
            (sigma_ci[BY] * BETA < idx_sigma_ci[BY] < sigma_ci[BY] / BETA))


def dist(idx_WCi, WCi):
    """
    Computes the distance between the wavelet feature vectors
    :param idx_WCi: Indexed image 2 last wavelet features
    :param wci: Query image 2 last wavelet features
    :return: Euclidean distance
    """
    wci = np.array([WC1, WC2, WC3])

    idx_W11 = idx_WCi[0]
    WC11 = WCi[0]

    idx_W12 = idx_WCi[1]['da']
    WC12 = WCi[1]['da']

    idx_W21 = idx_WCi[1]['ad']
    WC21 = WCi[1]['ad']

    idx_W22 = idx_WCi[1]['dd']
    WC22 = WCi[1]['dd']

    return W11 * np.sum(wci * euclidean(WC11, idx_W11)) \
           + W12 * np.sum(wci * euclidean(WC12, idx_W12)) \
           + W21 * np.sum(wci * euclidean(WC21, idx_W21)) \
           + W22 * np.sum(wci * euclidean(WC22, idx_W22))


def min_max(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr - mn) * (1.0 / (mx - mn))


def euclidean(a, b):
    return np.sqrt(np.sum((a-b)**2))
