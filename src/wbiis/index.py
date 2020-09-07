class Index:
    """
    Image index using wavelet features
    """
    def __init__(self, entries=None):

        if entries is None:
            entries = []

        self.entries = entries


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
