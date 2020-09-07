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
    def __init__(self, path='', wc1_16=None, std_dev_wc1_8=None, l5_wc1_8=None):
        """
        :param path: The image path
        :param wc1_16: 16x16 4-layer 2-D fast wavelet transform submatrices
        :param std_dev_wc1_8: 8x􏰊8 corner submatrices standard deviations
        :param l5_wc1_8: 8x8 5-level 2-D fast wavelet transform
        """

        self.path = path
        self.wc1_16 = wc1_16
        self.std_devWc1_8 = std_dev_wc1_8
        self.l5_wc1_8 = l5_wc1_8
