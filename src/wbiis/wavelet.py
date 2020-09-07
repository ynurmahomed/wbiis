import cv2
import numpy as np
import pywt


def get_wavelet_features(img, wavelet, level):
    """
    Computes the wavelet features.
    :param img: Source image
    :param wavelet: Wavelet type
    :param level: Decomposition level
    :return: 4-tuple with features and processing time
    """
    im_array = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    c1 = im_array[:, :, 1]
    c2 = im_array[:, :, 2]
    c3 = im_array[:, :, 0]

    coeffs_c1 = pywt.wavedec2(c1, wavelet, level=level)
    coeffs_c2 = pywt.wavedec2(c2, wavelet, level=level)
    coeffs_c3 = pywt.wavedec2(c3, wavelet, level=level)

    w_c1 = coeffs_c1[:2]
    w_c2 = coeffs_c2[:2]
    w_c3 = coeffs_c3[:2]

    sigma_c1 = np.std(coeffs_c1[0])
    sigma_c2 = np.std(coeffs_c2[0])
    sigma_c3 = np.std(coeffs_c3[0])

    level += 1
    l5_coeffs_c1 = pywt.wavedec2(c1, wavelet, level=level)
    l5_coeffs_c2 = pywt.wavedec2(c2, wavelet, level=level)
    l5_coeffs_c3 = pywt.wavedec2(c3, wavelet, level=level)
    l5_w_c1 = l5_coeffs_c1[:2]
    l5_w_c2 = l5_coeffs_c2[:2]
    l5_w_c3 = l5_coeffs_c3[:2]

    w_c = [w_c1, w_c2, w_c3]
    sigma_c = [sigma_c1, sigma_c2, sigma_c3]
    l5_w_c = [l5_w_c1, l5_w_c2, l5_w_c3]

    return w_c.copy(), sigma_c.copy(), l5_w_c.copy()


def _show_wavelet_image(coeffs, level):
    # normalize each coefficient array independently for better visibility
    coeffs[0] /= np.abs(coeffs[0]).max()
    for detail_level in range(level):
        coeffs[detail_level + 1] = [d / np.abs(d).max() for d in coeffs[detail_level + 1]]
    arr, slices = pywt.coeffs_to_array(coeffs)
    _show_image(arr)


def _show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
