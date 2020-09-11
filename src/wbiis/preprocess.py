import pickle
import cv2
import os
import time

from .index import Entry, Index
from .wavelet import get_wavelet_features
from .constants import INDEX_NAME


def preprocess_images(img_folder, resize_dimensions, thumbs_folder):
    """
    Preprocess images
    :param img_folder: Folder to scan for images
    :param resize_dimensions: (width, height) tuple
    :param thumbs_folder: Folder name for the thumbnails
    :return:
    """
    print('Generating thumbnails...')
    start = time.process_time()

    create_thumbnails_folder(img_folder, thumbs_folder)

    for file in os.listdir(img_folder):

        path = os.path.join(img_folder, file)

        if ignore(path):
            continue

        img = cv2.imread(path)
        if img is None:
            raise ValueError('Error: Folder should consist of only image files. Please remove {0}'.format(path))
        img = cv2.resize(img, resize_dimensions)

        out = os.path.join(img_folder, thumbs_folder, file)
        cv2.imwrite(out, img)

    end = time.process_time()
    print("{0:.2f}s".format(end - start))


def build_index(folder, wavelet, level):
    """
    Builds the index from wavelet features and saves to disk
    :param folder: Folder where the images to index are located
    :param wavelet: Wavelet type to compute the features
    :param level: Decomposition level
    :return:
    """
    print('Building index...')
    start = time.process_time()

    index = Index(wavelet, level)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        path = os.path.abspath(path)

        if ignore(path):
            continue

        img = cv2.imread(path)
        WCi, sigma_ci, l5_WCi = get_wavelet_features(img, wavelet, level)
        entry = Entry(path, WCi, sigma_ci, l5_WCi)
        index.entries.append(entry)

    with open(os.path.join(folder, INDEX_NAME), 'wb') as f:
        pickle.dump(index, f, pickle.HIGHEST_PROTOCOL)

    end = time.process_time()
    print("{0:.2f}s".format(end - start))

    return index


def ignore(path):
    return os.path.isdir(path) or os.path.basename(path).startswith('.')


def create_thumbnails_folder(img_folder, thumbs_folder):
    try:
        os.mkdir(os.path.join(img_folder, thumbs_folder))
    except FileExistsError:
        pass
