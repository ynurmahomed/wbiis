import os
import sys

from wbiis.preprocess import preprocess_images, build_index

HEIGHT = 128
WIDTH = 128

THUMBS_FOLDER = 'thumbs'

LEVEL = 4
WAVELET = 'db8'


def main(args):
    img_folder = args[1]
    preprocess_images(img_folder, (HEIGHT, WIDTH), THUMBS_FOLDER)
    build_index(os.path.join(img_folder, THUMBS_FOLDER), WAVELET, LEVEL)


if __name__ == '__main__':
    main(sys.argv)
