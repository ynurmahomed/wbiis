import argparse
import cv2
import numpy as np
import os
import pickle
import sys
import time

from wbiis.constants import GALLERY_COLUMNS, HEIGHT, INDEX_NAME, LEVEL, SHOW_IMG_MAX, THUMBS_FOLDER, WAVELET, WIDTH
from wbiis.preprocess import preprocess_images, build_index
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(prog='wbiis')

    subparsers = parser.add_subparsers(title='subcommands')

    index_parser = subparsers.add_parser('index', help='index images')
    index_parser.set_defaults(func=index)
    index_parser.add_argument('path', help='image folder to be indexed. Defaults to cwd', nargs='?',
                              default=os.getcwd())

    search_parser = subparsers.add_parser('search', help='search images')
    search_parser.set_defaults(func=search)
    search_parser.add_argument('path', help='image folder. Defaults to cwd', nargs='?', default=os.getcwd())
    search_parser.add_argument('query', help='query image')
    search_parser.add_argument('--n-results', help='number of results to return', type=int)
    search_parser.add_argument('--save', help='save the results', action='store_true')
    search_parser.add_argument('--show', help='display the query image and results', action='store_true')

    args = parser.parse_args()
    if 'func' not in args:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args.func(args)


def index(args):
    img_folder = args.path
    preprocess_images(img_folder, (HEIGHT, WIDTH), THUMBS_FOLDER)
    build_index(os.path.join(img_folder, THUMBS_FOLDER), WAVELET, LEVEL)


def search(args):
    n_results = args.n_results

    # TODO prompt to build the index if not present
    index_file = os.path.join(args.path, THUMBS_FOLDER, INDEX_NAME)
    with open(index_file, 'rb') as f:
        idx = pickle.load(f)

    # TODO resize image if bigger than 128x128
    img = cv2.imread(args.query)
    if img is None:
        raise ValueError('{0} is not an image'.format(args.query))

    start = time.process_time()
    results = idx.search(img, n_results)
    end = time.process_time()

    for i, (d, e) in enumerate(results):
        print('{0} {1} {2:.2f}'.format(i + 1, e.path, d))
    print("{0:.2f}s".format(end - start))

    if args.save:
        save(args.query, results)

    if args.show:
        show(img, results)


def save(query, results):
    name = os.path.basename(query).split('.')[0]
    with ZipFile('{0}.zip'.format(name), 'w') as z:
        for (d, e) in results:
            arcname = os.path.basename(e.path)
            filename = os.path.abspath(e.path)
            z.write(filename, arcname)
    print('Results saved in {0}.zip'.format(name))


def show(query_img, results):
    opencv_bg = 50
    n_results = len(results) if len(results) <= SHOW_IMG_MAX else SHOW_IMG_MAX
    remain = ((((n_results // GALLERY_COLUMNS) + 1) * GALLERY_COLUMNS) - n_results) % GALLERY_COLUMNS
    fill = [np.full(query_img.shape, opencv_bg, dtype=np.uint8) for _ in range(remain)]
    to_show = [cv2.imread(e.path) for _, e in results[:n_results]]
    g = gallery(np.array(to_show + fill), ncols=GALLERY_COLUMNS)
    cv2.imshow('Query image', query_img)
    cv2.imshow('Results', g)
    cv2.waitKey(0)


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result


if __name__ == '__main__':
    main()
