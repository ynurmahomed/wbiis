import argparse
import cv2
import os
import pickle
import sys
import time

from wbiis.constants import HEIGHT, INDEX_NAME, LEVEL, THUMBS_FOLDER, WAVELET, WIDTH
from wbiis.preprocess import preprocess_images, build_index


def main():
    parser = argparse.ArgumentParser(prog='wbiis')

    subparsers = parser.add_subparsers(title='subcommands')

    index_parser = subparsers.add_parser('index', help='index images')
    index_parser.set_defaults(func=index)
    index_parser.add_argument('path', help='image folder to be indexed', nargs='?', default=os.getcwd())

    search_parser = subparsers.add_parser('search', help='search images')
    search_parser.set_defaults(func=search)
    search_parser.add_argument('path', help='image folder', nargs='?', default=os.getcwd())
    search_parser.add_argument('query', help='query image')
    search_parser.add_argument('--n-results', help='number of results to return', type=int)
    search_parser.add_argument('--save', help='name of the output file')

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

    index_file = os.path.join(args.path, THUMBS_FOLDER, INDEX_NAME)
    with open(index_file, 'rb') as f:
        idx = pickle.load(f)

    img = cv2.imread(args.query)
    if img is None:
        raise ValueError('{0} is not an image'.format(args.query))

    start = time.process_time()
    results = idx.search(img, n_results)
    end = time.process_time()

    for i, (d, e) in enumerate(results):
        print('{0} {1} {2:.2f}'.format(i + 1, e.path, d))
    print("{0:.2f}s".format(end - start))



if __name__ == '__main__':
    main()
