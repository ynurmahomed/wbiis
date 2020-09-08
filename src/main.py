import argparse
import cv2
import os
import pickle

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
    args.func(args)


def index(args):
    img_folder = args.path
    preprocess_images(img_folder, (HEIGHT, WIDTH), THUMBS_FOLDER)
    build_index(os.path.join(img_folder, THUMBS_FOLDER), WAVELET, LEVEL)


# TODO:  If an image in the database differs from the querying image too much
#  when we compare the 8x8x3=192 dimensional feature vector, we discard it
def search(args):
    n_results = args.n_results

    index_file = os.path.join(args.path, THUMBS_FOLDER, INDEX_NAME)
    with open(index_file, 'rb') as f:
        idx = pickle.load(f)

    img = cv2.imread(args.query)
    if img is None:
        raise ValueError('{0} is not an image'.format(args.query))

    results = idx.search(img, n_results)
    for i, (d, e) in enumerate(results):
        print('{0} {1} {2:.2f}'.format(i+1, e.path, d))


if __name__ == '__main__':
    main()
