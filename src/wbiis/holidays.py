import os

from wbiis.constants import INDEX_NAME, THUMBS_FOLDER


def print_scores(path, query, results):
    """
    Computes print scores (true/false positive rates, precision, recall and accuracy) for
    the holiday dataset.
    If query image is in the result it is ignored.
    :param path: Database path
    :param query: Query image path
    :param results: Query results
    :return: None
    """
    def fname(f):
        base = os.path.basename(f)
        return os.path.splitext(base)[0]

    db = sorted([int(fname(f)) for f in os.listdir(os.path.join(path, THUMBS_FOLDER)) if not f == INDEX_NAME])
    q = int(fname(query))
    results = [int(fname(e.path)) for _, e in results if not int(fname(e.path)) == q]
    positives = [d for d in db if d > q and d - q < 100]
    negatives = [d for d in db if d < q or d - q >= 100]
    tp = len([r for r in results if r in positives])
    fp = len([r for r in results if r in negatives])
    tn = len([r for r in negatives if r not in results])
    p = len(positives)
    n = len(negatives)
    recall = tp / p
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    accuracy = (tp + tn) / (p + n)

    print('TP {0} FP {1} Precision {2:.2f} Recall {3:.2f} Accuracy {4:.2f}'.format(tp, fp, precision, recall, accuracy))


def print_inline(results):
    """
    Prints holidays the query image name followed by a sequence of rank and result image name.
    Useful for the holidays_map.py script
    :param results: Query results
    :return: None
    """
    r = ['{0} {1}'.format(i, os.path.basename(e.path)) for i, (d, e) in enumerate(results)]
    print(' '.join(r))
