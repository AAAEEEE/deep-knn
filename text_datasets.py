import csv
import glob
import io
import os
import json
import shutil
import tarfile
import tempfile
import zipfile

import numpy

import chainer

from nlp_utils import make_vocab
from nlp_utils import normalize_text
from nlp_utils import split_text
from nlp_utils import transform_to_array
from nlp_utils import transform_snli_to_array

URL_DBPEDIA = 'https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz'  # NOQA
URL_IMDB = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
URL_OTHER_BASE = 'https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/'  # NOQA
URL_SNLI = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'


def download_dbpedia():
    path = chainer.dataset.cached_download(URL_DBPEDIA)
    tf = tarfile.open(path, 'r')
    return tf


def read_dbpedia(tf, split, shrink=1, char_based=False):
    dataset = []
    f = tf.extractfile('dbpedia_csv/{}.csv'.format(split))
    f = io.StringIO(f.read().decode('utf-8'))
    for i, (label, title, text) in enumerate(csv.reader(f)):
        if i % shrink != 0:
            continue
        label = int(label) - 1  # Index begins from 1
        tokens = split_text(normalize_text(text), char_based)
        dataset.append((tokens, label))
    return dataset


def get_dbpedia(vocab=None, shrink=1, char_based=False):
    tf = download_dbpedia()

    print('read dbpedia')
    train = read_dbpedia(tf, 'train', shrink=shrink, char_based=char_based)
    test = read_dbpedia(tf, 'test', shrink=shrink, char_based=char_based)

    if vocab is None:
        print('constract vocabulary based on frequency')
        vocab = make_vocab(train)

    train = transform_to_array(train, vocab)
    test = transform_to_array(test, vocab)

    return train, test, vocab


def download_imdb():
    path = chainer.dataset.cached_download(URL_IMDB)
    tf = tarfile.open(path, 'r')
    # To read many files fast, tarfile is untared
    path = tempfile.mkdtemp()
    tf.extractall(path)
    return path


def read_imdb(path, split,
              shrink=1, fine_grained=False, char_based=False):
    fg_label_dict = {'1': 0, '2': 0, '3': 1, '4': 1,
                     '7': 2, '8': 2, '9': 3, '10': 3}

    def read_and_label(posneg, label):
        dataset = []
        target = os.path.join(path, 'aclImdb', split, posneg, '*')
        for i, f_path in enumerate(glob.glob(target)):
            if i % shrink != 0:
                continue
            with io.open(f_path, encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            tokens = split_text(normalize_text(text), char_based)
            if fine_grained:
                # extract from f_path. e.g. /pos/200_8.txt -> 8
                label = fg_label_dict[f_path.split('_')[-1][:-4]]
                dataset.append((tokens, label))
            else:
                dataset.append((tokens, label))
        return dataset

    pos_dataset = read_and_label('pos', 0)
    neg_dataset = read_and_label('neg', 1)
    return pos_dataset + neg_dataset


def get_imdb(vocab=None, shrink=1, fine_grained=False,
             char_based=False):
    tmp_path = download_imdb()

    print('read imdb')
    train = read_imdb(tmp_path, 'train',
                      shrink=shrink, fine_grained=fine_grained,
                      char_based=char_based)
    test = read_imdb(tmp_path, 'test',
                     shrink=shrink, fine_grained=fine_grained,
                     char_based=char_based)

    shutil.rmtree(tmp_path)

    if vocab is None:
        print('constract vocabulary based on frequency')
        vocab = make_vocab(train)

    train = transform_to_array(train, vocab)
    test = transform_to_array(test, vocab)

    return train, test, vocab


def download_other_dataset(name):
    if name in ['custrev', 'mpqa', 'rt-polarity', 'subj']:
        files = [name + '.all']
    elif name == 'TREC':
        files = [name + suff for suff in ['.train.all', '.test.all']]
    else:
        files = [name + suff for suff in ['.train', '.test']]
    file_paths = []
    for f_name in files:
        url = os.path.join(URL_OTHER_BASE, f_name)
        path = chainer.dataset.cached_download(url)
        file_paths.append(path)
    return file_paths


def read_other_dataset(path, shrink=1, char_based=False):
    dataset = []
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        for i, l in enumerate(f):
            if i % shrink != 0 or not len(l.strip()) >= 3:
                continue
            label, text = l.strip().split(None, 1)
            label = int(label)
            tokens = split_text(normalize_text(text), char_based)
            dataset.append((tokens, label))
    return dataset


def get_other_text_dataset(name, vocab=None, shrink=1,
                           char_based=False, seed=777):
    assert(name in ['TREC', 'stsa.binary', 'stsa.fine',
                    'custrev', 'mpqa', 'rt-polarity', 'subj'])
    datasets = download_other_dataset(name)
    train = read_other_dataset(
        datasets[0], shrink=shrink, char_based=char_based)
    if len(datasets) == 2:
        test = read_other_dataset(
            datasets[1], shrink=shrink, char_based=char_based)
    else:
        numpy.random.seed(seed)
        alldata = numpy.random.permutation(train)
        train = alldata[:-len(alldata) // 10]
        test = alldata[-len(alldata) // 10:]

    if vocab is None:
        print('constract vocabulary based on frequency')
        vocab = make_vocab(train)

    train = transform_to_array(train, vocab)
    test = transform_to_array(test, vocab)

    return train, test, vocab


def download_snli():
    print('download snli')
    # path = os.path.join(DATA_DIR, 'snli.zip')
    # if not os.path.exists(path):
    #     urllib.request.urlretrieve(URL_SNLI, path)
    path = chainer.dataset.cached_download(URL_SNLI)
    with zipfile.ZipFile(path, 'r') as zf:
        zf.extractall()
    path = os.path.abspath(os.path.join(path, os.pardir))
    path = os.path.join(path, 'snli_1.0')
    return path


def most_common(lst):
    return max(set(lst), key=lst.count)


def read_snli(path, split, shrink=1, char_based=False):
    path = os.path.join(path, 'snli_1.0_{}.jsonl'.format(split))
    dataset = []
    labels = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    with open(path) as f:
        for i, x in enumerate(f.readlines()):            
            if i % shrink != 0:
                continue
            x = json.loads(x)
            if x['gold_label'] in labels:
                label = labels[x['gold_label']]
            else:
                label = labels[most_common(x['annotator_labels'])]
            premise = split_text(normalize_text(x['sentence1']), char_based)
            hypothesis = split_text(normalize_text(x['sentence2']), char_based)
            dataset.append((premise, hypothesis, label))
    return dataset


def get_snli(vocab=None, shrink=1, char_based=False, combine=False):
    path = download_snli()

    print('read snli')
    path = 'snli_1.0'
    train = read_snli(path, 'train', shrink=shrink, char_based=char_based)
    test = read_snli(path, 'dev', shrink=shrink, char_based=char_based)

    if vocab is None:
        print('construct vocabulary based on frequency')
        train_premise = [(x, z) for x, y, z in train]
        train_hypothesis = [(y, z) for x, y, z in train]
        train_all = train_premise + train_hypothesis
        vocab = make_vocab(train_all, special=['$$$'] if combine else [])

    train = transform_snli_to_array(train, vocab, combine=combine)
    test = transform_snli_to_array(test, vocab, combine=combine)

    return train, test, vocab


if __name__ == '__main__':
    train, test, vocab = get_snli()
    print(len(train))
