import sys
import json

import chainer

import nets
import text_datasets


def setup_model(args):
    sys.stderr.write(json.dumps(args.__dict__, indent=2) + '\n')
    setup = json.load(open(args.model_setup))
    sys.stderr.write(json.dumps(setup, indent=2) + '\n')

    # Load a dataset
    dataset = setup['dataset']
    if dataset == 'dbpedia':
        train, test, vocab = text_datasets.get_dbpedia(
            char_based=setup['char_based'])
    elif dataset == 'snli':
        train, test, vocab = text_datasets.get_snli(
            char_based=setup['char_based'],
            combine=setup['combine_snli'])
    elif dataset.startswith('imdb.'):
        train, test, vocab = text_datasets.get_imdb(
            fine_grained=dataset.endswith('.fine'),
            char_based=setup['char_based'])
    elif dataset in ['TREC', 'stsa.binary', 'stsa.fine',
                     'custrev', 'mpqa', 'rt-polarity', 'subj']:
        train, test, vocab = text_datasets.get_other_text_dataset(
            dataset, char_based=setup['char_based'])

    # vocab = json.load(open(setup['vocab_path']))
    n_class = setup['n_class']
    print('# train data: {}'.format(len(train)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    print('# class: {}'.format(n_class))

    # Setup a model
    if setup['model'] == 'rnn':
        Encoder = nets.RNNEncoder
    elif setup['model'] == 'bilstm':
        Encoder = nets.BiLSTMEncoder
    elif setup['model'] == 'cnn':
        Encoder = nets.CNNEncoder
    elif setup['model'] == 'bow':
        Encoder = nets.BOWMLPEncoder
    encoder = Encoder(n_layers=setup['layer'], n_vocab=len(vocab),
                      n_units=setup['unit'], dropout=setup['dropout'])
    if dataset == 'snli':
        model = nets.SNLIClassifier(encoder, combine=setup['combine_snli'])
    else:
        model = nets.TextClassifier(encoder, n_class)
    chainer.serializers.load_npz(setup['model_path'], model)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    return model, train, test, vocab, setup
