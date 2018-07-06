import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

embed_init = chainer.initializers.Uniform(.25)


def sequence_embed(embed, xs, dropout=0.):
    """Efficient embedding function for variable-length sequences

    This output is equally to
    "return [F.dropout(embed(x), ratio=dropout) for x in xs]".
    However, calling the functions is one-shot and faster.

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): i-th element in the list is an input variable,
            which is a :math:`(L_i, )`-shaped int array.
        dropout (float): Dropout ratio.

    Returns:
        list of ~chainer.Variable: Output variables. i-th element in the
        list is an output variable, which is a :math:`(L_i, N)`-shaped
        float array. :math:`(N)` is the number of dimensions of word embedding.

    """
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


def block_embed(embed, x, dropout=0.):
    """Embedding function followed by convolution

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which
            is a :math:`(B, L)`-shaped int array. Its first dimension
            :math:`(B)` is assumed to be the *minibatch dimension*.
            The second dimension :math:`(L)` is the length of padded
            sentences.
        dropout (float): Dropout ratio.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(B, N, L, 1)`. :math:`(N)` is the number of dimensions
        of word embedding.

    """
    e = embed(x)
    e = F.dropout(e, ratio=dropout)
    e = F.transpose(e, (0, 2, 1))
    e = e[:, :, :, None]
    return e


class TextClassifier(chainer.Chain):

    """A classifier using a given encoder.

     This chain encodes a sentence and classifies it into classes.

     Args:
         encoder (Link): A callable encoder, which extracts a feature.
             Input is a list of variables whose shapes are
             "(sentence_length, )".
             Output is a variable whose shape is "(batchsize, n_units)".
         n_class (int): The number of classes to be predicted.

     """

    def __init__(self, encoder, n_class, dropout=0.1):
        super(TextClassifier, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            if type(encoder) is BiLSTMEncoder:  # bilstm make twice as big
                self.output = L.Linear(2 * encoder.out_units, n_class)
            else:
                self.output = L.Linear(encoder.out_units, n_class)
        self.dropout = dropout
        self.n_dknn_layers = self.encoder.n_dknn_layers

    def __call__(self, xs, ys):
        concat_outputs = self.predict(xs)
        concat_truths = F.concat(ys, axis=0)

        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        return loss

    def get_onehot_grad(self, xs, ys=None):
        if ys is None:
            with chainer.using_config('train', False):
                ys = self.predict(xs, argmax=True)
                ys = F.expand_dims(ys, axis=1)
                ys = [y for y in ys]
        encodings, exs = self.encoder.get_grad(xs)
        outputs = self.output(encodings)
        concat_truths = F.concat(ys, axis=0)
        loss = F.softmax_cross_entropy(outputs, concat_truths)

        if isinstance(exs, tuple):
            exs_grad = chainer.grad([loss], exs)
            ex_sections = np.cumsum([ex.shape[0] for ex in exs[:-1]])
            exs = F.concat(exs, axis=0)
            exs_grad = F.concat(exs_grad, axis=0)
            onehot_grad = F.sum(exs_grad * exs, axis=1)
            onehot_grad = F.split_axis(onehot_grad, ex_sections, axis=0)
        else:
            exs_grad = chainer.grad([loss], [exs])[0]
            # (batch_size, n_dim, max_length, 1)
            assert exs_grad.shape == exs.shape
            onehot_grad = F.squeeze(F.sum(exs_grad * exs, 1), 2)
            lengths = [len(x) for x in xs]
            onehot_grad = [x[:l] for x, l in zip(onehot_grad, lengths)]
        return onehot_grad

    def predict(self, xs, softmax=False, argmax=False, dknn=False):
        if dknn:
            encodings, dknn_layers = self.encoder(xs, dknn=True)
        else:
            encodings = self.encoder(xs, dknn=False)
        encodings = F.dropout(encodings, ratio=self.dropout)
        outputs = self.output(encodings)
        if softmax:
            outputs = F.softmax(outputs).data
        elif argmax:
            outputs = self.xp.argmax(outputs.data, axis=1)
        if dknn:
            return outputs, dknn_layers
        else:
            return outputs


class SNLIClassifier(chainer.Chain):

    """A classifier using a given encoder.

     This chain encodes a sentence and classifies it into classes.

     Args:
         encoder (Link): A callable encoder, which extracts a feature.
             Input is a list of variables whose shapes are
             "(sentence_length, )".
             Output is a variable whose shape is "(batchsize, n_units)".
         n_class (int): The number of classes to be predicted.

     """

    def __init__(self, encoder, n_class=3, n_layers=3, dropout=0.1,
                 combine=False):
        super(SNLIClassifier, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            if type(encoder) is BiLSTMEncoder:  # bilstm make twice as big
                self.mlp = MLP(n_layers, encoder.out_units * 4 * 2, dropout)
                self.output = L.Linear(encoder.out_units * 4 * 2, n_class)
            else:
                self.mlp = MLP(n_layers, encoder.out_units * 4, dropout)
                self.output = L.Linear(encoder.out_units * 4, n_class)

        self.dropout = dropout
        if combine:
            self.n_dknn_layers = self.mlp.n_dknn_layers + \
                                 self.encoder.n_dknn_layers
        else:
            self.n_dknn_layers = self.mlp.n_dknn_layers + 1
        self.combine = combine

    def __call__(self, xs, ys):
        concat_outputs = self.predict(xs)
        concat_truths = F.concat(ys, axis=0)

        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        return loss

    def get_onehot_grad(self, xs, ys=None):
        if ys is None:
            with chainer.using_config('train', False):
                ys = self.predict(xs, argmax=True)
        assert not self.combine
        u, exs_prem = self.encoder.get_grad(xs[0])
        v, exs_hypo = self.encoder.get_grad(xs[1])
        encodings = F.concat((u, v, F.absolute(u-v), u*v), axis=1)
        outputs = self.output(self.mlp(encodings, no_dropout=True))
        loss = F.softmax_cross_entropy(outputs, ys)

        exs = exs_hypo
        lengths = [len(x) for x in xs[1]]

        if isinstance(exs, tuple):
            exs_grad = chainer.grad([loss], exs)
            ex_sections = np.cumsum([ex.shape[0] for ex in exs[:-1]])
            exs = F.concat(exs, axis=0)
            exs_grad = F.concat(exs_grad, axis=0)
            onehot_grad = F.sum(exs_grad * exs, axis=1)
            onehot_grad = F.split_axis(onehot_grad, ex_sections, axis=0)
        else:
            exs_grad = chainer.grad([loss], [exs])[0]
            # (batch_size, n_dim, max_length, 1)
            assert exs_grad.shape == exs.shape
            onehot_grad = F.squeeze(F.sum(exs_grad * exs, 1), 2)
            onehot_grad = [x[:l] for x, l in zip(onehot_grad, lengths)]
        return onehot_grad

    def predict(self, xs, softmax=False, argmax=False, dknn=False):
        dknn_layers = []
        if self.combine:
            if dknn:
                encodings, dknn_layers = self.encoder(xs, dknn=True)
            else:
                encodings = self.encoder(xs, dknn=False)
        else:
            u = self.encoder(xs[0], dknn=False)
            v = self.encoder(xs[1], dknn=False)
            encodings = F.concat((u, v, F.absolute(u-v), u*v), axis=1)
            dknn_layers = [encodings]
        # encodings = F.dropout(encodings, ratio=self.dropout)
        # don't think we need because mlp uses dropout at beginning

        if dknn:
            outputs, _dknn_layers = self.mlp(encodings, dknn=True)
            dknn_layers = dknn_layers + _dknn_layers
        else:
            outputs = self.mlp(encodings, dknn=False)

        outputs = self.output(outputs)
        if softmax:
            outputs = F.softmax(outputs).data
        elif argmax:
            outputs = self.xp.argmax(outputs.data, axis=1)
        if dknn:
            return outputs, dknn_layers
        else:
            return outputs


class RNNEncoder(chainer.Chain):

    """A LSTM-RNN Encoder with Word Embedding.

    This model encodes a sentence sequentially using LSTM.

    Args:
        n_layers (int): The number of LSTM layers.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of a LSTM layer and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        super(RNNEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units,
                                   initialW=embed_init)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, dropout)

        self.n_layers = n_layers
        self.out_units = n_units
        self.dropout = dropout
        self.n_dknn_layers = n_layers

    def get_grad(self, xs):
        exs = sequence_embed(self.embed, xs, dropout=0.)
        last_h, last_c, ys = self.encoder(None, None, exs)
        assert(last_h.shape == (self.n_layers, len(xs), self.out_units))
        return last_h[-1], exs

    def __call__(self, xs, dknn=False):
        exs = sequence_embed(self.embed, xs, self.dropout)
        last_h, last_c, ys = self.encoder(None, None, exs)
        assert(last_h.shape == (self.n_layers, len(xs), self.out_units))
        if dknn:
            # if doing deep knn, also return all the LSTM layers
            # last_h: n_layers * (batch_size, n_units)
            return last_h[-1], last_h
        return last_h[-1]


class BiLSTMEncoder(chainer.Chain):

    """A LSTM-RNN Encoder with Word Embedding.

    This model encodes a sentence sequentially using LSTM.

    Args:
        n_layers (int): The number of LSTM layers.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of a LSTM layer and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        super(BiLSTMEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units,
                                   initialW=embed_init)
            self.encoder_forward = L.NStepLSTM(
                    n_layers, n_units, n_units, dropout)
            self.encoder_backward = L.NStepLSTM(
                    n_layers, n_units, n_units, dropout)

        self.n_layers = n_layers
        self.out_units = n_units
        self.dropout = dropout
        self.n_dknn_layers = n_layers

    def get_grad(self, xs):
        exs = sequence_embed(self.embed, xs, dropout=0.)
        fwd_last_h, _, ys = self.encoder_forward(None, None, exs)
        bwd_last_h, _, ys = self.encoder_backward(None, None, exs)

        last_h = F.concat((fwd_last_h, bwd_last_h), axis=2)
        assert(last_h.shape == (self.n_layers, len(xs), 2 * self.out_units))
        return last_h[-1], exs

    def __call__(self, xs, dknn=False):
        exs = sequence_embed(self.embed, xs, self.dropout)
        fwd_last_h, _, ys = self.encoder_forward(None, None, exs)
        bwd_last_h, _, ys = self.encoder_backward(None, None, exs)

        last_h = F.concat((fwd_last_h, bwd_last_h), axis=2)
        assert(last_h.shape == (self.n_layers, len(xs), 2 * self.out_units))
        if dknn:
            # if doing deep knn, also return all the LSTM layers
            # last_h: n_layers * (batch_size, n_units)
            return last_h[-1], last_h
        return last_h[-1]


class CNNEncoder(chainer.Chain):

    """A CNN encoder with word embedding.

    This model encodes a sentence as a set of n-gram chunks
    using convolutional filters.
    Following the convolution, max-pooling is applied over time.
    Finally, the output is fed into a multilayer perceptron.

    Args:
        n_layers (int): The number of layers of MLP.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of MLP and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        out_units = n_units // 3
        super(CNNEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, ignore_label=-1,
                                   initialW=embed_init)
            self.cnn_w3 = L.Convolution2D(
                n_units, out_units, ksize=(3, 1), stride=1, pad=(2, 0),
                nobias=True)
            self.cnn_w4 = L.Convolution2D(
                n_units, out_units, ksize=(4, 1), stride=1, pad=(3, 0),
                nobias=True)
            self.cnn_w5 = L.Convolution2D(
                n_units, out_units, ksize=(5, 1), stride=1, pad=(4, 0),
                nobias=True)
            self.mlp = MLP(n_layers, out_units * 3, dropout)

        self.out_units = out_units * 3
        self.dropout = dropout
        self.n_dknn_layers = self.mlp.n_dknn_layers + 1

    def get_grad(self, xs):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block, dropout=0.)
        h_w3 = F.max(self.cnn_w3(ex_block), axis=2)
        h_w4 = F.max(self.cnn_w4(ex_block), axis=2)
        h_w5 = F.max(self.cnn_w5(ex_block), axis=2)
        h = F.concat([h_w3, h_w4, h_w5], axis=1)
        h = F.relu(h)
        # h = F.dropout(h, ratio=self.dropout)
        return self.mlp(h, no_dropout=True), ex_block

    def __call__(self, xs, dknn=False):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block, self.dropout)
        h_w3 = F.max(self.cnn_w3(ex_block), axis=2)
        h_w4 = F.max(self.cnn_w4(ex_block), axis=2)
        h_w5 = F.max(self.cnn_w5(ex_block), axis=2)
        h = F.concat([h_w3, h_w4, h_w5], axis=1)
        h = F.relu(h)
        h = F.dropout(h, ratio=self.dropout)
        if dknn:
            # return the last CNN hidden followed by MLP hiddens
            output, layers = self.mlp(h, dknn=True)
            return output, [F.squeeze(h, 2)] + layers
        else:
            return self.mlp(h, dknn=False)


class MLP(chainer.ChainList):

    """A multilayer perceptron.

    Args:
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units in a hidden or output layer.
        dropout (float): The dropout ratio.

    """
    def __init__(self, n_layers, n_units, dropout=0.1):
        super(MLP, self).__init__()
        for i in range(n_layers):
            self.add_link(L.Linear(None, n_units))
        self.dropout = dropout
        self.out_units = n_units
        self.n_dknn_layers = n_layers

    def __call__(self, x, dknn=False, no_dropout=False):
        ratio = 0. if no_dropout else self.dropout
        dknn_layers = []
        for i, link in enumerate(self.children()):
            x = F.dropout(x, ratio=ratio)
            x = F.relu(link(x))
            dknn_layers.append(x)
        if dknn:
            return x, dknn_layers
        else:
            return x


class BOWEncoder(chainer.Chain):

    """A BoW encoder with word embedding.

    This model encodes a sentence as just a set of words by averaging.

    Args:
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_vocab, n_units, dropout=0.1):
        super(BOWEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, ignore_label=-1,
                                   initialW=embed_init)

        self.out_units = n_units
        self.dropout = dropout
        self.n_dknn_layers = 1

    def get_grad(self, xs):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block)
        x_len = self.xp.array([len(x) for x in xs], np.int32)[:, None, None]
        h = F.sum(ex_block, axis=2) / x_len
        return h, ex_block

    def __call__(self, xs, dknn=False):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block)
        x_len = self.xp.array([len(x) for x in xs], np.int32)[:, None, None]
        h = F.sum(ex_block, axis=2) / x_len
        if dknn:
            return h, [F.squeeze(h, 2)]
        else:
            return h


class BOWMLPEncoder(chainer.Chain):

    """A BOW encoder with word embedding and MLP.

    This model encodes a sentence as just a set of words by averaging.
    Additionally, its output is fed into a multilayer perceptron.

    Args:
        n_layers (int): The number of layers of MLP.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of MLP and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        super(BOWMLPEncoder, self).__init__()
        with self.init_scope():
            self.bow_encoder = BOWEncoder(n_vocab, n_units, dropout)
            self.mlp_encoder = MLP(n_layers, n_units, dropout)

        self.out_units = n_units
        self.n_dknn_layers = 1 + self.mlp_encoder.n_dknn_layers

    def get_grad(self, xs):
        h, ex_block = self.bow_encoder(xs, dknn=True)
        output = self.mlp_encoder(h, no_dropout=True)
        return output, ex_block

    def __call__(self, xs, dknn=False):
        if dknn:
            h, hs = self.bow_encoder(xs, dknn=True)
            output, dknn_layers = self.mlp_encoder(h, dknn=True)
            return output, hs + dknn_layers
        else:
            return self.mlp_encoder(self.bow_encoder(xs))
