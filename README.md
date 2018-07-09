# Deep k-Nearest Neighbors and Interpretable NLP

This repository contains the official code for the TODO VENUE paper. It contains code to run Deep k-Nearest Neighbors for a variety of models and architectures.

It also contains code for common interpretation techniques in natural language processing, such as leave one out and gradient based saliency maps. We additionally include code to produce visualizations like the ones seen on our paper's [supplementary website](https://sites.google.com/view/language-dknn/).

## Dependencies

This code is written in python. Dependencies include:

* Python 2/3
* [Chainer](https://chainer.org/)
* tqdm
* numpy

If you want to do efficient nearest neighbor lookup:
* Scikit-Learn (for KDTree)
* nearpy (for locally sensitive hashing)

If you want to visualize saliency maps:
* matplotlib



TODO:
remove combine snli

# TODO
    # 75 neighbors at each layer, or 75 neighbors total?
    # before or after relu?
    # Consider using a different distance than euclidean, cosine?



This code is built off chainers text classification demo.



## Word Vectors

In our paper, we used GloVe word vectors, though any pretrained vectors should work fine (word2vec, fastText, etc.). To obtain GloVe vectors, run the following commands.

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
```

Then pass the pretrained vectors in when training by using the command line argument ```python train_text_classifier.py --word_vectors glove.840B.300d.txt``` 


## References

Please consider citing [1](#dknn-language) if you found this code or our work beneficial to your research.

### TODO title

[1] Eric Wallace\* and Shi Feng\* and Jordan Boyd-Graber, [*TODO title*](PAPER LINK HERE)

```
@article{TODO,
  title={TODO},
  author={TODO},
  journal={TODO},  
  year={2018},  
}
```

(\* These authors contributed equally.)


## Contact

For issues with code or suggested improvements, feel free to open a pull request.

To contact the authors, reach out to Eric Wallace (ewallac2@umd.edu) and Shi Feng (shifeng@cs.umd.edu).
