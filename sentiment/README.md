# TensorFlow.js Example: Sentiment Analysis

This demo shows how to perform text sentiment analysis on text using the Layers
API of TensorFlow.js.

It demonstrates loading a pretrained model hosted at a URL, using
`tf.loadModel()`.

Two model variants are provided (CNN and LSTM).  These were trained on a set of
25,000 movie reviews from IMDB, labelled as having positive or negative
sentiment.  This dataset is
[provided by Python Keras](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification),
and the models were trained in Keras as well, based on the
[imdb_cnn](https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py)
and
[imdb_lstm](https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py)
examples.

To launch the demo, do

```sh
yarn
yarn watch
```

[See this example live!](https://storage.googleapis.com/tfjs-examples/sentiment/dist/index.html)
