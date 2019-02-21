# TensorFlow.js Example: Sentiment Analysis

This demo shows how to perform text sentiment analysis on text using the Layers
API of TensorFlow.js.

It demonstrates loading a pretrained model hosted at a URL, using
`tf.loadLayersModel()`.

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

## Training your own model in tfjs-node

To train the model using tfjs-node, do

```sh
yarn
yarn train <MODEL_TYPE>
```

where `MODEL_TYPE` is a required argument that specifies what type of model is to be
trained. The available options are:

- `flatten`: A model that flattens the embedding vectors of all words in the sequence.
- `cnn`: A 1D convolutional model.
- `simpleRNN`: A model that uses a SimpleRNN layer (`tf.layers.simpleRNN`)
- `lstm`: A model that uses a LSTM laayer (`tf.layers.lstm`)
- `bidirectionalLSTM`: A model that uses a bidirectional LSTM layer
  (`tf.layers.bidirectional` and `tf.layers.lstm`)

By default, the training happens on the CPU using the Eigen kernels from tfjs-node.
You can make the training happen on GPU by adding the `--gpu` flag to the command, e.g.,

```sh
yarn train --gpu <MODEL_TYPE>
```

The training process will download the training data and metadata form the web
if they haven't been downloaded before. After the model training completes, the model
will be saved to the `dist/resources` folder, alongside a `metadata.json` file.
Then when you run `yarn watch`, you will see a "Load local model" button in the web
page, which allows you to use the locally-trained model for inference in the browser.

Other arguments of the `yarn train` command include:

- `--maxLen` allows you to specify the sequence length.
- `--numWords` allows you to specify the vocabulary size.
- `--embeddingSize` allows you to adjust the dimensionality of the embedding vectors.
- `--epochs`, `--batchSize`, and `--validationSplit` are training-related settings.
- `--modelSavePath` allows you to specify where to store the model and metadata after
  training completes.

The detailed code for training are in the file [train.js](./train.js).
