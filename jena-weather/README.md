# TensorFlow.js Example: Jena Weather

This demo showcases
- visualization of temporal sequential data with the
  [tfjs-vis](https://www.npmjs.com/package/@tensorflow/tfjs-vis) library
- prediction of future values based on sequential input data using
  various model types including
  - linear regressors
  - multilayer perceptrons (MLPs)
  - recurrent neural networks (RNNs, to be added)
- underfitting, overfitting, and various techniques for reducing overfitting, including
  - L2 regularization
  - dropout
  - recurrent dropout (to be added)

The data used in this demo is the
[Jena weather archive dataset](https://www.kaggle.com/pankrzysiu/weather-archive-jena).

This example also showcases the usage of the following important APIs in
TensorFlow.js

- `tf.data.generator()`: How to create `tf.data.Dataset` objects from generator
  functions.
- `tf.Model.fitDataset()`: How to use a `tf.data.Dataset` object to train a
  `tf.Model` and use another `tf.data.Dataset` object to perform validation
  of the model at the end of every training epoch.
- `tfvis.show.fitCallbacks()`: How to use the convenient method to plot
  training-set and validation-set losses at the end of batches and epochs of
  model training.

## Training RNNs

This example shows how to predict temperature using a few different types of
models, including linear regressors, multilayer perceptrons, and recurrent
neural networks (RNNs). While training of the first two types of models
happens in the browser, the training of RNNs is conducted in Node.js, due to
their heavier computational load and longer training time.

For example, to train a gated recurrent unit (GRU) model, use shell commands:

```sh
yarn
yarn train-rnn
```

By default, the training happens on the CPU using the Eigen ops from tfjs-node.
If you have a CUDA-enabled GPU and the necessary drivers and libraries (CUDA and
CuDNN) installed, you can train the model using the CUDA/CuDNN ops from
tfjs-node-gpu. For that, just add the `--gpu` flag:

```sh
yarn
yarn train-rnn --gpu
```

You can also calculate the prediction error (mean absolute error) based on a
commonsense baseline method that is not machine learning: just predict the
temperature as the latest temperature data point in the input features.
This can be done with the dummy `--modelType` flag value `baseline`, i.e.,

```sh
yarn
yarn train-rnn --modelType baseline
```

The training code is in the file [train-rnn.js](./train-rnn.js).
