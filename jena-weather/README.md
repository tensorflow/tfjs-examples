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

## Training RNNs

This exapmle shows how to predict temperature using a few different types of
models, including linear regressors, multilayer perceptrons, and recurrent
neural networks (RNNs). While training of the first two types of models
happen in the browser, the training of RNNs happen in Node.js.

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

The training code is in the file [train-rnn.js](train-rnn.js).
