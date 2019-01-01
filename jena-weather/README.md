# TensorFlow.js Example: Jena Weather

This example illustrates how to visualize time-series data with tfjs-vis.

The data used in this demo is the
[Jena weather archive dataset](https://www.kaggle.com/pankrzysiu/weather-archive-jena).

This demo showcases
- visualization of temporal sequential data with the
  [tfjs-vis](https://www.npmjs.com/package/@tensorflow/tfjs-vis) library
- prediction of future values based on sequential input data using
  various model types including
  - linear regressors
  - multilayer perceptrons (MLPs)
  - recurrent neural networks (RNNs)
- underfitting, overfitting, and various techniques for reducing overfitting, including
  - L2 regularization
  - dropout
  - recurrent dropout
