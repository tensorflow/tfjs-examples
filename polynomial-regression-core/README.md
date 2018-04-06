# TensorFlow.js Example: Fitting a curve to synthetic data

This example shows you how to use TensorFlow.js operations and optimizers (the lower level api) to write a simple model that learns the coefficients of polynomial that we want to use to describe our data. In this toy example, we generate synthetic data by adding some noise to a polynomial function. Then starting with random coefficients, we train a model to learn the true coefficients that data was generated with.

[See this example live!](https://storage.googleapis.com/tfjs-examples/polynomial-regression-core/dist/index.html)
