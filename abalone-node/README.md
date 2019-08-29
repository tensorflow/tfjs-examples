# TensorFlow.js Example: Abalone Age

This example shows how to predicting the age of abalone from physical measurements under Node.js

The data set available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Abalone).

This example shows how to
* load a `Dataset` from a local csv file
* prepare the Dataset for training
* create a `tf.LayersModel` from scratch
* train the model through `model.fitDataset()`
* save the trained model to a local folder.

To launch the demo, do

```sh
yarn
yarn train
```

By default, the training uses tfjs-node, which runs on the CPU.
If you have a CUDA-enabled GPU and have the CUDA and CuDNN libraries
set up properly on your system, you can run the training on the GPU
by replacing the tfjs-node package with tfjs-node-gpu.
