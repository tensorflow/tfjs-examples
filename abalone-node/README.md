# TensorFlow.js Example: Abalone Age

This example shows how to predict the age of abalone from physical measurements using TensorFlow.js with Node.js.

The data set available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Abalone).

This example shows how to:
* load a `Dataset` from a local csv file.
* prepare the Dataset for training.
* create a `tf.LayersModel` from scratch.
* train the model through `model.fitDataset()`.
* save the trained model to a local folder.

To launch the demo, run the following command:

```sh
yarn
yarn train
```

The result logs 100 Epochs as well as a predicted result similar to the following:

```
...
Epoch 100 / 100
eta=0.0 =================================================>
402ms 57414us/step - loss=7.42 val_loss=5.60
The actual test abalone age is 10, the inference result from the model is 11.929240226745605
```

By default, the training uses tfjs-node, which runs on the CPU.
If you have a CUDA-enabled GPU and have the CUDA and CuDNN libraries
set up properly on your system, you can run the training on the GPU
by replacing the tfjs-node package with tfjs-node-gpu.
