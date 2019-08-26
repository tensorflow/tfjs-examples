# TensorFlow.js Example: Abalone Age

This example shows how to predicting the age of abalone from physical measurements under Node.js

The data set available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Abalone).

This example shows how to
* load a `CSVDataset` from a local csv file
* prepare the Dataset for training
* create a `tf.LayersModel` from scratch
* train the model through `model.fitDataset()`
* save the trained model to a local folder.

To launch the demo, do

```sh
yarn
yarn start
```
