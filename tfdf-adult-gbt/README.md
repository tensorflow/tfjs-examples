# Tensorflow Decision Forests Adult GBT Demo

## Contents

The demo shows how to use the Tensorflow.JS decision forests package
to run a convereted model.

## Converting Model
1. Create a Python TensorFlow Decision Forests model (see https://www.tensorflow.
org/decision_forests)

2. Save the model (will be exported as a SavedModel)

3. Run the model through the tensorflowjs_converter (see https://www.tensorflow.org/js/guide/conversion)

4. Use the tfjs-tfdf library to run the converted model in the web (see https://github.com/tensorflow/tfjs/tree/master/tfjs-tfdf)

## Demo

The demo in index.html is based on https://achoum.github.io/yggdrasil_decision_forests_js_example/example.html and shows the same model predicting the probability a person has income >= $50,000, but using the converted TensorFlow.JS model.
