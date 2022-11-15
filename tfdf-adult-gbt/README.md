# Tensorflow Decision Forests Adult GBT Demo

[See this example live!](https://storage.googleapis.com/tfjs-examples/tfdf-adult-gbt/index.html)

## Contents

The demo shows how to use the Tensorflow.JS decision forests package
to run a convereted model.

## Converting Model
1. Create a [Python TensorFlow Decision Forests model](https://www.tensorflow.org/decision_forests).

2. Save the model (will be exported as a SavedModel).

3. Run the model through the [tensorflowjs_converter](https://www.tensorflow.org/js/guide/conversion).
```sh
$ tensorflowjs_converter /path/to/saved_model /path/to/tfjs_model
```

4. Use the [tfjs-tfdf library](https://github.com/tensorflow/tfjs/tree/master/tfjs-tfdf) to run the converted model in the web.

## Demo

The demo in the index.html file is based on the [Javascript library](https://achoum.github.io/yggdrasil_decision_forests_js_example/example.html) and shows the same model predicting the probability a person has income >= $50,000, but using the converted TensorFlow.JS model.
