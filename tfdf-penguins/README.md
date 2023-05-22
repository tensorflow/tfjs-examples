# Tensorflow Decision Forests Penguins Demo

[See this example live!](https://storage.googleapis.com/tfjs-examples/tfdf-penguins/index.html)

## Contents

The demo shows how to use the Tensorflow.js decision forests package
to run a converted model.

## Converting Model

1. Create a [Python TensorFlow Decision Forests model](https://www.tensorflow.org/decision_forests).

2. Save the model (will be exported as a SavedModel).

3. Run the model through the [tensorflowjs_converter](https://www.tensorflow.org/js/guide/conversion).
```sh
$ tensorflowjs_converter /path/to/saved_model /path/to/tfjs_model
```

4. Use the [tfjs-tfdf library](https://github.com/tensorflow/tfjs/tree/master/tfjs-tfdf) to run the converted model in the web.

## Demo

The demo in the index.html file is based on the [SimpleML for Sheets tutorial](https://simplemlforsheets.com/tutorial.html) and shows how to predict the species of a penguin based on other information about it.
