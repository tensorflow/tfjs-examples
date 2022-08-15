# TensorFlow.js Example: Train LSTM to Generate Text

[See this example live!](https://storage.googleapis.com/tfjs-examples/lstm-text-generation/dist/index.html)

## Overview

This example illustrates how to use TensorFlow.js to train a LSTM model to
generate random text based on the patterns in a text corpus such as
Nietzsche's writing or the source code of TensorFlow.js itself.

The LSTM model operates at the character level. It takes a tensor of
shape `[numExamples, sampleLen, charSetSize]` as the input. The input is a
one-hot encoding of sequences of `sampleLen` characters. The characters
belong to a set of `charSetSize` unique characters. With the input, the model
outputs a tensor of shape `[numExamples, charSetSize]`, which represents the
model's predicted probabilites of the character that follows the input sequence.
The application then draws a random sample based on the predicted
probabilities to get the next character. Once the next character is obtained,
its one-hot encoding is concatenated with the previous input sequence to form
the input for the next time step. This process is repeated in order to generate
a character sequence of a given length. The randomness (diversity) is controlled
by a temperature parameter.

The UI allows creation of models consisting of a single
[LSTM layer](https://js.tensorflow.org/api/latest/#layers.lstm) or multiple,
stacked LSTM layers.

This example also illustrates how to save a trained model in the browser's
IndexedDB using TensorFlow.js's
[model saving API](https://js.tensorflow.org/tutorials/model-save-load.html),
so that the result of the training
may persist across browser sessions. Once a previously-trained model is loaded
from the IndexedDB, it can be used in text generation and/or further training.

This example is inspired by the LSTM text generation example from Keras:
https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

## Usage

### Running the Web Demo

The web demo supports model training and text generation. To launch the demo, do:

```sh
yarn && yarn watch
```

### Training Models in Node.js

Training a model in Node.js should give you a faster performance than the browser
environment.

To start a training job, enter command lines such as:

```sh
yarn
yarn train shakespeare \
    --lstmLayerSize 128,128 \
    --epochs 120 \
    --savePath ./my-shakespeare-model
```

- The first argument to `yarn train` (`shakespeare`) specifies what text corpus
  to train the model on. See the console output of `yarn train --help` for a set
  of supported text data. You can also provide the path to a file containing
  your own text corpus.
- The argument `--lstmLayerSize 128,128` specifies that the next-character
  prediction model should contain two LSTM layers stacked on top of each other,
  each with 128 units.
- The flag `--epochs` is used to specify the number of training epochs.
- The argument `--savePath ...` lets the training script save the model at the
  specified path once the training completes

If you have a CUDA-enabled GPU set up properly on your system, you can
add the `--gpu` flag to the command line to train the model on the GPU, which
should give you a further performance boost.

### Generating Text in Node.js using Saved Model Files

The example command line above generates a set of model files in the
`./my-shakespeare-model` folder after the completion of the training. You can
load the model and use it to generate text. For example:

```sh
yarn gen shakespeare ./my-shakespeare-model/model.json \
    --genLength 250 \
    --temperature 0.6
```

The command will randomly sample a snippet of text from the shakespeare
text corpus and use it as the seed to generate text.

- The first argument (`shakespeare`) specifies the text corpus.
- The second argument specifies the path to the saved JSON file for the
  model, which has been generated in the previous section.
- The `--genLength` flag allows you to speicify how many characters
  to generate.
- The `--temperature` flag allows you to specify the stochacity (randomness)
  of the generation processs. It should be a number greater than or equal to
  zero. The higher the value is, the more random the generated text will be.
