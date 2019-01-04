# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Train a simple model for the Iris dataset; Export the model and weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import keras
import numpy as np
import tensorflowjs as tfjs

import iris_data


def train(epochs,
          artifacts_dir,
          sequential=False):
  """Train a Keras model for Iris data classification and save result as JSON.

  Args:
    epochs: Number of epochs to traing the Keras model for.
    artifacts_dir: Directory to save the model artifacts (model topology JSON,
      weights and weight manifest) in.
    sequential: Whether to use a Keras Sequential model, instead of the default
      functional model.

  Returns:
    Final classification accuracy on the training set.
  """
  data_x, data_y = iris_data.load()

  if sequential:
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        10, input_shape=[data_x.shape[1]], use_bias=True, activation='sigmoid',
        name='Dense1'))
    model.add(keras.layers.Dense(
        3, use_bias=True, activation='softmax', name='Dense2'))
  else:
    iris_x = keras.layers.Input((4,))
    dense1 = keras.layers.Dense(
        10, use_bias=True, name='Dense1', activation='sigmoid')(iris_x)
    dense2 = keras.layers.Dense(
        3, use_bias=True, name='Dense2', activation='softmax')(dense1)
    # pylint:disable=redefined-variable-type
    model = keras.models.Model(inputs=[iris_x], outputs=[dense2])
    # pylint:enable=redefined-variable-type
  model.compile(loss='categorical_crossentropy', optimizer='adam')

  model.fit(data_x, data_y, batch_size=8, epochs=epochs)

  # Run prediction on the training set.
  pred_ys = np.argmax(model.predict(data_x), axis=1)
  true_ys = np.argmax(data_y, axis=1)
  final_train_accuracy = np.mean((pred_ys == true_ys).astype(np.float32))
  print('Accuracy on the training set: %g' % final_train_accuracy)

  tfjs.converters.save_keras_model(model, artifacts_dir)

  return final_train_accuracy


def main():
  train(FLAGS.epochs, FLAGS.artifacts_dir, sequential=FLAGS.sequential)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Iris model training and serialization')
  parser.add_argument(
      '--sequential',
      action='store_true',
      help='Use a Keras Sequential model, instead of the default functional '
      'model.')
  parser.add_argument(
      '--epochs',
      type=int,
      default=100,
      help='Number of epochs to train the Keras model for.')
  parser.add_argument(
      '--artifacts_dir',
      type=str,
      default='/tmp/iris.keras',
      help='Local path for saving the TensorFlow.js artifacts.')

  FLAGS, _ = parser.parse_known_args()
  main()
