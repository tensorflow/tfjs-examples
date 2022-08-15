/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Create a model for the Fashion-MNIST image classification problem.
 *
 * Based on:
 *   https://github.com/cmasch/zalando-fashion-mnist/blob/master/Simple_Convolutional_Neural_Network_Fashion-MNIST.ipynb
 */
export function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.batchNormalization({
    inputShape: [28, 28, 1]
  }));
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 4,
    padding: 'same',
    activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({poolSize: 2}));
  model.add(tf.layers.dropout({rate: 0.1}));
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 4,
    activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({poolSize: 2}));
  model.add(tf.layers.dropout({rate: 0.3}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units: 256, activation: 'relu'}));
  model.add(tf.layers.dropout({rate: 0.5}));
  model.add(tf.layers.dense({units: 64, activation: 'relu'}));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  compileModel(model);
  return model;
}

export function compileModel(model) {
  const optimizer = 'adam';
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
}
