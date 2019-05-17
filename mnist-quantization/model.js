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

export function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dropout({rate: 0.25}));
  model.add(tf.layers.dense({units: 512, activation: 'relu'}));
  model.add(tf.layers.dropout({rate: 0.5}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  compileModel(model);
  return model;
}

export function compileModel(model) {
  const optimizer = 'rmsprop';
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
}
