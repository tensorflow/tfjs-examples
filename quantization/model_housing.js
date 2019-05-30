/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {featureColumns} from './data_housing';

export function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 80,
    activation: 'relu',
    inputShape: [featureColumns.length]
  }));
  model.add(tf.layers.dropout({rate: 0.2}));
  model.add(tf.layers.dense({
    units: 120,
    activation: 'relu',
  }));
  model.add(tf.layers.dropout({rate: 0.1}));
  model.add(tf.layers.dense({
    units: 20,
    activation: 'relu',
  }));
  model.add(tf.layers.dropout({rate: 0.1}));
  model.add(tf.layers.dense({
    units: 10,
    activation: 'relu',
  }));
  model.add(tf.layers.dropout({rate: 0.1}));
  model.add(tf.layers.dense({units: 1}));
  compileModel(model);
  return model;
}

export function compileModel(model) {
  model.compile({
    loss: 'meanAbsoluteError',
    optimizer: 'adam'
  });
}