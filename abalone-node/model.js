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

const tf = require('@tensorflow/tfjs-node');

/**
 * Builds and returns Multi Layer Perceptron Regression Model.
 *
 * @param {number} inputShape The input shape of the model.
 * @returns {tf.Sequential} The multi layer perceptron regression mode  l.
 */
function createModel(inputShape) {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: inputShape,
    activation: 'sigmoid',
    units: 50,
  }));
  model.add(tf.layers.dense({
    activation: 'sigmoid',
    units: 50,
  }));
  model.add(tf.layers.dense({
    units: 1,
  }));
  model.compile({optimizer: tf.train.sgd(0.01), loss: 'meanSquaredError'});
  return model;
}

module.exports = createModel;
