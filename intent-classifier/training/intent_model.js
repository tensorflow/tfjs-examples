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
 * Returns a new ready-to-train tf.Model that classifies 512 dimensional vectors
 * into one of labels.length categories.
 *
 * The intended use is for the inputs to be embeddings from the universal
 * sentence encoder and labels to be the categories we want to classify those
 * sentence into.
 *
 * @param {string[]} labels
 *
 * @return {tf.Model} the model instance
 */
function getModel(labels) {
  const NUM_CLASSES = labels.length;
  const EMBEDDING_DIMS = 512;

  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [EMBEDDING_DIMS],
    units: NUM_CLASSES,
    activation: 'softmax',
  }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

module.exports = {
  getModel,
};
