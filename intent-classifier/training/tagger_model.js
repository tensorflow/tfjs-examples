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
 * Return a tf.Model to tag tokens in an input sequence.
 *
 * @param {Object} opts
 * @return {tf.Model} the model instance
 */
function getModel(opts) {
  const {embeddingDims, sequenceLength, modelType, numLabels, weights} = opts;

  const model = tf.sequential();
  model.add(tf.layers.inputLayer({
    inputShape: [sequenceLength, embeddingDims],
  }));

  // This function can return one of three different model architectures:
  //  'lstm': A unidirectional (one layer) LSTM
  //  'bidirectional-lstm': A bidirectional (one layer) LSTM
  //  'dense': A single layer dense network. This can be a baseline.

  let lstmLayer;
  switch (modelType) {
    case 'lstm':
      lstmLayer = tf.layers.lstm({
        units: sequenceLength,
        recurrentInitializer: 'glorotNormal',
        returnSequences: true,
      });
      break;
    case 'bidirectional-lstm':
      lstmLayer = tf.layers.lstm({
        units: sequenceLength,
        recurrentInitializer: 'glorotNormal',
        returnSequences: true,
      });
      lstmLayer = tf.layers.bidirectional({layer: lstmLayer});
      break;
  }

  if (modelType.match('lstm')) {
    model.add(lstmLayer);
    const dense = tf.layers.dense({units: numLabels});
    model.add(tf.layers.timeDistributed({layer: dense}));
    model.add(tf.layers.activation({activation: 'softmax'}));
  } else {
    const dense = tf.layers.dense({units: numLabels, activation: 'softmax'});
    model.add(dense);
  }

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
    // Optionally weight misclassifications for different tags differently.
    // This allows us to prioritize getting certain classes correct.
    classWeight: weights,
  });

  model.summary();

  return model;
}


module.exports = {getModel};
