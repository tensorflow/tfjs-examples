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

let tf;

// TODO(cais): Fix.
tf = require('@tensorflow/tfjs-node-gpu');  // TODO(cais): Parameterize.

import {featureColumns, getDatasetStats, getNormalizedDatasets} from './data_housing';

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
  // const optimizer = tf.train.rmsprop(1e-3);
  model.compile({
    loss: 'meanAbsoluteError',
    optimizer: 'adam'
  });
  return model;
}

async function main() {
  const {count, featureMeans, featureStddevs, labelMean, labelStddev} =
      await getDatasetStats();
  const batchSize = 128;
  const {trainXs, trainYs, valXs, valYs} = await getNormalizedDatasets(
      count, featureMeans, featureStddevs, labelMean, labelStddev, batchSize);

  const model = createModel();
  model.summary();

  await model.fit(trainXs, trainYs,  {
    epochs: 500,
    validationData: [valXs, valYs]
  });
}

if (require.main === module) {
  main();
}
