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
tf = require('@tensorflow/tfjs-node');

import {featureColumns, getDatasetStats, getNormalizedDatasets} from './data_housing';

export function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 8,
    activation: 'relu',
    inputShape: [featureColumns.length]
  }));
  model.add(tf.layers.dense({
    units: 8,
    activation: 'relu',
  }));
  model.add(tf.layers.dense({units: 1}));
  const optimizer = tf.train.sgd(0.02);
  model.compile({
    loss: 'meanAbsoluteError',
    optimizer
  });
  return model;
}

async function main() {
  const {count, featureMeans, featureStddevs, labelMean, labelStddev} =
      await getDatasetStats();
  const batchSize = 32;
  const {traintDataset, valDataset} = await getNormalizedDatasets(
      count, featureMeans, featureStddevs, labelMean, labelStddev, batchSize);

  const model = createModel();
  model.summary();

  await model.fitDataset(traintDataset,  {
    epochs: 10,
    validationData: valDataset
  });
  // console.log(normalizedDataset);  // DEBUG
  // const iterator = await normalizedDataset.iterator();
  // const item = await iterator.next();
  // item.value.ys.print();

  // console.log(item.value.xs);
  // console.log(item.value.ys);
  // const moments = tf.moments(item.value.xs);
  // moments.mean.print();
  // moments.variance.print();
}

if (require.main === module) {
  main();
}
