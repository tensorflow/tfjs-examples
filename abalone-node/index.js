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

const tf = require('@tensorflow/tfjs-node-gpu');

/**
 * Builds and returns Multi Layer Perceptron Regression Model.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression mode  l.
 */
function getModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [8],
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

/**
 * Load a local csv file and prepare the data for training. Data source:
 * https://archive.ics.uci.edu/ml/datasets/Abalone
 *
 * @returns {tf.data.CSVDataset} The loaded and prepared Dataset.
 */
function getDataset() {
  const dataset = tf.data.csv(
      'file://abalone.csv',
      {hasHeader: true, columnConfigs: {'rings': {isLabel: true}}});
  // Convert features and labels.
  return dataset
      .map(row => {
        const rawFeatures = row['xs'];
        const rawLabel = row['ys'];
        const convertedFeatures = Object.keys(rawFeatures).map(key => {
          switch (rawFeatures[key]) {
            case 'F':
              return 0;
            case 'M':
              return 1;
            case 'I':
              return 2;
            default:
              return Number(rawFeatures[key]);
          }
        });
        const convertedLabel = [rawLabel['rings']];
        return {xs: convertedFeatures, ys: convertedLabel};
      })
      .shuffle(1000)
      .batch(500);
}

/**
 * Train a model with dataset, then save the model to a local folder.
 */
async function run() {
  const dataset = getDataset();
  const model = getModel();

  await model.fitDataset(dataset, {
    epochs: 100,
    callbacks: {
      onEpochEnd: async (epoch) => {
        console.log(`Epoch ${epoch + 1} of ${100} completed.`);
      }
    }
  });

  await model.save(`file://trainedModel`);
}

run();
