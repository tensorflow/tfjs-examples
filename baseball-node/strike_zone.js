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

const tf = require('@tensorflow/tfjs');
const normalize = require('./utils').normalize;

const TRAIN_DATA_PATH =
    'https://storage.googleapis.com/mlb-pitch-data/strike_zone_training_data.csv';
const TEST_DATA_PATH =
    'https://storage.googleapis.com/mlb-pitch-data/strike_zone_test_data.csv';

// Constants from training data:
const PX_MIN = -2.65170604056843;
const PX_MAX = 2.842899614;
const PZ_MIN = -2.01705841594049;
const PZ_MAX = 6.06644249133382;
const SZ_TOP_MIN = 2.85;
const SZ_TOP_MAX = 4.241794863019148;
const SZ_BOT_MIN = 1.248894636863092;
const SZ_BOT_MAX = 2.2130980270561516;

const TRAINING_DATA_LENGTH = 10000;
const TEST_DATA_LENGTH = 200;

// Converts a row from the CSV into features and labels.
// Each feature field is normalized within training data constants:
const csvTransform = ({xs, ys}) => {
  const values = [
    normalize(xs.px, PX_MIN, PX_MAX), normalize(xs.pz, PZ_MIN, PZ_MAX),
    normalize(xs.sz_top, SZ_TOP_MIN, SZ_TOP_MAX),
    normalize(xs.sz_bot, SZ_BOT_MIN, SZ_BOT_MAX), xs.left_handed_batter
  ];
  return {xs: values, ys: ys.is_strike};
};

const trainingData =
    tf.data.csv(TRAIN_DATA_PATH, {columnConfigs: {is_strike: {isLabel: true}}})
        .map(csvTransform)
        .shuffle(TRAINING_DATA_LENGTH)
        .batch(50);

const testValidationData =
    tf.data.csv(TEST_DATA_PATH, {columnConfigs: {is_strike: {isLabel: true}}})
        .map(csvTransform)
        .batch(TEST_DATA_LENGTH);

const model = tf.sequential();
model.add(tf.layers.dense({units: 20, activation: 'relu', inputShape: [5]}));
model.add(tf.layers.dense({units: 10, activation: 'relu'}));
model.add(tf.layers.dense({units: 2, activation: 'softmax'}));
model.compile({
  optimizer: tf.train.adam(),
  loss: 'sparseCategoricalCrossentropy',
  metrics: ['accuracy']
});

module.exports = {
  model,
  testValidationData,
  trainingData,
  TEST_DATA_LENGTH
};
