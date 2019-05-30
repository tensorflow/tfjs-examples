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

const HOUSING_CSV_URL = 'https://storage.googleapis.com/learnjs-data/csv-datasets/california_housing_train_10k.csv';

export const featureColumns = [
  'longitude', 'latitude', 'housing_median_age', 'total_rooms',
  'total_bedrooms', 'population', 'households',  'median_income'];
const labelColumn = 'median_house_value';

/**
 * Calculate the column-by-column statistics of the housing CSV dataset.
 *
 * @return An object consisting of the following fields:
 *   count {number} Number of data rows.
 *   featureMeans {number[]} Each element is the arithmetic mean over all values
 *     in a column. Ordered by the feature columns in the CSV dataset.
 *   featureStddevs {number[]} Each element is the standard deviation over all
 *     values in a column. Ordered by the columsn in the in the CSV dataset.
 *   labelMean {number} The arithmetic mean of the label column.
 *   labeStddev {number} The standard deviation of the albel column.
 */
export async function getDatasetStats() {
  const featureValues = {};
  featureColumns.forEach(feature => {
    featureValues[feature] = [];
  });
  const labelValues = [];

  const dataset = tf.data.csv(HOUSING_CSV_URL, {
    columnConfigs: {
      [labelColumn]: {
        isLabel: true
      }
    }
  });
  const iterator = await dataset.iterator();
  let count = 0;
  while (true) {
    const item = await iterator.next();
    if (item.done) {
      break;
    }
    featureColumns.forEach(feature => {
      if (item.value.xs[feature] == null) {
        throw new Error(`item #{count} lacks feature ${feature}`);
      }
      featureValues[feature].push(item.value.xs[feature]);
    });
    labelValues.push(item.value.ys[labelColumn]);
    count++;
  }

  return tf.tidy(() => {
    const featureMeans = {};
    const featureStddevs = {};
    featureColumns.forEach(feature => {
      const {mean, variance} = tf.moments(featureValues[feature]);
      featureMeans[feature] = mean.arraySync();
      featureStddevs[feature] = tf.sqrt(variance).arraySync();
    });

    const moments = tf.moments(labelValues);
    const labelMean = moments.mean.arraySync();
    const labelStddev = tf.sqrt(moments.variance).arraySync();
    return {
      count,
      featureMeans,
      featureStddevs,
      labelMean,
      labelStddev
    };
  });
}

/**
 * Get a dataset with the features and label z-normalized,
 * the dataset is split into three xs-ys tensor pairs: for training,
 * validation and evaluation.
 *
 * @param {number} count Number of rows in the CSV dataset, computed beforehand.
 * @param {{[feature: string]: number}} featureMeans Arithmetic means of the
 *   features. Use for normalization.
 * @param {[feature: string]: number} featureStddevs Standard deviations of the
 *   features. Used for normalization.
 * @param {number} labelMean Arithmetic mean of the label. Used for
 *   normalization.
 * @param {number} labelStddev Standard deviation of the label. Used for
 *   normalization.
 * @param {number} validationSplit Validation spilt, must be >0 and <1.
 * @param {number} evaluationSplit Evaluation split, must be >0 and <1.
 * @returns An object consisting of the following keys:
 *   trainXs {tf.Tensor} training feature tensor
 *   trainYs {tf.Tensor} training label tensor
 *   valXs {tf.Tensor} validation feature tensor
 *   valYs {tf.Tensor} validation label tensor
 *   evalXs {tf.Tensor} evaluation feature tensor
 *   evalYs {tf.Tensor} evaluation label tensor.
 */
export async function getNormalizedDatasets(
    count, featureMeans, featureStddevs, labelMean, labelStddev,
    validationSplit, evaluationSplit) {
  tf.util.assert(
      validationSplit > 0 && validationSplit < 1,
      () => `validationSplit is expected to be >0 and <1, ` +
            `but got ${validationSplit}`);
  tf.util.assert(
      evaluationSplit > 0 && evaluationSplit < 1,
      () => `evaluationSplit is expected to be >0 and <1, ` +
            `but got ${evaluationSplit}`);
  tf.util.assert(
      validationSplit + evaluationSplit < 1,
      () => `The sum of validationSplit and evaluationSplit exceeds 1`);

  const dataset = tf.data.csv(HOUSING_CSV_URL, {
    columnConfigs: {
      [labelColumn]: {
        isLabel: true
      }
    }
  });

  const featureValues = [];
  const labelValues = [];
  const indices = [];
  const iterator = await dataset.iterator();
  for (let i = 0; i < count; ++i) {
    const {value, done} = await iterator.next();
    if (done) {
      break;
    }
    featureColumns.map(feature => {
      featureValues.push(
          (value.xs[feature] - featureMeans[feature]) /
          featureStddevs[feature]);
    });
    labelValues.push((value.ys[labelColumn] - labelMean) / labelStddev);
    indices.push(i);
  }

  const xs = tf.tensor2d(featureValues, [count, featureColumns.length]);
  const ys = tf.tensor2d(labelValues, [count, 1]);

  // Set random seed to fix shuffling order and therefore to fix the
  // training, validation, and evaluation splits.
  Math.seedrandom('1337');
  tf.util.shuffle(indices);

  const numTrain = Math.round(count * (1 - validationSplit - evaluationSplit));
  const numVal = Math.round(count * validationSplit);
  const trainXs = xs.gather(indices.slice(0, numTrain));
  const trainYs = ys.gather(indices.slice(0, numTrain));
  const valXs = xs.gather(indices.slice(numTrain, numTrain + numVal));
  const valYs = ys.gather(indices.slice(numTrain, numTrain + numVal));
  const evalXs = xs.gather(indices.slice(numTrain + numVal));
  const evalYs = ys.gather(indices.slice(numTrain + numVal));

  return {trainXs, trainYs, valXs, valYs, evalXs, evalYs};

}
