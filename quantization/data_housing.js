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
    console.log(labelMean, labelStddev);  // DEBUG
    return {
      count,
      featureMeans,
      featureStddevs,
      labelMean,
      labelStddev
    };
  });
}

export async function getNormalizedDatasets(
    count, featureMeans, featureStddevs, labelMean, labelStddev, batchSize) {
  const dataset = tf.data.csv(HOUSING_CSV_URL, {
    columnConfigs: {
      [labelColumn]: {
        isLabel: true
      }
    }
  }).shuffle(count).map(data => {
    const xs = data.xs;
    const xsTensor = tf.tensor1d(featureColumns.map(feature =>
        (xs[feature] - featureMeans[feature]) / featureStddevs[feature]));
    const ysTensor =
        tf.tensor1d([(data.ys[labelColumn] - labelMean) / labelStddev]);
    return {
      xs: xsTensor,
      ys: ysTensor
    };
  }).batch(batchSize);

  const numTrain = Math.round(count * 0.7);
  const numVal = count - numTrain;
  const traintDataset = dataset.take(numTrain);
  const valDataset = dataset.take(numVal);
  return {traintDataset, valDataset};
}
