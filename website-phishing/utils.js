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
import * as tf from '@tensorflow/tfjs';
import * as Papa from 'papaparse';

const BASE_URL =
    'https://gist.githubusercontent.com/ManrajGrover/6589d3fd3eb9a0719d2a83128741dfc1/raw/d0a86602a87bfe147c240e87e6a9641786cafc19/';

/**
 *
 * @param {Array<Object>} data Downloaded data.
 *
 * @returns {Promise.Array<number[]>} Resolves to data with values parsed as floats.
 */
const parseCsv = async (data) => {
  return new Promise(resolve => {
    data = data.map((row) => {
      return Object.keys(row).sort().map(key => parseFloat(row[key]));
    });
    resolve(data);
  });
};

/**
 * Downloads and returns the csv.
 *
 * @param {string} filename Name of file to be loaded.
 *
 * @returns {Promise.Array<number[]>} Resolves to parsed csv data.
 */
export const loadCsv = async (filename) => {
  return new Promise(resolve => {
    const url = `${BASE_URL}${filename}.csv`;

    console.log(`  * Downloading data from: ${url}`);
    Papa.parse(url, {
      download: true,
      header: true,
      complete: (results) => {
        resolve(parseCsv(results['data']));
      }
    })
  });
};

/**
 * Shuffles data and label using Fisher-Yates algorithm.
 */
export const shuffle = (data, label) => {
  let counter = data.length;
  let temp = 0;
  let index = 0;
  while (counter > 0) {
    index = (Math.random() * counter) | 0;
    counter--;
    // data:
    temp = data[counter];
    data[counter] = data[index];
    data[index] = temp;
    // label:
    temp = label[counter];
    label[counter] = label[index];
    label[index] = temp;
  }
};

/**
 * Calculate the arithmetic mean of a vector.
 *
 * @param {Array} vector The vector represented as an Array of Numbers.
 *
 * @returns {number} The arithmetic mean.
 */
const mean = (vector) => {
  let sum = 0;
  for (const x of vector) {
    sum += x;
  }
  return sum / vector.length;
};

/**
 * Calculate the standard deviation of a vector.
 *
 * @param {Array} vector The vector represented as an Array of Numbers.
 *
 * @returns {number} The standard deviation.
 */
const stddev = (vector) => {
  let squareSum = 0;
  const vectorMean = mean(vector);
  for (const x of vector) {
    squareSum += (x - vectorMean) * (x - vectorMean);
  }
  return Math.sqrt(squareSum / (vector.length - 1));
};

/**
 * Normalize a vector by its mean and standard deviation.
 *
 * @param {Array} vector Vector to be normalized.
 * @param {number} vectorMean Mean to be used.
 * @param {number} vectorStddev Standard Deviation to be used.
 *
 * @returns {Array} Normalized vector.
 */
const normalizeVector = (vector, vectorMean, vectorStddev) => {
  return vector.map(x => (x - vectorMean) / vectorStddev);
};

/**
 * Normalizes the dataset
 *
 * @param {Array} dataset Dataset to be normalized.
 * @param {boolean} isTrainData Whether it is training data or not.
 * @param {Array} vectorMeans Mean of each column of dataset.
 * @param {Array} vectorStddevs Standard deviation of each column of dataset.
 *
 * @returns {Object} Contains normalized dataset, mean of each vector column,
 *                   standard deviation of each vector column.
 */
export const normalizeDataset =
    (dataset, isTrainData = true, vectorMeans = [], vectorStddevs = []) => {
      const numFeatures = dataset[0].length;
      let vectorMean;
      let vectorStddev;

      for (let i = 0; i < numFeatures; i++) {
        const vector = dataset.map(row => row[i]);

        if (isTrainData) {
          vectorMean = mean(vector);
          vectorStddev = stddev(vector);

          vectorMeans.push(vectorMean);
          vectorStddevs.push(vectorStddev);
        } else {
          vectorMean = vectorMeans[i];
          vectorStddev = vectorStddevs[i];
        }

        const vectorNormalized =
            normalizeVector(vector, vectorMean, vectorStddev);

        vectorNormalized.forEach((value, index) => {
          dataset[index][i] = value;
        });
      }

      return {dataset, vectorMeans, vectorStddevs};
    };

/**
 * Binarizes a tensor based on threshold.
 *
 * @param {tf.Tensor} y Tensor to be binarized.
 */
const binarize = (y) => {
  const condition = y.greater(tf.scalar(0.5));
  return tf.where(condition, tf.onesLike(y), tf.zerosLike(y));
};

/**
 * Builds and returns Confusion Matrix.
 *
 * @param {tf.Tensor} y Tensor containing actual target.
 * @param {tf.Tensor} yPred Tensor containing predicted probabilities.
 */
export const getConfusionMatrix = (y, yPred) => {
  // [[TN, FP],
  //  [FN, TP]]
  const confusionMatrix = [[0, 0], [0, 0]];

  const yData = y.dataSync();
  const yPredData = binarize(yPred).dataSync();

  for (let index = 0; index < yData.length; ++index) {
    confusionMatrix[yData[index]][yPredData[index]] += 1;
  }

  return confusionMatrix;
};

/**
 * Calculates and returns precision score.
 *
 * @param {Array<number[]>} confusionMatrix Confusion Matrix.
 */
export const getPrecisionScore = (confusionMatrix) => {
  return confusionMatrix[1][1] /
      (confusionMatrix[1][1] + confusionMatrix[0][1]);
};

/**
 * Calculates and returns recall score.
 *
 * @param {Array<number[]>} confusionMatrix Confusion Matrix.
 */
export const getRecallScore = (confusionMatrix) => {
  return confusionMatrix[1][1] /
      (confusionMatrix[1][1] + confusionMatrix[1][0]);
};
