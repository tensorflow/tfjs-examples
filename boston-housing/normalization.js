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
 * Calculates the mean and standard deviation of each column of a data array.
 *
 * @param {Array} dataset Dataset from which to calculate the mean and std.
 *
 * @returns {Object} Contains the mean of each vector column,
 *                   standard deviation of each vector column.
 */
export const determineMeanAndStd =
    (dataset) => {
      const numFeatures = dataset[0].length;
      let vectorMeans = [];
      let vectorStddevs = [];
      for (let i = 0; i < numFeatures; i++) {
        const vector = dataset.map(row => row[i]);
        vectorMeans.push(mean(vector));
        vectorStddevs.push(stddev(vector));
      }
      return {vectorMeans, vectorStddevs};
    }

/**
 * Normalizes the dataset to zero-mean, unit standard deviation by subtracting
 * the supplied mean and dividing by the supplied standard deviation.
 *
 * @param {Array} dataset Dataset to be normalized.
 * @param {Array} vectorMeans Mean of each column of dataset.
 * @param {Array} vectorStddevs Standard deviation of each column of dataset.
 *
 * @returns {Array} Normalized dataset.
 */
export const normalizeDataset = (dataset, vectorMeans, vectorStddevs) => {
  const numFeatures = dataset[0].length;
  for (let i = 0; i < numFeatures; i++) {
    const vector = dataset.map(row => row[i]);
    const vectorNormalized =
        normalizeVector(vector, vectorMeans[i], vectorStddevs[i]);

    vectorNormalized.forEach((value, index) => {
      dataset[index][i] = value;
    });
  }
  return dataset;
};
