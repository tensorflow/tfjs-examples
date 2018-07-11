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
const Papa = require('papaparse');

// Change this to Google cloud link once data is pushed
// Probably would need to add logic to unzip gzip before parsing csv
const BASE_URL =
  'https://gist.githubusercontent.com/ManrajGrover/a4b2b6bf0abda231b4b49af8b9950688/raw/661367f1ab938642ff0d216276b77ace5d288b04/';

async function parseCsv(data) {
  return new Promise(resolve => {
    data = data.map((row) => {
      return Object.keys(row).sort().map(key => parseFloat(row[key]));
    });

    resolve(data);
  });
}

/**
 *  Downloads and returns the csv
 */
export async function loadCsv(filename) {
  return new Promise(resolve => {
    const url = `${BASE_URL}${filename}.csv`;

    console.log(`  * Downloading from: ${url}`);
    Papa.parse(url, {
      download: true,
      header: true,
      complete: (results) => {
        resolve(parseCsv(results['data']));
      }
    })
  });
}

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
 * Calculate the arithmetic mean of a vector
 *
 * @param {Array} vector The vector represented as an Array of Numbers.
 *
 * @returns The arithmetic mean.
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
 * @returns The standard deviation.
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
 * @param {Array} vector Vector to be normalized
 * @param {float} vectorMean Mean to be used
 * @param {float} vectorStddev Standard Deviation to be used
 */
const normalizeVector = (vector, vectorMean, vectorStddev) => {
  return vector.map(x => (x - vectorMean) / vectorStddev);
};

/**
 * Normalizes the dataset
 *
 * @param {Array} dataset Dataset to be normalized
 * @param {Boolean} isTrainData Whether it is training data or not
 * @param {Array} vectorMeans Mean of each column of dataset
 * @param {Array} vectorStddevs Standard deviation of each column of dataset
 *
 * @returns {Object} Contains normalized dataset, mean of each vector column,
 *                   standard deviation of each vector column
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

    return { dataset, vectorMeans, vectorStddevs };
  };
