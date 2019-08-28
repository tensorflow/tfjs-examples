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
 * Load a local csv file and prepare the data for training. Data source:
 * https://archive.ics.uci.edu/ml/datasets/Abalone
 *
 * @param {string} csvPath The path to csv file.
 * @returns {tf.data.Dataset} The loaded and prepared Dataset.
 */
async function createDataset(csvPath) {
  const dataset = tf.data.csv(
      csvPath, {hasHeader: true, columnConfigs: {'rings': {isLabel: true}}});
  const numOfColumns = (await dataset.columnNames()).length - 1;
  // Convert features and labels.
  return {
    dataset: dataset.map(row => {
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
    }),
    numOfColumns: numOfColumns
  };
}

module.exports = createDataset;
