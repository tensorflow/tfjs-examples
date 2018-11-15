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

import * as tfd from '@tensorflow/tfjs-data';

// Boston Housing data constants:
const BASE_URL =
  'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/';

const TRAIN_FILENAME = 'boston-housing-train.csv';
const VALIDATION_FILENAME = 'boston-housing-validation.csv';
const TEST_FILENAME = 'boston-housing-test.csv';

/** Helper class to handle loading training and test data. */
export class BostonHousingDataset {
  constructor() {
    this.trainDataset = null;
    this.validationDataset = null;
    this.testDataset = null;
    this.numFeatures = null;
  }

  static async create() {
    const result = new BostonHousingDataset();
    await result.loadData();
    return result;
  }

  /** Loads training and test data. */
  async loadData() {
    console.log('* Downloading data *');

    const trainData = await this.prepareDataset(`${BASE_URL}${TRAIN_FILENAME}`);
    // Sets number of features so it can be used in the model. Need to exclude
    // the column of label.
    this.trainDataset = trainData.dataset;
    this.numFeatures = trainData.numFeatures;
    this.validationDataset =
      (await this.prepareDataset(`${BASE_URL}${VALIDATION_FILENAME}`))
      .dataset;
    this.testDataset =
      (await this.prepareDataset(`${BASE_URL}${TEST_FILENAME}`)).dataset;
  }

  /**
   * Prepare dataset from provided url.
   */
  async prepareDataset(url) {
    // We want to predict the column "medv", which represents a median value of
    // a home (in $1000s), so we mark it as a label.
    const csvDataset = tfd.csv(url, {
      columnConfigs: {
        medv: {
          isLabel: true
        }
      }
    });

    // Convert rows from object form (keyed by column name) to array form.
    const convertedDataset =
      csvDataset.map(([rawFeatures, rawLabel]) => {
        const convertedFeatures = Object.values(rawFeatures);
        const convertedLabel = Object.values(rawLabel);
        return [convertedFeatures, convertedLabel];
      });

    return {
      dataset: convertedDataset.shuffle(100),
      // Number of features is the number of column names minus one for the
      // label column.
      numFeatures: (await csvDataset.columnNames()).length - 1
    };
  }
}

export const featureDescriptions = [
  'Crime rate', 'Land zone size', 'Industrial proportion', 'Next to river',
  'Nitric oxide concentration', 'Number of rooms per house', 'Age of housing',
  'Distance to commute', 'Distance to highway', 'Tax rate', 'School class size',
  'School drop-out rate'
];
