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

import * as ui from './ui';


// Boston Housing CSV
const BOSTON_HOUSING_CSV_URL =
    'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/train-data.csv'
// Amazon consumer reviews from Kaggle
const PRODUCT_REVIEW_CSV_URL =
    'https://storage.googleapis.com/learnjs-data/csv-datasets/1429_1.csv'
// Banking from Kaggle
const BANKING_CSV_URL =
    'https://storage.googleapis.com/learnjs-data/csv-datasets/banking.csv'
// ML Survey data from Kaggle
// https://www.kaggle.com/kaggle/kaggle-survey-2018#multipleChoiceResponses.csv
const KAGGLE_2018_SURVEY_CSV_URL =
    'https://storage.googleapis.com/learnjs-data/csv-datasets/multipleChoiceResponses.csv'


const countRowsHandler = async () => {
  const url = ui.getQueryElement().value;
  ui.updateStatus(`Attempting to connect to CSV resource at ${url}`);
  const myData = tf.data.csv(url);
  ui.updateStatus('Got the data connection ... counting records in CSV');
  let i = 0;
  ui.updateRowCountOutput(`counting...`);
  const updateFn = (x) => {
    i += 1;
    if ((i % 1000) === 0) {
      ui.updateStatus(`Counting ... ${i} rows of data in the CSV so far...`);
    }
  };
  await myData.forEach((x) => updateFn(x));
  ui.updateStatus(`Done counting rows.`);
  ui.updateRowCountOutput(`Counted ${i} rows of data in the CSV.`);
};

const getColumnNamesHandler = async () => {
  const url = ui.getQueryElement().value;
  ui.updateStatus(`Attempting to connect to CSV resource at ${url}`);
  const myData = tf.data.csv(url);
  ui.updateStatus('Got the data connection ... determining the column names');
  ui.updateColumnNamesOutput('Determining column names.');
  const columnNames = await myData.columnNames();
  ui.updateStatus('Done getting column names.');
  ui.updateColumnNamesOutput(columnNames.join(', '));
};

const getSampleRowHandler = async () => {
  const url = ui.getQueryElement().value;
  ui.updateStatus(`Attempting to connect to CSV resource at ${url}`);
  const myData = tf.data.csv(url);
  ui.updateStatus('Got the data connection ...collecting first sample');
  // const columnNames = await myData.columnNames();
  const sampleIndex = ui.getSampleIndex();
  const sample = await myData.skip(sampleIndex - 1).take(1).collectAll();
  ui.updateStatus(`Done getting sample number #${sampleIndex}.`);
  ui.updateSampleRowMessage(`Done getting sample number #${sampleIndex}.`);
  ui.updateSampleRowOutput(sample[0]);
};

const resetOutputMessages = () => {
  ui.updateRowCountOutput('click "Count rows"');
  ui.updateColumnNamesOutput('click "Get column names"');
  ui.updateSampleRowMessage('select an index and click "Get a sample row"');
  ui.updateSampleRowOutput([]);
};

// Set up handlers
document.addEventListener('DOMContentLoaded', async () => {
  // console.log(tf.version);
  resetOutputMessages();

  // Helper to connect preset URL buttons.
  const connectURLButton = (buttonId, url) => {
    document.getElementById(buttonId).addEventListener('click', async () => {
      ui.getQueryElement().value = url;
      resetOutputMessages();
    }, false);
  };

  connectURLButton('productReviewButton', PRODUCT_REVIEW_CSV_URL);
  connectURLButton('bostonButton', BOSTON_HOUSING_CSV_URL);
  connectURLButton('bankingButton', BANKING_CSV_URL);
  connectURLButton('kaggleSurveyButton', KAGGLE_2018_SURVEY_CSV_URL);

  // Connect action buttons.
  document.getElementById('countRows')
      .addEventListener('click', countRowsHandler, false);
  document.getElementById('getColumnNames')
      .addEventListener('click', getColumnNamesHandler, false);
  document.getElementById('getSampleRow')
      .addEventListener('click', getSampleRowHandler, false);
}, false);
