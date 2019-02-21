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
    'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv';
// Jena Climate CSV
const JENA_CLIMATE_CSV_URL =
    'https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv';
// Dresses Sales data
// Originally from https://www.openml.org/d/23381
const DRESSES_SALES_CSV_URL =
    'https://storage.googleapis.com/learnjs-data/csv-datasets/dresses-sales-openml.csv';
// State University of New York Campus Data from NYS.gov
// Originally from
// https://data.ny.gov/Education/State-University-of-New-York-SUNY-Campus-Locations/3cij-nwhw
const SUNY_CSV_URL =
    'https://storage.googleapis.com/learnjs-data/csv-datasets/State_University_of_New_York__SUNY__Campus_Locations_with_Websites__Enrollment_and_Select_Program_Offerings.csv';


/**
 * Builds a CSV Dataset object using the URL specified in the UI.  Then iterates
 * over all eleemnts in that dataset to count them.  Updates the UI accordingly.
 */
async function countRowsHandler() {
  const url = ui.getQueryElement().value;
  ui.updateStatus(`Building data object to connect to ${url}`);
  const myData = tf.data.csv(url);
  let i = 0;
  ui.updateRowCountOutput(`counting...`);
  const updateFn = x => {
    i += 1;
    if (i % 1000 === 0) {
      ui.updateStatus(`Counting ... ${i} rows of data in the CSV so far...`);
    }
  };
  try {
    ui.updateStatus('Attempting to count records in CSV.');
    // Note that `tf.data.Dataset.forEachAsync()` is an async function.  Without
    // the `await` here, there is no control over when the updataFn's will
    // execute, thus, they will likely execute *after* we update the status with
    // the final count, resulting in a display of "Counted 0 rows.".
    await myData.forEachAsync(x => updateFn(x));
  } catch (e) {
    const errorMsg = `Caught an error iterating over ${url}.  ` +
        `This URL might not be valid or might not support CORS requests.` +
        `  Check the developer console for CORS errors.` + e;
    ui.updateRowCountOutput(errorMsg);
    return;
  }
  ui.updateStatus(`Done counting rows.`);
  ui.updateRowCountOutput(`Counted ${i} rows of data in the CSV.`);
};

/**
 * Builds a CSV Dataset object using the URL specified in the UI.  Then connects
 * with the dataset object to retrieve the column names.  Updates the UI
 * accordingly.
 */
async function getColumnNamesHandler() {
  ui.updateColumnNamesOutput([]);
  const url = ui.getQueryElement().value;
  ui.updateStatus(`Attempting to connect to CSV resource at ${url}`);
  const myData = tf.data.csv(url);
  ui.updateStatus('Got the data connection ... determining the column names');
  ui.updateColumnNamesMessage('Determining column names.');
  try {
    const columnNames = await myData.columnNames();
    ui.updateStatus('Done getting column names.');
    ui.updateColumnNamesMessage('');
    ui.updateColumnNamesOutput(columnNames);
  } catch (e) {
    const errorMsg = `Caught an error retrieving column names from ${url}.  ` +
        `This URL might not be valid or might not support CORS requests.` +
        `  Check the developer console for CORS errors.` + e;
    ui.updateColumnNamesMessage(errorMsg);
    return;
  }
};

/**
 * Accesses the CSV to collect a single specified row.  The row index
 * is specified by the UI element managed in the ui library.
 */
async function getSampleRowHandler() {
  const url = ui.getQueryElement().value;
  ui.updateStatus(`Attempting to connect to CSV resource at ${url}`);
  const myData = tf.data.csv(url);
  ui.updateStatus('Got the data connection ... getting requested sample');
  const sampleIndex = ui.getSampleIndex();
  if (sampleIndex < 0 || isNaN(sampleIndex)) {
    const msg = `Can not get samples with negative or NaN index.  (Requested ${
        sampleIndex}).`;
    ui.updateStatus(msg);
    ui.updateSampleRowMessage(msg);
    ui.updateSampleRowOutput([]);
    return;
  }
  let sample;
  try {
    sample = await myData.skip(sampleIndex).take(1).toArray();
  } catch (e) {
    let errorMsg = `Caught an error retrieving sample from ${url}.  `;
    errorMsg +=
        'This URL might not be valid or might not support CORS requests.';
    errorMsg += '  Check the developer console for CORS errors.';
    errorMsg += e;
    ui.updateSampleRowMessage(errorMsg);
    return;
  }
  if (sample.length === 0) {
    // When samples are requested beyond the end of the CSV, the data will
    // return empty.
    const msg = `Can not get sample index ${
        sampleIndex}.  This may be beyond the end of the dataset.`;
    ui.updateStatus(msg);
    ui.updateSampleRowMessage(msg);
    ui.updateSampleRowOutput([]);
    return;
  }
  ui.updateStatus(`Done getting sample ${sampleIndex}.`);
  ui.updateSampleRowMessage(`Done getting sample ${sampleIndex}.`);
  ui.updateSampleRowOutput(sample[0]);
};

/** Clears output messages and tables. */
const resetOutputMessages = () => {
  ui.updateRowCountOutput('click "Count rows"');
  ui.updateColumnNamesMessage('click "Get column names"');
  ui.updateColumnNamesOutput([]);
  ui.updateSampleRowMessage('select an index and click "Get a sample row"');
  ui.updateSampleRowOutput([]);
};

/** Sets up handlers for the user affordences, including all buttons. */
document.addEventListener('DOMContentLoaded', async () => {
  resetOutputMessages();

  // Helper to connect preset URL buttons.
  const connectURLButton = (buttonId, url, statusMessage) => {
    document.getElementById(buttonId).addEventListener('click', async () => {
      ui.getQueryElement().value = url;
      resetOutputMessages();
      ui.updateStatus(statusMessage);
    }, false);
  };

  connectURLButton(
      'jena-climate-button', JENA_CLIMATE_CSV_URL,
      `Jena climate data is a record of atmospheric conditions taken over a ` +
          `period of time.  In this dataset, 14 different quantities (such ` +
          `as air temperature, atmospheric pressure, humidity, wind ` +
          `direction, and so on) were recorded every 10 minutes, over ` +
          `several years.  Note that counting all the rows of this dataset` +
          `might take a while`);
  connectURLButton(
      'boston-button', BOSTON_HOUSING_CSV_URL,
      `"Boston Housing" is a commonly used dataset in introductory ML problems.`);
  connectURLButton(
      'dresses-button', DRESSES_SALES_CSV_URL,
      `This dataset contains attributes of dresses and their recommendations ` +
          `according to their sales. Provided courtesy of OpenML. Find more ` +
          `curated ML datasets at https://www.openml.org/d/23381`);
  connectURLButton(
      'suny-button', SUNY_CSV_URL,
      `Campuses which comprise the State University of New York (SUNY) System. ` +
          `Highlights information on Undergraduate and Graduate enrollment ` +
          `as well as some program area offerings.  Find more datasets at ` +
          `https://data.ny.gov/`);

  // Connect action buttons.
  document.getElementById('count-rows')
      .addEventListener('click', countRowsHandler, false);
  document.getElementById('get-column-names')
      .addEventListener('click', getColumnNamesHandler, false);
  document.getElementById('get-sample-row')
      .addEventListener('click', getSampleRowHandler, false);

  // Connect sample index to fetch on change.
  document.getElementById('which-sample-input')
      .addEventListener('change', getSampleRowHandler, false);
}, false);
