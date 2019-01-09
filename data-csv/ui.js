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

const statusElement = document.getElementById('status');
const rowCountOutputElement = document.getElementById('rowCountOutput');
const columnNamesOutputElement = document.getElementById('columnNamesOutput');
const sampleRowMessageElement = document.getElementById('sampleRowMessage');
const sampleRowOutputContainerElement =
    document.getElementById('sampleRowOutputContainer');
const whichSampleInputElement = document.getElementById('whichSampleInput');

export const updateStatus = (message) => {
  console.log(message);
  statusElement.value = message;
};

export const updateRowCountOutput = (message) => {
  console.log(message);
  rowCountOutputElement.textContent = message;
};

export const updateColumnNamesOutput = (message) => {
  console.log(message);
  columnNamesOutputElement.textContent = message;
};

export const updateSampleRowMessage = (message) => {
  console.log(message);
  sampleRowMessageElement.textContent = message;
};

// tslint:disable-next-line: no-any
export const updateSampleRowOutput = (rawRow) => {
  sampleRowOutputContainerElement.textContent = '';
  const row = rawRow;
  for (const key in row) {
    if (row.hasOwnProperty(key)) {
      sampleRowOutputContainerElement.textContent += key + ':';
      sampleRowOutputContainerElement.textContent += row[key] + ' ';
    }
  }
};

export const getSampleIndex =
    () => {
      return whichSampleInputElement.valueAsNumber;
    }

export const getQueryElement = () => document.getElementById('queryURL');
