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
const columnNamesMessageElement = document.getElementById('columnNamesMessage');
const columnNamesOutputContainerElement =
    document.getElementById('columnNamesOutputContainer');
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

export const updateColumnNamesMessage = (message) => {
  console.log(message);
  columnNamesMessageElement.textContent = message;
};

export const updateColumnNamesOutput = (colNames) => {
  const container = columnNamesOutputContainerElement;
  container.align = 'left';
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  const olList = document.createElement('ol');
  for (const name of colNames) {
    const item = document.createElement('li');
    item.textContent = name;
    olList.appendChild(item);
  }
  container.appendChild(olList);
};

export const updateSampleRowMessage = (message) => {
  console.log(message);
  sampleRowMessageElement.textContent = message;
};

export const updateSampleRowOutput = (rawRow) => {
  sampleRowOutputContainerElement.textContent = '';
  const row = rawRow;
  for (const key in row) {
    if (row.hasOwnProperty(key)) {
      const oneKeyRow = document.createElement('div');
      oneKeyRow.className = 'divTableRow';
      oneKeyRow.align = 'left';
      const keyDiv = document.createElement('div');
      const valueDiv = document.createElement('div');
      // TODO(bileschi): There is probably a better way to style this.
      keyDiv.className = 'divTableCellKey';
      valueDiv.className = 'divTableCell';
      keyDiv.textContent = key + ': ';
      valueDiv.textContent = row[key];
      oneKeyRow.appendChild(keyDiv);
      oneKeyRow.appendChild(valueDiv);
      // add the div child to updateSampleRowOutput
      sampleRowOutputContainerElement.appendChild(oneKeyRow);
    }
  }
};

export const getSampleIndex =
    () => {
      return whichSampleInputElement.valueAsNumber;
    }

export const getQueryElement = () => document.getElementById('queryURL');
