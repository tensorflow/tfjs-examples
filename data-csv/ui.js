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
const rowCountOutputElement = document.getElementById('row-count-output');
const columnNamesMessageElement =
    document.getElementById('column-names-message');
const columnNamesOutputContainerElement =
    document.getElementById('column-names-output-container');
const sampleRowMessageElement = document.getElementById('sample-row-message');
const sampleRowOutputContainerElement =
    document.getElementById('sample-row-output-container');
const whichSampleInputElement = document.getElementById('which-sample-input');

/** Updates the large message at the top of the info table */
export function updateStatus(message) {
  statusElement.value = message;
};

/** Updates the message in the "count rows" output row. */
export function updateRowCountOutput(message) {
  rowCountOutputElement.textContent = message;
};

/** Updates the message in the "column names" output row. */
export function updateColumnNamesMessage(message) {
  columnNamesMessageElement.textContent = message;
};


/**
 *  Creates an HTML ordered list to display the column names, represented in the
 * `colNames` argument as a list of strings.
 */
export function updateColumnNamesOutput(colNames) {
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

// Updates the message in the "sample" output row.
export function updateSampleRowMessage(message) {
  sampleRowMessageElement.textContent = message;
};

/**
 * Creates an HTML table, using div elements, to display the key-value pairs
 *  represented in the input `rawRow`. This HTML table is inserted into the
 * sample row.
 */
export function updateSampleRowOutput(rawRow) {
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

// Returns current value of the selected sample index as a number.
export function getSampleIndex() {
  return whichSampleInputElement.valueAsNumber;
}

// Returns the currently specified URL, from which to access the CSV file.
export const getQueryElement = () => {
  return document.getElementById('query-url');
}
