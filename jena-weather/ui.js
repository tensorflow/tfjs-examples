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

const statusElement = document.getElementById('status');

export function logStatus(message) {
  statusElement.innerText = message;
}

const timeSpanSelect = document.getElementById('time-span');
const selectSeries1 = document.getElementById('data-series-1');
const selectSeries2 = document.getElementById('data-series-2');
const dataNormalizedCheckbox = document.getElementById('data-normalized');

export function populateSelects(dataObj) {
  const columnNames = ['None'].concat(dataObj.getDataColumnNames());
  for (const selectSeries of [selectSeries1, selectSeries2]) {
    while (selectSeries.firstChild) {
      selectSeries.removeChild(selectSeries.firstChild);
    }
    console.log(columnNames);
    for (const name of columnNames) {
      const option = document.createElement('option');
      option.setAttribute('value', name);
      option.textContent = name;
      selectSeries.appendChild(option);
    }
  }

  if (columnNames.indexOf('T (degC)') !== -1) {
    selectSeries1.value = 'T (degC)';
  }
  if (columnNames.indexOf('p (mbar)') !== -1) {
    selectSeries2.value = 'p (mbar)';
  }
  timeSpanSelect.value = 'week';
  dataNormalizedCheckbox.checked = true;
}
