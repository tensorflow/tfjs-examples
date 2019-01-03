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

import {plotData} from './index';

const statusElement = document.getElementById('status');
const timeSpanSelect = document.getElementById('time-span');
const selectSeries1 = document.getElementById('data-series-1');
const selectSeries2 = document.getElementById('data-series-2');
const dataNormalizedCheckbox = document.getElementById('data-normalized');
const dateTimeRangeSpan = document.getElementById('date-time-range');
const dataPrevButton = document.getElementById('data-prev');
const dataNextButton = document.getElementById('data-next');
const dataScatterCheckbox = document.getElementById('data-scatter');

export function logStatus(message) {
  statusElement.innerText = message;
}

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

export const TIME_SPAN_RANGE_MAP = {
  hour: 6,
  day: 6 * 24,
  week: 6 * 24 * 7,
  tenDays: 6 * 24 * 10,
  month: 6 * 24 * 30,
  year: 6 * 24 * 365,
  full: null
};

export const TIME_SPAN_STRIDE_MAP = {
  day: 1,
  week: 1,
  tenDays: 6,
  month: 6,
  year: 6 * 6,
  full: 6 * 24
};

export let currBeginIndex = 0;

export function updateDateTimeRangeSpan(jenaWeatherData) {
  const timeSpan = timeSpanSelect.value;
  const currEndIndex = currBeginIndex + TIME_SPAN_RANGE_MAP[timeSpan];
  const begin =
      new Date(jenaWeatherData.getTime(currBeginIndex)).toLocaleDateString();
  const end =
      new Date(jenaWeatherData.getTime(currEndIndex)).toLocaleDateString();
  dateTimeRangeSpan.textContent = `${begin} - ${end}`;
}

export function updateScatterCheckbox() {
  const series1 = selectSeries1.value;
  const series2 = selectSeries2.value;
  dataScatterCheckbox.disabled = series1 === 'None' || series2 === 'None';
}

dataPrevButton.addEventListener('click', () => {
  const timeSpan = timeSpanSelect.value;
  currBeginIndex -= Math.round(TIME_SPAN_RANGE_MAP[timeSpan] / 8);
  if (currBeginIndex >= 0) {
    plotData();
  } else {
    currBeginIndex = 0;
  }
});

dataNextButton.addEventListener('click', () => {
  const timeSpan = timeSpanSelect.value;
  currBeginIndex += Math.round(TIME_SPAN_RANGE_MAP[timeSpan] / 8);
  plotData();
});

timeSpanSelect.addEventListener('change', () => {
  plotData();
});
selectSeries1.addEventListener('change', plotData);
selectSeries2.addEventListener('change', plotData);
dataNormalizedCheckbox.addEventListener('change', plotData);
dataScatterCheckbox.addEventListener('change', plotData);

export function getDataVizOptions() {
  return {
    timeSpan: timeSpanSelect.value,
    series1: selectSeries1.value,
    series2: selectSeries2.value,
    normalize: dataNormalizedCheckbox.checked,
    scatter: dataScatterCheckbox.checked
  };
}