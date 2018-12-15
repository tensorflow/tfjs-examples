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

/**
 * Addition RNN example.
 *
 * Based on Python Keras example:
 *   https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

import {JenaWeatherData} from './data';
import {logStatus, populateSelects} from './ui';

let jenaWeatherData;

const timeSpanSelect = document.getElementById('time-span');
const selectSeries1 = document.getElementById('data-series-1');
const selectSeries2 = document.getElementById('data-series-2');
const dataChartContainer = document.getElementById('data-chart');
const dataNormalizedCheckbox = document.getElementById('data-normalized');
const dataScatterCheckbox = document.getElementById('data-scatter');

const dataPrevButton = document.getElementById('data-prev');
const dataNextButton = document.getElementById('data-next');
const dateTimeRangeSpan = document.getElementById('date-time-range');

const TIME_SPAN_RANGE_MAP = {
  day: 6 * 24,
  week: 6 * 24 * 7,
  month: 6 * 24 * 30,
  year: 6 * 24 * 365,
  full: null
};
const TIME_SPAN_STRIDE_MAP = {
  day: 1,
  week: 1,
  month: 6,
  year: 6 * 6,
  full: 6 * 24
};

let currBeginIndex = 0;
function plotData() {
  logStatus('Rendering data plot...');
  const timeSpan = timeSpanSelect.value;
  const series1 = selectSeries1.value;
  const series2 = selectSeries2.value;
  const normalize = dataNormalizedCheckbox.checked;
  const scatter = dataScatterCheckbox.checked;

  const plotAgainstEachOther =
      scatter && series1 !== 'None' && series2 !== 'None';
  const values = [];
  const series = [];
  if (plotAgainstEachOther) {
    // Plot the two series against each other.
    const includeTime = false;
    const xs = jenaWeatherData.getColumnData(
        series1, includeTime, normalize, currBeginIndex,
        TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]);
    const ys = jenaWeatherData.getColumnData(
        series2, includeTime, normalize, currBeginIndex,
        TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]);
    values.push(xs.map((x, i) => {
      return {x, y: ys[i]};
    }));
    series.push(`${series1} - ${series2}`);

    tfvis.render.scatterplot({values, series}, dataChartContainer, {
      width: dataChartContainer.offsetWidth * 0.6,
      height: dataChartContainer.offsetWidth * 0.5,
      xLabel: series1,
      yLabel: series2
    });
    updateDateTimeRangeSpan();
    logStatus('Done rendering data plot.');
  } else {
    // Plot one or two series agains time.
    const includeTime = true;
    if (series1 !== 'None') {
      values.push(jenaWeatherData.getColumnData(
          series1, includeTime, normalize, currBeginIndex,
          TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]));
      series.push(series1);
    }
    if (series2 !== 'None') {
      values.push(jenaWeatherData.getColumnData(
          series2, includeTime, normalize, currBeginIndex,
          TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]));
      series.push(series2);
    }

    // NOTE(cais): On a Linux workstation running latest Chrome, the length
    // limit seems to be around 120k.
    tfvis.render.linechart({values, series: series}, dataChartContainer, {
      width: dataChartContainer.offsetWidth * 0.95,
      height: dataChartContainer.offsetWidth * 0.3,
      xLabel: 'Time',
      yLabel: series.length === 1 ? series[0] : '',
    });
    updateDateTimeRangeSpan();
    logStatus('Done rendering data plot.');
  }

  updateScatterCheckbox();
}

function updateDateTimeRangeSpan() {
  const timeSpan = timeSpanSelect.value;
  const currEndIndex = currBeginIndex + TIME_SPAN_RANGE_MAP[timeSpan];
  const begin =
      new Date(jenaWeatherData.getTime(currBeginIndex)).toLocaleDateString();
  const end =
      new Date(jenaWeatherData.getTime(currEndIndex)).toLocaleDateString();
  dateTimeRangeSpan.textContent = `${begin} - ${end}`;
}

function updateScatterCheckbox() {
  const series1 = selectSeries1.value;
  const series2 = selectSeries2.value;
  dataScatterCheckbox.disabled = series1 === 'None' || series2 === 'None';
}

timeSpanSelect.addEventListener('change', () => {
  currBeginIndex = 0;
  plotData();
});
selectSeries1.addEventListener('change', plotData);
selectSeries2.addEventListener('change', plotData);
dataNormalizedCheckbox.addEventListener('change', plotData);
dataScatterCheckbox.addEventListener('change', plotData);

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

async function run() {
  logStatus('Loading Jena weather data...');
  jenaWeatherData = new JenaWeatherData();
  await jenaWeatherData.load();
  logStatus('Done loading Jena weather data.');

  populateSelects(jenaWeatherData);
  plotData();

  // const dataColumnName = 'Tdew (degC)';
}

run();
