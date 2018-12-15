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

function plotData() {
  logStatus('Rendering data plot...');
  const timeSpan = timeSpanSelect.value;
  const series1 = selectSeries1.value;
  const series2 = selectSeries2.value;
  const normalize = dataNormalizedCheckbox.checked;

  const includeTime = true;
  // NOTE(cais): On a Linux workstation running latest Chrome, the length
  // limit seems to be around 120k.
  const values = [];
  const seriesNames = [];
  if (series1 != 'None') {
    values.push(jenaWeatherData.getColumnData(
        series1, includeTime, normalize, 0, TIME_SPAN_RANGE_MAP[timeSpan],
        TIME_SPAN_STRIDE_MAP[timeSpan]));
    seriesNames.push(series1);
  }
  if (series2 != 'None') {
    values.push(jenaWeatherData.getColumnData(
        series2, includeTime, normalize, 0, TIME_SPAN_RANGE_MAP[timeSpan],
        TIME_SPAN_STRIDE_MAP[timeSpan]));
    seriesNames.push(series2);
  }
  tfvis.render.linechart({values, series: seriesNames}, dataChartContainer, {
    width: dataChartContainer.offsetWidth * 0.95,
    height: 300,
    xLabel: 'Time',
    yLabel: seriesNames.length === 1 ? seriesNames[0] : '',
  });
  logStatus('Done rendering data plot.');
}

timeSpanSelect.addEventListener('change', plotData);
selectSeries1.addEventListener('change', plotData);
selectSeries2.addEventListener('change', plotData);
dataNormalizedCheckbox.addEventListener('change', plotData);

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
