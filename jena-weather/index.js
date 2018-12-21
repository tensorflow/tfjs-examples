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
 * Weather Prediction Example
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

const trainModelButton = document.getElementById('train-model');

const TIME_SPAN_RANGE_MAP = {
  hour: 6,
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
    let seriesLabel1 = series1;
    let seriesLabel2 = series2;
    if (normalize) {
      seriesLabel1 += ' (normalized)';
      seriesLabel2 += ' (normalized)';
    }
    series.push(`${seriesLabel1} - ${seriesLabel2}`);

    tfvis.render.scatterplot({values, series}, dataChartContainer, {
      width: dataChartContainer.offsetWidth * 0.7,
      height: dataChartContainer.offsetWidth * 0.5,
      xLabel: seriesLabel1,
      yLabel: seriesLabel2
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
      series.push(normalize ? `${series1} (normalized)` : series1);
    }
    if (series2 !== 'None') {
      values.push(jenaWeatherData.getColumnData(
          series2, includeTime, normalize, currBeginIndex,
          TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]));
      series.push(normalize ? `${series2} (normalized)` : series2);
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

trainModelButton.addEventListener('click', async () => {


  // Test iteratorFn.
  const shuffle = true;
  const lookBack = 10 * 24 * 6;  // Look back 10 days.
  const step = 6;  // 1-hour steps.
  const delay = 24 * 6;  // Predict the weather 1 day later.
  const batchSize = 128;
  const minIndex = 0;
  const maxIndex = 200000;
  const normalize = true;  // TODO(cais): Set to true.

  // Construct model.
  const numFeatures = 13;  // TODO(cais): Do not hardcode.
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape: [Math.floor(lookBack / step), numFeatures]}));
  model.add(tf.layers.dense({units: 32, activation: 'relu'}));
  model.add(tf.layers.dense({units: 1}));
  model.compile({loss: 'meanAbsoluteError', optimizer: 'rmsprop'});
  model.summary();

  // const lookBack = 24 * 6;  // Look back 1 days.
  // const step = 6;  // 1-hour steps.
  // const delay = 24 * 6;  // Predict the weather 1 day later.
  // const batchSize = 1;
  // const minIndex = 0;
  // const maxIndex = 200000;
  // const normalize = false;

  const iteratorFn = jenaWeatherData.getIteratorFn(
      shuffle, lookBack, delay, batchSize, step, minIndex, maxIndex, normalize);
  // TODO(cais): Use the following when the API is available.
  // const dataset = tf.data.generator(iteratorFn);
  // // let out = await dataset.iterator();
  // // console.log(out.value[0].shape, out.value[1].shape);
  // out.value[1].print();
  // // out = await dataset.iterator();
  // out.value[1].print();

  const epochs = 20;
  const batchesPerEpoch = 500;  // TODO(cais): 500.
  for (let i = 0; i < epochs; ++i) {
    const t0 = tf.util.now();
    for (let j = 0; j < batchesPerEpoch; ++j) {
      const item = iteratorFn();
      const trainLoss = await model.trainOnBatch(item.value[0], item.value[1]);
      console.log(
          `epoch ${i + 1}/${epochs} batch ${j + 1}/${batchesPerEpoch}: ` +
          `trainLoss=${trainLoss.toFixed(6)}`);  // DEBUG
      tf.dispose(item.value);
    }
    const t1 = tf.util.now();
    console.log(`${(t1 - t0) / batchesPerEpoch} ms/batch`);
  }
});

async function run() {
  logStatus('Loading Jena weather data (41.2 MB)...');
  jenaWeatherData = new JenaWeatherData();
  await jenaWeatherData.load();
  logStatus('Done loading Jena weather data.');

  populateSelects(jenaWeatherData);
  plotData();
}

run();
