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
 * Weather Prediction Example.
 *
 * - Visualizes data using tfjs-vis.
 * - Trains simple models (linear regressor and MLPs) and visualizes the
 *   training processes.
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

import {JenaWeatherData} from './data';
import {buildModel, trainModel} from './models';
import {currBeginIndex, getDataVizOptions, logStatus, populateSelects, TIME_SPAN_RANGE_MAP, TIME_SPAN_STRIDE_MAP, updateDateTimeRangeSpan, updateScatterCheckbox} from './ui';

const dataChartContainer = document.getElementById('data-chart');
const trainModelButton = document.getElementById('train-model');
const modelTypeSelect = document.getElementById('model-type');
const includeDateTimeSelect =
    document.getElementById('include-date-time-features');
const epochsInput = document.getElementById('epochs');

let jenaWeatherData;

/**
 * Render data chart.
 *
 * The rendered visualization obeys:
 *
 * - The dropdown menus for the timeseries.
 * - The "Plot against each other" checkbox.
 * - The "Normalize data" checkbox.
 *
 * Depending on the status of the UI contorls, the chart may be
 *
 * - A line chart that plots one or two timeseries against time, or
 * - A scatter plot that plots two timeseries against on another.
 */
export function plotData() {
  logStatus('Rendering data plot...');
  const {timeSpan, series1, series2, normalize, scatter} = getDataVizOptions();

  if (scatter && series1 !== 'None' && series2 !== 'None') {
    // Plot the two series against each other.
    makeTimeSeriesScatterPlot(series1, series2, timeSpan, normalize);
  } else {
    // Plot one or two series agains time.
    makeTimeSeriesChart(
        series1, series2, timeSpan, normalize, dataChartContainer);
  }

  updateDateTimeRangeSpan(jenaWeatherData);
  updateScatterCheckbox();
  logStatus('Done rendering chart.');
}

/**
 * Plot zero, one or two time series against time.
 *
 * @param {string} series1 Name of timeseries 1 (x-axis).
 * @param {string} series2 Name of timeseries 2 (y-axis).
 * @param {string} timeSpan Name of the time span. Must be a member of
 *   `TIME_SPAN_STRIDE_MAP`.
 * @param {boolean} normalize Whether to use normalized for the two
 *   timeseries.
 * @param {HTMLDivElement} chartConatiner The div element in which
 *   the charts will be rendered.
 */
function makeTimeSeriesChart(
    series1, series2, timeSpan, normalize, chartConatiner) {
  const values = [];
  const series = [];
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
  tfvis.render.linechart(chartConatiner, {values, series: series}, {
    width: chartConatiner.offsetWidth * 0.95,
    height: chartConatiner.offsetWidth * 0.3,
    xLabel: 'Time',
    yLabel: series.length === 1 ? series[0] : '',
  });
}

/**
 * Make a scatter plot of two timeseries.
 *
 * The scatter plot plots the two timeseries against each other.
 *
 * @param {string} series1 Name of timeseries 1 (x-axis).
 * @param {string} series2 Name of timeseries 2 (y-axis).
 * @param {string} timeSpan Name of the time span. Must be a member of
 *   `TIME_SPAN_STRIDE_MAP`.
 * @param {boolean} normalize Whether to use normalized for the two
 *   timeseries.
 */
function makeTimeSeriesScatterPlot(series1, series2, timeSpan, normalize) {
  const includeTime = false;
  const xs = jenaWeatherData.getColumnData(
      series1, includeTime, normalize, currBeginIndex,
      TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]);
  const ys = jenaWeatherData.getColumnData(
      series2, includeTime, normalize, currBeginIndex,
      TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]);
  const values = [xs.map((x, i) => {
    return {x, y: ys[i]};
  })];
  let seriesLabel1 = series1;
  let seriesLabel2 = series2;
  if (normalize) {
    seriesLabel1 += ' (normalized)';
    seriesLabel2 += ' (normalized)';
  }
  const series = [`${seriesLabel1} - ${seriesLabel2}`];

  tfvis.render.scatterplot(dataChartContainer, {values, series}, {
    width: dataChartContainer.offsetWidth * 0.7,
    height: dataChartContainer.offsetWidth * 0.5,
    xLabel: seriesLabel1,
    yLabel: seriesLabel2
  });
}

trainModelButton.addEventListener('click', async () => {
  logStatus('Training model...');
  trainModelButton.disabled = true;
  trainModelButton.textContent = 'Training model. Please wait...'
  // Test iteratorFn.
  const lookBack = 10 * 24 * 6;  // Look back 10 days.
  const step = 6;                // 1-hour steps.
  const delay = 24 * 6;          // Predict the weather 1 day later.
  const batchSize = 128;
  const normalize = true;
  const includeDateTime = includeDateTimeSelect.checked;
  const modelType = modelTypeSelect.value;

  console.log('Creating model...');
  let numFeatures = jenaWeatherData.getDataColumnNames().length;
  const model = buildModel(modelType, Math.floor(lookBack / step), numFeatures);

  // Draw a summary of the model with tfjs-vis visor.
  const surface =
      tfvis.visor().surface({tab: modelType, name: 'Model Summary'});
  tfvis.show.modelSummary(surface, model);

  const trainingSurface =
      tfvis.visor().surface({tab: modelType, name: 'Model Training'});

  console.log('Starting model training...');
  const epochs = +epochsInput.value;
  await trainModel(
      model, jenaWeatherData, normalize, includeDateTime,
      lookBack, step, delay, batchSize, epochs,
      tfvis.show.fitCallbacks(trainingSurface, ['loss', 'val_loss'], {
        callbacks: ['onBatchEnd', 'onEpochEnd']
      }));

  logStatus('Model training complete...');

  if (modelType.indexOf('mlp') === 0) {
    visualizeModelLayers(
        modelType, [model.layers[1], model.layers[2]],
        ['Dense Layer 1', 'Dense Layer 2']);
  } else if (modelType.indexOf('linear-regression') === 0) {
    visualizeModelLayers(modelType, [model.layers[1]], ['Dense Layer 1']);
  }

  trainModelButton.textContent = 'Train model';
  trainModelButton.disabled = false;
});

/**
 * Visualize layers of a model.
 *
 * @param {string} tab Name of the tfjs-vis visor tab on which the visualization
 *   will be made.
 * @param {tf.layers.Layer[]} layers An array of layers to visualize.
 * @param {string[]} layerNames Names of the layers, to be used to label the
 *   tfvis surfaces. Must have the same length as `layers`.
 */
function visualizeModelLayers(tab, layers, layerNames) {
  layers.forEach((layer, i) => {
    const surface = tfvis.visor().surface({tab, name: layerNames[i]});
    tfvis.show.layer(surface, layer);
  });
}

async function run() {
  logStatus('Loading Jena weather data (41.2 MB)...');
  jenaWeatherData = new JenaWeatherData();
  await jenaWeatherData.load();
  logStatus('Done loading Jena weather data.');
  console.log(
      'standard deviation of the T (degC) column: ' +
      jenaWeatherData.getMeanAndStddev('T (degC)').stddev.toFixed(4));

  console.log('Populating data-series selects...');
  populateSelects(jenaWeatherData);

  console.log('Plotting data...');
  plotData();
}

run();
