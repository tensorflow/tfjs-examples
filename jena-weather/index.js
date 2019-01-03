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
    makeTimeSeriesChart(series1, series2, timeSpan, normalize);
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
 */
function makeTimeSeriesChart(series1, series2, timeSpan, normalize) {
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
  console.log('Calling linerchart');  // DEBUG
  tfvis.render.linechart({values, series: series}, dataChartContainer, {
    width: dataChartContainer.offsetWidth * 0.95,
    height: dataChartContainer.offsetWidth * 0.3,
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

  tfvis.render.scatterplot({values, series}, dataChartContainer, {
    width: dataChartContainer.offsetWidth * 0.7,
    height: dataChartContainer.offsetWidth * 0.5,
    xLabel: seriesLabel1,
    yLabel: seriesLabel2
  });
}

/**
 * Build a linear-regression model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @returns {tf.Model} A TensorFlow.js tf.Model instance.
 */
function buildLinearRegressionModel(inputShape) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape}));
  model.add(tf.layers.dense({units: 1}));
  return model;
}

/**
 * Build a GRU model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @param {tf.regularizer.Regularizer} kernelRegularizer An optional
 *   regularizer for the kernel of the first (hdiden) dense layer of the MLP.
 *   If not specified, no weight regularization will be included in the MLP.
 * @param {number} dropoutRate Dropout rate of an optional dropout layer
 *   inserted between the two dense layers of the MLP. Optional. If not
 *   specified, no dropout layers will be included in the MLP.
 * @returns {tf.Model} A TensorFlow.js tf.Model instance.
 */
function buildMLPModel(inputShape, kernelRegularizer, dropoutRate) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape}));
  model.add(
      tf.layers.dense({units: 32, kernelRegularizer, activation: 'relu'}));
  if (dropoutRate > 0) {
    model.add(tf.layers.dropout({rate: dropoutRate}));
  }
  model.add(tf.layers.dense({units: 1}));
  return model;
}

/**
 * Build a GRU model for the temperature-prediction problem.
 *
 * TODO(cais): Move this to a tfjs-node training script, as training
 *  the GRU in the browser turns out to be too slow.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @returns {tf.Model} A TensorFlow.js GRU model.
 */
// function buildGRUModel(inputShape) {
//   // TODO(cais): Add recurrent dropout.
//   const model = tf.sequential();
//   const rnnUnits = 32;
//   model.add(tf.layers.gru({units: rnnUnits, inputShape}));
//   model.add(tf.layers.dense({units: 1}));
//   return model;
// }

/**
 * Build a model for the temperature-prediction problem.
 *
 * @param {string} modelType Model type.
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @returns A compiled instance of `tf.Model`.
 */
function buildModel(modelType, inputShape) {
  console.log(`modelType = ${modelType}`);
  let model;
  if (modelType === 'mlp') {
    model = buildMLPModel(inputShape);
  } else if (modelType === 'mlp-l2') {
    model = buildMLPModel(inputShape, tf.regularizers.l2());
  } else if (modelType === 'linear-regression') {
    model = buildLinearRegressionModel(inputShape);
  } else if (modelType === 'mlp-dropout') {
    const regularizer = null;
    const dropoutRate = 0.25;
    model = buildMLPModel(inputShape, regularizer, dropoutRate);
  } else if (modelType === 'gru') {
    model = buildGRUModel(inputShape);
  } else {
    throw new Error(`Unsupported model type: ${modelType}`);
  }

  model.compile({loss: 'meanAbsoluteError', optimizer: 'rmsprop'});
  return model;
}

let lossValues;
let valLossValues;

/**
 * Plot loss and accuracy curves.
 *
 * TODO(cais): This should be replaced by tfvis.show.fitCallbacks() once
 *   fitDataset() is used.
 */
function plotLoss(modelType, epoch, loss, valLoss) {
  const lossSurface =
      tfvis.visor().surface({tab: modelType, name: 'Loss curves'});

  if (epoch <= 1) {
    lossValues = [];
    valLossValues = [];
  }
  lossValues.push({x: epoch, y: loss});
  valLossValues.push({x: epoch, y: valLoss});

  // const values = [{x: 1, y: 10}, {x: 2, y: 20}, {x: 3, y: 5}];
  tfvis.render.linechart(
      {values: [lossValues, valLossValues], series: ['train', 'val']},
      lossSurface, {xLabel: 'Epoch #', yLabel: 'Loss'});
}

trainModelButton.addEventListener('click', async () => {
  logStatus('Training model...');
  trainModelButton.disabled = true;
  // Test iteratorFn.
  const shuffle = true;
  const lookBack = 10 * 24 * 6;  // Look back 10 days.
  const step = 6;                // 1-hour steps.
  const delay = 24 * 6;          // Predict the weather 1 day later.
  const batchSize = 128;
  const minIndex = 0;
  const maxIndex = 200000;
  const normalize = true;
  const includeDateTime = includeDateTimeSelect.checked;
  console.log(`includeDateTime = ${includeDateTime}`);

  // Construct model.
  const modelType = modelTypeSelect.value;
  let numFeatures = jenaWeatherData.getDataColumnNames().length;
  if (includeDateTime) {
    numFeatures += 2;
  }
  const model =
      buildModel(modelType, [Math.floor(lookBack / step), numFeatures]);

  // Draw a summary of the model with tfjs-vis visor.
  const surface =
      tfvis.visor().surface({tab: modelType, name: 'Model Summary'});
  tfvis.show.modelSummary(surface, model);

  const trainNextBatchFn = jenaWeatherData.getNextBatchFunction(
      shuffle, lookBack, delay, batchSize, step, minIndex, maxIndex, normalize,
      includeDateTime);

  // TODO(cais): Use the following when the API is available.
  // const dataset = tf.data.generator(iteratorFn);
  // await model.fitDataset(dataset, {epochs, batchesPerEpoch, ...});
  const epochs = +epochsInput.value;
  const batchesPerEpoch = 500;
  const displayEvery = 100;
  for (let i = 0; i < epochs; ++i) {
    const t0 = tf.util.now();
    let totalTrainLoss = 0;
    let numSeen = 0;
    for (let j = 0; j < batchesPerEpoch; ++j) {
      const item = trainNextBatchFn();
      const trainLoss = await model.trainOnBatch(item.value[0], item.value[1]);

      numSeen += item.value[0].shape[0];
      totalTrainLoss += item.value[0].shape[0] * trainLoss;
      if ((j + 1) % displayEvery === 0) {
        console.log(
            `epoch ${i + 1}/${epochs} batch ${j + 1}/${batchesPerEpoch}: ` +
            `trainLoss=${trainLoss.toFixed(6)}`);
      }
      tf.dispose(item.value);
    }
    const t1 = tf.util.now();
    const epochTrainLoss = totalTrainLoss / numSeen;

    // Perform validation.
    const valMinIndex = 200001;
    const valMaxIndex = 300000;
    const valNextBatchFn = jenaWeatherData.getNextBatchFunction(
        false, lookBack, delay, batchSize, step, valMinIndex, valMaxIndex,
        normalize, includeDateTime);
    const valT0 = tf.util.now();
    const valSteps = Math.floor((300000 - 200001 - lookBack) / batchSize);
    tf.tidy(() => {
      console.log(`Running validation: valSteps=${valSteps}`);
      let totalValLoss = tf.scalar(0);
      numSeen = 0;
      for (let j = 0; j < valSteps; ++j) {
        if (j % displayEvery === 0) {
          console.log(`  Validation: step ${j}/${valSteps}`);
        }
        const item = valNextBatchFn();
        const evalOut =
            model.evaluate(item.value[0], item.value[1], {batchSize});
        const numExamples = item.value[0].shape[0];
        totalValLoss = tf.tidy(
            () => totalValLoss.add(evalOut.mulStrict(tf.scalar(numExamples))));
        numSeen += numExamples;
        tf.dispose([item.value, evalOut]);
      }
      const valLoss = totalValLoss.divStrict(tf.scalar(numSeen)).dataSync()[0];
      const valT1 = tf.util.now();
      const valMsPerBatch = (valT1 - valT0) / valSteps;
      console.log(
          `epoch ${i + 1}/${epochs}: trainLoss=${epochTrainLoss.toFixed(6)}; ` +
          `valLoss=${valLoss.toFixed(6)} ` +
          `(train: ${((t1 - t0) / batchesPerEpoch).toFixed(1)} ms/batch; ` +
          `val: ${valMsPerBatch.toFixed(1)} ms/batch)\n`);
      plotLoss(modelType, i + 1, epochTrainLoss, valLoss)

      tf.dispose(valLoss);
    });
  }
  trainModelButton.disabled = false;
  logStatus('Model training complete...');

  if (modelType.indexOf('mlp') === 0) {
    visualizeModelLayers(
        modelType, [model.layers[1], model.layers[2]],
        ['Dense Layer 1', 'Dense Layer 2']);
  } else if (modelType.indexOf('linear-regression') === 0) {
    visualizeModelLayers(modelType, [model.layers[1]], ['Dense Layer 1']);
  }
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
