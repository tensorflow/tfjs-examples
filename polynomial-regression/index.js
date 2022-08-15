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

import * as tf from '@tensorflow/tfjs';

const canvas = document.getElementById('canvas');
const order = 3;
// Convert world coordinates to canvas ones.
function world2canvas(canvas, x, y) {
  return [x + canvas.width / 2, -y + canvas.height / 2];
}
// Draw x and y axes in the canvas.
function drawAxes(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  const leftCoord = world2canvas(canvas, -canvas.width / 2, 0);
  const rightCoord = world2canvas(canvas, canvas.width / 2, 0);
  ctx.moveTo(leftCoord[0], leftCoord[1]);
  ctx.lineTo(rightCoord[0], rightCoord[1]);
  ctx.stroke();
  const topCoord = world2canvas(canvas, 0, canvas.height / 2);
  const bottomCoord = world2canvas(canvas, 0, -canvas.height / 2);
  ctx.moveTo(topCoord[0], topCoord[1]);
  ctx.lineTo(bottomCoord[0], bottomCoord[1]);
  ctx.stroke();
}

// Draw x and y data in the canvas.
//
// Also draws the x and y axes.
//
// Args:
//   canvas: The canvas to draw the data in.
//   xyData: An Array of [x, y] Arrays.
function drawXYData(canvas, xyData) {
  drawAxes(canvas);
  const ctx = canvas.getContext('2d');
  for (let i = 0; i < xyData.length; ++i) {
    ctx.beginPath();
    const x = xyData[i][0];
    const y = xyData[i][1];
    const canvasCoord = world2canvas(canvas, x, y);
    ctx.arc(canvasCoord[0], canvasCoord[1], 4, 0, Math.PI * 2, true);
    ctx.stroke();
  }
}

// Calculate the arithmetic mean of a vector.
//
// Args:
//   vector: The vector represented as an Array of Numbers.
//
// Returns:
//   The arithmetic mean.
function mean(vector) {
  let sum = 0;
  for (const x of vector) {
    sum += x;
  }
  return sum / vector.length;
}

// Calculate the standard deviation of a vector.
//
// Args:
//   vector: The vector represented as an Array of Numbers.
//
// Returns:
//   The standard deviation.
function stddev(vector) {
  let squareSum = 0;
  const vectorMean = mean(vector);
  for (const x of vector) {
    squareSum += (x - vectorMean) * (x - vectorMean);
  }
  return Math.sqrt(squareSum / (vector.length - 1));
}

// Normalize a vector by its mean and standard deviation.
function normalizeVector(vector, vectorMean, vectorStddev) {
  return vector.map(x => (x - vectorMean) / vectorStddev);
}

// Convert x-y data to normalized Tensors.
//
// Args:
//   xyData: An Array of [x, y] Number Arrays.
//   order: The order of the polynomial to generate data for. Assumed to be
//     a non-negative integer.
//
// Returns: An array consisting of the following
//   xPowerMeans: Arithmetic means of the powers of x, from order `1` to
//      order `order`
//   xPowerStddevs: Standard deviations of the powers of x.
//   Normalized powers of x: an Tensor2D of shape [batchSize, order + 1].
//     The first column is all ones; the following columns are powers of x
//     from order `1` to `order`.
//   yMean: Arithmetic mean of y.
//   yStddev: Standard deviation of y.
//   Normalized powers of y: an Tensor2D of shape [batchSize, 1].
function toNormalizedTensors(xyData, order) {
  const batchSize = xyData.length;
  const xData = xyData.map(xy => xy[0]);
  const yData = xyData.map(xy => xy[1]);
  const yMean = mean(yData);
  const yStddev = stddev(yData);
  const yNormalized = normalizeVector(yData, yMean, yStddev);
  const normalizedXPowers = [];
  const xPowerMeans = [];
  const xPowerStddevs = [];
  for (let i = 0; i < order; ++i) {
    const xPower = xData.map(x => Math.pow(x, i + 1));
    const xPowerMean = mean(xPower);
    xPowerMeans.push(xPowerMean);
    const xPowerStddev = stddev(xPower);
    xPowerStddevs.push(xPowerStddev);
    const normalizedXPower = normalizeVector(xPower, xPowerMean, xPowerStddev);
    normalizedXPowers.push(normalizedXPower);
  }
  const xArrayData = [];
  for (let i = 0; i < xData.length; ++i) {
    for (let j = 0; j < order + 1; ++j) {
      if (j === 0) {
        xArrayData.push(1);
      } else {
        xArrayData.push(normalizedXPowers[j - 1][i]);
      }
    }
  }
  return [
    xPowerMeans, xPowerStddevs, tf.tensor2d(xArrayData, [batchSize, order + 1]),
    yMean, yStddev, tf.tensor2d(yNormalized, [batchSize, 1])
  ];
}

// Fit a model for polynomial regression.
//
// Args:
//   xyData: An Array of [x, y] Number Arrays.
//   epochs: How many epochs to train for.
//   learningRate: Learning rate.
//
// Returns: An Array consisting of the following:
//   The trained keras Model instance.
//   xPowerMeans: Arithmetic means of the powers of x, from order `1` to
//      order `order`
//   xPowerStddevs: Standard deviations of the powers of x.
//   yMean: Arithmetic mean of y.
//   yStddev: Standard deviation of y.
async function fitModel(xyData, epochs, learningRate) {
  const batchSize = xyData.length;
  const outputs = toNormalizedTensors(xyData, order);
  const xPowerMeans = outputs[0];
  const xPowerStddevs = outputs[1];
  const xData = outputs[2];
  const yMean = outputs[3];
  const yStddev = outputs[4];
  const yData = outputs[5];
  const input = tf.input({shape: [order + 1]});
  const linearLayer =
      tf.layers.dense({units: 1, kernelInitializer: 'Zeros', useBias: false});
  const output = linearLayer.apply(input);
  const model = tf.model({inputs: input, outputs: output});
  const sgd = tf.train.sgd(learningRate);
  model.compile({optimizer: sgd, loss: 'meanSquaredError'});
  await model.fit(xData, yData, {
    batchSize: batchSize,
    epochs: epochs,
  });
  console.log(
      'Model weights (normalized):',
      model.trainableWeights[0].read().dataSync());
  return [model, xPowerMeans, xPowerStddevs, yMean, yStddev];
}

// Render the predictions made by the model.
function renderModelPredictions(
    canvas, order, model, xPowerMeans, xPowerStddevs, yMean, yStddev) {
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  let x = -0.5 * width;
  const xStep = 0.02 * width;
  const xs = [];
  const xPowers = [];
  let n = 0;
  while (x < 0.5 * width) {
    xs.push(x);
    let d = 1;
    for (let j = 0; j < order + 1; ++j) {
      xPowers.push(
          j === 0 ? d : ((d - xPowerMeans[j - 1]) / xPowerStddevs[j - 1]));
      d *= x;
    }
    x += xStep;
    n++;
  }

  const predictOut = model.predict(tf.tensor2d(xPowers, [n, order + 1]));
  const normalizedYs = predictOut.dataSync();
  ctx.beginPath();
  let canvasXY = world2canvas(canvas, xs[0], normalizedYs[0] * yStddev + yMean);
  ctx.moveTo(canvasXY[0], canvasXY[1]);
  for (let i = 1; i < n; ++i) {
    canvasXY = world2canvas(canvas, xs[i], normalizedYs[i] * yStddev + yMean);
    ctx.lineTo(canvasXY[0], canvasXY[1]);
    ctx.stroke();
  }
}

// Generate x-y data based on the size of the canvas.
function generateXYData(canvas, coeffs) {
  const data = [];
  for (let x = -canvas.width / 2; x < canvas.width / 2;
       x += canvas.width / 25) {
    data.push([
      x, coeffs[0] * x * x * x + coeffs[1] * x * x + coeffs[2] * x + coeffs[3]
    ]);
  }
  return data;
}

// Fit a model and render the data and predictions.
async function fitAndRender() {
  const epochs = +epochsElement.value;
  const learningRate = +learningRateElement.value;
  if (!isFinite(epochs) || !isFinite(learningRate)) {
    return;
  }
  const coeffs = [
    +cubicCoeffElement.value, +quadCoeffElement.value,
    +linearCoeffElement.value, +constCoeffElement.value
  ];
  console.log('True coefficients: ' + JSON.stringify(coeffs));
  let xyData = generateXYData(canvas, coeffs);
  drawXYData(canvas, xyData);
  const fitOutputs = await fitModel(xyData, epochs, learningRate);
  const model = fitOutputs[0];
  const xPowerMeans = fitOutputs[1];
  const xPowerStddevs = fitOutputs[2];
  const yMean = fitOutputs[3];
  const yStddev = fitOutputs[4];
  await renderModelPredictions(
      canvas, order, model, xPowerMeans, xPowerStddevs, yMean, yStddev);
}

const cubicCoeffElement = document.getElementById('cubic-coeff');
const quadCoeffElement = document.getElementById('quad-coeff');
const linearCoeffElement = document.getElementById('linear-coeff');
const constCoeffElement = document.getElementById('const-coeff');

const epochsElement = document.getElementById('epochs');
const learningRateElement = document.getElementById('learning-rate');

cubicCoeffElement.addEventListener('keyup', fitAndRender);
quadCoeffElement.addEventListener('keyup', fitAndRender);
linearCoeffElement.addEventListener('keyup', fitAndRender);
constCoeffElement.addEventListener('keyup', fitAndRender);

epochsElement.addEventListener('keyup', fitAndRender);
learningRateElement.addEventListener('keyup', fitAndRender);

fitAndRender();
