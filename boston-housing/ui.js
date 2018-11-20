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

import {linearRegressionModel, multiLayerPerceptronRegressionModel1Hidden, multiLayerPerceptronRegressionModel2Hidden, run} from '.';

const statusElement = document.getElementById('status');
export function updateStatus(message) {
  statusElement.innerText = message;
};

const baselineStatusElement = document.getElementById('baselineStatus');
export function updateBaselineStatus(message) {
  baselineStatusElement.innerText = message;
};

export function updateModelStatus(message, modelName) {
  const statElement = document.querySelector(`#${modelName} .status`);
  statElement.innerText = message;
};

const NUM_TOP_WEIGHTS_TO_DISPLAY = 5;
/**
 * Updates the weights output area to include information about the weights
 * learned in a simple linear model.
 * @param {List} weightsList list of objects with 'value':number and
 *     'description':string
 */
export function updateWeightDescription(weightsList) {
  const inspectionHeadlineElement =
      document.getElementById('inspectionHeadline');
  inspectionHeadlineElement.innerText =
      `Top ${NUM_TOP_WEIGHTS_TO_DISPLAY} weights by magnitude`;
  // Sort weights objects by descending absolute value.
  weightsList.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  var table = document.getElementById('myTable');
  // Clear out table contents
  table.innerHTML = '';
  // Add new rows to table.
  weightsList.forEach((weight, i) => {
    if (i < NUM_TOP_WEIGHTS_TO_DISPLAY) {
      let row = table.insertRow(-1);
      let cell1 = row.insertCell(0);
      let cell2 = row.insertCell(1);
      if (weight.value < 0) {
        cell2.setAttribute('class', 'negativeWeight');
      } else {
        cell2.setAttribute('class', 'positiveWeight');
      }
      cell1.innerHTML = weight.description;
      cell2.innerHTML = weight.value.toFixed(4);
    }
  });
};

export async function setup() {
  const trainSimpleLinearRegression = document.getElementById('simple-mlr');
  const trainNeuralNetworkLinearRegression1Hidden =
      document.getElementById('nn-mlr-1hidden');
  const trainNeuralNetworkLinearRegression2Hidden =
      document.getElementById('nn-mlr-2hidden');

  trainSimpleLinearRegression.addEventListener('click', async (e) => {
    const model = linearRegressionModel();
    await run(model, 'linear', true);
  }, false);

  trainNeuralNetworkLinearRegression1Hidden.addEventListener(
      'click', async () => {
        const model = multiLayerPerceptronRegressionModel1Hidden();
        await run(model, 'oneHidden', false);
      }, false);

  trainNeuralNetworkLinearRegression2Hidden.addEventListener(
      'click', async () => {
        const model = multiLayerPerceptronRegressionModel2Hidden();
        await run(model, 'twoHidden', false);
      }, false);
};
