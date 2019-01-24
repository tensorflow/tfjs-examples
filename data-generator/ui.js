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

const generateSampleDataMessageElement =
    document.getElementById('generated-sample-data-message');
const generatedDataContainerElement =
    document.getElementById('generated-data-container');
const toArrayContainerElement = document.getElementById('to-array-container');
const batchSizeElement = document.getElementById('generator-batch');
const takeElement = document.getElementById('generator-take');

/** Returns current value of the batchSize a number. */
export function getBatchSize() {
  return batchSizeElement.valueAsNumber;
}

/** Returns current value of the number to take a number. */
export function getTake() {
  return takeElement.valueAsNumber;
}

/**
 * Fills in the data in the Game Simulation.
 * TODO(bileschi): describe the format of the input `generatedArray`
 */
export function displaySimulation(sample, featuresAndLabel) {
  document.getElementById('sim-p1-1').innerText = sample[0][0];
  document.getElementById('sim-p1-2').innerText = sample[0][1];
  document.getElementById('sim-p1-3').innerText = sample[0][2];
  document.getElementById('sim-p2-1').innerText = sample[1][0];
  document.getElementById('sim-p2-2').innerText = sample[1][1];
  document.getElementById('sim-p2-3').innerText = sample[1][2];
  document.getElementById('sim-result').innerText = sample[2];
  document.getElementById('sim-features-and-label').innerText = JSON.stringify(featuresAndLabel);
};

/**
 * Creates an HTML table, using div elements, to display the generated sample
 * data.
 * TODO(bileschi): describe the format of the input `generatedArray`
 */
export async function displayBatches(arr) {
  toArrayContainerElement.textContent = '';
  let i = 0;
  for (const batch of arr) {
    const oneKeyRow = document.createElement('div');
    oneKeyRow.className = 'divTableRow';
    oneKeyRow.align = 'left';
    const featuresDiv = document.createElement('div');
    const labelDiv = document.createElement('div');
    // TODO(bileschi): Style this better.
    featuresDiv.className = 'divTableCell';
    labelDiv.className = 'divTableCell';
    featuresDiv.textContent = batch.features;
    labelDiv.textContent = batch.label
    oneKeyRow.appendChild(featuresDiv);
    oneKeyRow.appendChild(labelDiv);
    // add the div child to updateSampleRowOutput
    toArrayContainerElement.appendChild(oneKeyRow);
  }
};