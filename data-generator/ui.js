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
const statusElement = document.getElementById('train-model-message');
const lossContainerElement = document.getElementById('training-loss-canvas');
const accuracyContainerElement =
    document.getElementById('training-accuracy-canvas');
const numSimulationsSoFarElement =
    document.getElementById('num-simulations-so-far');
const batchesPerEpochElement = document.getElementById('batches-per-epoch');
const epochsToTrainElement = document.getElementById('epochs-to-train');
const expectedSimulationsElement =
    document.getElementById('expected-simulations');
export const useOneHotElement = document.getElementById('use-one-hot');

/** borrowd from mnist.  probably remove */
export function logStatus(message) {
  statusElement.innerText = message;
}

/** Returns current value of the batchSize a number. */
export function getBatchSize() {
  return batchSizeElement.valueAsNumber;
}

/** Returns current value of the number to take a number. */
export function getTake() {
  return takeElement.valueAsNumber;
}

export function getLossContainer() {
  return lossContainerElement;
}

export function getAccuracyContainer() {
  return accuracyContainerElement;
}

export function getBatchesPerEpoch() {
  return batchesPerEpochElement.valueAsNumber;
}

export function getEpochsToTrain() {
  return epochsToTrainElement.valueAsNumber;
}

export function displayNumSimulationsSoFar(numSimulationsSoFar) {
  numSimulationsSoFarElement.innerText = numSimulationsSoFar;
}

export function getUseOneHot() {
  return useOneHotElement.checked;
}

export function displayExpectedSimulations() {
  const expectedSimulations =
      getBatchSize() * getBatchesPerEpoch() * getEpochsToTrain();
  expectedSimulationsElement.innerText = expectedSimulations;
}

function featuresAndLabelsToPrettyString(features) {
  const basicArray = [];
  for (const value of features) {
    basicArray.push(value);
  }
  return basicArray;
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
  const features = featuresAndLabel.features.dataSync();
  const label = featuresAndLabel.label.dataSync();
  document.getElementById('sim-features').innerText =
      JSON.stringify(featuresAndLabelsToPrettyString(features));
  document.getElementById('sim-label').innerText = label;
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
