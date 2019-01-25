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

export const lossContainerElement =
    document.getElementById('training-loss-canvas');
export const accuracyContainerElement =
    document.getElementById('training-accuracy-canvas');

const toArrayContainerElement = document.getElementById('to-array-container');
const batchSizeElement = document.getElementById('generator-batch');
const takeElement = document.getElementById('generator-take');
const statusElement = document.getElementById('train-model-message');
const numSimulationsSoFarElement =
    document.getElementById('num-simulations-so-far');
const batchesPerEpochElement = document.getElementById('batches-per-epoch');
const epochsToTrainElement = document.getElementById('epochs-to-train');
const expectedSimulationsElement =
    document.getElementById('expected-simulations');
const inputCard1Element = document.getElementById('input-card-1');
const inputCard2Element = document.getElementById('input-card-2');
const inputCard3Element = document.getElementById('input-card-3');


export const useOneHotElement = document.getElementById('use-one-hot');

export function getBatchSize() {
  return batchSizeElement.valueAsNumber;
}

export function getTake() {
  return takeElement.valueAsNumber;
}

export function getBatchesPerEpoch() {
  return batchesPerEpochElement.valueAsNumber;
}

export function getEpochsToTrain() {
  return epochsToTrainElement.valueAsNumber;
}

export function getInputCard1() {
  return inputCard1Element.valueAsNumber;
}

export function getInputCard2() {
  return inputCard2Element.valueAsNumber;
}

export function getInputCard3() {
  return inputCard3Element.valueAsNumber;
}

export function getUseOneHot() {
  return useOneHotElement.checked;
}

/** Updates display of simulation count. */
export function displayNumSimulationsSoFar(numSimulationsSoFar) {
  numSimulationsSoFarElement.innerText = numSimulationsSoFar;
}

/** Updates message for training results.  Used for training speed. */
export function displayTrainLogMessage(message) {
  statusElement.innerText = message;
}

/** Updates display of expected number of simulations to train model. */
export function displayExpectedSimulations() {
  const expectedSimulations =
      getBatchSize() * getBatchesPerEpoch() * getEpochsToTrain();
  expectedSimulationsElement.innerText = expectedSimulations;
}

/** Updates display of prediction from model. */
export function displayPrediction(prediction) {
  document.getElementById('prediction').innerText =
      `${prediction.dataSync()[0].toFixed(3)}`;
}

/** Helper to display processed version of game state. */
function featuresAndLabelsToPrettyString(features) {
  const basicArray = [];
  for (const value of features) {
    basicArray.push(value);
  }
  return basicArray;
}

/**
 * Fills in the data in the Game Simulation.
 * @param {[number[], number[], number]} sample  A game state.  The first
 *     element the sample is an array of player 1's hand.  The second element is
 *     an array of player 2's hand.  The third element is 1 if player 1 wins, 0
 *     otherwise.
 * @param {features, label} featuresAndLabel The processed version of the
 *     sample, suitable to feed into the model.
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
 *
 * @param {[number[], number[], number][]} arr A list of game states, each
 *     element a triple.  The first element of each game state is an array of
 *     player 1's hand.  The second element is an array of player 2's hand.  The
 *     third element is 1 if player 1 wins, 0 otherwise.
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
