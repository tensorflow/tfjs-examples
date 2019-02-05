import {gameToFeaturesAndLabel} from '.';

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

import * as game from './game';

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

export function getInputCards() {
  const cards = [];
  for (let i = 0; i < game.GAME_STATE.num_cards_per_hand; i++) {
    cards.push(document.getElementById(`input-card-${i}`).valueAsNumber);
  }
  return cards;
}

/** Updates display of simulation count. */
export function displayNumSimulationsSoFar() {
  numSimulationsSoFarElement.innerText = game.GAME_STATE.num_simulations_so_far;
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
export function displayPrediction(text) {
  document.getElementById('prediction').innerText = text;
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
 * @param {player1, player2, win]} sample  A game state.  The first
 *     element the sample is an array of player 1's hand.  The second element is
 *     an array of player 2's hand.  The third element is 1 if player 1 wins, 0
 *     otherwise.
 * @param {features, label} featuresAndLabel The processed version of the
 *     sample, suitable to feed into the model.
 */
export function displaySimulation(sample, featuresAndLabel) {
  const player1Row = document.getElementById('player1-row');
  player1Row.innerHTML = '';
  // Player 1 simulation cells.
  for (let i = 0; i < game.GAME_STATE.num_cards_per_hand; i++) {
    const newDiv = document.createElement('div');
    newDiv.className = 'divTableCell';
    newDiv.innerText = sample.player1Hand[i];
    player1Row.appendChild(newDiv);
  }

  const player2Row = document.getElementById('player2-row');
  player2Row.innerHTML = '';
  // Player 2 simulation cells.
  for (let i = 0; i < game.GAME_STATE.num_cards_per_hand; i++) {
    const newDiv = document.createElement('div');
    newDiv.className = 'divTableCell';
    newDiv.innerText = sample.player2Hand[i];
    player2Row.appendChild(newDiv);
  }

  const resultRow = document.getElementById('result-row');
  resultRow.innerHTML = '';
  // Result row.
  const newDiv = document.createElement('div');
  newDiv.className = 'divTableCell';
  newDiv.innerText = sample.player1Win;
  resultRow.appendChild(newDiv);

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

export function updatePredictionInputs() {
  const container = document.getElementById('prediction-input');
  container.innerHTML = '';
  for (let i = 0; i < game.GAME_STATE.num_cards_per_hand; i++) {
    const newH4 = document.createElement('h4');
    newH4.innerText = `card ${i} `;
    const newInput = document.createElement('input');
    newInput.type = 'number';
    newInput.id = `input-card-${i}`;
    newInput.value = 13;
    newH4.appendChild(newInput);
    container.appendChild(newH4);
  }
}

export function enableTrainButton() {
  document.getElementById('train-model-using-fit-dataset')
      .removeAttribute('disabled');
}

export function disableTrainButton() {
  document.getElementById('train-model-using-fit-dataset')
      .setAttribute('disabled', true);
}

export function enableStopButton() {
  document.getElementById('stop-training').removeAttribute('disabled');
}

export function disableStopButton() {
  document.getElementById('stop-training').setAttribute('disabled', true);
}

export function enablePredictButton() {
  document.getElementById('predict').removeAttribute('disabled');
}

export function disablePredictButton() {
  document.getElementById('predict').setAttribute('disabled', true);
}
