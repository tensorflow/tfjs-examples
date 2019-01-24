
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

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';


import * as game from './game';
import * as ui from './ui';

/**
 * Returns a dataset which will yield unlimited plays of the game.
 */
export const GAME_GENERATOR_DATASET = tf.data.generator(() => {
  const value = game.generateOnePlay();
  const done = false;
  return {value, done};
});

/**
 * Holds game state of most recent simulation to allow for re-calculation
 * of feature representation.
 */
let SAMPLE_GAME_STATE;

/**
 * Takes the state of one complete game and returns features suitable for
 * training.  Specifically it removes features identifying the opponent' hand
 * and divides the training features(player 1's hand) from the target (whether
 * player one wins).
 * @param {*} gameState
 */
export function gameToFeaturesAndLabelOneHot(gameState) {
  // const features = gameState[0];
  const features = tf.concat([
    tf.oneHot(tf.scalar(gameState[0][0], 'int32'), game.MAX_CARD_VALUE),
    tf.oneHot(tf.scalar(gameState[0][1], 'int32'), game.MAX_CARD_VALUE),
    tf.oneHot(tf.scalar(gameState[0][2], 'int32'), game.MAX_CARD_VALUE)
  ]);
  const label = tf.scalar(gameState[2]);
  return {features, label};
}

export function gameToFeaturesAndLabelRaw(gameState) {
  const features = tf.tensor1d(gameState[0]);
  const label = tf.scalar(gameState[2]);
  return {features, label};
}

export function gameToFeaturesAndLabel(gameState) {
  if (ui.getUseOneHot()) {
    return gameToFeaturesAndLabelOneHot(gameState);
  }
  return gameToFeaturesAndLabelRaw(gameState);
}

/**
 * Collects one random play of the game.  Processes the sample to generate
 * features and labels representation of the play.  Calls a UI method to render
 * the sample and the processed sample.
 * @param {bool} wantNewGame : If true, a new game is generated.
 */
async function simulateGameHandler(wantNewGame) {
  if (wantNewGame) {
    SAMPLE_GAME_STATE = game.generateOnePlay();
  }
  const featuresAndLabel = gameToFeaturesAndLabel(SAMPLE_GAME_STATE);
  ui.displaySimulation(SAMPLE_GAME_STATE, featuresAndLabel);
  ui.displayNumSimulationsSoFar(game.NUM_SIMULATIONS_SO_FAR);
}

/** @see datasetToArrayHandler */
async function datasetToArray() {
  return GAME_GENERATOR_DATASET.map(gameToFeaturesAndLabel)
      .batch(ui.getBatchSize())
      .take(ui.getTake())
      .toArray();
}

/**
 * Creates a dataset pipeline from GAME_GENERATOR_DATASET by:
 * 1) Applying the function gameToFeaturesAndlabel
 * 2) Taking the first N samples of the dataset
 * 3) Batching the dataset to batches of size B
 *
 * It then executes the dataset by filling an array.  Finally, it passes this
 * array to the UI to render in a table.
 */
async function datasetToArrayHandler() {
  const arr = await datasetToArray();
  ui.displayBatches(arr);
  ui.displayNumSimulationsSoFar(game.NUM_SIMULATIONS_SO_FAR);
}

function createLinearModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: 3, units: 1, activation: 'sigmoid'}));
  return model;
}

function createDNNModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: 30, units: 10, activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'relu'}));
  model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  return model;
}

async function trainModelUsingFitDataset(model, dataset) {
  const EPOCHS = ui.getEpochsToTrain();
  const BATCHES_PER_EPOCH = ui.getBatchesPerEpoch();
  const VALIDATION_BATCHES = 10;
  const trainLogs = [];
  const beginMs = performance.now();
  const fitDatasetArgs = {
    batchesPerEpoch: BATCHES_PER_EPOCH,
    epochs: EPOCHS,
    validationData: dataset,
    validationBatches: VALIDATION_BATCHES,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // Plot the loss and accuracy values at the end of every training epoch.
        const secPerEpoch =
            (performance.now() - beginMs) / (1000 * (epoch + 1));
        ui.logStatus(
            `Training model... Approximately ` +
            `${secPerEpoch.toFixed(4)} seconds per epoch`);
        trainLogs.push(logs);
        tfvis.show.history(
            ui.getLossContainer(), trainLogs, ['loss', 'val_loss'])
        tfvis.show.history(
            ui.getAccuracyContainer(), trainLogs, ['acc', 'val_acc'])
        ui.displayNumSimulationsSoFar(game.NUM_SIMULATIONS_SO_FAR);
      },
    }
  };
  await model.fitDataset(dataset, fitDatasetArgs);
}

async function trainModelUsingFitDatasetHandler() {
  console.log('I am train model using fit DATASET handler');
  // const model = createLinearModel();
  const model = createDNNModel();
  model.compile({
    optimizer: 'rmsprop',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  const dataset =
      GAME_GENERATOR_DATASET.map(gameToFeaturesAndLabel)
          .map(a => [tf.tensor1d(a.features), tf.tensor1d([a.label])])
          .batch(ui.getBatchSize());
  trainModelUsingFitDataset(model, dataset);
}

function featureTypeClickHandler() {
  ui.displayBatches([]);
  // Only update the sample game features if there is already a sample game.
  if (SAMPLE_GAME_STATE != null) {
    simulateGameHandler(false);
  }
}

/** Sets up handlers for the user affordences, including all buttons. */
document.addEventListener('DOMContentLoaded', async () => {
  console.log('content loaded... connecting buttons.');
  document.getElementById('simulate-game')
      .addEventListener('click', () => simulateGameHandler(true), false);
  document.getElementById('dataset-to-array')
      .addEventListener('click', datasetToArrayHandler, false);
  document.getElementById('train-model-using-fit-dataset')
      .addEventListener('click', trainModelUsingFitDatasetHandler, false);
  ui.displayNumSimulationsSoFar(game.NUM_SIMULATIONS_SO_FAR);
  document.getElementById('generator-batch')
      .addEventListener('change', ui.displayExpectedSimulations, false);
  document.getElementById('batches-per-epoch')
      .addEventListener('change', ui.displayExpectedSimulations, false);
  document.getElementById('epochs-to-train')
      .addEventListener('change', ui.displayExpectedSimulations, false);
  document.getElementById('use-one-hot')
      .addEventListener('click', featureTypeClickHandler, false);
  ui.displayExpectedSimulations();
});
