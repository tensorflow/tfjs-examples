
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
 * Holds the model to be trained & evaluated.
 */
let GLOBAL_MODEL;

/**
 * Takes the state of one complete game and returns features suitable for
 * training.  Returns an object containing features = player1's hand represented
 * using oneHot encoding, and label = whether player 1 won.
 * @param {*} gameState
 */
function gameToFeaturesAndLabelOneHot(gameState) {
  return tf.tidy(() => {
    const player1Hand = tf.tensor1d(gameState[0], 'int32');
    const handOneHot = tf.oneHot(
        tf.sub(player1Hand, tf.scalar(1, 'int32')), game.MAX_CARD_VALUE);
    const features = tf.sum(handOneHot, 0);
    const label = tf.tensor1d([gameState[2]]);
    return {features, label};
  });
}

/**
 * Takes the state of one complete game and returns features suitable for
 * training.  Returns an object containing features = player1's hand
 * and label = whether player 1 won.
 * @param {*} gameState
 */
function gameToFeaturesAndLabelRaw(gameState) {
  const features = tf.tensor1d(gameState[0]);
  const label = tf.tensor1d([gameState[2]]);
  return {features, label};
}

/**
 * Takes the state of one complete game and returns features suitable for
 * training.  Depending if 'oneHot' is selected, Player 1's hand is represented
 * either literally, as an array of three numbers or using oneHotRepresentation.
 * @param {*} gameState
 */
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

/**
 * This is pulled into a separate function to isolate the async code.
 *  @see datasetToArrayHandler
 */
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

/**
 * Returns a three layer sequential model suitable for predicting win state from
 * feature representation.  The input shape depends on whether oneHot
 * representation is used.
 */
function createDNNModel() {
  GLOBAL_MODEL = tf.sequential();
  GLOBAL_MODEL.add(tf.layers.dense({
    inputShape:
        [ui.getUseOneHot() ? game.MAX_CARD_VALUE : game.NUM_CARDS_PER_HAND],
    units: 10,
    activation: 'relu'
  }));
  GLOBAL_MODEL.add(tf.layers.dense({units: 10, activation: 'relu'}));
  GLOBAL_MODEL.add(tf.layers.dense({units: 10, activation: 'relu'}));
  GLOBAL_MODEL.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  return GLOBAL_MODEL;
}


/**
 * Trains a the provided model on the provided dataset using model.fitDataset.
 * Schedules a callback at the end of every epoch to update the UI with
 * graphs showing loss and accuracy, as well as training speed and the current
 * prediction for the manually entered hand.
 * @param {tf.Model} model
 * @param {tf.data.Dataset} dataset
 */
async function trainModelUsingFitDataset(model, dataset) {
  const trainLogs = [];
  const beginMs = performance.now();
  const fitDatasetArgs = {
    batchesPerEpoch: ui.getBatchesPerEpoch(),
    epochs: ui.getEpochsToTrain(),
    validationData: dataset,
    validationBatches: 10,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // Plot the loss and accuracy values at the end of every training epoch.
        const secPerEpoch =
            (performance.now() - beginMs) / (1000 * (epoch + 1));
        ui.displayTrainLogMessage(
            `Training model... Approximately ` +
            `${secPerEpoch.toFixed(4)} seconds per epoch`);
        trainLogs.push(logs);
        tfvis.show.history(
            ui.lossContainerElement, trainLogs, ['loss', 'val_loss'])
        tfvis.show.history(
            ui.accuracyContainerElement, trainLogs, ['acc', 'val_acc'])
        ui.displayNumSimulationsSoFar(game.NUM_SIMULATIONS_SO_FAR);
        // Update the prediction.
        predictHandler();
      },
    }
  };
  await model.fitDataset(dataset, fitDatasetArgs);
}

/**
 * Constructs a new model and trains it on a dataset pipeline built off of
 * GAME_GENERATOR_DATASET.  The dataset pipeline performs feature calculation
 * and batching.
 * @see trainModelUsingFitDataset for training details.
 */
async function trainModelUsingFitDatasetHandler() {
  const model = createDNNModel();
  model.compile({
    optimizer: 'rmsprop',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  const dataset = GAME_GENERATOR_DATASET.map(gameToFeaturesAndLabel)
                      .map(a => [a.features, a.label])
                      .batch(ui.getBatchSize());
  trainModelUsingFitDataset(model, dataset);
}

/**
 * Handler for changing the feature representation between raw and oneHot.
 * Updates the simulation output and clears a batch sample, if it exists.
 */
function featureTypeClickHandler() {
  ui.displayBatches([]);
  // Only update the sample game features if there is already a sample game.
  if (SAMPLE_GAME_STATE != null) {
    simulateGameHandler(false);
  }
}

/**
 * Applies the model to the manually entered hand value and updates the UI with
 * the model's prediction.
 */
function predictHandler() {
  const cards = [ui.getInputCard1(), ui.getInputCard2(), ui.getInputCard3()];
  const features = gameToFeaturesAndLabel([cards, [1, 2, 3], 1]).features;
  const output = GLOBAL_MODEL.predict(features.expandDims(0));
  ui.displayPrediction(output);
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
  document.getElementById('predict').addEventListener(
      'click', predictHandler, false);
  ui.displayExpectedSimulations();
});
