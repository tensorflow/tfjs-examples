
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
 * Takes the state of one complete game and returns features suitable for
 * training.  Specifically it removes features identifying the opponent' hand
 * and divides the training features(player 1's hand) from the target (whether
 * player one wins).
 * @param {*} gameState
 */
export function gameToFeaturesAndLabel(gameState) {
  const features = gameState[0];
  const label = gameState[2];
  return {features, label};
}

/**
 * Collects one random play of the game.  Processes the sample to generate
 * features and labels representation of the play.  Calls a UI method to render
 * the sample and the processed sample.
 */
async function simulateGameHandler() {
  const sample = game.generateOnePlay();
  const featuresAndLabel = gameToFeaturesAndLabel(sample);
  ui.displaySimulation(sample, featuresAndLabel);
}

/** @see datasetToArrayHandler */
async function datasetToArray() {
  const arr = await GAME_GENERATOR_DATASET.map(gameToFeaturesAndLabel)
                  .take(ui.getTake())
                  .batch(ui.getBatchSize())
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
  ui.displayBatches(datasetToArray());
}

function createLinearModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1 }));
  return model;
}

async function trainModelUsingFitDataset(model, dataset) {
  const fitDatasetConfig = {
    epochs: params.epochs,
    validationData: validationDataset,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // Plot the loss and accuracy values at the end of every training epoch.
        const secPerEpoch =
          (performance.now() - beginMs) / (1000 * (epoch + 1));
        ui.status(
          `Training model... Approximately ` +
          `${secPerEpoch.toFixed(4)} seconds per epoch`);
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ['loss', 'val_loss'])
        tfvis.show.history(accContainer, trainLogs, ['acc', 'val_acc'])
        const [[xTest, yTest]] = await validationDataset.toArray();
        calculateAndDrawConfusionMatrix(model, xTest, yTest);
      },
    }
  };
  await model.fitDataset(dataset, fitDatasetConfig);
}




async function trainModelUsingFitDatasetHandler() {
  console.log('I am train model using fit DATASET handler');
  const model = createLinearModel();
  model.compile({
    optimizer: 'rmsprop',
    loss: 'mse',
    metrics: ['accuracy'],
  });
  trainModelUsingFitDataset(model, GAME_GENERATOR_DATASET);
}

/** Sets up handlers for the user affordences, including all buttons. */
document.addEventListener('DOMContentLoaded', async () => {
  console.log('content loaded... connecting buttons.');
  document.getElementById('simulate-game')
      .addEventListener('click', simulateGameHandler, false);
  document.getElementById('dataset-to-array')
      .addEventListener('click', datasetToArrayHandler, false);
  document.getElementById('train-model-using-fit-dataset')
      .addEventListener('click', trainModelUsingFitDatasetHandler, false);
});
