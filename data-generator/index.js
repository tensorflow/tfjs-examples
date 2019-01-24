
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
 * Collects one random play of the game.  Processes the sample to generate  features and labels representation
 * of the play.  Calls a UI method to render the sample and the processed sample.
 */
async function simulateGameHandler() {
  const sample = game.generateOnePlay();
  const featuresAndLabel = gameToFeaturesAndLabel(sample);
    ui.displaySimulation(sample, featuresAndLabel);
}

/**
 * Creates a dataset pipeline from GAME_GENERATOR_DATASET by:
 * 1) Applying the function gameToFeaturesAndlabel
 * 2) Taking the first N samples of the dataset
 * 3) Batching the dataset to batches of size B
 * 
 * It then executes the dataset by filling an array.  Finally, it passes this array
 * to the UI to render in a table.
 */
async function datasetToArrayHandler() {
  const arr = await GAME_GENERATOR_DATASET
    .map(gameToFeaturesAndLabel)
    .take(ui.getTake())
    .batch(ui.getBatchSize())
    .toArray();
  ui.displayBatches(arr);
}

/** Sets up handlers for the user affordences, including all buttons. */
document.addEventListener('DOMContentLoaded', async () => {
  console.log('content loaded... connecting buttons.');
  document.getElementById('simulate-game')
    .addEventListener('click', simulateGameHandler, false);
  document.getElementById('dataset-to-array')
      .addEventListener('click', datasetToArrayHandler, false);
});
