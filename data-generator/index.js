
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

export function playNTimes(numRows) {
  const rows = [];
  for (let i = 0; i < numRows; i++) {
    rows.push(game.generateOnePlay());
  }
  return rows;
}

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

/** Sets up handlers for the user affordences, including all buttons. */
document.addEventListener('DOMContentLoaded', async () => {
  console.log('content loaded... connecting buttons.');
  document.getElementById('simulate-game')
    .addEventListener('click', async () => {
      const sample = playNTimes(1)[0];
      const featuresAndLabel = gameToFeaturesAndLabel(sample);
        ui.updateSimulationOutput(sample, featuresAndLabel);
      }, false);
  document.getElementById('dataset-to-array')
      .addEventListener('click', async () => {
        ui.datasetToArrayHandler(GAME_GENERATOR_DATASET.map(gameToFeaturesAndLabel));
      }, false);
});
