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

import {ALL_ACTIONS, getStateTensor, SnakeGame} from './snake_game';
import {renderSnakeGame} from './snake_graphics';

const gameCanvas = document.getElementById('game-canvas');

const loadHostedModelButton = document.getElementById('load-hosted-model');

const stepButton = document.getElementById('step');
const resetButton = document.getElementById('reset');
const autoPlayStopButton = document.getElementById('auto-play-stop');
const gameStatusSpan = document.getElementById('game-status');
const showQValuesCheckbox = document.getElementById('show-q-values');

let game;
let qNet;

let cumulativeReward = 0;
let cumulativeFruits = 0;
let autoPlaying = false;
let autoPlayIntervalJob;

/** Reset the game state. */
async function reset() {
  if (game == null) {
    return;
  }
  game.reset();
  await calcQValuesAndBestAction();
  renderSnakeGame(gameCanvas, game,
      showQValuesCheckbox.checked ? currentQValues : null);
  gameStatusSpan.textContent = 'Game started.';
  stepButton.disabled = false;
  autoPlayStopButton.disabled = false;
}

/**
 * Play a game for one step.
 *
 * - Use the current best action to forward one step in the game.
 * - Accumulate to the cumulative reward.
 * - Determine if the game is over and update the UI accordingly.
 * - If the game has not ended, calculate the current Q-values and best action.
 * - Render the game in the canvas.
 */
async function step() {
  const {reward, done, fruitEaten} = game.step(bestAction);
  invalidateQValuesAndBestAction();
  cumulativeReward += reward;
  if (fruitEaten) {
    cumulativeFruits++;
  }
  gameStatusSpan.textContent =
      `Reward=${cumulativeReward.toFixed(1)}; Fruits=${cumulativeFruits}`;
  if (done) {
    gameStatusSpan.textContent += '. Game Over!';
    cumulativeReward = 0;
    cumulativeFruits = 0;
    if (autoPlayIntervalJob) {
      clearInterval(autoPlayIntervalJob);
      autoPlayStopButton.click();
    }
    autoPlayStopButton.disabled = true;
    stepButton.disabled = true;
  }
  await calcQValuesAndBestAction();
  renderSnakeGame(gameCanvas, game,
      showQValuesCheckbox.checked ? currentQValues : null);
}

let currentQValues;
let bestAction;

/** Calculate the current Q-values and the best action. */
async function calcQValuesAndBestAction() {
  if (currentQValues != null) {
    return;
  }
  tf.tidy(() => {
    const stateTensor = getStateTensor(game.getState(), game.height, game.width);
    const predictOut = qNet.predict(stateTensor);
    currentQValues = predictOut.dataSync();
    bestAction = ALL_ACTIONS[predictOut.argMax(-1).dataSync()[0]];
  });
}

function invalidateQValuesAndBestAction() {
  currentQValues = null;
  bestAction = null;
}

const LOCAL_MODEL_URL = './dqn/model.json';
const REMOTE_MODEL_URL = 'https://storage.googleapis.com/tfjs-examples/snake-dqn/models/model.json';

function enableGameButtons() {
  autoPlayStopButton.disabled = false;
  stepButton.disabled = false;
  resetButton.disabled = false;
}

async function initGame() {
  game = new SnakeGame({
    height: 9,
    width: 9,
    numFruits: 1,
    initLen: 2
  });

  // Warm up qNet.
  for (let i = 0; i < 3; ++i) {
    qNet.predict(getStateTensor(game.getState(), game.height, game.width));
  }

  await reset();

  stepButton.addEventListener('click', step);

  autoPlayStopButton.addEventListener('click', () => {
    if (autoPlaying) {
      autoPlayStopButton.textContent = 'Auto Play';
      if (autoPlayIntervalJob) {
        clearInterval(autoPlayIntervalJob);
      }
    } else {
      autoPlayIntervalJob = setInterval(() => {
        step(game, qNet);
      }, 100);
      autoPlayStopButton.textContent = 'Stop';
    }
    autoPlaying = !autoPlaying;
    stepButton.disabled = autoPlaying;
  });

  resetButton.addEventListener('click',  () => reset(game));
}

(async function() {
  try {
    qNet = await tf.loadLayersModel(LOCAL_MODEL_URL);
    loadHostedModelButton.textContent = `Loaded model from ${LOCAL_MODEL_URL}`;
    initGame();
    enableGameButtons();
  } catch (err) {
    console.log('Loading local model failed.');
    loadHostedModelButton.disabled = false;
  }

  loadHostedModelButton.addEventListener('click', async () => {
    try {
      qNet = await tf.loadLayersModel(REMOTE_MODEL_URL);
      loadHostedModelButton.textContent = `Loaded hosted model.`;
      loadHostedModelButton.disabled = true;
      initGame();
      enableGameButtons();
    } catch (err) {
      loadHostedModelButton.textContent = 'Failed to load model.'
      loadHostedModelButton.disabled = true;
    }
  });
})();
