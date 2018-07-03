/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import embed from 'vega-embed';
import * as tf from '@tensorflow/tfjs';

import {CartPole} from './cart_pole';
import {SaveablePolicyNetwork} from './index';
import {mean} from './utils';

const appStatus = document.getElementById('app-status');
const storedModelStatusInput = document.getElementById('stored-model-status');
const deleteStoredModelButton = document.getElementById('delete-stored-model');
const cartPoleCanvas = document.getElementById('cart-pole-canvas');

const numIterationsInput = document.getElementById('num-iterations');
const gamesPerIterationInput = document.getElementById('games-per-iteration');
const discountRateInput = document.getElementById('discount-rate');
const maxStepsPerGameInput = document.getElementById('max-steps-per-game');
const learningRateInput = document.getElementById('learning-rate');

const trainButton = document.getElementById('train');
const testButton = document.getElementById('test');
const iterationStatus = document.getElementById('iteration-status');
const iterationProgress = document.getElementById('iteration-progress');
const trainStatus = document.getElementById('train-status');
const trainProgress = document.getElementById('train-progress');

let policyNet;

function logStatus(message) {
  appStatus.textContent = message;
}

export function onGameEnd(gameCount, totalGames) {
  iterationStatus.textContent = `Game ${gameCount} of ${totalGames}`;
  iterationProgress.value = gameCount / totalGames * 100;
}

function onIterationEnd(iterationCount, totalIterations) {
  trainStatus.textContent =
      `Iteration ${iterationCount} of ${totalIterations}`;
  trainProgress.value =iterationCount / totalIterations * 100;
}

let meanStepValues = [];
function plotSteps() {
  embed(
      '#steps-canvas', {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'data': {'values': meanStepValues},
        'mark': 'line',
        'encoding': {
          'x': {'field': 'iteration', 'type': 'ordinal'},
          'y': {'field': 'meanSteps', 'type': 'quantitative'},
        },
        'width': 300,
      },
      {});
}

// TODO(cais): Move to ui.js.
function disableUI() {
  trainButton.disabled = true;
  testButton.disabled = true;
  deleteStoredModelButton.disabled = true;
}

function enableUI() {
  trainButton.disabled = false;
  testButton.disabled = false;
  deleteStoredModelButton.disabled = false;
}

async function updateLocallyStoredModelStatus() {
  const modelInfo = await policyNet.checkStoredModelStatus();
  if (modelInfo == null) {
    storedModelStatusInput.value = 'No stored model.';
    deleteStoredModelButton.disabled = true;
  } else {
    storedModelStatusInput.value =
        `Saved @ ${modelInfo.dateSaved.toISOString()}`;
    deleteStoredModelButton.disabled = false;
  }
}

export async function setUpUI() {
  const cartPole = new CartPole(true);
  policyNet = new SaveablePolicyNetwork(5);

  console.log('policyNet:', policyNet);  // DEBUG
  if (await policyNet.checkStoredModelStatus() != null) {
    policyNet.loadModel();
  }
  await updateLocallyStoredModelStatus();

  cartPole.render(cartPoleCanvas);

  deleteStoredModelButton.addEventListener('click', async () => {
    if (confirm(
        `Are you sure you want to delete the locally-stored model?`)) {
      await policyNet.removeModel();
    }
    policyNet = null;
    await updateLocallyStoredModelStatus();
  });

  trainButton.addEventListener('click', async () => {
    disableUI();
    const trainIterations = Number.parseInt(numIterationsInput.value);
    const gamesPerIteration = Number.parseInt(gamesPerIterationInput.value);
    const discountRate = Number.parseFloat(discountRateInput.value);
    const maxStepsPerGame = Number.parseInt(maxStepsPerGameInput.value);
    const learningRate = Number.parseFloat(learningRateInput.value);
    // TODO(cais): Value sanity checks.

    const optimizer = tf.train.adam(learningRate);

    meanStepValues = [];
    onIterationEnd(0, trainIterations);
    for (let i = 0; i < trainIterations; ++i) {
      const gameSteps = await policyNet.train(
          cartPole, optimizer, discountRate, gamesPerIteration, maxStepsPerGame);
      meanStepValues.push({
        iteration: i + 1,
        meanSteps: mean(gameSteps)
      });
      console.log(`# of tensors: ${tf.memory().numTensors}`);  // DEBUG
      plotSteps();
      onIterationEnd(i + 1, trainIterations);
      await tf.nextFrame();
    }
    await policyNet.saveModel();
    await updateLocallyStoredModelStatus();
    enableUI();
  });

  // TODO(cais): Move to index.js?
  testButton.addEventListener('click', async () => {
    disableUI();
    let isDone = false;
    const cartPole = new CartPole(true);
    cartPole.setRandomState();
    let steps = 0;
    while (!isDone) {
      steps++;
      tf.tidy(() => {
        const action = policyNet.getLogitsAndActions(
            cartPole.getStateTensor())[1].dataSync()[0];
        logStatus(`Test in progress. Action: ${action === 1 ? '←' : ' →'} (Step ${steps})`);
        isDone = cartPole.update(action);
        cartPole.render(cartPoleCanvas);
      });
      await tf.nextFrame();
    }
    logStatus(`Testing finished. Survived ${steps} step(s).`);
    console.log(`# of tensors: ${tf.memory().numTensors}`);  // DEBUG
    enableUI();
  });
}
