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

import * as tf from '@tensorflow/tfjs';
import embed from 'vega-embed';

import {CartPole} from './cart_pole';

/**
 * Policy network for controlling the cart-pole system.
 */
class PolicyNetwork {
  /**
   * Constructor of PolicyNetwork.
   *
   * @param {number} hiddenLayerSize Size of the hidden layer.
   */
  constructor(hiddenLayerSize) {
    this.model_ = tf.sequential();
    this.model_.add(tf.layers.dense({
      units: hiddenLayerSize,
      activation: 'elu',
      inputShape: [4]
    }));
    this.model_.add(tf.layers.dense({units: 1}));
    this.oneTensor_ = tf.scalar(1);
  }

  getGradientsAndSaveActions(inputTensor) {
    const f = () => tf.tidy(() => {
      const [logits, actions] = this.getLogitsAndActions(inputTensor);
      this.currentActions_ = actions.dataSync();
      const labels = this.oneTensor_.sub(
          tf.tensor2d(this.currentActions_, actions.shape));
      return tf.sigmoidCrossEntropyWithLogits(labels, logits).asScalar();
    });
    return tf.variableGrads(f);
  }

  getCurrentActions() {
    return this.currentActions_;
  }

  /**
   * Get action based  on a state tensor.
   *
   * @param {tf.Tensor} inputs A tf.Tensor instance of shape `[batchSize, 4]`.
   * @returns {Float32Array} 0-1 action values for all the examples in the batch,
   *   length = batchSize.
   */
  getLogitsAndActions(inputs) {
    return tf.tidy(() => {
      const logits = this.model_.predict(inputs);

      // Get the probability of the left word action.
      const leftProb = tf.sigmoid(logits);
      // Probabilites of the left and right actions.
      const leftRightProbs =
          tf.concat([leftProb, this.oneTensor_.sub(leftProb)], 1);
      const actions = tf.multinomial(leftRightProbs, 1, null, true);
      return [logits, actions];
    });
  }

  async train(cartPoleSystem,
              optimizer,
              discountRate,
              numGames,
              maxStepsPerGame) {
      const allGradients = [];
      const allRewards = [];
      const gameSteps = [];
    onGameEnd(0, numGames);
    for (let i = 0; i < numGames; ++i) {
      cartPoleSystem.setRandomState();
      const gameRewards = [];
      const gameGradients = [];
      for (let j = 0; j < maxStepsPerGame; ++j) {
        const gradients = tf.tidy(() => {
          const inputTensor = cartPoleSystem.getStateTensor();
          return this.getGradientsAndSaveActions(inputTensor).grads;
        });

        this.pushGradients_(gameGradients, gradients);
        const action = this.currentActions_[0];
        const isDone = cartPoleSystem.update(action);
        // cartPoleSystem.render(cartPoleCanvas);
        // wait tf.nextFrame();
        if (isDone) {
          gameRewards.push(0);
          onGameEnd(i + 1, numGames);
          break;
        } else {
          gameRewards.push(1);
        }
        if (j >= maxStepsPerGame) {
          onGameEnd(i + 1, numGames);
          break;
        }
      }
      gameSteps.push(gameRewards.length);
      this.pushGradients_(allGradients, gameGradients);
      allRewards.push(gameRewards);
      await tf.nextFrame();
    }
    console.log(`game steps = ${gameSteps}, mean = ${mean(gameSteps)}`);

    tf.tidy(() => {
      const normalizedRewards =
          discountAndNormalizeRewards(allRewards, discountRate);
      const gradientsToApply =
          scaleAndAverageGradients(allGradients, normalizedRewards);
      optimizer.applyGradients(gradientsToApply);
    });
    tf.dispose(allGradients);
    return gameSteps;
  }

  pushGradients_(record, gradients) {
    for (const key in gradients) {
      if (key in record) {
        record[key].push(gradients[key]);
      } else {
        record[key] = [gradients[key]];
      }
    }
  }
}

function mean(xs) {
  return xs.reduce((x, prev) => prev + x) / xs.length;
}

function discountRewards(rewards, discountRate) {
  const discounted = [];
  for (let i = rewards.length - 1; i >=0; --i) {
    const reward = rewards[i];
    const prevReward =
        discounted.length > 0 ? discounted[discounted.length - 1] : 0;
    discounted.push(discountRate * prevReward + reward);
  }
  discounted.reverse();
  return discounted;
}

function discountAndNormalizeRewards(rewardSequences, discountRate) {
  return tf.tidy(() => {
    const discounted = [];
    for (const sequence of rewardSequences) {
      discounted.push(discountRewards(sequence, discountRate))
    }

    // Compute the overall mean and stddev.
    const flattened = [];
    for (const sequence of discounted) {
      flattened.push(...sequence);
    }
    const [mean, std] = tf.tidy(() => {
      const r = tf.tensor1d(flattened);
      const mean = tf.mean(r);
      const std = tf.sqrt(tf.mean(tf.square(r.sub(mean))));
      return [mean.dataSync()[0], std.dataSync()[0]];
    });

    // TODO(cais): Maybe normalized should be a tf.Tensor.
    const normalized = [];
    for (const rs of discounted) {
      normalized.push(rs.map(r => (r - mean) / std));
    }
    return normalized;
  });
}

function scaleAndAverageGradients(allGradients, normalizedRewards) {
  return tf.tidy(() => {
    const rewardScalars = [];
    for (const rewardSequence of normalizedRewards) {
      const rewardScalarSequence = rewardSequence.map(r => tf.scalar(r));
      rewardScalars.push(rewardScalarSequence);
    }

    // TODO(cais): Use tighter tidy() scopes.
    const gradients = {};

    for (const varName in allGradients) {
      const varGradients = allGradients[varName];

      const numGames = varGradients.length;
      gradients[varName] = tf.tidy(() => {
        let numGradients = 0;
        let sum = tf.zerosLike(varGradients[0][0]);
        for (let g = 0; g < numGames; ++g) {
          const numSteps = varGradients[g].length;
          for (let s = 0; s < numSteps; ++s) {
            // TODO(cais): Use broadcasting, vectorized multiplication for
            //   performance?
            const scaledGradients =
                varGradients[g][s].mul(rewardScalars[g][s]);
            sum = sum.add(scaledGradients);
            numGradients++;
          }
        }
        return sum.div(tf.scalar(numGradients));
      });
    }
    return gradients;
  });
}

const policyNet = new PolicyNetwork(5);

const cartPole = new CartPole(true);

// TODO(cais): Move to ui.js.
const cartPoleCanvas = document.getElementById('cart-pole-canvas');
const numIterationsInput = document.getElementById('num-iterations');
const leftButton = document.getElementById('left');
const rightButton = document.getElementById('right');
const trainButton = document.getElementById('train');
const testButton = document.getElementById('test');
const testResult = document.getElementById('test-result');
const iterationStatus = document.getElementById('iteration-status');
const iterationProgress = document.getElementById('iteration-progress');
const trainStatus = document.getElementById('train-status');
const trainProgress = document.getElementById('train-progress');

cartPole.render(cartPoleCanvas);

leftButton.addEventListener('click', () => {
  cartPole.update(-1);
  cartPole.render(cartPoleCanvas);
});

rightButton.addEventListener('click', () => {
  cartPole.update(1);
  cartPole.render(cartPoleCanvas);
});


let meanStepValues = [];

function onGameEnd(gameCount, totalGames) {
  iterationStatus.textContent = `Game ${gameCount} of ${totalGames}`;
  iterationProgress.value = gameCount / totalGames * 100;
}

function onIterationEnd(iterationCount, totalIterations) {
  trainStatus.textContent =
      `Iteration ${iterationCount} of ${totalIterations}`;
  trainProgress.value =iterationCount / totalIterations * 100;
}

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
}

function enableUI() {
  trainButton.disabled = false;
  testButton.disabled = false;
}


trainButton.addEventListener('click', async () => {
  disableUI();
  const trainIterations = Number.parseInt(numIterationsInput.value);
  // TODO(cais): Value sanity checks.
  const discountRate = 0.95;
  const numGames = 20;
  const maxStepsPerGame = 200;
  const learningRate = 0.05;

  const optimizer = tf.train.adam(learningRate);

  meanStepValues = [];
  onIterationEnd(0, trainIterations);
  for (let i = 0; i < trainIterations; ++i) {
    const gameSteps = await policyNet.train(
        cartPole, optimizer, discountRate, numGames, maxStepsPerGame);
    meanStepValues.push({
      iteration: i + 1,
      meanSteps: mean(gameSteps)
    });
    console.log(`# of tensors: ${tf.memory().numTensors}`);  // DEBUG
    plotSteps();
    onIterationEnd(i + 1, trainIterations);
    await tf.nextFrame();
  }
  enableUI();
});

testButton.addEventListener('click', async () => {
  disableUI();
  let isDone = false;
  const cartPole = new CartPole(true);
  cartPole.setRandomState();
  let steps = 0;
  while (!isDone) {
    steps++;
    tf.tidy(() => {
      const action =
          policyNet.getLogitsAndActions(cartPole.getStateTensor())[1].dataSync()[0];
      isDone = cartPole.update(action);
      cartPole.render(cartPoleCanvas);
    });
    await tf.nextFrame();
  }
  testResult.textContent = `Survived ${steps} step(s).`;
  enableUI();
});
