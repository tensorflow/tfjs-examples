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

import {onGameEnd, setUpUI} from './ui';

/**
 * Policy network for controlling the cart-pole system.
 */
class PolicyNetwork {
  /**
   * Constructor of PolicyNetwork.
   *
   * @param {number | number[]} hiddenLayerSizes Size of the hidden layer, as
   *   a single number (for a single hidden layer) or an Array of numbers (for
   *   any number of hidden layers).
   */
  constructor(hiddenLayerSizes) {
    this.model_ = tf.sequential();
    this.model_.add(tf.layers.dense({
      units: hiddenLayerSizes,
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

export class SaveablePolicyNetwork extends PolicyNetwork {
  constructor(hiddenLayerSizes) {
    super(hiddenLayerSizes);
    this.MODEL_SAVE_PATH_ = 'indexeddb://cart-pole-v1';
  }

  async saveModel() {
    return await this.model_.save(this.MODEL_SAVE_PATH_);
  }

  async loadModel() {
    const modelsInfo = await tf.io.listModels();
    if (this.MODEL_SAVE_PATH_ in modelsInfo) {
      console.log(`Loading existing model...`);
      this.model_ = await tf.loadModel(this.MODEL_SAVE_PATH_);
      console.log(`Loaded model from ${this.MODEL_SAVE_PATH_}`);
    } else {
      throw new Error(
          `Cannot find model at ${this.MODEL_SAVE_PATH_}. ` +
          `Creating model from scratch.`);
    }
  }

  /**
   * Remove the locally saved model from IndexedDB.
   */
  async removeModel() {
    if (await this.checkStoredModelStatus() == null) {
      throw new Error(
          'Cannot remove locally saved model because it does not exist.');
    }
    return await tf.io.removeModel(this.MODEL_SAVE_PATH_);
  }

  /**
   * Check the status of locally saved model.
   *
   * @returns If the locally saved model exists, the model info as a JSON
   *   object. Else, `undefined`.
   */
  async checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[this.MODEL_SAVE_PATH_];
  }
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

setUpUI();
