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

/**
 * TensorFlow.js Reinforcement Learning Example: Balancing a Cart-Pole System.
 *
 * The simulation, training, testing and visualization parts are written
 * purely in JavaScript and can run in the web browser with WebGL acceleration.
 *
 * This reinforcement learning (RL) problem was proposed in:
 *
 * - Barto, Sutton, and Anderson, "Neuronlike Adaptive Elements That Can Solve
 *   Difficult Learning Control Problems," IEEE Trans. Syst., Man, Cybern.,
 *   Vol. SMC-13, pp. 834--846, Sept.--Oct. 1983
 * - Sutton, "Temporal Aspects of Credit Assignment in Reinforcement Learning",
 *   Ph.D. Dissertation, Department of Computer and Information Science,
 *   University of Massachusetts, Amherst, 1984.
 *
 * It later became one of OpenAI's gym environmnets:
 *   https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
 */

import * as tf from '@tensorflow/tfjs';

import {maybeRenderDuringTraining, onGameEnd, setUpUI} from './ui';

/**
 * Policy network for controlling the cart-pole system.
 */
class PolicyNetwork {
  /**
   * Constructor of PolicyNetwork.
   *
   * @param {number | number[] | tf.Model} hiddenLayerSizes
   *   Can be any of the following
   *   - Size of the hidden layer, as a single number (for a single hidden
   *     layer)
   *   - An Array of numbers (for any number of hidden layers).
   *   - An instance of tf.Model.
   */
  constructor(hiddenLayerSizesOrModel) {
    if (hiddenLayerSizesOrModel instanceof tf.Model) {
      this.model_ = hiddenLayerSizesOrModel;
    } else {
      this.createModel_(hiddenLayerSizesOrModel);
    }


    this.oneTensor_ = tf.scalar(1);
  }

  /**
   * Create the underlying model of this policy network.
   *
   * @param {number | number[]} hiddenLayerSizes Size of the hidden layer, as
   *   a single number (for a single hidden layer) or an Array of numbers (for
   *   any number of hidden layers).
   */
  createModel_(hiddenLayerSizes) {
    if (!Array.isArray(hiddenLayerSizes)) {
      hiddenLayerSizes = [hiddenLayerSizes];
    }
    this.model_ = tf.sequential();
    for (const hiddenLayerSize of hiddenLayerSizes) {
      this.model_.add(tf.layers.dense({
        units: hiddenLayerSize,
        activation: 'elu',
        inputShape: [4]
      }));
    }
    this.model_.add(tf.layers.dense({units: 1}));
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

  /**
   * Train the policy network's model.
   *
   * @param {CartPole} cartPoleSystem The cart-pole system object to use during
   *   training.
   * @param {tf.train.Optimizer} optimizer An instance of TensorFlow.js
   *   Optimizer to use for training.
   * @param {number} discountRate Reward discounting rate: a number between 0
   *   and 1.
   * @param {number} numGames Number of game to play for each model parameter
   *   update.
   * @param {number} maxStepsPerGame Maximum number of steps to perform during
   *   a game. If this number is reached, the game will end immediately.
   * @returns {number[]} The number of steps completed in the `numGames` games
   *   in this round of training.
   */
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

        await maybeRenderDuringTraining(cartPoleSystem);

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

  /**
   * Push a new dictionary of gradients into records.
   *
   * @param {{[varName: string]: tf.Tensor[]}} record The record of variable
   *   gradient: a map from variable name to the Array of gradient values for
   *   the variable.
   * @param {{[varName: string]: tf.Tensor}} gradients The new gradients to push
   *   into `record`: a map from variable name to the gradient Tensor.
   */
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

// The IndexedDB path where the model of the policy network will be saved.
const MODEL_SAVE_PATH_ = 'indexeddb://cart-pole-v1';

/**
 * A subclass of PolicyNetwork that supports saving and loading.
 */
export class SaveablePolicyNetwork extends PolicyNetwork {

  /**
   * Constructor of SaveablePolicyNetwork
   *
   * @param {number | number[]} hiddenLayerSizesOrModel
   */
  constructor(hiddenLayerSizesOrModel) {
    super(hiddenLayerSizesOrModel);
  }

  /**
   * Save the model to IndexedDB.
   */
  async saveModel() {
    return await this.model_.save(MODEL_SAVE_PATH_);
  }

  /**
   * Load the model fom IndexedDB.
   *
   * @returns {SaveablePolicyNetwork} The instance of loaded
   *   `SaveablePolicyNetwork`.
   * @throws {Error} If no model can be found in IndexedDB.
   */
  static async loadModel() {
    const modelsInfo = await tf.io.listModels();
    if (MODEL_SAVE_PATH_ in modelsInfo) {
      console.log(`Loading existing model...`);
      const model = await tf.loadModel(MODEL_SAVE_PATH_);
      console.log(`Loaded model from ${MODEL_SAVE_PATH_}`);
      return new SaveablePolicyNetwork(model);
    } else {
      throw new Error(`Cannot find model at ${MODEL_SAVE_PATH_}.`);
    }
  }

  /**
   * Check the status of locally saved model.
   *
   * @returns If the locally saved model exists, the model info as a JSON
   *   object. Else, `undefined`.
   */
  static async checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[MODEL_SAVE_PATH_];
  }

  /**
   * Remove the locally saved model from IndexedDB.
   */
  async removeModel() {
    return await tf.io.removeModel(MODEL_SAVE_PATH_);
  }

  /**
   * Get the sizes of the hidden layers.
   *
   * @returns {number | number[]} If the model has only one hidden layer,
   *   return the size of the layer as a single number. If the model has
   *   multiple hidden layers, return the sizes as an Array of numbers.
   */
  hiddenLayerSizes() {
    const sizes = [];
    for (let i = 0; i < this.model_.layers.length - 1; ++i) {
      sizes.push(this.model_.layers[i].units);
    }
    return sizes.length === 1 ? sizes[0] : sizes;
  }
}

/**
 * Discount the reward values.
 *
 * @param {number[]} rewards The reward values to be discounted.
 * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
 *   0.95.
 * @returns {tf.Tensor} The discounted reward values as a 1D tf.Tensor.
 */
function discountRewards(rewards, discountRate) {
  const discountedBuffer = tf.buffer([rewards.length]);
  let prev = 0;
  for (let i = rewards.length - 1; i >= 0; --i) {
    const current = discountRate * prev + rewards[i];
    discountedBuffer.set(current, i);
    prev = current;
  }
  return discountedBuffer.toTensor();
}

/**
 * Discount and normalize reward values.
 *
 * This function perrforms two steps:
 *
 * 1. Discounts the reward values using `discountRate`.
 * 2. Normalize the reward values with the global reward mean and standard
 *    deviation.
 *
 * @param {number[][]} rewardSequences Sequences of reward values.
 * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
 *   0.95.
 * @returns {tf.Tensor[]} The discounted and normalize reward values as an
 *   Array of tf.Tensor.
 */
function discountAndNormalizeRewards(rewardSequences, discountRate) {
  return tf.tidy(() => {
    const discounted = [];
    for (const sequence of rewardSequences) {
      discounted.push(discountRewards(sequence, discountRate))
    }
    // Compute the overall mean and stddev.
    const concatenated = tf.concat(discounted);
    const mean = tf.mean(concatenated);
    const std = tf.sqrt(tf.mean(tf.square(concatenated.sub(mean))));
    // Normalize the reward sequences using the mean and std.
    const normalized = discounted.map(rs => rs.sub(mean).div(std));
    return normalized;
  });
}

/**
 * Scale the gradient values using normalized reward values and compute average.
 *
 * The gradient values are scaled by the normalized reward values. Then they
 * are averaged across all games and all steps.
 *
 * @param {{[varName: string]: tf.Tensor[][]}} allGradients A map from variable
 *   name to all the gradient values for the variable across all games and all
 *   steps.
 * @param {tf.Tensor[]} normalizedRewards An Array of normalized reward values
 *   for all the games. Each element of the Array is a 1D tf.Tensor of which
 *   the length equals the number of steps in the game.
 * @returns {{[varName: string]: tf.Tensor}} Scaled and averaged gradients
 *   for the variables.
 */
function scaleAndAverageGradients(allGradients, normalizedRewards) {
  return tf.tidy(() => {
    const gradients = {};
    for (const varName in allGradients) {
      gradients[varName] = tf.tidy(() => {
        // Stack gradients together.
        const varGradients = allGradients[varName].map(
            varGameGradients => tf.stack(varGameGradients));
        // Expand dimensions of reward tensors to prepare for multiplication
        // with broadcasting.
        const expandedDims = [];
        for (let i = 0; i < varGradients[0].rank - 1; ++i) {
          expandedDims.push(1);
        }
        const reshapedNormalizedRewards = normalizedRewards.map(
            rs => rs.reshape(rs.shape.concat(expandedDims)));
        for (let g = 0; g < varGradients.length; ++g) {
          // This mul() call uses broadcasting.
          varGradients[g] = varGradients[g].mul(reshapedNormalizedRewards[g]);
        }
        // Concatenate the scaled gradients together, then average them across
        // all the steps of all the games.
        return tf.mean(tf.concat(varGradients, 0), 0);
      });
    }
    return gradients;
  });
}

setUpUI();
