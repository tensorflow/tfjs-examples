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

import * as argparse from 'argparse';

// The value of tf (TensorFlow.js-Node module) will be set dynamically
// depending on the value of the --gpu flag below.
let tf;

import {SnakeGameAgent} from './agent';
import {copyWeights} from './dqn';
import {SnakeGame} from './snake_game';

/**
 * Train an agent to play the snake game.
 *
 * @param {SnakeGameAgent} agent The agent to train.
 * @param {number} batchSize Batch size for training.
 * @param {number} gamma Reward discount rate. Must be a number >= 0 and <= 1.
 * @param {number} learnigRate
 * @param {number} cumulativeRewardThreshold The threshold of cumulative reward
 *   from a single game. The training stops as soon as this threshold is achieved.
 * @param {number} syncEveryFrames The frequency at which the weights are copied
 *   from the online DQN of the agent to the target DQN, in number of frames.
 * @param {string} savePath Path to which the online DQN of the agent will be
 *   saved upon the completion of the training.
 */
export async function train(
    agent, batchSize, gamma, learningRate, cumulativeRewardThreshold,
    syncEveryFrames, savePath) {
  for (let i = 0; i < agent.replayBufferSize; ++i) {
    agent.playStep();
  }

  const optimizer = tf.train.adam(learningRate);
  while (true) {
    agent.trainOnReplayBatch(batchSize, gamma, optimizer);
    const {cumulativeReward, done} = agent.playStep();
    if (done) {
      console.log(`Frame #${agent.frameCount}: ` +
          `cumulativeReward = ${cumulativeReward}`);
      if (cumulativeReward >= cumulativeRewardThreshold) {
        // TODO(cais): Save online network.
        break;
      }
    }
    if (agent.frameCount % syncEveryFrames === 0) {
      console.log('Copying weights from online network to target network');
      copyWeights(agent.targetNetwork, agent.onlineNetwork);
    }
  }

  await agent.onlineNetwork.save(`file://${savePath}`);
}

export function parseArguments() {
  const parser = new argparse.ArgumentParser({
    description: 'Training script for a DQN that plays the snake game'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Whether to use tfjs-node-gpu for training ' +
    '(requires CUDA GPU, drivers, and libraries).'
  });
  parser.addArgument('--height', {
    type: 'int',
    defaultValue: 9,
    help: 'Height of the game board.'
  });
  parser.addArgument('--width', {
    type: 'int',
    defaultValue: 9,
    help: 'Width of the game board.'
  });
  parser.addArgument('--numFruits', {
    type: 'int',
    defaultValue: 1,
    help: 'Number of fruits present on the board at any given time.'
  });
  parser.addArgument('--initLen', {
    type: 'int',
    defaultValue: 2,
    help: 'Initial length of the snake, in number of squares.'
  });
  parser.addArgument('--cumulativeRewardThreshold', {
    type: 'float',
    defaultValue: 200,
    help: 'Threshold for cumulative reward (its moving average over the 100 ' +
    'latest games. Training stops as soon as this threshold is reached.'
  });
  parser.addArgument('--replayBufferSize', {
    type: 'int',
    defaultValue: 1e4,
    help: 'Length of the replay memory buffer.'
  });
  parser.addArgument('--epsilonInit', {
    type: 'float',
    defaultValue: 1,
    help: 'Initial value of epsilon, used for the epsilon-greedy algorithm.'
  });
  parser.addArgument('--epsilonFinal', {
    type: 'float',
    defaultValue: 1,
    help: 'Final value of epsilon, used for the epsilon-greedy algorithm.'
  });
  parser.addArgument('--epsilonDecayFrames', {
    type: 'int',
    defaultValue: 2e5,
    help: 'Number of frames of game over which the value of epsilon ' +
    'decays from epsilonInit to epsilonFinal'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 64,
    help: 'Batch size for DQN training.'
  });
  parser.addArgument('--gamma', {
    type: 'float',
    defaultValue: 0.99,
    help: 'Reward discount rate.'
  });
  parser.addArgument('--learningRate', {
    type: 'float',
    defaultValue: 1e-3,
    help: 'Learning rate for DQN training.'
  });
  parser.addArgument('--syncEveryFrames', {
    type: 'int',
    defaultValue: 1e3,
    help: 'Frequency at which weights are sync\'ed from the online network ' +
    'to the target network.'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArguments();
  if (args.gpu) {
    tf = require('@tensorflow/tfjs-node-gpu');
  } else {
    tf = require('@tensorflow/tfjs-node');
  }
  console.log(`args: ${JSON.stringify(args, null, 2)}`);

  const game = new SnakeGame({
    height: args.height,
    width: args.width,
    numFruits: args.numFruits,
    initLen: args.initLen
  });
  const agent = new SnakeGameAgent(game, {
    gamma: args.gamma,
    replayBufferSize: args.replayBufferSize,
    epsilonInit: args.epsilonInit,
    epsilonFinal: args.epsilonFinal,
    epsilonDecayFrames: args.epsilonDecayFrames,
    batchSize: args.batchSize,
    learningRate: args.learningRate
  });

  await train(
      agent, args.batchSize, args.gamma, args.learningRate,
      args.cumulativeRewardThreshold, args.syncEveryFrames, args.savePath);
}

if (require.main === module) {
  main();
}
