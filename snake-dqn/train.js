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

import {SnakeGameAgent} from './agent';

/**
 * Train an agent to play the snake game.
 *
 * @param {SnakeGameAgent} agent The agent to train.
 * @param {number} cumulativeRewardThreshold The threshold of cumulative reward
 *   from a single game. The training stops as soon as this threshold is achieved.
 * @param {number} syncPerFrame The frequency at which the weights are copied from
 *   the online DQN of the agent to the target DQN, in number of frames.
 * @param {string} savePath Path to which the online DQN of the agent will be saved
 *   upon the completion of the training.
 */
train(agent, cumulativeRewardThreshold, syncPerFrame, savePath) {
  for (let i = 0; i < this.replayBufferSize; ++i) {
    this.playStep();
  }

  while (true) {
    // console.log('Calling trainOnReplayBatch()');  // DEBUG
    this.trainOnReplayBatch();
    const {cumulativeReward, done} = this.playStep();
    if (done) {
      console.log(`Frame #${this.frameCount}: ` +
          `cumulativeReward = ${cumulativeReward}`);
      if (cumulativeReward >= cumulativeRewardThreshold) {
        // TODO(cais): Save online network.
        break;
      }
    }
    if (this.frameCount % syncPerFrame === 0) {
      console.log('Copying weights from online network to target network');
      copyWeights(this.targetNetwork, this.onlineNetwork);
    }
  }
}
