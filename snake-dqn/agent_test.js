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

import {SnakeGame} from "./snake_game";
import { SnakeGameAgent } from "./agent";

describe('SnakeGameAgent', () => {
  it('playStep', () => {
    const game = new SnakeGame({
      height: 9,
      width: 9,
      numFruits: 1,
      initLen: 2
    });
    const agent = new SnakeGameAgent(game, {
      replayBufferSize: 100,
      epsilonInit: 1,
      epsilonFinal: 0.1,
      epsilonNumFrames: 10,
      batchSize: 32,
      learningRate: 1e-3
    });

    const numGames = 40;
    let bufferIndex = 0;
    for (let n = 0; n < numGames; ++n) {
      // At the beginnig of a game, the cumulative reward ought to be 0.
      expect(agent.cumulativeReward_).toEqual(0);
      let out = null;
      let outPrev = null;
      for (let m = 0; m < 10; ++m) {
        const currentState = agent.game_.getState();
        out = agent.playStep();
        // Check the content of the replay buffer.
        expect(agent.replayMemory_.buffer[bufferIndex % 100][0])
            .toEqual(currentState);
        expect(agent.replayMemory_.buffer[bufferIndex % 100][1])
            .toEqual(out.action);

        expect(agent.replayMemory_.buffer[bufferIndex % 100][2]).toEqual(
            outPrev == null ? out.cumulativeReward :
            out.cumulativeReward - outPrev.cumulativeReward);
        expect(agent.replayMemory_.buffer[bufferIndex % 100][3]).toEqual(out.done);
        expect(agent.replayMemory_.buffer[bufferIndex % 100][4])
            .toEqual(out.done ? undefined : agent.game_.getState());
        bufferIndex++;
        if (out.done) {
          break;
        }
        outPrev = out;
      }
      agent.reset();
    }
  });
});