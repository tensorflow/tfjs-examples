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


import {ArgumentParser} from 'argparse';
import {Board, Game} from './game';
import {HumanAgent} from './human_agent';

global.fetch = require('node-fetch');

function parseArguments() {
  const parser = new ArgumentParser({description: 'Exercises go-moku game'});
  parser.addArgument('--gpu', {action: 'storeTrue', help: 'Use GPU'});
  return parser.parseArgs();
}

async function main() {
  const args = parseArguments();
  if (args.gpu) {
    console.log('Using GPU for training.');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU for training.');
    require('@tensorflow/tfjs-node');
  }

  console.log('I am gomoku');
  const width = 3;
  const height = 3;
  const nInRow = 3
  const board = new Board({width, height, nInRow});
  board.initBoard();
  const game = new Game(board);

  // Human vs. Human
  const human1 = new HumanAgent();
  human1.setPlayerIndex(1);
  const human2 = new HumanAgent();
  human2.setPlayerIndex(2);

  const whoGoesFirst = 0;
  const showBoardEachTime = true;
  game.startPlay(human1, human2, whoGoesFirst, showBoardEachTime);
}

if (require.main === module) {
  main();
}
