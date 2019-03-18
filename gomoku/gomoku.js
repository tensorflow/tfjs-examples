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
import { Board } from './game';

global.fetch = require('node-fetch');

function parseArguments() {
  const parser =
      new ArgumentParser({description: 'Exercises go-moku game'});
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use GPU'
  });
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
  const b = new Board({});
  console.log(JSON.stringify(b));
  b.initBoard();
}

main();
