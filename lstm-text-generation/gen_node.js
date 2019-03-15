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

/**
 * Use a trained next-character prediction model to generate some text. 
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as argparse from 'argparse';

import * as tf from '@tensorflow/tfjs';

import {maybeDownload, TextData, TEXT_DATA_URLS} from './data';
import {generateText} from './model';

function parseArgs() {
  const parser = argparse.ArgumentParser({
    description: 'Train an lstm-text-generation model.'
  });
  parser.addArgument('textDatasetName', {
    type: 'string',
    choices: Object.keys(TEXT_DATA_URLS),
    help: 'Name of the text dataset'
  });
  parser.addArgument('modelJSONPath', {
    type: 'string',
    help: 'Path to the trained next-char prediction model saved on disk ' +
    '(e.g., ./my-model/model.json)'
  });
  parser.addArgument('--genLength', {
    type: 'int',
    defaultValue: 200,
    help: 'Length of the text to generate.'
  });
  parser.addArgument('--temperature', {
    type: 'float',
    defaultValue: 0.5,
    help: 'Temperature value to use for text generation. Higher values ' +
    'lead to more random-looking generation results.'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use CUDA GPU for training.'
  });
  parser.addArgument('--sampleStep', {
    type: 'int',
    defaultValue: 3,
    help: 'Step length: how many characters to skip between one example ' +
    'extracted from the text data to the next.'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArgs();

  if (args.gpu) {
    console.log('Using GPU');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU');
    require('@tensorflow/tfjs-node');
  }

  // Load the model.
  const model = await tf.loadModel(`file://${args.modelJSONPath}`);

  const sampleLen = model.inputs[0].shape[1];

  // Create the text data object.
  const textDataURL = TEXT_DATA_URLS[args.textDatasetName].url;
  const localTextDataPath = path.join(os.tmpdir(), path.basename(textDataURL));
  await maybeDownload(textDataURL, localTextDataPath);
  const text = fs.readFileSync(localTextDataPath, {encoding: 'utf-8'});
  const textData = new TextData('text-data', text, sampleLen, args.sampleStep);

  // Get a seed text from the text data object.
  const [seed, seedIndices] = textData.getRandomSlice();
  
  console.log(`Seed text:\n"${seed}"\n`);

  const generated = await generateText(
      model, textData, seedIndices, args.genLength, args.temperature);

  console.log(`Generated text:\n"${generated}"\n`);
}

main();
