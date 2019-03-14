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
 * Training of a next-char prediction model.
 */

import * as fs from 'fs';
import * as https from 'https';
import * as os from 'os';
import * as path from 'path';

import * as argparse from 'argparse';

import {maybeDownload, TextData, TEXT_DATA_URLS} from './data';
import {createModel, compileModel, fitModel, generateText} from './model';

function parseArgs() {
  const parser = argparse.ArgumentParser({
    description: 'Train an lstm-text-generation model.'
  });
  parser.addArgument('textDatasetName', {
    type: 'string',
    choices: Object.keys(TEXT_DATA_URLS),
    help: 'Name of the text dataset'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use CUDA GPU for training.'
  });
  parser.addArgument('--sampleLen', {
    type: 'int',
    defaultValue: 60,
    help: 'Sample length: Length of each input sequence to the model, in ' +
    'number of characters.'
  });
  parser.addArgument('--sampleStep', {
    type: 'int',
    defaultValue: 3,
    help: 'Step length: how many characters to skip between one example ' +
    'extracted from the text data to the next.'
  });
  parser.addArgument('--learningRate', {
    type: 'float',
    defaultValue: 1e-2,
    help: 'Learning rate to be used during training'
  });
  parser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 150,
    help: 'Number of training epochs'
  });
  parser.addArgument('--examplesPerEpoch', {
    type: 'int',
    defaultValue: 10000,
    help: 'Number of examples to sample from the text in each training epoch.'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 128,
    help: 'Batch size for training.'
  });
  parser.addArgument('--validationSplit', {
    type: 'float',
    defaultValue: 0.0625,
    help: 'Validation split for training.'
  });
  parser.addArgument('--displayLength', {
    type: 'int',
    defaultValue: 120,
    help: 'Length of the sampled text to display after each epoch of training.'
  });
  parser.addArgument('--savePath', {
    type: 'string',
    help: 'Path to which the model will be saved (optional)'
  });
  parser.addArgument('--lstmLayerSize', {
    type: 'string',
    defaultValue: '128,128',
    help: 'LSTM layer size. Can be a single number or an array of numbers ' +
    'separated by commas (E.g., "256", "256,128")'
  });  // TODO(cais): Support
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

  // Create the text data object.
  const textDataURL = TEXT_DATA_URLS[args.textDatasetName].url;
  const localTextDataPath = path.join(os.tmpdir(), path.basename(textDataURL));
  await maybeDownload(textDataURL, localTextDataPath);
  const text = fs.readFileSync(localTextDataPath, {encoding: 'utf-8'});
  const textData =
      new TextData('text-data', text, args.sampleLen, args.sampleStep);

  // Convert lstmLayerSize from string to number array before handing it
  // to `createModel()`.
  const lstmLayerSize = args.lstmLayerSize.indexOf(',') === -1 ?
      Number.parseInt(args.lstmLayerSize) :
      args.lstmLayerSize.split(',').map(x => Number.parseInt(x));

  const model = createModel(
      textData.sampleLen(), textData.charSetSize(), lstmLayerSize);
  compileModel(model, args.learningRate);

  // Get a seed text for display in the course of model training.
  const [seed, seedIndices] = textData.getRandomSlice();
  console.log(`Seed text:\n"${seed}"\n`);

  const DISPLAY_TEMPERATURES = [0, 0.25, 0.5, 0.75];

  let epochCount = 0;
  await fitModel(
      model, textData, args.epochs, args.examplesPerEpoch, args.batchSize,
      args.validationSplit, {
        onTrainBegin: async () => {
          epochCount++;
          console.log(`Epoch ${epochCount} of ${args.epochs}:`);
        },
        onTrainEnd: async () => {
          DISPLAY_TEMPERATURES.forEach(async temperature => {
            const generated = await generateText(
                model, textData, seedIndices, args.displayLength, temperature);
            console.log(
                `Generated text (temperature=${temperature}):\n` +
                `"${generated}"\n`);
          });
        }
      });

  if (args.savePath != null && args.savePath.length > 0) {
    await model.save(`file://${args.savePath}`);
    console.log(`Saved model to ${args.savePath}`);
  }
}

main();
