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

import * as fs from 'fs';

import {TextData} from './data';
import {createModel, compileModel, fitModel} from './model';
import '@tensorflow/tfjs-node';

async function main() {
  const text = fs.readFileSync('./nietzsche.txt',  {encoding: 'utf-8'});

  const sampleLength = 40;
  const sampleStep = 3;
  const textData =  new TextData('text-data', text, sampleLength, sampleStep);

  const lstmLayerSize = 256;
  const model = createModel(
      textData.sampleLen(), textData.charSetSize(), lstmLayerSize);
  compileModel(model, 1e-2);

  const epochs = 100;
  const examplesPerEpoch = 2048;
  const batchSize = 128;
  const validationSplit = 0.0625;
  await fitModel(
      model, textData, epochs, examplesPerEpoch, batchSize, validationSplit);
}

main();