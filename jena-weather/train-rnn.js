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

// TODO(cais): Add --gpu flag.
// import '@tensorflow/tfjs-node';
import '@tensorflow/tfjs-node-gpu';

import {JenaWeatherData} from './data';
import {trainModel} from './models';

global.fetch = require('node-fetch');

async function main() {
  const jenaWeatherData = new JenaWeatherData();
  console.log(`Loading Jena weather data...`);
  await jenaWeatherData.load();

  const shuffle = true;
  const lookBack = 10 * 24 * 6;  // Look back 10 days.
  const step = 6;                // 1-hour steps.
  const delay = 24 * 6;          // Predict the weather 1 day later.
  const batchSize = 128;
  const minIndex = 0;
  const maxIndex = 200000;
  const normalize = true;
  const includeDateTime = false;
  const epochs = 20;
  const displayEvery = 5;

  const modelType = 'gru';
  let numFeatures = jenaWeatherData.getDataColumnNames().length;
  const model = buildModel(modelType, Math.floor(lookBack / step), numFeatures);

  await trainModel(
      model, jenaWeatherData, shuffle, normalize, includeDateTime, lookBack,
      step, delay, batchSize, minIndex, maxIndex, epochs, displayEvery);
}

main();
