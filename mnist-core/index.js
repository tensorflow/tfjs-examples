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

import {MnistData} from './data';
import * as model from './model';
import * as ui from './ui';

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

async function train() {
  ui.isTraining();
  await model.train(data, ui.trainingLog);
}

async function test() {
  const testExamples = 50;
  const batch = data.nextTestBatch(testExamples);
  let begin = Date.now();
  const predictions = model.predict(batch.xs);
  let end = Date.now();
  console.log(`${end - begin}`);
  const labels = model.classesFromLabel(batch.labels);

  ui.showTestResults(batch, predictions, labels);
}

async function mnist() {
  await load();
  await train();
  test();
}
mnist();
