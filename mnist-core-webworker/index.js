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

import * as Comlink from 'comlink';

import * as ui from './ui';

async function load(model) {
  await model.load();
}

async function train(model) {
  ui.isTraining();
  await model.train(Comlink.proxyValue(ui.trainingLog));
}

async function test(model) {
  const testExamples = 50;
  const {imagesData, predictions, labels} = await model.test(testExamples);

  ui.showTestResults(imagesData, predictions, labels, testExamples);
}

async function mnist() {
  ui.animate();
  const Model = Comlink.proxy(new Worker('model.js'));
  const model = await new Model();

  await load(model);
  await train(model);
  test(model);
}
mnist();
