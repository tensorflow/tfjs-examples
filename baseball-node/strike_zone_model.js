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

const tf = require('@tensorflow/tfjs');

// Constants from training data:
const PX_MIN = -2.65170604056843;
const PX_MAX = 2.842899614;
const PZ_MIN = -2.01705841594049;
const PZ_MAX = 6.06644249133382;
const SZ_TOP_MIN = 2.85;
const SZ_TOP_MAX = 4.241794863019148;
const SZ_BOT_MIN = 1.248894636863092;
const SZ_BOT_MAX = 2.2130980270561516;

// Build model...
const fields = [
  {key: 'px', min: PX_MIN, max: PX_MAX}, {key: 'pz', min: PZ_MIN, max: PZ_MAX},
  {key: 'sz_top', min: SZ_TOP_MIN, max: SZ_TOP_MAX},
  {key: 'sz_bot', min: SZ_BOT_MIN, max: SZ_BOT_MAX}, {key: 'left_handed_batter'}
];

// TODO - figure out pitch data.
// const data = new Pitch

const model = tf.sequential();

model.add(tf.layers.dense(
    {units: 20, activation: 'relu', inputShape: [fields.length]}));
model.add(tf.layers.dense({units: 10, activation: 'relu'}));
model.add(tf.layers.dense({units: 2, activation: 'softmax'}));

model.compile({
  optimizer: tf.train.adam(),
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

module.exports = {model};
