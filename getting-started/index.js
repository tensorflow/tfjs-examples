/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

// This tiny example illustrates how little code is necessary build /
// train / predict from a model in TensorFlow.js.  Edit this code
// and refresh the index.html to quickly explore the API.

// Tiny TFJS train / predict example.
async function run() {
  // Create a simple model.
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  // Generate some synthetic data for training. (y = 2x - 1)
  const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
  const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  // Train the model using the data.
  await model.fit(xs, ys, {epochs: 250});

  // Use the model to do inference on a data point the model hasn't seen.
  // Should print approximately 39.
  document.getElementById('micro-out-div').innerText =
      model.predict(tf.tensor2d([20], [1, 1])).dataSync();
}

run();
