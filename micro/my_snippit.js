// Copyright 2018 Google LLC. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// TFJS microfiddle.
var output = []
console.log = function(x){
  output.push(x);
  document.getElementById('fiddle_out_div').innerHTML = output.join("<br>");
}
// Edit snippet below.

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
for (var i = 0; i < 5; ++i) {
  console.log(i);
  model.fit(xs, ys);
}

// Use the model to do inference on a data point the model hasn't seen:
model.predict(tf.tensor2d([5], [1, 1])).print();
