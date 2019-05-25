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

let tf;

tf = require('@tensorflow/tfjs-node-gpu');  // TODO(cais): Parameterize.

const AUTO_MPG_CSV_URL = 'https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv';

let dataset = tf.data.csv(AUTO_MPG_CSV_URL, {
  columnConfigs: {
    mpg: {
      isLabel: true
    }
  }
}).shuffle(1000).map(data => {
  const xs = data.xs;
//   console.log(data);  // DEBUG
//   const xsTensor =
//       tf.tensor1d([xs.cylinders, xs.displacement, xs.horsepower, xs.weight, xs.acceleration]);
  const xsTensor = tf.tensor1d([xs.cylinders, xs.displacement]);
//   console.log(xs);
//   xsTensor.print();  // DEBUG
  const ysTensor =
      tf.tensor1d([data.ys.mpg])
//   ysTensor.print();  // DEBUG
  return {
    xs: xsTensor,
    ys: ysTensor
  };
}).batch(64);

console.log(dataset);


function buildModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 16,
    activation: 'relu',
    inputShape: [2]
  }));
  model.add(tf.layers.dense({units: 1}));
  const learningRate = 5e-2;
  model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.sgd(learningRate)
    // optimizer: 'adam'
  });
  return model;
}

async function run() {
  const iter = await dataset.iterator();
  console.log(await iter.next());

  const model = buildModel();
  model.summary();

  await model.fitDataset(dataset, {
    epochs: 100
  });
}

run();
