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

/**
 * Based on
 * https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
 */

import * as tf from '@tensorflow/tfjs';

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const epsilon = tf.scalar(1e-5);

async function generatePatterns(model, layerName, filterIndex) {
  const imageH = model.inputs[0].shape[1];
  const imageW = model.inputs[0].shape[2];

  const layerOutput = model.getLayer(layerName).output;
  console.log(`layerOutput.shape:`, layerOutput.shape);  // DEBUG
  const auxModel = tf.model({inputs: model.inputs, outputs: layerOutput});
  // console.log('auxModel:'); // DEBUG
  auxModel.summary();  // DEBUG
  const lossFunction = (input) => {
    // ----------------------------------
    console.log(`Calling apply(): input.shape: ${input.shape}`);  // DEBUG
    const out = auxModel.predict(input).gather([filterIndex], 3);
    // ----------------------------------
    // function relu2(x) {
    //   return tf.tidy(() => {
    //     return x.relu().relu();
    //   });
    // }
    // const out = relu2(tf.tidy(() => input.relu().square().mean().add(0.001)));
    // console.log(`out.shape = ${out.shape}`);  // DEBUG
    // ----------------------------------
    // const feedDict = new tf.models.FeedDict();  // DEBUG
    // feedDict.add(model.inputs[0], input);
    // console.log(feedDict);  // DEBUG
    // const out = tf.models.execute(layerOutput, feedDict);
    console.log(`out.shape = ${out.shape}`);
    return out;
  }  // TODO(cais): Simplify.

  const gradients = tf.grad(lossFunction);

  // const inputImage = tf.variable(tf.randomUniform([1, imageH, imageW, 3], 0, 1).mul(tf.scalar(20)).add(tf.scalar(128)));
  let inputImage = tf.tidy(() => tf.randomUniform([1, imageH, imageW, 3], 0, 1).mul(tf.scalar(20)).add(tf.scalar(128)));

  const iterations = 2;  // TODO(cais): 40.
  for (let i = 0; i < iterations; ++i) {
    console.log(`Iteration ${i + 1}/${iterations}`);  // DEBUG
    console.log(inputImage.isDisposed);  // dEBUG
    let grads = gradients(inputImage);
    console.log(`grads.shape: ${JSON.stringify(grads.shape)}`);  // DEBUG
    const norm = tf.sqrt(tf.mean(tf.square(grads))).add(epsilon);
    // console.log(`norm.shape: ${JSON.stringify(norm.shape)}`);  // DEBUG
    norm.print();
    grads = grads.div(norm);
    const newInputImage = inputImage.add(grads);
    inputImage.dispose();
    inputImage = newInputImage;
  }
}

async function run() {
  status('Loading model...');
  console.log(tf.models.execute);

  const model = await tf.loadModel(MOBILENET_MODEL_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  // model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status('');

  console.log('Calling generate patterns');  // DEBUG
  generatePatterns(model, 'conv_pw_8_relu', 0);
};

//
// UI
//

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

run();
