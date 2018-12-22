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

const fs = require('fs');
const Jimp = require('jimp');

const tf = require('@tensorflow/tfjs');

require('@tensorflow/tfjs-node-gpu');
// TODO(cais): Try the tfjs-node version of this.

// const MODEL_PATH =
//     // tslint:disable-next-line:max-line-length
//     'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
// const MODEL_PATH = 'https://storage.googleapis.com/tfjs-speech-model-test/mobilenetv2_tfjs/model.json';
// const MODEL_PATH = 'https://storage.googleapis.com/tfjs-speech-model-test/mobilenetv2_0.35_tfjs/model.json';
// const MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json';
const MODEL_PATH = 'file://./vgg16_tfjs/model.json';

const epsilon = tf.scalar(1e-5);

// const filterCanvas = document.getElementById('filter');

// TODO(cais): Deduplicate with index.js.
async function generatePatterns(model, layerName, filterIndex) {
  const imageH = model.inputs[0].shape[1];
  const imageW = model.inputs[0].shape[2];
  const imageDepth = model.inputs[0].shape[3];

  const layerOutput = model.getLayer(layerName).output;
  const auxModel = tf.model({inputs: model.inputs, outputs: layerOutput});
  const lossFunction = (input) => {
    // ----------------------------------
    // console.log(`Calling apply(): input.shape: ${input.shape}`);  // DEBUG
    const out = auxModel.apply(input, {training: true}).gather([filterIndex], 3);
    return out;
  }  // TODO(cais): Simplify.

  let image = tf.tidy(() => tf.randomUniform([1, imageH, imageW, imageDepth], 0, 1).mul(20).add(128));
  const gradients = tf.grad(lossFunction);

  const iterations = 100;  // TODO(cais): 40.
  const stepSize = 1;
  for (let i = 0; i < iterations; ++i) {
    console.log(`Iteration ${i + 1}/${iterations}`);  // DEBUG
    const scaledGrads = tf.tidy(() => {
      const grads = gradients(image);
      // console.log(`grads.shape: ${JSON.stringify(grads.shape)}`);  // DEBUG
      const norm = tf.sqrt(tf.mean(tf.square(grads))).add(epsilon);
      // console.log(`norm.shape: ${JSON.stringify(norm.shape)}`);  // DEBUG
      // norm.print();
      return grads.div(norm);
    });
    const newInputImage = tf.tidy(() => image.add(scaledGrads.mul(stepSize)).clipByValue(0, 255));
    scaledGrads.dispose();
    image.dispose();
    image = newInputImage;
    // clipByValue(0, 255)
    // const filterImage = tf.tidy(() => inputImage.squeeze([0]).asType('int32'));
    // await tf.toPixels(filterImage, filterCanvas);
    // filterImage.dispose();
    // console.log(tf.memory().numTensors);  // DEBUG
  }

  const imageData = image.clipByValue(0, 255).asType('int32').dataSync();
  console.log(imageData.length);
  const bufferLen = imageH * imageW * 4;
  const buffer = new Uint8Array(bufferLen);
  let index = 0;
  for (let i = 0; i < imageH; ++i) {
    for (let j = 0; j < imageW; ++j) {
      const inIndex = 3 * (i * imageW + j);
      buffer.set([imageData[inIndex]], index++);
      buffer.set([imageData[inIndex + 1]], index++);
      buffer.set([imageData[inIndex + 2]], index++);
      buffer.set([255], index++);
    }
  }
  console.log(`index = ${index}`);

  new Jimp({data: new Buffer(buffer), width: imageW, height: imageH}, (err, img) => {
    console.log(img);  // DEBUG
    img.write('filter.png');
  });

//   console.log(inputImage.dataSync());
//   const image = new Jimp(imageH, imageW, (err, image) => {
//   });
//   console.log(image.bitmap.data.length); 
//   image.scan(0, 0, imageH, imageW, function(x, y, idx) {
//     // x, y is the position of this pixel on the image
//     // idx is the position start position of this rgba tuple in the bitmap Buffer
//     // this is the image
   
//     // var red = this.bitmap.data[idx + 0];
//     // var green = this.bitmap.data[idx + 1];
//     // var blue = this.bitmap.data[idx + 2];
//     // var alpha = this.bitmap.data[idx + 3];
//     this.bitmap.data[idx + 0] = 100;
//     // console.log(red, green, blue, alpha);
   
//     // rgba values run from 0 - 255
//     // e.g. this.bitmap.data[idx] = 0; // removes red from this pixel
//   });
}

async function run() {
  console.log('Loading model...');
  // console.log(tf.models.execute);

  const model = await tf.loadModel(MODEL_PATH);
  // const model = await tf.loadModel('./vgg16_tfjs/model.json');
  model.summary();  //

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  // model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  console.log('Calling generate patterns');  // DEBUG
  // generatePatterns(model, 'conv_pw_8', 0);
  // generatePatterns(model, 'conv_pw_8_relu', 10);
  // generatePatterns(model, 'conv_pw_9_relu', 11);
  // generatePatterns(model, 'conv_pw_14', 0);
  // generatePatterns(model, 'block_1_depthwise', 3);

  // MobileNetV2 alpha=1.0.
//   generatePatterns(model, 'block_1_depthwise_relu', 50);  // Interesting.
  // generatePatterns(model, 'block_2_depthwise_relu', 2);  // Interesting.
  // generatePatterns(model, 'block_3_depthwise_relu', 0);  // Interesting.
  // generatePatterns(model, 'block_4_depthwise_relu', 100);  // Interesting.
  // generatePatterns(model, 'block_4_depthwise_relu', 128);  // Interesting.
  // Becomes extremely slow when you reach block 5.
//   generatePatterns(model, 'block_5_depthwise_relu', 1);  // Interesting.

  // MobileNetV2 alpha=0.35.
  // generatePatterns(model, 'block_1_depthwise_relu', 0);  // Interesting.
  // generatePatterns(model, 'block_1_depthwise_relu', 7);  // Interesting.
  // generatePatterns(model, 'block_5_depthwise_relu', 1);  // Interesting.
  // generatePatterns(model, 'block_6_depthwise_relu', 2);  // Interesting.
  // generatePatterns(model, 'block_8_depthwise_relu', 2);  // Interesting.
  // generatePatterns(model, 'block_10_depthwise_relu', 100);  // Interesting.
  // generatePatterns(model, 'block_12_depthwise_relu', 100);  // Interesting.
  
  // MNIST model.
//   generatePatterns(model, 'max_pooling2d_1', 1);  // Interesting.

  // VGG16
  generatePatterns(model, 'block5_conv3', 0);  // Interesting.
};

run();
