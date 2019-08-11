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

import * as argparse from 'argparse';
import * as fs from 'fs';
import * as jimp from 'jimp';
import * as path from 'path';
const ProgressBar = require('progress');

import {IMAGENET_CLASSES} from './imagenet_classes';

// The `tf` module will be loaded dynamically depending on whether
// `--gpu` is specified in the command-line flags.
let tf;

function parseArgs() {
  const parser = new argparse.ArgumentParser({
    description:
        'TensorFlow.js Quantization Example: Evaluating an MNIST Model',
    addHelp: true
  });
  parser.addArgument('modelSavePath', {
    type: 'string',
    help: 'Path at which the model to be evaluated is saved.'
  });
  parser.addArgument('imageDir', {
    type: 'string',
    help: 'Path at the directory under which the test images are stored.'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for evaluation (requires CUDA-enabled ' +
    'GPU and supporting drivers and libraries.'
  });
  return parser.parseArgs();
}

async function readImageTensorFromFile(filePath, height, width) {
  return new Promise((resolve, reject) => {
    jimp.read(filePath, (err, image) => {
      if (err) {
        reject(err);
      } else {
        const h = image.bitmap.height;
        const w = image.bitmap.width;
        const buffer = tf.buffer([1, h, w, 3], 'float32');
        image.scan(0, 0, w, h, function(x, y, index) {
        buffer.set(image.bitmap.data[index], 0, y, x, 0);
        buffer.set(image.bitmap.data[index + 1], 0, y, x, 1);
        buffer.set(image.bitmap.data[index + 2], 0, y, x, 2);
      });
      resolve(tf.tidy(() => tf.image.resizeBilinear(
          buffer.toTensor(), [height, width]).div(255)));
      }
    });
  });
}

async function main() {
  const args = parseArgs();
  if (args.gpu) {
    tf = require('@tensorflow/tfjs-node-gpu');
  } else {
    tf = require('@tensorflow/tfjs-node');
  }

  console.log(`Loading model from ${args.modelSavePath}...`);
  const model = await tf.loadLayersModel(`file://${args.modelSavePath}`);

  const imageH = model.inputs[0].shape[2];
  const imageW = model.inputs[0].shape[2];

  // Load the images into a tensor.
  const dirContent = fs.readdirSync(args.imageDir);
  dirContent.sort();
  const numImages = dirContent.length;
  console.log(`Reading ${numImages} images...`);
  const progressBar = new ProgressBar('[:bar]', {
    total: numImages,
    width: 80,
    head: '>'
  });
  const imageTensors = [];
  const truthLabels = [];
  for (const fileName of dirContent) {
    const truthLabel = fileName.split('.')[0].split('_')[2];
    truthLabels.push(truthLabel);
    const imageFilePath = path.join(args.imageDir, fileName);
    const imageTensor =
        await readImageTensorFromFile(imageFilePath, imageH, imageW);
    imageTensors.push(imageTensor);
    progressBar.tick();
  }

  const stackedImageTensor = tf.concat(imageTensors, 0);
  console.log('Calling model.predict()...');
  const t0 = new Date().getTime();
  const {top1Indices, top5Indices} = tf.tidy(() => {
    const probs = model.predict(stackedImageTensor, {batchSize: 64});
    return {
      top1Indices: probs.argMax(-1).arraySync(),
      top5Indices: probs.topk(5).indices.arraySync()
    };
  });
  console.log(`model.predict() took ${(new Date().getTime() - t0).toFixed(2)} ms`);

  let numCorrectTop1 = 0;
  let numCorrectTop5 = 0;
  top1Indices.forEach((top1Index, i) => {
    const truthLabel = truthLabels[i];
    if (IMAGENET_CLASSES[top1Index].indexOf(truthLabel) !== -1) {
      numCorrectTop1++;
    }
    for (let k = 0; k < 5; ++k) {
      if (IMAGENET_CLASSES[top5Indices[i][k]].indexOf(truthLabel) !== -1) {
        numCorrectTop5++;
        break;
      }
    }
  });
  console.log(
      `#total = ${numImages}; #correct(top-1) = ${numCorrectTop1}; ` +
      `accuracy(top-1) = ${(numCorrectTop1 / numImages).toFixed(3)}; ` +
      `#correct(top-5) = ${numCorrectTop5}; ` +
      `accuracy(top-5) = ${(numCorrectTop5 / numImages).toFixed(3)}\n`);
  tf.dispose([imageTensors, stackedImageTensor]);
}

if (require.main === module) {
  main();
}
