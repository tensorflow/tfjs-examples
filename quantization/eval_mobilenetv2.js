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
import * as jimp from 'jimp';

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
  parser.addArgument('imageFilePath', {  // TODO(cais): Change to image dir path.
    type: 'string',
    help: 'Path at which the image for inference is stored.'
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
      resolve(tf.tidy(
          () => tf.image.resizeBilinear(buffer.toTensor(), [height, width])));
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
  model.summary();

  const imageH = model.inputs[0].shape[2];
  const imageW = model.inputs[0].shape[2];
  console.log(`imageH = ${imageH}; imageW = ${imageW}`);
  let imageTensor =
      await readImageTensorFromFile(args.imageFilePath, imageH, imageW);
//   imageTensor.print();  // DEBUG
  imageTensor = imageTensor.div(255);
  // TODO(cais): Put normalization in readImageTensorFromFile();
//   imageTensor.min().print();
//   imageTensor.max().print();
//   console.log(imageTensor.shape);  // DEBUG
  model.predict(imageTensor).argMax(-1).print();
  // TODO(cais): tidy().

//   console.log(`Performing prediction...`);
//   const t0 = tf.util.now();
//   const evalOutput = model.evaluate(testImages, testLabels);
//   const t1 = tf.util.now();
//   console.log(`\nEvaluation took ${(t1 - t0).toFixed(2)} ms.`);
//   console.log(
//       `\nEvaluation result:\n` +
//       `  Loss = ${evalOutput[0].dataSync()[0].toFixed(6)}; `+
//       `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(6)}`);
}

if (require.main === module) {
  main();
}
