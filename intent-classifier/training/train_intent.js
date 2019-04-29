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

const path = require('path');
const fs = require('fs');

const argparse = require('argparse');
const tf = require('@tensorflow/tfjs');
const fileIO = require('@tensorflow/tfjs-node/dist/io/file_system');
const mkdirp = require('mkdirp');

const {getModel} = require('./intent_model');
const {loadJSON} = require('./util');


async function run(
    dataPath, metadataPath, outputFolder, epochs, validationSplit = 0.15) {
  const {xsArr, ysArr} = loadJSON(dataPath);
  const metadata = loadJSON(metadataPath);

  const xs = tf.tensor(xsArr, metadata.xsShape);
  const ys = tf.tensor(ysArr, metadata.ysShape);

  const model = getModel(metadata.labels);

  // We use model.fit as the whole dataset comfortably fits in memory.
  await model.fit(xs, ys, {epochs, validationSplit});

  mkdirp(outputFolder);
  await model.save(fileIO.fileSystem(outputFolder));

  const metaOutPath = path.resolve(outputFolder, 'intent_metadata.json');
  const metadataStr = JSON.stringify(metadata, null, 2);
  fs.writeFileSync(metaOutPath, metadataStr, {encoding: 'utf8'});
}


(async function() {
  const parser = new argparse.ArgumentParser();
  const defaultDataPath =
      path.resolve(__dirname, './data/intents_as_tensors.json');
  const defaultMetadataPath =
      path.resolve(__dirname, './data/intent_metadata.json');
  const defaultOutFolder = path.resolve(__dirname, './models/intent/');

  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for training (required CUDA and CuDNN)',
  });
  parser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 32,
    help: 'Number of epochs',
  });
  parser.addArgument('--dataPath', {
    type: 'string',
    defaultValue: defaultDataPath,
  });
  parser.addArgument('--metadataPath', {
    type: 'string',
    defaultValue: defaultMetadataPath,
  });
  parser.addArgument('--outFolder', {
    type: 'string',
    defaultValue: defaultOutFolder,
  });

  const args = parser.parseArgs();

  if (args.gpu) {
    console.log('Using GPU.');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU.');
    require('@tensorflow/tfjs-node');
  }

  await run(args.dataPath, args.metadataPath, args.outFolder, args.epochs);
})();
