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

/**
 * Takes a CSV file of queries and intents and embeds all the queries using the
 * universal sentence encoder and one-hot-encodes the queries.
 * The output of this script will be an JSON file with the following format
 *
 *  {
 *    xs: number[][], // Array of tensor-like objects for queries
 *    ys: number[][], // Array of tensor-like objects for intents (labels)
 *  }
 *
 * The script will also output a `metadata.json` file with the following format:
 *
 *  {
 *    labels: string[], array of strings to invert one-hot encodings
 *    xsShape: [number, number], tensor shape of xs
 *    ysShape: [number, number], tensor shape of ys
 *  }
 */

const fs = require('fs');
const path = require('path');
const mkdirp = require('mkdirp');

const tf = require('@tensorflow/tfjs');
const argparse = require('argparse');
global.fetch = require('node-fetch');
const useLoader = require('@tensorflow-models/universal-sentence-encoder');

const {loadCSV, chunk} = require('./util');


async function csvToTensors(data, labels, use, batchSize) {
  // Only keep the queries for the intents we are interested in.
  const filtered = data.filter(d => labels.indexOf(d.intent) >= 0);
  console.log(`Converting ${filtered.length} queries to tensors.`);

  // Shuffle the data before we tensorify it and write it out.
  tf.util.shuffle(filtered);

  const xs = filtered.map(d => d.query);
  const ys = filtered.map(d => labels.indexOf(d.intent));

  const xsBatches = chunk(xs, batchSize);
  const ysBatches = chunk(ys, batchSize);

  tf.util.assert(
      xsBatches.length === ysBatches.length,
      () => 'batched xs and ys do not have the same length');

  const xsBatchedTensors = [];
  const ysBatchedTensors = [];

  for (let i = 0; i < xsBatches.length; i++) {
    console.time(
        `Converted batch ${i} of ${xsBatches.length} (${batchSize}): `);

    const queries = xsBatches[i];
    const batchLabels = ysBatches[i];

    // Convert the labels to one-hot representation
    // And embed the query with the universal sentence encoder
    const oneHot = tf.oneHot(batchLabels, labels.length);
    const embedding = await use.embed(queries);

    xsBatchedTensors.push(embedding);
    ysBatchedTensors.push(oneHot);

    console.timeEnd(
        `Converted batch ${i} of ${xsBatches.length} (${batchSize}): `);
  }

  const xsTensor = tf.concat(xsBatchedTensors);
  const ysTensor = tf.concat(ysBatchedTensors);

  tf.dispose([xsBatchedTensors, ysBatchedTensors]);

  return {
    xs: xsTensor,
    ys: ysTensor,
  };
}


async function run(srcPath, outFolder, batchSize) {
  console.log('Start');
  const use = await useLoader.load();
  console.log('Loaded Universal Sentence Encoder');

  const LABELS = [
    'AddToPlaylist',
    'GetWeather',
    'PlayMusic',
  ];

  const csvData = loadCSV(srcPath);
  const {xs, ys} = await csvToTensors(csvData, LABELS, use, batchSize);
  const [xsArr, ysArr] = await Promise.all([xs.array(), ys.array()]);

  // Write out the tensorified data
  mkdirp(outFolder);

  const dataOutPath = path.resolve(outFolder, 'intents_as_tensors.json');
  const dataStr = JSON.stringify({xsArr, ysArr}, null, 2);
  fs.writeFileSync(dataOutPath, dataStr, {encoding: 'utf8'});

  // Write out the related metadata
  const metaOutPath = path.resolve(outFolder, 'intent_metadata.json');
  const metadata = {
    labels: LABELS,
    xsShape: xs.shape,
    ysShape: ys.shape,
  };

  const metadataStr = JSON.stringify(metadata, null, 2);
  fs.writeFileSync(metaOutPath, metadataStr, {encoding: 'utf8'});
}


(async function() {
  const parser = new argparse.ArgumentParser();
  const defaultSrcPath = path.resolve(__dirname, './data/intents.csv');
  const defaultOutFolder = path.resolve(__dirname, './data/');

  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for training (required CUDA and CuDNN)',
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 32,
    help: 'Batch size to be used during model training',
  });
  parser.addArgument('--srcPath', {
    type: 'string',
    defaultValue: defaultSrcPath,
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

  await run(args.srcPath, args.outFolder, args.batchSize);
})();
