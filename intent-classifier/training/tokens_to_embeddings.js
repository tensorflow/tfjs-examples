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
 * Takes a CSV file of queries and intents and embeds all the tokens in the
 * queries using the universal sentence encoder. The output of this script
 * will be an NDJSON where each line is an array in the following format:
 *
 *  [ "token": string, "embedding": [] ]
 */


const fs = require('fs');
const path = require('path');

const tf = require('@tensorflow/tfjs');
const ndjson = require('ndjson');
const argparse = require('argparse');

const {tokenizeSentence, loadCSV, chunk, flatMap, unique} = require('./util');

// Set up a global fetch variable to allow loading model weights via HTTP.
global.fetch = require('node-fetch');
const useLoader = require('@tensorflow-models/universal-sentence-encoder');

/**
 * Embed the input sentences using univeversal sentence encoder.
 * @param {string[]} sentences sentences to embed
 * @param {UniversalSentenceEncoder} use instance of the model
 * @param {number} batchSize how many tokens to embed in one call
 * @param {string} outputPath A path to an outputfile to write to.
 */
async function embedTokens(sentences, use, batchSize, outputPath) {
  // Set up file writer for ndjson
  const fd = fs.openSync(outputPath, 'w');
  const serialize = ndjson.serialize();
  serialize.on('data', line => {
    fs.appendFileSync(fd, line, 'utf8');
  });

  const tokens = flatMap(sentences, tokenizeSentence);
  const uniqueTokens = unique(tokens);

  console.log(
      `Got ${sentences.length} sentences with` +
      ` ${tokens.length} tokens where ${uniqueTokens.length} are unique`);

  const batchedTokens = chunk(uniqueTokens, batchSize);

  for (let i = 0; i < batchedTokens.length; i++) {
    console.time(
        `Converted batch ${i} of ${batchedTokens.length} (${batchSize}): `);

    const tokensFromBatch = batchedTokens[i];
    const embedding = await use.embed(tokensFromBatch);
    const embeddingArr = await embedding.array();

    for (let j = 0; j < tokensFromBatch.length; j++) {
      serialize.write([tokensFromBatch[j], embeddingArr[j]]);
    }

    tf.dispose([embedding]);

    console.timeEnd(
        `Converted batch ${i} of ${batchedTokens.length} (${batchSize}): `);
  }

  serialize.end();
  fs.closeSync(fd);
}

async function run(srcPath, outPath, batchSize) {
  console.log('Start');
  const use = await useLoader.load();
  console.log('Loaded Universal Sentence Encoder');

  const csvData = loadCSV(srcPath);
  const queries = csvData.map(q => q.query);

  await embedTokens(queries, use, batchSize, outPath);
  console.log('Done');
}


(async function() {
  const parser = new argparse.ArgumentParser();
  const defaultSrcPath = path.resolve(__dirname, './data/intents.csv');
  const defaultOutPath =
      path.resolve(__dirname, './data/token_embeddings.ndjson');

  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for training (required CUDA and CuDNN)',
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 512,
    help: 'Batch size to be used during model training',
  });
  parser.addArgument('--logDir', {
    type: 'string',
    help: 'Directory to which the TensorBoard summaries will be saved ' +
        'during training.',
  });
  parser.addArgument('--srcPath', {
    type: 'string',
    defaultValue: defaultSrcPath,
  });
  parser.addArgument('--outPath', {
    type: 'string',
    defaultValue: defaultOutPath,
  });

  const args = parser.parseArgs();

  if (args.gpu) {
    console.log('Using GPU.');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU.');
    require('@tensorflow/tfjs-node');
  }

  await run(args.srcPath, args.outPath, args.batchSize);
})();
