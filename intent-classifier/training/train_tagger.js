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

const fs = require('fs');
const path = require('path');

const ndjson = require('ndjson');
const argparse = require('argparse');
const tf = require('@tensorflow/tfjs');
const fileIO = require('@tensorflow/tfjs-node/dist/io/file_system');
const mkdirp = require('mkdirp');

const {TAGS} = require('./util');

const {getModel} = require('./tagger_model');

const EMBEDDING_SIZE = 512;

async function loadNDJSON(path) {
  const records = [];

  const parser = fs.createReadStream(path).pipe(ndjson.parse());

  parser.on('data', obj => records.push(obj));

  return new Promise(function(resolve, reject) {
    parser.on('end', () => resolve(records));
    parser.on('error', e => reject(e));
  });
}

/**
 * Returns a generator function that yields batches of data for training.
 *
 * @param {string} embeddingsPath path to ndjson file with token embeddings
 * @param {string} taggedTokensPath path to ndjson with tagged tokens
 * @param {number} sequenceLength max length of sequence
 * @param {number} batchSize size of batch
 */
async function getDataIterator(
    embeddingsPath, taggedTokensPath, sequenceLength, batchSize) {
  // Load token embeddings and convert to tensors
  let tokenEmbeddingsTuples = await loadNDJSON(embeddingsPath);
  tokenEmbeddingsTuples = tokenEmbeddingsTuples.map(([token, embedding]) => {
    const embeddingAsTensor = tf.tensor1d(embedding);
    return [token, embeddingAsTensor];
  });
  // Add an 'embedding' for the __PAD__ token. We will encode it with tf.ones.
  tokenEmbeddingsTuples.push([TAGS[2], tf.ones([EMBEDDING_SIZE])]);
  const tokenEmbeddings = new Map(tokenEmbeddingsTuples);


  // Load the tagged intent tokens and convert the labels to one-hot
  // We will tranform the queries into tensors dynamically in the generator
  // function below.
  const taggedIntentTokens = await loadNDJSON(taggedTokensPath);
  const labelOneHots = new Map(
      TAGS.map(tag => [tag, tf.oneHot(TAGS.indexOf(tag), TAGS.length)]));

  console.log('Data Loaded');

  /**
   * Returns a batch of data for training. Will assemble a tensor of shape
   * [batchSize, sequenceLength, EMBEDDING_SIZE] that represent a batch of
   * queries whose tokens have been embedded with USE.
   */
  function* getNextBatch() {
    let xs = [];
    let ys = [];

    let toDispose = [];

    // Loop through all the tokenized sentences
    for (let idx = 0; idx < taggedIntentTokens.length; idx++) {
      const sentence = taggedIntentTokens[idx];
      const features = sentence[0];

      // Each example is converted to an array of length `sequenceLength`,
      // adding padding tokens if necessary and truncating the sentence
      // if it is too long.
      const exampleX = [];
      const exampleY = [];
      for (let index = 0; index < sequenceLength; index++) {
        let token;
        let tag;
        if (index < features.length) {
          const tuple = features[index];
          token = tuple.token;
          tag = tuple.tag;
        } else {
          // PADDING
          token = TAGS[TAGS.PAD_IDX];
          tag = TAGS[TAGS.PAD_IDX];
        }

        // Note that we reuse the tensors for a given token or tag.
        const tokenEmbedding = tokenEmbeddings.get(token);
        const tagOnehot = labelOneHots.get(tag);

        exampleX.push(tokenEmbedding);
        exampleY.push(tagOnehot);

        tf.util.assert(
            tokenEmbedding != null,
            () => console.log(`Error getting token embedding for ${token}`));

        tf.util.assertShapesMatch(
            tokenEmbedding.shape, [EMBEDDING_SIZE],
            () => console.log(`Wrong shape for token embedding of ${token}`));

        tf.util.assert(
            tagOnehot != null,
            () => console.log(`Error getting label onehot for ${tag}`));

        tf.util.assertShapesMatch(
            tagOnehot.shape, [TAGS.length],
            () => console.log(`Wrong shape for label onehot for ${tag}`));
      }

      // Add an example to the batch
      const xStacked = tf.stack(exampleX);
      const yStacked = tf.stack(exampleY);
      xs.push(xStacked);
      ys.push(yStacked);

      // We will dispose of these tensors once the higher rank tensor is
      // created for the whole batch.
      toDispose.push(xStacked);
      toDispose.push(yStacked);

      // Create a new batch and yield it.
      if (idx > 0 && idx % (batchSize - 1) === 0) {
        const batchedXS = tf.stack(xs);
        const batchedYS = tf.stack(ys);

        yield {xs: batchedXS, ys: batchedYS};

        tf.dispose([batchedXS, batchedYS, toDispose]);
        toDispose = [];
        xs = [];
        ys = [];
      }
    }
  }

  return getNextBatch;
}

/*
 * Train the model
 */
async function run(
    embeddingsPath, taggedTokensPath, outFolder, modelOpts, trainingOpts) {
  const dataIterator = await getDataIterator(
      embeddingsPath,
      taggedTokensPath,
      modelOpts.sequenceLength,
      trainingOpts.batchSize,
  );
  const dataset = {iterator: dataIterator};
  const model = getModel(modelOpts);

  console.log('Start training', trainingOpts.epochs);
  await model.fitDataset(dataset, {
    epochs: trainingOpts.epochs,
  });

  mkdirp(outFolder);
  console.log(`Saving model to ${outFolder}`);
  await model.save(fileIO.fileSystem(outFolder));


  // Write out the related metadata
  const metaOutPath = path.resolve(outFolder, 'tagger_metadata.json');
  const metadata = {
    labels: TAGS,
    sequenceLength: modelOpts.sequenceLength,
    embeddingSize: EMBEDDING_SIZE,
  };

  const metadataStr = JSON.stringify(metadata, null, 2);
  fs.writeFileSync(metaOutPath, metadataStr, {encoding: 'utf8'});
}


(async function() {
  const parser = new argparse.ArgumentParser();

  const defaultEmbeddingsPath =
      path.resolve(__dirname, './data/token_embeddings.ndjson');
  const defaultTaggedTokensPath =
      path.resolve(__dirname, './data/intents_tagged_tokens.ndjson');
  const defaultOutFolder =
      path.resolve(__dirname, './models/bidirectional-tagger/');

  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for training (required CUDA and CuDNN)',
  });
  parser.addArgument('--boostLOCTag', {
    action: 'storeTrue',
    help: 'Penalize incorrect LOC tags more than other errors',
    defaultValue: false,
  });
  parser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 20,
    help: 'Number of epochs',
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 32,
    help: 'Number of elements in a batch',
  });
  parser.addArgument('--embeddingsPath', {
    type: 'string',
    defaultValue: defaultEmbeddingsPath,
  });
  parser.addArgument('--taggedTokensPath', {
    type: 'string',
    defaultValue: defaultTaggedTokensPath,
  });
  parser.addArgument('--outFolder', {
    type: 'string',
    defaultValue: defaultOutFolder,
  });
  parser.addArgument('--modelType', {
    type: 'string',
    choices: ['bidirectional-lstm', 'lstm', 'dense'],
    defaultValue: 'bidirectional-lstm',
  });
  parser.addArgument('--sequenceLength', {
    type: 'int',
    defaultValue: 30,
    help: 'Max length of sequence that model can predict on',
  });

  const args = parser.parseArgs();

  if (args.gpu) {
    console.log('Attempting to use GPU.');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU.');
    require('@tensorflow/tfjs-node');
  }


  const modelOpts = {
    modelType: args.modelType,
    sequenceLength: args.sequenceLength,
    embeddingDims: EMBEDDING_SIZE,
    numLabels: TAGS.length,
  };

  if (args.boostLOCTag) {
    modelOpts.weights = [1, 1.5, 1];
  }

  const trainingOpts = {
    epochs: args.epochs,
    batchSize: args.batchSize,
  };

  let outFolder = args.outFolder;
  const modelType = modelOpts.modelType;
  if (modelType != 'bidirectional-lstm' && outFolder === defaultOutFolder) {
    switch (modelType) {
      case 'lstm':
        outFolder = path.resolve(__dirname, './models/lstm-tagger/');
        break;
      case 'dense':
        outFolder = path.resolve(__dirname, './models/dense-tagger/');
        break;
    }
  }

  await run(
      args.embeddingsPath, args.taggedTokensPath, outFolder, modelOpts,
      trainingOpts);
})();
