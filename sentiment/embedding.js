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
 * Utilites for extracting the embedding matrix and output them as files.
 */

import {writeFileSync} from 'fs';
import * as tf from '@tensorflow/tfjs';

/**
 * Extract the first embedding matrix from a TensorFlow.js model.
 *
 * @param {tf.model} model An instance of tf.Model, assumed to contain an
 *   Embedding layer.
 * @retuns {tf.Tensor} The embedding matrix from the first Embedding
 *   layer encoutnered while iterating through all layers of the model.
 * @throws Error if no embedding layer can be found in the model.
 */
function extractEmbeddingMatrix(model) {
  for (const layer of model.layers) {
    if (layer.getClassName() === 'Embedding') {
      const embed = layer.getWeights()[0];
      tf.util.assert(
        embed.rank === 2,
        `Expected the rank of an embedding matrix to be 2, ` + 
        `but got ${embed.rank}`);
      return embed;
    }
  }
  throw new Error('Cannot find any Embedding layer in model.');
}

/**
 * Write the values of the first embedding matrix of a model to files.
 * 
 * The word labels are writen as well. The vectors and labels files are
 * directly loadable into the Embedding Projector
 * (https://projector.tensorflow.org/).
 *
 * @param {tf.model} model An instance of tf.Model, assumed to contain an
 *   Embedding layer.
 * @param {string} prefix Path prefix for writing the vectors and labels files.
 *   For exapmle if `prefix` is `/tmp/embed`, then 
 *   - the vectors will be written to `/tmp/embed_vectors.tsv`
 *   - the labels will be written to `/tmp/embed_labels.tsv`
 * @param {{[word: string]: number}} wordIndex A dictionary mapping words to
 *   their integer indices.
 * @param {number} indexFrom The basevalue of the integer indices.
 */
export async function writeEmbeddingMatrixAndLabels(
    model, prefix, wordIndex, indexFrom) {
  tf.util.assert(
      prefix != null && prefix.length > 0,
      `Null, undefined or empty path prefix`);

  const embed = extractEmbeddingMatrix(model);

  const numWords = embed.shape[0];
  const embedDims = embed.shape[1];
  const embedData = await embed.data();
  
  // Write the ebmedding matrix to file.
  let vectorsStr = '';
  let index = 0;
  for (let i = 0; i < numWords; ++i) {
    for (let j = 0; j < embedDims; ++j) {
      vectorsStr += embedData[index++].toFixed(5);
      if (j < embedDims - 1) {
        vectorsStr += '\t';
      } else {
        vectorsStr += '\n';
      }
    }
  }

  const vectorsFilePath = `${prefix}_vectors.tsv`;
  writeFileSync(vectorsFilePath, vectorsStr, {encoding: 'utf-8'});
  console.log(
      `Written embedding vectors (${numWords} * ${embedDims}) to: ` +
      `${vectorsFilePath}`);

  // Collect and write the word labels.
  const indexToWord = {};
  for (const word in wordIndex) {
    indexToWord[wordIndex[word]] = word;
  }

  let labelsStr = '';
  for(let i = 0; i < numWords; ++i) {
    if (i >= indexFrom) {
      labelsStr += indexToWord[i - indexFrom];
    } else {
      labelsStr += 'not-a-word';
    }
    labelsStr += '\n';
  }

  const labelsFilePath = `${prefix}_labels.tsv`;
  writeFileSync(labelsFilePath, labelsStr, {encoding: 'utf-8'});
  console.log(
      `Written embedding labels (${numWords}) to: ` +
      `${labelsFilePath}`);
}
