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

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as https from 'https';
import * as os from 'os';
import * as path from 'path';

import {OOV_INDEX, PAD_INDEX, padSequences} from './sequence_utils';

// `import` doesn't seem to work with extract-zip.
const extract = require('extract-zip');

const DATA_ZIP_URL =
    'https://storage.googleapis.com/learnjs-data/imdb/imdb_tfjs_data.zip';
const METADATA_TEMPLATE_URL =
    'https://storage.googleapis.com/learnjs-data/imdb/metadata.json.zip';

/**
 * Load IMDB data features from a local file.
 *
 * @param {string} filePath Data file on local filesystem.
 * @param {string} numWords Number of words in the vocabulary. Word indices
 *   that exceed this limit will be marked as `OOV_INDEX`.
 * @param {string} maxLen Length of each sequence. Longer sequences will be
 *   pre-truncated; shorter ones will be pre-padded.
 * @return {tf.Tensor} The dataset represented as a 2D `tf.Tensor` of shape
 *   `[]` and dtype `int32` .
 */
function loadFeatures(filePath, numWords, maxLen) {
  const buffer = fs.readFileSync(filePath);
  const numBytes = buffer.byteLength;

  let sequences = [];
  let seq = [];
  let index = 0;

  while (index < numBytes) {
    const value = buffer.readInt32LE(index);
    if (value === 1) {
      // A new sequence has started.
      if (index > 0) {
        sequences.push(seq);
      }
      seq = [];
    } else {
      // Sequence continues.
      seq.push(value >= numWords ? OOV_INDEX : value);
    }
    index += 4;
  }
  if (seq.length > 0) {
    sequences.push(seq);
  }
  const paddedSequences =
      padSequences(sequences, maxLen, 'pre', 'pre');
  return tf.tensor2d(
      paddedSequences, [paddedSequences.length, maxLen], 'int32');
}

/**
 * Load IMDB targets from a file.
 *
 * @param {string} filePath Path to the binary targets file.
 * @return {tf.Tensor} The targets as `tf.Tensor` of shape `[numExamples, 1]`
 *   and dtype `float32`. It has 0 or 1 values.
 */
function loadTargets(filePath) {
  const buffer = fs.readFileSync(filePath);
  const numBytes = buffer.byteLength;

  let ys = [];
  for (let i = 0; i < numBytes; ++i) {
    ys.push(buffer.readUInt8(i));
  }
  return tf.tensor2d(ys, [ys.length, 1], 'float32');
}

/**
 * Get a file by downloading it if necessary.
 *
 * @param {string} sourceURL URL to download the file from.
 * @param {string} destPath Destination file path on local filesystem.
 */
async function maybeDownload(sourceURL, destPath) {
  return new Promise(async (resolve, reject) => {
    if (!fs.existsSync(destPath) || fs.lstatSync(destPath).size === 0) {
      const localZipFile = fs.createWriteStream(destPath);
      console.log(`Downloading file from ${sourceURL} ...`);
      https.get(sourceURL, response => {
        response.pipe(localZipFile);
        localZipFile.on('finish', () => {
          localZipFile.close(() => resolve());
        });
        localZipFile.on('error', err => reject(err));
      });
    } else {
      return resolve();
    }
  });
}

/**
 * Get extracted files.
 *
 * If the files are already extracted, this will be a no-op.
 *
 * @param {string} sourcePath Source zip file path.
 * @param {string} destDir Extraction destination directory.
 */
async function maybeExtract(sourcePath, destDir) {
  return new Promise((resolve, reject) => {
    if (fs.existsSync(destDir)) {
      return resolve();
    }
    console.log(`Extracting: ${sourcePath} --> ${destDir}`);
    extract(sourcePath, {dir: destDir}, err => {
      if (err == null) {
        return resolve();
      } else {
        return reject(err);
      }
    });
  });
}

const ZIP_SUFFIX = '.zip';

/**
 * Get the IMDB data through file downloading and extraction.
 *
 * If the files already exist on the local file system, the download and/or
 * extraction steps will be skipped.
 */
async function maybeDownloadAndExtract() {
  const zipDownloadDest = path.join(os.tmpdir(), path.basename(DATA_ZIP_URL));
  await maybeDownload(DATA_ZIP_URL, zipDownloadDest);

  const zipExtractDir =
      zipDownloadDest.slice(0, zipDownloadDest.length - ZIP_SUFFIX.length);
  await maybeExtract(zipDownloadDest, zipExtractDir);
  return zipExtractDir;
}

/**
 * Load data by downloading and extracting files if necessary.
 *
 * @param {number} numWords Number of words to in the vocabulary.
 * @param {number} len Length of each sequence. Longer sequences will
 *   be pre-truncated and shorter ones will be pre-padded.
 * @return
 *   xTrain: Training data as a `tf.Tensor` of shape
 *     `[numExamples, len]` and `int32` dtype.
 *   yTrain: Targets for the training data, as a `tf.Tensor` of
 *     `[numExamples, 1]` and `float32` dtype. The values are 0 or 1.
 *   xTest: The same as `xTrain`, but for the test dataset.
 *   yTest: The same as `yTrain`, but for the test dataset.
 */
export async function loadData(numWords, len) {
  const dataDir = await maybeDownloadAndExtract();

  const trainFeaturePath = path.join(dataDir, 'imdb_train_data.bin');
  const xTrain = loadFeatures(trainFeaturePath, numWords, len);
  const testFeaturePath = path.join(dataDir, 'imdb_test_data.bin');
  const xTest = loadFeatures(testFeaturePath, numWords, len);
  const trainTargetsPath = path.join(dataDir, 'imdb_train_targets.bin');
  const yTrain = loadTargets(trainTargetsPath);
  const testTargetsPath = path.join(dataDir, 'imdb_test_targets.bin');
  const yTest = loadTargets(testTargetsPath);

  tf.util.assert(
      xTrain.shape[0] === yTrain.shape[0],
      `Mismatch in number of examples between xTrain and yTrain`);
  tf.util.assert(
      xTest.shape[0] === yTest.shape[0],
      `Mismatch in number of examples between xTest and yTest`);
  return {xTrain, yTrain, xTest, yTest};
}

/**
 * Load a metadata template by downloading and extracting files if necessary.
 *
 * @return A JSON object that is the metadata template.
 */
export async function loadMetadataTemplate() {
  const baseName = path.basename(METADATA_TEMPLATE_URL);
  const zipDownloadDest = path.join(os.tmpdir(), baseName);
  await maybeDownload(METADATA_TEMPLATE_URL, zipDownloadDest);

  const zipExtractDir =
      zipDownloadDest.slice(0, zipDownloadDest.length - ZIP_SUFFIX.length);
  await maybeExtract(zipDownloadDest, zipExtractDir);

  return JSON.parse(fs.readFileSync(
      path.join(zipExtractDir,
                baseName.slice(0, baseName.length - ZIP_SUFFIX.length))));
}
