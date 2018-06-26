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

import * as tf from '@tensorflow/tfjs';

/**
 * A class for text data.
 *
 * This class manages the following:
 *
 * - Converting training data (as a string) into one hot-encoded vectors.
 * - Drawing random slices from the training data. This is useful for training
 *   models and obtaining the seed text for model-based text generation.
 */
export class TextData {
  /**
   * Constructor of TextData.
   *
   * @param {string} dataIdentifier An identifier for this instance of TextData.
   * @param {string} textString The training test data.
   * @param {number} sampleLen Length of each training example, i.e., the input
   *   sequence length expected by the LSTM model.
   * @param {number} sampleStep How many characters to skip when going from one
   *   example of the training data (in `textString`) to the next.
   */
  constructor(dataIdentifier, textString, sampleLen, sampleStep) {
    if (!dataIdentifier) {
      throw new Error('Model identifier is not provided.');
    }

    this._dataIdentifier = dataIdentifier;

    this._textString = textString;
    this._textLen = textString.length;
    this._sampleLen = sampleLen;
    this._sampleStep = sampleStep;

    this._getCharSet();
    this._convertAllTextToIndices();
    this._generateExampleBeginIndices();
  }

  /**
   * Get data identifier.
   *
   * @returns {string} The data identifier.
   */
  dataIdentifier() {
    return this._dataIdentifier;
  }

  /**
   * Get length of the training text data.
   *
   * @returns {number} Length of training text data.
   */
  textLen() {
    return this._textLen;
  }

  /**
   * Get the length of each training example.
   */
  sampleLen() {
    return this._sampleLen;
  }

  /**
   * Get the size of the character set.
   *
   * @returns {number} Size of the character set, i.e., how many unique
   *   characters there are in the training text data.
   */
  charSetSize() {
    return this._charSetSize;
  }

  /**
   * Generate the next epoch of data for training models.
   *
   * @param {number} numExamples Number examples to generate.
   * @returns {[tf.Tensor, tf.Tensor]} `xs` and `ys` Tensors.
   *   `xs` has the shape of `[numExamples, this.sampleLen, this.charSetSize]`.
   *   `ys` has the shape of `[numExamples, this.charSetSize]`.
   */
  nextDataEpoch(numExamples) {
    const xsBuffer = new tf.TensorBuffer([
        numExamples, this._sampleLen, this._charSetSize]);
    const ysBuffer  = new tf.TensorBuffer([numExamples, this._charSetSize]);
    for (let i = 0; i < numExamples; ++i) {
      const beginIndex = this._exampleBeginIndices[
          this._examplePosition % this._exampleBeginIndices.length];
      for (let j = 0; j < this._sampleLen; ++j) {
        xsBuffer.set(1, i, j, this._indices[beginIndex + j]);
      }
      ysBuffer.set(1, i, this._indices[beginIndex + this._sampleLen]);
      this._examplePosition++;
    }
    return [xsBuffer.toTensor(), ysBuffer.toTensor()];
  }

  /**
   * Get the unique character at given index from the character set.
   *
   * @param {number} index
   * @returns {string} The unique character at `index` of the character set.
   */
  getFromCharSet(index) {
    return this._charSet[index];
  }

  /**
   * Convert text string to integer indices.
   *
   * @param {string} text Input text.
   * @returns {number[]} Indices of the characters of `text`.
   */
  textToIndices(text) {
    const indices = [];
    for (let i = 0; i < text.length; ++i) {
      indices.push(this._charSet.indexOf(text[i]));
    }
    return indices;
  }

  /**
   * Get a random slice of text data.
   *
   * @returns {[string, number[]} The string and index representation of the
   *   same slice.
   */
  getRandomSlice() {
    const startIndex =
        Math.round(Math.random() * (this._textLen - this._sampleLen - 1));
    const textSlice = this._slice(startIndex, startIndex + this._sampleLen);
    return [textSlice, this.textToIndices(textSlice)];
  }

  /**
   * Get a slice of the training text data.
   *
   * @param {number} startIndex
   * @param {number} endIndex
   * @param {bool} useIndices Whether to return the indices instead of string.
   * @returns {string | Uint16Array} The result of the slicing.
   */
  _slice(startIndex, endIndex) {
    return this._textString.slice(startIndex, endIndex);
  }

  /**
   * Get the set of unique characters from text.
   */
  _getCharSet() {
    this._charSet = [];
    for (let i = 0; i < this._textLen; ++i) {
      if (this._charSet.indexOf(this._textString[i]) === -1) {
        this._charSet.push(this._textString[i]);
      }
    }
    this._charSetSize = this._charSet.length;
  }

  /**
   * Convert all training text to integer indices.
   */
  _convertAllTextToIndices() {
    this._indices = new Uint16Array(this.textToIndices(this._textString));
  }

  /**
   * Generate the example-begin indices; shuffle them randomly.
   */
  _generateExampleBeginIndices() {
    // Prepare beginning indices of examples.
    this._exampleBeginIndices = [];
    for (let i = 0;
        i < this._textLen - this._sampleLen - 1;
        i += this._sampleStep) {
      this._exampleBeginIndices.push(i);
    }

    // Randomly shuffle the beginning indices.
    tf.util.shuffle(this._exampleBeginIndices);
    this._examplePosition = 0;
  }
}
