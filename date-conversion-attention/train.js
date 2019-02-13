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
 * Training an attention LSTM sequence-to-sequence decoder to translate
 * various date formats into the ISO date format.
 *
 * Inspired by and loosely based on
 * https://github.com/wanasit/katakana/blob/master/notebooks/Attention-based%20Sequence-to-Sequence%20in%20Keras.ipynb
 */

import * as fs from 'fs';
import * as shelljs from 'shelljs';
import * as argparse from 'argparse';
import * as tf from '@tensorflow/tfjs';
import * as dateFormat from './date_format';
import {createModel, runSeq2SeqInference} from './model';

/**
 * Generate sets of data for training.
 *
 * @param {number} trainSplit Trainining split. Must be >0 and <1.
 * @param {number} valSplit Validatoin split. Must be >0 and <1.
 * @return An `Object` consisting of
 *   - trainEncoderInput, as a `tf.Tensor` of shape
 *     `[numTrainExapmles, inputLength]`
 *   - trainDecoderInput, as a `tf.Tensor` of shape
 *     `[numTrainExapmles, outputLength]`. The first element of every
 *     example has been set as START_CODE (the sequence-start symbol).
 *   - trainDecoderOuptut, as a one-hot encoded `tf.Tensor` of shape
 *     `[numTrainExamples, outputLength, outputVocabSize]`.
 *   - valEncoderInput, same as trainEncoderInput, but for the validation set.
 *   - valDecoderInput, same as trainDecoderInput, but for the validation set.
 *   - valDecoderOutput, same as trainDecoderOuptut, but for the validation
 *     set.
 *   - testDateTuples, date tuples ([year, month, day]) for the test set.
 */
export function generateDataForTraining(trainSplit = 0.8, valSplit = 0.15) {
  tf.util.assert(
      trainSplit > 0 && valSplit > 0 && trainSplit + valSplit <= 1,
      `Invalid trainSplit (${trainSplit}) and valSplit (${valSplit})`);

  const dateTuples = [];
  const MIN_YEAR = 1950;
  const MAX_YEAR = 2050;
  for (let year = MIN_YEAR; year < MAX_YEAR; ++year) {
    for (let month = 1; month <= 12; ++month) {
      for (let day = 1; day <= 28; ++day) {
        dateTuples.push([year, month, day]);
      }
    }
  }
  tf.util.shuffle(dateTuples);

  const numTrain = Math.floor(dateTuples.length * trainSplit);
  const numVal = Math.floor(dateTuples.length * valSplit);

  function dateTuplesToTensor(dateTuples) {
    return tf.tidy(() => {
      const inputs =
          dateFormat.INPUT_FNS.map(fn => dateTuples.map(tuple => fn(tuple)));
      const inputStrings = [];
      inputs.forEach(inputs => inputStrings.push(...inputs));
      const encoderInput =
          dateFormat.encodeInputDateStrings(inputStrings);
      const trainTargetStrings = dateTuples.map(
          tuple => dateFormat.dateTupleToYYYYDashMMDashDD(tuple));
      let decoderInput =
          dateFormat.encodeOutputDateStrings(trainTargetStrings)
          .asType('float32');
      // One-step time shift: The decoder input is shifted to the left by
      // one time step with respect to the encoder input. This accounts for
      // the step-by-step decoding that happens during inference time.
      decoderInput = tf.concat([
        tf.ones([decoderInput.shape[0], 1]).mul(dateFormat.START_CODE),
        decoderInput.slice(
            [0, 0], [decoderInput.shape[0], decoderInput.shape[1] - 1])
      ], 1).tile([dateFormat.INPUT_FNS.length, 1]);
      const decoderOutput = tf.oneHot(
          dateFormat.encodeOutputDateStrings(trainTargetStrings),
          dateFormat.OUTPUT_VOCAB.length).tile(
              [dateFormat.INPUT_FNS.length, 1, 1]);
      return {encoderInput, decoderInput, decoderOutput};
    });
  }

  const {
    encoderInput: trainEncoderInput,
    decoderInput: trainDecoderInput,
    decoderOutput: trainDecoderOutput
  } = dateTuplesToTensor(dateTuples.slice(0, numTrain));
  const {
    encoderInput: valEncoderInput,
    decoderInput: valDecoderInput,
    decoderOutput: valDecoderOutput
  } = dateTuplesToTensor(dateTuples.slice(numTrain, numTrain + numVal));
  const testDateTuples =
      dateTuples.slice(numTrain + numVal, dateTuples.length);
  return {
    trainEncoderInput,
    trainDecoderInput,
    trainDecoderOutput,
    valEncoderInput,
    valDecoderInput,
    valDecoderOutput,
    testDateTuples
  };
}

function parseArguments() {
  const argParser = new argparse.ArgumentParser({
    description:
        'Train an attention-based date-conversion model in TensorFlow.js'
  });
  argParser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu to train the model. Requires CUDA/CuDNN.'
  });
  argParser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 2,
    help: 'Number of epochs to train the model for'
  });
  argParser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 128,
    help: 'Batch size to be used during model training'
  });
  argParser.addArgument('--savePath', {
    type: 'string',
    defaultValue: './dist/model',
  });
  return argParser.parseArgs();
}

async function run() {
  const args = parseArguments();
  if (args.gpu) {
    console.log('Using GPU');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU');
    require('@tensorflow/tfjs-node');
  }

  const model = createModel(
      dateFormat.INPUT_VOCAB.length, dateFormat.OUTPUT_VOCAB.length,
      dateFormat.INPUT_LENGTH, dateFormat.OUTPUT_LENGTH);
  model.summary();

  const {
    trainEncoderInput,
    trainDecoderInput,
    trainDecoderOutput,
    valEncoderInput,
    valDecoderInput,
    valDecoderOutput,
    testDateTuples
  } = generateDataForTraining();

  await model.fit(
      [trainEncoderInput, trainDecoderInput], trainDecoderOutput, {
        epochs: args.epochs,
        batchSize: args.batchSize,
        shuffle: true,
        validationData: [[valEncoderInput, valDecoderInput], valDecoderOutput]
      });

  // Save the model.
  if (args.savePath != null && args.savePath.length) {
    if (!fs.existsSync(args.savePath)) {
      shelljs.mkdir('-p', args.savePath);
    }
    const saveURL = `file://${args.savePath}`
    await model.save(saveURL);
    console.log(`Saved model to ${saveURL}`);
  }

  // Run seq2seq inference tests and print the results to console.
  const numTests = 10;
  for (let n = 0; n < numTests; ++n) {
    for (const testInputFn of dateFormat.INPUT_FNS) {
      const inputStr = testInputFn(testDateTuples[n]);
      console.log('\n-----------------------');
      console.log(`Input string: ${inputStr}`);
      const correctAnswer =
          dateFormat.dateTupleToYYYYDashMMDashDD(testDateTuples[n]);
      console.log(`Correct answer: ${correctAnswer}`);

      const {outputStr} = await runSeq2SeqInference(model, inputStr);
      const isCorrect = outputStr === correctAnswer;
      console.log(
          `Model output: ${outputStr} (${isCorrect ? 'OK' : 'WRONG'})` );
    }
  }
}

if (require.main === module) {
  run();
}
