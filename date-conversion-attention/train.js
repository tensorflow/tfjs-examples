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

const argparse = require('argparse');
const tf = require('@tensorflow/tfjs');
// TODO(cais): Put under a command-line arg.
require('@tensorflow/tfjs-node');

const dateFormat = require('./date_format');

/**
 * A custom layer used to obtain the last time step of an RNN sequential
 * output.
 */
class GetLastTimestepLayer extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.supportMasking = true;
  }

  computeOutputShape(inputShape) {
    const outputShape = inputShape.slice();
    outputShape.splice(outputShape.length - 2, 1);
    return outputShape;
  }

  call(input) {
    if (Array.isArray(input)) {
      input = input[0];
    }
    const inputRank = input.shape.length;
    tf.util.assert(inputRank === 3, `Invalid input rank: ${inputRank}`);
    // TODO(cais): Use chaining API.
    return tf.squeeze(tf.gather(input, [input.shape[1] - 1], 1), [1]);
  }

  static get className() {
    return 'GetLastTimestepLayer';
  }
}

function createModel(inputDictSize, outputDictSize, inputLength, outputLength) {
  const embeddingDims = 64;
  const lstmUnits = 64;

  const encoderInput = tf.input({shape: [inputLength]});
  const decoderInput = tf.input({shape: [outputLength]});

  let encoder = tf.layers.embedding({
    inputDim: inputDictSize,
    outputDim: embeddingDims,
    inputLength,
    maskZero: true
  }).apply(encoderInput);
  encoder = tf.layers.lstm({
    units: lstmUnits,
    returnSequences: true
  }).apply(encoder);

  const encoderLast = new GetLastTimestepLayer({
    name: 'encoderLast'
  }).apply(encoder);

  let decoder = tf.layers.embedding({
    inputDim: outputDictSize,
    outputDim: embeddingDims,
    inputLength: outputLength,
    maskZero: true
  }).apply(decoderInput);
  decoder = tf.layers.lstm({
    units: lstmUnits,
    returnSequences: true
  }).apply(decoder, {initialState: [encoderLast, encoderLast]});

  let attention = tf.layers.dot({axes: [2, 2]}).apply([decoder, encoder]);
  attention = tf.layers.activation({
    activation: 'softmax',
    name: 'attention'
  }).apply(attention);

  const context = tf.layers.dot({
    axes: [2, 1],
    name: 'context'
  }).apply([attention, encoder]);
  const deocderCombinedContext =
      tf.layers.concatenate().apply([context, decoder]);
  let output = tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: lstmUnits,
      activation: 'tanh'
    })
  }).apply(deocderCombinedContext);
  output = tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: outputDictSize,
      activation: 'softmax'
    })
  }).apply(output);

  const model = tf.model({
    inputs: [encoderInput, decoderInput],
    outputs: output
    // outputs: attention
  });
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam'
  });
  return model;
}

// TODO(cais): Need unit test for this.
function generateBatchesForTraining(trainSplit = 0.8, valSplit = 0.15) {
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
  console.log(dateTuples.length);

  const numTrain = Math.floor(dateTuples.length * trainSplit);
  const numVal = Math.floor(dateTuples.length * valSplit);

  const inputFns = [
    dateFormat.dateTupleToDDMMMYYYY,
    dateFormat.dateTupleToMMDDYY,
    dateFormat.dateTupleToMMSlashDDSlashYY,
    dateFormat.dateTupleToMMSlashDDSlashYYYY
  ];

  // TODO(cais): Use tf.tidy().
  function dateTuplesToTensor(dateTuples) {
    const inputs = inputFns.map(fn => dateTuples.map(tuple => fn(tuple)));
    const inputStrings = [];
    inputs.forEach(inputs => inputStrings.push(...inputs));
    const encoderInput =
        dateFormat.encodeInputDateStrings(inputStrings);
    const trainTargetStrings = dateTuples.map(
        tuple => dateFormat.dateTupleToYYYYDashMMDashDD(tuple));
    let decoderInput = dateFormat.encodeOutputDateStrings(trainTargetStrings);
    // One-step time shift: The decoder input is shifted to the left by
    // one time step with respect to the encoder input. This accounts for
    // the step-by-step decoding that happens during inference time.
    decoderInput = tf.concat([
      tf.ones([decoderInput.shape[0], 1]).mul(dateFormat.START_CODE),
      decoderInput.slice(
          [0, 0], [decoderInput.shape[0], decoderInput.shape[1] - 1])
    ], 1).tile([inputFns.length, 1]);  
    const decoderOutput =
        dateFormat.encodeOutputDateStrings(trainTargetStrings, true)
        .tile([inputFns.length, 1, 1]);
    return {encoderInput, decoderInput, decoderOutput};
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

  return {
    trainEncoderInput,
    trainDecoderInput,
    trainDecoderOutput,
    valEncoderInput,
    valDecoderInput,
    valDecoderOutput,
    testDateTuples: dateTuples.slice(numTrain + numVal)
  };
}

/**
 * Perform sequence-to-sequence decoding for date conversion.
 *
 * @param {tf.Model} model The model to be used for the sequence-to-sequence
 *   decoding, with two inputs: 
 *   1. Encoder input of shape `[numExamples, inputLength]`
 *   2. Decoder input of shape `[numExamples, outputLength]`
 *   and one output:
 *   1. Decoder softmax probability output of shape
 *      `[numExamples, outputLength, outputVocabularySize]`
 * @param {string} inputStr Input date string to be converted.
 * @return {string} The converted date string.
 */
async function runSeq2SeqInference(model, inputStr) {
  const encoderInput = dateFormat.encodeInputDateStrings([inputStr]);
  const decoderInput = tf.buffer([1, dateFormat.OUTPUT_LENGTH]);
  decoderInput.set(dateFormat.START_CODE, 0, 0);

  for (let i = 1; i < dateFormat.OUTPUT_LENGTH; ++i) {
    const predictOut = model.predict(
        [encoderInput, decoderInput.toTensor()]);
    const output = (await predictOut.argMax(2).data())[i - 1];
    predictOut.dispose();
    decoderInput.set(output, 0, i);
  }
  const predictOut = model.predict(
      [encoderInput, decoderInput.toTensor()]);
  const finalOutput =
      (await predictOut.argMax(2).data())[dateFormat.OUTPUT_LENGTH - 1]

  let outputStr = '';
  for (let i = 1; i < decoderInput.shape[1]; ++i) {
    outputStr += dateFormat.OUTPUT_VOCAB[decoderInput.get(0, i)];
  }
  outputStr += dateFormat.OUTPUT_VOCAB[finalOutput];
  return outputStr;
}

function parseArguments() {
  const argParser = new argparse.ArgumentParser({
    description:
        'Train an attention-based date-conversion model in TensorFlow.js'
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
  return argParser.parseArgs();
}

async function run() {
  const args = parseArguments();

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
  } = generateBatchesForTraining();

  const history = await model.fit(
      [trainEncoderInput, trainDecoderInput], trainDecoderOutput, {
        epochs: args.epochs,
        batchSize: args.batchSize,
        shuffle: true,
        validationData: [[valEncoderInput, valDecoderInput], valDecoderOutput]
      });
  console.log(history.history);

  // Run seq2seq inference tests and print the results to console.
  const numTests = 10;
  const testInputFns = [
    dateFormat.dateTupleToDDMMMYYYY,
    dateFormat.dateTupleToMMDDYY,
    dateFormat.dateTupleToMMSlashDDSlashYY,
    dateFormat.dateTupleToMMSlashDDSlashYYYY
  ];
  for (let n = 0; n < numTests; ++n) {
    for (const testInputFn of testInputFns) {
      const inputStr = testInputFn(testDateTuples[n]);
      console.log('\n-----------------------');
      console.log(`Input string: ${inputStr}`);
      const correctAnswer =
          dateFormat.dateTupleToYYYYDashMMDashDD(testDateTuples[n]);
      console.log(`Correct answer: ${correctAnswer}`);

      const outputStr = await runSeq2SeqInference(model, inputStr);
      const isCorrect = outputStr === correctAnswer;
      console.log(
          `Model output: ${outputStr} (${isCorrect ? 'OK' : 'WRONG'})` );
    }
  }
}

if (require.main === module) {
  run();
}
