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

const tf = require('@tensorflow/tfjs');
// TODO(cais): Put under a command-line arg.
require('@tensorflow/tfjs-node');

const dateFormat = require('./date_format');

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

  call(input, kwargs) {
    if (Array.isArray(input)) {
      input = input[0];
    }
    // console.log('In GetLastTimestepLayer.call():', input);  // DEBUG
    const inputRank = input.shape.length;
    tf.util.assert(inputRank === 3, `Invalid input rank: ${inputRank}`);
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

//   console.log(encoder);  // DEBUG
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
  console.log(`numTrain = ${numTrain}`);  // DEBUG

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
    decoderInput = tf.concat([
      tf.ones([decoderInput.shape[0], 1]).mul(dateFormat.START_CODE),
      decoderInput.slice(
          [0, 1], [decoderInput.shape[0], decoderInput.shape[1] - 1])
    ], 1).tile([inputFns.length, 1]);  // One-step time shift.
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

  // const trainInputTensors = trainInputs.map(
  //     strings => dateFormat.encodeInputDateStrings(strings));
  // console.log(trainInputTensors.length);
  // const
  // console.log(trainInputs[0].length);
  // console.log(trainInputs[0][0]);
  // console.log(trainInputs[1].length);
  // console.log(trainInputs[1][0]);
}

// DEBUG
async function run() {
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
  console.log(trainEncoderInput.shape);  // DEBUG
  console.log(trainDecoderInput.shape);  // DEBUG

  const history = await model.fit(
      [trainEncoderInput, trainDecoderInput], trainDecoderOutput, {
        epochs: 1,  // TODO(cais): Make this a command-line arg.
        batchSize: 1024,  // TODO(cais): Make this a command-line arg.
        validationData: [[valEncoderInput, valDecoderInput], valDecoderOutput]
      });
  console.log(history.history);

  // Run inference.
  const numTests = 10;
  for (let n = 0; n < numTests; ++n) {
    const inputStr = dateFormat.dateTupleToDDMMMYYYY(testDateTuples[n]);
    console.log('\n-----------------------');
    console.log(`Input string: ${inputStr}`);  // DEBUG
    const correctAnswer =
        dateFormat.dateTupleToYYYYDashMMDashDD(testDateTuples[n]);
    console.log(`Correct answer: ${correctAnswer}`);  // DEBUG

    const testEncoderInput = dateFormat.encodeInputDateStrings([inputStr]);
    const testDecoderInput = tf.buffer([1, dateFormat.OUTPUT_LENGTH]);
    testDecoderInput.set(dateFormat.START_CODE, 0, 0);

    for (let i = 1; i < dateFormat.OUTPUT_LENGTH; ++i) {
      const output = model.predict(
          [testEncoderInput, testDecoderInput.toTensor()])
          .argMax(2).dataSync();
      testDecoderInput.set(output[i], 0, i);
    }
    const finalOutput = model.predict(
        [testEncoderInput, testDecoderInput.toTensor()])
        .argMax(2).dataSync()[0];
    console.log(finalOutput);  // DEBUG

    let outputStr = '';
    for (let i = 1; i < testDecoderInput.shape[1]; ++i) {
      outputStr += dateFormat.OUTPUT_VOCAB[testDecoderInput.get(0, i)];
    }
    console.log(`Model output: ${outputStr}`);
  }
}

run();
