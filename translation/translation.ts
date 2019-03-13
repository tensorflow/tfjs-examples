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
 * Train a simple LSTM model for character-level language translation.
 * This is based on the Tensorflow.js example at:
 *   https://github.com/tensorflow/tfjs-examples/blob/master/translation/python/translation.py
 *
 * The training data can be downloaded with a command like the following example:
 *   wget http://www.manythings.org/anki/fra-eng.zip
 *
 * Author: Huan LI <zixia@zixia.net>
 * 2019, https://github.com/huan
 *
 */
import fs from 'fs';
import path from 'path';

import {ArgumentParser} from 'argparse';
import readline from 'readline';
import mkdirp from 'mkdirp';

const {zip} = require('zip-array');
const invertKv = require('invert-kv');

import * as tf from '@tensorflow/tfjs';

let FLAGS = {} as any;

async function readData (dataFile: string) {
  // Vectorize the data.
  const inputTexts: string[] = [];
  const targetTexts: string[] = [];

  const inputCharacters = new Set<string>();
  const targetCharacters = new Set<string>();

  const fileStream = fs.createReadStream(dataFile);
  const rl = readline.createInterface({
    input: fileStream,
    output: process.stdout,
    terminal: false,
  });

  let lineNumber = 0;
  rl.on('line', line => {
    if (++lineNumber > FLAGS.num_samples) {
      rl.close();
      return;
    }

    let [inputText, targetText] = line.split('\t');
    // We use "tab" as the "start sequence" character for the targets, and "\n"
    // as "end sequence" character.
    targetText = '\t' + targetText + '\n';

    inputTexts.push(inputText);
    targetTexts.push(targetText);

    for (const char of inputText) {
      if (!inputCharacters.has(char)) {
        inputCharacters.add(char);
      }
    }
    for (const char of targetText) {
      if (!targetCharacters.has(char)) {
        targetCharacters.add(char);
      }
    }
  })

  await new Promise(r => rl.on('close', r));

  const inputCharacterList = [...inputCharacters].sort();
  const targetCharacterList = [...targetCharacters].sort();

  const numEncoderTokens = inputCharacterList.length;
  const numDecoderTokens = targetCharacterList.length;

  // Math.max() does not work with very large arrays because of the stack limitation
  const maxEncoderSeqLength = inputTexts.map(text => text.length)
      .reduceRight((prev, curr) => curr > prev ? curr : prev, 0);
  const maxDecoderSeqLength = targetTexts.map(text => text.length)
      .reduceRight((prev, curr) => curr > prev ? curr : prev, 0);

  console.log('Number of samples:', inputTexts.length);
  console.log('Number of unique input tokens:', numEncoderTokens);
  console.log('Number of unique output tokens:', numDecoderTokens);
  console.log('Max sequence length for inputs:', maxEncoderSeqLength);
  console.log('Max sequence length for outputs:', maxDecoderSeqLength);

  const inputTokenIndex = inputCharacterList.reduce(
    (prev, curr, idx) => (prev[curr] = idx, prev),
    {} as {[char: string]: number},
  );
  const targetTokenIndex = targetCharacterList.reduce(
    (prev, curr, idx) => (prev[curr] = idx, prev),
    {} as {[char: string]: number},
  );

  // Save the token indices to file.
  const metadataJsonPath = path.join(
    FLAGS.artifacts_dir,
    'metadata.json',
  );

  if (!fs.existsSync(path.dirname(metadataJsonPath))) {
    mkdirp.sync(path.dirname(metadataJsonPath));
  }

  const metadata = {
    'input_token_index': inputTokenIndex,
    'target_token_index': targetTokenIndex,
    'max_encoder_seq_length': maxEncoderSeqLength,
    'max_decoder_seq_length': maxDecoderSeqLength,
  };

  fs.writeFileSync(metadataJsonPath, JSON.stringify(metadata));
  console.log('Saved metadata at: ', metadataJsonPath);

  const encoderInputDataBuf = tf.buffer<tf.Rank.R3>([
    inputTexts.length,
    maxEncoderSeqLength,
    numEncoderTokens,
  ]);
  const decoderInputDataBuf = tf.buffer<tf.Rank.R3>([
    inputTexts.length,
    maxDecoderSeqLength,
    numDecoderTokens,
  ]);
  const decoderTargetDataBuf = tf.buffer<tf.Rank.R3>([
    inputTexts.length,
    maxDecoderSeqLength,
    numDecoderTokens,
  ]);

  for (
    const [i, [inputText, targetText]]
    of (zip(inputTexts, targetTexts).entries() as IterableIterator<[number, [string, string]]>)
  ) {
    for (const [t, char] of inputText.split('').entries()) {
      // encoder_input_data[i, t, input_token_index[char]] = 1.
      encoderInputDataBuf.set(1, i, t, inputTokenIndex[char]);
    }

    for (const [t, char] of targetText.split('').entries()) {
      // decoder_target_data is ahead of decoder_input_data by one timestep
      decoderInputDataBuf.set(1, i, t, targetTokenIndex[char]);
      if (t > 0) {
        // decoder_target_data will be ahead by one timestep
        // and will not include the start character.
        decoderTargetDataBuf.set(1, i, t - 1, targetTokenIndex[char]);
      }
    }
  }

  const encoderInputData = encoderInputDataBuf.toTensor();
  const decoderInputData = decoderInputDataBuf.toTensor();
  const decoderTargetData = decoderTargetDataBuf.toTensor();

  return {
    inputTexts,
    maxEncoderSeqLength,
    maxDecoderSeqLength,
    numEncoderTokens,
    numDecoderTokens,
    inputTokenIndex,
    targetTokenIndex,
    encoderInputData,
    decoderInputData,
    decoderTargetData,
  };
}

/**
Create a Keras model for the seq2seq translation.

Args:
  num_encoder_tokens: Total number of distinct tokens in the inputs
    to the encoder.
  num_decoder_tokens: Total number of distinct tokens in the outputs
    to/from the decoder
  latent_dim: Number of latent dimensions in the LSTMs.

Returns:
  encoder_inputs: Instance of `keras.Input`, symbolic tensor as input to
    the encoder LSTM.
  encoder_states: Instance of `keras.Input`, symbolic tensor for output
    states (h and c) from the encoder LSTM.
  decoder_inputs: Instance of `keras.Input`, symbolic tensor as input to
    the decoder LSTM.
  decoder_lstm: `keras.Layer` instance, the decoder LSTM.
  decoder_dense: `keras.Layer` instance, the Dense layer in the decoder.
  model: `keras.Model` instance, the entire translation model that can be
    used in training.
*/
function seq2seqModel (
  numEncoderTokens: number,
  numDecoderTokens: number,
  latentDim: number,
) {
  // Define an input sequence and process it.
  const encoderInputs = tf.layers.input({
    shape: [null, numEncoderTokens] as number[],
    name: 'encoderInputs',
  });

  const encoder = tf.layers.lstm({
    units: latentDim,
    returnState: true,
    name: 'encoderLstm',
  });
  const [, stateH, stateC] = encoder.apply(encoderInputs) as tf.SymbolicTensor[];
  // We discard `encoder_outputs` and only keep the states.
  const encoderStates = [stateH, stateC];

  // Set up the decoder, using `encoder_states` as initial state.
  const decoderInputs = tf.layers.input({
    shape: [null, numDecoderTokens] as number[],
    name: 'decoderInputs',
  });
  // We set up our decoder to return full output sequences,
  // and to return internal states as well. We don't use the
  // return states in the training model, but we will use them in inference.
  const decoderLstm = tf.layers.lstm({
    units: FLAGS.latent_dim,
    returnSequences: true,
    returnState: true,
    name: 'decoderLstm',
  });

  const [decoderOutputs, ] = decoderLstm.apply(
    [decoderInputs, ...encoderStates],
  ) as tf.Tensor[];

  const decoderDense = tf.layers.dense({
    units: numDecoderTokens,
    activation: 'softmax',
    name: 'decoderDense',
  });

  const decoderDenseOutputs = decoderDense.apply(decoderOutputs) as tf.SymbolicTensor;

  // Define the model that will turn
  // `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  const model = tf.model({
    inputs: [encoderInputs, decoderInputs],
    outputs: decoderDenseOutputs,
    name: 'seq2seqModel',
  });
  return {
    encoderInputs,
    encoderStates,
    decoderInputs,
    decoderLstm,
    decoderDense,
    model,
  };
}

/**
Decode (i.e., translate) an encoded sentence.

Args:
  input_seq: A `numpy.ndarray` of shape
    `(1, max_encoder_seq_length, num_encoder_tokens)`.
  encoder_model: A `keras.Model` instance for the encoder.
  decoder_model: A `keras.Model` instance for the decoder.
  num_decoder_tokens: Number of unique tokens for the decoder.
  target_begin_index: An `int`: the index for the beginning token of the
    decoder.
  reverse_target_char_index: A lookup table for the target characters, i.e.,
    a map from `int` index to target character.
  max_decoder_seq_length: Maximum allowed sequence length output by the
    decoder.

Returns:
  The result of the decoding (i.e., translation) as a string.
"""
*/
async function decodeSequence (
  inputSeq: tf.Tensor,
  encoderModel: tf.LayersModel,
  decoderModel: tf.LayersModel,
  numDecoderTokens: number,
  targetBeginIndex: number,
  reverseTargetCharIndex: {[indice: number]: string},
  maxDecoderSeqLength: number,
) {
  // Encode the input as state vectors.
  let statesValue = encoderModel.predict(inputSeq) as tf.Tensor[];

  // Generate empty target sequence of length 1.
  let targetSeq = tf.buffer<tf.Rank.R3>([
    1,
    1,
    numDecoderTokens,
  ]);

  // Populate the first character of target sequence with the start character.
  targetSeq.set(1, 0, 0, targetBeginIndex);

  // Sampling loop for a batch of sequences
  // (to simplify, here we assume a batch of size 1).
  let stopCondition = false;
  let decodedSentence = '';
  while (!stopCondition) {
    const [outputTokens, h, c] = decoderModel.predict(
      [targetSeq.toTensor(), ...statesValue]
    ) as [
      tf.Tensor<tf.Rank.R3>,
      tf.Tensor<tf.Rank.R2>,
      tf.Tensor<tf.Rank.R2>,
    ];

    // Sample a token
    const sampledTokenIndex = await outputTokens
                                      .squeeze()
                                      .argMax(-1)
                                      .array() as number;

    const sampledChar = reverseTargetCharIndex[sampledTokenIndex];
    decodedSentence += sampledChar;

    // Exit condition: either hit max length
    // or find stop character.
    if ( sampledChar === '\n'
      || decodedSentence.length > maxDecoderSeqLength
    ) {
      stopCondition = true;
    }

    // Update the target sequence (of length 1).
    targetSeq = tf.buffer<tf.Rank.R3>([1, 1, numDecoderTokens], 'float32');
    targetSeq.set(1, 0, 0, sampledTokenIndex);

    // Update states
    statesValue = [h, c];
  }
  return decodedSentence;
}

async function main () {
  if (FLAGS.gpu) {
    console.log('Using GPU');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU');
    require('@tensorflow/tfjs-node');
  }

  const {
    inputTexts,
    maxDecoderSeqLength,
    numEncoderTokens,
    numDecoderTokens,
    targetTokenIndex,
    encoderInputData,
    decoderInputData,
    decoderTargetData,
  } = await readData(FLAGS.data_path);

  const {
    encoderInputs,
    encoderStates,
    decoderInputs,
    decoderLstm,
    decoderDense,
    model,
  } = seq2seqModel(
    numEncoderTokens,
    numDecoderTokens,
    FLAGS.latent_dim,
  );

  // Run training.
  model.compile({
    optimizer: 'rmsprop',
    loss: 'categoricalCrossentropy',
  });

  model.summary();

  await model.fit(
    [
      encoderInputData,
      decoderInputData,
    ],
    decoderTargetData,
    {
      batchSize: FLAGS.batch_size,
      epochs: FLAGS.epochs,
      validationSplit: 0.2,
    },
  );

  await model.save('file://' + FLAGS.artifacts_dir);

  // tfjs.converters.save_keras_model(model, FLAGS.artifacts_dir)

  // Next: inference mode (sampling).
  // Here's the drill:
  // 1) encode input and retrieve initial decoder state
  // 2) run one step of decoder with this initial state
  // and a "start of sequence" token as target.
  // Output will be the next target token
  // 3) Repeat with the current target token and current states

  // Define sampling models
  const encoderModel = tf.model({
    inputs: encoderInputs,
    outputs: encoderStates,
    name: 'encoderModel',
  });

  const decoderStateInputH = tf.layers.input({
    shape: [FLAGS.latent_dim],
    name: 'decoderStateInputHidden',
  });
  const decoderStateInputC = tf.layers.input({
    shape: FLAGS.latent_dim,
    name: 'decoderStateInputCell',
  });
  const decoderStatesInputs = [decoderStateInputH, decoderStateInputC];
  let [decoderOutputs, stateH, stateC] = decoderLstm.apply(
      [decoderInputs, ...decoderStatesInputs]
  ) as tf.SymbolicTensor[];

  const decoderStates = [stateH, stateC];
  decoderOutputs = decoderDense.apply(decoderOutputs) as tf.SymbolicTensor;
  const decoderModel = tf.model({
    inputs: [decoderInputs, ...decoderStatesInputs],
    outputs: [decoderOutputs, ...decoderStates],
    name: 'decoderModel',
  });

  // Reverse-lookup token index to decode sequences back to
  // something readable.
  const reverseTargetCharIndex = invertKv(targetTokenIndex) as {[indice: number]: string};

  const targetBeginIndex = targetTokenIndex['\t'];

  for (let seqIndex = 0; seqIndex < FLAGS.num_test_sentences; seqIndex++) {
    // Take one sequence (part of the training set)
    // for trying out decoding.
    const inputSeq = encoderInputData.slice(seqIndex, 1);

    // Get expected output
    const targetSeqVoc = decoderTargetData.slice(seqIndex, 1).squeeze([0]) as tf.Tensor2D;
    const targetSeqTensor = targetSeqVoc.argMax(-1) as tf.Tensor1D;

    const targetSeqList = await targetSeqTensor.array();

    // One-hot to index
    const targetSeq = targetSeqList.map(indice => reverseTargetCharIndex[indice]);

    // Array to string
    const targetSeqStr = targetSeq.join('').replace('\n', '');
    const decodedSentence = await decodeSequence(
      inputSeq, encoderModel, decoderModel, numDecoderTokens,
      targetBeginIndex, reverseTargetCharIndex, maxDecoderSeqLength,
    );
    console.log('-');
    console.log('Input sentence:', inputTexts[seqIndex]);
    console.log('Target sentence:', targetSeqStr);
    console.log('Decoded sentence:', decodedSentence);
  }
}


const parser = new ArgumentParser({
  version: '0.0.1',
  addHelp: true,
  description: 'Keras seq2seq translation model training and serialization',
});

parser.addArgument(
  ['data_path'],
  {
    type: 'string',
    help: 'Path to the training data, e.g., ~/ml-data/fra-eng/fra.txt',
  },
);
parser.addArgument(
  '--batch_size',
  {
    type: 'int',
    defaultValue: 64,
    help: 'Training batch size.'
  }
);
parser.addArgument(
  '--epochs',
  {
    type: 'int',
    defaultValue: 200,
    help: 'Number of training epochs.',
  },
);
parser.addArgument(
  '--latent_dim',
  {
    type: 'int',
    defaultValue: 256,
    help: 'Latent dimensionality of the encoding space.',
  },
);
parser.addArgument(
  '--num_samples',
  {
    type: 'int',
    defaultValue: 10000,
    help: 'Number of samples to train on.',
  }
);
parser.addArgument(
  '--num_test_sentences',
  {
    type: 'int',
    defaultValue: 100,
    help: 'Number of example sentences to test at the end of the training.',
  },
);
parser.addArgument(
  '--artifacts_dir',
  {
    type: 'string',
    defaultValue: '/tmp/translation.keras',
    help: 'Local path for saving the TensorFlow.js artifacts.',
  },
);
parser.addArgument('--gpu', {
  action: 'storeTrue',
  help: 'Use tfjs-node-gpu to train the model. Requires CUDA/CuDNN.'
});

[FLAGS,] = parser.parseKnownArgs();
main();
